"""Run exact finite-state checks and CPU total-cost comparisons.

No fake LLM sleep, model cost extrapolation or remote inference is used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deltaweave.engine import Engine, Query
from deltaweave.store import Store

MODES = ("full", "positive", "coarse", "predicate", "adaptive")


def selected(ctx, args):
    return ctx.query(Query("items", where={"project": args["project"], "enabled": True},
                           select=("value",), order_by=("priority",), limit=2))


def root_task(ctx, args):
    choices = ctx.query(Query("control", where={"id": args["id"]}, select=("project",)))
    return None if not choices else ctx.call("select", choices[0])


def attach(engine):
    engine.register("select", selected)
    engine.register("root", root_task)


def oracle(tables, root_id):
    controls = [r for r in tables["control"] if r["id"] == root_id]
    if not controls:
        return None
    rows = sorted((r for r in tables["items"] if r["project"] == controls[0]["project"] and r["enabled"] is True),
                  key=lambda x: (x["priority"], x["id"]))[:2]
    return [{"value": r["value"]} for r in rows]


def finite_checks(indexed=True):
    """All 64 configurations of 3 rows, then each single-field toggle.

    An independent direct Python oracle never reads engine caches/dirty sets.
    Each starting state is independent; all transitions stay finite and exact.
    """
    results = {mode: {"checks": 0, "mismatches": 0, "first_failure": None} for mode in MODES}
    for bits in itertools.product((False, True), repeat=6):
        tables = {"control": [{"id": "q", "project": "p"}],
                  "items": [{"id": str(i), "project": "p", "enabled": bits[i],
                             "priority": i, "value": int(bits[i+3])} for i in range(3)]}
        for changed in range(6):
            revised = deepcopy(tables)
            row = revised["items"][changed % 3]
            field = "enabled" if changed < 3 else "value"
            row[field] = not row[field] if field == "enabled" else 1 - row[field]
            gold = oracle(revised, "q")
            for mode in MODES:
                store = Store(tables, indexed=indexed)
                engine = Engine(store, mode)
                attach(engine)
                assert engine.run("root", {"id": "q"}) == oracle(tables, "q")
                store.commit(upserts={"items": [row]})
                prediction = engine.run("root", {"id": "q"})
                result = results[mode]
                result["checks"] += 1
                if prediction != gold:
                    result["mismatches"] += 1
                    if result["first_failure"] is None:
                        result["first_failure"] = {"before": tables, "upsert": row, "gold": gold, "actual": prediction}
    for mode in ("full", "coarse", "predicate", "adaptive"):
        if results[mode]["mismatches"]:
            raise AssertionError(f"Correctness gate failed: {mode}")
    return results


def workload(n, pattern, seed=42, steps=16, queries=24):
    rng = random.Random(seed)
    projects = [f"p{i}" for i in range(queries)]
    tables = {"control": [{"id": f"q{i}", "project": p} for i, p in enumerate(projects)],
              "items": [{"id": f"r{i:06d}", "project": projects[i % queries],
                         "enabled": i % 2 == 0, "priority": i, "value": i % 7, "note": "initial"} for i in range(n)]}
    current = deepcopy(tables)
    events = []
    for step in range(steps):
        if pattern == "unrelated":
            updates = [{"id": f"outside{step}", "project": "outside", "enabled": True, "priority": 0, "value": step}]
        elif pattern == "filtered":
            updates = [{"id": f"excluded{step}", "project": projects[step % queries], "enabled": False, "priority": 0, "value": step}]
        elif pattern == "same_observation":
            index = (step * 17) % n
            row = deepcopy(current["items"][index]); row["note"] = f"revised-{step}"
            current["items"][index] = row; updates = [row]
        elif pattern == "high_fanout":
            updates = []
            for index in range(n):
                row = deepcopy(current["items"][index]); row["value"] += 1
                current["items"][index] = row; updates.append(row)
        elif pattern == "mixed":
            index = rng.randrange(n)
            row = deepcopy(current["items"][index]); row["enabled"] = not row["enabled"]
            current["items"][index] = row; updates = [row]
        elif pattern == "low_reuse":
            new_controls = [{"id": f"fresh-q{step}-{i}", "project": f"fresh-p{step}-{i}"} for i in range(queries)]
            new_items = [{"id": f"fresh-r{step}-{i}", "project": f"fresh-p{step}-{i}",
                          "enabled": True, "priority": 0, "value": step+i} for i in range(queries)]
            events.append({"upserts": {"items": new_items, "control": new_controls},
                           "requests": [r["id"] for r in new_controls]})
            continue
        else:
            raise ValueError(pattern)
        events.append({"upserts": {"items": updates}, "requests": [f"q{i}" for i in range(queries)]})
    return tables, events


def one_run(tables, events, mode, indexed=True):
    started = time.perf_counter()
    store = Store(tables, indexed=indexed); engine = Engine(store, mode); attach(engine)
    begin = time.perf_counter()
    first = [engine.run("root", {"id": row["id"]}) for row in tables["control"]]
    cold_seconds = time.perf_counter() - started
    row_map = {table: {r["id"]: deepcopy(r) for r in rows} for table, rows in tables.items()}
    mismatches = 0; commit_seconds = 0.0; query_wall = time.perf_counter() - begin
    index_update_seconds = 0.0
    oracle_seconds = 0.0
    for event in events:
        tick = time.perf_counter()
        store.commit(upserts=event["upserts"])
        commit_seconds += time.perf_counter() - tick
        index_update_seconds += store.last_commit_stats["index_update_seconds"]
        tick = time.perf_counter()
        for table, rows in event["upserts"].items():
            row_map[table].update({r["id"]: deepcopy(r) for r in rows})
        expected = [oracle({t: list(rs.values()) for t, rs in row_map.items()}, q) for q in event["requests"]]
        oracle_seconds += time.perf_counter() - tick
        tick = time.perf_counter()
        actual = [engine.run("root", {"id": q}) for q in event["requests"]]
        query_wall += time.perf_counter() - tick
        mismatches += sum(a != b for a, b in zip(actual, expected))
    total = time.perf_counter() - started - oracle_seconds
    if mode != "positive" and mismatches:
        raise AssertionError((mode, mismatches))
    return {"mode": mode, "indexed": indexed, "total_seconds_including_cold_and_commits": total,
            "cold_seconds": cold_seconds, "commit_seconds": commit_seconds, "request_seconds": query_wall,
            "requests": len(first) + sum(len(e["requests"]) for e in events), "mismatches": mismatches,
            "stats": engine.stats,
            "index_initialization": store.initial_stats, "index_update_seconds": index_update_seconds,
            "snapshot_history_note": "Store retains full immutable snapshots; memory growth is not production-bounded.",
            "query_index_note": "All modes share exact scalar-posting indexes when enabled; not a SQL/IVM comparison."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"experiments"/"updates-v1.json")
    parser.add_argument("--sizes", type=int, nargs="+", default=[96, 960])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-off", action="store_true", help="Match the unindexed exploratory implementation")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {"protocol": "research/protocol-v1.md", "python": platform.python_version(), "platform": platform.platform(),
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT/"src/deltaweave/engine.py", ROOT/"src/deltaweave/store.py"]},
              "indexed": not args.index_off, "seed": args.seed, "steps": args.steps, "repeats": args.repeats,
              "finite_checks": finite_checks(not args.index_off), "measurements": [], "aggregates": []}
    for n in args.sizes:
        for pattern in ("unrelated", "filtered", "same_observation", "mixed", "high_fanout", "low_reuse"):
            tables, events = workload(n, pattern, seed=args.seed, steps=args.steps)
            for repeat in range(args.repeats):
                # Rotate execution order to avoid giving a mode all cold starts.
                order = MODES[repeat % len(MODES):] + MODES[:repeat % len(MODES)]
                for mode in order:
                    row = {"rows": n, "pattern": pattern, "repeat": repeat, **one_run(tables, events, mode, not args.index_off)}
                    report["measurements"].append(row)
            for mode in MODES:
                rows = [r for r in report["measurements"] if r["rows"] == n and r["pattern"] == pattern and r["mode"] == mode]
                report["aggregates"].append({"rows": n, "pattern": pattern, "mode": mode,
                    "median_total_seconds": statistics.median(r["total_seconds_including_cold_and_commits"] for r in rows),
                    "min_total_seconds": min(r["total_seconds_including_cold_and_commits"] for r in rows),
                    "max_total_seconds": max(r["total_seconds_including_cold_and_commits"] for r in rows),
                    "mismatches_per_run": [r["mismatches"] for r in rows],
                    "query_executions": rows[0]["stats"]["query_executions"],
                    "query_scans": rows[0]["stats"]["query_scans"]})
            print(json.dumps({"completed": [n, pattern]}), flush=True)
    args.output.parent.mkdir(exist_ok=True, parents=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "finite_checks": report["finite_checks"]}), flush=True)


if __name__ == "__main__":
    main()
