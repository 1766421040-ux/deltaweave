#!/usr/bin/env python3
"""CPU-only declared-dependency replay; no LLM and no online requests.

PYTHONPATH=src python3 scripts/benchmark_domain.py --output experiments/domain-v2.json

Public metadata observations and synthetic interventions are separately labeled.
All timings are actual local CPU measurements, not projected model speedups.
The JSON is saved even when comparisons fail, and existing output is never replaced.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from time import perf_counter
from typing import Any

from packaging.version import Version

from deltaweave import Engine, Store
from deltaweave.domain import TABLES, register_domain, tables_from_catalog
from deltaweave.oracle import brute_force, check_selection

ROOT = Path(__file__).resolve().parents[1]
MODES = ("full", "coarse", "predicate", "adaptive", "positive")
REQUIRED_MODES = ("full", "coarse", "predicate", "adaptive")
ENVIRONMENT = {
    "python_full_version": "3.11.8", "python_version": "3.11", "sys_platform": "linux",
    "os_name": "posix", "platform_python_implementation": "CPython", "implementation_name": "cpython",
    "implementation_version": "3.11.8", "platform_machine": "x86_64", "platform_release": "synthetic-release",
    "platform_system": "Linux", "platform_version": "synthetic-version",
}


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def release(project: str, version: str = "1", dependencies=(), **overrides: Any) -> dict[str, Any]:
    return {"id": f"{project}-{version}", "project": project, "version": version,
            "requires_python": ">=3.8", "requires_dist": list(dependencies), "provides_extra": [],
            "yanked": False, **overrides}


def tables(*rows: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    result = {key: [] for key in TABLES}
    result["releases"] = list(rows)
    result["projects"] = [{"id": name, "project": name, "complete": True} for name in sorted({row["project"] for row in rows})]
    result["environment"] = [{"id": field, "environment": "benchmark", "field": field, "value": value} for field, value in ENVIRONMENT.items()]
    return result


def args(*requirements: str, **overrides: Any) -> dict[str, Any]:
    return {"requirements": list(requirements), "environment": "benchmark", "as_of": "2026-10-28", **overrides}


def step(label: str, *, upserts=None, deletes=None, arguments=None) -> dict[str, Any]:
    return {"label": label, "upserts": upserts or {}, "deletes": deletes or {}, "arguments": arguments}


def chain(name: str, data: dict, arguments: dict, events: list, **metadata: Any) -> dict[str, Any]:
    return {"id": name, "origin": "synthetic_controlled_world", "tables": data, "arguments": arguments,
            "steps": [step("initial")] + events, **metadata}


def controlled_chains() -> list[dict[str, Any]]:
    abi = {"id": "abi-rule", "project": "a", "kind": "require_env", "field": "ssl_abi", "operator": "version", "specifier": ">=3"}
    env = {"id": "ssl", "environment": "benchmark", "field": "ssl_abi", "value": "1"}
    alias_data = tables(release("a"), release("b"))
    alias_data["aliases"] = [{"id": "alias", "alias": "legacy", "target": "a"}]
    clock_data = tables(release("a"))
    clock_data["policies"] = [{"id": "pause", "project": "a", "kind": "deny", "valid_from": "2026-11-01", "valid_until": "2026-12-01"}]
    marker_data = tables(release("a", dependencies=['b; sys_platform == "linux"']), release("b"))
    return [
        chain("C1-new-policy-new-environment", tables(release("a")), args("a"), [
            step("new-exception-needs-unobserved-field", upserts={"policies": [abi]}),
            step("field-shows-violation", upserts={"environment": [env]}),
            step("field-recovers", upserts={"environment": [{**env, "value": "3"}]}),
        ]),
        chain("C2-withdrawal-and-insertion", tables(release("a", dependencies=["b>=1"]), release("b")), args("a"), [
            step("withdraw-last-candidate", deletes={"releases": ["b-1"]}),
            step("insert-new-candidate", upserts={"releases": [release("b", "2")]}),
        ]),
        chain("C3-same-value-new-provenance", tables(release("a")), args("a"), [
            step("replace-artifact-id-same-value", deletes={"releases": ["a-1"]}, upserts={"releases": [release("a", id="new-proof")]}),
            step("retract-new-proof", deletes={"releases": ["new-proof"]}),
        ]),
        chain("C4-atomic-cross-row-update", tables(release("a", dependencies=["b==2"]), release("b", "1")), args("a"), [
            step("atomic-constraint-and-pool-swap", deletes={"releases": ["b-1"]},
                 upserts={"releases": [release("a", dependencies=["b==1"]), release("b", "2")]}),
        ], note="Both complete snapshots conflict; this replay checks committed states. Engine interleaving tests separately check a commit during execution."),
        chain("C5-alias-and-policy", alias_data, args("legacy"), [
            step("irrelevant-policy-on-other-project", upserts={"policies": [{"id": "block-b", "project": "b", "kind": "deny"}]}),
            step("alias-now-points-to-blocked-project", upserts={"aliases": [{"id": "alias", "alias": "legacy", "target": "b"}]}),
            step("withdraw-policy", deletes={"policies": ["block-b"]}),
        ]),
        chain("C6-clock-without-database-write", clock_data, args("a"), [
            step("policy-begins", arguments=args("a", as_of="2026-11-01")),
            step("policy-expires", arguments=args("a", as_of="2026-12-01")),
        ]),
        chain("C7-unknown-marker-field", marker_data, args("a"), [
            step("remove-required-marker-observation", deletes={"environment": ["sys_platform"]}),
            step("provide-other-platform", upserts={"environment": [{"id": "sys_platform", "environment": "benchmark", "field": "sys_platform", "value": "win32"}]}),
        ]),
        chain("C8-finite-pool-version-addition", tables(release("a", "1")), args("a>=2"), [
            step("insert-first-admissible-version", upserts={"releases": [release("a", "2")]}),
            step("higher-version-has-unknown-dependency", upserts={"releases": [release("a", "3", ["outside-selected-universe"])]}),
        ]),
    ]


def public_chains(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    names = {
        "requests": {"requests", "urllib3", "idna", "charset-normalizer", "certifi"},
        "httpx": {"httpx", "httpcore", "anyio", "idna", "certifi", "sniffio", "h11", "typing-extensions", "exceptiongroup"},
    }
    output = []
    for root, projects in names.items():
        records = []
        for project in sorted(projects):
            candidates = sorted([row for row in catalog["records"] if row["project"] == project and row["yanked"] is False], key=lambda row: Version(row["version"]), reverse=True)
            records.extend(candidates[:2 if project == root else 1])
        data = tables_from_catalog({"records": records}, environments={"benchmark": ENVIRONMENT})
        root_candidates = sorted([row for row in data["releases"] if row["project"] == root], key=lambda row: Version(row["version"]), reverse=True)
        output.append(chain(f"public-{root}", data, args(root), [
            step("synthetic-withdrawal-of-latest-selected-root", deletes={"releases": [root_candidates[0]["id"]]}),
        ], origin="verified_public_metadata_with_synthetic_intervention",
            observed_at=catalog["observed_at"], source_records=[{"project": row["project"], "version": row["version"], "url": row["metadata_url"], "sha256": row["metadata_sha256"]} for row in records],
            note="Current observed metadata only. The deletion is a counterfactual, not a claim about actual PyPI history. Nonroot projects retain one candidate for bounded oracle enumeration."))
    return output


def random_chains(seed: int = 51, count: int = 64) -> list[dict[str, Any]]:
    rng, output = random.Random(seed), []
    for index in range(count):
        names = [f"w{index:03d}p{i}" for i in range(rng.randint(2, 4))]
        rows = []
        for project in names:
            for version in ("1", "2"):
                deps = []
                if rng.random() < .65:
                    deps.append(rng.choice(names) + rng.choice((">=1", ">=2", "<2")))
                if rng.random() < .30:
                    deps.append(rng.choice(names) + '[fast]>=1; extra == "fast"')
                if rng.random() < .08:
                    deps.append("unknown-project>=1")
                rows.append(release(project, version, deps, provides_extra=["fast"], yanked=rng.random() < .12))
        requested = names[0] + rng.choice(("", "[fast]", ">=2"))
        chosen = rng.choice(rows)
        if index % 3 == 0:
            event = step("toggle-yank", upserts={"releases": [{**chosen, "yanked": not chosen["yanked"]}]})
        elif index % 3 == 1:
            event = step("new-deny-policy", upserts={"policies": [{"id": "new-deny", "project": rng.choice(names), "kind": "deny", "versions": ">=2"}]})
        else:
            event = step("withdraw-selected-record", deletes={"releases": [chosen["id"]]})
        output.append(chain(f"random-{index:03d}", tables(*rows), args(requested), [event],
                            origin="seeded_synthetic_finite_graph", seed=seed))
    return output


def apply_reference(data: dict, event: dict) -> dict:
    """Independent plain-dictionary transition, not Store.commit."""
    result = copy.deepcopy(data)
    for table in set(event["deletes"]) | set(event["upserts"]):
        remove = set(event["deletes"].get(table, []))
        incoming = {row["id"]: copy.deepcopy(row) for row in event["upserts"].get(table, [])}
        result[table] = [row for row in result[table] if row["id"] not in remove and row["id"] not in incoming] + list(incoming.values())
    return result


def _current_citations(result: dict, data: dict) -> bool:
    available = {f"{table}:{row['id']}" for table, rows in data.items() for row in rows}
    return all(value in available for value in result["citations"])


def replay_chain(case: dict, *, modes=MODES, max_assignments: int = 200000) -> dict[str, Any]:
    engines, setup_ms = {}, {}
    for mode in modes:
        start = perf_counter()
        store = Store(copy.deepcopy(case["tables"]))
        engine = Engine(store, mode=mode)
        register_domain(engine)
        engines[mode] = store, engine
        setup_ms[mode] = (perf_counter() - start) * 1000
    data, current_args = copy.deepcopy(case["tables"]), copy.deepcopy(case["arguments"])
    records, previous = [], {}
    for event in case["steps"]:
        data = apply_reference(data, event)
        if event["arguments"] is not None:
            current_args = copy.deepcopy(event["arguments"])
        start = perf_counter()
        reference = brute_force(data, current_args, max_assignments=max_assignments)
        oracle_ms = (perf_counter() - start) * 1000
        outputs = {}
        for mode, (store, engine) in engines.items():
            start = perf_counter()
            wrote = bool(event["upserts"] or event["deletes"])
            if wrote:
                store.commit(upserts=event["upserts"], deletes=event["deletes"])
            prediction = engine.run("resolve", current_args)
            elapsed_ms = (perf_counter() - start) * 1000
            # Compare storage transitions independently of the production solver.
            store_equal = all(sorted(store.snapshot.rows(table), key=lambda row: str(row["id"])) == sorted(data[table], key=lambda row: str(row["id"])) for table in TABLES)
            witness = check_selection(data, current_args, prediction["selection"]) if prediction["status"] == "resolved" else None
            same_status = prediction["status"] == reference["status"]
            citations_current = _current_citations(prediction, data)
            valid = same_status and citations_current and store_equal and (witness is None or witness["valid"])
            prior = previous.get(mode)
            outputs[mode] = {
                "prediction": prediction, "status_matches_oracle": same_status, "witness": witness,
                "citation_rows_current": citations_current, "store_matches_reference": store_equal,
                "agreement": valid, "elapsed_ms": elapsed_ms, "revision": store.snapshot.revision,
                "stats": copy.deepcopy(engine.last_receipt["stats"]),
                "execution_policy": copy.deepcopy(engine.last_receipt.get("execution_policy")),
                "commit_stats": copy.deepcopy(store.last_commit_stats) if wrote else {},
                "same_status_and_selection_as_previous": prior is not None and (prior["status"], prior["selection"]) == (prediction["status"], prediction["selection"]),
                "citations_changed_from_previous": prior is not None and prior["citations"] != prediction["citations"],
            }
            previous[mode] = prediction
        records.append({"label": event["label"], "event": event, "arguments": copy.deepcopy(current_args),
                        "data_sha256": digest(data), "oracle": reference, "oracle_elapsed_ms": oracle_ms, "methods": outputs})
    return {**{key: value for key, value in case.items() if key not in {"tables", "steps", "arguments"}},
            "initial_data_sha256": digest(case["tables"]), "initial_tables": case["tables"],
            "setup_ms": setup_ms, "states": records}


def summarize(cases: list[dict]) -> dict[str, Any]:
    states = [state for case in cases for state in case["states"]]
    summary = {"chains": len(cases), "states": len(states), "reference_status_counts": {status: sum(state["oracle"]["status"] == status for state in states) for status in ("resolved", "conflict", "incomplete")}, "methods": {}}
    for mode in MODES:
        rows = [state["methods"][mode] for state in states]
        emitted = [row for row in rows if row["prediction"]["status"] == "resolved"]
        agreement = sum(row["agreement"] for row in rows)
        summary["methods"][mode] = {
            "agreement_count": agreement, "state_denominator": len(rows), "agreement_rate": agreement / len(rows) if rows else None,
            "resolved_witnesses_valid": sum(row["witness"]["valid"] for row in emitted), "resolved_denominator": len(emitted),
            "whole_chains_agree": sum(all(state["methods"][mode]["agreement"] for state in case["states"]) for case in cases), "chain_denominator": len(cases),
            "setup_ms": sum(case["setup_ms"][mode] for case in cases), "commit_and_query_ms": sum(row["elapsed_ms"] for row in rows),
            "task_executions": sum(row["stats"].get("task_executions", 0) for row in rows),
            "query_executions": sum(row["stats"].get("query_executions", 0) for row in rows),
            "query_scans": sum(row["stats"].get("query_scans", 0) for row in rows),
            "adaptive_full_runs": sum(row["stats"].get("adaptive_full_runs", 0) for row in rows),
            "adaptive_coarse_runs": sum(row["stats"].get("adaptive_coarse_runs", 0) for row in rows),
            "adaptive_fallback_runs": sum(row["stats"].get("adaptive_fallback_runs", 0) for row in rows),
            "same_selection_but_citations_changed": sum(row["same_status_and_selection_as_previous"] and row["citations_changed_from_previous"] for row in rows),
        }
    summary["required_modes_all_agree"] = all(summary["methods"][mode]["agreement_count"] == len(states) for mode in REQUIRED_MODES)
    return summary


def benchmark(catalog_path: Path, *, seed: int = 51, worlds: int = 64) -> dict[str, Any]:
    source_paths = [Path("src/deltaweave") / name for name in ("domain.py", "oracle.py", "engine.py", "store.py")]
    source_paths += [Path("scripts/benchmark_domain.py")]
    hashes = {str(path): file_hash(ROOT / path) for path in source_paths}
    catalog = json.loads(catalog_path.read_text())
    scenarios = controlled_chains() + public_chains(catalog) + random_chains(seed, worlds)
    start = perf_counter()
    results = [replay_chain(item) for item in scenarios]
    unchanged = all(file_hash(ROOT / path) == value for path, value in hashes.items())
    return {
        "format": "deltaweave.domain-benchmark.v2", "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_used": False, "seed": seed, "random_worlds": worlds, "source_code_sha256": hashes,
        "source_code_unchanged_during_run": unchanged, "catalog_sha256": file_hash(catalog_path),
        "catalog_scope": catalog["scope"], "environment": ENVIRONMENT,
        "measurement_contract": {
            "truth": "Independent Cartesian enumeration and witness checks; shared packaging parsing risk remains.",
            "incomplete": "An unknown/unsupported classification, not proof of no solution or a valid resolved answer.",
            "cost": "Actual local CPU setup plus Store.commit and Engine.run; oracle checking excluded from method timings and recorded separately.",
            "timing_limit": "Single local replay with concurrent workload not controlled; timings are diagnostic observations and cannot establish comparative speedups.",
            "scope": "One tracked resolver task with separately tracked relation queries; no LLM, install or runtime compatibility tests.",
            "provenance": "Citations include all search observations, not a minimal proof. Same selected values can still change citations and task output.",
            "public_data": "Two selected subuniverses from current verified public metadata; their withdrawal events are explicitly synthetic.",
            "atomicity": "Replay visits committed snapshots; concurrent mid-execution update behavior is covered by engine unit tests.",
            "positive": "Intentionally incomplete old-positive-dependency baseline; excluded from required-mode agreement gate.",
        },
        "summary": summarize(results), "cases": results,
        "total_audit_elapsed_ms": (perf_counter() - start) * 1000,
    }


def save_report(report: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data/public/pypi-catalog.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/domain-v2.json")
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--worlds", type=int, default=64)
    options = parser.parse_args(argv)
    if options.output.exists():
        parser.error("output exists; refusing to overwrite")
    if options.worlds < 0:
        parser.error("worlds must be nonnegative")
    report = benchmark(options.catalog, seed=options.seed, worlds=options.worlds)
    save_report(report, options.output)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"]["required_modes_all_agree"] and report["source_code_unchanged_during_run"] else 2


if __name__ == "__main__":
    sys.exit(main())
