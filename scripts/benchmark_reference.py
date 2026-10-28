"""Freeze 16 synthetic update chains, audit them on CPU, then compare views.

python scripts/benchmark_reference.py prepare --output experiments/reference-v1
python scripts/benchmark_reference.py run --bundle experiments/reference-v1 \
  --view reference --model .cache/deepseek-r1-distill-qwen-1.5b \
  --adapter runs/behavior/adapter --output runs/reference-confirmation

No model is loaded by prepare. Outputs must be new paths. The optional run
keeps every session receipt; a blocked model response is never repaired.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random

from deltaweave.domain import TABLES, register_domain
from deltaweave.engine import Engine
from deltaweave.model import evaluation_config
from deltaweave.oracle import brute_force, check_selection
from deltaweave.session import KnowledgeSession, REFERENCE_PROMPT_VERSION
from deltaweave.store import Store

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261031
VIEWS = ("full", "reference_context", "reference")
FAMILIES = ("new_policy_environment", "withdrawal_insertion", "same_value_provenance",
            "atomic_cross_rows", "alias_policy", "clock_without_write",
            "unknown_field", "finite_pool")
SOURCE_FILES = tuple(f"src/deltaweave/{name}.py" for name in (
    "engine", "store", "domain", "oracle", "session", "evidence", "provenance",
    "learning_data", "model", "cli")) + (
    "scripts/audit_tokenizer.py", "scripts/benchmark_reference.py",
    "research/protocol-reference-v1.md")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def release(project, version="1.0", dependencies=(), **extra):
    return {"id": f"{project}@{version}", "project": project, "version": version,
            "requires_python": ">=3.8", "requires_dist": list(dependencies),
            "provides_extra": [], "yanked": False, **extra}


def world(*records):
    result = {name: [] for name in TABLES}
    result["releases"] = list(records)
    result["projects"] = [{"id": name, "project": name, "complete": True}
                          for name in sorted({r["project"] for r in records})]
    result["environment"] = [
        {"id": field, "environment": "confirmation", "field": field, "value": value}
        for field, value in {"python_full_version": "3.12.4", "sys_platform": "linux"}.items()]
    return result


def event(label, *, upserts=None, deletes=None, query=None, published=True):
    return {"label": label, "upserts": upserts or {}, "deletes": deletes or {},
            "query": query, "publication_ready": published}


def build_cases(seed=SEED):
    """New instances/combinations of known mechanisms, not new mechanism claims."""
    rng, cases = random.Random(seed), []
    for family_index, family in enumerate(FAMILIES):
        for replica in range(2):
            stem = f"ref{family_index}{replica}-{rng.randrange(16**6):06x}"
            a, b = stem + "-app", stem + "-lib"
            query = {"requirements": [a], "environment": "confirmation", "as_of": "2027-02-01"}
            initial = event("initial")
            if family == "new_policy_environment":
                data = world(release(a, dependencies=[b + ">=1"]), release(b))
                policy = {"id": "platform-policy", "project": b, "kind": "require_env",
                          "field": "crypto_level", "operator": "version", "specifier": ">=4"}
                events = [initial, event("new-transitive-policy", upserts={"policies": [policy]}),
                          event("provide-new-field", upserts={"environment": [{"id": "crypto", "environment": "confirmation",
                              "field": "crypto_level", "value": "3" if replica == 0 else "5"}]})]
            elif family == "withdrawal_insertion":
                data = world(release(a, dependencies=[b + ">=2"]), release(b, "2.0"))
                events = [initial, event("withdraw-only-admissible", deletes={"releases": [b + "@2.0"]}),
                          event("insert-replacement", upserts={"releases": [release(b, "3.1")]})]
            elif family == "same_value_provenance":
                data = world(release(a, dependencies=[b]), release(b))
                replacement = release(b, id=b + "@replacement")
                events = [initial, event("same-value-new-artifact", deletes={"releases": [b + "@1.0"]}, upserts={"releases": [replacement]}),
                          event("retract-new-artifact", deletes={"releases": [replacement["id"]]})]
            elif family == "atomic_cross_rows":
                data = world(release(a, dependencies=[b + "==3"]), release(b, "2.0"))
                events = [initial, event("swap-demand-and-candidate-atomically", deletes={"releases": [b + "@2.0"]},
                    upserts={"releases": [release(a, dependencies=[b + "==2"]), release(b, "3.0")]}),
                    event("insert-matching-version", upserts={"releases": [release(b, "2.0")]})]
            elif family == "alias_policy":
                data = world(release(a), release(b, "4.0"))
                alias = {"id": "routing", "alias": stem + "-legacy", "target": a}
                data["aliases"] = [alias]
                query["requirements"] = [alias["alias"]]
                events = [event("not-yet-published", published=False),
                    event("publish-with-unrelated-deny", upserts={"policies": [{"id": "deny-lib", "project": b, "kind": "deny", "versions": ">=4"}]}),
                    event("alias-enters-denied-project", upserts={"aliases": [{**alias, "target": b}]})]
            elif family == "clock_without_write":
                data = world(release(a, dependencies=[b]), release(b))
                data["policies"] = [{"id": "embargo", "project": b, "kind": "deny", "valid_from": "2027-03-01", "valid_until": "2027-04-01"}]
                events = [initial, event("time-enters-policy", query={**query, "as_of": "2027-03-15"}),
                          event("time-exits-policy", query={**query, "as_of": "2027-04-01"})]
            elif family == "unknown_field":
                data = world(release(a, dependencies=[b + '; sys_platform == "linux"']), release(b))
                field = "sys_platform" if replica == 0 else "python_full_version"
                events = [initial, event("remove-explicit-environment", deletes={"environment": [field]}),
                          event("provide-changed-environment", upserts={"environment": [{"id": field,
                              "environment": "confirmation", "field": field,
                              "value": "win32" if replica == 0 else "3.13.1"}]})]
            else:
                data = world(release(a), release(b))
                query["requirements"] = [a + ">=2"]
                events = [initial, event("add-first-admissible", upserts={"releases": [release(a, "2.4", [b])]}),
                          event("withdraw-completeness-not-release", deletes={"projects": [b]})]
            # Replica 1 has six additional real transitive dependencies. Keep
            # these dependencies when an event replaces the root metadata.
            if replica == 1:
                leaves = [stem + f"-leaf{i}" for i in range(6)]
                for name in leaves:
                    data["projects"].append({"id": name, "project": name, "complete": True})
                    data["releases"].extend([release(name), release(name, "2.0")])
                for record in data["releases"] + [r for e in events for r in e["upserts"].get("releases", [])]:
                    if record["project"] == a:
                        record["requires_dist"] += [name + ">=1" for name in leaves]
            cases.append({"id": stem, "family": family, "replica": replica,
                          "origin": "synthetic_confirmation_not_real_package_incident",
                          "tables": data, "query": query, "prevalidate": replica == 1, "events": events})
    return cases


def transition(tables, change):
    """Independent plain-data replay; do not use Store for the oracle world."""
    result = deepcopy(tables)
    for table in set(change["deletes"]) | set(change["upserts"]):
        inserted = {row["id"]: deepcopy(row) for row in change["upserts"].get(table, [])}
        removed = set(change["deletes"].get(table, [])) | set(inserted)
        result[table] = [row for row in result[table] if row["id"] not in removed] + list(inserted.values())
    return result


def expected_action(state):
    """Independent interface specification; no call to production decide()."""
    if not state["publication_ready"]:
        return {"action": "wait", "arguments": {"reason": "snapshot_not_ready"}}
    if state["python"] is None:
        return {"action": "ask", "arguments": {"field": "python"}}
    obs = state["observation"]
    if obs is None or obs["validated_at"] != state["target_revision"]:
        return {"action": "refresh", "arguments": {"tool": "resolve"}}
    if obs["status"] == "incomplete":
        return {"action": "abstain", "arguments": {"reason": "incomplete_catalog"}}
    return {"action": "answer", "arguments": {key: obs[key] for key in ("status", "selection", "citations")}}


def reference_valid(step, result):
    proof = step.get("proof_record")
    if not proof:
        return False
    try:
        semantic = proof["semantic"]
        payload = {key: proof[key] for key in ("query", "runtime_binding", "resolver_result", "solver_receipt")}
        full = {"action": "answer", "arguments": {key: proof["resolver_result"][key] for key in ("status", "selection", "citations")}}
        receipt = proof["solver_receipt"]
        return (proof["content_sha256"] == digest(semantic)
            and proof["handle"] == "h" + digest(semantic)[:16]
            and proof["proof_sha256"] == digest(payload)
            and proof["runtime_binding"] == result["runtime_binding"]
            and proof["query"] == result["query"]
            and semantic == {"query": result["query"], "answer": full}
            and receipt["name"] == "resolve" and receipt["status"] == "valid"
            and receipt["validated_at"] == result["snapshot_revision"]
            and receipt["args"] == result["query"]
            and receipt["value_sha256"] == digest(proof["resolver_result"])
            and receipt["version"] == result["runtime_binding"]["registry_versions"]["resolve"]
            and proof["resolver_result"]["status"] in {"resolved", "conflict"}
            and step["raw_action"] == expected_action(step["state"])
            and step["raw_action"]["arguments"]["selection"] == {"answer_ref": proof["handle"]}
            and step["raw_action"]["arguments"]["citations"] == ["proof:" + proof["handle"]]
            and step["expanded_answer"] == full)
    except (KeyError, TypeError):
        return False


def assess(result, tables, query, published, reference):
    available = {f"{name}:{row['id']}" for name, rows in tables.items() for row in rows}
    python_rows = [r for r in tables["environment"] if r["environment"] == query["environment"] and r["field"] == "python_full_version"]
    expected_terminal = ("wait" if not published else "ask" if len(python_rows) != 1 or python_rows[0].get("value") is None
                         else "abstain" if reference["status"] == "incomplete" else "answer")
    final = result["answer"]
    valid = bool(final and final["action"] == expected_terminal)
    full_answer = bool(final and final["action"] == "answer")
    if valid and full_answer:
        payload = final["arguments"]
        valid = payload["status"] == reference["status"] and set(payload["citations"]) <= available
        if payload["status"] == "resolved":
            valid = valid and check_selection(tables, query, payload["selection"])["valid"]
            valid = valid and all(any(f"releases:{r['id']}" in payload["citations"] for r in tables["releases"]
                if r["project"] == name and r["version"] == version) for name, version in payload["selection"].items())
        else:
            valid = valid and payload["selection"] == {}
    elif valid:
        expected = {"wait": {"reason": "snapshot_not_ready"}, "ask": {"field": "python"},
                    "abstain": {"reason": "incomplete_catalog"}}[expected_terminal]
        valid = final["arguments"] == expected
    step_checks = []
    for step in result["steps"]:
        raw_valid = (step["raw_action"] == expected_action(step["state"])
                     and (step["receipt"]["kind"] == "rule" or step["receipt"].get("completed") is True))
        ref_answer = (result["evidence_view"] != "full" and step["raw_action"] is not None
                      and step["raw_action"].get("action") == "answer")
        integrity = reference_valid(step, result) if ref_answer and step["accepted"] else None
        expanded = bool(ref_answer and step.get("expanded_answer") is not None)
        step_checks.append({"raw_contract_valid": raw_valid, "accepted": step["accepted"],
                            "reference_answer_attempt": ref_answer,
                            "reference_expanded": expanded,
                            "reference_integrity": integrity,
                            "wrong_reference_accepted": bool(expanded and (not raw_valid or not integrity)),
                            "fresh_model": step["receipt"]["kind"] == "model" and not step["receipt"]["cache_hit"]})
    return {"expected_terminal": expected_terminal, "oracle_status": reference["status"],
            "oracle_appropriate_terminal": valid, "full_answer_emitted": full_answer,
            "full_answer_oracle_valid": valid if full_answer else None, "step_checks": step_checks}


def summarize(records):
    steps = [(step, check) for record in records for step, check in zip(record["result"]["steps"], record["audit"]["step_checks"])]
    fresh = [(step, check) for step, check in steps if check["fresh_model"]]
    accepted_refs = [check for _, check in steps if check["reference_expanded"]]
    full = [r for r in records if r["audit"]["full_answer_emitted"]]
    def ratio(n, d):
        return {"numerator": n, "denominator": d, "rate": n / d if d else None}
    costs = {}
    for field in ("input_tokens", "new_tokens", "seconds"):
        values = [step["receipt"].get(field) for step, _ in fresh]
        missing = sum(not isinstance(v, (int, float)) or isinstance(v, bool) for v in values)
        costs[field] = {"total": sum(values) if not missing else None, "missing_receipts": missing,
                        "fresh_generation_denominator": len(fresh)}
    return {"sessions": len(records),
        "fresh_raw_contract": ratio(sum(check["raw_contract_valid"] for _, check in fresh), len(fresh)),
        "all_sessions_completed": ratio(sum(r["result"]["status"] == "completed" for r in records), len(records)),
        "oracle_appropriate_terminal": ratio(sum(r["audit"]["oracle_appropriate_terminal"] for r in records), len(records)),
        "wrong_reference_acceptance": ratio(sum(c["wrong_reference_accepted"] for c in accepted_refs), len(accepted_refs)),
        "full_answer_oracle_agreement": ratio(sum(r["audit"]["full_answer_oracle_valid"] for r in full), len(full)),
        "cache_hits": sum(step["receipt"]["kind"] == "model" and step["receipt"]["cache_hit"] for step, _ in steps),
        "rule_actions": sum(step["receipt"]["kind"] == "rule" for step, _ in steps),
        "terminal_statuses": dict(Counter(r["result"]["status"] for r in records)),
        "expected_terminal_actions": dict(Counter(r["audit"]["expected_terminal"] for r in records)),
        "fresh_generation_cost": costs}


def run_view(cases, config, view, *, policy=None, policy_identity=None, on_record=None):
    if view not in VIEWS:
        raise ValueError("Unknown evidence view")
    records = []
    for case in cases:
        data, query = deepcopy(case["tables"]), deepcopy(case["query"])
        store = Store(data)
        engine = Engine(store, mode=config["engine_mode"])
        register_domain(engine)
        session = KnowledgeSession(engine, policy=policy, policy_identity=policy_identity,
            evidence_view=view, semantic_cache=config["semantic_cache"])
        for index, change in enumerate(case["events"]):
            data = transition(data, change)
            if change["query"] is not None:
                query = deepcopy(change["query"])
            if change["upserts"] or change["deletes"]:
                store.commit(upserts=change["upserts"], deletes=change["deletes"])
            if any(sorted(store.snapshot.rows(t), key=lambda r: r["id"]) != sorted(data[t], key=lambda r: r["id"]) for t in TABLES):
                raise AssertionError("Production store differs from independent transition")
            reference = brute_force(data, query, max_assignments=config["oracle_max_assignments"])
            result = session.request(query, max_actions=config["max_actions"], prevalidate=case["prevalidate"],
                                     publication_ready=change["publication_ready"])
            record = {"case_id": case["id"], "family": case["family"], "replica": case["replica"],
                      "event_index": index, "event": change, "query": deepcopy(query),
                      "world_sha256": digest(data), "oracle": reference, "result": result,
                      "audit": assess(result, data, query, change["publication_ready"], reference)}
            records.append(record)
            if on_record:
                on_record(record)
    return {"view": view, "records": records, "summary": summarize(records)}


def configuration():
    return {"schema": "deltaweave.reference-confirmation.v1", "seed": SEED,
            "views": list(VIEWS), "chains": 16, "sessions_per_view": 48,
            "engine_mode": "coarse", "max_actions": 3, "semantic_cache": True,
            "oracle_max_assignments": 200000, "reference_prompt_version": REFERENCE_PROMPT_VERSION,
            "generation": evaluation_config(max_input_tokens=2048),
            "scope": "Closed one-reference transport/control, not model evidence reasoning; synthetic new instances of known update families."}


def source_hashes():
    return {name: file_hash(ROOT / name) for name in SOURCE_FILES}


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cases, config = build_cases(), configuration()
    write_new(output / "cases.json", cases)
    write_new(output / "config.json", config)
    manifest = {"schema": config["schema"], "frozen_at": datetime.now(timezone.utc).isoformat(),
                "inputs": {name: file_hash(output / name) for name in ("cases.json", "config.json")},
                "source_sha256": source_hashes()}
    # Freeze inputs/config/source before oracle/rule execution; no model call.
    write_new(output / "manifest.json", manifest)
    views = {view: run_view(cases, config, view) for view in VIEWS}
    equal_answers = all([r["result"]["answer"] for r in views[v]["records"]] ==
                        [r["result"]["answer"] for r in views["full"]["records"]] for v in VIEWS)
    passed = equal_answers and all(v["summary"]["oracle_appropriate_terminal"]["numerator"] == 48
        and v["summary"]["all_sessions_completed"]["numerator"] == 48
        and v["summary"]["wrong_reference_acceptance"]["numerator"] == 0 for v in views.values())
    audit = {"passed": passed, "model_used": False, "manifest_sha256": file_hash(output / "manifest.json"),
             "identical_full_answers_across_views": equal_answers, "views": views}
    write_new(output / "rule-audit.json", audit)
    write_new(output / "audit-seal.json", {"manifest_sha256": file_hash(output / "manifest.json"),
               "rule_audit_sha256": file_hash(output / "rule-audit.json"), "passed": passed})
    return audit


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["inputs"].items():
        if name not in {"cases.json", "config.json"} or file_hash(bundle / name) != expected:
            raise ValueError("Frozen input/config changed")
    if set(manifest["inputs"]) != {"cases.json", "config.json"} or manifest["source_sha256"] != source_hashes():
        raise ValueError("Frozen source set or source bytes changed")
    seal = json.loads((bundle / "audit-seal.json").read_text())
    if seal != {"manifest_sha256": file_hash(bundle / "manifest.json"), "rule_audit_sha256": file_hash(bundle / "rule-audit.json"), "passed": True}:
        raise ValueError("Rule audit not sealed/passed")
    audit = json.loads((bundle / "rule-audit.json").read_text())
    config = json.loads((bundle / "config.json").read_text())
    cases = json.loads((bundle / "cases.json").read_text())
    if not audit["passed"] or config != configuration() or cases != build_cases():
        raise ValueError("Bundle differs from registered protocol")
    return cases, config, file_hash(bundle / "manifest.json")


def run_model(bundle, output, view, model, adapter=None):
    cases, config, manifest_hash = verify_bundle(bundle)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    from deltaweave.cli import make_local_policy
    policy, identity, model_binding = make_local_policy(Path(model), None if adapter is None else Path(adapter))
    if model_binding["config"] != config["generation"]:
        raise ValueError("Loaded generation settings differ from frozen settings")
    write_new(output / "run-binding.json", {"manifest_sha256": manifest_hash, "view": view, "model": model_binding})
    def checkpoint(record):
        write_new(output / f"{record['case_id']}-{record['event_index']}.json", record)
        print(json.dumps({"case": record["case_id"], "event": record["event_index"], "status": record["result"]["status"]}), flush=True)
    result = run_view(cases, config, view, policy=policy, policy_identity=identity, on_record=checkpoint)
    verify_bundle(bundle)  # Detect source/data edits during a long model run.
    report = {"schema": config["schema"], "manifest_sha256": manifest_hash,
              "model_used": True, "model": model_binding, **result,
              "scope": config["scope"], "failures_retained": True}
    write_new(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--bundle", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--view", choices=VIEWS, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        audit = prepare(args.output)
        print(json.dumps({"passed": audit["passed"], "views": {v: x["summary"] for v, x in audit["views"].items()}}, indent=2))
        if not audit["passed"]:
            raise SystemExit(1)
    else:
        report = run_model(args.bundle, args.output, args.view, args.model, args.adapter)
        print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
