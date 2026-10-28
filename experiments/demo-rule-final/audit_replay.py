"""Independently audit this rule-only SQLite replay; never invoke its solver.

Run from the source checkout after editable installation. The output is created
once; an existing audit is never overwritten. The oracle shares packaging's
PEP 440/508 parser with production, so parser errors remain a common-mode risk.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from deltaweave.oracle import brute_force, check_selection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def normalized(tables):
    return {name: sorted(rows, key=lambda row: canonical(row["id"]))
            for name, rows in tables.items()}


def main():
    report_path = HERE / "report.json"
    report = json.loads(report_path.read_text())
    catalog_path = ROOT / "data/public/pypi-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    # Rebuild declared rows independently; no Store.commit, source loader,
    # tracked query, production solver or CLI fixture helper is used here.
    tables = {name: [] for name in
              ("projects", "releases", "aliases", "policies", "environment")}
    for item in catalog["records"]:
        row = deepcopy(item)
        project = re.sub(r"[-_.]+", "-", row["project"]).lower()
        row.update(project=project, id=f"{project}=={row['version']}")
        tables["releases"].append(row)
    tables["projects"] = [{"id": name, "project": name, "complete": True}
                           for name in sorted({r["project"] for r in tables["releases"]})]
    environment = {
        "python_full_version": "3.11.9", "python_version": "3.11",
        "implementation_name": "cpython", "implementation_version": "3.11.9",
        "os_name": "posix", "sys_platform": "darwin", "platform_machine": "arm64",
        "platform_system": "Darwin", "platform_release": "declared",
        "platform_version": "declared", "platform_python_implementation": "CPython",
    }
    tables["environment"] = [
        {"id": f"declared-mac:{key}", "environment": "declared-mac",
         "field": key, "value": value} for key, value in environment.items()]
    events = [
        {"name": "initial", "upserts": {}, "deletes": {}},
        {"name": "new_org_constraint", "upserts": {"policies": [
            {"id": "demo-deny", "project": "requests", "kind": "deny", "versions": ">=2.32"}]}, "deletes": {}},
        {"name": "unrelated_project_policy", "upserts": {"policies": [
            {"id": "unrelated-deny", "project": "anyio", "kind": "deny", "versions": ">=4"}]}, "deletes": {}},
        {"name": "remove_org_constraint", "upserts": {}, "deletes": {"policies": ["demo-deny"]}},
        {"name": "withdraw_selected_file", "upserts": {}, "deletes": {"releases": ["requests==2.32.5"]}},
        {"name": "remove_completeness_certificate", "upserts": {}, "deletes": {"projects": ["requests"]}},
    ]
    query = {"requirements": ["requests>=2.31"], "environment": "declared-mac", "as_of": "2026-10-28"}
    versions = {"controller_python": "1", "resolve": "finite-domain-v1",
                "training_sample": "finite-resolver-answer-v1"}
    checks = {"six_events": len(report["runs"]) == 6,
              "catalog_hash": report["catalog_sha256"] == hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
              "model_not_loaded": report["model"] is None}
    rows, previous_content = [], None
    for revision, (run, event) in enumerate(zip(report["runs"], events)):
        for name, ids in event["deletes"].items():
            tables[name] = [row for row in tables[name] if row["id"] not in ids]
        for name, replacements in event["upserts"].items():
            for replacement in replacements:
                tables[name] = [row for row in tables[name] if row["id"] != replacement["id"]]
                tables[name].append(deepcopy(replacement))
        result, binding = run["result"], run["source_binding"]
        manifest = json.loads((HERE / run["dataset_manifest"]).read_text())
        sample = manifest["samples"][0]
        content = {key: value for key, value in sample.items()
                   if key not in {"snapshot_revision", "content_sha256"}}
        semantic = {key: manifest[key] for key in
                    ("snapshot_revision", "snapshot_sha256", "runtime_binding", "samples")}
        snapshot_hash = digest({"tables": normalized(tables), "context": {}})
        runtime = {"snapshot_revision": revision, "snapshot_sha256": snapshot_hash,
                   "registry_versions": versions, "registry_fingerprint": digest(versions)}
        oracle = brute_force(tables, query)
        action, answer = result["answer"]["action"], result["answer"]["arguments"]
        if oracle["status"] == "resolved":
            witness = check_selection(tables, query, answer.get("selection", {}))
            available = {f"{name}:{row['id']}" for name, records in tables.items() for row in records}
            citations = set(answer.get("citations", []))
            answer_valid = (action == "answer" and answer.get("status") == "resolved"
                            and witness["valid"] and citations <= available
                            and all(f"releases:{name}=={version}" in citations
                                    for name, version in answer["selection"].items()))
        else:
            answer_valid = oracle["status"] == "incomplete" and result["answer"] == {
                "action": "abstain", "arguments": {"reason": "incomplete_catalog"}}
        changed = [] if canonical(content) == previous_content else ["requests-baseline"]
        previous_content = canonical(content)
        local = {
            "event_is_declared_overlay": run["event"] == event,
            "completed": result["status"] == "completed" and result["closed_loop_contract_success"] is True,
            "oracle_answer_valid": answer_valid,
            "rule_not_model": result["raw_model_success"] is None
                and result["fresh_model_calls"] == 0 and result["model_cache_hits"] == 0
                and all(step["receipt"]["kind"] == "rule" for step in result["steps"]),
            "source_identity": binding["source_id"] == report["source"]["source_id"],
            "source_local_revision_mapping": binding["source_revision"] == binding["store_revision"] == result["snapshot_revision"] == revision,
            "source_snapshot_id": binding["snapshot_id"] == f"{binding['source_id']}@{revision}",
            "snapshot_hash_all_equal": binding["published_snapshot_sha256"] == result["snapshot_sha256"] == manifest["snapshot_sha256"] == snapshot_hash,
            "runtime_binding_equal": result["runtime_binding"] == manifest["runtime_binding"] == runtime,
            "manifest_hash": digest(semantic) == run["dataset_sha256"] == manifest["manifest_sha256"],
            "sample_content_and_target": digest(content) == sample["content_sha256"]
                and sample["target"] == result["answer"] and sample["query"] == query
                and sample["observation"]["status"] == oracle["status"],
            "changed_sample_ids": changed == run["changed_sample_ids"] == manifest["changed_sample_ids"],
            "training_receipt_binding": manifest["receipts"]["requests-baseline"]["name"] == "training_sample"
                and manifest["receipts"]["requests-baseline"]["validated_at"] == revision
                and manifest["receipts"]["requests-baseline"]["value_sha256"] == digest(content),
            "tool_receipt_binding": all(receipt["validated_at"] == revision for receipt in result["tool_receipts"])
                and all(receipt["args"] == query and receipt["value_sha256"] == digest(sample["observation"])
                        for receipt in result["tool_receipts"] if receipt["name"] == "resolve"),
        }
        if revision:
            local["source_commit_receipt"] = run["source_receipt"] == {
                "source_id": binding["source_id"], "revision": revision,
                "request_id": f"demo-{revision}", "duplicate": False}
        rows.append({"event": event["name"], "oracle_status": oracle["status"],
                     "oracle_assignments_examined": oracle["assignments_examined"],
                     "answer_action": action, "selected_requests": answer.get("selection", {}).get("requests"),
                     "changed_sample_ids": changed, "checks": local, "passed": all(local.values())})
    # Read the actual database with a read-only URI and fixed SQL. SQLite adapter
    # loading/outbox code is not part of this independent reconstruction.
    with sqlite3.connect(f"file:{HERE / 'knowledge.sqlite'}?mode=ro", uri=True) as db:
        names = [row[0] for row in db.execute("SELECT name FROM relations")]
        persisted = {name: [] for name in names}
        for name, payload in db.execute("SELECT relation_name,payload FROM records"):
            persisted[name].append(json.loads(payload))
        identity, revision, context = db.execute("SELECT source_id,revision,context FROM metadata").fetchone()
        outbox = db.execute("SELECT seq,request_id,request_sha256,payload FROM outbox ORDER BY seq").fetchall()
    checks.update(final_database_matches_reconstructed_events=normalized(persisted) == normalized(tables),
                  final_source_revision=revision == 5, final_source_identity=identity == report["source"]["source_id"],
                  final_context=json.loads(context) == {}, contiguous_outbox=[row[0] for row in outbox] == list(range(1, 6)))
    for seq, request, request_hash, payload in outbox:
        expected = {"expected_revision": seq - 1, "upserts": events[seq]["upserts"], "deletes": events[seq]["deletes"]}
        checks[f"outbox_{seq}"] = request == f"demo-{seq}" and json.loads(payload) == expected and request_hash == digest(expected)
    sources = [f"src/deltaweave/{name}.py" for name in
               ("cli", "domain", "oracle", "sqlite_source", "session", "lineage", "engine", "store", "provenance")]
    sources.append(str(Path(__file__).resolve().relative_to(ROOT)))
    audit = {"format": "deltaweave.demo-independent-audit.v1", "passed": all(checks.values()) and all(row["passed"] for row in rows),
             "model_used": False, "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
             "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
             "method": "Independent public catalog/event reconstruction without production converter, Store.commit, source loader or solver; brute-force oracle and witness checks; direct read-only SQLite/outbox checks; independent canonical hashes for snapshots, samples and manifests. Shared packaging parser remains a common-mode risk. Citation row existence and selected-release support are checked, not minimal proof completeness.",
             "rows": rows, "checks": checks}
    with (HERE / "independent-audit.json").open("x") as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"passed": audit["passed"], "states": len(rows),
                      "failed": {row["event"]: [key for key, value in row["checks"].items() if not value]
                                 for row in rows if not row["passed"]}}, indent=2))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
