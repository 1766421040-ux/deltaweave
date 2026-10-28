"""Local, read-only knowledge demo. Model execution is an explicit option."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from .domain import register_domain, tables_from_catalog
from .engine import Engine
from .lineage import TrainingLedger
from .session import KnowledgeSession, REFERENCE_PROMPT_VERSION
from .sqlite_source import SQLiteSource, SQLitePublisher
from .store import Store, canonical

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = {"python_full_version": "3.11.9", "python_version": "3.11", "implementation_name": "cpython",
               "implementation_version": "3.11.9", "os_name": "posix", "sys_platform": "darwin",
               "platform_machine": "arm64", "platform_system": "Darwin", "platform_release": "declared",
               "platform_version": "declared", "platform_python_implementation": "CPython"}


def make_local_policy(model_path, adapter_path):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from .model import HFPolicy, model_provenance, adapter_provenance, evaluation_config, load_tokenizer, case_seed, contract, source_provenance
    base = model_provenance(model_path)
    adapter = adapter_provenance(adapter_path, base)
    config = evaluation_config(max_input_tokens=2048)
    identity = hashlib.sha256(canonical({"base": base, "adapter": adapter, "config": config,
                                         "source": source_provenance()}).encode()).hexdigest()
    policy = HFPolicy(model_path, adapter_path, load_tokenizer(model_path), config)
    def predict(prompt):
        prediction = policy.predict(prompt, case_seed(prompt, config["seed"]))
        try:
            parsed = contract.parse_generated_response(prediction["raw"])
        except ValueError:
            parsed = None
        return parsed, prediction
    return predict, identity, {"base": base, "adapter": adapter, "config": config,
                               "identity": identity, "model_load_seconds": policy.load_seconds}


def demo(output, *, mode="coarse", model_path=None, adapter_path=None, evidence_view="full"):
    source_sha256 = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ("src/deltaweave/cli.py", "src/deltaweave/session.py", "src/deltaweave/evidence.py",
                     "src/deltaweave/engine.py", "src/deltaweave/store.py", "src/deltaweave/domain.py",
                     "src/deltaweave/provenance.py", "src/deltaweave/lineage.py", "src/deltaweave/sqlite_source.py",
                     "src/deltaweave/learning_data.py")}
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new demo directory; old evidence is never overwritten")
    output.mkdir(parents=True)
    catalog_path = ROOT / "data/public/pypi-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    tables = tables_from_catalog(catalog, environments={"declared-mac": ENVIRONMENT})
    source = SQLiteSource.create(output / "knowledge.sqlite", tables)
    publisher = SQLitePublisher.load(source)
    engine = Engine(publisher.store, mode=mode); register_domain(engine)
    ledger = TrainingLedger(engine)
    query = {"requirements": ["requests>=2.31"], "environment": "declared-mac", "as_of": "2026-10-28"}
    ledger.add("requests-baseline", query)
    if model_path is None:
        policy, identity, model_binding = None, "rule-v1", None
    else:
        policy, identity, model_binding = make_local_policy(model_path, adapter_path)
    session = KnowledgeSession(engine, policy=policy, policy_identity=identity, evidence_view=evidence_view)
    events = [
        {"name": "initial", "upserts": {}, "deletes": {}},
        {"name": "new_org_constraint", "upserts": {"policies": [{"id": "demo-deny", "project": "requests", "kind": "deny", "versions": ">=2.32"}]}, "deletes": {}},
        {"name": "unrelated_project_policy", "upserts": {"policies": [{"id": "unrelated-deny", "project": "anyio", "kind": "deny", "versions": ">=4"}]}, "deletes": {}},
        {"name": "remove_org_constraint", "upserts": {}, "deletes": {"policies": ["demo-deny"]}},
        {"name": "withdraw_selected_file", "upserts": {}, "deletes": {"releases": ["requests==2.32.5"]}},
        {"name": "remove_completeness_certificate", "upserts": {}, "deletes": {"projects": ["requests"]}},
    ]
    runs = []
    for index, event in enumerate(events):
        source_receipt = None
        if index:
            source_receipt = source.commit(upserts=event["upserts"], deletes=event["deletes"],
                                           expected_revision=publisher.source_revision,
                                           request_id=f"demo-{index}")
            publisher.poll()
        # Validate through the shared graph before answering after the first
        # query. The raw model is still allowed to fail and be explicitly blocked.
        result = session.request(query, prevalidate=True)
        source_binding = publisher.binding(result["snapshot_revision"])
        if source_binding["published_snapshot_sha256"] != result["snapshot_sha256"]:
            raise RuntimeError("Published database snapshot does not match the session receipt")
        manifest_path, manifest = ledger.freeze(output / "datasets")
        runs.append({"event": event, "source_receipt": source_receipt, "source_binding": source_binding, "result": result,
                     "dataset_manifest": str(manifest_path.relative_to(output)),
                     "dataset_sha256": manifest["manifest_sha256"], "changed_sample_ids": manifest["changed_sample_ids"]})
        print(json.dumps({"event": event["name"], "status": result["status"], "answer": result["answer"]}, ensure_ascii=False), flush=True)
    report = {"schema_version": 1, "engine": mode, "model": model_binding, "evidence_view": evidence_view,
              "reference_prompt_version": None if evidence_view == "full" else REFERENCE_PROMPT_VERSION,
              "source_sha256": source_sha256,
              "source": {"kind": "managed_sqlite", "source_id": publisher.source_id, "path": "knowledge.sqlite"},
              "catalog_sha256": hashlib.sha256(catalog_path.read_bytes()).hexdigest(), "runs": runs,
              "stats": engine.stats, "scope": "Actual local SQLite updates and optional local model. All events after initial are synthetic experimental overlays, not real PyPI incidents."}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("demo", help="Replay durable SQLite changes, tool traces and immutable training datasets")
    d.add_argument("--output", type=Path, required=True)
    d.add_argument("--mode", choices=("full", "coarse", "predicate", "adaptive"), default="coarse")
    d.add_argument("--model", type=Path, help="Explicit local official model directory; omitted runs strong rule policy")
    d.add_argument("--adapter", type=Path, help="Optional verified trained adapter directory")
    d.add_argument("--evidence-view", choices=("full", "reference", "reference_context"), default="full",
                   help="Full payload copying, or optional exact reference followed by verified host expansion")
    r = sub.add_parser("resolve", help="Resolve declared constraints using the selected public catalog")
    r.add_argument("requirements", nargs="+")
    r.add_argument("--as-of", default="2026-10-28")
    args = parser.parse_args()
    if args.command == "demo":
        if args.adapter and not args.model:
            parser.error("--adapter requires --model")
        demo(args.output, mode=args.mode, model_path=args.model, adapter_path=args.adapter,
             evidence_view=args.evidence_view)
    else:
        catalog = json.loads((ROOT / "data/public/pypi-catalog.json").read_text())
        engine = Engine(Store(tables_from_catalog(catalog, environments={"declared-mac": ENVIRONMENT})))
        register_domain(engine)
        print(json.dumps(engine.run("resolve", {"requirements": args.requirements, "environment": "declared-mac", "as_of": args.as_of}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
