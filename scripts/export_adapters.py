"""Validate completed runs and copy byte-identical public adapters, CPU only.

This does not publish to GitHub, load a model, redact a configuration, or infer
quality from successful training. Missing final records are explicitly pending.
"""
from __future__ import annotations

from collections import Counter
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deltaweave import model

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json", "training_manifest.json")
RUN_NAMES = {"qa": "train_qa.jsonl", "behavior": "train_behavior.jsonl"}


class ExportPending(RuntimeError):
    pass


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=model.contract._unique_object,
                      parse_constant=model.contract._reject_constant)


def public_path_audit(value, location="root"):
    """Fail rather than silently rewrite a file whose hash binds an experiment."""
    if isinstance(value, dict):
        for key, item in value.items():
            public_path_audit(key, location + ".key")
            public_path_audit(item, location + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            public_path_audit(item, f"{location}[{index}]")
    elif isinstance(value, str):
        if (value.startswith(("/", "~", "file://")) or re.match(r"^[A-Za-z]:[\\/]", value)
                or any(prefix in value for prefix in ("/Users/", "/home/", "/private/", "\\Users\\"))):
            raise ValueError(f"Local absolute path in public metadata at {location}; no automatic redaction performed")


def validate_run(run_dir: Path, name: str, project_root: Path = ROOT):
    """Check reports against bytes/data/source, without requiring base weights."""
    run_dir, project_root = Path(run_dir), Path(project_root)
    if name not in RUN_NAMES:
        raise ValueError("Expected the registered qa or behavior training route")
    required = ("summary.json", "run_manifest.json", "loss-log.jsonl",
                *("adapter/" + item for item in ADAPTER_FILES))
    missing = [item for item in required if not (run_dir / item).is_file()]
    if missing:
        raise ExportPending(f"{name}: completed artifacts not yet available: {', '.join(missing)}")
    if (run_dir / "failure.json").exists():
        raise ValueError(f"{name}: run contains a failure receipt")
    summary = read_json(run_dir / "summary.json")
    binding = read_json(run_dir / "run_manifest.json")
    if summary.get("status") != "completed":
        raise ExportPending(f"{name}: summary has not confirmed completion")
    if summary.get("binding_sha256") != model.object_hash(binding):
        raise ValueError(f"{name}: training binding hash mismatch")
    if binding.get("config_sha256") != model.object_hash(binding.get("config")):
        raise ValueError(f"{name}: training config hash mismatch")
    base = binding.get("base", {})
    if (base.get("model") != model.MODEL_ID or base.get("revision") != model.REVISION
            or set(base.get("files", {})) != set(model.MODEL_FILES)):
        raise ValueError(f"{name}: base identity/file contract differs from the pinned official source")
    data_path = project_root / "data" / "learning" / RUN_NAMES[name]
    rows = model.load_rows(data_path, training=True)
    if binding.get("data_sha256") != model.file_hash(data_path) or binding.get("ordered_rows_sha256") != model.object_hash(rows):
        raise ValueError(f"{name}: dataset differs from the registered run")
    if binding.get("dataset", {}).get("rows") != len(rows):
        raise ValueError(f"{name}: dataset row count mismatch")
    source = binding.get("source", {}).get("files_sha256", {})
    if set(source) != set(model.SOURCE_FILES):
        raise ValueError(f"{name}: source file contract is incomplete")
    for relative, sha in source.items():
        if model.file_hash(project_root / relative) != sha:
            raise ValueError(f"{name}: frozen source differs: {relative}")
    provenance = model.adapter_provenance(run_dir / "adapter", binding["base"])
    metadata = provenance["metadata"]
    if (metadata.get("config") != binding["config"] or metadata.get("data_sha256") != binding["data_sha256"]
            or metadata.get("training_source") != binding["source"]
            or metadata.get("run_manifest_sha256") != model.file_hash(run_dir / "run_manifest.json")):
        raise ValueError(f"{name}: adapter-to-training-report binding mismatch")
    lines = (run_dir / "loss-log.jsonl").read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError(f"{name}: empty/blank loss log")
    losses = [json.loads(line, object_pairs_hook=model.contract._unique_object,
                         parse_constant=model.contract._reject_constant) for line in lines]
    exposures = Counter()
    known_ids = {row["id"] for row in rows}
    for expected_step, entry in enumerate(losses, 1):
        if entry.get("step") != expected_step:
            raise ValueError(f"{name}: loss log steps are not contiguous")
        if (len(entry.get("row_ids", [])) != binding["config"]["gradient_accumulation"]
                or any(identifier not in known_ids for identifier in entry["row_ids"])):
            raise ValueError(f"{name}: invalid microbatch exposure record")
        for key in ("supervised_tokens", "sequence_tokens"):
            if not isinstance(entry.get(key), int) or isinstance(entry[key], bool) or entry[key] <= 0:
                raise ValueError(f"{name}: invalid logged token count")
        for key in ("loss", "seconds", "grad_norm_before_clip"):
            if (not isinstance(entry.get(key), (int, float)) or isinstance(entry[key], bool)
                    or not math.isfinite(entry[key]) or entry[key] < 0):
                raise ValueError(f"{name}: invalid loss/timing/gradient record")
        exposures.update(entry["row_ids"])
    actual_steps = len(losses)
    supervised = sum(item["supervised_tokens"] for item in losses)
    sequence_tokens = sum(item["sequence_tokens"] for item in losses)
    examples = sum(exposures.values())
    if (actual_steps != summary.get("completed_optimizer_steps") or actual_steps != metadata.get("completed_optimizer_steps")
            or supervised != summary.get("seen_supervised_tokens") or supervised != metadata.get("seen_supervised_tokens")
            or sequence_tokens != summary.get("seen_sequence_tokens")
            or dict(exposures) != summary.get("example_exposures")
            or summary.get("examples_seen") != examples or summary.get("microbatches") != examples
            or summary.get("dataset_equivalent_epochs") != examples / len(rows)
            or summary.get("final_loss") != losses[-1]["loss"]
            or summary.get("last_window_supervised_tokens") != losses[-1]["supervised_tokens"]
            or metadata.get("last_window_supervised_tokens") != losses[-1]["supervised_tokens"]):
        raise ValueError(f"{name}: summary/adapter token budget or exposure does not match loss log")
    target = binding["config"].get("target_supervised_tokens")
    if (summary.get("target_supervised_tokens") != target
            or summary.get("token_budget_overshoot") != (supervised - target if target is not None else 0)):
        raise ValueError(f"{name}: target/overshoot mismatch")
    model.validate_training_budget(metadata)
    expected_artifacts = {"loss-log.jsonl", *("adapter/" + item for item in ADAPTER_FILES)}
    if set(summary.get("artifact_sha256", {})) != expected_artifacts:
        raise ValueError(f"{name}: summary artifact hash set is incomplete")
    for relative, sha in summary["artifact_sha256"].items():
        if model.file_hash(run_dir / relative) != sha:
            raise ValueError(f"{name}: final training artifact checksum mismatch: {relative}")
    for item in (binding, summary, metadata, read_json(run_dir / "adapter" / "adapter_config.json"), losses):
        public_path_audit(item)
    by_id = {row["id"]: row for row in rows}
    observed_kinds = Counter()
    for identifier, count in exposures.items():
        observed_kinds[by_id[identifier]["kind"]] += count
    return {"route": name, "status": "completed_training_records_verified",
            "adapter_files": {item: {"sha256": model.file_hash(run_dir / "adapter" / item),
                                      "bytes": (run_dir / "adapter" / item).stat().st_size} for item in ADAPTER_FILES},
            "original_report_sha256": {item: model.file_hash(run_dir / item) for item in ("summary.json", "run_manifest.json", "loss-log.jsonl")},
            "coverage": {"unique_rows_seen": len(exposures), "dataset_rows": len(rows),
                         "unique_groups_seen": len({by_id[i]["group"] for i in exposures}),
                         "dataset_groups": len({row["group"] for row in rows}),
                         "example_exposures_by_kind": dict(sorted(observed_kinds.items()))},
            "training_records": {"summary": summary, "run_manifest": binding, "loss_log": losses},
            "transformation": "none: all three exported adapter files are byte-identical copies",
            "quality_claim": None}


def export_adapters(runs_dir: Path, output_dir: Path, *, dry_run=False, project_root=ROOT):
    runs_dir, output_dir = Path(runs_dir), Path(output_dir)
    if output_dir.exists() and not dry_run:
        raise FileExistsError("Export destination must be new; existing artifacts are immutable")
    # Both routes must be complete before any destination is created.
    checked = {name: validate_run(runs_dir / name, name, project_root) for name in RUN_NAMES}
    manifest = {"schema_version": 1, "status": "validation_passed" if dry_run else "exported_local_artifacts",
                "scope": "CPU file/data/source/run-record audit only; no model loading, training, or quality measurement",
                "model": model.MODEL_ID, "revision": model.REVISION,
                "exporter_sha256": model.file_hash(Path(__file__)), "runs": checked}
    public_path_audit(manifest)
    if dry_run:
        return manifest
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, evidence in checked.items():
        destination = output_dir / name
        destination.mkdir()
        for filename, expected in evidence["adapter_files"].items():
            shutil.copyfile(runs_dir / name / "adapter" / filename, destination / filename)
            if model.file_hash(destination / filename) != expected["sha256"]:
                raise ValueError("Source changed while exporting; destination has no completion manifest")
    # The manifest is written last. A partial copy is never declared complete.
    model.atomic_new_json(output_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--dry-run", action="store_true", help="Validate complete records without copying or creating output")
    args = parser.parse_args()
    try:
        result = export_adapters(args.runs, args.output, dry_run=args.dry_run)
    except ExportPending as exc:
        print(json.dumps({"status": "pending", "reason": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": result["status"], "routes": list(result["runs"]),
                      "quality_claim": None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
