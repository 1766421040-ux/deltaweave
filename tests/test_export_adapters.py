"""Synthetic records test the exporter; these are not experiment evidence."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_adapters", ROOT / "scripts" / "export_adapters.py")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
m = exporter.model


def make_fixture_run(root, name):
    project = root / "project"
    run = root / "runs" / name
    adapter = run / "adapter"
    adapter.mkdir(parents=True)
    data_dir = project / "data" / "learning"
    data_dir.mkdir(parents=True, exist_ok=True)
    row = {"id": name + "-fixture", "split": "train", "group": "fixture-group", "kind": "fixture-only",
           "prompt": "Synthetic exporter fixture, not a model task.", "target": {"action": "answer"}}
    data = data_dir / exporter.RUN_NAMES[name]
    data.write_text(m.canonical(row) + "\n")
    source_files = {}
    for relative in m.SOURCE_FILES:
        file = project / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("TEST FIXTURE SOURCE ONLY\n")
        source_files[relative] = m.file_hash(file)
    source = {"files_sha256": source_files, "packages": {"fixture": True}}
    base = {"model": m.MODEL_ID, "revision": m.REVISION,
            "files": {name: {"sha256": "0" * 64, "bytes": 1} for name in m.MODEL_FILES},
            "download_manifest_sha256": "0" * 64}
    config = {"steps": 512, "target_supervised_tokens": 100, "gradient_accumulation": 4,
              "r": 4, "lora_alpha": 8, "lora_dropout": 0.05, "target_modules": m.TARGET_MODULES}
    binding = {"base": base, "config": config, "config_sha256": m.object_hash(config),
               "data_sha256": m.file_hash(data), "ordered_rows_sha256": m.object_hash([row]),
               "dataset": {"rows": 1}, "source": source}
    m.write_new_json(run / "run_manifest.json", binding)
    adapter_config = {"base_model_name_or_path": m.MODEL_ID, "revision": m.REVISION,
                      **{key: config[key] for key in ("r", "lora_alpha", "lora_dropout", "target_modules")}}
    m.write_new_json(adapter / "adapter_config.json", adapter_config)
    (adapter / "adapter_model.safetensors").write_bytes(b"TEST FIXTURE ONLY: not actual safetensors")
    metadata = {"status": "completed", "base": base, "config": config, "config_sha256": m.object_hash(config),
                "completed_optimizer_steps": 1, "seen_supervised_tokens": 168, "last_window_supervised_tokens": 168,
                "data_sha256": binding["data_sha256"], "training_source": source,
                "run_manifest_sha256": m.file_hash(run / "run_manifest.json"),
                "adapter_files": {file: {"sha256": m.file_hash(adapter / file), "bytes": (adapter / file).stat().st_size}
                                  for file in ("adapter_config.json", "adapter_model.safetensors")}}
    m.write_new_json(adapter / "training_manifest.json", metadata)
    loss = {"step": 1, "row_ids": [row["id"]] * 4, "supervised_tokens": 168, "sequence_tokens": 512,
            "loss": 0.5, "seconds": 1.0, "grad_norm_before_clip": 0.1}
    (run / "loss-log.jsonl").write_text(m.canonical(loss) + "\n")
    summary = {"status": "completed", "binding_sha256": m.object_hash(binding),
               "completed_optimizer_steps": 1, "seen_supervised_tokens": 168, "seen_sequence_tokens": 512,
               "example_exposures": {row["id"]: 4}, "examples_seen": 4, "microbatches": 4,
               "dataset_equivalent_epochs": 4.0, "final_loss": 0.5, "last_window_supervised_tokens": 168,
               "target_supervised_tokens": 100, "token_budget_overshoot": 68,
               "artifact_sha256": {file: m.file_hash(run / file) for file in ("loss-log.jsonl", *("adapter/" + f for f in exporter.ADAPTER_FILES))}}
    m.write_new_json(run / "summary.json", summary)
    return project, run


class ExportAdapterTests(unittest.TestCase):
    def test_missing_final_report_is_pending_without_creating_destination(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaises(exporter.ExportPending):
                exporter.export_adapters(root / "runs", root / "artifacts", project_root=root / "project")
            self.assertFalse((root / "artifacts").exists())

    def test_byte_identical_export_loadable_by_existing_provenance_checker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, qa = make_fixture_run(root, "qa")
            make_fixture_run(root, "behavior")
            dry = exporter.export_adapters(root / "runs", root / "artifacts", project_root=project, dry_run=True)
            self.assertEqual(dry["status"], "validation_passed")
            self.assertFalse((root / "artifacts").exists())
            result = exporter.export_adapters(root / "runs", root / "artifacts", project_root=project)
            self.assertEqual(result["status"], "exported_local_artifacts")
            for name in exporter.RUN_NAMES:
                for file in exporter.ADAPTER_FILES:
                    self.assertEqual((root / "runs" / name / "adapter" / file).read_bytes(), (root / "artifacts" / name / file).read_bytes())
                m.adapter_provenance(root / "artifacts" / name, result["runs"][name]["training_records"]["run_manifest"]["base"])
            with self.assertRaises(FileExistsError):
                exporter.export_adapters(root / "runs", root / "artifacts", project_root=project)

    def test_changed_loss_or_dataset_or_weights_fail(self):
        for relative in ("loss-log.jsonl", "adapter/adapter_model.safetensors"):
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                project, run = make_fixture_run(root, "qa")
                file = run / relative
                file.write_bytes(file.read_bytes().replace(b"0.5", b"0.9") if "loss" in relative else b"changed")
                with self.assertRaises(ValueError):
                    exporter.validate_run(run, "qa", project)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, run = make_fixture_run(root, "qa")
            data = project / "data/learning/train_qa.jsonl"
            data.write_text(data.read_text().replace("Synthetic exporter", "Different exporter"))
            with self.assertRaisesRegex(ValueError, "dataset"):
                exporter.validate_run(run, "qa", project)

    def test_absolute_paths_are_rejected_not_redacted(self):
        for value in ("/Users/example/model", "C:\\Users\\example\\model", "prefix /home/example/model", "file:///tmp/model"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "no automatic redaction"):
                exporter.public_path_audit({"source": value})
        exporter.public_path_audit({"model": m.MODEL_ID, "source": "src/deltaweave/model.py"})


if __name__ == "__main__":
    unittest.main()
