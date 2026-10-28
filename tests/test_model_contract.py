"""CPU contract tests; fake policy fixtures never load the real model weights."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from deltaweave import model as runner

_SPEC = importlib.util.spec_from_file_location("tokenizer_test_fixture", ROOT / "tests" / "test_tokenizer_contract.py")
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)
RecordingTokenizer = _FIXTURE.RecordingTokenizer


def row(identifier="one", split="dev", prompt="Which snapshot applies?"):
    return {"id": identifier, "split": split, "group": "group-" + identifier,
            "kind": "fixture-only", "prompt": prompt,
            "target": {"action": "answer", "arguments": {"version": "2.0"}}}


def write_rows(path, rows):
    path.write_text("".join(runner.canonical(r) + "\n" for r in rows), encoding="utf-8")


def fixture_base(path):
    """Clearly fake bytes for provenance unit tests, never an executable checkpoint."""
    path.mkdir()
    files = {}
    for name in runner.MODEL_FILES:
        content = ("TEST FIXTURE ONLY: " + name).encode()
        (path / name).write_bytes(content)
        files[name] = {"sha256": runner.file_hash(path / name), "bytes": len(content)}
    runner.write_new_json(path / "download-manifest.json", {"model": runner.MODEL_ID, "revision": runner.REVISION, "files": files})
    return runner.model_provenance(path)


class DataAndTrainingContractTests(unittest.TestCase):
    def test_train_refuses_holdout_mixed_split_duplicate_and_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            bad = [[row(split="dev")], [row(split="future")], [row(split="train"), row("two", "test")],
                   [row(split="train"), row(split="train")], []]
            for rows in bad:
                with self.subTest(rows=rows):
                    write_rows(path, rows)
                    with self.assertRaises(ValueError):
                        runner.load_rows(path, training=True)
            write_rows(path, [row(split="future")])
            self.assertEqual(runner.load_rows(path)[0]["split"], "future")

    def test_only_prompt_enters_condition_and_no_truncation(self):
        tokenizer = RecordingTokenizer()
        sample = row(split="train")
        sample["id"] = "LABEL_METADATA_SENTINEL"
        sample["kind"] = "GOLD_KIND_SENTINEL"
        prepared, stats = runner.prepare_training_rows([sample], tokenizer)
        ex = prepared[0]
        condition = tokenizer.decode(ex["input_ids"][:ex["prompt_length"]])
        self.assertIn(sample["prompt"], condition)
        self.assertNotIn(sample["id"], condition)
        self.assertNotIn(sample["kind"], condition)
        self.assertNotIn('"action":"answer"', condition)
        self.assertEqual(stats["supervised_tokens_total"], sum(v != -100 for v in ex["labels"]))
        with self.assertRaisesRegex(ValueError, "truncation"):
            runner.prepare_training_rows([sample], tokenizer, max_length=5)

    def test_bucket_orders_are_seeded_complete_and_epoch_specific(self):
        lengths = [10 + i % 3 for i in range(77)]
        first = runner.epoch_order(lengths, 42, 0)
        self.assertEqual(first, runner.epoch_order(lengths, 42, 0))
        self.assertEqual(sorted(first), list(range(77)))
        self.assertNotEqual(first, runner.epoch_order(lengths, 42, 1))
        self.assertNotEqual(first, runner.epoch_order(lengths, 43, 0))
        stream = runner.index_stream(lengths, 42)
        self.assertEqual([next(stream) for _ in range(154)], first + runner.epoch_order(lengths, 42, 1))

    def test_token_weighted_accumulation_matches_pooled_gradient(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Optional CPU tensor gradient check requires PyTorch")
        # No language-model forward or GPU: independent scalar-gradient oracle.
        logits = torch.tensor([[0.2, 0.7], [0.1, -0.3], [1.2, 0.2], [-0.5, 0.5]], requires_grad=True)
        labels = torch.tensor([1, 0, 1, 0])
        pooled = torch.nn.functional.cross_entropy(logits, labels)
        expected = torch.autograd.grad(pooled, logits)[0]
        weights = runner.accumulation_weights([1, 3])
        accumulated = (torch.nn.functional.cross_entropy(logits[:1], labels[:1]) * weights[0]
                       + torch.nn.functional.cross_entropy(logits[1:], labels[1:]) * weights[1])
        actual = torch.autograd.grad(accumulated, logits)[0]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))
        self.assertNotEqual(weights, [0.5, 0.5])

    def test_first_complete_window_budget_validation(self):
        good = {"config": {"steps": 512, "target_supervised_tokens": 100},
                "completed_optimizer_steps": 3, "seen_supervised_tokens": 120,
                "last_window_supervised_tokens": 40}
        runner.validate_training_budget(good)
        for updates in ({"seen_supervised_tokens": 99}, {"seen_supervised_tokens": 140},
                        {"completed_optimizer_steps": 513}, {"last_window_supervised_tokens": 0}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                runner.validate_training_budget({**good, **updates})
        with self.assertRaises(ValueError):
            runner.validate_training_budget({"config": {"steps": 4, "target_supervised_tokens": None}, "completed_optimizer_steps": 3})

    def test_dry_run_does_not_load_model_and_output_is_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_base(root / "base")
            write_rows(root / "train.jsonl", [row(split="train")])
            with patch.object(runner, "load_tokenizer", return_value=RecordingTokenizer()):
                result = runner.train_adapter(root / "train.jsonl", root / "base", root / "run", steps=512,
                                              target_supervised_tokens=100, dry_run=True)
            self.assertEqual(result["status"], "dry_run_passed")
            self.assertFalse(result["weights_loaded"])
            self.assertFalse((root / "run" / "adapter").exists())
            with self.assertRaises(FileExistsError):
                runner.train_adapter(root / "train.jsonl", root / "base", root / "run", steps=1, dry_run=True)

    def test_model_files_are_rehashed_not_just_manifest_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "base"
            fixture_base(root)
            (root / "model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                runner.model_provenance(root)

    def test_cpu_fixture_training_stops_on_complete_token_budget_window(self):
        try:
            import torch
            import transformers
            import peft
        except ImportError:
            self.skipTest("Optional fake-model CPU loop test requires local ML libraries")
        class TinyFixture(torch.nn.Module):
            """Scalar loss oracle, not a language model or real safetensors export."""
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.3))
                self.config = SimpleNamespace(use_cache=True)
                self.calls = 0
            def gradient_checkpointing_enable(self, **kwargs):
                pass
            def enable_input_require_grads(self):
                pass
            def forward(self, **batch):
                self.calls += 1
                return SimpleNamespace(loss=torch.nn.functional.softplus(self.weight))
            def save_pretrained(self, path, **kwargs):
                path.mkdir()
                config = self.peft_config["default"]
                runner.write_new_json(path / "adapter_config.json", {
                    "base_model_name_or_path": config.base_model_name_or_path, "revision": config.revision,
                    "r": config.r, "lora_alpha": config.lora_alpha, "lora_dropout": config.lora_dropout,
                    "target_modules": sorted(config.target_modules)})
                (path / "adapter_model.safetensors").write_bytes(b"TEST FIXTURE ONLY, not executable weights")
        fixture = TinyFixture()
        def wrap(model, config):
            model.peft_config = {"default": config}
            return model
        tokenizer = RecordingTokenizer()
        sample = row(split="train")
        examples, _ = runner.prepare_training_rows([sample], tokenizer)
        window_tokens = 4 * len(examples[0]["completion_ids"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = fixture_base(root / "base")
            write_rows(root / "train.jsonl", [sample])
            with patch.object(runner, "load_tokenizer", return_value=tokenizer), \
                 patch.object(transformers.AutoModelForCausalLM, "from_pretrained", return_value=fixture), \
                 patch.object(peft, "get_peft_model", side_effect=wrap), redirect_stdout(io.StringIO()):
                result = runner.train_adapter(root / "train.jsonl", root / "base", root / "run",
                    steps=8, device="cpu", target_supervised_tokens=window_tokens + 1)
            self.assertEqual(result["completed_optimizer_steps"], 2)
            self.assertEqual(result["seen_supervised_tokens"], window_tokens * 2)
            self.assertEqual(result["token_budget_overshoot"], window_tokens - 1)
            self.assertEqual(fixture.calls, 8)
            self.assertEqual(result["examples_seen"], 8)
            provenance = runner.adapter_provenance(root / "run" / "adapter", base)
            self.assertEqual(provenance["metadata"]["completed_optimizer_steps"], 2)


class ScoringAndReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = RecordingTokenizer()
        self.row = row()
        self.config = runner.evaluation_config(max_new_tokens=2048)

    def record(self, raw=None, eos=True, seconds=1.0):
        raw = raw if raw is not None else '</think>{"action":"answer","arguments":{"version":"2.0"}}'
        ids = self.tokenizer.encode(raw) + ([self.tokenizer.eos_token_id] if eos else [])
        return runner.generation_record(self.tokenizer, ids,
            input_tokens=len(runner.contract.encode_prompt(self.tokenizer, self.row["prompt"])),
            seconds=seconds, timed_out=False, config=self.config)

    def test_exact_payload_no_missing_extra_keys_types_or_array_repair(self):
        good = self.record()["raw"]
        self.assertTrue(runner.score_response(good, self.row["target"], True)["strict_success"])
        self.assertFalse(runner.score_response(good, self.row["target"], False)["strict_success"])
        for target in ({"action": "answer"}, {"action": "answer", "arguments": {"version": "1.0"}},
                       {"action": "answer", "arguments": {"version": "2.0"}, "extra": None}):
            self.assertFalse(runner.score_response(good, target, True)["strict_success"])
        self.assertFalse(runner.exact_json_equal({"x": True}, {"x": 1}))
        self.assertFalse(runner.exact_json_equal({"x": [1, 2]}, {"x": [2, 1]}))

    def test_auxiliary_allows_only_a_sole_fence_without_value_change(self):
        fenced = 'Reason.\n</think>\n```json\n{"action":"answer","arguments":{"version":"2.0"}}\n```'
        result = runner.score_response(fenced, self.row["target"], True)
        self.assertFalse(result["strict_success"])
        self.assertTrue(result["auxiliary_success"])
        self.assertEqual(result["auxiliary_mode"], "sole_json_fence")
        for raw in (fenced + '\n{"action":"answer"}', fenced.replace("```json", "Here it is:\n```json"),
                    fenced.replace('"2.0"', '"1.0"'), fenced.replace('"action":"answer"', '"action":"answer","action":"answer"')):
            self.assertFalse(runner.score_response(raw, self.row["target"], True)["auxiliary_success"])

    def test_eos_cap_and_timeout_preserve_failures(self):
        self.assertTrue(self.record()["completed"])
        unterminated = self.record(eos=False)
        self.assertFalse(unterminated["completed"])
        self.assertEqual(unterminated["stop_reason"], "unterminated")
        self.config["max_new_tokens"] = unterminated["new_tokens"]
        capped = self.record(eos=False)
        self.assertTrue(capped["hit_token_cap"])
        self.assertEqual(capped["stop_reason"], "token_cap")
        timed = self.record(seconds=121.0)
        self.assertTrue(timed["eos"])
        self.assertTrue(timed["timed_out"])
        self.assertFalse(timed["completed"])
        self.assertEqual(timed["timeout_overshoot_seconds"], 1.0)

    def test_prompt_seed_independent_of_id_and_stable(self):
        self.assertEqual(runner.case_seed("public prompt", 42), runner.case_seed("public prompt", 42))
        self.assertNotEqual(runner.case_seed("public prompt", 42), runner.case_seed("public prompt 2", 42))
        self.assertNotEqual(runner.case_seed("public prompt", 42), runner.case_seed("public prompt", 43))

    def test_receipt_recomputes_raw_score_and_settings_binding(self):
        receipt = runner.make_receipt(self.row, self.record(), "setting-a", self.config)
        runner.validate_receipt(receipt, self.row, "setting-a", self.config, self.tokenizer)
        with self.assertRaises(ValueError):
            runner.validate_receipt(receipt, self.row, "setting-b", self.config, self.tokenizer)
        for field, value in (("raw", '</think>{"action":"abstain"}'), ("completed", False), ("generated_ids_sha256", "forged")):
            altered = json.loads(json.dumps(receipt))
            altered["payload"]["prediction"][field] = value
            altered["receipt_sha256"] = runner.object_hash(altered["payload"])
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.validate_receipt(altered, self.row, "setting-a", self.config, self.tokenizer)
        changed = {**self.row, "prompt": "Other prompt"}
        with self.assertRaises(ValueError):
            runner.validate_receipt(receipt, changed, "setting-a", self.config, self.tokenizer)


class EvaluationResumeTests(unittest.TestCase):
    def test_interruption_resume_reuses_only_validated_case_receipts(self):
        tokenizer = RecordingTokenizer()
        calls = []
        class FakePolicy:
            load_seconds = 0.0
            interrupt = True
            def __init__(self, model_path, adapter_path, actual_tokenizer, config):
                self.config = config
                self.tokenizer = actual_tokenizer
            def predict(self, prompt, seed):
                calls.append(prompt)
                if self.interrupt and len(calls) == 2:
                    raise KeyboardInterrupt("Fixture interruption")
                ids = tokenizer.encode('</think>{"action":"answer","arguments":{"version":"2.0"}}') + [tokenizer.eos_token_id]
                return runner.generation_record(tokenizer, ids,
                    input_tokens=len(runner.contract.encode_prompt(tokenizer, prompt)),
                    seconds=0.1, timed_out=False, config=self.config)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture_base(root / "base")
            data = root / "dev.jsonl"
            write_rows(data, [row(), row("two", prompt="Second question")])
            output = root / "evaluation.json"
            with patch.object(runner, "load_tokenizer", return_value=tokenizer), patch.object(runner, "HFPolicy", FakePolicy), redirect_stdout(io.StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    runner.evaluate_model(data, root / "base", output)
                self.assertFalse(output.exists())
                with self.assertRaisesRegex(ValueError, "binding mismatch"):
                    runner.evaluate_model(data, root / "base", output, seed=43, resume=True)
                self.assertEqual(len(calls), 2)  # Binding failure happens before another forward.
                FakePolicy.interrupt = False
                result = runner.evaluate_model(data, root / "base", output, resume=True)
            self.assertEqual(result["cached_cases"], 1)
            self.assertEqual(result["new_cases"], 1)
            self.assertEqual(len(calls), 3)
            self.assertEqual(result["metrics"]["strict_successes"], 2)
            self.assertEqual(len(list(output.with_suffix(".json.receipts").glob("*.json"))), 3)
            with self.assertRaises(FileExistsError):
                runner.evaluate_model(data, root / "base", output, resume=True)

    def test_limit_dev_cannot_shrink_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "test.jsonl"
            write_rows(data, [row(split="test")])
            with self.assertRaisesRegex(ValueError, "only.*dev"):
                runner.evaluate_model(data, root / "absent-model", root / "output.json", limit_dev=1)


if __name__ == "__main__":
    unittest.main()
