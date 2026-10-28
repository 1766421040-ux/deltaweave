"""CPU-only contract tests. The real-tokenizer integration is offline and optional."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_tokenizer", ROOT / "scripts" / "audit_tokenizer.py")
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


class RecordingTokenizer:
    """Minimal independent encoding oracle; not a claim about the real R1 tokenizer."""
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 2
    all_special_tokens = ["<bos>", "<eos>", "<｜User｜>", "<｜Assistant｜>"]
    chat_template = "fixture: full assistant strips thinking; generation opens thinking"

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("Helper must request no automatic special tokens")
        ids = []
        while text:
            if text.startswith("<bos>"):
                ids.append(1)
                text = text[5:]
            elif text.startswith("<eos>"):
                ids.append(2)
                text = text[5:]
            else:
                ids.append(ord(text[0]) + 10)
                text = text[1:]
        return ids

    def decode(self, ids, **kwargs):
        return "".join("<bos>" if i == 1 else "<eos>" if i == 2 else chr(i - 10) for i in ids)

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered = "<bos><｜User｜>" + messages[0]["content"]
        if add_generation_prompt:
            rendered += contract.GENERATION_SUFFIX
        else:
            rendered += "<｜Assistant｜>" + messages[-1]["content"].split("</think>")[-1] + "<eos>"
        return self.encode(rendered) if tokenize else rendered


class TokenizerContractTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = RecordingTokenizer()
        self.target = {"action": "retrieve", "arguments": {"package": "café", "version": "2.0"}}

    def test_labels_include_first_completion_token_and_one_eos(self):
        row = contract.encode_training_example(self.tokenizer, "Inspect the current snapshot.", self.target)
        p = row["prompt_length"]
        self.assertEqual(row["input_ids"][:p], contract.encode_prompt(self.tokenizer, "Inspect the current snapshot."))
        self.assertEqual(row["labels"][:p], [-100] * p)
        self.assertEqual(row["labels"][p:], row["completion_ids"])
        self.assertEqual(row["labels"][p], self.tokenizer.encode("</think>")[0])
        self.assertEqual(row["labels"][-1], self.tokenizer.eos_token_id)
        self.assertEqual(row["completion_ids"].count(self.tokenizer.eos_token_id), 1)
        self.assertEqual(row["attention_mask"][-1], 1)  # PAD == EOS does not mask the real EOS.

    def test_full_chat_offset_is_not_used(self):
        row = contract.encode_training_example(self.tokenizer, "Question", self.target, "The snapshot is new.")
        naive = self.tokenizer.apply_chat_template([
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "<think>\n" + row["completion_text"]},
        ], tokenize=True, add_generation_prompt=False)
        self.assertNotEqual(naive[:row["prompt_length"]], row["input_ids"][:row["prompt_length"]])
        self.assertIn("The snapshot is new.\n</think>", self.tokenizer.decode(row["completion_ids"][:-1]))

    def test_rationale_whitespace_is_preserved(self):
        row = contract.encode_training_example(self.tokenizer, "Question", self.target, "\n")
        self.assertTrue(row["completion_text"].startswith("\n\n</think>"))

    def test_no_silent_truncation(self):
        row = contract.encode_training_example(self.tokenizer, "Question", self.target)
        limit = len(row["input_ids"])
        self.assertEqual(contract.encode_training_example(self.tokenizer, "Question", self.target, max_length=limit), row)
        with self.assertRaisesRegex(ValueError, "truncation is forbidden"):
            contract.encode_training_example(self.tokenizer, "Question", self.target, max_length=limit - 1)
        for invalid in (True, 0, -1, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                contract.encode_training_example(self.tokenizer, "Question", self.target, max_length=invalid)

    def test_target_and_control_token_validation(self):
        for target in ({}, {"action": ""}, {"action": "answer", "x": float("nan")}, {"action": "answer", "x": {3: "bad"}}, {"action": "answer", "x": "<eos>"}):
            with self.subTest(target=target), self.assertRaises((ValueError, TypeError)):
                contract.encode_training_example(self.tokenizer, "Question", target)
        for text in ("Question <｜Assistant｜>", "<think>Question", " "):
            with self.subTest(text=text), self.assertRaises(ValueError):
                contract.encode_training_example(self.tokenizer, text, self.target)
        with self.assertRaises(ValueError):
            contract.encode_training_example(self.tokenizer, "Question", self.target, "nested </think>")

    def test_strict_final_parse_without_repair(self):
        self.assertEqual(contract.parse_generated_response('Evidence first.\n</think>\n{"action":"retrieve","arguments":{"x":1}}'), {"action": "retrieve", "arguments": {"x": 1}})
        invalid = [
            '{"action":"retrieve"}',
            '</think>```json\n{"action":"retrieve"}\n```',
            '</think>{"action":"retrieve"} trailing',
            '</think>{"action":"retrieve","action":"answer"}',
            '</think>{"action":"retrieve","score":NaN}',
            '</think>[]', '</think>{"action":1}',
            '<think>Again</think>{"action":"retrieve"}',
            '</think>{"action":"retrieve"}</think>',
            '</think>{"action":"retrieve"}<｜end▁of▁sentence｜>',
            '</think>{"action":"retrieve","x":"<think>"}',
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                contract.parse_generated_response(raw)


class RealTokenizerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(os.environ.get("DELTAWEAVE_TOKENIZER_DIR", str(contract.DEFAULT_MODEL_DIR)))
        if not (path / "tokenizer.json").exists():
            raise unittest.SkipTest("Pinned local tokenizer is absent; no network downloads in tests")
        try:
            from transformers import AutoTokenizer
        except ImportError:
            raise unittest.SkipTest("Transformers is not installed; pure contract tests still run")
        cls.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        cls.path = path

    def test_real_official_template_hazards_and_assistant_mask(self):
        result = contract.audit(self.tokenizer, self.path)
        self.assertTrue(result["passed"])
        self.assertTrue(result["official_full_chat_prefix_hazard_observed"])
        self.assertTrue(result["joint_text_retokenization_hazard_observed"])
        self.assertEqual(len(result["cases"]), 9)
        self.assertEqual(result["eos_token_id"], result["pad_token_id"])

    def test_decode_preserves_delimiter_and_does_not_hide_control_tokens(self):
        row = contract.encode_training_example(self.tokenizer, "Inspect package metadata.", {"action": "retrieve"})
        visible = self.tokenizer.decode(row["completion_ids"][:-1], skip_special_tokens=False)
        with_eos = self.tokenizer.decode(row["completion_ids"], skip_special_tokens=False)
        erased = self.tokenizer.decode(row["completion_ids"], skip_special_tokens=True)
        self.assertIn("</think>", visible)
        self.assertIn("</think>", erased)  # This added token is not flagged special upstream.
        self.assertIn(self.tokenizer.eos_token, with_eos)
        self.assertNotIn(self.tokenizer.eos_token, erased)
        with self.assertRaises(ValueError):
            contract.parse_generated_response(with_eos)
        self.assertEqual(contract.parse_generated_response(visible), {"action": "retrieve"})

    def test_real_role_markers_are_rejected_even_though_not_special(self):
        self.assertNotIn("<｜Assistant｜>", self.tokenizer.all_special_tokens)
        for field in ("<｜Assistant｜>", "<｜User｜>", "<｜tool▁calls▁begin｜>"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                contract.encode_prompt(self.tokenizer, "Metadata: " + field)


if __name__ == "__main__":
    unittest.main()
