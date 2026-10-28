"""Audit the local R1 tokenizer without loading weights or running a forward pass.

The helpers are the tokenizer contract for training and free generation. A prompt
is tokenized exactly as it will be supplied at inference; completion IDs are
encoded separately and concatenated. Never infer this boundary from a rendered
full-chat assistant message: the official R1 template removes thinking content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
REVISION = "ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562"
DEFAULT_MODEL_DIR = ROOT / ".cache" / "deepseek-r1-distill-qwen-1.5b"
GENERATION_SUFFIX = "<｜Assistant｜><think>\n"
CLOSE_THINK = "</think>"
TOKENIZER_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _decode(tokenizer: Any, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _no_control_tokens(tokenizer: Any, text: str, field: str) -> None:
    # Data may contain arbitrary punctuation, but not forged model control tokens.
    forbidden = set(tokenizer.all_special_tokens) | {"<think>", CLOSE_THINK}
    # R1 role markers, like thinking delimiters, are added tokens which upstream
    # does not mark special. Check the control-token namespace independently.
    if "<｜" in text or any(token and token in text for token in forbidden):
        raise ValueError(f"{field} contains a reserved model control token")


def _json_value(value: Any) -> None:
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _json_value(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _json_value(item)


def encode_prompt(tokenizer: Any, user_content: str) -> list[int]:
    """Return the exact input IDs to use for both training and inference."""
    if not isinstance(user_content, str) or not user_content.strip():
        raise ValueError("user_content must be a nonempty string")
    _no_control_tokens(tokenizer, user_content, "user_content")
    messages = [{"role": "user", "content": user_content}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not prompt_text.endswith(GENERATION_SUFFIX):
        raise ValueError("Tokenizer does not provide the audited official R1 thinking prefix")
    ids = list(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))
    if not ids or ids[0] != tokenizer.bos_token_id:
        raise ValueError("Official prompt must begin with BOS")
    if _decode(tokenizer, ids) != prompt_text:
        raise ValueError("Prompt token decoding does not round-trip exactly")
    return ids


def render_completion(tokenizer: Any, target: Mapping[str, Any], rationale: str = "") -> str:
    """Serialize one action object after a closed thinking segment.

    Empty rationale is an explicit short-action adaptation, not a claim to retain
    R1's reasoning capability. The project action schema must be checked upstream.
    """
    if not isinstance(target, Mapping) or not isinstance(target.get("action"), str) or not target["action"]:
        raise ValueError("target must be an object with a nonempty string action")
    if not isinstance(rationale, str):
        raise ValueError("rationale must be a string")
    payload = dict(target)
    _json_value(payload)
    rendered = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    _no_control_tokens(tokenizer, rendered, "target")
    _no_control_tokens(tokenizer, rationale, "rationale")
    # No leading whitespace is silently removed: these bytes are part of the target.
    return (rationale + "\n" if rationale else "") + CLOSE_THINK + "\n" + rendered


def encode_training_example(
    tokenizer: Any,
    user_content: str,
    target: Mapping[str, Any],
    rationale: str = "",
    *,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Build assistant-only labels, including the first completion token and EOS.

    Concatenating IDs is intentional. Re-tokenizing concatenated text can change
    tokenization at the boundary and must not be substituted by a data collator.
    Right padding must subsequently set padded labels to -100. Attention masks
    must come from sequence lengths, not ID != pad_id, because R1 PAD == EOS.
    """
    prompt_ids = encode_prompt(tokenizer, user_content)
    completion_text = render_completion(tokenizer, target, rationale)
    completion_ids = _encode(tokenizer, completion_text)
    eos_id = tokenizer.eos_token_id
    if not isinstance(eos_id, int) or eos_id in completion_ids:
        raise ValueError("Completion must contain no embedded EOS; append exactly one EOS ID")
    if not completion_ids or _decode(tokenizer, completion_ids) != completion_text:
        raise ValueError("Completion token decoding does not round-trip exactly")
    completion_ids.append(eos_id)
    input_ids = prompt_ids + completion_ids
    if max_length is not None:
        if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if len(input_ids) > max_length:
            raise ValueError(f"Example has {len(input_ids)} tokens, exceeding max_length={max_length}; truncation is forbidden")
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
        "prompt_length": len(prompt_ids),
        "completion_ids": completion_ids,
        "completion_text": completion_text,
    }


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON constant: {value}")


def parse_generated_response(raw_continuation: str) -> dict[str, Any]:
    """Parse the continuation after the prompt's opening <think>, without repair.

    The caller removes only the actual terminal EOS token by ID before decoding;
    it must not use skip_special_tokens=True, which can hide unexpected control
    tokens. In the pinned tokenizer, </think> is an added token but is not marked
    special; this flag does not remove that particular delimiter.
    Truncated or timed-out generations remain failures even when parseable.
    """
    if not isinstance(raw_continuation, str) or raw_continuation.count(CLOSE_THINK) != 1:
        raise ValueError("Expected exactly one closing thinking delimiter")
    _, payload = raw_continuation.split(CLOSE_THINK, 1)
    if "<think>" in raw_continuation or "<｜" in raw_continuation:
        raise ValueError("Unexpected model control token in generated continuation")
    obj = json.loads(payload.strip(), object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(obj, dict) or not isinstance(obj.get("action"), str) or not obj["action"]:
        raise ValueError("Final output must be a JSON object with a nonempty string action")
    # Validate the task-specific action/argument schema separately; this function
    # neither chooses an action nor substitutes a guessed target for invalid output.
    return obj


def audit(tokenizer: Any, model_dir: Path | None = None) -> dict[str, Any]:
    """Exercise real rendering, mask, Unicode and cross-boundary tokenization."""
    prompts = [
        "Choose the next dependency-update action using only the supplied versioned evidence.",
        'Snapshot v7. Package alpha requires beta>=2; the previous answer used beta==1.\nReturn one action.',
        '检查版本证据。Package café has extras ["http","測試"]. Python: 3.12.',
    ]
    target = {"action": "answer", "arguments": {"package": "café", "version": "2.0"}}
    rationales = ["", "The supplied snapshot supersedes the previous evidence.", "\n"]
    cases = []
    for p_index, prompt in enumerate(prompts):
        for r_index, rationale in enumerate(rationales):
            example = encode_training_example(tokenizer, prompt, target, rationale)
            prompt_ids = encode_prompt(tokenizer, prompt)
            full_ids = list(tokenizer.apply_chat_template([
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "<think>\n" + example["completion_text"]},
            ], tokenize=True, add_generation_prompt=False))
            joint_ids = _encode(tokenizer, _decode(tokenizer, prompt_ids) + example["completion_text"])
            if (example["labels"][:len(prompt_ids)] != [-100] * len(prompt_ids)
                    or example["labels"][len(prompt_ids):] != example["completion_ids"]
                    or example["labels"][-1] != tokenizer.eos_token_id
                    or parse_generated_response(example["completion_text"]) != target):
                raise ValueError("Assistant mask or generated-output round-trip audit failed")
            cases.append({
                "prompt_fixture": p_index, "rationale_fixture": r_index,
                "prompt_tokens": len(prompt_ids), "completion_tokens_including_eos": len(example["completion_ids"]),
                "official_full_chat_prefix_equal": full_ids[:len(prompt_ids)] == prompt_ids,
                "joint_text_encoding_equals_explicit_ids": joint_ids == example["input_ids"][:-1],
                "first_supervised_token_id": example["labels"][len(prompt_ids)],
                "assistant_only_mask_passed": True, "roundtrip_passed": True,
            })
    report: dict[str, Any] = {
        "schema_version": 1, "audit_kind": "tokenizer_only_no_weights_no_forward",
        "expected_model": MODEL_ID, "expected_revision": REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "closing_think_is_registered_special_token": CLOSE_THINK in tokenizer.all_special_tokens,
        "official_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "passed": all(item["assistant_only_mask_passed"] and item["roundtrip_passed"] for item in cases),
        "official_full_chat_prefix_hazard_observed": any(not x["official_full_chat_prefix_equal"] for x in cases),
        "joint_text_retokenization_hazard_observed": any(not x["joint_text_encoding_equals_explicit_ids"] for x in cases),
        "cases": cases,
        "not_tested": ["model_forward", "memory_peak", "training", "generation_quality", "reasoning_retention"],
    }
    if model_dir is not None:
        report["files_sha256"] = {name: sha256(model_dir / name) for name in TOKENIZER_FILES}
        manifest_path = model_dir / "download-manifest.json"
        report["source_manifest_status"] = "pending_download_manifest"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["model"] != MODEL_ID or manifest["revision"] != REVISION:
                raise ValueError("Downloaded model identity differs from pinned source")
            if any(manifest["files"][name]["sha256"] != digest for name, digest in report["files_sha256"].items()):
                raise ValueError("Tokenizer/config hash differs from download manifest")
            report["source_manifest_status"] = "tokenizer_config_files_match_download_manifest"
            report["download_manifest_sha256"] = sha256(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR, help="Existing local tokenizer directory; never downloaded")
    parser.add_argument("--output", type=Path, help="Optional new audit report path; refuses replacement")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    report = audit(tokenizer, args.model)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
