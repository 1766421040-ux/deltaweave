"""Local-only LoRA training and auditable free-generation contracts.

Heavy ML imports are lazy: dataset/receipt/scoring checks run on a plain Python
installation. No function downloads weights or calls a remote model service.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import time
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("_deltaweave_tokenizer_contract", ROOT / "scripts" / "audit_tokenizer.py")
contract = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(contract)
MODEL_ID, REVISION = contract.MODEL_ID, contract.REVISION
MODEL_FILES = ("config.json", "generation_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "LICENSE", "README.md")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
SOURCE_FILES = ("src/deltaweave/model.py", "scripts/audit_tokenizer.py", "scripts/train_adapter.py", "scripts/evaluate_model.py")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def object_hash(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, value: Any) -> None:
    """Refuse to overwrite published or previous-run evidence."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_new_json(path: Path, value: Any) -> None:
    """A crash must not leave a partial receipt that looks reusable."""
    path = Path(path)
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    write_new_json(temporary, value)
    try:
        os.link(temporary, path)  # Atomic creation; fails instead of replacing.
    finally:
        temporary.unlink(missing_ok=True)


def load_rows(path: Path, *, training: bool = False) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row at line {line_number}")
            row = json.loads(line, object_pairs_hook=contract._unique_object, parse_constant=contract._reject_constant)
            if not isinstance(row, dict) or not {"id", "split", "group", "prompt", "target", "kind"} <= row.keys():
                raise ValueError(f"Missing row contract fields at line {line_number}")
            for field in ("id", "split", "group", "prompt", "kind"):
                if not isinstance(row[field], str) or not row[field].strip():
                    raise ValueError(f"{field} must be a nonempty string at line {line_number}")
            if row["id"] in seen:
                raise ValueError(f"Duplicate row id at line {line_number}")
            seen.add(row["id"])
            if row["split"] not in {"train", "dev", "test", "future"}:
                raise ValueError("split must be train, dev, test, or future")
            if training and row["split"] != "train":
                raise ValueError("Training rejects every dev/test row")
            target = row["target"]
            if not isinstance(target, dict) or not isinstance(target.get("action"), str) or not target["action"]:
                raise ValueError("target must be an action JSON object")
            canonical(row)  # Reject NaN and non-JSON values before hashing.
            rows.append(row)
    if not rows:
        raise ValueError("Dataset is empty")
    if len({row["split"] for row in rows}) != 1:
        raise ValueError("A dataset file must contain exactly one split")
    return rows


def source_provenance() -> dict[str, Any]:
    versions = {}
    for name in ("torch", "transformers", "tokenizers", "peft", "safetensors", "accelerate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"files_sha256": {name: file_hash(ROOT / name) for name in SOURCE_FILES},
            "python": platform.python_version(), "system": platform.system(),
            "machine": platform.machine(), "packages": versions}


def model_provenance(model_path: Path) -> dict[str, Any]:
    """Rehash all pinned downloaded files, including weights, without loading them."""
    model_path = Path(model_path)
    manifest_path = model_path / "download-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != MODEL_ID or manifest.get("revision") != REVISION:
        raise ValueError("Base identity does not match the pinned official model")
    if set(manifest.get("files", {})) != set(MODEL_FILES):
        raise ValueError("Base manifest must contain the complete seven-file contract")
    files = {}
    for name in MODEL_FILES:
        entry = manifest["files"][name]
        path = model_path / name
        actual = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        if actual["sha256"] != entry.get("sha256") or actual["bytes"] != entry.get("bytes"):
            raise ValueError(f"Pinned model file checksum/size mismatch: {name}")
        files[name] = actual
    return {"model": MODEL_ID, "revision": REVISION, "files": files,
            "download_manifest_sha256": file_hash(manifest_path)}


def load_tokenizer(model_path: Path) -> Any:
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)


def prepare_training_rows(rows: list[dict[str, Any]], tokenizer: Any, max_length: int = 1024) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = []
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Training rejects every dev/test row")
        try:
            # Only prompt reaches encode_prompt. id/group/kind/target are not
            # condition inputs; target is exclusively the supervised continuation.
            encoded = contract.encode_training_example(tokenizer, row["prompt"], row["target"], rationale="", max_length=max_length)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Training row {row['id']!r} failed token audit: {exc}") from exc
        examples.append(encoded)
    lengths = [len(e["input_ids"]) for e in examples]
    supervised = [len(e["completion_ids"]) for e in examples]
    return examples, {"rows": len(rows), "groups": len({r["group"] for r in rows}),
        "sequence_tokens_total": sum(lengths), "supervised_tokens_total": sum(supervised),
        "sequence_tokens_min": min(lengths), "sequence_tokens_max": max(lengths),
        "supervised_tokens_min": min(supervised), "supervised_tokens_max": max(supervised),
        "actions": dict(sorted(Counter(r["target"]["action"] for r in rows).items())),
        "kinds": dict(sorted(Counter(r["kind"] for r in rows).items())),
        "truncated_rows": 0, "rationale": "empty: directly supervise closing think and action JSON"}


def epoch_order(lengths: list[int], seed: int, epoch: int, bucket_size: int = 64) -> list[int]:
    if not lengths or bucket_size < 1:
        raise ValueError("Nonempty lengths and positive bucket size required")
    rng = random.Random(f"deltaweave-length-bucket-v1:{seed}:{epoch}")
    indices = list(range(len(lengths)))
    rng.shuffle(indices)
    buckets = [indices[i:i + bucket_size] for i in range(0, len(indices), bucket_size)]
    # Stable sort keeps randomized ties; each epoch covers every row once.
    for bucket in buckets:
        bucket.sort(key=lambda index: lengths[index])
    rng.shuffle(buckets)
    return [index for bucket in buckets for index in bucket]


def index_stream(lengths: list[int], seed: int) -> Iterator[int]:
    epoch = 0
    while True:
        yield from epoch_order(lengths, seed, epoch)
        epoch += 1


def accumulation_weights(counts: list[int]) -> list[float]:
    if not counts or any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in counts):
        raise ValueError("Each microbatch needs positive supervised token count")
    total = sum(counts)
    return [n / total for n in counts]


def sampled_memory(torch: Any = None, device: str = "cpu") -> dict[str, int]:
    sample = {}
    try:
        if sys.platform.startswith("linux"):
            resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
            sample["rss_bytes"] = resident_pages * os.sysconf("SC_PAGE_SIZE")
        elif sys.platform == "darwin":
            output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True, timeout=2)
            sample["rss_bytes"] = int(output.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    if torch is not None and device == "mps":
        sample["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
        sample["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return sample


def _synchronize(torch: Any, device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _check_device(torch: Any, device: str) -> None:
    if device not in {"cpu", "mps"}:
        raise ValueError("Only local cpu or mps execution is supported")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; no silent CPU or cloud fallback")


def train_adapter(data_path: Path, model_path: Path, output_dir: Path, *, steps: int, seed: int = 42,
                  device: str = "mps", dry_run: bool = False,
                  target_supervised_tokens: int | None = None) -> dict[str, Any]:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("steps must be a positive optimizer-step count")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if device not in {"cpu", "mps"}:
        raise ValueError("device must be local cpu or mps")
    if target_supervised_tokens is not None and (not isinstance(target_supervised_tokens, int)
            or isinstance(target_supervised_tokens, bool) or target_supervised_tokens < 1):
        raise ValueError("target_supervised_tokens must be a positive integer")
    output_dir, data_path, model_path = Path(output_dir), Path(data_path), Path(model_path)
    if output_dir.exists():
        raise FileExistsError("Training output must be a new directory")
    rows = load_rows(data_path, training=True)
    base = model_provenance(model_path)
    tokenizer = load_tokenizer(model_path)
    examples, stats = prepare_training_rows(rows, tokenizer)
    config = {"steps": steps, "seed": seed, "device": device, "base_dtype": "bfloat16",
              "r": 4, "lora_alpha": 8, "lora_dropout": 0.05, "target_modules": TARGET_MODULES,
              "optimizer": "AdamW", "learning_rate": 2e-4, "betas": [0.9, 0.999], "eps": 1e-8,
              "weight_decay": 0.01, "batch_size": 1, "gradient_accumulation": 4,
              "max_length": 1024, "gradient_checkpointing": True, "max_grad_norm": 1.0,
              "loss_normalization": "supervised_tokens_across_full_accumulation_window",
              "data_order": "seeded_length_buckets_64_no_drop_last", "scheduler": "constant",
              "target_supervised_tokens": target_supervised_tokens,
              "stopping_rule": "first_complete_optimizer_window_at_or_above_token_budget" if target_supervised_tokens else "fixed_optimizer_steps",
              "completion_contract": "explicit_ids_close_think_json_eos_v1"}
    binding = {"schema_version": 1, "base": base, "config": config, "config_sha256": object_hash(config),
               "data_sha256": file_hash(data_path), "ordered_rows_sha256": object_hash(rows),
               "source": source_provenance(), "tokenizer_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
               "dataset": stats}
    output_dir.mkdir(parents=True, exist_ok=False)
    write_new_json(output_dir / "run_manifest.json", binding)
    if dry_run:
        result = {"status": "dry_run_passed", "weights_loaded": False, "forward_executed": False,
                  "binding_sha256": object_hash(binding), "dataset": stats}
        write_new_json(output_dir / "dry-run.json", result)
        return result
    started = time.perf_counter()
    logs = []
    try:
        import torch
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        _check_device(torch, device)
        random.seed(seed)
        torch.manual_seed(seed)
        load_start = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
            local_files_only=True, trust_remote_code=False, attn_implementation="sdpa").to(device)
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, lora_dropout=0.05,
            target_modules=TARGET_MODULES, bias="none", task_type="CAUSAL_LM"))
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=2e-4, betas=(0.9, 0.999), eps=1e-8,
                                      weight_decay=0.01, foreach=False)
        _synchronize(torch, device)
        load_seconds = time.perf_counter() - load_start
        model.train()
        stream = index_stream([len(e["input_ids"]) for e in examples], seed)
        exposures = Counter()
        training_start = time.perf_counter()
        seen_supervised_tokens = 0
        with (output_dir / "loss-log.jsonl").open("x", encoding="utf-8") as loss_stream:
            for step in range(1, steps + 1):
                step_start = time.perf_counter()
                indices = [next(stream) for _ in range(4)]
                counts = [len(examples[i]["completion_ids"]) for i in indices]
                weights = accumulation_weights(counts)
                optimizer.zero_grad(set_to_none=True)
                mean_loss = 0.0
                for index, weight in zip(indices, weights):
                    item = examples[index]
                    batch = {key: torch.tensor([item[key]], device=device, dtype=torch.long)
                             for key in ("input_ids", "attention_mask", "labels")}
                    output = model(**batch)
                    loss = output.loss
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("Non-finite training loss")
                    (loss * weight).backward()
                    mean_loss += float(loss.detach().cpu()) * weight
                    exposures[rows[index]["id"]] += 1
                    del output, loss, batch
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                if not bool(torch.isfinite(grad_norm)):
                    raise RuntimeError("Non-finite gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                _synchronize(torch, device)
                memory = sampled_memory(torch, device)
                entry = {"step": step, "loss": mean_loss, "supervised_tokens": sum(counts),
                         "sequence_tokens": sum(len(examples[i]["input_ids"]) for i in indices),
                         "row_ids": [rows[i]["id"] for i in indices], "grad_norm_before_clip": float(grad_norm.detach().cpu()),
                         "seconds": time.perf_counter() - step_start, "memory_sample": memory}
                logs.append(entry)
                seen_supervised_tokens += sum(counts)
                loss_stream.write(canonical(entry) + "\n")
                loss_stream.flush()
                if step == 1 or step % 5 == 0 or step == steps:
                    print(canonical({"step": step, "steps": steps, "loss": mean_loss,
                                     "seen_supervised_tokens": seen_supervised_tokens,
                                     "elapsed_seconds": time.perf_counter() - training_start}), flush=True)
                    if device == "mps":
                        torch.mps.empty_cache()
                del grad_norm
                if target_supervised_tokens is not None and seen_supervised_tokens >= target_supervised_tokens:
                    break
        training_seconds = time.perf_counter() - training_start
        if target_supervised_tokens is not None and seen_supervised_tokens < target_supervised_tokens:
            raise RuntimeError("Maximum optimizer steps reached before the supervised-token budget; adapter is not exported")
        adapter_dir = output_dir / "adapter"
        # PEFT otherwise writes the machine's absolute source directory.
        model.peft_config["default"].base_model_name_or_path = MODEL_ID
        model.peft_config["default"].revision = REVISION
        model.save_pretrained(adapter_dir, safe_serialization=True)
        adapter_files = {name: {"sha256": file_hash(adapter_dir / name), "bytes": (adapter_dir / name).stat().st_size}
                         for name in ("adapter_config.json", "adapter_model.safetensors")}
        metadata = {"schema_version": 1, "status": "completed", "base": base,
                    "data_sha256": binding["data_sha256"], "config": config, "config_sha256": object_hash(config),
                    "run_manifest_sha256": file_hash(output_dir / "run_manifest.json"),
                    "training_source": binding["source"], "adapter_files": adapter_files,
                    "completed_optimizer_steps": len(logs), "seen_supervised_tokens": seen_supervised_tokens,
                    "last_window_supervised_tokens": logs[-1]["supervised_tokens"], "optimizer_state_exported": False}
        write_new_json(adapter_dir / "training_manifest.json", metadata)
        memory_keys = {key for row in logs for key in row["memory_sample"]}
        result = {"status": "completed", "binding_sha256": object_hash(binding),
                  "completed_optimizer_steps": len(logs), "microbatches": len(logs) * 4,
                  "examples_seen": len(logs) * 4, "dataset_equivalent_epochs": len(logs) * 4 / len(rows),
                  "target_supervised_tokens": target_supervised_tokens,
                  "token_budget_overshoot": seen_supervised_tokens - target_supervised_tokens if target_supervised_tokens else 0,
                  "last_window_supervised_tokens": logs[-1]["supervised_tokens"],
                  "load_seconds": load_seconds, "training_seconds": training_seconds,
                  "total_seconds": time.perf_counter() - started,
                  "trainable_parameters": sum(p.numel() for p in parameters),
                  "trainable_dtypes": sorted({str(p.dtype) for p in parameters}),
                  "seen_supervised_tokens": sum(x["supervised_tokens"] for x in logs),
                  "seen_sequence_tokens": sum(x["sequence_tokens"] for x in logs),
                  "example_exposures": dict(sorted(exposures.items())), "final_loss": logs[-1]["loss"],
                  "maximum_observed_memory_samples": {key: max(x["memory_sample"].get(key, 0) for x in logs) for key in memory_keys},
                  "memory_note": "Post-step MPS/current RSS samples only; sample maxima are not true peaks.",
                  "determinism_note": "Seed and data order are fixed; bitwise cross-device/kernel reproducibility is not claimed.",
                  "artifact_sha256": {"loss-log.jsonl": file_hash(output_dir / "loss-log.jsonl"),
                                      "adapter/training_manifest.json": file_hash(adapter_dir / "training_manifest.json"),
                                      **{f"adapter/{name}": value["sha256"] for name, value in adapter_files.items()}}}
        write_new_json(output_dir / "summary.json", result)
        return result
    except BaseException as exc:
        write_new_json(output_dir / "failure.json", {"status": "failed", "error_type": type(exc).__name__,
            "completed_logged_steps": len(logs), "elapsed_seconds": time.perf_counter() - started,
            "binding_sha256": object_hash(binding), "successful_adapter_claim": False})
        raise


def adapter_provenance(adapter_path: Path | None, base: dict[str, Any]) -> dict[str, Any] | None:
    if adapter_path is None:
        return None
    path = Path(adapter_path)
    manifest_path = path / "training_manifest.json"
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed" or metadata.get("base") != base:
        raise ValueError("Adapter is incomplete or belongs to a different base snapshot")
    if metadata.get("config_sha256") != object_hash(metadata.get("config")):
        raise ValueError("Adapter training config hash mismatch")
    validate_training_budget(metadata)
    if set(metadata.get("adapter_files", {})) != {"adapter_config.json", "adapter_model.safetensors"}:
        raise ValueError("Adapter manifest file contract mismatch")
    for name, expected in metadata["adapter_files"].items():
        if file_hash(path / name) != expected["sha256"] or (path / name).stat().st_size != expected["bytes"]:
            raise ValueError(f"Adapter file checksum/size mismatch: {name}")
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODEL_ID or config.get("revision") != REVISION:
        raise ValueError("Adapter configuration has a different base identity")
    training_config = metadata["config"]
    if any(config.get(key) != training_config.get(key) for key in ("r", "lora_alpha", "lora_dropout")):
        raise ValueError("Adapter LoRA configuration differs from registered training config")
    if sorted(config.get("target_modules", [])) != sorted(training_config.get("target_modules", [])):
        raise ValueError("Adapter target modules differ from registered training config")
    return {"training_manifest_sha256": file_hash(manifest_path), "metadata": metadata}


def validate_training_budget(metadata: dict[str, Any]) -> None:
    config = metadata["config"]
    steps = metadata.get("completed_optimizer_steps")
    if not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= config["steps"]:
        raise ValueError("Invalid completed optimizer-step count")
    target = config.get("target_supervised_tokens")
    if target is None:
        if steps != config["steps"]:
            raise ValueError("Adapter did not complete its registered training steps")
    else:
        seen, last = metadata.get("seen_supervised_tokens"), metadata.get("last_window_supervised_tokens")
        if (not isinstance(seen, int) or isinstance(seen, bool) or not isinstance(last, int)
                or isinstance(last, bool) or not 0 < last <= seen or seen < target or seen - last >= target):
            raise ValueError("Adapter did not stop at the first completed window reaching the token budget")


def case_seed(prompt: str, seed: int) -> int:
    digest = hashlib.sha256((f"deltaweave-generation-v1:{seed}\0" + prompt).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def exact_json_equal(left: Any, right: Any) -> bool:
    """Object order is ignored; extra keys, array order and scalar types matter."""
    return canonical(left) == canonical(right)


def score_response(raw: str, target: dict[str, Any], completed: bool) -> dict[str, Any]:
    strict, auxiliary, error = None, None, None
    try:
        strict = contract.parse_generated_response(raw)
    except (ValueError, TypeError) as exc:
        error = str(exc)
    auxiliary = strict
    auxiliary_mode = "strict" if strict is not None else None
    if strict is None and raw.count("</think>") == 1:
        before, after = raw.split("</think>", 1)
        # Auxiliary score allows only a sole JSON fence in the final segment.
        # It never selects among objects, removes prose, or changes claim values.
        match = re.fullmatch(r"\s*```(?:json)?[ \t]*\n(.*?)\n```\s*", after, flags=re.DOTALL)
        if match:
            try:
                auxiliary = contract.parse_generated_response(before + "</think>" + match.group(1))
                auxiliary_mode = "sole_json_fence"
            except (ValueError, TypeError):
                pass
    return {"strict_parsed": strict, "strict_parse_error": error,
            "auxiliary_parsed": auxiliary, "auxiliary_mode": auxiliary_mode,
            "format_valid": strict is not None,
            "strict_success": bool(completed and strict is not None and exact_json_equal(strict, target)),
            "auxiliary_success": bool(completed and auxiliary is not None and exact_json_equal(auxiliary, target)),
            "strict_action_match": bool(completed and strict is not None and strict.get("action") == target["action"]),
            "auxiliary_action_match": bool(completed and auxiliary is not None and auxiliary.get("action") == target["action"])}


def generation_record(tokenizer: Any, generated_ids: list[int], *, input_tokens: int, seconds: float,
                      timed_out: bool, config: dict[str, Any], error: dict[str, str] | None = None) -> dict[str, Any]:
    ended = bool(generated_ids and generated_ids[-1] == tokenizer.eos_token_id)
    hit_token_cap = len(generated_ids) >= config["max_new_tokens"] and not ended
    # A deadline reached during the last step counts as timeout even if that step
    # happened to emit EOS. Record this separately from physical EOS termination.
    timed_out = bool(timed_out or seconds >= config["max_seconds"])
    raw = tokenizer.decode(generated_ids[:-1] if ended else generated_ids,
                           skip_special_tokens=False, clean_up_tokenization_spaces=False)
    completed = ended and not timed_out and not hit_token_cap and error is None
    if error is not None:
        stop_reason = "error"
    elif timed_out:
        stop_reason = "timeout"
    elif hit_token_cap:
        stop_reason = "token_cap"
    elif ended:
        stop_reason = "eos"
    else:
        stop_reason = "unterminated"
    close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    boundaries = [i for i in range(len(generated_ids)) if generated_ids[i:i + len(close_ids)] == close_ids]
    boundary = boundaries[0] if len(boundaries) == 1 else None
    payload_end = len(generated_ids) - (1 if ended else 0)
    return {"raw": raw, "generated_ids": generated_ids, "generated_ids_sha256": object_hash(generated_ids),
            "input_tokens": input_tokens, "new_tokens": len(generated_ids), "seconds": seconds,
            "tokens_per_second": len(generated_ids) / seconds if seconds > 0 else None,
            "eos": ended, "timed_out": timed_out, "hit_token_cap": hit_token_cap,
            "completed": completed, "stop_reason": stop_reason, "error": error,
            "timeout_overshoot_seconds": max(0.0, seconds - config["max_seconds"]),
            "thinking_tokens": boundary,
            "final_segment_tokens": payload_end - boundary - len(close_ids) if boundary is not None else None}


class HFPolicy:
    """One local model load; predict accepts only a public prompt, never a row/gold."""
    def __init__(self, model_path: Path, adapter_path: Path | None, tokenizer: Any, config: dict[str, Any]):
        import torch
        from transformers import AutoModelForCausalLM
        self.torch, self.tokenizer, self.config = torch, tokenizer, config
        self.device = config["device"]
        _check_device(torch, self.device)
        start = time.perf_counter()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
            local_files_only=True, trust_remote_code=False, attn_implementation="sdpa").to(self.device)
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False, local_files_only=True)
        self.model.eval()
        _synchronize(torch, self.device)
        self.load_seconds = time.perf_counter() - start

    def predict(self, prompt: str, seed: int) -> dict[str, Any]:
        from transformers import StoppingCriteria, StoppingCriteriaList
        torch, tokenizer, config = self.torch, self.tokenizer, self.config
        input_count, generated = 0, []
        start = time.perf_counter()
        error = None
        class Deadline(StoppingCriteria):
            def __init__(self):
                self.timed_out = False
            def __call__(self, input_ids, scores, **kwargs):
                self.timed_out = time.perf_counter() - start >= config["max_seconds"]
                return self.timed_out
        deadline = Deadline()
        try:
            ids = contract.encode_prompt(tokenizer, prompt)
            input_count = len(ids)
            if input_count > config["max_input_tokens"]:
                raise ValueError("Prompt exceeds max_input_tokens; no truncation permitted")
            inputs = torch.tensor([ids], device=self.device, dtype=torch.long)
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                    max_new_tokens=config["max_new_tokens"], do_sample=True,
                    temperature=config["temperature"], top_p=config["top_p"], top_k=0,
                    repetition_penalty=1.0, num_beams=1,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                    stopping_criteria=StoppingCriteriaList([deadline]), use_cache=True)
            _synchronize(torch, self.device)
            generated = output[0, len(ids):].tolist()
            del output, inputs
        except Exception as exc:
            # OOM and malformed-input rows remain denominator-preserving failures.
            error = {"type": type(exc).__name__, "message": str(exc)[:1000]}
        elapsed = time.perf_counter() - start
        record = generation_record(tokenizer, generated, input_tokens=input_count, seconds=elapsed,
                                   timed_out=deadline.timed_out, config=config, error=error)
        record["memory_sample"] = sampled_memory(torch, self.device)
        if self.device == "mps":
            torch.mps.empty_cache()
        return record


def evaluation_config(*, seed: int = 42, device: str = "mps", max_new_tokens: int = 2048,
                      max_seconds: float = 120.0, max_input_tokens: int = 1024) -> dict[str, Any]:
    for name, value in (("max_new_tokens", max_new_tokens), ("max_input_tokens", max_input_tokens)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool) or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if device not in {"cpu", "mps"}:
        raise ValueError("device must be local cpu or mps")
    return {"seed": seed, "device": device, "dtype": "bfloat16", "max_new_tokens": max_new_tokens,
            "max_seconds": float(max_seconds), "max_input_tokens": max_input_tokens,
            "do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 0,
            "num_beams": 1, "repetition_penalty": 1.0, "use_cache": True,
            "seed_derivation": "sha256_deltaweave-generation-v1_seed_prompt_mod_2**31",
            "completion_contract": "explicit_ids_close_think_json_eos_v1",
            "auxiliary_parser": "sole_json_fence_only_no_value_repair_v1"}


def receipt_path(directory: Path, row: dict[str, Any]) -> Path:
    return directory / (hashlib.sha256(row["id"].encode()).hexdigest() + ".json")


def make_receipt(row: dict[str, Any], prediction: dict[str, Any], settings_sha256: str, config: dict[str, Any]) -> dict[str, Any]:
    payload = {"schema_version": 1, "settings_sha256": settings_sha256, "row_sha256": object_hash(row),
               "id": row["id"], "split": row["split"], "group": row["group"], "kind": row["kind"],
               "prompt_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(),
               "case_seed": case_seed(row["prompt"], config["seed"]), "prediction": prediction,
               "score": score_response(prediction["raw"], row["target"], prediction["completed"])}
    return {"receipt_sha256": object_hash(payload), "payload": payload}


def validate_receipt(receipt: dict[str, Any], row: dict[str, Any], settings_sha256: str,
                     config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    if set(receipt) != {"receipt_sha256", "payload"} or receipt["receipt_sha256"] != object_hash(receipt["payload"]):
        raise ValueError("Receipt integrity hash mismatch")
    payload = receipt["payload"]
    prediction = payload.get("prediction", {})
    ids = prediction.get("generated_ids")
    seconds = prediction.get("seconds")
    if (not isinstance(ids, list) or any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in ids)
            or len(ids) > config["max_new_tokens"] or not isinstance(seconds, (int, float))
            or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0):
        raise ValueError("Invalid cached generation IDs/time")
    recomputed = generation_record(tokenizer, ids, input_tokens=prediction["input_tokens"], seconds=seconds,
        timed_out=prediction["timed_out"], config=config, error=prediction.get("error"))
    for key, value in recomputed.items():
        if not exact_json_equal(prediction.get(key), value):
            raise ValueError(f"Cached prediction cannot be reconstructed: {key}")
    if prediction.get("error") is None and prediction["input_tokens"] != len(contract.encode_prompt(tokenizer, row["prompt"])):
        raise ValueError("Cached prompt token count mismatch")
    expected = make_receipt(row, prediction, settings_sha256, config)
    if not exact_json_equal(receipt, expected):
        raise ValueError("Receipt row/settings/seed/score binding mismatch")
    return payload


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(cases)
    if not count:
        raise ValueError("Cannot summarize no cases")
    def count_score(key):
        return sum(bool(c["score"][key]) for c in cases)
    actions = {}
    for action in sorted({c["target"]["action"] for c in cases}):
        selected = [c for c in cases if c["target"]["action"] == action]
        actions[action] = {"count": len(selected), "strict_successes": sum(c["score"]["strict_success"] for c in selected),
                           "auxiliary_successes": sum(c["score"]["auxiliary_success"] for c in selected)}
    latencies = sorted(c["prediction"]["seconds"] for c in cases)
    groups = {c["group"] for c in cases}
    correct_groups = sum(all(c["score"]["strict_success"] for c in cases if c["group"] == g) for g in groups)
    return {"cases": count, "strict_successes": count_score("strict_success"),
            "strict_success_rate": count_score("strict_success") / count,
            "auxiliary_successes": count_score("auxiliary_success"),
            "auxiliary_success_rate": count_score("auxiliary_success") / count,
            "format_valid": count_score("format_valid"), "strict_action_matches": count_score("strict_action_match"),
            "stop_reasons": dict(sorted(Counter(c["prediction"]["stop_reason"] for c in cases).items())),
            "eos_count": sum(c["prediction"]["eos"] for c in cases),
            "timeout_count": sum(c["prediction"]["timed_out"] for c in cases),
            "token_cap_count": sum(c["prediction"]["hit_token_cap"] for c in cases),
            "new_tokens_total": sum(c["prediction"]["new_tokens"] for c in cases),
            "generation_seconds_total": sum(latencies),
            "latency_p50_seconds": latencies[max(0, math.ceil(count * 0.5) - 1)],
            "latency_p95_seconds": latencies[max(0, math.ceil(count * 0.95) - 1)],
            "represented_groups": len(groups), "all_observed_group_rows_correct": correct_groups,
            "group_metric_note": "Only rows in this evaluated file/subset; not a completeness claim for unseen group variants.",
            "by_target_action": actions}


def evaluate_model(data_path: Path, model_path: Path, output_path: Path, *, adapter_path: Path | None = None,
                   seed: int = 42, device: str = "mps", max_new_tokens: int = 2048,
                   max_seconds: float = 120.0, max_input_tokens: int = 1024,
                   limit_dev: int | None = None, resume: bool = False) -> dict[str, Any]:
    output_path, data_path, model_path = Path(output_path), Path(data_path), Path(model_path)
    if output_path.exists():
        raise FileExistsError("Completed evaluation outputs are immutable; resume only unfinished receipt directories")
    config = evaluation_config(seed=seed, device=device, max_new_tokens=max_new_tokens,
                               max_seconds=max_seconds, max_input_tokens=max_input_tokens)
    rows = load_rows(data_path)
    full_count = len(rows)
    if limit_dev is not None:
        if rows[0]["split"] != "dev" or not isinstance(limit_dev, int) or isinstance(limit_dev, bool) or limit_dev < 1:
            raise ValueError("--limit-dev is positive and allowed only for an entirely dev dataset")
        rows = sorted(rows, key=lambda row: row["id"])[:limit_dev]
    base = model_provenance(model_path)
    adapter = adapter_provenance(adapter_path, base)
    tokenizer = load_tokenizer(model_path)
    binding = {"schema_version": 1, "base": base, "adapter": adapter, "config": config,
               "data_sha256": file_hash(data_path), "source": source_provenance(),
               "rows_sha256": object_hash(rows), "selected_ids": [r["id"] for r in rows],
               "split": rows[0]["split"], "limit_dev": limit_dev, "full_file_rows": full_count,
               "tokenizer_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
    settings_sha256 = object_hash(binding)
    cache_dir = output_path.with_suffix(output_path.suffix + ".receipts")
    if cache_dir.exists():
        if not resume:
            raise FileExistsError("Receipt directory exists; explicitly use --resume with identical settings")
        previous = json.loads((cache_dir / "run_manifest.json").read_text(encoding="utf-8"))
        if not exact_json_equal(previous, binding):
            raise ValueError("Resume settings/data/source/model/adapter binding mismatch")
    else:
        if resume:
            raise FileNotFoundError("--resume requires an existing unfinished receipt directory")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(exist_ok=False)
        write_new_json(cache_dir / "run_manifest.json", binding)
    allowed = {receipt_path(cache_dir, row).name for row in rows} | {"run_manifest.json"}
    unexpected = [p.name for p in cache_dir.iterdir() if p.name not in allowed and ".partial-" not in p.name]
    if unexpected:
        raise ValueError("Unexpected cached receipts; refusing cross-run reuse")
    cases, cached, missing = [], {}, []
    # Validate *all* existing receipts before any model load.
    for row in rows:
        path = receipt_path(cache_dir, row)
        if path.exists():
            cached[row["id"]] = validate_receipt(json.loads(path.read_text()), row, settings_sha256, config, tokenizer)
        else:
            missing.append(row["id"])
    started = time.perf_counter()
    policy = HFPolicy(model_path, adapter_path, tokenizer, config) if missing else None
    for index, row in enumerate(rows, 1):
        if row["id"] in cached:
            payload = cached[row["id"]]
        else:
            prediction = policy.predict(row["prompt"], case_seed(row["prompt"], seed))
            receipt = make_receipt(row, prediction, settings_sha256, config)
            payload = validate_receipt(receipt, row, settings_sha256, config, tokenizer)
            atomic_new_json(receipt_path(cache_dir, row), receipt)
        cases.append({**payload, "prompt": row["prompt"], "target": row["target"]})
        print(canonical({"case": index, "total": len(rows), "id": row["id"],
                         "cached": row["id"] in cached, "stop": payload["prediction"]["stop_reason"],
                         "strict_success": payload["score"]["strict_success"]}), flush=True)
    report = {"schema_version": 1, "status": "completed", "settings_sha256": settings_sha256,
              "binding": binding, "metrics": summarize_cases(cases), "cases": cases,
              "cached_cases": len(cached), "new_cases": len(missing),
              "this_invocation_seconds": time.perf_counter() - started,
              "model_load_seconds": policy.load_seconds if policy is not None else 0,
              "receipts_sha256": {receipt_path(cache_dir, row).name: file_hash(receipt_path(cache_dir, row)) for row in rows},
              "memory_note": "Per-case MPS/current RSS samples are not true peaks.",
              "auxiliary_note": "Only a sole JSON fence is tolerated; action/payload/extra keys must still exactly match. Main score is strict.",
              "termination_note": "Any timeout, cap without EOS, or generation error is a failure, even if a partial JSON parses."}
    atomic_new_json(output_path, report)
    return report
