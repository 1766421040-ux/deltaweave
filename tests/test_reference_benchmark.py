"""Independent confirmation audit tests; no model/GPU execution."""
from copy import deepcopy
import json

import pytest

import benchmark_reference as benchmark


def correct_policy(prompt):
    # A test double, never included in published model measurements.
    state = json.loads(prompt.split("\nSTATE=", 1)[1])
    return benchmark.expected_action(state), {"completed": True, "input_tokens": 12, "new_tokens": 6, "seconds": 0.01}


def test_confirmation_has_16_independent_chains_and_48_states():
    cases = benchmark.build_cases()
    assert len(cases) == 16 and sum(len(c["events"]) for c in cases) == 48
    assert {c["family"] for c in cases} == set(benchmark.FAMILIES)
    assert len({c["id"] for c in cases}) == 16
    assert cases == benchmark.build_cases()
    assert cases != benchmark.build_cases(benchmark.SEED + 1)


def test_all_views_match_oracle_with_long_payloads_and_nonanswer_controls():
    cases, config = benchmark.build_cases(), benchmark.configuration()
    results = {view: benchmark.run_view(cases, config, view) for view in benchmark.VIEWS}
    full = results["full"]
    assert max(len(r["result"]["answer"]["arguments"].get("citations", [])) for r in full["records"]) >= 25
    assert max(len(r["result"]["answer"]["arguments"].get("selection", {})) for r in full["records"]) >= 5
    for view, report in results.items():
        summary = report["summary"]
        assert summary["all_sessions_completed"]["numerator"] == 48
        assert summary["oracle_appropriate_terminal"]["numerator"] == 48
        assert summary["fresh_raw_contract"] == {"numerator": 0, "denominator": 0, "rate": None}
        assert summary["wrong_reference_acceptance"]["numerator"] == 0
        assert {"wait", "ask", "abstain", "answer"} <= summary["expected_terminal_actions"].keys()
        assert [r["result"]["answer"] for r in report["records"]] == [r["result"]["answer"] for r in full["records"]]
        assert any(s["raw_action"]["action"] == "refresh" for r in report["records"] for s in r["result"]["steps"])


def test_cached_failures_are_not_counted_as_new_model_generations():
    case = deepcopy(benchmark.build_cases()[0])
    case["events"] = [benchmark.event("first"), benchmark.event("same-input")]
    calls = []
    def failed(prompt):
        calls.append(prompt)
        return None, {"completed": True, "input_tokens": 17, "new_tokens": 9, "seconds": .1}
    report = benchmark.run_view([case], benchmark.configuration(), "reference", policy=failed, policy_identity="test-invalid")
    summary = report["summary"]
    assert len(calls) == 1 and summary["cache_hits"] == 1
    assert summary["fresh_raw_contract"] == {"numerator": 0, "denominator": 1, "rate": 0.0}
    assert summary["all_sessions_completed"]["numerator"] == 0
    assert summary["full_answer_oracle_agreement"]["rate"] is None
    assert summary["wrong_reference_acceptance"]["rate"] is None
    assert summary["fresh_generation_cost"]["new_tokens"]["total"] == 9


def test_auditor_detects_wrong_binding_even_if_host_claims_accepted():
    case = benchmark.build_cases()[0]
    record = benchmark.run_view([case], benchmark.configuration(), "reference")["records"][0]
    corrupted = deepcopy(record["result"])
    corrupted["steps"][-1]["proof_record"]["runtime_binding"]["snapshot_revision"] += 1
    check = benchmark.assess(corrupted, case["tables"], case["query"], True, record["oracle"])
    assert check["step_checks"][-1]["wrong_reference_accepted"] is True
    # Conversely, a guard blocking before expansion must not be counted as an
    # incorrect delivered answer merely because raw-action comparison passed.
    corrupted["steps"][-1]["expanded_answer"] = None
    corrupted["answer"] = None
    corrupted["status"] = "blocked_evidence_binding"
    check = benchmark.assess(corrupted, case["tables"], case["query"], True, record["oracle"])
    assert check["step_checks"][-1]["wrong_reference_accepted"] is False
    assert check["oracle_appropriate_terminal"] is False


def test_oracle_detects_plausible_but_unavailable_rendered_version():
    case = benchmark.build_cases()[0]
    record = benchmark.run_view([case], benchmark.configuration(), "full")["records"][0]
    corrupted = deepcopy(record["result"])
    project = next(iter(corrupted["answer"]["arguments"]["selection"]))
    corrupted["answer"]["arguments"]["selection"][project] = "99.0"
    check = benchmark.assess(corrupted, case["tables"], case["query"], True, record["oracle"])
    assert check["full_answer_oracle_valid"] is False


def test_freeze_precedes_audit_refuses_overwrite_and_detects_input_tampering(tmp_path, monkeypatch):
    output = tmp_path / "bundle"
    original = benchmark.run_view
    def check_frozen(*args, **kwargs):
        assert (output / "manifest.json").exists()
        return original(*args, **kwargs)
    monkeypatch.setattr(benchmark, "run_view", check_frozen)
    audit = benchmark.prepare(output)
    assert audit["passed"] and audit["model_used"] is False
    benchmark.verify_bundle(output)
    with pytest.raises(FileExistsError):
        benchmark.prepare(output)
    with (output / "cases.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="Frozen input"):
        benchmark.verify_bundle(output)


def test_missing_token_receipt_is_unknown_not_zero():
    case = benchmark.build_cases()[0]
    def without_cost(prompt):
        action, _ = correct_policy(prompt)
        return action, {"completed": True}
    report = benchmark.run_view([case], benchmark.configuration(), "reference", policy=without_cost, policy_identity="test-no-cost")
    assert report["summary"]["fresh_generation_cost"]["new_tokens"]["total"] is None
    assert report["summary"]["fresh_generation_cost"]["new_tokens"]["missing_receipts"] > 0
