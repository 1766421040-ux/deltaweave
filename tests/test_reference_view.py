"""CPU reference-transport tests; no model-quality claims or generation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from deltaweave.engine import Engine, Query
from deltaweave.evidence import EvidenceBindingError, ReferenceRegistry
from deltaweave.learning_data import CONTRACT, decide, prompt_for
from deltaweave.provenance import execution_binding
from deltaweave.session import KnowledgeSession, normalize_state, reference_prompt_for, REFERENCE_PROMPT_VERSION
from deltaweave.store import Store, canonical


QUERY = {"requirements": ["p>=1"], "environment": "local", "as_of": "2026-10-28"}


def resolve(ctx, args):
    rows = ctx.query(Query("items", where={"project": "p"}))
    return {"status": "resolved" if rows else "incomplete",
            "selection": {row["package"]: row["version"] for row in rows},
            "citations": [row["id"] for row in rows], "issues": [],
            "constraints": [{"package": row["package"], "required": ">=1"} for row in rows],
            "scope": "finite test universe"}


def setup(view="reference", policy=None):
    store = Store({"environment": [{"id": "py", "environment": "local",
                    "field": "python_full_version", "value": "3.11.9"}],
                   "items": [{"id": f"m{i}", "project": "p", "package": f"pkg{i}",
                              "version": "1.0"} for i in range(5)]})
    engine = Engine(store); engine.register("resolve", resolve)
    session = KnowledgeSession(engine, evidence_view=view, policy=policy,
        policy_identity="test-frozen-policy-seed42" if policy else None)
    return store, session


def correct(prompt):
    return decide(json.loads(prompt.split("\nSTATE=", 1)[1])), {"completed": True}


def test_three_views_preserve_full_public_answer_and_separate_model_action():
    responses = {view: setup(view, correct)[1].request(QUERY)
                 for view in ("full", "reference", "reference_context")}
    full = responses["full"]
    assert all(result["status"] == "completed" for result in responses.values())
    assert all(result["answer"] == full["answer"] for result in responses.values())
    assert len(full["answer"]["arguments"]["selection"]) == 5
    assert full["raw_action"] == full["answer"]
    assert full["expanded_answer"] is None
    compact, context = responses["reference"], responses["reference_context"]
    assert compact["raw_action"] == context["raw_action"]
    assert compact["raw_action"] != compact["answer"]
    assert compact["expanded_answer"] == compact["answer"]
    assert context["steps"][-1]["state"]["full_evidence"] == full["steps"][-1]["state"]["observation"]
    assert "full_evidence" not in compact["steps"][-1]["state"]
    assert compact["proof_record"]["resolver_result"]["constraints"]
    assert compact["proof_record"]["semantic"]["answer"] == compact["answer"]
    assert compact["proof_record"]["solver_receipt"]["validated_at"] == compact["snapshot_revision"]


@pytest.mark.parametrize("view", ["reference", "reference_context"])
def test_unrelated_commit_reuses_raw_action_but_rebinds_current_host_observation(view):
    calls = []
    def policy(prompt):
        calls.append(prompt)
        return correct(prompt)
    store, session = setup(view, policy)
    first = session.request(QUERY)
    store.commit(upserts={"items": [{"id": "unrelated", "project": "q", "package": "other", "version": "9"}]})
    second = session.request(QUERY, prevalidate=True)
    assert len(calls) == 2  # Initial refresh + answer, no new generation on reuse.
    assert second["fresh_model_calls"] == 0 and second["model_cache_hits"] == 1
    assert first["proof_record"]["handle"] == second["proof_record"]["handle"]
    assert first["proof_record"]["content_sha256"] == second["proof_record"]["content_sha256"]
    assert first["proof_record"]["proof_sha256"] != second["proof_record"]["proof_sha256"]
    assert first["proof_record"]["runtime_binding"]["snapshot_revision"] == 0
    assert second["proof_record"]["runtime_binding"]["snapshot_revision"] == 1
    assert second["answer"] == first["answer"]
    if view == "reference_context":
        normalized = normalize_state(second["steps"][-1]["state"])
        assert normalized["full_evidence"]["validated_at"] == normalized["target_revision"] == 100
        assert normalized["full_evidence"]["computed_at"] == 98


def test_stale_reference_still_requires_refresh_and_changed_answer_changes_handle():
    store, session = setup(policy=correct)
    first = session.request(QUERY)
    store.commit(upserts={"items": [{"id": "m0", "project": "p", "package": "pkg0", "version": "2.0"}]})
    second = session.request(QUERY)
    assert [step["raw_action"]["action"] for step in second["steps"]] == ["refresh", "answer"]
    assert second["steps"][0]["expanded_answer"] is None
    assert second["proof_record"]["handle"] != first["proof_record"]["handle"]
    assert second["answer"]["arguments"]["selection"]["pkg0"] == "2.0"


def test_replayed_old_reference_is_blocked_without_repairing_raw_action():
    replay = []
    def policy(prompt):
        action, receipt = correct(prompt)
        if action["action"] == "answer":
            if replay:
                return deepcopy(replay[0]), receipt
            replay.append(deepcopy(action))
        return action, receipt
    store, session = setup(policy=policy)
    first = session.request(QUERY)
    store.commit(upserts={"items": [{"id": "m0", "project": "p", "package": "pkg0", "version": "2.0"}]})
    second = session.request(QUERY, prevalidate=True)
    assert second["status"] == "blocked_model_contract"
    assert second["answer"] is None and second["expanded_answer"] is None
    assert second["raw_action"] == first["raw_action"]
    assert second["proof_record"]["handle"] != first["proof_record"]["handle"]


def test_host_record_is_immutable_and_rejects_wrong_snapshot_registry_and_mutation():
    store, session = setup()
    response = session.request(QUERY)
    record = session._observations[canonical(QUERY)]["proof"]
    with pytest.raises(FrozenInstanceError):
        record.handle = "changed"
    detached = record.export()
    detached["resolver_result"]["selection"].clear()
    assert record.expand(response["raw_action"], query=QUERY,
                         binding=response["runtime_binding"]) == response["answer"]
    with pytest.raises(EvidenceBindingError):
        replace(record, _proof="{}").export()
    store.commit(upserts={"items": [{"id": "other", "project": "q", "package": "z", "version": "9"}]})
    with pytest.raises(EvidenceBindingError, match="bound"):
        record.expand(response["raw_action"], query=QUERY,
                      binding=execution_binding(session.engine, store.snapshot))
    changed_registry = deepcopy(response["runtime_binding"])
    changed_registry["registry_versions"]["resolve"] = "2"
    with pytest.raises(EvidenceBindingError, match="bound"):
        record.expand(response["raw_action"], query=QUERY, binding=changed_registry)
    session._observations[canonical(QUERY)]["proof"] = replace(record, _proof="{}")
    # With the old observation retained, corruption is blocked before even an
    # otherwise valid refresh action can smuggle through an unverified record.
    assert session.request(QUERY)["status"] == "blocked_evidence_binding"


def test_short_handle_collision_fails_closed(monkeypatch):
    import deltaweave.evidence as evidence
    monkeypatch.setattr(evidence, "short_handle", lambda digest: "hforcedcollision")
    store, session = setup()
    assert session.request(QUERY)["status"] == "completed"
    store.commit(upserts={"items": [{"id": "m0", "project": "p", "package": "pkg0", "version": "2.0"}]})
    result = session.request(QUERY, prevalidate=True)
    assert result["status"] == "blocked_evidence_binding"
    assert result["answer"] is None


def test_concurrent_commit_during_model_callback_cannot_rebind_local_record():
    committed = []
    store = None
    def policy(prompt):
        action, receipt = correct(prompt)
        if action["action"] == "answer" and not committed:
            store.commit(upserts={"items": [{"id": "m0", "project": "p", "package": "pkg0", "version": "2.0"}]})
            committed.append(True)
        return action, receipt
    store, session = setup(policy=policy)
    first = session.request(QUERY)
    assert first["status"] == "completed"
    assert first["snapshot_revision"] == first["proof_record"]["runtime_binding"]["snapshot_revision"] == 0
    assert first["answer"]["arguments"]["selection"]["pkg0"] == "1.0"
    assert session.request(QUERY)["answer"]["arguments"]["selection"]["pkg0"] == "2.0"


def test_unknown_reference_and_incomplete_evidence_cannot_expand():
    store, session = setup()
    response = session.request(QUERY)
    record = session._observations[canonical(QUERY)]["proof"]
    unknown = deepcopy(response["raw_action"])
    unknown["arguments"]["selection"]["answer_ref"] = "hunknown"
    with pytest.raises(EvidenceBindingError, match="exactly"):
        record.expand(unknown, query=QUERY, binding=response["runtime_binding"])
    store.commit(deletes={"items": [f"m{i}" for i in range(5)]})
    result = session.request(QUERY)
    assert result["answer"]["action"] == "abstain"
    assert result["expanded_answer"] is None
    incomplete = session._observations[canonical(QUERY)]["proof"]
    with pytest.raises(EvidenceBindingError, match="Incomplete"):
        incomplete.expand(response["raw_action"], query=QUERY, binding=result["runtime_binding"])


def test_full_prompt_remains_byte_identical_to_original_contract():
    observed = []
    def policy(prompt):
        observed.append(prompt)
        return correct(prompt)
    _, session = setup("full", policy)
    response = session.request(QUERY)
    assert observed == [prompt_for(normalize_state(step["state"])) for step in response["steps"]]
    assert all(prompt.startswith(CONTRACT) for prompt in observed)
    assert all("For conflict, selection is {}." in prompt for prompt in observed)
    assert all("Reference response protocol" not in prompt for prompt in observed)


@pytest.mark.parametrize("view", ["reference", "reference_context"])
def test_reference_conflict_prompt_has_one_consistent_answer_rule(view):
    observed = []
    def policy(prompt):
        observed.append(prompt)
        return correct(prompt)
    _, session = setup(view, policy)
    def conflict(ctx, args):
        result = resolve(ctx, args)
        result.update(status="conflict", selection={})
        return result
    session.engine.register("resolve", conflict, version="conflict-fixture")
    response = session.request(QUERY)
    assert response["status"] == "completed"
    assert response["answer"]["arguments"]["status"] == "conflict"
    assert response["answer"]["arguments"]["selection"] == {}
    assert response["raw_action"]["arguments"]["selection"] == {"answer_ref": response["proof_record"]["handle"]}
    assert response["reference_prompt_version"] == REFERENCE_PROMPT_VERSION
    for prompt in observed:
        assert "For conflict, selection is {}." not in prompt
        assert '"selection":{"demo":"2.0"}' not in prompt
        assert "For BOTH resolved and conflict" in prompt
        assert prompt.count("\n5. ") == 1
        for line in CONTRACT.splitlines():
            if any(line.startswith(str(i) + ". ") for i in range(1, 5)):
                assert line in prompt  # Publication/ask/refresh/abstain semantics unchanged.


def test_reference_prompt_fails_closed_if_full_contract_changes(monkeypatch):
    import deltaweave.session as module
    monkeypatch.setattr(module, "prompt_for", lambda state: "New incompatible answer contract")
    with pytest.raises(EvidenceBindingError, match="explicit review"):
        reference_prompt_for({})
