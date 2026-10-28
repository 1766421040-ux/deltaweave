from deltaweave.engine import Engine, Query
from deltaweave.store import Store
from deltaweave.learning_data import decide, build_group
from deltaweave.session import KnowledgeSession, normalize_state


def resolve(ctx, args):
    rows = ctx.query(Query("items", where={"project": "p"}))
    return {"status": "resolved" if rows else "incomplete", "selection": {"p": rows[0]["version"]} if rows else {},
            "citations": [r["id"] for r in rows], "issues": []}


def setup(policy=None):
    store = Store({"environment": [{"id": "py", "environment": "local", "field": "python_full_version", "value": "3.11.9"}],
                   "items": [{"id": "p", "project": "p", "version": "1.0"}]})
    engine = Engine(store); engine.register("resolve", resolve)
    return store, KnowledgeSession(engine, policy=policy,
                                   policy_identity="test-model-v1-seed42" if policy else None)


QUERY = {"requirements": ["p>=1"], "environment": "local", "as_of": "2026-10-28"}


def test_normalization_preserves_contract_for_all_fixture_states():
    for split in ("train", "dev", "test", "future"):
        for row in build_group(split, 1, 21):
            assert decide(row["state"]) == decide(normalize_state(row["state"]))


def test_rule_agent_refreshes_then_retracts_and_never_mixes_snapshots():
    store, session = setup()
    first = session.request(QUERY)
    assert first["status"] == "completed"
    assert [s["action"]["action"] for s in first["steps"]] == ["refresh", "answer"]
    store.commit(deletes={"items": ["p"]})
    second = session.request(QUERY)
    assert second["answer"]["action"] == "abstain"
    assert [s["action"]["action"] for s in second["steps"]] == ["refresh", "abstain"]


def test_invalid_model_output_is_blocked_not_repaired():
    store, session = setup(lambda prompt: ({"action": "answer", "arguments": {}}, {"completed": True}))
    result = session.request(QUERY)
    assert result["status"] == "blocked_model_contract"
    assert result["answer"] is None
    assert result["raw_model_success"] is False
import json
import pytest


def test_same_source_revision_solver_version_change_invalidates_observation():
    store, session = setup()
    assert session.request(QUERY)["answer"]["arguments"]["status"] == "resolved"
    def new_resolve(ctx, args):
        return {"status": "conflict", "selection": {}, "citations": ["p"], "issues": []}
    session.engine.register("resolve", new_resolve, version="2")
    result = session.request(QUERY)
    assert result["snapshot_revision"] == 0
    assert [step["action"]["action"] for step in result["steps"]] == ["refresh", "answer"]
    assert result["answer"]["arguments"]["status"] == "conflict"
    assert result["runtime_binding"]["registry_versions"]["resolve"] == "2"


def test_completed_wrong_prediction_is_cached_and_still_blocked():
    calls = []
    def wrong(prompt):
        calls.append(prompt)
        return {"action": "answer", "arguments": {}}, {"completed": True, "raw_text": "wrong", "seconds": 1.5}
    store, session = setup(wrong)
    first, second = session.request(QUERY), session.request(QUERY)
    assert first["status"] == second["status"] == "blocked_model_contract"
    assert len(calls) == 1
    assert first["raw_model_success"] is False
    assert second["raw_model_success"] is None
    assert second["fresh_model_calls"] == 0 and second["model_cache_hits"] == 1
    assert second["closed_loop_contract_success"] is False
    assert second["steps"][0]["action"] == first["steps"][0]["action"]
    receipt = second["steps"][0]["receipt"]
    assert receipt["origin_receipt"]["raw_text"] == "wrong"
    assert "seconds" not in receipt  # Original generation latency is not a fresh call.


def test_cache_only_success_is_not_reported_as_new_raw_generation():
    calls = []
    def correct(prompt):
        calls.append(prompt)
        return decide(json.loads(prompt.split("\nSTATE=", 1)[1])), {"completed": True}
    store, session = setup(correct)
    first = session.request(QUERY)
    second = session.request(QUERY)
    assert first["fresh_model_calls"] == 2
    assert first["raw_model_success"] is True
    assert second["fresh_model_calls"] == 0 and second["model_cache_hits"] == 1
    assert second["raw_model_success"] is None
    assert second["closed_loop_contract_success"] is True
    assert len(calls) == 2


def test_policy_identity_required_and_readonly_and_positive_rejected():
    store, session = setup()
    policy = lambda prompt: (None, {"completed": True})
    with pytest.raises(ValueError, match="identity"):
        KnowledgeSession(session.engine, policy=policy)
    with pytest.raises(AttributeError):
        session.policy = policy
    with pytest.raises(AttributeError):
        session.policy_identity = "new"
    with pytest.raises(ValueError, match="Positive"):
        KnowledgeSession(Engine(store, mode="positive"))


def test_mid_request_registry_change_blocks_completion():
    store, session = setup()
    def changing(prompt):
        session.engine.register("resolve", lambda ctx, args: {
            "status": "conflict", "selection": {}, "citations": [], "issues": []}, version="2")
        return decide(json.loads(prompt.split("\nSTATE=", 1)[1])), {"completed": True}
    session = KnowledgeSession(session.engine, policy=changing, policy_identity="changing-test-v1")
    result = session.request(QUERY)
    assert result["status"] == "blocked_runtime_binding"
    assert result["answer"] is None and result["raw_model_success"] is False


def test_session_pins_source_snapshot_across_model_callback_commit():
    store, rule_session = setup()
    committed = []
    def updating(prompt):
        if not committed:
            store.commit(upserts={"items": [{"id": "p", "project": "p", "version": "2.0"}]})
            committed.append(True)
        return decide(json.loads(prompt.split("\nSTATE=", 1)[1])), {"completed": True}
    session = KnowledgeSession(rule_session.engine, policy=updating, policy_identity="updating-test-v1")
    result = session.request(QUERY)
    assert result["snapshot_revision"] == 0 and store.snapshot.revision == 1
    assert result["answer"]["arguments"]["selection"] == {"p": "1.0"}
    assert session.request(QUERY)["answer"]["arguments"]["selection"] == {"p": "2.0"}


def test_session_keeps_its_receipt_when_another_request_finishes_after_tool_run(monkeypatch):
    from threading import Thread
    store, session = setup()
    engine = session.engine
    atomic_run = engine.run_with_receipt
    other = {}

    def interleaved(name, args, snapshot=None):
        pair = atomic_run(name, args, snapshot=snapshot)
        def compete():
            try:
                store.commit(upserts={"items": [{"id": "p", "project": "p", "version": "2.0"}]})
                other["value"] = engine.run("controller_python", {"environment": "local"})
            except BaseException as exc:
                other["error"] = exc
        thread = Thread(target=compete)
        thread.start(); thread.join(5)
        assert not thread.is_alive()
        assert "error" not in other
        return pair

    monkeypatch.setattr(engine, "run_with_receipt", interleaved)
    result = session.request(QUERY)
    assert result["status"] == "completed"
    assert result["answer"]["arguments"]["selection"] == {"p": "1.0"}
    receipt = result["tool_receipts"][0]
    assert receipt["name"] == "resolve"
    assert receipt["validated_at"] == result["snapshot_revision"] == 0
    assert result["steps"][-1]["state"]["observation"]["computed_at"] == 0
    assert engine.last_receipt["name"] == "controller_python"
    assert engine.last_receipt["validated_at"] == 1
