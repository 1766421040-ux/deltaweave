import json
import pytest
from deltaweave.engine import Engine, Query
from deltaweave.store import Store
from deltaweave.lineage import TrainingLedger, route_failure
from deltaweave.provenance import snapshot_fingerprint
from deltaweave.session import KnowledgeSession


def resolver(ctx, args):
    rows = ctx.query(Query("facts", where={"project": args["project"]}))
    return {"status": "resolved" if rows else "incomplete",
            "selection": {args["project"]: rows[0]["value"]} if rows else {},
            "citations": [r["id"] for r in rows]}


def test_frozen_labels_remain_immutable_while_new_manifest_retracts(tmp_path):
    store = Store({"facts": [{"id": "f", "project": "p", "value": "1.0"}]})
    engine = Engine(store); engine.register("resolve", resolver)
    ledger = TrainingLedger(engine); ledger.add("s", {"project": "p"})
    first_path, first = ledger.freeze(tmp_path)
    first_bytes = first_path.read_bytes()
    assert not ledger.needs_revalidation
    store.commit(deletes={"facts": ["f"]})
    assert ledger.needs_revalidation
    second_path, second = ledger.freeze(tmp_path)
    assert second["samples"][0]["target"]["action"] == "abstain"
    assert second["changed_sample_ids"] == ["s"]
    assert first_bytes == first_path.read_bytes()
    assert first["manifest_sha256"] != second["manifest_sha256"]
    assert json.loads(second_path.read_text())["samples"][0]["snapshot_revision"] == store.snapshot.revision


def test_unrelated_change_advances_validation_not_label(tmp_path):
    store = Store({"facts": [{"id": "f", "project": "p", "value": "1.0"}]})
    engine = Engine(store); engine.register("resolve", resolver)
    ledger = TrainingLedger(engine); ledger.add("s", {"project": "p"})
    ledger.freeze(tmp_path)
    store.commit(upserts={"facts": [{"id": "other", "project": "q", "value": "9"}]})
    _, manifest = ledger.freeze(tmp_path)
    assert manifest["changed_sample_ids"] == []
    assert not ledger.needs_revalidation


def test_diagnosis_does_not_train_to_cover_missing_data():
    assert route_failure(source_complete=False, snapshot_consistent=True,
                         current_evidence=False, model_contract_passed=False) == "collect_missing_evidence"
    assert route_failure(source_complete=True, snapshot_consistent=True,
                         current_evidence=True, model_contract_passed=False) == "collect_behavior_errors_for_sft_review"


def test_solver_version_invalidates_ledger_without_database_commit(tmp_path):
    store = Store({"facts": [{"id": "f", "project": "p", "value": "1"}]})
    engine = Engine(store); engine.register("resolve", resolver)
    ledger = TrainingLedger(engine); ledger.add("s", {"project": "p"})
    _, first = ledger.freeze(tmp_path)
    engine.register("resolve", lambda ctx, args: {
        "status": "conflict", "selection": {}, "citations": ["f"]}, version="2")
    assert ledger.needs_revalidation
    _, second = ledger.freeze(tmp_path)
    assert not ledger.needs_revalidation
    assert second["snapshot_revision"] == first["snapshot_revision"] == 0
    assert second["manifest_sha256"] != first["manifest_sha256"]
    assert second["runtime_binding"]["registry_versions"]["resolve"] == "2"
    assert second["samples"][0]["target"]["arguments"]["status"] == "conflict"


def test_shared_snapshot_fingerprint_is_row_order_independent(tmp_path):
    facts = [{"id": "z", "project": "p", "value": "1"}, {"id": "a", "project": "p", "value": "2"}]
    env = [{"id": "py", "environment": "x", "field": "python_full_version", "value": "3.11"}]
    store = Store({"facts": facts, "environment": env})
    other = Store({"environment": env, "facts": list(reversed(facts))})
    assert snapshot_fingerprint(store.snapshot) == snapshot_fingerprint(other.snapshot)
    engine = Engine(store)
    def complete(ctx, args):
        return {**resolver(ctx, args), "issues": []}
    engine.register("resolve", complete)
    session = KnowledgeSession(engine)
    ledger = TrainingLedger(engine)
    query = {"project": "p", "requirements": ["p"], "environment": "x", "as_of": "2026-10-28"}
    ledger.add("s", query)
    _, manifest = ledger.freeze(tmp_path)
    response = session.request(query)
    assert response["snapshot_sha256"] == manifest["snapshot_sha256"]
    assert response["runtime_binding"]["registry_fingerprint"] == manifest["runtime_binding"]["registry_fingerprint"]


def test_failed_manifest_write_is_atomic_and_does_not_advance_ledger(tmp_path, monkeypatch):
    import deltaweave.provenance as provenance
    store = Store({"facts": [{"id": "f", "project": "p", "value": "1"}]})
    engine = Engine(store); engine.register("resolve", resolver)
    ledger = TrainingLedger(engine); ledger.add("s", {"project": "p"})
    old_path, _ = ledger.freeze(tmp_path)
    old_bytes = old_path.read_bytes()
    store.commit(upserts={"facts": [{"id": "f", "project": "p", "value": "2"}]})
    def partial_failure(value, stream, **kwargs):
        stream.write('{"partial":')
        raise OSError("simulated write failure")
    monkeypatch.setattr(provenance.json, "dump", partial_failure)
    with pytest.raises(OSError, match="simulated"):
        ledger.freeze(tmp_path)
    assert list(tmp_path.iterdir()) == [old_path]
    assert old_path.read_bytes() == old_bytes
    assert ledger.needs_revalidation


def test_heuristic_engine_cannot_label_training_data():
    with pytest.raises(ValueError, match="Positive"):
        TrainingLedger(Engine(Store({"facts": []}), mode="positive"))


def test_ledger_binds_sample_to_its_atomic_receipt_under_interleaving(tmp_path, monkeypatch):
    from threading import Thread
    store = Store({"facts": [{"id": "f", "project": "p", "value": "1.0"}]})
    engine = Engine(store); engine.register("resolve", resolver)
    ledger = TrainingLedger(engine); ledger.add("s", {"project": "p"})
    atomic_run = engine.run_with_receipt
    other = {}

    def interleaved(name, args, snapshot=None):
        pair = atomic_run(name, args, snapshot=snapshot)
        def compete():
            try:
                store.commit(upserts={"facts": [{"id": "f", "project": "p", "value": "2.0"}]})
                other["value"] = engine.run("resolve", {"project": "p"})
            except BaseException as exc:
                other["error"] = exc
        thread = Thread(target=compete)
        thread.start(); thread.join(5)
        assert not thread.is_alive()
        assert "error" not in other
        return pair

    monkeypatch.setattr(engine, "run_with_receipt", interleaved)
    _, manifest = ledger.freeze(tmp_path)
    assert manifest["samples"][0]["target"]["arguments"]["selection"] == {"p": "1.0"}
    receipt = manifest["receipts"]["s"]
    assert receipt["name"] == "training_sample"
    assert receipt["validated_at"] == manifest["snapshot_revision"] == 0
    assert other["value"]["selection"] == {"p": "2.0"}
    assert ledger.needs_revalidation
