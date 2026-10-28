import json
from deltaweave.learning_data import build_group, decide, write_datasets


def test_validity_uses_certificate_not_generation_time():
    rows = build_group("test", 0, 42)
    by_kind = {r["kind"]: r for r in rows}
    assert by_kind["certified_reuse"]["target"]["action"] == "answer"
    assert by_kind["stale"]["target"]["action"] == "refresh"
    # Same stale answer text, different validity yields a different action.
    state = by_kind["stale"]["state"]
    state["observation"]["validated_at"] = state["target_revision"]
    assert decide(state)["action"] == "answer"


def test_precedence_and_no_implied_global_unsat():
    by_kind = {r["kind"]: r for r in build_group("future", 1, 88)}
    assert by_kind["not_published"]["target"]["action"] == "wait"
    assert by_kind["missing_python"]["target"]["action"] == "ask"
    assert by_kind["incomplete"]["target"]["action"] == "abstain"


def test_splits_are_disjoint_and_reproducible(tmp_path):
    manifest = write_datasets(tmp_path)
    assert write_datasets(tmp_path) == manifest
    groups = []
    for name in ("train_behavior.jsonl", "dev.jsonl", "test.jsonl", "future.jsonl"):
        rows = [json.loads(line) for line in (tmp_path / name).read_text().splitlines()]
        groups.append({r["group"] for r in rows})
        assert all(r["target"] == decide(r["state"]) for r in rows)
    assert sum(map(len, groups)) == len(set.union(*groups))
