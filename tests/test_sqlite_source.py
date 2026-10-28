"""Real temporary SQLite transactions, crash-window rollback and engine replay."""

import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
from threading import Barrier
import unittest

from deltaweave import Engine, Query, RevisionConflict
from deltaweave.domain import register_domain
from deltaweave.provenance import snapshot_fingerprint
from deltaweave.sqlite_source import (
    OutboxGap, SourceIntegrityError, SourceMismatch, SQLitePublisher, SQLiteSource,
)
from scripts.benchmark_domain import args, release, tables


class SQLiteSourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "source.sqlite"

    def tearDown(self):
        self.directory.cleanup()

    def test_new_path_only_and_open_never_creates_or_overwrites(self):
        with self.assertRaises(FileNotFoundError):
            SQLiteSource(self.path)
        self.assertFalse(self.path.exists())
        self.path.write_text("unrelated file")
        with self.assertRaises(FileExistsError):
            SQLiteSource.create(self.path, {"items": []})
        self.assertEqual(self.path.read_text(), "unrelated file")

    def test_reopen_persists_context_rows_and_empty_relations(self):
        source = SQLiteSource.create(self.path, {"items": [{"id": "a", "value": 1}], "empty": []}, {"scope": "finite"})
        original = source.snapshot()
        original.tables["items"][0]["value"] = "outside mutation"
        original.context["scope"] = "changed"
        receipt = source.commit(upserts={"items": [{"id": "a", "value": 2}]}, expected_revision=0, request_id="first")
        reopened = SQLiteSource(self.path)
        snapshot = reopened.snapshot()
        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(snapshot.source_id, receipt["source_id"])
        self.assertEqual(snapshot.tables, {"empty": [], "items": [{"id": "a", "value": 2}]})
        self.assertEqual(snapshot.context, {"scope": "finite"})
        self.assertEqual(len(reopened.changes(0)["events"]), 1)

    def test_compare_and_swap_and_duplicate_command_are_atomic(self):
        source = SQLiteSource.create(self.path, {"t": [{"id": "a", "value": 0}]})
        command = {"upserts": {"t": [{"id": "a", "value": 1}]}, "expected_revision": 0, "request_id": "retry-id"}
        first = source.commit(**command)
        duplicate = SQLiteSource(self.path).commit(**command)
        self.assertEqual(first["revision"], duplicate["revision"])
        self.assertFalse(first["duplicate"])
        self.assertTrue(duplicate["duplicate"])
        with self.assertRaises(SourceIntegrityError):
            source.commit(**{**command, "upserts": {"t": [{"id": "a", "value": 2}]}})
        with self.assertRaises(RevisionConflict):
            source.commit(upserts={"t": [{"id": "a", "value": 3}]}, expected_revision=0, request_id="another-command")
        self.assertEqual(source.snapshot().tables["t"], [{"id": "a", "value": 1}])
        self.assertEqual(source.snapshot().revision, 1)

    def test_two_concurrent_cas_writers_cannot_both_commit(self):
        source = SQLiteSource.create(self.path, {"t": []})
        barrier = Barrier(2)
        def update(value):
            client = SQLiteSource(self.path)
            barrier.wait(timeout=5)
            try:
                client.commit(upserts={"t": [{"id": "winner", "value": value}]}, expected_revision=0, request_id=value)
                return "committed"
            except RevisionConflict:
                return "conflict"
        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(update, ["alpha", "beta"]))
        self.assertEqual(sorted(outcomes), ["committed", "conflict"])
        self.assertEqual(source.snapshot().revision, 1)
        self.assertEqual(len(source.changes(0)["events"]), 1)

    def test_read_transaction_cannot_mix_metadata_and_later_rows(self):
        source = SQLiteSource.create(self.path, {"t": [{"id": "a", "value": 0}, {"id": "b", "value": 0}]})
        # WAL allows a writer to finish while this test's read snapshot is open.
        with sqlite3.connect(self.path) as connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0], "wal")
            # Materialize WAL/SHM before opening read-only URI connections.
            connection.execute("CREATE TABLE snapshot_test_anchor (value INTEGER)")
        class MidReadSource(SQLiteSource):
            def _metadata(self, connection):
                metadata = super()._metadata(connection)
                if getattr(self, "armed", False):
                    self.armed = False
                    source.commit(upserts={"t": [{"id": "a", "value": 1}, {"id": "b", "value": 1}]}, expected_revision=0, request_id="during-read")
                return metadata
        reader = MidReadSource(self.path)
        reader.armed = True
        old = reader.snapshot()
        self.assertEqual(old.revision, 0)
        self.assertEqual([row["value"] for row in old.tables["t"]], [0, 0])
        current = source.snapshot()
        self.assertEqual(current.revision, 1)
        self.assertEqual([row["value"] for row in current.tables["t"]], [1, 1])

    def test_sql_failure_after_row_writes_rolls_back_records_revision_and_outbox(self):
        source = SQLiteSource.create(self.path, {"t": [{"id": "a", "value": 0}]})
        # Inject a deterministic storage failure at the last write stage.
        with sqlite3.connect(self.path) as connection:
            connection.execute("CREATE TRIGGER reject_outbox BEFORE INSERT ON outbox BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with self.assertRaises(sqlite3.IntegrityError):
            source.commit(upserts={"t": [{"id": "a", "value": 9}]}, expected_revision=0, request_id="will-rollback")
        self.assertEqual(source.snapshot().revision, 0)
        self.assertEqual(source.snapshot().tables["t"], [{"id": "a", "value": 0}])
        self.assertEqual(source.changes(0)["events"], [])

    def test_store_validation_rejects_whole_invalid_batch(self):
        source = SQLiteSource.create(self.path, {"t": [{"id": "a", "value": 0}], "u": []})
        with self.assertRaises(ValueError):
            source.commit(upserts={"t": [{"id": "a", "value": 1}], "u": [{"id": "x"}, {"id": "x"}]}, expected_revision=0, request_id="invalid")
        self.assertEqual(source.snapshot().revision, 0)
        self.assertEqual(source.snapshot().tables["t"][0]["value"], 0)

    def test_publisher_keeps_engine_and_publishes_new_policy_then_new_environment(self):
        source = SQLiteSource.create(self.path, tables(release("a")))
        publisher = SQLitePublisher.load(source)
        store = publisher.store
        engine = Engine(store, mode="predicate")
        register_domain(engine)
        old_snapshot = store.snapshot
        request = args("a")
        self.assertEqual(engine.run("resolve", request)["status"], "resolved")
        policy = {"id": "abi", "project": "a", "kind": "require_env", "field": "ssl_abi", "operator": "version", "specifier": ">=3"}
        source.commit(upserts={"policies": [policy]}, expected_revision=0, request_id="policy")
        # No implicit background publication before the explicit poll.
        self.assertEqual(engine.run("resolve", request)["status"], "resolved")
        self.assertEqual(publisher.poll()["applied"], 1)
        self.assertIs(publisher.store, store)
        self.assertEqual(engine.run("resolve", request)["status"], "incomplete")
        source.commit(upserts={"environment": [{"id": "ssl", "environment": "benchmark", "field": "ssl_abi", "value": "3"}]}, expected_revision=1, request_id="environment")
        publisher.poll()
        self.assertEqual(engine.run("resolve", request)["status"], "resolved")
        self.assertEqual(old_snapshot.rows("policies"), [])
        self.assertEqual(publisher.poll()["applied"], 0)

    def test_order_duplicates_and_foreign_events(self):
        source = SQLiteSource.create(self.path, {"t": []})
        publisher = SQLitePublisher.load(source)
        source.commit(upserts={"t": [{"id": "x", "value": 1}]}, expected_revision=0, request_id="one")
        source.commit(upserts={"t": [{"id": "x", "value": 2}]}, expected_revision=1, request_id="two")
        first, second = source.changes(0)["events"]
        with self.assertRaises(OutboxGap):
            publisher.publish(second)
        self.assertEqual(publisher.store.snapshot.revision, 0)
        self.assertTrue(publisher.publish(first))
        self.assertFalse(publisher.publish(copy.deepcopy(first)))
        self.assertEqual(publisher.store.snapshot.revision, 1)
        self.assertTrue(publisher.publish(second))
        with self.assertRaises(SourceMismatch):
            publisher.publish({**second, "source_id": "foreign"})
        with self.assertRaises(SourceIntegrityError):
            publisher.publish({**first, "upserts": {"t": [{"id": "x", "value": "tampered"}]}})

    def test_outbox_gap_fails_before_publishing_and_store_write_requires_reload(self):
        source = SQLiteSource.create(self.path, {"t": []})
        publisher = SQLitePublisher.load(source)
        for index in range(2):
            source.commit(upserts={"t": [{"id": "x", "value": index}]}, expected_revision=index, request_id=str(index))
        with sqlite3.connect(self.path) as connection:
            connection.execute("DELETE FROM outbox WHERE seq=1")
        with self.assertRaises(OutboxGap):
            publisher.poll()
        self.assertEqual(publisher.store.snapshot.revision, 0)
        reloaded = SQLitePublisher.load(source)
        self.assertEqual(reloaded.source_revision, 2)
        self.assertEqual(reloaded.store.snapshot.revision, 0)
        reloaded.store.commit(upserts={"t": [{"id": "outside"}]})
        with self.assertRaises(SourceMismatch):
            reloaded.poll()

    def test_restart_tracks_source_revision_separately_from_local_store_revision(self):
        source = SQLiteSource.create(self.path, {"t": []})
        source.commit(upserts={"t": [{"id": "a", "value": 1}]}, expected_revision=0, request_id="one")
        publisher = SQLitePublisher.load(SQLiteSource(self.path))
        self.assertEqual((publisher.source_revision, publisher.store.snapshot.revision), (1, 0))
        initial_hash = snapshot_fingerprint(publisher.store.snapshot)
        self.assertEqual(publisher.binding()["published_snapshot_sha256"], initial_hash)
        source.commit(upserts={"t": [{"id": "a", "value": 2}]}, expected_revision=1, request_id="two")
        self.assertEqual(publisher.poll()["source_revision"], 2)
        self.assertEqual(publisher.store.snapshot.revision, 1)
        self.assertEqual(publisher.binding(0)["source_revision"], 1)
        self.assertEqual(publisher.binding(1)["source_revision"], 2)
        self.assertEqual(publisher.binding(0)["published_snapshot_sha256"], initial_hash)
        self.assertEqual(publisher.binding(1)["published_snapshot_sha256"], snapshot_fingerprint(publisher.store.snapshot))
        self.assertNotEqual(publisher.binding(0)["snapshot_id"], publisher.binding(1)["snapshot_id"])
        with self.assertRaises(SourceMismatch):
            publisher.binding(True)

    def test_ids_and_sql_looking_values_are_bound_data(self):
        source = SQLiteSource.create(self.path, {"t; DROP TABLE metadata;--": [{"id": 1, "value": "integer"}, {"id": "1", "value": "string"}]})
        source.commit(upserts={"t; DROP TABLE metadata;--": [{"id": "1", "value": "'); DELETE FROM records; --"}]}, expected_revision=0, request_id="sql-looking-value")
        rows = source.snapshot().tables["t; DROP TABLE metadata;--"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(next(row["value"] for row in rows if type(row["id"]) is int), "integer")


if __name__ == "__main__":
    unittest.main()
