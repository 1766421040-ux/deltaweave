"""Managed local SQLite source with transactional JSON rows and an ordered outbox.

source = SQLiteSource.create(new_path, tables)  # refuses every existing path
publisher = SQLitePublisher.load(source)
engine = Engine(publisher.store)               # retained across poll() calls
source.commit(upserts={...}, expected_revision=0, request_id="stable-command-id")
publisher.poll()                              # publishes complete transactions

Opening SQLiteSource(path) never creates a file. Only databases initialized by
this module are supported; SQL identifiers/statements are fixed, all user values
are bound parameters. The same Store validator defines accepted row mutations.
The prototype retains every outbox entry and all in-memory snapshots. It is not
a generic external-database CDC connector, and it makes no bounded-storage claim.

Source revisions are durable. Store revisions start at zero when loading a
source snapshot; the publisher explicitly tracks both. The publisher exclusively
owns Store writes, while engines can read its immutable snapshots. On a gap or
an out-of-band Store write, fail and explicitly reload a publisher AND its engine;
never silently replace an existing engine's Store. Each source transaction is
published atomically, but a poll containing several transactions is not one
combined transaction. No background thread or automatic wall-clock polling runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any
from uuid import uuid4

from .provenance import snapshot_fingerprint
from .store import RevisionConflict, Store, canonical, id_key


class SourceIntegrityError(ValueError):
    pass


class OutboxGap(SourceIntegrityError):
    pass


class SourceMismatch(SourceIntegrityError):
    pass


@dataclass(frozen=True)
class SourceSnapshot:
    source_id: str
    revision: int
    tables: dict[str, list[dict]]
    context: Any


def _request(upserts, deletes, expected_revision):
    return {"upserts": {} if upserts is None else upserts,
            "deletes": {} if deletes is None else deletes,
            "expected_revision": expected_revision}


def _hash(value):
    return sha256(canonical(value).encode("utf-8")).hexdigest()


class SQLiteSource:
    FORMAT = "deltaweave-sqlite-v1"

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve(strict=True)
        self._source_id = None
        self._source_id = self.snapshot().source_id

    @classmethod
    def create(cls, path: str | Path, tables: dict, context: Any = None) -> SQLiteSource:
        validated = Store(tables, context, indexed=False).snapshot
        target = Path(path).absolute()
        # O_EXCL also rejects existing symlinks; nothing is overwritten.
        fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        provisional = cls.__new__(cls)
        provisional.path, provisional._source_id = target.resolve(), None
        with provisional._transaction(write=True) as connection:
            connection.execute("CREATE TABLE metadata (singleton INTEGER PRIMARY KEY CHECK(singleton=1), format TEXT NOT NULL, source_id TEXT NOT NULL, revision INTEGER NOT NULL, context TEXT NOT NULL)")
            connection.execute("CREATE TABLE relations (name TEXT PRIMARY KEY)")
            connection.execute("CREATE TABLE records (relation_name TEXT NOT NULL REFERENCES relations(name), row_key TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(relation_name,row_key))")
            connection.execute("CREATE TABLE outbox (seq INTEGER PRIMARY KEY, request_id TEXT NOT NULL UNIQUE, request_sha256 TEXT NOT NULL, payload TEXT NOT NULL)")
            connection.execute("INSERT INTO metadata VALUES (1,?,?,0,?)", (cls.FORMAT, str(uuid4()), canonical(validated.context)))
            for table, rows in validated.tables.items():
                connection.execute("INSERT INTO relations VALUES (?)", (table,))
                connection.executemany("INSERT INTO records VALUES (?,?,?)", [(table, id_key(row["id"]), canonical(row)) for row in rows])
        return cls(target)

    @contextmanager
    def _transaction(self, *, write=False):
        # mode=rw/ro prevents sqlite3.connect from silently creating a new file.
        mode = "rw" if write else "ro"
        connection = sqlite3.connect(self.path.as_uri() + f"?mode={mode}", uri=True, isolation_level=None, timeout=5)
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            if write:
                connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _metadata(self, connection):
        try:
            rows = connection.execute("SELECT format,source_id,revision,context FROM metadata WHERE singleton=1").fetchall()
        except sqlite3.DatabaseError as exc:
            raise SourceIntegrityError("Not a managed DeltaWeave SQLite source") from exc
        if len(rows) != 1 or rows[0][0] != self.FORMAT:
            raise SourceIntegrityError("Unsupported source format")
        _, identity, revision, context = rows[0]
        if self._source_id is not None and identity != self._source_id:
            raise SourceMismatch("Source identity changed; explicit reload required")
        if type(revision) is not int or revision < 0:
            raise SourceIntegrityError("Invalid source revision")
        return identity, revision, json.loads(context)

    def _snapshot(self, connection):
        identity, revision, context = self._metadata(connection)
        tables = {row[0]: [] for row in connection.execute("SELECT name FROM relations ORDER BY name")}
        for table, key, payload in connection.execute("SELECT relation_name,row_key,payload FROM records ORDER BY relation_name,row_key"):
            row = json.loads(payload)
            if table not in tables or id_key(row["id"]) != key:
                raise SourceIntegrityError("Record identity does not match stored key")
            tables[table].append(row)
        return SourceSnapshot(identity, revision, tables, context)

    def snapshot(self) -> SourceSnapshot:
        """Detached rows/context from one SQLite read transaction."""
        with self._transaction() as connection:
            return self._snapshot(connection)

    def commit(self, *, upserts=None, deletes=None, expected_revision: int, request_id: str) -> dict:
        if type(expected_revision) is not int or expected_revision < 0:
            raise TypeError("expected_revision must be a nonnegative integer")
        if not isinstance(request_id, str) or not request_id:
            raise TypeError("request_id must be a nonempty stable string")
        encoded = canonical(_request(upserts, deletes, expected_revision))
        # Freeze the same command bytes for validation, SQL rows and the outbox;
        # callers retaining mutable dictionaries cannot change one stage alone.
        request = json.loads(encoded)
        fingerprint = sha256(encoded.encode("utf-8")).hexdigest()
        with self._transaction(write=True) as connection:
            identity, revision, _ = self._metadata(connection)
            existing = connection.execute("SELECT seq,request_sha256 FROM outbox WHERE request_id=?", (request_id,)).fetchone()
            if existing is not None:
                if existing[1] != fingerprint:
                    raise SourceIntegrityError("request_id reused with different command")
                return {"source_id": identity, "revision": existing[0], "request_id": request_id, "duplicate": True}
            if revision != expected_revision:
                raise RevisionConflict(f"Expected source revision {expected_revision}; current {revision}")
            current = self._snapshot(connection)
            # Validation and transition are independently materialized before any
            # SQL row mutation. This is intentionally O(database size), not fast CDC.
            temporary = Store(current.tables, current.context, indexed=False)
            changed = temporary.commit(upserts=request["upserts"], deletes=request["deletes"], expected_revision=0)
            for event in changed.events:
                if event.after is None:
                    connection.execute("DELETE FROM records WHERE relation_name=? AND row_key=?", (event.table, id_key(event.id)))
                else:
                    connection.execute("INSERT INTO records VALUES (?,?,?) ON CONFLICT(relation_name,row_key) DO UPDATE SET payload=excluded.payload", (event.table, id_key(event.id), canonical(event.after)))
            next_revision = revision + 1
            connection.execute("INSERT INTO outbox VALUES (?,?,?,?)", (next_revision, request_id, fingerprint, encoded))
            connection.execute("UPDATE metadata SET revision=? WHERE singleton=1", (next_revision,))
            return {"source_id": identity, "revision": next_revision, "request_id": request_id, "duplicate": False}

    def changes(self, after: int) -> dict:
        if type(after) is not int or after < 0:
            raise TypeError("after must be a nonnegative integer")
        with self._transaction() as connection:
            identity, head, _ = self._metadata(connection)
            if after > head:
                raise OutboxGap("Consumer revision is ahead of the source")
            rows = connection.execute("SELECT seq,request_id,request_sha256,payload FROM outbox WHERE seq>? AND seq<=? ORDER BY seq", (after, head)).fetchall()
            if len(rows) != head - after or any(row[0] != after + index for index, row in enumerate(rows, 1)):
                raise OutboxGap("Outbox gap; explicit snapshot reload required")
            events = []
            for seq, request_id, fingerprint, payload in rows:
                command = json.loads(payload)
                if _hash(command) != fingerprint or command.get("expected_revision") != seq - 1:
                    raise SourceIntegrityError("Outbox command hash or revision invalid")
                events.append({"source_id": identity, "seq": seq, "request_id": request_id,
                               "request_sha256": fingerprint, **command})
            return {"source_id": identity, "revision": head, "events": events}


class SQLitePublisher:
    def __init__(self, source: SQLiteSource, snapshot: SourceSnapshot, *, indexed: bool = True):
        self.source, self.source_id = source, snapshot.source_id
        self.store = Store(snapshot.tables, snapshot.context, indexed=indexed)
        self.source_revision = snapshot.revision
        self._store_revision = self.store.snapshot.revision
        self._source_revisions = {self._store_revision: self.source_revision}
        self._published_snapshots = {self._store_revision: self.store.snapshot}
        self._lock = RLock()

    @classmethod
    def load(cls, source: SQLiteSource, *, indexed: bool = True) -> SQLitePublisher:
        return cls(source, source.snapshot(), indexed=indexed)

    def _check_store(self):
        if self.store.snapshot.revision != self._store_revision:
            raise SourceMismatch("Store changed outside this publisher; explicit reload required")

    def publish(self, event: dict) -> bool:
        """Apply one complete source command; a repeated sequence is a no-op."""
        with self._lock:
            event = json.loads(canonical(event))
            self._check_store()
            if event.get("source_id") != self.source_id:
                raise SourceMismatch("Event belongs to another source")
            seq = event.get("seq")
            if type(seq) is not int or seq < 1:
                raise SourceIntegrityError("Invalid event sequence")
            request = _request(event.get("upserts"), event.get("deletes"), event.get("expected_revision"))
            if _hash(request) != event.get("request_sha256") or request["expected_revision"] != seq - 1:
                raise SourceIntegrityError("Invalid event command fingerprint")
            if seq <= self.source_revision:
                return False
            if seq != self.source_revision + 1:
                raise OutboxGap("Out-of-order event; explicit retry or reload required")
            snapshot = self.store.commit(upserts=request["upserts"], deletes=request["deletes"], expected_revision=self._store_revision)
            self._store_revision = snapshot.revision
            self.source_revision = seq
            self._source_revisions[snapshot.revision] = seq
            self._published_snapshots[snapshot.revision] = snapshot
            return True

    def binding(self, store_revision: int | None = None) -> dict:
        """Identify the durable source version behind a pinned local snapshot.

        Retaining this mapping avoids equating Store revision zero after restart
        with source revision zero, and also handles a concurrent later poll.
        """
        with self._lock:
            self._check_store()
            local = self._store_revision if store_revision is None else store_revision
            if type(local) is not int or local not in self._source_revisions:
                raise SourceMismatch("No source binding for this local snapshot")
            revision = self._source_revisions[local]
            return {"source_id": self.source_id, "source_revision": revision,
                    "store_revision": local, "snapshot_id": f"{self.source_id}@{revision}",
                    "published_snapshot_sha256": snapshot_fingerprint(self._published_snapshots[local])}

    def poll(self) -> dict:
        with self._lock:
            self._check_store()
            batch = self.source.changes(self.source_revision)
            if batch["source_id"] != self.source_id:
                raise SourceMismatch("Source changed")
            applied = sum(self.publish(event) for event in batch["events"])
            return {"applied": applied, **self.binding()}
