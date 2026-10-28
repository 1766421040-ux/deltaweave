"""Small immutable JSON snapshot store; no external DB or model dependency.

Every commit is atomic and retains earlier snapshots and before/after events.
Public values are defensive copies. History is deliberately retained in memory;
this reference implementation does not claim bounded storage or durable CDC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from threading import RLock
from time import perf_counter
from types import MappingProxyType
from typing import Any, Mapping


def canonical(value: Any) -> str:
    """Strict finite JSON with type-sensitive, stable encoding."""
    def check(item: Any) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise TypeError("Expected finite JSON (string object keys, no tuples/NaN)")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def id_key(value: Any) -> str:
    if type(value) not in (str, int):
        raise TypeError("Row id must be a string or integer, excluding bool")
    return canonical(value)


class RevisionConflict(ValueError):
    """The caller's expected revision is no longer current."""


@dataclass(frozen=True)
class Change:
    table: str
    _id: str = field(repr=False)
    _before: str | None = field(repr=False)
    _after: str | None = field(repr=False)

    @property
    def id(self) -> str | int:
        return json.loads(self._id)

    @property
    def before(self) -> dict | None:
        return None if self._before is None else json.loads(self._before)

    @property
    def after(self) -> dict | None:
        return None if self._after is None else json.loads(self._after)

    old = before
    new = after


@dataclass(frozen=True)
class Snapshot:
    revision: int
    _tables: Mapping[str, Mapping[str, str]] = field(repr=False, compare=False)
    _context: str = field(repr=False)
    events: tuple[Change, ...] = ()
    _owner: object = field(default=None, repr=False, compare=False)

    def rows(self, table: str) -> list[dict]:
        return [json.loads(row) for row in self._tables[table].values()]

    @property
    def tables(self) -> dict[str, list[dict]]:
        return {table: self.rows(table) for table in self._tables}

    @property
    def context(self) -> Any:
        return json.loads(self._context)


class Store:
    def __init__(self, tables: dict[str, list[dict]], context: Any = None):
        if type(tables) is not dict or not all(type(name) is str for name in tables):
            raise TypeError("tables must be a string-keyed dict")
        self._lock = RLock()
        self._owner = object()
        encoded: dict[str, dict[str, str]] = {}
        for table, rows in tables.items():
            encoded[table] = self._encode_rows(rows)
        initial = self._make_snapshot(0, encoded, canonical({} if context is None else context), ())
        self._history = [initial]
        self.last_commit_stats: dict[str, int | float] = {}

    @staticmethod
    def _encode_rows(rows: list[dict]) -> dict[str, str]:
        if type(rows) is not list:
            raise TypeError("table rows/upserts must be lists")
        result = {}
        for row in rows:
            if type(row) is not dict or "id" not in row:
                raise ValueError("Each row must be an object with id")
            key = id_key(row["id"])
            if key in result:
                raise ValueError("Duplicate row id in batch")
            result[key] = canonical(row)
        return result

    def _make_snapshot(self, revision, tables, context, events):
        frozen = MappingProxyType({name: MappingProxyType(dict(rows))
                                   for name, rows in tables.items()})
        return Snapshot(revision, frozen, context, tuple(events), self._owner)

    @property
    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._history[-1]

    def owns(self, snapshot: Snapshot) -> bool:
        with self._lock:
            return (isinstance(snapshot, Snapshot) and snapshot._owner is self._owner
                    and 0 <= snapshot.revision < len(self._history)
                    and self._history[snapshot.revision] is snapshot)

    def events_between(self, before: int, after: int) -> tuple[Change, ...]:
        with self._lock:
            if not (0 <= before <= after < len(self._history)):
                raise ValueError("Invalid revision interval")
            return tuple(event for snapshot in self._history[before + 1:after + 1]
                         for event in snapshot.events)

    def commit(self, upserts: dict[str, list[dict]] | None = None,
               deletes: dict[str, list[str | int]] | None = None,
               expected_revision: int | None = None) -> Snapshot:
        started = perf_counter()
        with self._lock:
            current = self._history[-1]
            if expected_revision is not None:
                if type(expected_revision) is not int:
                    raise TypeError("expected_revision must be an integer")
                if expected_revision != current.revision:
                    raise RevisionConflict(f"Expected {expected_revision}; current {current.revision}")
            upserts = {} if upserts is None else upserts
            deletes = {} if deletes is None else deletes
            if type(upserts) is not dict or type(deletes) is not dict:
                raise TypeError("upserts/deletes must be dictionaries")
            encoded = {}
            removed = {}
            for table, rows in upserts.items():
                if table not in current._tables:
                    raise KeyError(table)
                encoded[table] = self._encode_rows(rows)
            for table, ids in deletes.items():
                if table not in current._tables:
                    raise KeyError(table)
                if type(ids) is not list:
                    raise TypeError("delete ids must be a list")
                keys = [id_key(item) for item in ids]
                if len(keys) != len(set(keys)):
                    raise ValueError("Duplicate deletion id")
                removed[table] = keys
                if set(keys) & set(encoded.get(table, {})):
                    raise ValueError("Cannot upsert and delete the same id in one commit")
            # Validate the complete batch before constructing/publishing any change.
            tables = {name: dict(rows) for name, rows in current._tables.items()}
            events = []
            for table in sorted(set(encoded) | set(removed)):
                for key in sorted(removed.get(table, [])):
                    old = tables[table].pop(key, None)
                    if old is not None:
                        events.append(Change(table, key, old, None))
                for key, new in sorted(encoded.get(table, {}).items()):
                    old = tables[table].get(key)
                    if old != new:
                        tables[table][key] = new
                        events.append(Change(table, key, old, new))
            snapshot = self._make_snapshot(current.revision + 1, tables,
                                           current._context, events)
            self._history.append(snapshot)
            self.last_commit_stats = {"events": len(events),
                                      "commit_seconds": perf_counter() - started}
            return snapshot
