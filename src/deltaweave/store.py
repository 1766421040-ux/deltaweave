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


def _index_token(value: Any) -> str | None:
    """None return means non-scalar; JSON null has the string token 'null'."""
    if value is not None and type(value) not in (str, bool, int, float):
        return None
    # Equality distinguishes JSON numeric types but treats +/-0.0 equally.
    return canonical(0.0 if type(value) is float and value == 0 else value)


def _terms(row: dict) -> set[tuple[str, str]]:
    return {(name, token) for name, value in row.items()
            if (token := _index_token(value)) is not None}


def _build_index(rows: Mapping[str, str]):
    fields: dict[str, dict[str, set[str]]] = {}
    postings = 0
    for key, encoded in rows.items():
        for name, token in _terms(json.loads(encoded)):
            fields.setdefault(name, {}).setdefault(token, set()).add(key)
            postings += 1
    return MappingProxyType({name: MappingProxyType({token: frozenset(ids)
                             for token, ids in values.items()})
                             for name, values in fields.items()}), postings


def _update_index(index, events):
    """Copy only touched field maps/postings; immutable untouched maps are shared."""
    fields = dict(index)
    touched: dict[str, dict] = {}
    changed_postings: dict[tuple[str, str], set[str]] = {}
    posting_delta = 0
    for event in events:
        before = set() if event.before is None else _terms(event.before)
        after = set() if event.after is None else _terms(event.after)
        for terms, add in ((before - after, False), (after - before, True)):
            for name, token in terms:
                if name not in touched:
                    touched[name] = dict(fields.get(name, {}))
                pair = name, token
                if pair not in changed_postings:
                    changed_postings[pair] = set(touched[name].get(token, ()))
                if add:
                    changed_postings[pair].add(event._id)
                    posting_delta += 1
                else:
                    changed_postings[pair].discard(event._id)
                    posting_delta -= 1
    for (name, token), ids in changed_postings.items():
        if ids:
            touched[name][token] = frozenset(ids)
        else:
            touched[name].pop(token, None)
    for name, values in touched.items():
        if values:
            fields[name] = MappingProxyType(values)
        else:
            fields.pop(name, None)
    return MappingProxyType(fields), posting_delta


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
    _indexes: Mapping | None = field(default=None, repr=False, compare=False)
    index_postings: int = 0
    row_count: int = 0

    def rows(self, table: str) -> list[dict]:
        return [json.loads(row) for row in self._tables[table].values()]

    @property
    def indexed(self) -> bool:
        return self._indexes is not None

    def query_rows(self, table: str, where: dict) -> list[dict]:
        """Return defensive candidate rows; caller applies the exact predicate.

        Indexed snapshots intersect scalar postings. Unindexed snapshots return
        the relation scan, so both paths use the same engine filter/sort/project.
        Candidate ids are ordered stably, independent of insertion/event order.
        """
        rows = self._tables[table]
        if type(where) is not dict or not all(type(name) is str for name in where):
            raise TypeError("where must be a string-keyed dictionary")
        tokens = [(name, _index_token(value)) for name, value in where.items()]
        if any(token is None for _, token in tokens):
            raise TypeError("Query equality values must be finite JSON scalars")
        if self._indexes is None or not tokens:
            keys = sorted(rows)
        else:
            index = self._indexes[table]
            sets = [index.get(name, {}).get(token, frozenset()) for name, token in tokens]
            sets.sort(key=len)
            candidates = set(sets[0])
            for posting in sets[1:]:
                candidates.intersection_update(posting)
                if not candidates:
                    break
            keys = sorted(candidates)
        return [json.loads(rows[key]) for key in keys]

    @property
    def tables(self) -> dict[str, list[dict]]:
        return {table: self.rows(table) for table in self._tables}

    @property
    def context(self) -> Any:
        return json.loads(self._context)


class Store:
    def __init__(self, tables: dict[str, list[dict]], context: Any = None, *, indexed: bool = True):
        started = perf_counter()
        if type(indexed) is not bool:
            raise TypeError("indexed must be bool")
        if type(tables) is not dict or not all(type(name) is str for name in tables):
            raise TypeError("tables must be a string-keyed dict")
        self._lock = RLock()
        self._owner = object()
        encoded: dict[str, dict[str, str]] = {}
        for table, rows in tables.items():
            encoded[table] = self._encode_rows(rows)
        index_started = perf_counter()
        indexes = {} if indexed else None
        postings = 0
        if indexed:
            for table, rows in encoded.items():
                indexes[table], count = _build_index(rows)
                postings += count
        index_seconds = perf_counter() - index_started
        initial = self._make_snapshot(0, encoded, canonical({} if context is None else context), (), indexes, postings)
        self._history = [initial]
        self._event_prefix = [0]
        self.last_commit_stats: dict[str, int | float] = {}
        self.initial_stats = {"indexed": indexed, "rows": sum(map(len, encoded.values())),
                              "index_postings": postings, "index_build_seconds": index_seconds,
                              "initialization_seconds": perf_counter() - started}

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

    def _make_snapshot(self, revision, tables, context, events, indexes, postings):
        frozen = MappingProxyType({name: rows if isinstance(rows, MappingProxyType) else MappingProxyType(dict(rows))
                                   for name, rows in tables.items()})
        return Snapshot(revision=revision, _tables=frozen, _context=context, events=tuple(events),
                        _owner=self._owner,
                        _indexes=None if indexes is None else MappingProxyType(dict(indexes)),
                        index_postings=postings, row_count=sum(map(len, tables.values())))

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

    def event_count_between(self, before: int, after: int) -> int:
        """Complete event count in O(1); maintained inside atomic commits."""
        with self._lock:
            if not (0 <= before <= after < len(self._history)):
                raise ValueError("Invalid revision interval")
            return self._event_prefix[after] - self._event_prefix[before]

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
            tables = dict(current._tables)
            events = []
            for table in sorted(set(encoded) | set(removed)):
                previous_count = len(events)
                tables[table] = dict(tables[table])
                for key in sorted(removed.get(table, [])):
                    old = tables[table].pop(key, None)
                    if old is not None:
                        events.append(Change(table, key, old, None))
                for key, new in sorted(encoded.get(table, {}).items()):
                    old = tables[table].get(key)
                    if old != new:
                        tables[table][key] = new
                        events.append(Change(table, key, old, new))
                if len(events) == previous_count:
                    tables[table] = current._tables[table]
            index_started = perf_counter()
            indexes = None if current._indexes is None else dict(current._indexes)
            postings = current.index_postings
            if indexes is not None:
                by_table: dict[str, list[Change]] = {}
                for event in events:
                    by_table.setdefault(event.table, []).append(event)
                for table, changes in by_table.items():
                    indexes[table], delta = _update_index(indexes[table], changes)
                    postings += delta
            index_seconds = perf_counter() - index_started
            snapshot = self._make_snapshot(current.revision + 1, tables,
                                           current._context, events, indexes, postings)
            self._history.append(snapshot)
            self._event_prefix.append(self._event_prefix[-1] + len(events))
            self.last_commit_stats = {"events": len(events), "indexed": snapshot.indexed,
                                      "index_postings": postings,
                                      "index_update_seconds": index_seconds,
                                      "commit_seconds": perf_counter() - started}
            return snapshot
