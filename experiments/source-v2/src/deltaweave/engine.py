"""Deterministic, read-only dynamic traces over immutable JSON snapshots.

The guarantee is equivalence to full execution, not correctness of an agent's
answer. Callables must expose every mutable read through ctx.query/ctx.call and
bump their version when code, prompts, model or decoding changes. Hidden I/O,
randomness and external effects are outside this contract.

All modes share the same exact query executor and optional snapshot equality
index (Store defaults to indexed=True). Coarse guards use explicit scope, or
project equality when present, otherwise the entire relation. Predicate
guards use full equality membership. Positive guards intentionally miss inserts.
No mode claims a globally minimal validation frontier.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from threading import RLock
from time import perf_counter
from typing import Any, Callable

from .store import Snapshot, Store, canonical, id_key


def _scalar(value):
    if value is not None and type(value) not in (str, bool, int, float):
        raise TypeError("Query equality/order values must be JSON scalars")
    canonical(value)


def _equal(left, right):
    return type(left) is type(right) and left == right


def _matches(row, where):
    return row is not None and all(key in row and _equal(row[key], value)
                                   for key, value in where.items())


def _order(value):
    _scalar(value)
    rank = {type(None): 0, bool: 1, int: 2, float: 3, str: 4}[type(value)]
    return rank, "" if value is None else value


@dataclass(frozen=True, init=False)
class Query:
    table: str
    select: tuple[str, ...] | None
    order_by: tuple[str, ...]
    limit: int | None
    _where: str = field(repr=False)
    _scope: str = field(repr=False)

    def __init__(self, table: str, where: dict | None = None,
                 select: tuple[str, ...] | None = None,
                 order_by: tuple[str, ...] = (), limit: int | None = None,
                 scope: dict | None = None):
        if type(table) is not str:
            raise TypeError("Query table must be a string")
        where = {} if where is None else where
        if type(where) is not dict or not all(type(k) is str for k in where):
            raise TypeError("where must be a string-keyed dictionary")
        for value in where.values():
            _scalar(value)
        if scope is None:
            scope = {"project": where["project"]} if "project" in where else {}
        if type(scope) is not dict or not all(
                key in where and _equal(value, where[key]) for key, value in scope.items()):
            raise ValueError("scope must be an equality subset of where")
        for fields in (select, order_by):
            if fields is not None and (type(fields) is not tuple or
                    not all(type(item) is str for item in fields) or
                    len(set(fields)) != len(fields)):
                raise TypeError("select/order_by must be tuples of unique field names")
        if order_by is None:
            raise TypeError("order_by cannot be None")
        if limit is not None and (type(limit) is not int or limit < 0):
            raise ValueError("limit must be a nonnegative integer or None")
        for key, value in {"table": table, "select": select, "order_by": order_by,
                           "limit": limit, "_where": canonical(where),
                           "_scope": canonical(scope)}.items():
            object.__setattr__(self, key, value)

    @property
    def where(self):
        return json.loads(self._where)

    @property
    def scope(self):
        return json.loads(self._scope)

    def descriptor(self):
        return {"table": self.table, "where": self.where, "scope": self.scope,
                "select": None if self.select is None else list(self.select),
                "order_by": list(self.order_by), "limit": self.limit}

    @property
    def key(self):
        return canonical(self.descriptor())


@dataclass(frozen=True)
class _Dependency:
    kind: str
    value: str
    query: Query | None = None
    name: str | None = None
    args: str | None = None


@dataclass(frozen=True)
class _Trace:
    key: str
    name: str
    version: str
    args: str
    value: str
    dependencies: tuple[_Dependency, ...]
    computed_at: int
    validated_at: int
    epoch: int


@dataclass(frozen=True)
class _QueryTrace:
    query: Query
    value: str
    returned_ids: frozenset[str]
    computed_at: int
    validated_at: int


class Context:
    """Only these tracked reads are supported inside registered callables."""
    def __init__(self, engine: Engine, snapshot: Snapshot):
        self._engine = engine
        self._snapshot = snapshot
        self._dependencies: list[_Dependency] = []
        self._child_seconds = 0.0

    def query(self, query: Query) -> list[dict]:
        started = perf_counter()
        try:
            trace = self._engine._query(query, self._snapshot)
            self._dependencies.append(_Dependency("query", trace.value, query=query))
            return json.loads(trace.value)
        finally:
            self._child_seconds += perf_counter() - started

    def call(self, name: str, args: Any) -> Any:
        started = perf_counter()
        try:
            encoded = canonical(args)
            trace = self._engine._task(name, encoded, self._snapshot)
            self._dependencies.append(_Dependency("task", trace.value, name=name, args=encoded))
            return json.loads(trace.value)
        finally:
            self._child_seconds += perf_counter() - started

    @property
    def context(self) -> Any:
        value = self._snapshot._context
        self._dependencies.append(_Dependency("context", value))
        return json.loads(value)


class Engine:
    def __init__(self, store: Store, mode: str = "predicate"):
        if mode not in {"full", "positive", "coarse", "predicate"}:
            raise ValueError("Unknown engine mode")
        self._store = store
        self._mode = mode
        self._functions: dict[str, tuple[str, Callable]] = {}
        self._known_functions: dict[tuple[str, str], Callable] = {}
        self._epoch = 0
        self._tasks: dict[str, dict[int, _Trace]] = {}
        self._queries: dict[str, dict[int, _QueryTrace]] = {}
        self._active: set[str] = set()
        self._lock = RLock()
        self._last_receipt = None
        self.stats: dict[str, int | float] = {}
        self.reset_stats()

    @property
    def store(self):
        return self._store

    @property
    def mode(self):
        return self._mode

    def reset_stats(self):
        self.stats = {name: 0 for name in (
            "runs", "task_executions", "task_reuses", "task_validations",
            "query_executions", "query_reuses", "query_scans", "guard_checks",
            "guard_matches", "observation_changes", "dependency_checks",
            "trace_writes", "dependency_edges_written", "query_receipt_writes")}
        self.stats.update({name: 0.0 for name in (
            "run_seconds", "query_seconds", "guard_seconds", "task_self_seconds",
            "validation_self_seconds")})

    def register(self, name: str, function: Callable, version: str = "1"):
        if type(name) is not str or not name or type(version) is not str or not callable(function):
            raise TypeError("register expects name, callable, and string version")
        with self._lock:
            if self._active:
                raise RuntimeError("Cannot register during an execution")
            old = self._known_functions.get((name, version))
            if old is not None and old is not function:
                raise ValueError("A changed callable must use a new version")
            if self._functions.get(name) != (version, function):
                self._epoch += 1
            self._known_functions[name, version] = function
            self._functions[name] = version, function

    @property
    def last_receipt(self):
        return None if self._last_receipt is None else json.loads(canonical(self._last_receipt))

    def run(self, name: str, args: Any, snapshot: Snapshot | None = None) -> Any:
        with self._lock:
            if self._active:
                raise RuntimeError("Nested execution must use ctx.call")
            snapshot = self.store.snapshot if snapshot is None else snapshot
            if not self.store.owns(snapshot):
                raise ValueError("Snapshot does not belong to this store")
            encoded = canonical(args)
            before = dict(self.stats)
            started = perf_counter()
            self.stats["runs"] += 1
            self._last_receipt = None
            try:
                trace = self._task(name, encoded, snapshot)
                self._last_receipt = self._receipt(trace)
                return json.loads(trace.value)
            except Exception as exc:
                self._last_receipt = {"status": "error", "snapshot": snapshot.revision,
                                      "error_type": type(exc).__name__}
                raise
            finally:
                self.stats["run_seconds"] += perf_counter() - started
                if self._last_receipt is not None:
                    self._last_receipt["stats"] = {key: self.stats[key] - before[key]
                                                   for key in self.stats}

    @staticmethod
    def _prior(records, revision):
        eligible = [key for key in records if key <= revision]
        return records[max(eligible)] if eligible else None

    def _task(self, name, args, snapshot):
        version, function = self._functions[name]
        key = canonical([name, version, json.loads(args)])
        if key in self._active:
            raise RuntimeError("Cyclic task call detected")
        records = self._tasks.setdefault(key, {})
        prior = self._prior(records, snapshot.revision)
        if prior is not None and prior.validated_at == snapshot.revision and prior.epoch == self._epoch:
            self.stats["task_reuses"] += 1
            return prior
        self._active.add(key)
        try:
            if prior is not None and self.mode != "full":
                self.stats["task_validations"] += 1
                validation_start = perf_counter()
                child_seconds = 0.0
                unchanged = True
                for dependency in prior.dependencies:
                    self.stats["dependency_checks"] += 1
                    child_start = perf_counter()
                    if dependency.kind == "query":
                        value = self._query(dependency.query, snapshot).value
                    elif dependency.kind == "task":
                        value = self._task(dependency.name, dependency.args, snapshot).value
                    else:
                        value = snapshot._context
                    child_seconds += perf_counter() - child_start
                    if value != dependency.value:
                        unchanged = False
                        break  # Do not evaluate the now-obsolete remainder of this trace.
                self.stats["validation_self_seconds"] += max(
                    0.0, perf_counter() - validation_start - child_seconds)
                if unchanged:
                    trace = _Trace(key, name, version, args, prior.value, prior.dependencies,
                                   prior.computed_at, snapshot.revision, self._epoch)
                    records[snapshot.revision] = trace
                    self.stats["task_reuses"] += 1
                    self._count_trace(trace)
                    return trace
            ctx = Context(self, snapshot)
            self.stats["task_executions"] += 1
            start = perf_counter()
            try:
                result = canonical(function(ctx, json.loads(args)))
            finally:
                self.stats["task_self_seconds"] += max(0.0, perf_counter() - start - ctx._child_seconds)
            # Always replace the readset, even when the public value is unchanged.
            trace = _Trace(key, name, version, args, result, tuple(ctx._dependencies),
                           snapshot.revision, snapshot.revision, self._epoch)
            records[snapshot.revision] = trace
            if prior is not None and prior.value != result:
                self.stats["observation_changes"] += 1
            self._count_trace(trace)
            return trace
        finally:
            self._active.remove(key)

    def _count_trace(self, trace):
        self.stats["trace_writes"] += 1
        self.stats["dependency_edges_written"] += len(trace.dependencies)

    def _query(self, query: Query, snapshot: Snapshot):
        if not isinstance(query, Query):
            raise TypeError("ctx.query expects Query")
        records = self._queries.setdefault(query.key, {})
        prior = self._prior(records, snapshot.revision)
        if prior is not None and prior.validated_at == snapshot.revision:
            self.stats["query_reuses"] += 1
            return prior
        if prior is not None and self.mode != "full":
            guard_start = perf_counter()
            changed = False
            where = query.scope if self.mode == "coarse" else query.where
            for event in self.store.events_between(prior.validated_at, snapshot.revision):
                self.stats["guard_checks"] += 1
                if event.table != query.table:
                    continue
                if self.mode == "positive":
                    matches = event._id in prior.returned_ids
                else:
                    matches = _matches(event.before, where) or _matches(event.after, where)
                if matches:
                    changed = True
                    self.stats["guard_matches"] += 1
                    break
            self.stats["guard_seconds"] += perf_counter() - guard_start
            if not changed:
                trace = _QueryTrace(query, prior.value, prior.returned_ids,
                                    prior.computed_at, snapshot.revision)
                records[snapshot.revision] = trace
                self.stats["query_reuses"] += 1
                self.stats["query_receipt_writes"] += 1
                return trace
        start = perf_counter()
        self.stats["query_executions"] += 1
        try:
            selected = []
            where = query.where
            for row in snapshot.query_rows(query.table, where):
                self.stats["query_scans"] += 1
                if _matches(row, where):
                    selected.append(row)
            selected.sort(key=lambda row: tuple(_order(row[field]) for field in query.order_by)
                          + (_order(row["id"]),))
            if query.limit is not None:
                selected = selected[:query.limit]
            ids = frozenset(id_key(row["id"]) for row in selected)
            projected = selected if query.select is None else [
                {field: row[field] for field in query.select} for row in selected]
            value = canonical(projected)
            trace = _QueryTrace(query, value, ids, snapshot.revision, snapshot.revision)
            records[snapshot.revision] = trace
            self.stats["query_receipt_writes"] += 1
            return trace
        finally:
            self.stats["query_seconds"] += perf_counter() - start

    @staticmethod
    def _digest(encoded):
        return sha256(encoded.encode("utf-8")).hexdigest()

    def _receipt(self, trace):
        dependencies = []
        for dependency in trace.dependencies:
            item = {"kind": dependency.kind, "value_sha256": self._digest(dependency.value)}
            if dependency.query is not None:
                item["query"] = dependency.query.descriptor()
            if dependency.name is not None:
                item.update(name=dependency.name, args=json.loads(dependency.args))
            dependencies.append(item)
        return {"status": "valid", "kind": "task", "name": trace.name,
                "mode": self.mode,
                "validation_contract": ("positive_rows_heuristic" if self.mode == "positive"
                                        else "conservative_observation"),
                "version": trace.version, "args": json.loads(trace.args),
                "key_sha256": self._digest(trace.key), "value_sha256": self._digest(trace.value),
                "computed_at": trace.computed_at, "validated_at": trace.validated_at,
                "registry_epoch": trace.epoch, "dependencies": dependencies}

    def receipts(self) -> list[dict]:
        with self._lock:
            result = [self._receipt(trace) for records in self._tasks.values()
                      for trace in records.values()]
            for records in self._queries.values():
                for trace in records.values():
                    result.append({"status": "valid", "kind": "query",
                        "mode": self.mode,
                        "validation_contract": ("positive_rows_heuristic" if self.mode == "positive"
                                                else "conservative_observation"),
                        "query": trace.query.descriptor(),
                        "value_sha256": self._digest(trace.value),
                        "returned_ids": sorted((json.loads(key) for key in trace.returned_ids),
                                               key=_order),
                        "computed_at": trace.computed_at, "validated_at": trace.validated_at})
            return result
