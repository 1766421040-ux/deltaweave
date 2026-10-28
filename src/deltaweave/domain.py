"""Bounded declared-dependency resolution, not installation/runtime compatibility.

All tables must exist (empty tables are meaningful):
* projects: {id, project: normalized name, complete: bool}. ``complete=True``
  certifies ONLY the selected finite release universe, never all of PyPI.
* releases: {id, project, version, requires_python: str, requires_dist: [str],
  provides_extra: [str], yanked: bool}. One metadata artifact per project/version.
* aliases: {id, alias: normalized name, target: normalized name}.
* environment: {id, environment: name, field: str, value: JSON scalar}.
* policies: {id, project: normalized name or '*', kind, versions?: PEP440,
  when?: {environment_field: exact_value}, valid_from?: YYYY-MM-DD,
  valid_until?: YYYY-MM-DD}. Time intervals are [from, until).
  kind='deny' forbids matching releases; kind='require' adds ``requirement``;
  kind='require_env' needs ``field`` and either operator='eq', value=... or
  operator='version', specifier=PEP440. Unknown kinds/operators are incomplete.

solve(ctx, args): args = {requirements: [PEP508], environment: str,
as_of: YYYY-MM-DD, max_states?: positive int}. No wall clock or host marker
defaults are used. Markers require all referenced environment fields; extras
use the union of the base context and each requested extra. Direct URLs and
undeclared extras are explicitly incomplete. Yanked releases are always
excluded, including exact pins (a stated policy, not pip's exception behavior).
Pre-releases require explicit PEP440 admission; there is no implicit fallback.

Outputs have status resolved/conflict/incomplete, selection {name: version},
constraints, citations ('table:id' observations), issues, and scope. Conflict
means no solution in the certified finite domain. Search limits, absent project
certificates or unsupported metadata mean incomplete, never proof of conflict.
The solver uses tracked queries only; the independent oracle is in oracle.py.
"""

from __future__ import annotations

import copy
import json
import re
from datetime import date
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

SCOPE = "Selected finite release universe; declared constraints only, not runtime compatibility or all PyPI."
TABLES = ("projects", "releases", "aliases", "policies", "environment")
MARKER_FIELDS = frozenset({
    "implementation_name", "implementation_version", "os_name", "platform_machine",
    "platform_release", "platform_system", "platform_version",
    "python_full_version", "platform_python_implementation", "python_version", "sys_platform",
})


class _Unknown(Exception):
    pass


class _NoSolution(Exception):
    pass


def _day(value: Any) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise _Unknown("dates must use YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise _Unknown("invalid calendar date") from exc


def _spec(value: Any) -> SpecifierSet:
    if not isinstance(value, str):
        raise _Unknown("specifier must be a string")
    try:
        return SpecifierSet(value)
    except InvalidSpecifier as exc:
        raise _Unknown(f"unsupported specifier: {value}") from exc


class _Resolver:
    def __init__(self, ctx: Any, args: dict[str, Any]):
        self.ctx = ctx
        self.args = args
        self.env_name = args.get("environment")
        if not isinstance(self.env_name, str) or not self.env_name:
            raise _Unknown("environment name is required")
        self.as_of = _day(args.get("as_of"))
        self.roots = args.get("requirements")
        if not isinstance(self.roots, list) or not self.roots or not all(isinstance(x, str) for x in self.roots):
            raise _Unknown("requirements must be a nonempty list of PEP508 strings")
        self.limit = args.get("max_states", 10000)
        if type(self.limit) is not int or self.limit < 1:
            raise _Unknown("max_states must be a positive integer")
        self.states = 0
        self.observed: set[str] = set()
        self.queries: dict[str, list[dict[str, Any]]] = {}
        self.issues: set[str] = set()

    def rows(self, table: str, **where: Any) -> list[dict[str, Any]]:
        # Import here keeps the schema adapter usable before an engine is created.
        from .engine import Query
        key = json.dumps([table, where], sort_keys=True)
        if key not in self.queries:
            try:
                found = self.ctx.query(Query(table=table, where=where))
            except KeyError as exc:
                raise _Unknown(f"required table missing: {table}") from exc
            self.queries[key] = found
            self.observed.update(f"{table}:{row['id']}" for row in found)
        return self.queries[key]

    def env(self, field: str) -> Any:
        found = self.rows("environment", environment=self.env_name, field=field)
        if len(found) != 1 or "value" not in found[0]:
            raise _Unknown(f"environment field unavailable or ambiguous: {field}")
        return found[0]["value"]

    def name(self, raw: str) -> str:
        name = canonicalize_name(raw)
        seen: set[str] = set()
        while True:
            if name in seen:
                raise _Unknown(f"alias cycle: {name}")
            seen.add(name)
            aliases = self.rows("aliases", alias=name)
            if not aliases:
                return name
            if len(aliases) != 1 or not isinstance(aliases[0].get("target"), str):
                raise _Unknown(f"ambiguous or malformed alias: {name}")
            name = canonicalize_name(aliases[0]["target"])

    def marker(self, req: Requirement, extras: set[str]) -> bool:
        if req.marker is None:
            return True
        # Remove literal strings before discovering variable names. This avoids
        # packaging.evaluate's silent fallback to the current host environment.
        expression = re.sub(r"'[^']*'|\"[^\"]*\"", "", str(req.marker))
        variables = set(re.findall(r"\b[a-zA-Z_][a-zA-Z_0-9]*\b", expression)) - {"and", "or", "in", "not"}
        if variables - MARKER_FIELDS - {"extra"}:
            raise _Unknown("unsupported marker variables")
        env = {field: self.env(field) for field in sorted(variables - {"extra"})}
        if not all(isinstance(value, str) for value in env.values()):
            raise _Unknown("PEP508 marker environment values must be strings")
        try:
            return any(req.marker.evaluate({**env, "extra": extra}) for extra in [""] + sorted(extras))
        except (ValueError, KeyError) as exc:
            raise _Unknown(f"cannot evaluate marker: {req.marker}") from exc

    def policies(self, project: str, version: Version) -> list[dict[str, Any]]:
        active = []
        # Empty queries are dependencies, so a future exception can invalidate.
        for row in self.rows("policies", project="*") + self.rows("policies", project=project):
            begin = _day(row["valid_from"]) if row.get("valid_from") is not None else None
            end = _day(row["valid_until"]) if row.get("valid_until") is not None else None
            if begin and end and begin >= end:
                raise _Unknown("invalid policy time interval")
            if (begin and self.as_of < begin) or (end and self.as_of >= end):
                continue
            if not _spec(row.get("versions", "")).contains(version, prereleases=True):
                continue
            when = row.get("when", {})
            if not isinstance(when, dict):
                raise _Unknown("policy when must be an exact environment mapping")
            matches = True
            for field, expected in sorted(when.items()):
                actual = self.env(field)
                if type(actual) is not type(expected) or actual != expected:
                    matches = False
            if matches:
                active.append(row)
        return active

    def candidates(self, project: str) -> list[dict[str, Any]]:
        certificates = self.rows("projects", project=project)
        if len(certificates) != 1 or certificates[0].get("complete") is not True:
            raise _Unknown(f"no complete finite-universe certificate: {project}")
        rows = self.rows("releases", project=project)
        if not rows:
            raise _NoSolution(f"certified empty candidate universe: {project}")
        versions = []
        for row in rows:
            try:
                version = Version(row["version"])
            except (KeyError, TypeError, InvalidVersion) as exc:
                raise _Unknown(f"invalid release version in {project}") from exc
            versions.append(version)
        if len(set(versions)) != len(versions):
            raise _Unknown(f"multiple metadata artifacts for one version: {project}")
        return [row for _, row in sorted(zip(versions, rows), key=lambda pair: pair[0], reverse=True)]

    def analyse(self, selected: dict[str, dict[str, Any]]) -> tuple[dict[str, list[Requirement]], list[dict[str, Any]]]:
        needs: dict[str, list[Requirement]] = {}
        extras: dict[str, set[str]] = {}
        entries: dict[str, dict[str, Any]] = {}

        def record(kind: str, project: str, requirement: str, source: str) -> None:
            row = {"kind": kind, "project": project, "requirement": requirement, "source": source}
            entries[json.dumps(row, sort_keys=True)] = row

        def add(text: str, source: str, parent_extras: set[str], kind: str) -> None:
            try:
                req = Requirement(text)
            except (InvalidRequirement, TypeError) as exc:
                raise _Unknown(f"invalid requirement: {text}") from exc
            if not self.marker(req, parent_extras):
                return
            if req.url:
                raise _Unknown(f"direct URL requirements are unsupported: {req.name}")
            name = self.name(req.name)
            record(kind, name, str(req), source)
            requirements = needs.setdefault(name, [])
            if str(req) not in {str(old) for old in requirements}:
                requirements.append(req)
            extras.setdefault(name, set()).update(canonicalize_name(x) for x in req.extras)

        for text in self.roots:
            add(text, "root", set(), "root")
        prior = None
        while True:
            signature = tuple((key, tuple(sorted(map(str, needs[key]))), tuple(sorted(extras[key]))) for key in sorted(needs))
            if signature == prior:
                break
            prior = signature
            for project in sorted(needs):
                if project not in selected:
                    continue
                row = selected[project]
                version = Version(row["version"])
                if not all(req.specifier.contains(version) for req in needs[project]):
                    raise _NoSolution(f"version constraints disagree: {project}")
                if type(row.get("yanked")) is not bool:
                    raise _Unknown(f"unknown yank state: {project} {version}")
                if row["yanked"]:
                    raise _NoSolution(f"yanked candidate excluded: {project} {version}")
                dependencies, available_extras = row.get("requires_dist"), row.get("provides_extra")
                if not isinstance(dependencies, list) or not all(isinstance(x, str) for x in dependencies):
                    raise _Unknown(f"missing dependency metadata: {project}")
                if not isinstance(available_extras, list) or not all(isinstance(x, str) for x in available_extras):
                    raise _Unknown(f"missing extras metadata: {project}")
                if extras[project] - {canonicalize_name(x) for x in available_extras}:
                    raise _Unknown(f"undeclared requested extra: {project}")
                python_spec = _spec(row.get("requires_python"))
                if str(python_spec):
                    py = self.env("python_full_version")
                    try:
                        okay = python_spec.contains(Version(str(py)), prereleases=True)
                    except InvalidVersion as exc:
                        raise _Unknown("invalid python_full_version") from exc
                    record("python", project, str(python_spec), f"releases:{row['id']}")
                    if not okay:
                        raise _NoSolution(f"Python constraint rejects {project} {version}")
                for text in dependencies:
                    add(text, f"releases:{row['id']}", extras[project], "dependency")
                for policy in self.policies(project, version):
                    source = f"policies:{policy['id']}"
                    kind = policy.get("kind")
                    if kind == "deny":
                        raise _NoSolution(f"policy denies {project} {version}: {policy['id']}")
                    if kind == "require":
                        add(policy.get("requirement"), source, extras[project], "policy")
                    elif kind == "require_env":
                        field, operator = policy.get("field"), policy.get("operator")
                        if not isinstance(field, str):
                            raise _Unknown("require_env needs a field")
                        actual = self.env(field)
                        if operator == "eq" and "value" in policy:
                            okay = type(actual) is type(policy["value"]) and actual == policy["value"]
                            description = f"{field} == {policy['value']!r}"
                        elif operator == "version":
                            constraint = _spec(policy.get("specifier"))
                            try:
                                okay = constraint.contains(Version(str(actual)), prereleases=True)
                            except InvalidVersion as exc:
                                raise _Unknown(f"environment field is not a version: {field}") from exc
                            description = f"{field} {constraint}"
                        else:
                            raise _Unknown("unsupported require_env operator")
                        record("environment", project, description, source)
                        if not okay:
                            raise _NoSolution(f"environment policy rejects {project}: {policy['id']}")
                    elif kind != "deny":
                        raise _Unknown(f"unsupported policy kind: {kind}")
        return needs, [entries[key] for key in sorted(entries)]

    def search(self, selected: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]] | None:
        self.states += 1
        if self.states > self.limit:
            raise _Unknown("search budget exhausted")
        try:
            needs, constraints = self.analyse(selected)
            pending = sorted(set(needs) - set(selected))
            if not pending:
                return selected, constraints
            project = pending[0]
            candidates = self.candidates(project)
        except _NoSolution as exc:
            self.issues.add(str(exc))
            return None
        except _Unknown as exc:
            self.issues.add(str(exc))
            raise
        unknowns: set[str] = set()
        for row in candidates:
            try:
                result = self.search({**selected, project: row})
                if result is not None:
                    return result
            except _Unknown as exc:
                unknowns.add(str(exc))
                if self.states > self.limit:
                    break
        if unknowns:
            raise _Unknown("; ".join(sorted(unknowns)))
        return None


def solve(ctx: Any, args: dict[str, Any]) -> dict[str, Any]:
    """Tracked deterministic baseline. Register directly as an engine node."""
    result: dict[str, Any] = {"status": "incomplete", "selection": {}, "constraints": [], "citations": [], "issues": [], "scope": SCOPE}
    resolver = None
    try:
        if not isinstance(args, dict):
            raise _Unknown("arguments must be an object")
        resolver = _Resolver(ctx, args)
        found = resolver.search({})
        if found is None:
            result["status"] = "conflict"
            result["issues"] = sorted(resolver.issues)
        else:
            selected, constraints = found
            result.update(status="resolved", selection={key: row["version"] for key, row in sorted(selected.items())}, constraints=constraints)
    except _Unknown as exc:
        result["issues"] = [str(exc)]
    if resolver is not None:
        # These are observations used in the search, not a minimal unsat core.
        result["citations"] = sorted(resolver.observed)
    return result


def register_domain(engine: Any, name: str = "resolve") -> None:
    engine.register(name, solve, version="finite-domain-v1")


def tables_from_catalog(catalog: dict[str, Any], *, environments: dict[str, dict[str, Any]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Adapt verified selected wheel metadata without inventing missing projects.

    Pass explicit environments; no values are inferred from the running machine.
    Project completeness refers to records actually included in this catalog.
    A catalog `missing` entry does not become a certified empty project.
    """
    tables: dict[str, list[dict[str, Any]]] = {name: [] for name in TABLES}
    projects = set()
    for original in catalog["records"]:
        row = copy.deepcopy(original)
        project = canonicalize_name(row["project"])
        row.update(project=project, id=f"{project}=={row['version']}")
        tables["releases"].append(row)
        projects.add(project)
    tables["projects"] = [{"id": project, "project": project, "complete": True} for project in sorted(projects)]
    for env, fields in sorted((environments or {}).items()):
        for field, value in sorted(fields.items()):
            tables["environment"].append({"id": f"{env}:{field}", "environment": env, "field": field, "value": value})
    return tables
