"""Independent bounded enumeration oracle for domain.py's declared schema.

This module never imports or invokes the production solver. It enumerates
candidate assignments, then checks their dependency closure from scratch.
Both implementations deliberately share packaging's PEP440/508 parser; parser
bugs are therefore a common-mode risk, not independently validated semantics.
The oracle is for small finite universes, not a scalable package installer.
"""

from __future__ import annotations

import itertools
import re
from datetime import date
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

_FIELDS = {"implementation_name", "implementation_version", "os_name", "platform_machine", "platform_release", "platform_system", "platform_version", "python_full_version", "platform_python_implementation", "python_version", "sys_platform"}


class _Undetermined(Exception):
    pass


class _Rejected(Exception):
    pass


class _World:
    def __init__(self, tables: dict[str, list[dict[str, Any]]], args: dict[str, Any]):
        if any(key not in tables for key in ("projects", "releases", "aliases", "policies", "environment")):
            raise _Undetermined("schema table absent")
        self.tables, self.args = tables, args
        self.env_name = args.get("environment")
        if not isinstance(self.env_name, str) or not self.env_name:
            raise _Undetermined("environment name required")
        self.today = self.day(args.get("as_of"))
        roots = args.get("requirements")
        if not isinstance(roots, list) or not roots or not all(isinstance(x, str) for x in roots):
            raise _Undetermined("requirements must be a nonempty string list")
        self.roots = roots

    @staticmethod
    def day(value: Any) -> date:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise _Undetermined("date must be YYYY-MM-DD")
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise _Undetermined("invalid date") from exc

    @staticmethod
    def spec(value: Any) -> SpecifierSet:
        try:
            if not isinstance(value, str):
                raise _Undetermined("missing string specifier")
            return SpecifierSet(value)
        except InvalidSpecifier as exc:
            raise _Undetermined("invalid specifier") from exc

    def rows(self, table: str, **fields: Any) -> list[dict[str, Any]]:
        return [row for row in self.tables[table] if all(key in row and type(row[key]) is type(value) and row[key] == value for key, value in fields.items())]

    def env(self, field: str) -> Any:
        rows = self.rows("environment", environment=self.env_name, field=field)
        if len(rows) != 1 or "value" not in rows[0]:
            raise _Undetermined(f"missing or ambiguous environment: {field}")
        return rows[0]["value"]

    def name(self, value: str) -> str:
        name, visited = canonicalize_name(value), set()
        while True:
            if name in visited:
                raise _Undetermined("alias cycle")
            visited.add(name)
            matches = self.rows("aliases", alias=name)
            if not matches:
                return name
            if len(matches) != 1 or not isinstance(matches[0].get("target"), str):
                raise _Undetermined("ambiguous alias")
            name = canonicalize_name(matches[0]["target"])

    def requirement(self, text: str, extras: set[str]) -> Requirement | None:
        try:
            req = Requirement(text)
        except (InvalidRequirement, TypeError) as exc:
            raise _Undetermined("invalid PEP508 requirement") from exc
        if req.marker is not None:
            raw = re.sub(r"'[^']*'|\"[^\"]*\"", "", str(req.marker))
            names = set(re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", raw)) - {"and", "or", "not", "in"}
            if names - _FIELDS - {"extra"}:
                raise _Undetermined("unsupported marker variables")
            values = {key: self.env(key) for key in names - {"extra"}}
            if not all(isinstance(value, str) for value in values.values()):
                raise _Undetermined("marker environment values must be strings")
            try:
                applies = any(req.marker.evaluate({**values, "extra": value}) for value in {""} | extras)
            except (KeyError, ValueError) as exc:
                raise _Undetermined("invalid marker evaluation") from exc
            if not applies:
                return None
        if req.url:
            raise _Undetermined("URL requirement unsupported")
        return req

    def candidates(self, project: str) -> list[dict[str, Any]]:
        rows = self.rows("projects", project=project)
        if len(rows) != 1 or rows[0].get("complete") is not True:
            raise _Undetermined(f"uncertified project: {project}")
        releases = self.rows("releases", project=project)
        try:
            versions = [Version(row["version"]) for row in releases]
        except (KeyError, TypeError, InvalidVersion) as exc:
            raise _Undetermined("invalid candidate version") from exc
        if len(set(versions)) != len(versions):
            raise _Undetermined("ambiguous release artifacts")
        return sorted(releases, key=lambda row: Version(row["version"]), reverse=True)

    def policies(self, project: str, version: Version) -> list[dict[str, Any]]:
        result = []
        for row in self.rows("policies", project=project) + self.rows("policies", project="*"):
            start = self.day(row["valid_from"]) if row.get("valid_from") is not None else None
            stop = self.day(row["valid_until"]) if row.get("valid_until") is not None else None
            if start and stop and start >= stop:
                raise _Undetermined("invalid policy interval")
            if (start and self.today < start) or (stop and self.today >= stop):
                continue
            if not self.spec(row.get("versions", "")).contains(version, prereleases=True):
                continue
            guard = row.get("when", {})
            if not isinstance(guard, dict):
                raise _Undetermined("invalid policy guard")
            checks = [(self.env(field), expected) for field, expected in sorted(guard.items())]
            if all(type(actual) is type(expected) and actual == expected for actual, expected in checks):
                result.append(row)
        return result

    def outgoing(self, project: str, row: dict[str, Any], extras: set[str], *, check: bool) -> list[str]:
        dependencies = row.get("requires_dist")
        if not isinstance(dependencies, list) or not all(isinstance(item, str) for item in dependencies):
            raise _Undetermined("missing dependency metadata")
        version = Version(row["version"])
        if check:
            if type(row.get("yanked")) is not bool:
                raise _Undetermined("unknown yank state")
            if row["yanked"]:
                raise _Rejected("yanked")
            available = row.get("provides_extra")
            if not isinstance(available, list) or not all(isinstance(item, str) for item in available):
                raise _Undetermined("missing extras metadata")
            if extras - {canonicalize_name(item) for item in available}:
                raise _Undetermined("undeclared extra")
            python_spec = self.spec(row.get("requires_python"))
            if str(python_spec):
                try:
                    py = Version(str(self.env("python_full_version")))
                except InvalidVersion as exc:
                    raise _Undetermined("invalid python_full_version") from exc
                if not python_spec.contains(py, prereleases=True):
                    raise _Rejected("Python constraint")
        outgoing = list(dependencies)
        for policy in self.policies(project, version):
            kind = policy.get("kind")
            if kind == "require":
                outgoing.append(policy.get("requirement"))
            elif kind == "deny":
                if check:
                    raise _Rejected("deny policy")
            elif kind == "require_env":
                if not check:
                    continue
                field = policy.get("field")
                if not isinstance(field, str):
                    raise _Undetermined("environment policy needs field")
                actual = self.env(field)
                if policy.get("operator") == "eq" and "value" in policy:
                    okay = type(actual) is type(policy["value"]) and actual == policy["value"]
                elif policy.get("operator") == "version":
                    try:
                        okay = self.spec(policy.get("specifier")).contains(Version(str(actual)), prereleases=True)
                    except InvalidVersion as exc:
                        raise _Undetermined("invalid environment version") from exc
                else:
                    raise _Undetermined("unsupported environment policy")
                if not okay:
                    raise _Rejected("environment policy")
            else:
                raise _Undetermined("unknown policy kind")
        return outgoing

    def possible_projects(self) -> tuple[list[str], set[str]]:
        """Overapproximate relevant projects, independent of an assignment.

        This only reduces the Cartesian enumeration domain. Unknown candidate
        metadata cannot establish a conflict; it is retained as uncertainty.
        """
        extras: dict[str, set[str]] = {}
        unknowns: set[str] = set()

        def add(text: str, active_extras: set[str]) -> None:
            req = self.requirement(text, active_extras)
            if req is not None:
                key = self.name(req.name)
                extras.setdefault(key, set()).update(canonicalize_name(x) for x in req.extras)

        for root in self.roots:
            add(root, set())
        previous = None
        while previous != tuple((key, tuple(sorted(value))) for key, value in sorted(extras.items())):
            previous = tuple((key, tuple(sorted(value))) for key, value in sorted(extras.items()))
            for key in list(extras):
                try:
                    candidates = self.candidates(key)
                except _Undetermined as exc:
                    unknowns.add(str(exc))
                    continue
                for row in candidates:
                    try:
                        for text in self.outgoing(key, row, extras[key], check=False):
                            add(text, extras[key])
                    except _Undetermined as exc:
                        unknowns.add(str(exc))
        return sorted(extras), unknowns

    def check(self, assignment: dict[str, dict[str, Any] | None]) -> dict[str, str]:
        demands: dict[str, list[Requirement]] = {}
        extras: dict[str, set[str]] = {}

        def add(text: str, active_extras: set[str]) -> None:
            req = self.requirement(text, active_extras)
            if req is not None:
                key = self.name(req.name)
                old = demands.setdefault(key, [])
                if str(req) not in {str(other) for other in old}:
                    old.append(req)
                extras.setdefault(key, set()).update(canonicalize_name(x) for x in req.extras)

        for root in self.roots:
            add(root, set())
        processed = None
        while True:
            signature = tuple((key, tuple(sorted(map(str, value))), tuple(sorted(extras[key]))) for key, value in sorted(demands.items()))
            if signature == processed:
                break
            processed = signature
            for project in list(demands):
                self.candidates(project)  # completeness is part of the contract
                candidate = assignment.get(project)
                if candidate is None:
                    raise _Rejected("required project not selected")
                version = Version(candidate["version"])
                if any(not requirement.specifier.contains(version) for requirement in demands[project]):
                    raise _Rejected("version constraint")
                for text in self.outgoing(project, candidate, extras[project], check=True):
                    add(text, extras[project])
        return {project: assignment[project]["version"] for project in sorted(demands)}


def brute_force(tables: dict[str, list[dict[str, Any]]], args: dict[str, Any], *, max_assignments: int = 200000) -> dict[str, Any]:
    """Enumerate finite assignments; budget exhaustion is incomplete, not unsat."""
    result: dict[str, Any] = {"status": "incomplete", "selection": {}, "issues": [], "assignments_examined": 0}
    try:
        if type(max_assignments) is not int or max_assignments < 1:
            raise _Undetermined("max_assignments must be positive")
        world = _World(tables, args)
        projects, _closure_unknowns = world.possible_projects()
        # A malformed candidate excluded by the actual root constraints cannot
        # make a fully disproven assignment unknown. Closure is an overestimate;
        # only assignment checks can contribute uncertainty to the verdict.
        unknowns: set[str] = set()
        domains = []
        for project in projects:
            try:
                domains.append(world.candidates(project) + [None])
            except _Undetermined:
                domains.append([None])
        for values in itertools.product(*domains):
            if result["assignments_examined"] >= max_assignments:
                raise _Undetermined("oracle enumeration budget exhausted")
            result["assignments_examined"] += 1
            try:
                selection = world.check(dict(zip(projects, values)))
                return {**result, "status": "resolved", "selection": selection}
            except _Rejected:
                continue
            except _Undetermined as exc:
                unknowns.add(str(exc))
        result["status"] = "incomplete" if unknowns else "conflict"
        result["issues"] = sorted(unknowns)
    except _Undetermined as exc:
        result["issues"] = [str(exc)]
    return result


def check_selection(tables: dict[str, list[dict[str, Any]]], args: dict[str, Any], selection: dict[str, str]) -> dict[str, Any]:
    """Check a reported witness without running the production search."""
    try:
        world = _World(tables, args)
        assignment = {}
        for project, version in selection.items():
            candidates = [row for row in world.candidates(project) if row["version"] == version]
            if len(candidates) != 1:
                raise _Rejected("witness release unavailable or ambiguous")
            assignment[project] = candidates[0]
        actual = world.check(assignment)
        if actual != selection:
            raise _Rejected("witness has unused or missing projects")
        return {"valid": True, "issues": []}
    except (_Undetermined, _Rejected) as exc:
        return {"valid": False, "issues": [str(exc)]}
