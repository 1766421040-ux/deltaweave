"""CPU semantic tests, independent witnesses and incremental update regressions."""

import copy
import itertools
import json
from pathlib import Path
import unittest

from deltaweave import Engine, Store
from deltaweave.domain import TABLES, register_domain, tables_from_catalog
from deltaweave.oracle import brute_force, check_selection


def release(name, version="1", deps=(), extras=(), **overrides):
    return {"id": f"{name}-{version}", "project": name, "version": version,
            "requires_python": ">=3.8", "requires_dist": list(deps),
            "provides_extra": list(extras), "yanked": False, **overrides}


def world(*releases):
    data = {table: [] for table in TABLES}
    data["releases"] = list(releases)
    data["projects"] = [{"id": name, "project": name, "complete": True}
                        for name in sorted({row["project"] for row in releases})]
    fields = {"python_full_version": "3.11.8", "python_version": "3.11",
              "sys_platform": "linux", "os_name": "posix",
              "platform_python_implementation": "CPython", "implementation_name": "cpython",
              "implementation_version": "3.11.8", "platform_machine": "x86_64",
              "platform_release": "test-release", "platform_system": "Linux", "platform_version": "test-version"}
    data["environment"] = [{"id": key, "environment": "test", "field": key, "value": value}
                           for key, value in fields.items()]
    return data


def arguments(*requirements, **overrides):
    return {"requirements": list(requirements), "environment": "test", "as_of": "2026-10-28", **overrides}


def runner(tables, mode="predicate"):
    store = Store(tables)
    engine = Engine(store, mode=mode)
    register_domain(engine)
    return store, engine


class DomainTests(unittest.TestCase):
    def assert_reference(self, tables, args, expected):
        _, engine = runner(tables)
        actual = engine.run("resolve", args)
        self.assertEqual(actual["status"], expected, actual)
        reference = brute_force(tables, args)
        self.assertEqual(reference["status"], expected, reference)
        if expected == "resolved":
            self.assertTrue(check_selection(tables, args, actual["selection"])["valid"], actual)
        return actual

    def test_public_requests_and_httpx_are_declared_constraints_only(self):
        path = Path(__file__).resolve().parents[1] / "data/public/pypi-catalog.json"
        catalog = json.loads(path.read_text())
        env = {row["field"]: row["value"] for row in world()["environment"]}
        tables = tables_from_catalog(catalog, environments={"test": env})
        for root in ("requests>=2.31", "httpx>=0.27"):
            with self.subTest(root=root):
                actual = self.assert_reference(tables, arguments(root), "resolved")
                self.assertIn("not runtime compatibility", actual["scope"])
                self.assertTrue(actual["constraints"])
                self.assertTrue(actual["citations"])
        # An optional dependency outside the selected universe is unknown.
        self.assert_reference(tables, arguments("requests[socks]"), "incomplete")

    def test_backtracking_and_late_transitive_constraint(self):
        data = world(release("a", "1"), release("a", "2"), release("z", deps=["a<2"]))
        answer = self.assert_reference(data, arguments("a", "z"), "resolved")
        self.assertEqual(answer["selection"], {"a": "1", "z": "1"})
        self.assert_reference(data, arguments("a>=2", "z"), "conflict")

    def test_requested_extras_union_after_project_already_selected(self):
        data = world(release("a", deps=["b"]), release("b", deps=['d; extra == "fast"'], extras=["fast"]),
                     release("c", deps=["b[fast]"]), release("d"))
        base = self.assert_reference(data, arguments("a"), "resolved")
        with_extra = self.assert_reference(data, arguments("a", "c"), "resolved")
        self.assertNotIn("d", base["selection"])
        self.assertEqual(set(with_extra["selection"]), {"a", "b", "c", "d"})

    def test_markers_have_no_host_environment_fallback(self):
        data = world(release("a", deps=['b; python_version < "3.12"']), release("b"))
        data["environment"] = [row for row in data["environment"] if row["field"] != "python_version"]
        self.assert_reference(data, arguments("a"), "incomplete")
        data["environment"].append({"id": "pv", "environment": "test", "field": "python_version", "value": "3.12"})
        result = self.assert_reference(data, arguments("a"), "resolved")
        self.assertEqual(result["selection"], {"a": "1"})

    def test_missing_project_is_not_certified_empty(self):
        data = world(release("a", deps=["b"]))
        self.assert_reference(data, arguments("a"), "incomplete")
        data["projects"].append({"id": "b", "project": "b", "complete": True})
        self.assert_reference(data, arguments("a"), "conflict")

    def test_yanked_release_excluded_even_for_exact_pin(self):
        data = world(release("a", "1"), release("a", "2", yanked=True))
        answer = self.assert_reference(data, arguments("a"), "resolved")
        self.assertEqual(answer["selection"], {"a": "1"})
        self.assert_reference(data, arguments("a==2"), "conflict")

    def test_unknown_extras_urls_and_metadata_are_not_silent_success(self):
        for data, args in [
            (world(release("a")), arguments("a[not-declared]")),
            (world(release("a")), arguments("a @ https://example.invalid/a.whl")),
            (world(release("a", requires_dist=None)), arguments("a")),
            (world(release("a", requires_python=None)), arguments("a")),
            (world(release("a", "1"), release("a", "1.0")), arguments("a")),
        ]:
            with self.subTest(args=args, releases=data["releases"]):
                self.assert_reference(data, args, "incomplete")

    def test_new_exception_creates_new_environment_read_then_recovers(self):
        policy = {"id": "abi", "project": "a", "kind": "require_env", "field": "ssl_abi", "operator": "version", "specifier": ">=3"}
        for mode in ("full", "coarse", "predicate", "adaptive"):
            with self.subTest(mode=mode):
                store, engine = runner(world(release("a")), mode)
                args = arguments("a")
                self.assertEqual(engine.run("resolve", args)["status"], "resolved")
                store.commit(upserts={"policies": [policy]})
                self.assertEqual(engine.run("resolve", args)["status"], "incomplete")
                env = {"id": "ssl", "environment": "test", "field": "ssl_abi", "value": "1"}
                store.commit(upserts={"environment": [env]})
                self.assertEqual(engine.run("resolve", args)["status"], "conflict")
                store.commit(upserts={"environment": [{**env, "value": "3"}]})
                self.assertEqual(engine.run("resolve", args)["status"], "resolved")
                self.assertEqual(brute_force(store.snapshot.tables, args)["status"], "resolved")
        # The deliberately weak baseline really misses insertion into an empty query.
        store, engine = runner(world(release("a")), "positive")
        engine.run("resolve", arguments("a"))
        store.commit(upserts={"policies": [policy]})
        self.assertEqual(engine.run("resolve", arguments("a"))["status"], "resolved")

    def test_policy_adds_new_package_dependency(self):
        store, engine = runner(world(release("a"), release("b", "1"), release("b", "2")))
        args = arguments("a")
        self.assertEqual(engine.run("resolve", args)["selection"], {"a": "1"})
        policy = {"id": "new", "project": "a", "kind": "require", "requirement": "b>=2", "when": {"sys_platform": "linux"}}
        store.commit(upserts={"policies": [policy]})
        result = engine.run("resolve", args)
        self.assertEqual(result["selection"], {"a": "1", "b": "2"})
        self.assertTrue(check_selection(store.snapshot.tables, args, result["selection"])["valid"])

    def test_alias_insertion_and_cycles(self):
        data = world(release("a"))
        store, engine = runner(data)
        args = arguments("legacy")
        self.assertEqual(engine.run("resolve", args)["status"], "incomplete")
        store.commit(upserts={"aliases": [{"id": "old", "alias": "legacy", "target": "a"}]})
        self.assertEqual(engine.run("resolve", args)["selection"], {"a": "1"})
        store.commit(upserts={"aliases": [{"id": "cycle", "alias": "a", "target": "legacy"}]})
        self.assertEqual(engine.run("resolve", args)["status"], "incomplete")
        self.assertEqual(brute_force(store.snapshot.tables, args)["status"], "incomplete")

    def test_policy_dates_are_half_open_and_time_is_explicit(self):
        data = world(release("a"))
        data["policies"] = [{"id": "pause", "project": "*", "kind": "deny", "valid_from": "2026-10-01", "valid_until": "2026-11-01"}]
        for day, status in [("2026-09-30", "resolved"), ("2026-10-01", "conflict"), ("2026-11-01", "resolved")]:
            self.assert_reference(data, arguments("a", as_of=day), status)

    def test_withdrawal_and_equal_version_source_replacement(self):
        store, engine = runner(world(release("a")))
        args = arguments("a")
        initial = engine.run("resolve", args)
        store.commit(deletes={"releases": ["a-1"]}, upserts={"releases": [release("a", id="replacement")]})
        replaced = engine.run("resolve", args)
        self.assertEqual(initial["selection"], replaced["selection"])
        self.assertNotIn("releases:a-1", replaced["citations"])
        self.assertIn("releases:replacement", replaced["citations"])
        store.commit(deletes={"releases": ["replacement"]})
        self.assertEqual(engine.run("resolve", args)["status"], "conflict")

    def test_unrelated_project_update_does_not_execute_solver(self):
        store, engine = runner(world(release("a"), release("z")))
        engine.run("resolve", arguments("a"))
        store.commit(upserts={"releases": [release("z", "2")]})
        self.assertEqual(engine.run("resolve", arguments("a"))["status"], "resolved")
        self.assertEqual(engine.last_receipt["stats"]["task_executions"], 0)

    def test_search_and_enumeration_budgets_cannot_prove_conflict(self):
        data = world(release("a", "1"), release("a", "2", deps=["b>=2"]), release("b"))
        _, engine = runner(data)
        self.assertEqual(engine.run("resolve", arguments("a", max_states=1))["status"], "incomplete")
        self.assertEqual(brute_force(data, arguments("a"), max_assignments=1)["status"], "incomplete")
        self.assert_reference(data, arguments("a"), "resolved")

    def test_cyclic_dependencies_reach_finite_closure(self):
        data = world(release("a", deps=["b>=1"]), release("b", deps=["a>=1"]))
        self.assert_reference(data, arguments("a"), "resolved")
        self.assert_reference(data, arguments("a", "b<1"), "conflict")

    def test_small_cartesian_worlds_against_independent_enumerator(self):
        for a2_requirement, b2_yanked, root in itertools.product(("b<2", "b>=2"), (True, False), ("a", "a>=2")):
            data = world(release("a", "1", deps=["b<2"]), release("a", "2", deps=[a2_requirement]),
                         release("b", "1"), release("b", "2", yanked=b2_yanked))
            args = arguments(root, "b>=2")
            expected = brute_force(data, args)["status"]
            self.assert_reference(data, args, expected)

    def test_oracle_rejects_unsupported_witness_not_just_matching_status(self):
        data = world(release("a", deps=["b>=2"]), release("b", "1"), release("b", "2"))
        args = arguments("a")
        self.assertFalse(check_selection(data, args, {"a": "1", "b": "1"})["valid"])
        self.assertFalse(check_selection(data, args, {"a": "1"})["valid"])
        self.assertTrue(check_selection(data, args, {"a": "1", "b": "2"})["valid"])

    def test_excluded_unknown_candidate_does_not_pollute_conflict_proof(self):
        data = world(release("a", "1", deps=["other @ https://example.invalid/other.whl"]),
                     release("a", "2", deps=["b>=2"]), release("b", "1"))
        self.assert_reference(data, arguments("a>=2"), "conflict")

    def test_marker_scalar_does_not_coerce_unknown_to_string(self):
        data = world(release("a", deps=['b; sys_platform == "linux"']), release("b"))
        for row in data["environment"]:
            if row["field"] == "sys_platform":
                row["value"] = None
        self.assert_reference(data, arguments("a"), "incomplete")


if __name__ == "__main__":
    unittest.main()
