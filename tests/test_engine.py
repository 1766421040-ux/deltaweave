"""CPU-only adversarial update semantics, not agent-quality measurements."""
from dataclasses import FrozenInstanceError
import itertools
from threading import Event, Thread
import unittest

from deltaweave import Engine, Query, RevisionConflict, Store


class StoreTests(unittest.TestCase):
    def test_defensive_values_old_snapshots_and_atomic_commit(self):
        initial = {"items": [{"id": "a", "value": {"nested": [1]}}], "policy": []}
        context = {"env": ["old"]}
        store = Store(initial, context)
        old = store.snapshot
        initial["items"][0]["value"]["nested"].append(2)
        context["env"].append("mutated")
        old.rows("items")[0]["value"]["nested"].append(3)
        old.tables["items"].clear()
        old.context["env"].clear()
        self.assertEqual(old.rows("items")[0]["value"], {"nested": [1]})
        self.assertEqual(old.context, {"env": ["old"]})
        with self.assertRaises(FrozenInstanceError):
            old.revision = 99
        with self.assertRaises(TypeError):
            old._tables["items"]["a"] = "changed"
        new = store.commit(upserts={"items": [{"id": "a", "value": 2}],
                                    "policy": [{"id": "p", "allow": True}]},
                           expected_revision=0)
        self.assertEqual(new.revision, 1)
        self.assertEqual(len(new.events), 2)
        self.assertEqual(old.rows("items")[0]["value"], {"nested": [1]})
        new.events[0].before["value"] = "mutated"
        self.assertEqual(new.events[0].old["value"], {"nested": [1]})
        self.assertEqual(new.events[0].new["value"], 2)
        with self.assertRaises(RevisionConflict):
            store.commit(deletes={"items": ["a"]}, expected_revision=0)
        self.assertIs(store.snapshot, new)
        # Invalid last table cannot partially commit a valid first mutation.
        with self.assertRaises(ValueError):
            store.commit(upserts={"items": [{"id": "a", "value": 8}],
                                  "policy": [{"id": "p"}, {"id": "p"}]})
        self.assertIs(store.snapshot, new)
        with self.assertRaises(ValueError):
            store.commit(upserts={"items": [{"id": "a"}]}, deletes={"items": ["a"]})
        self.assertIs(store.snapshot, new)

    def test_noop_commit_has_no_events_and_strict_json(self):
        store = Store({"t": [{"id": 1, "n": 2}]})
        snapshot = store.commit(upserts={"t": [{"n": 2, "id": 1}]}, deletes={"t": ["missing"]})
        self.assertEqual(snapshot.revision, 1)
        self.assertEqual(snapshot.events, ())
        self.assertEqual(store.events_between(0, 1), ())
        for bad in ({"id": True}, {"id": "a", "bad": float("nan")}, {"id": None}):
            with self.assertRaises(TypeError):
                store.commit(upserts={"t": [bad]})
        self.assertEqual(store.snapshot.revision, 1)

    def test_index_is_immutable_and_shares_untouched_relations(self):
        store = Store({"t": [{"id": "a", "x": 1}], "untouched": [{"id": "u", "x": 2}]})
        old = store.snapshot
        self.assertTrue(old.indexed)
        self.assertEqual(old.index_postings, 4)
        with self.assertRaises(TypeError):
            old._indexes["t"]["x"]["1"] = frozenset()
        new = store.commit(upserts={"t": [{"id": "a", "x": 2, "nested": [1]}]})
        self.assertIs(old._tables["untouched"], new._tables["untouched"])
        self.assertIs(old._indexes["untouched"], new._indexes["untouched"])
        self.assertEqual(old.query_rows("t", {"x": 1}), [{"id": "a", "x": 1}])
        self.assertEqual(new.query_rows("t", {"x": 1}), [])
        self.assertEqual(new.index_postings, 4)  # Nested value is not indexed.
        same = store.commit(upserts={"t": [{"id": "a", "x": 2, "nested": [1]}]})
        self.assertIs(same._tables["t"], new._tables["t"])
        self.assertIs(same._indexes["t"], new._indexes["t"])
        self.assertGreaterEqual(store.initial_stats["initialization_seconds"], store.initial_stats["index_build_seconds"])
        self.assertGreaterEqual(store.last_commit_stats["commit_seconds"], store.last_commit_stats["index_update_seconds"])
        scan = Store({"t": []}, indexed=False)
        self.assertFalse(scan.snapshot.indexed)
        self.assertEqual(scan.initial_stats["index_postings"], 0)


class EngineTests(unittest.TestCase):
    def test_result_receipt_pair_is_atomic_against_a_concurrent_run(self):
        store = Store({"t": [{"id": "row", "value": 1}]})
        engine = Engine(store)
        engine.register("read", lambda ctx, args: ctx.query(Query("t")))
        old_snapshot = store.snapshot
        first_computed, competitor_attempted, competitor_done = Event(), Event(), Event()
        real_run = engine.run
        other = {}

        def intercepted(name, args, snapshot=None):
            value = real_run(name, args, snapshot=snapshot)
            first_computed.set()
            self.assertTrue(competitor_attempted.wait(5))
            # The competitor reached Engine.run but cannot overwrite its receipt
            # until the outer atomic operation has copied the first receipt.
            self.assertFalse(competitor_done.wait(0.1))
            return value

        def compete():
            try:
                if not first_computed.wait(5):
                    raise AssertionError("Initial run did not finish")
                store.commit(upserts={"t": [{"id": "row", "value": 2}]})
                competitor_attempted.set()
                other["value"] = real_run("read", {})
            except BaseException as exc:
                other["error"] = exc
                competitor_attempted.set()
            finally:
                competitor_done.set()

        engine.run = intercepted
        thread = Thread(target=compete)
        thread.start()
        value, receipt = engine.run_with_receipt("read", {}, snapshot=old_snapshot)
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", other)
        self.assertEqual(value, [{"id": "row", "value": 1}])
        self.assertEqual(other["value"], [{"id": "row", "value": 2}])
        self.assertEqual(receipt["validated_at"], 0)
        self.assertEqual(engine.last_receipt["validated_at"], 1)
        # The returned pair does not expose the stored receipt or trace value.
        receipt["dependencies"].clear()
        value[0]["value"] = 999
        self.assertTrue(engine.last_receipt["dependencies"])
        self.assertEqual(real_run("read", {}, snapshot=old_snapshot)[0]["value"], 1)

    def make_query_engine(self, mode, rows, query):
        store = Store({"t": rows})
        engine = Engine(store, mode=mode)
        engine.register("read", lambda ctx, args: ctx.query(query))
        return store, engine

    def test_c1_empty_query_insertion_positive_is_intentionally_incomplete(self):
        for mode in ("full", "positive", "coarse", "predicate", "adaptive"):
            with self.subTest(mode=mode):
                store, engine = self.make_query_engine(mode, [], Query("t", {"project": "p"}))
                self.assertEqual(engine.run("read", {}), [])
                store.commit(upserts={"t": [{"id": "new", "project": "p"}]})
                expected = [] if mode == "positive" else [{"id": "new", "project": "p"}]
                self.assertEqual(engine.run("read", {}), expected)

    def test_c2_new_top_candidate_without_old_positive_edge(self):
        for mode in ("full", "positive", "coarse", "predicate", "adaptive"):
            store, engine = self.make_query_engine(mode, [{"id": "old", "rank": 3}],
                                                   Query("t", order_by=("rank",), limit=1))
            self.assertEqual(engine.run("read", {})[0]["id"], "old")
            store.commit(upserts={"t": [{"id": "new", "rank": 1}]})
            expected = "old" if mode == "positive" else "new"
            self.assertEqual(engine.run("read", {})[0]["id"], expected)

    def test_c3_move_out_delete_and_same_count_replacement(self):
        for mode in ("full", "coarse", "predicate", "adaptive"):
            store, engine = self.make_query_engine(mode, [{"id": "a", "enabled": True}],
                                                   Query("t", {"enabled": True}))
            self.assertEqual(len(engine.run("read", {})), 1)
            store.commit(upserts={"t": [{"id": "a", "enabled": False}]})
            self.assertEqual(engine.run("read", {}), [])
            store.commit(upserts={"t": [{"id": "b", "enabled": True}]})
            self.assertEqual(engine.run("read", {})[0]["id"], "b")
            store.commit(deletes={"t": ["b"]}, upserts={"t": [{"id": "c", "enabled": True}]})
            self.assertEqual(engine.run("read", {})[0]["id"], "c")
            store.commit(deletes={"t": ["c"]})
            self.assertEqual(engine.run("read", {}), [])

    def test_c4_projection_is_exact_and_provenance_separate(self):
        store, engine = self.make_query_engine("predicate",
            [{"id": "a", "project": "p", "n": 1, "noise": "before"}],
            Query("t", {"project": "p"}, select=("n",)))
        self.assertEqual(engine.run("read", {}), [{"n": 1}])
        store.commit(upserts={"t": [{"id": "a", "project": "p", "n": 1, "noise": "after"}]})
        self.assertEqual(engine.run("read", {}), [{"n": 1}])
        self.assertEqual(engine.last_receipt["computed_at"], 0)
        self.assertEqual(engine.last_receipt["validated_at"], 1)
        self.assertEqual(engine.last_receipt["stats"]["task_executions"], 0)
        self.assertEqual(engine.last_receipt["stats"]["query_executions"], 1)
        self.assertEqual(engine.receipts()[-1]["returned_ids"], ["a"])
        # Changed id but same projected value still updates query provenance.
        store.commit(deletes={"t": ["a"]}, upserts={"t": [{"id": "b", "project": "p", "n": 1}]})
        self.assertEqual(engine.run("read", {}), [{"n": 1}])
        self.assertEqual(engine.receipts()[-1]["returned_ids"], ["b"])
        store.commit(upserts={"t": [{"id": "b", "project": "p", "n": 2}]})
        self.assertEqual(engine.run("read", {}), [{"n": 2}])

    def test_c5_scoped_coarse_is_strong_and_predicate_has_same_query_executor(self):
        query = Query("t", {"project": "p", "enabled": True}, select=("v",))
        outcomes = {}
        for mode in ("coarse", "predicate"):
            store, engine = self.make_query_engine(mode, [{"id": "a", "project": "p", "enabled": True, "v": 1}], query)
            engine.run("read", {})
            # Different project: both guards avoid a query, including coarse.
            store.commit(upserts={"t": [{"id": "q", "project": "q", "enabled": True, "v": 9}]})
            engine.run("read", {})
            self.assertEqual(engine.last_receipt["stats"]["query_executions"], 0)
            # Same project but nonmember: only the finer predicate avoids a scan.
            store.commit(upserts={"t": [{"id": "b", "project": "p", "enabled": False, "v": 8}]})
            self.assertEqual(engine.run("read", {}), [{"v": 1}])
            outcomes[mode] = engine.last_receipt["stats"]
        self.assertEqual(outcomes["coarse"]["query_scans"], 1)
        self.assertEqual(outcomes["predicate"]["query_scans"], 0)
        self.assertEqual(outcomes["coarse"]["task_executions"], 0)
        self.assertEqual(outcomes["predicate"]["task_executions"], 0)
        with self.assertRaises(ValueError):
            Query("t", {"project": "p"}, scope={"project": "different"})

    def test_readset_switch_output_unchanged_then_new_dependency_changes(self):
        store = Store({"t": [{"id": "flag", "value": "a"},
                              {"id": "a", "value": 7}, {"id": "b", "value": 7}]})
        engine = Engine(store)
        def choose(ctx, args):
            flag = ctx.query(Query("t", {"id": "flag"}))[0]["value"]
            return ctx.query(Query("t", {"id": flag}))[0]["value"]
        engine.register("choose", choose)
        engine.register("outer", lambda ctx, args: {"answer": ctx.call("choose", {})})
        self.assertEqual(engine.run("outer", {}), {"answer": 7})
        store.commit(upserts={"t": [{"id": "flag", "value": "b"}]})
        self.assertEqual(engine.run("outer", {}), {"answer": 7})
        self.assertEqual(engine.last_receipt["computed_at"], 0)
        current = [r for r in engine.receipts() if r.get("name") == "choose" and r["validated_at"] == 1][0]
        self.assertEqual(current["dependencies"][1]["query"]["where"], {"id": "b"})
        store.commit(upserts={"t": [{"id": "b", "value": 8}]})
        self.assertEqual(engine.run("outer", {}), {"answer": 8})

    def test_short_circuit_obsolete_branch_before_validating_missing_input(self):
        store = Store({"t": [{"id": "flag", "value": "a"},
                              {"id": "a", "value": 7}, {"id": "b", "value": 9}]})
        engine = Engine(store)
        engine.register("branch", lambda ctx, args: ctx.query(Query("t", {"id": args["id"]}))[0]["value"])
        def root(ctx, args):
            flag = ctx.query(Query("t", {"id": "flag"}))[0]["value"]
            return ctx.call("branch", {"id": flag})
        engine.register("root", root)
        self.assertEqual(engine.run("root", {}), 7)
        store.commit(upserts={"t": [{"id": "flag", "value": "b"}]}, deletes={"t": ["a"]})
        # Revalidating obsolete branch(a) would raise IndexError.
        self.assertEqual(engine.run("root", {}), 9)

    def test_model_code_version_invalidates_parent_on_same_snapshot(self):
        store = Store({"t": []})
        engine = Engine(store)
        engine.register("model", lambda ctx, args: 1, version="weights-a")
        engine.register("root", lambda ctx, args: ctx.call("model", {}))
        self.assertEqual(engine.run("root", {}), 1)
        engine.register("model", lambda ctx, args: 2, version="weights-b")
        self.assertEqual(engine.run("root", {}), 2)
        with self.assertRaises(ValueError):
            engine.register("model", lambda ctx, args: 3, version="weights-b")
        engine.register("root", lambda ctx, args: 99, version="prompt-v2")
        self.assertEqual(engine.run("root", {}), 99)

    def test_held_snapshot_during_commit_and_out_of_order_snapshot_requests(self):
        for mode in ("full", "coarse", "predicate", "adaptive"):
            store = Store({"t": [{"id": "a", "value": 1}, {"id": "b", "value": 1}]})
            engine = Engine(store, mode)
            def root(ctx, args):
                a = ctx.query(Query("t", {"id": "a"}))[0]["value"]
                if store.snapshot.revision == 0:
                    store.commit(upserts={"t": [{"id": "a", "value": 2}, {"id": "b", "value": 2}]})
                b = ctx.query(Query("t", {"id": "b"}))[0]["value"]
                return [a, b]
            engine.register("root", root)
            old = store.snapshot
            self.assertEqual(engine.run("root", {}, snapshot=old), [1, 1])
            self.assertEqual(engine.run("root", {}), [2, 2])
            self.assertEqual(engine.run("root", {}, snapshot=old), [1, 1])
            self.assertEqual(engine.run("root", {}), [2, 2])
            with self.assertRaises(ValueError):
                engine.run("root", {}, snapshot=Store({"t": []}).snapshot)

    def test_return_args_query_context_and_receipt_mutation_cannot_poison_cache(self):
        store = Store({"t": [{"id": "a", "nested": [1]}]}, {"constant": [2]})
        engine = Engine(store)
        def run(ctx, args):
            rows = ctx.query(Query("t"))
            rows[0]["nested"].append(7)
            data = ctx.context
            data["constant"].append(8)
            args["x"].append(9)
            return [rows, data, args]
        engine.register("run", run)
        args = {"x": [3]}
        expected = [[{"id": "a", "nested": [1, 7]}], {"constant": [2, 8]}, {"x": [3, 9]}]
        value = engine.run("run", args)
        self.assertEqual(value, expected)
        value[0].clear()
        engine.last_receipt["dependencies"].clear()
        engine.receipts()[0]["dependencies"].clear()
        self.assertEqual(engine.run("run", args), expected)
        self.assertEqual(args, {"x": [3]})
        store.commit()
        self.assertEqual(engine.run("run", args), expected)
        self.assertEqual(engine.last_receipt["computed_at"], 0)
        self.assertEqual(store.snapshot.rows("t"), [{"id": "a", "nested": [1]}])

    def test_full_memo_shares_task_and_query_within_each_snapshot(self):
        store = Store({"t": [{"id": "a", "value": 1}]})
        engine = Engine(store, "full")
        engine.register("read", lambda ctx, args: ctx.query(Query("t")))
        engine.register("root", lambda ctx, args: [ctx.call("read", {}), ctx.call("read", {})])
        engine.run("root", {})
        self.assertEqual(engine.stats["task_executions"], 2)
        self.assertEqual(engine.stats["query_executions"], 1)
        engine.run("root", {})
        self.assertEqual(engine.stats["task_executions"], 2)
        store.commit()
        engine.run("root", {})
        self.assertEqual(engine.stats["task_executions"], 4)
        self.assertEqual(engine.stats["query_executions"], 2)

    def test_query_missing_null_type_sort_limit_and_input_immutability(self):
        where = {"v": None}
        query = Query("t", where, select=("id",), order_by=("rank",), limit=1)
        where["v"] = True
        query.where["v"] = False
        rows = [{"id": "b", "rank": 1, "v": None}, {"id": "a", "rank": 1, "v": None},
                {"id": "c", "rank": 0}, {"id": "d", "rank": 0, "v": False}]
        _, engine = self.make_query_engine("predicate", rows, query)
        self.assertEqual(engine.run("read", {}), [{"id": "a"}])
        _, engine2 = self.make_query_engine("predicate", [{"id": "a", "v": True}, {"id": "b", "v": 1}], Query("t", {"v": 1}))
        self.assertEqual(engine2.run("read", {})[0]["id"], "b")
        _, empty = self.make_query_engine("predicate", rows, Query("t", select=(), limit=0))
        self.assertEqual(empty.run("read", {}), [])
        _, missing = self.make_query_engine("predicate", rows, Query("t", select=("absent",)))
        with self.assertRaises(KeyError):
            missing.run("read", {})

    def test_cycles_and_failures_do_not_publish_successful_root(self):
        engine = Engine(Store({"t": []}))
        engine.register("loop", lambda ctx, args: ctx.call("loop", args))
        with self.assertRaisesRegex(RuntimeError, "Cyclic"):
            engine.run("loop", {})
        self.assertEqual(engine.last_receipt["status"], "error")
        self.assertEqual(engine.receipts(), [])

    def test_mode_and_store_cannot_be_swapped_under_existing_cache(self):
        engine = Engine(Store({"t": []}), "positive")
        with self.assertRaises(AttributeError):
            engine.mode = "predicate"
        with self.assertRaises(AttributeError):
            engine.store = Store({"t": []})
        engine.register("read", lambda ctx, args: ctx.query(Query("t")))
        engine.run("read", {})
        self.assertEqual(engine.last_receipt["validation_contract"], "positive_rows_heuristic")

    def test_finite_two_step_state_space_matches_full_recompute(self):
        # Exact finite oracle: direct Python filtering/sorting/projection, outside engine.
        variants = [None, {"id": "a", "project": "p", "enabled": True, "v": 1},
                    {"id": "a", "project": "p", "enabled": False, "v": 2},
                    {"id": "a", "project": "q", "enabled": True, "v": 3}]
        query = Query("t", {"project": "p", "enabled": True}, select=("v",))
        for mode, first, second in itertools.product(("coarse", "predicate", "adaptive"), variants, variants):
            store, engine = self.make_query_engine(mode, [], query)
            self.assertEqual(engine.run("read", {}), [])
            for row in (first, second):
                if row is None:
                    store.commit(deletes={"t": ["a"]})
                else:
                    store.commit(upserts={"t": [row]})
                expected = [] if row is None or row["project"] != "p" or not row["enabled"] else [{"v": row["v"]}]
                self.assertEqual(engine.run("read", {}), expected)

    def test_index_and_scan_have_identical_semantics_through_updates(self):
        initial = [{"id": "null", "v": None}, {"id": "missing"},
                   {"id": "boolean", "v": True}, {"id": "integer", "v": 1},
                   {"id": "float", "v": 1.0}, {"id": "zero", "v": -0.0},
                   {"id": "nested", "v": [1, 2]}]
        operations = [
            {"upserts": {"t": [{"id": "missing", "v": None}, {"id": "nested", "v": 1}]}},
            {"upserts": {"t": [{"id": "integer", "v": [1]}, {"id": "null", "other": 4}]}},
            {"deletes": {"t": ["float", "zero"]}, "upserts": {"t": [{"id": "new", "v": 0.0}]}},
            {"deletes": {"t": ["new", "missing", "nested", "boolean"]}},
        ]
        for mode in ("full", "positive", "coarse", "predicate", "adaptive"):
            stores = [Store({"t": initial}, indexed=indexed) for indexed in (True, False)]
            engines = [Engine(store, mode) for store in stores]
            for engine in engines:
                engine.register("read", lambda ctx, args: ctx.query(Query("t", args, select=("id",))))
            for operation in [None] + operations:
                if operation:
                    for store in stores:
                        store.commit(**operation)
                for where in ({}, {"v": None}, {"v": True}, {"v": 1}, {"v": 1.0},
                              {"v": 0.0}, {"v": -0.0}, {"absent": None}):
                    self.assertEqual(engines[0].run("read", where), engines[1].run("read", where))
            self.assertGreater(engines[1].stats["query_scans"], engines[0].stats["query_scans"])

    def test_index_matches_all_equalities_but_none_is_not_missing(self):
        store = Store({"t": [{"id": 1, "project": "p", "v": None},
                              {"id": 2, "project": "p"},
                              {"id": 3, "project": "q", "v": None}]})
        self.assertEqual(store.snapshot.query_rows("t", {"project": "p", "v": None}),
                         [{"id": 1, "project": "p", "v": None}])
        with self.assertRaises(TypeError):
            store.snapshot.query_rows("t", {"v": [1]})

    def test_adaptive_fixed_threshold_and_event_prefix(self):
        store = Store({"t": [{"id": number, "v": 0} for number in range(4)]})
        engine = Engine(store, "adaptive")
        engine.register("read", lambda ctx, args: ctx.query(Query("t", {"id": 0})))
        engine.run("read", {})
        self.assertEqual(engine.last_receipt["execution_policy"]["reason"], "cold_root")
        store.commit(upserts={"t": [{"id": 1, "v": 1}]})
        self.assertEqual(store.event_count_between(0, 1), 1)
        engine.run("read", {})
        policy = engine.last_receipt["execution_policy"]
        self.assertEqual(policy["event_density"], .25)
        self.assertEqual(policy["selected_mode"], "coarse")
        store.commit(upserts={"t": [{"id": 2, "v": 1}, {"id": 3, "v": 1}]})
        engine.run("read", {})
        self.assertEqual(engine.last_receipt["execution_policy"]["selected_mode"], "full")
        self.assertEqual(engine.last_receipt["stats"]["adaptive_fallback_runs"], 1)
        self.assertEqual(engine.last_receipt["stats"]["task_executions"], 1)
        self.assertEqual(engine.last_receipt["stats"]["guard_checks"], 0)
        self.assertEqual(store.event_count_between(0, 2), 3)
        self.assertEqual(store.event_count_between(1, 2), 2)
        store.commit()
        self.assertEqual(store.event_count_between(2, 3), 0)
        with self.assertRaises(RevisionConflict):
            store.commit(deletes={"t": [0]}, expected_revision=0)
        self.assertEqual(store.event_count_between(0, 3), 3)
        with self.assertRaises(ValueError):
            store.event_count_between(3, 2)
        self.assertEqual(Engine(store).mode, "coarse")

    def test_adaptive_full_fallback_replaces_readset_then_coarse_is_sound(self):
        padding = [{"id": f"pad-{i}", "value": 0} for i in range(8)]
        store = Store({"t": [{"id": "flag", "value": "a"},
                              {"id": "a", "value": 7}, {"id": "b", "value": 7}] + padding})
        engine = Engine(store, "adaptive")
        def choose(ctx, args):
            flag = ctx.query(Query("t", {"id": "flag"}))[0]["value"]
            return ctx.query(Query("t", {"id": flag}))[0]["value"]
        engine.register("choose", choose)
        engine.register("root", lambda ctx, args: [ctx.call("choose", {}), ctx.call("choose", {})])
        self.assertEqual(engine.run("root", {}), [7, 7])
        high = [{"id": "flag", "value": "b"}] + [{**row, "value": 1} for row in padding]
        store.commit(upserts={"t": high})
        self.assertEqual(engine.run("root", {}), [7, 7])
        self.assertEqual(engine.last_receipt["execution_policy"]["reason"], "high_event_density")
        self.assertEqual(engine.last_receipt["stats"]["task_executions"], 2)
        # A second root request in the same snapshot shares valid memo values.
        self.assertEqual(engine.run("root", {}), [7, 7])
        self.assertEqual(engine.last_receipt["stats"]["task_executions"], 0)
        store.commit(upserts={"t": [{"id": "b", "value": 8}]})
        self.assertEqual(engine.run("root", {}), [8, 8])
        self.assertEqual(engine.last_receipt["execution_policy"]["selected_mode"], "coarse")
        self.assertGreaterEqual(engine.last_receipt["stats"]["adaptive_policy_seconds"], 0)

    def test_adaptive_code_epoch_and_obsolete_branch_validation(self):
        padding = [{"id": f"pad-{i}", "value": 0} for i in range(12)]
        store = Store({"t": [{"id": "flag", "value": "a"},
                              {"id": "a", "value": 7}, {"id": "b", "value": 9}] + padding})
        engine = Engine(store, "adaptive")
        engine.register("branch", lambda ctx, args: ctx.query(Query("t", {"id": args["id"]}))[0]["value"])
        def root(ctx, args):
            flag = ctx.query(Query("t", {"id": "flag"}))[0]["value"]
            return ctx.call("branch", {"id": flag})
        engine.register("root", root)
        self.assertEqual(engine.run("root", {}), 7)
        store.commit(upserts={"t": [{"id": "flag", "value": "b"}]}, deletes={"t": ["a"]})
        self.assertEqual(engine.run("root", {}), 9)
        self.assertEqual(engine.last_receipt["execution_policy"]["selected_mode"], "coarse")
        engine.register("branch", lambda ctx, args: 10, version="2")
        self.assertEqual(engine.run("root", {}), 10)
        self.assertEqual(engine.last_receipt["execution_policy"]["event_density"], 0)


if __name__ == "__main__":
    unittest.main()
