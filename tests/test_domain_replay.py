"""Replay invariants and evidence boundaries; no model inference."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.benchmark_domain import (
    MODES, REQUIRED_MODES, ROOT, apply_reference, controlled_chains,
    public_chains, random_chains, replay_chain, save_report, summarize,
)


class DomainReplayTests(unittest.TestCase):
    def test_controlled_families_distinguish_sound_and_positive_only_modes(self):
        cases = [replay_chain(item) for item in controlled_chains()]
        self.assertEqual(len(cases), 8)
        result = summarize(cases)
        self.assertTrue(result["required_modes_all_agree"], json.dumps(result, indent=2))
        self.assertLess(result["methods"]["positive"]["agreement_count"], result["states"])
        for case in cases:
            for state in case["states"]:
                for mode in REQUIRED_MODES:
                    self.assertTrue(state["methods"][mode]["store_matches_reference"])

    def test_clock_changes_arguments_without_incrementing_store_revision(self):
        case = next(case for case in controlled_chains() if case["id"].startswith("C6-"))
        result = replay_chain(case)
        self.assertEqual([state["oracle"]["status"] for state in result["states"]], ["resolved", "conflict", "resolved"])
        for mode in MODES:
            self.assertEqual([state["methods"][mode]["revision"] for state in result["states"]], [0, 0, 0])

    def test_same_selection_does_not_hide_changed_provenance(self):
        case = next(case for case in controlled_chains() if case["id"].startswith("C3-"))
        result = replay_chain(case)
        changed = result["states"][1]["methods"]["predicate"]
        self.assertTrue(changed["same_status_and_selection_as_previous"])
        self.assertTrue(changed["citations_changed_from_previous"])
        self.assertEqual(result["states"][2]["methods"]["predicate"]["prediction"]["status"], "conflict")

    def test_seeded_graphs_are_reproducible_and_witnesses_are_checked(self):
        first = random_chains(51, 8)
        self.assertEqual(first, random_chains(51, 8))
        self.assertNotEqual(first, random_chains(52, 8))
        reports = [replay_chain(case) for case in first]
        self.assertTrue(summarize(reports)["required_modes_all_agree"])
        for report in reports:
            for state in report["states"]:
                for mode in REQUIRED_MODES:
                    row = state["methods"][mode]
                    if row["prediction"]["status"] == "resolved":
                        self.assertTrue(row["witness"]["valid"])

    def test_public_metadata_and_synthetic_events_are_distinguished(self):
        catalog = json.loads((ROOT / "data/public/pypi-catalog.json").read_text())
        for case in public_chains(catalog):
            self.assertEqual(case["origin"], "verified_public_metadata_with_synthetic_intervention")
            self.assertTrue(case["source_records"])
            self.assertIn("synthetic", case["steps"][1]["label"])
            report = replay_chain(case)
            self.assertTrue(all(state["methods"][mode]["agreement"] for state in report["states"] for mode in REQUIRED_MODES))

    def test_reference_transition_does_not_mutate_original_and_save_refuses_overwrite(self):
        case = controlled_chains()[0]
        prior = copy.deepcopy(case["tables"])
        changed = apply_reference(case["tables"], case["steps"][1])
        self.assertEqual(case["tables"], prior)
        self.assertNotEqual(changed, prior)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            save_report({"sentinel": True}, output)
            with self.assertRaises(FileExistsError):
                save_report({"sentinel": False}, output)
            self.assertEqual(json.loads(output.read_text()), {"sentinel": True})


if __name__ == "__main__":
    unittest.main()
