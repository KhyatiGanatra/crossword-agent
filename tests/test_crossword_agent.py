from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from crossword_agent.model import FixtureModel, ModelError, parse_json_object
from crossword_agent.puzzle import derive_entries, load_fixture
from crossword_agent.run_history import append_run_record, build_run_record
from crossword_agent.sweep_solver import SweepFillSolver, solve_with_sweep_fill
from crossword_agent.types import JsonReply, SolveConfig
from evals.metrics import score_result

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "data" / "dev" / "mini-001.json"
OFFLINE = SolveConfig(strong_model=None, reasoning_model=None, audit_model=None)


class SharedModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = load_fixture(FIXTURE_PATH)
        self.entries = derive_entries(self.fixture.puzzle)

    def test_derives_across_and_down_entries(self) -> None:
        self.assertEqual(len(self.entries), 10)
        self.assertEqual(self.entries["1A"].length, 3)
        self.assertEqual(self.entries["1D"].length, 5)
        self.assertEqual(self.entries["5D"].length, 3)


class SweepFillSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = load_fixture(FIXTURE_PATH)
        self.puzzle = self.fixture.puzzle
        self.entries = derive_entries(self.puzzle)

    def test_offline_fixture_solves_exactly_without_seeing_gold(self) -> None:
        model = FixtureModel(self.fixture.candidate_bank)
        result = solve_with_sweep_fill(self.puzzle, model, OFFLINE)

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.grid, self.fixture.solution)
        self.assertEqual(result.unverified_entries, ())
        self.assertEqual([event.action for event in result.events][:2], ["start", "sweep"])
        self.assertEqual(result.events[-1].action, "finish")
        self.assertEqual(result.model_calls, model.calls)
        self.assertEqual(result.estimated_cost_usd, 0.0)
        # The sweep prompt carries no letters; no prompt ever contains a solution row.
        self.assertNotIn("pattern=", model.prompts[0])
        for prompt in model.prompts:
            for row in self.fixture.solution:
                if "#" in row:
                    self.assertNotIn(row, prompt)
            self.assertNotIn("solution", prompt.lower())

        metrics = score_result(self.puzzle, self.fixture.solution, result)
        self.assertTrue(metrics["exact_puzzle"])
        self.assertTrue(metrics["self_report_correct"])
        self.assertEqual(metrics["reasoning_tokens"], 0)

    def test_wrong_length_and_wrong_alternative_are_outscored(self) -> None:
        # 1A offers LOAN (wrong length) before EAR; 1D offers EAGER before EMBER.
        model = FixtureModel(self.fixture.candidate_bank)
        result = solve_with_sweep_fill(self.puzzle, model, OFFLINE)
        self.assertEqual(result.assignments["1A"], "EAR")
        self.assertEqual(result.assignments["1D"], "EMBER")

    def test_fill_abstains_when_a_weak_candidate_blocks_stronger_ones(self) -> None:
        solver = SweepFillSolver(self.puzzle, FixtureModel({}), OFFLINE)
        gold = {
            entry_id: "".join(self.fixture.solution[row][col] for row, col in entry.cells)
            for entry_id, entry in self.entries.items()
        }
        for entry_id, answer in gold.items():
            solver.cands[entry_id] = {answer: 1.0}
        solver.cands["1D"] = {"EAGER": 0.35}  # only a weak, conflicting candidate

        grid, assigned = solver.fill()

        self.assertEqual(assigned["4A"][0], "EMBER")
        self.assertEqual(assigned["6A"][0], "ABUSE")
        self.assertEqual(solver.pattern(grid, "1D"), "EMBER")  # implied by crossings
        self.assertFalse(solver.verified("1D", assigned))  # never proposed for that clue
        self.assertTrue(solver.verified("4A", assigned))
        self.assertIn("1D", solver.weak_set(assigned, 0.55))

    def test_trusted_grid_reveals_only_verified_letters(self) -> None:
        solver = SweepFillSolver(self.puzzle, FixtureModel({}), OFFLINE)
        solver.cands["1A"] = {"EAR": 1.0}
        solver.cands["5D"] = {"XEN": 1.0}
        assigned = {"1A": ("EAR", 1.0, 3.0), "5D": ("XEN", 0.2, 1.4)}

        trusted = solver.trusted_grid(assigned, 0.55)

        self.assertEqual(solver.pattern(trusted, "1A"), "EAR")
        self.assertEqual(solver.pattern(trusted, "5D"), "...")


    def test_weak_entry_prompt_pattern_never_echoes_a_full_fill(self) -> None:
        solver = SweepFillSolver(self.puzzle, FixtureModel({}), OFFLINE)
        for entry_id, entry in self.entries.items():
            gold = "".join(self.fixture.solution[row][col] for row, col in entry.cells)
            solver.cands[entry_id] = {gold: 1.0}
        solver.cands["1D"] = {"EAGER": 0.35}
        grid, assigned = solver.fill()
        weak = solver.weak_set(assigned, 0.55)
        self.assertIn("1D", weak)

        trusted, patterns = solver.prompt_patterns(assigned, weak, 0.55)

        pattern = patterns["1D"]
        self.assertEqual(len(pattern), 5)
        self.assertEqual(pattern.count("."), 1)  # fully determined by crossings, so one doubt is opened
        for shown, gold_letter in zip(pattern, "EMBER"):
            if shown != ".":
                self.assertEqual(shown, gold_letter)

    def test_repeated_proposals_get_a_capped_bonus(self) -> None:
        solver = SweepFillSolver(self.puzzle, FixtureModel({}), OFFLINE)
        for _ in range(6):
            solver._propose("1A", "EAR", 0.6)
        solver._propose("1A", "EAR", 1.0)
        self.assertAlmostEqual(solver.cands["1A"]["EAR"], 1.0 + 0.15 * 3)

    def test_parse_json_object_tolerates_prefix_and_suffix(self) -> None:
        self.assertEqual(
            parse_json_object('Sure! {"answers": {"1A": ["EAR"]}} hope this helps'),
            {"answers": {"1A": ["EAR"]}},
        )
        self.assertIsNone(parse_json_object("no json here"))
        self.assertIsNone(parse_json_object("[1, 2, 3]"))

    def test_recoverable_model_failure_is_skipped_not_fatal(self) -> None:
        class FlakyModel(FixtureModel):
            def complete_json(self, system, user, *, thinking=False, max_tokens=4000, model=None):
                if self.calls == 0:
                    self.calls += 1
                    return JsonReply(data=None, error="timeout: simulated", model="flaky")
                return super().complete_json(system, user, thinking=thinking, max_tokens=max_tokens, model=model)

        model = FlakyModel(self.fixture.candidate_bank)
        result = solve_with_sweep_fill(self.puzzle, model, OFFLINE)

        self.assertEqual(result.status, "complete")
        self.assertEqual(result.grid, self.fixture.solution)
        self.assertEqual(result.model_calls, 2)
        self.assertFalse(result.events[1].accepted)  # the sweep produced nothing

    def test_non_recoverable_model_error_stops_with_error_status(self) -> None:
        class BrokenModel:
            name = "broken"

            def complete_json(self, system, user, *, thinking=False, max_tokens=4000, model=None):
                raise ModelError("Nebius returned HTTP 401: bad key", category="http")

        result = solve_with_sweep_fill(self.puzzle, BrokenModel(), OFFLINE)
        self.assertEqual(result.status, "error")
        self.assertIn("HTTP 401", result.error or "")

    def test_call_budget_stops_the_solver(self) -> None:
        model = FixtureModel(self.fixture.candidate_bank)
        config = SolveConfig(strong_model=None, reasoning_model=None, clues_per_call=3, max_model_calls=1)
        result = solve_with_sweep_fill(self.puzzle, model, config)

        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(result.model_calls, 1)
        self.assertIn("budget", result.error or "")
        self.assertGreater(len(result.unverified_entries), 0)

    def test_run_history_records_metrics_without_puzzle_content(self) -> None:
        result = solve_with_sweep_fill(self.puzzle, FixtureModel(self.fixture.candidate_bank), OFFLINE)
        record = build_run_record(
            puzzle=self.puzzle,
            solution=self.fixture.solution,
            result=result,
            provider="fixture",
            model="fixture-candidate-bank",
            strategy="sweep-fill",
            configuration={"clues_per_call": 26, "candidates_per_entry": 5},
        )

        self.assertEqual(record["review"], "pass")
        self.assertTrue(record["self_reported_complete"])
        self.assertEqual(record["model_calls"], result.model_calls)
        serialized = json.dumps(record)
        self.assertNotIn(next(iter(self.puzzle.clues.values())), serialized)
        self.assertNotIn("solution", serialized)
        self.assertNotIn("grid", serialized)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "nested" / "runs.jsonl"
            append_run_record(path, record)
            append_run_record(path, record)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["run_id"], record["run_id"])

    def test_audit_confirmed_correction_is_adopted_by_the_fill(self) -> None:
        class AuditingModel(FixtureModel):
            def complete_json(self, system, user, *, thinking=False, max_tokens=4000, model=None):
                if "Proposed corrections" in user:
                    self.joint_prompt = user
                    return JsonReply(data={"accept": ["1A"], "reject": [], "alternatives": []}, model="audit")
                if "current fill" in user:
                    return JsonReply(data={"flags": [{"id": "1A", "answer": "EAR", "reason": "lend an ear"}]}, model="audit")
                return super().complete_json(system, user, thinking=thinking, max_tokens=max_tokens, model=model)

        model = AuditingModel({"1A": ("EAR",), "3D": ("RESIN",)})  # what the re-ask proposes
        solver = SweepFillSolver(self.puzzle, model, SolveConfig(strong_model=None, reasoning_model=None, audit_model="audit"))
        for entry_id, entry in self.entries.items():
            gold = "".join(self.fixture.solution[row][col] for row, col in entry.cells)
            solver.cands[entry_id] = {gold: 1.0}
        solver.cands["1A"] = {"EAT": 1.0}    # wrong, and mutually consistent with...
        solver.cands["3D"] = {"TESIN": 1.0}  # ...a wrong crossing (gold RESIN)
        grid, assigned = solver.fill()
        self.assertEqual(solver.pattern(grid, "1A"), "EAT")

        touched, grid, assigned = solver.audit_round(grid, assigned)

        self.assertEqual(touched, 2)  # the flagged entry and its changed crossing
        self.assertIn("3D", model.joint_prompt)  # the crossing effect TESIN -> RESIN was shown for confirmation
        self.assertEqual(solver.pattern(grid, "1A"), "EAR")
        self.assertEqual(solver.pattern(grid, "3D"), "RESIN")
        self.assertEqual(["".join(row) for row in grid], list(self.fixture.solution))

if __name__ == "__main__":
    unittest.main()
