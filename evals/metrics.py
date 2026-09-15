from __future__ import annotations

from typing import Any

from crossword_agent.puzzle import derive_entries, entry_pattern
from crossword_agent.types import Puzzle, SolveResult


def score_result(
    puzzle: Puzzle, solution: tuple[str, ...], result: SolveResult
) -> dict[str, Any]:
    """Deterministic metrics against the hidden gold grid. No model is involved."""

    entries = derive_entries(puzzle)
    open_cells = [
        (row, col)
        for row, solution_row in enumerate(solution)
        for col, cell in enumerate(solution_row)
        if cell != "#"
    ]
    correct_letters = sum(
        result.grid[row][col] == solution[row][col] for row, col in open_cells
    )
    correct_entries = sum(
        entry_pattern(entry, result.grid) == entry_pattern(entry, solution)
        for entry in entries.values()
    )
    grid_shape_valid = len(result.grid) == len(solution) and all(
        len(actual) == len(expected) and all(
            (actual_cell == "#") == (expected_cell == "#")
            for actual_cell, expected_cell in zip(actual, expected)
        )
        for actual, expected in zip(result.grid, solution)
    )
    exact_puzzle = result.grid == solution
    self_reported_complete = result.status == "complete"
    # Evaluation-side diagnostic only (contains answers): which entries were wrong, as id:got->want.
    wrong_entries = [
        f"{entry.id}:{entry_pattern(entry, result.grid)}->{entry_pattern(entry, solution)}"
        for entry in entries.values()
        if entry_pattern(entry, result.grid) != entry_pattern(entry, solution)
    ]
    return {
        "wrong_entries": wrong_entries,
        "letter_accuracy": correct_letters / len(open_cells),
        "entry_accuracy": correct_entries / len(entries),
        "exact_puzzle": exact_puzzle,
        "review": "pass" if exact_puzzle else "fail",
        "valid_grid_shape": grid_shape_valid,
        "status": result.status,
        # Did the solver's own "done" claim match reality? (precision of verified completion)
        "self_reported_complete": self_reported_complete,
        "self_report_correct": self_reported_complete and exact_puzzle,
        "unverified_entries": len(result.unverified_entries),
        "turns": len(result.events),
        "model_calls": result.model_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "unknown_cost_calls": result.unknown_cost_calls,
        "duration_ms": result.duration_ms,
        "revisions": sum(event.revisions for event in result.events),
    }
