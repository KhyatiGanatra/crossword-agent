from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .puzzle import derive_entries, entry_pattern
from .types import Puzzle, SolveResult


def build_run_record(
    *,
    puzzle: Puzzle,
    solution: tuple[str, ...],
    result: SolveResult,
    provider: str,
    model: str,
    strategy: str,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Metric-only record of one solve: no clues, grids, answers, or credentials."""

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
    exact = result.grid == solution
    return {
        "run_id": uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "puzzle_id": puzzle.id,
        "provider": provider,
        "model": model,
        "strategy": strategy,
        "configuration": configuration,
        "status": result.status,
        "review": "pass" if exact else "fail",
        "exact_puzzle": exact,
        "letter_accuracy": correct_letters / len(open_cells),
        "entry_accuracy": correct_entries / len(entries),
        "self_reported_complete": result.status == "complete",
        "unverified_entries": len(result.unverified_entries),
        "turns": len(result.events),
        "model_calls": result.model_calls,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens,
        "estimated_cost_usd": result.estimated_cost_usd,
        "unknown_cost_calls": result.unknown_cost_calls,
        "duration_ms": result.duration_ms,
        "error": result.error,
    }


def append_run_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
