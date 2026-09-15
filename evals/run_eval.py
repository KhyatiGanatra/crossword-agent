"""Evaluate the sweep-fill solver over a set of puzzle fixtures.

Gold solutions are loaded here and compared only after each solve; the solver never sees them.
Every run appends a metric-only record to the run log, and the summary reports exact-solve rate,
accuracy, cost, latency, and how often the solver's own "complete" claim was correct.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean

from crossword_agent.cli import (
    STRATEGY,
    add_solver_arguments,
    config_from_args,
    configuration_record,
    require_candidate_bank,
)
from crossword_agent.model import FixtureModel, ModelError, NebiusModel
from crossword_agent.puzzle import load_fixture
from crossword_agent.run_history import append_run_record, build_run_record
from crossword_agent.sweep_solver import solve_with_sweep_fill

from .metrics import score_result

REPO_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the crossword solver")
    parser.add_argument("puzzles", nargs="*", type=Path, help="Fixture files or directories (default: data/dev)")
    add_solver_arguments(parser)
    parser.add_argument("--repeats", type=int, default=1, help="Runs per puzzle; the solver is stochastic")
    parser.add_argument("--workers", type=int, default=1, help="Puzzles solved concurrently (each solve is network-bound)")
    parser.add_argument("--output", type=Path, help="Write one JSON line per run to this file")
    parser.add_argument(
        "--run-log",
        type=Path,
        default=Path(os.getenv("CROSSWORD_RUN_LOG", "eval-results/model-runs.jsonl")),
    )
    parser.add_argument("--no-run-log", action="store_true")
    return parser


def _fixture_paths(arguments: list[Path]) -> list[Path]:
    if not arguments:
        return sorted((REPO_ROOT / "data" / "dev").glob("*.json"))
    paths: list[Path] = []
    for argument in arguments:
        if argument.is_dir():
            paths.extend(sorted(argument.glob("*.json")))
        else:
            paths.append(argument)
    return paths


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = _fixture_paths(args.puzzles)
    config = config_from_args(args)
    try:
        nebius = (
            NebiusModel(model=args.model, timeout_seconds=args.request_timeout)
            if args.provider == "nebius"
            else None
        )
    except ModelError as exc:
        print(f"Configuration error: {exc}")
        return 2

    fixtures = []
    for path in paths:
        fixture = load_fixture(path)
        try:
            if nebius is None:
                require_candidate_bank(fixture)
        except ValueError as exc:
            print(f"Configuration error: {exc}")
            return 2
        fixtures.append(fixture)

    def run_one(job: tuple) -> tuple[dict, dict]:
        fixture, repeat = job
        model = nebius if nebius is not None else FixtureModel(fixture.candidate_bank)
        result = solve_with_sweep_fill(fixture.puzzle, model, config)
        metrics = score_result(fixture.puzzle, fixture.solution, result)
        record = {
            "puzzle_id": fixture.puzzle.id,
            "repeat": repeat + 1,
            "provider": model.name,
            "strategy": STRATEGY,
            **metrics,
            "error": result.error,
        }
        history = build_run_record(
            puzzle=fixture.puzzle,
            solution=fixture.solution,
            result=result,
            provider=args.provider,
            model=model.name,
            strategy=STRATEGY,
            configuration=configuration_record(args),
        )
        return record, history

    jobs = [(fixture, repeat) for fixture in fixtures for repeat in range(args.repeats)]
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for record, history in executor.map(run_one, jobs):
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if not args.no_run_log:
                try:
                    append_run_record(args.run_log, history)
                except OSError as exc:
                    print(f"run log warning: {exc}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")

    completed = [record for record in records if record["self_reported_complete"]]
    costs = [record["estimated_cost_usd"] for record in records]
    summary = {
        "puzzles": len(paths),
        "runs": len(records),
        "exact_puzzle_rate": mean(record["exact_puzzle"] for record in records),
        "mean_letter_accuracy": mean(record["letter_accuracy"] for record in records),
        "mean_entry_accuracy": mean(record["entry_accuracy"] for record in records),
        "complete_claim_rate": len(completed) / len(records) if records else 0.0,
        "complete_claim_precision": (
            mean(record["exact_puzzle"] for record in completed) if completed else None
        ),
        "mean_model_calls": mean(record["model_calls"] for record in records),
        "mean_reasoning_tokens": mean(record["reasoning_tokens"] for record in records),
        "mean_duration_s": mean(record["duration_ms"] for record in records) / 1000,
        "mean_cost_usd": mean(costs) if all(cost is not None for cost in costs) else None,
        "total_cost_usd": sum(costs) if all(cost is not None for cost in costs) else None,
        "total_unknown_cost_calls": sum(record["unknown_cost_calls"] for record in records),
    }
    print("summary " + json.dumps(summary, sort_keys=True))
    return 0 if all(record["exact_puzzle"] for record in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
