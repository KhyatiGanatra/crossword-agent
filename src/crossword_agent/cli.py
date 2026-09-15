from __future__ import annotations

import argparse
import os
from pathlib import Path

from .model import FixtureModel, ModelError, NebiusModel
from .puzzle import load_fixture
from .run_history import append_run_record, build_run_record
from .sweep_solver import solve_with_sweep_fill
from .types import PuzzleFixture, SolveConfig, TraceEvent

DEFAULT_PUZZLE = Path(__file__).resolve().parents[2] / "data" / "dev" / "mini-001.json"
STRATEGY = "sweep-fill"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _optional_model(value: str) -> str | None:
    return None if value.strip().lower() in {"", "none", "off"} else value


def _print_grid(grid: tuple[str, ...]) -> None:
    for row in grid:
        print("  " + " ".join("■" if cell == "#" else cell for cell in row), flush=True)


def _show_event(event: TraceEvent) -> None:
    marker = "✓" if event.accepted else "✗"
    print(f"\n[{event.turn:02d}] {marker} {event.action}: {event.message}", flush=True)
    if event.input_tokens or event.output_tokens:
        cost = (
            f"${event.estimated_cost_usd:.5f}" if event.estimated_cost_usd is not None else "unknown"
        )
        print(
            f"  usage: {event.input_tokens} input + {event.output_tokens} output tokens, cost {cost}",
            flush=True,
        )
    _print_grid(event.grid)
    if event.alternatives:
        print("  placed: " + ", ".join(event.alternatives), flush=True)
    for part in event.rationale.split("; ") if event.rationale else []:
        print("  " + part, flush=True)


def add_solver_arguments(parser: argparse.ArgumentParser) -> None:
    """Solver and provider flags shared by the CLI and the evaluation harness."""

    parser.add_argument("--provider", choices=("fixture", "nebius"), default="fixture")
    parser.add_argument(
        "--model",
        default=os.getenv("NEBIUS_MODEL", "moonshotai/Kimi-K3"),
        help="Sweep and repair model, run without reasoning (default: Kimi K3)",
    )
    parser.add_argument(
        "--strong-model",
        type=_optional_model,
        default=os.getenv("NEBIUS_STRONG_MODEL", "deepseek-ai/DeepSeek-V4-Pro"),
        help="Stronger no-reasoning model for unverified entries; 'none' disables (default: DeepSeek V4 Pro)",
    )
    parser.add_argument(
        "--reasoning-model",
        type=_optional_model,
        default=os.getenv("NEBIUS_REASONING_MODEL", "zai-org/GLM-5.3-Flash"),
        help="Bounded-reasoning model (reasoning_effort low) for the last unverified entries; 'none' disables (default: GLM-5.3-Flash)",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=float(os.getenv("NEBIUS_REQUEST_TIMEOUT", "180")),
        help="Seconds to wait for each Token Factory response (default: 180)",
    )
    parser.add_argument(
        "--audit-model",
        type=_optional_model,
        default=os.getenv("NEBIUS_AUDIT_MODEL", "zai-org/GLM-5.3-Flash"),
        help="Reasoning model that audits clue fit and proposes small corrections; 'none' disables (default: GLM-5.3-Flash)",
    )
    parser.add_argument("--audit-rounds", type=int, default=2)
    parser.add_argument("--candidates-per-entry", type=_positive_int, default=5)
    parser.add_argument("--clues-per-call", type=_positive_int, default=26)
    parser.add_argument("--repair-rounds", type=int, default=3)
    parser.add_argument("--escalation-rounds", type=int, default=2)
    parser.add_argument("--beam-width", type=_positive_int, default=300)
    parser.add_argument("--abstain-penalty", type=float, default=0.6)
    parser.add_argument("--verify-support", type=float, default=0.5)
    parser.add_argument("--max-model-calls", type=_positive_int, default=40, help="Paid-call ceiling per puzzle")
    parser.add_argument("--max-cost-usd", type=float, default=None, help="Stop before exceeding this cost per puzzle")
    parser.add_argument("--max-seconds", type=float, default=None, help="Stop before exceeding this wall clock per puzzle")


def config_from_args(args: argparse.Namespace) -> SolveConfig:
    return SolveConfig(
        strong_model=args.strong_model,
        reasoning_model=args.reasoning_model,
        candidates_per_entry=args.candidates_per_entry,
        clues_per_call=args.clues_per_call,
        repair_rounds=args.repair_rounds,
        escalation_rounds=args.escalation_rounds,
        audit_model=args.audit_model,
        audit_rounds=args.audit_rounds,
        beam_width=args.beam_width,
        abstain_penalty=args.abstain_penalty,
        verify_support=args.verify_support,
        max_model_calls=args.max_model_calls,
        max_cost_usd=args.max_cost_usd,
        max_seconds=args.max_seconds,
    )


def configuration_record(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model": args.model,
        "strong_model": args.strong_model,
        "reasoning_model": args.reasoning_model,
        "candidates_per_entry": args.candidates_per_entry,
        "clues_per_call": args.clues_per_call,
        "repair_rounds": args.repair_rounds,
        "escalation_rounds": args.escalation_rounds,
        "audit_model": args.audit_model,
        "audit_rounds": args.audit_rounds,
        "beam_width": args.beam_width,
        "abstain_penalty": args.abstain_penalty,
        "verify_support": args.verify_support,
        "max_model_calls": args.max_model_calls,
        "max_cost_usd": args.max_cost_usd,
        "max_seconds": args.max_seconds,
        "request_timeout_seconds": args.request_timeout,
    }


def require_candidate_bank(fixture: PuzzleFixture) -> None:
    """The offline provider only makes sense for fixtures that ship a candidate bank."""

    if not fixture.candidate_bank:
        raise ValueError(
            f"fixture {fixture.puzzle.id} has no candidate_bank, so the offline fixture provider "
            "cannot answer its clues; run with --provider nebius"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Solve a crossword with the sweep-fill agent and watch the grid fill stage by stage"
    )
    parser.add_argument("puzzle", nargs="?", type=Path, default=DEFAULT_PUZZLE)
    add_solver_arguments(parser)
    parser.add_argument(
        "--run-log",
        type=Path,
        default=Path(os.getenv("CROSSWORD_RUN_LOG", "eval-results/model-runs.jsonl")),
        help="Append model and outcome metrics to this JSONL file",
    )
    parser.add_argument("--no-run-log", action="store_true", help="Do not append this solve to the run log")
    parser.add_argument("--quiet", action="store_true", help="Print only the final summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    fixture = load_fixture(args.puzzle)
    try:
        if args.provider == "fixture":
            require_candidate_bank(fixture)
            model = FixtureModel(fixture.candidate_bank)
        else:
            model = NebiusModel(model=args.model, timeout_seconds=args.request_timeout)
    except (ModelError, ValueError) as exc:
        print(f"Configuration error: {exc}")
        return 2

    print(f"{fixture.puzzle.title} ({fixture.puzzle.id})")
    print(f"Provider: {model.name} | strategy: {STRATEGY}")
    config = config_from_args(args)
    result = solve_with_sweep_fill(
        fixture.puzzle,
        model,
        config,
        on_event=None if args.quiet else _show_event,
    )

    review = "pass" if result.grid == fixture.solution else "fail"
    print(f"\nResult: {result.status} | deterministic review against hidden gold: {review}")
    if args.quiet:
        _print_grid(result.grid)
    print(
        f"Stages: {len(result.events)} | model calls: {result.model_calls} | tokens: "
        f"{result.input_tokens} input + {result.output_tokens} output "
        f"({result.reasoning_tokens} reasoning) | elapsed: {result.duration_ms / 1000:.1f} s"
    )
    if args.provider == "fixture":
        print("Token Factory cost: $0.000000 (offline fixture provider)")
    elif result.estimated_cost_usd is not None:
        qualifier = (
            f" (known subtotal; {result.unknown_cost_calls} call(s) have unknown usage)"
            if result.unknown_cost_calls
            else ""
        )
        print(f"Estimated Token Factory cost: ${result.estimated_cost_usd:.6f}{qualifier}")
    else:
        print(f"Estimated Token Factory cost: unavailable; {result.unknown_cost_calls} call(s) have unknown cost")
    if result.unverified_entries:
        print("Unverified entries: " + ", ".join(result.unverified_entries))
    if result.error:
        print(f"Stop reason: {result.error}")
    if not args.no_run_log:
        record = build_run_record(
            puzzle=fixture.puzzle,
            solution=fixture.solution,
            result=result,
            provider=args.provider,
            model=model.name,
            strategy=STRATEGY,
            configuration=configuration_record(args),
        )
        try:
            append_run_record(args.run_log, record)
            print(f"Run log: {args.run_log}")
        except OSError as exc:
            print(f"Run log warning: {exc}")
    return 0 if review == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
