"""Summarise sweep-fill evaluation records.

Reads JSONL written by ``evals.run_eval --output`` (or the run log) and prints the numbers the
evaluation methodology asks for: exact-solve rate with confidence intervals, the near-miss
distribution, how often the solver's own "complete" claim was right, cost and latency, and a
breakdown by puzzle era and by puzzle. Deterministic; no model is involved.

Usage:
  uv run python -m evals.summarize eval-results/test-part1.jsonl eval-results/test-part2.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load(paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("strategy") not in (None, "sweep-fill"):
                continue
            if str(record.get("provider", "")).startswith("fixture"):
                continue
            records.append(record)
    return records


def era(puzzle_id: str) -> str:
    if puzzle_id.startswith("nyt-"):
        year = int(puzzle_id[4:8])
        return "pre-1993" if year < 1993 else "1993+"
    return "other"


def wilson(successes: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    if trials == 0:
        return (0.0, 0.0)
    p = successes / trials
    denominator = 1 + z * z / trials
    centre = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_by_puzzle(records: list[dict], key: str, samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% interval for the mean per-puzzle rate, resampling puzzles (repeats stay together)."""

    by_puzzle: dict[str, list[float]] = defaultdict(list)
    for record in records:
        by_puzzle[record["puzzle_id"]].append(float(record[key]))
    puzzle_means = [statistics.mean(values) for values in by_puzzle.values()]
    if len(puzzle_means) < 2:
        value = puzzle_means[0] if puzzle_means else 0.0
        return (value, value)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draw = [rng.choice(puzzle_means) for _ in puzzle_means]
        draws.append(statistics.mean(draw))
    draws.sort()
    return (draws[int(0.025 * samples)], draws[int(0.975 * samples) - 1])


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def block(title: str, records: list[dict]) -> None:
    n = len(records)
    if n == 0:
        print(f"\n## {title}: no runs")
        return
    puzzles = {record["puzzle_id"] for record in records}
    exact = [record for record in records if record["exact_puzzle"]]
    complete = [record for record in records if record.get("self_reported_complete") or record.get("status") == "complete"]
    false_claims = [record for record in complete if not record["exact_puzzle"]]
    cautious = [record for record in exact if record not in complete]
    letters = [record["letter_accuracy"] for record in records]
    costs = [record["estimated_cost_usd"] for record in records if record.get("estimated_cost_usd") is not None]
    seconds = [record["duration_ms"] / 1000 for record in records]
    calls = [record["model_calls"] for record in records]
    reasoning = [record.get("reasoning_tokens", 0) for record in records]
    low, high = wilson(len(exact), n)
    boot_low, boot_high = bootstrap_by_puzzle(records, "exact_puzzle")
    print(f"\n## {title}")
    print(f"runs: {n} over {len(puzzles)} puzzles")
    print(
        f"exact-solve rate: {len(exact) / n:.1%}  (Wilson 95% over runs {low:.1%}–{high:.1%}; "
        f"bootstrap 95% over puzzles {boot_low:.1%}–{boot_high:.1%})"
    )
    print(
        f"letters: mean {statistics.mean(letters):.1%}, median {statistics.median(letters):.1%}, "
        f"min {min(letters):.1%}"
    )
    near1 = sum(1 for value in letters if value >= 0.995)
    near5 = sum(1 for value in letters if value >= 0.974)
    print(
        f"near misses: {near1 / n:.1%} of runs within ~1 wrong letter, {near5 / n:.1%} within ~5 "
        f"(thresholds 99.5% / 97.4% of ~190 cells)"
    )
    precision = len(complete) - len(false_claims)
    print(
        f"claims: 'complete' claimed in {len(complete) / n:.1%} of runs; precision "
        f"{(precision / len(complete)) if complete else 0:.1%}; false claims {len(false_claims)}; "
        f"exact-but-cautious {len(cautious)}"
    )
    print(
        f"calls: mean {statistics.mean(calls):.1f}, p90 {percentile(calls, 0.9):.0f} | reasoning tokens: "
        f"mean {statistics.mean(reasoning):.0f}"
    )
    if costs:
        print(
            f"cost: mean ${statistics.mean(costs):.4f}, p90 ${percentile(costs, 0.9):.4f}, "
            f"max ${max(costs):.4f}, total ${sum(costs):.3f}"
        )
    print(
        f"latency: mean {statistics.mean(seconds):.1f} s, p50 {percentile(seconds, 0.5):.1f} s, "
        f"p90 {percentile(seconds, 0.9):.1f} s, max {max(seconds):.1f} s"
    )
    statuses = defaultdict(int)
    for record in records:
        statuses[record.get("status", "?")] += 1
    print("statuses: " + ", ".join(f"{status} {count}" for status, count in sorted(statuses.items())))


def per_puzzle(records: list[dict]) -> None:
    by_puzzle: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_puzzle[record["puzzle_id"]].append(record)
    print("\n## Per puzzle")
    print("| puzzle | era | runs | exact | mean letters | mean calls | mean cost | mean s | statuses |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---|")
    for puzzle_id in sorted(by_puzzle):
        rows = by_puzzle[puzzle_id]
        costs = [row["estimated_cost_usd"] for row in rows if row.get("estimated_cost_usd") is not None]
        statuses = ",".join(sorted({row.get("status", "?") for row in rows}))
        print(
            f"| {puzzle_id} | {era(puzzle_id)} | {len(rows)} | {sum(row['exact_puzzle'] for row in rows)} | "
            f"{statistics.mean(row['letter_accuracy'] for row in rows):.1%} | "
            f"{statistics.mean(row['model_calls'] for row in rows):.1f} | "
            f"${statistics.mean(costs):.4f} | {statistics.mean(row['duration_ms'] for row in rows) / 1000:.1f} | {statuses} |"
            if costs
            else f"| {puzzle_id} | {era(puzzle_id)} | {len(rows)} | {sum(row['exact_puzzle'] for row in rows)} | "
            f"{statistics.mean(row['letter_accuracy'] for row in rows):.1%} | "
            f"{statistics.mean(row['model_calls'] for row in rows):.1f} | n/a | "
            f"{statistics.mean(row['duration_ms'] for row in rows) / 1000:.1f} | {statuses} |"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise sweep-fill evaluation records")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--no-per-puzzle", action="store_true")
    args = parser.parse_args(argv)
    records = load(args.paths)
    if not records:
        print("no sweep-fill records found")
        return 1
    block("All runs", records)
    for label in ("1993+", "pre-1993", "other"):
        subset = [record for record in records if era(record["puzzle_id"]) == label]
        if subset:
            block(f"Era {label}", subset)
    if not args.no_per_puzzle:
        per_puzzle(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
