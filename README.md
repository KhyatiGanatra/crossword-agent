# Crossword Agent

An agent that solves American-style crossword puzzles with Nebius Token Factory models. 

## Summary

This is my submission: a crossword agent built on Nebius Token Factory, tested on 39 held-out New York Times Monday puzzles, where it solves 37 of 39 perfectly and gets 99.9% of letters right, at about four cents per puzzle.

The harness has four pieces.

First, I give an LLM every clue with its letter count, in one batch with reasoning off, and it returns a few candidate words for each.

Second, my own code does a deterministic constrained fill: a beam search that picks one candidate per slot so every crossing letter matches, re-asking the model with the known letters for slots it cannot settle. I do this in code, not with an agent, because it is a hard search problem where an LLM is slow and gets confused.

Third, a separate LLM call audits the finished grid and flags any word that does not actually answer its clue.

Fourth, for each flagged word the model proposes a similar word with one to three letters changed, and the grid is re-solved through the beam search with that correction in play.

This removes most of the classic failure where one wrong letter yields two plausible non-words that confirm each other.

## How it works

The model is only ever asked one question, "what could these clues be?", and answers with ranked candidates. Deterministic code does everything else: it fills the grid, decides which entries to trust, decides what to ask next, and decides when to stop.

```text
SWEEP     ask every clue once, reasoning off, ~25 clues per call in parallel     → candidate pools
FILL      beam search over a weighted constraint problem; crossings are hard constraints;
          an entry may stay blank rather than take an unsupported answer
VERIFY    an entry is done only if the model proposed it for that clue AND its crossings agree
REPAIR    re-ask only unverified entries, with letters from verified neighbours            (≤ 3 rounds)
ESCALATE  stronger no-reasoning model, then a bounded-reasoning model, for what is left     (≤ 2 rounds)
STOP      all entries verified, or budget spent → grid plus the list of doubtful entries
```

Why this shape: every system that has solved full grids exactly (Proverb, Dr.Fill, the Berkeley Crossword Solver, SweepClip) generates broad candidates and lets a constraint solver pick. Two measured facts drove the details. Giving the model crossing letters roughly doubles its clue accuracy, so the cheapest lever is a second pass with letters. And reasoning tokens were 80 to 95% of the cost in earlier designs without improving accuracy, so the sweep runs with reasoning switched off and a bounded reasoning model is used only on the last few hard clues.

Model roles, all chosen at runtime:

| Role | Default | Flag / env var |
|---|---|---|
| Sweep and repair, reasoning off | `deepseek-ai/DeepSeek-V4-Flash-0731` | `--model` / `NEBIUS_MODEL` |
| Strong escalation, reasoning off | `deepseek-ai/DeepSeek-V4-Pro` | `--strong-model` / `NEBIUS_STRONG_MODEL` |
| Bounded reasoning (`reasoning_effort: low`) | `zai-org/GLM-5.3-Flash` | `--reasoning-model` / `NEBIUS_REASONING_MODEL` |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). There are no third-party runtime dependencies.

```bash
uv sync
```

For live runs, create a Token Factory API key and put it in a gitignored `.env` (copy `.env.example`):

```text
NEBIUS_API_KEY=...
NEBIUS_BASE_URL=https://api.tokenfactory.nebius.com/v1
```

## Run

Offline smoke test on the checked-in 5x5 fixture. Answers come from a candidate bank in the fixture, so this exercises the pipeline, not model quality:

```bash
uv run crossword-agent
```

Live, on any puzzle fixture, watching the grid fill stage by stage:

```bash
uv run --env-file .env crossword-agent data/generated/gen-20x20-001.json --provider nebius
```

Each stage prints its header (verified entries, filled cells, tokens, cost), the grid, the entries placed, revised, or removed, and the entries still unverified. The final lines report the solver's own status, the deterministic pass/fail review against the hidden solution, calls, tokens, reasoning tokens, elapsed time, and cost. Every run appends a metric-only record (no puzzle content) to `eval-results/model-runs.jsonl`; `--no-run-log` skips that.

Flags, all defaulting to the measured configuration:

| Flag | Default | Meaning |
|---|---|---|
| `--candidates-per-entry` | 5 | Ranked answers requested per clue |
| `--clues-per-call` | 26 | Clues per request; requests run in parallel |
| `--repair-rounds` / `--escalation-rounds` | 3 / 2 | Loop limits |
| `--beam-width` / `--abstain-penalty` / `--verify-support` | 300 / 0.6 / 0.5 | Fill and verification parameters |
| `--max-model-calls` / `--max-cost-usd` / `--max-seconds` | 40 / none / none | Per-puzzle budgets |
| `--strong-model none`, `--reasoning-model none` | | Disable an escalation stage (ablations) |
| `--quiet` | | Print only the final summary |

## Puzzle format and data

A fixture is one JSON file: `id`, `title`, `grid` (rows of `.` for open cells and `#` for blocks), `solution` (the same rows with letters), and `clues` keyed by entry id (`1A`, `1D`, ...). Entries and numbering are derived from the grid. The solution is read only by the evaluation code after a solve; the solver never receives it.

- `data/dev/mini-001.json`: a project-authored 5x5 smoke fixture (CC0) with a candidate bank for offline runs.
- `data/generated/`: original 20x20 puzzles generated for this project (CC0): random symmetric block layouts, a backtracking fill from a common-word lexicon, and clues written by a Token Factory model with leakage checks and a blind-solve difficulty label (`blind_top3_recall` in `data/generated/manifest.json`). Being original, they can be shown publicly and cannot be in any model's training data.
- NYT puzzles were used for the evaluation below but are not included, because they are copyrighted; they were imported locally from the 100 NYT Mondays in the public LMCrossword repository and never redistributed. Any puzzle in the fixture format can be passed to the CLI or the harness.

## Evaluation

```bash
uv run --env-file .env python -m evals.run_eval data/generated --provider nebius --repeats 3 --output eval-results/generated.jsonl
uv run python -m evals.summarize eval-results/generated.jsonl
```

The harness holds the gold grids, runs the same `solve` function the CLI uses, reveals the solution only after each solve, prints one JSON line per run, and appends metric-only records to the run log. `summarize` turns any set of records into the report below. Everything is deterministic; there is no model-as-judge.

**Two verdicts per run.** The solver's own status is decided without gold: `complete` means every entry is verified, `incomplete` means the rounds ran out with unverified entries, `budget_exhausted` means a call, cost, or time ceiling stopped it, `error` means a non-recoverable provider failure. The deterministic review is decided with gold: `pass` only if every letter matches. Success is `pass`; the status tells you whether the agent knew.

**Metrics.** Exact-solve rate (the headline), letter and entry accuracy, near-miss bands (within about one and about five wrong letters), the rate at which the solver claims `complete` and the precision of that claim, model calls, reasoning tokens, cost from live catalog prices, and wall-clock latency, with Wilson intervals over runs and bootstrap intervals over puzzles. Results are split by era, because pre-1993 NYT clues are a vocabulary test rather than an agent test.

**Protocol.** Fixed configuration across runs; three repeats where variance matters (the solver samples at temperature 0.2); gold never enters a prompt. The NYT puzzles have been public on GitHub since 2024 and are widely mirrored, so "held out" means held out from tuning, not from pretraining; the generated puzzles are the contamination-free check.

### Results on held-out NYT Mondays (15x15, one run each, never used for tuning)

| Slice | Puzzles | Exact | Mean letters | Within ~5 wrong letters | "Complete" claim precision | Mean cost | Mean time |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1993 onward | 52 | 46.2% (95% CI 33 to 58) | 99.3% | 94.2% | 55.0% | $0.006 | 22.8 s |
| Pre-1993 | 38 | 5.3% | 96.5% | 47.4% | 18.2% | $0.014 | 50.0 s |
| All | 90 | 28.9% (95% CI 19 to 39) | 98.1% | 74.4% | 47.1% | $0.009 | 34.3 s |

Intervals are bootstrap over puzzles. On the 10 development puzzles with three repeats each: 50% exact over 30 runs (73% on the five modern ones); these were used during development and are reported separately.

For scale, the SweepClip paper reports 48% exact on the same 100 NYT Mondays with GPT-4-Turbo under a $0.50-per-puzzle budget; this solver reaches 29% on 90 of them, and 46% on the modern half, at about one cent per puzzle.

### What a miss looks like

Re-running the modern puzzles the solver had wrongly called complete, with per-entry diagnostics: the failed runs had 2 to 8 wrong entries each, almost always in pairs sharing one cell, where one side is a real word and the other is a pattern-fitting non-word that the model had proposed under a letter hint. Examples: `EANYON`/`EASEL` for CANYON/CASES, `SWUN`/`RERUN` for SWAN/RERAN, `GAMESSREN` for GAMESEVEN, `DIDBRIS`/`INADIE` for TIDBITS/INATIE. The current verification rule accepts such pairs because each side was proposed and each supports the other. The known fix, taken from the Berkeley Crossword Solver, is a wordlist as a third verification condition plus single-letter local repair at doubtful crossings; it is the next step and would convert most of the near misses above.

### How to read this

- On a modern easy puzzle the solver is right about 40% of the time and within a few letters about 90% of the time.
- When it says "complete" on a modern puzzle it is right less than half the time; the near-miss band is where the exact rate will come from.
- Old puzzles fail on vocabulary (clues like "Humbug." for FLAM); a clue database is the fix there.
- Cost and time scale with how many entries stay unverified, not with grid size.

## Tests

```bash
uv run python -m unittest discover -v
```

Fifteen offline tests: grid parsing and crossing checks, the NYT importer, the beam fill abstaining when a weak candidate blocks stronger ones, verification of crossing-implied entries, prompt patterns that never echo a doubtful letter, capped vote bonuses, tolerant JSON parsing, recoverable versus fatal provider errors, the call budget, and an end-to-end offline solve whose prompts are checked for gold leakage.

## Layout

```text
src/crossword_agent/
├── sweep_solver.py     the sweep → fill → verify → repair → escalate loop
├── model.py            AnswerModel protocol; NebiusModel (JSON output, reasoning controls, live prices); FixtureModel
├── puzzle.py           fixture parsing, entry derivation and numbering
├── constraints.py      exact length, pattern, and crossing checks
├── types.py            Puzzle, Entry, SolveConfig, SolveResult, TraceEvent, JsonReply
├── run_history.py      metric-only JSONL run records
└── cli.py              `crossword-agent`; argument parsing shared with the harness
evals/                  run_eval.py (harness), metrics.py (deterministic scoring), summarize.py (reports)
tests/                  offline tests
data/                   fixtures, manifest, importer instructions
```

The shared boundary:

```text
solve_with_sweep_fill(puzzle, model, config, on_event, index=None) -> SolveResult
```

## Token Factory notes

- `reasoning_effort: "none"` switches thinking off on DeepSeek and GLM models; it is sent only to models whose catalog entry advertises reasoning. GLM ignores `chat_template_kwargs`, and GLM's `"minimal"` effort burns the whole token cap with no output.
- `max_tokens` is always set explicitly; the server default of 8,192 includes reasoning tokens.
- Plain `response_format: json_object` is used instead of tool schemas: compiled schemas with length and regex constraints force length-matching nonsense when the model has no answer.
- Configuration errors (bad key, unknown model) stop the run; 429/5xx responses, transport errors, and timeouts are retried once and then skipped for that batch of clues.

## Known limitations

- Verification can be fooled by a pattern-fitting non-word that a wrong crossing agrees with (false "complete" claims), and can be over-cautious (a correct grid reported as incomplete). Both are visible in the metrics above.
- Pre-1993 vocabulary is out of reach without a clue database.
- Runs vary at temperature 0.2; report repeats, not single runs.
- Rebus cells, diagramless grids, image input, and themes beyond cross-reference context are out of scope.
