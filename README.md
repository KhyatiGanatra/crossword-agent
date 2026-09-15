# Crossword Agent

An agent that solves American-style crossword puzzles with open-weight models served by Token Factory.

## How it works

The model is only ever asked one question, "what could these clues be?", and answers with ranked candidates. Deterministic code does everything else: it fills the grid, decides which entries to trust, decides what to ask next, and decides when to stop.

```text
SWEEP     ask every clue once, reasoning off, ~25 clues per call in parallel     → candidate pools
FILL      beam search over a weighted constraint problem; crossings are hard constraints;
          an entry may stay blank rather than take an unsupported answer
VERIFY    an entry is done only if the model proposed it for that clue AND its crossings agree
REPAIR    re-ask only unverified entries, with letters from verified neighbours            (≤ 3 rounds)
ESCALATE  stronger no-reasoning model, then GLM-5.3-Flash at low reasoning effort, for what is left   (≤ 2 rounds)
AUDIT     a reasoning model checks every fill against its clue, proposes small corrections, and confirms
          them with the affected crossings in view; the fill decides whether to adopt them        (≤ 2 rounds)
STOP      all entries verified, or budget spent → grid plus the list of doubtful entries
```

The audit stage exists because nearly every miss is one shared cell where a real word crosses a pattern-fitting non-word that the model proposed under a letter hint (ISAT crossing ONOFFSTITCH). The audit model sees each fill next to its clue, names the ones that do not fit with a same-length replacement, then sees what that replacement does to the crossing entries and confirms or rejects the joint change. A confirmed answer enters the candidate pools with a strong prior and the flagged fills are discounted; the crossings it touches are re-asked with the contested cell blanked, so they count as verified only if the sweep model proposes their new fill independently. The deterministic fill still makes the final choice.

Why this shape: every system that has solved full grids exactly (Proverb, Dr.Fill, the Berkeley Crossword Solver, SweepClip) generates broad candidates and lets a constraint solver pick. Two measured facts drove the details. Giving the model crossing letters roughly doubles its clue accuracy, so the cheapest lever is a second pass with letters. And reasoning tokens dominated cost without improving clue accuracy, so the sweep runs with reasoning switched off and bounded reasoning is used only on the last few hard clues and in the audit.

Model roles, all chosen at runtime:

| Role | Default | Flag / env var |
|---|---|---|
| Sweep and repair, reasoning off | `moonshotai/Kimi-K3` (DeepSeek V4 Flash is the low-cost alternative) | `--model` / `TOKEN_FACTORY_MODEL` |
| Strong escalation, reasoning off | `deepseek-ai/DeepSeek-V4-Pro` | `--strong-model` / `TOKEN_FACTORY_STRONG_MODEL` |
| Bounded reasoning (`reasoning_effort: low`) | `zai-org/GLM-5.3-Flash` | `--reasoning-model` / `TOKEN_FACTORY_REASONING_MODEL` |
| Clue-fit audit (`reasoning_effort: low`) | `zai-org/GLM-5.3-Flash` | `--audit-model` / `TOKEN_FACTORY_AUDIT_MODEL` |

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). There are no third-party runtime dependencies.

```bash
uv sync
```

For live runs, create a Token Factory API key and put it in a gitignored `.env` (copy `.env.example`):

```text
TOKEN_FACTORY_API_KEY=...
TOKEN_FACTORY_BASE_URL=https://api.tokenfactory.nebius.com/v1
```

## Run

Offline smoke test on the checked-in 5x5 fixture. Answers come from a candidate bank in the fixture, so this exercises the pipeline, not model quality:

```bash
uv run crossword-agent
```

Live, on any puzzle in the fixture format, watching the grid fill stage by stage:

```bash
uv run --env-file .env crossword-agent path/to/puzzle.json --provider token-factory
```

Each stage prints its header (verified entries, filled cells, tokens, cost), the grid, the entries placed, revised, or removed, and the entries still unverified. The final lines report the solver's own status, the deterministic pass/fail review against the hidden solution, calls, tokens, reasoning tokens, elapsed time, and cost. Every run appends a metric-only record (no puzzle content) to `eval-results/model-runs.jsonl`; `--no-run-log` skips that.

Flags, all defaulting to the measured configuration:

| Flag | Default | Meaning |
|---|---|---|
| `--candidates-per-entry` | 5 | Ranked answers requested per clue |
| `--clues-per-call` | 26 | Clues per request; requests run in parallel |
| `--repair-rounds` / `--escalation-rounds` / `--audit-rounds` | 3 / 2 / 2 | Loop limits |
| `--beam-width` / `--abstain-penalty` / `--verify-support` | 300 / 0.6 / 0.5 | Fill and verification parameters |
| `--max-model-calls` / `--max-cost-usd` / `--max-seconds` | 40 / none / none | Per-puzzle budgets |
| `--strong-model none`, `--reasoning-model none`, `--audit-model none` | | Disable a stage (ablations) |
| `--quiet` | | Print only the final summary |

## Puzzle format

A puzzle is one JSON file: `id`, `title`, `grid` (rows of `.` for open cells and `#` for blocks), `solution` (the same rows with letters), and `clues` keyed by entry id (`1A`, `1D`, ...). Entries and numbering are derived from the grid. The solution is read only by the evaluation code after a solve; the solver never receives it. The checked-in `data/dev/mini-001.json` is a project-authored 5x5 (CC0) with a candidate bank for offline runs.

The evaluation below used New York Times Monday puzzles imported locally from the public LMCrossword research repository. They are copyrighted and are not included here.

## Evaluation

```bash
uv run --env-file .env python -m evals.run_eval <directory or fixture files> --provider token-factory --repeats 3 --workers 4 --output eval-results/run.jsonl
uv run python -m evals.summarize eval-results/run.jsonl
```

`--workers` solves that many puzzles concurrently (each solve is network-bound; four workers cut a 40-puzzle run from about 15 minutes to about 5).

The harness holds the gold grids, runs the same `solve` function the CLI uses, reveals the solution only after each solve, prints one JSON line per run, and appends metric-only records to the run log. `summarize` turns any set of records into a report. Scoring is deterministic; there is no model-as-judge in the scoring.

**Two verdicts per run.** The solver's own status is decided without gold: `complete` means every entry is verified, `incomplete` means the rounds ran out with unverified entries, `budget_exhausted` means a call, cost, or time ceiling stopped it, `error` means a non-recoverable provider failure. The deterministic review is decided with gold: `pass` only if every letter matches. Success is `pass`; the status tells you whether the agent knew.

**Metrics.** Exact-solve rate (the headline), letter and entry accuracy, near-miss bands (within about one and about five wrong letters), the rate at which the solver claims `complete` and the precision of that claim, model calls, reasoning tokens, cost from live catalog prices, and wall-clock latency, with Wilson intervals over runs and bootstrap intervals over puzzles.

**Protocol.** Fixed configuration across runs; repeats where variance matters (the solver samples at temperature 0.2); gold never enters a prompt. The evaluation set is 39 NYT Monday puzzles that were never used during development. These puzzles have been public on GitHub since 2024, so "held out" means held out from tuning, not from pretraining.

### Results

Three independent runs over the 39-puzzle evaluation set with the default configuration (Kimi K3 sweep, audit on):

| Run | Exact (match gold) | Mean letters | "Complete" claim precision | Model calls | Cost per puzzle |
|---|---:|---:|---:|---:|---:|
| 1 | 37 of 39 | 99.97% | 95% | 8.6 | $0.036 |
| 2 | 36 of 39 | 99.93% | 95% | 8.6 | $0.036 |
| 3 | 38 of 39 | 99.94% | 100% | 8.6 | $0.036 |
| **All** | **111 of 117 runs (94.9%, 95% CI 89 to 98)** | | | | |

Thirty-three puzzles were solved in every run and every puzzle was solved in at least one; each miss is a single wrong cell. The solver samples at temperature 0.2, so repeats are the right way to read any single number. Per-puzzle wall clock is 30 to 50 s at eight concurrent puzzles.

For scale, the SweepClip paper reports 48% exact on 100 NYT Mondays with GPT-4-Turbo under a $0.50-per-puzzle budget.

### What a miss looks like

A miss is almost always one cell shared by two crossing entries, where one side is a real word and the other a pattern-fitting non-word that the model proposed under a letter hint (ISAT crossing ONOFFSTITCH for ISAW/ONOFFSWITCH). Both were proposed and each supported the other, so verification accepted them. The audit stage is the answer to exactly this: it judges each fill against its clue, and the re-ask with the contested cell blanked makes the crossing earn its verification independently.

## Tests

```bash
uv run python -m unittest discover -v
```

Fourteen offline tests: grid parsing, the audit stage adopting a confirmed correction, the beam fill abstaining when a weak candidate blocks stronger ones, verification of crossing-implied entries, prompt patterns that never echo a doubtful letter, capped vote bonuses, tolerant JSON parsing, recoverable versus fatal provider errors, the call budget, and an end-to-end offline solve whose prompts are checked for gold leakage.

## Layout

```text
src/crossword_agent/
├── sweep_solver.py     the sweep → fill → verify → repair → escalate → audit loop
├── model.py            AnswerModel protocol; TokenFactoryModel (JSON output, reasoning controls, live prices); FixtureModel
├── puzzle.py           fixture parsing, entry derivation and numbering
├── types.py            Puzzle, Entry, SolveConfig, SolveResult, TraceEvent, JsonReply
├── run_history.py      metric-only JSONL run records
└── cli.py              `crossword-agent`; argument parsing shared with the harness
evals/                  run_eval.py (harness), metrics.py (deterministic scoring), summarize.py (reports)
tests/                  offline tests
data/dev/               the offline smoke fixture
```

The shared boundary:

```text
solve_with_sweep_fill(puzzle, model, config, on_event) -> SolveResult
```

## Token Factory notes

- `reasoning_effort: "none"` switches thinking off on DeepSeek and GLM models; it is sent only to models whose catalog entry advertises reasoning. GLM ignores `chat_template_kwargs`, and GLM's `"minimal"` effort burns the whole token cap with no output.
- `max_tokens` is always set explicitly; the server default of 8,192 includes reasoning tokens.
- Plain `response_format: json_object` is used instead of tool schemas: compiled schemas with length and regex constraints force length-matching nonsense when the model has no answer.
- Configuration errors (bad key, unknown model) stop the run; 429/5xx responses, transport errors, and timeouts are retried once and then skipped for that batch of clues.

## Scope and limitations

- Evaluated on American-style 15x15 puzzles. Puzzle-wide themes and revealers, rebus cells, diagramless grids, and image input are out of scope.
- Verification can still be fooled when a non-word and a wrong crossing agree with each other and survive the audit; runs vary at temperature 0.2, so report repeats, not single runs.
