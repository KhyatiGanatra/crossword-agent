"""The core solving loop: sweep -> fill -> verify -> repair -> escalate.

The model is only ever asked "what could these clues be?" and answers with ranked lists. Code owns
the grid, chooses a globally consistent fill by beam search over a weighted constraint problem,
decides which entries are trustworthy, and decides what to ask next. See
docs/sweep-fill-walkthrough.md for the full explanation.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .model import AnswerModel, ModelError
from .puzzle import derive_entries
from .types import Entry, JsonReply, Puzzle, SolveConfig, SolveResult, TraceEvent

ProgressCallback = Callable[[TraceEvent], None]

SYSTEM_PROMPT = (
    "You are an expert American crossword solver. For every clue, list up to {k} candidate answers, "
    "ordered from most to least likely. Answers use only uppercase letters A-Z with no spaces or "
    "punctuation and must have exactly the stated length. When a letter pattern is given (dots are "
    "unknown letters), every candidate must match it position by position. Return ONLY a JSON "
    'object of the form {{"answers": {{"1A": ["FIRST", "SECND"], "2D": ["..."]}}}} with one key '
    "per clue id."
)

RANK_PRIOR = {1: 1.0, 2: 0.6, 3: 0.45}
DEFAULT_PRIOR = 0.35
VOTE_BONUS = 0.15  # added per repeated proposal of the same answer, capped
MAX_VOTE_BONUSES = 3
DOUBT_SUPPORT = 0.8  # crossings below this are blanked when a weak entry's pattern is fully determined
NEUTRAL_SUPPORT = 0.35
SUPPORT_WEIGHT = 2.0
POOL_LIMIT = 8
STRONG_CLUES_PER_CALL = 10
REASONING_CLUES_PER_CALL = 8
NO_THINK_MAX_TOKENS = 2500
THINK_MAX_TOKENS = 6000
AUDIT_PRIOR = 1.6  # a correction confirmed with its crossings in view outranks a blind proposal
AUDIT_DOUBT = 0.5  # score multiplier applied to a fill the audit flagged as not fitting its clue
MAX_AUDIT_EDIT = 3  # a correction may change at most this many letters

AUDIT_SYSTEM = (
    "You are checking a filled American crossword. For each entry you get its clue and the current fill. "
    "Decide whether the current fill genuinely answers the clue as a real word, name, abbreviation, or "
    "phrase. For every entry that does NOT, propose the most likely correct answer of exactly the same "
    "length, preferring answers that differ from the current fill in as few letters as possible (one to "
    "three). Do not flag entries that fit. Return ONLY a JSON object of the form "
    '{"flags": [{"id": "30D", "answer": "ISAW", "reason": "vidi means I saw"}]}; use an empty list if '
    "everything fits."
)
JOINT_SYSTEM = (
    "You are verifying proposed corrections to a filled American crossword. Each correction changes "
    "letters shared with crossing entries, and the crossings' resulting fills are shown with their clues. "
    "Accept a correction only if every affected entry becomes a correct answer to its clue. If a "
    "correction is close but a different same-length answer would make all affected entries correct, "
    'give it under alternatives. Return ONLY a JSON object of the form {"accept": ["44A"], '
    '"reject": ["12D"], "alternatives": [{"id": "44A", "answer": "ONOFFSWITCH"}]}.'
)

Assignment = tuple[str, float, float]  # answer, support, score
Grid = list[list[str]]


def normalize_answer(answer: str) -> str:
    """Uppercase A-Z only: what a model answer must reduce to before length and pattern checks."""

    return "".join(character for character in answer.upper() if "A" <= character <= "Z")


@dataclass
class _Usage:
    calls: int = 0
    prompt: int = 0
    completion: int = 0
    reasoning: int = 0
    cost: float = 0.0
    unknown_cost_calls: int = 0
    errors: list[str] = field(default_factory=list)


class SweepFillSolver:
    def __init__(
        self,
        puzzle: Puzzle,
        model: AnswerModel,
        config: SolveConfig | None = None,
        on_event: ProgressCallback | None = None,
    ) -> None:
        self.puzzle = puzzle
        self.model = model
        self.config = config or SolveConfig()
        self.on_event = on_event
        self.entries: dict[str, Entry] = derive_entries(puzzle)
        self.ids = sorted(self.entries, key=lambda key: (int(key[:-1]), key[-1]))
        cell_owner: dict[tuple[int, int], list[tuple[str, int]]] = {}
        for entry_id, entry in self.entries.items():
            for position, cell in enumerate(entry.cells):
                cell_owner.setdefault(cell, []).append((entry_id, position))
        # crossings[entry] = [(own letter index, other entry, other letter index), ...]
        self.crossings: dict[str, list[tuple[int, str, int]]] = {entry_id: [] for entry_id in self.entries}
        for owners in cell_owner.values():
            for entry_id, position in owners:
                for other_id, other_position in owners:
                    if other_id != entry_id:
                        self.crossings[entry_id].append((position, other_id, other_position))
        self.cands: dict[str, dict[str, float]] = {entry_id: {} for entry_id in self.entries}
        self._best_prior: dict[str, dict[str, float]] = {entry_id: {} for entry_id in self.entries}
        self._votes: dict[str, dict[str, int]] = {entry_id: {} for entry_id in self.entries}
        self.usage = _Usage()
        self.events: list[TraceEvent] = []
        self.started = time.perf_counter()
        self.budget_reason: str | None = None
        self._previous_assigned: dict[str, Assignment] = {}
        self._previous_verified = 0
        self._stage_mark = (0, 0, 0.0, 0)

    # ------------------------------------------------------------------ prompts
    def references(self, entry_id: str) -> list[str]:
        clue = self.entries[entry_id].clue
        lowered = clue.casefold()
        suffixes = [suffix for suffix, word in (("A", "across"), ("D", "down")) if word in lowered]
        referenced: list[str] = []
        for number in re.findall(r"\d+", clue):
            for suffix in suffixes:
                candidate = f"{int(number)}{suffix}"
                if candidate in self.entries and candidate != entry_id and candidate not in referenced:
                    referenced.append(candidate)
        return referenced

    def clue_line(
        self, entry_id: str, grid: Grid | None, with_crossings: bool, pattern: str | None = None
    ) -> str:
        entry = self.entries[entry_id]
        line = f"{entry_id} ({entry.length})"
        if pattern is None and grid is not None:
            pattern = self.pattern(grid, entry_id)
        if pattern and pattern.strip("."):
            line += f" pattern={pattern}"
        line += f": {entry.clue}"
        extras: list[str] = []
        if grid is not None:
            for referenced_id in self.references(entry_id):
                extras.append(f"{referenced_id} is currently {self.pattern(grid, referenced_id)}")
        if with_crossings and grid is not None:
            crossings = [
                f"letter {own_index + 1} crosses {other_id} '{self.entries[other_id].clue}' = "
                f"{self.pattern(grid, other_id)}"
                for own_index, other_id, _ in self.crossings[entry_id]
            ]
            extras.append("crossings: " + "; ".join(crossings))
        if extras:
            line += "  [" + " | ".join(extras) + "]"
        return line

    def build_prompt(
        self,
        chunk_ids: list[str],
        grid: Grid | None,
        with_crossings: bool,
        patterns: dict[str, str] | None = None,
    ) -> str:
        head = "Puzzle: American-style crossword.\n"
        if grid is not None:
            head += "Some clues include a letter pattern from crossing answers; candidates MUST fit it.\n"
        return head + "Clues (id (length) [pattern]: clue):\n" + "\n".join(
            self.clue_line(entry_id, grid, with_crossings, (patterns or {}).get(entry_id))
            for entry_id in chunk_ids
        )

    # ------------------------------------------------------------------ model access
    def query(
        self,
        ids: list[str],
        grid: Grid | None,
        *,
        thinking: bool,
        bonus: float,
        clues_per_call: int,
        with_crossings: bool = False,
        model: str | None = None,
        enforce_pattern: bool = True,
        patterns: dict[str, str] | None = None,
    ) -> None:
        """Ask the model about ``ids`` and merge valid answers into the candidate pools."""

        if not ids:
            return
        config = self.config
        chunks = [ids[start : start + clues_per_call] for start in range(0, len(ids), clues_per_call)]
        remaining = config.max_model_calls - self.usage.calls
        if remaining <= 0:
            self.budget_reason = f"model-call budget of {config.max_model_calls} exhausted"
            return
        if config.max_cost_usd is not None and self.usage.cost >= config.max_cost_usd:
            self.budget_reason = f"cost budget of ${config.max_cost_usd:.4f} exhausted"
            return
        if config.max_seconds is not None and self.elapsed_seconds() >= config.max_seconds:
            self.budget_reason = f"time budget of {config.max_seconds:g} s exhausted"
            return
        if len(chunks) > remaining:
            chunks = chunks[:remaining]
            self.budget_reason = f"model-call budget of {config.max_model_calls} exhausted"

        system = SYSTEM_PROMPT.format(k=config.candidates_per_entry)
        self.usage.calls += len(chunks)

        def run(chunk_ids: list[str]) -> JsonReply:
            return self.model.complete_json(
                system,
                self.build_prompt(chunk_ids, grid, with_crossings, patterns),
                thinking=thinking,
                max_tokens=THINK_MAX_TOKENS if thinking else NO_THINK_MAX_TOKENS + 80 * len(chunk_ids),
                model=model,
            )

        with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as executor:
            replies = list(executor.map(run, chunks))

        for chunk_ids, reply in zip(chunks, replies):
            self.usage.prompt += reply.input_tokens
            self.usage.completion += reply.output_tokens
            self.usage.reasoning += reply.reasoning_tokens
            if reply.estimated_cost_usd is None:
                self.usage.unknown_cost_calls += 1
            else:
                self.usage.cost += reply.estimated_cost_usd
            if reply.error:
                self.usage.errors.append(f"{reply.model or self.model.name}: {reply.error}")
            answers = (reply.data or {}).get("answers", reply.data) if reply.data else {}
            if not isinstance(answers, dict):
                answers = {}
            for entry_id in chunk_ids:
                raw = answers.get(entry_id) or []
                if not isinstance(raw, list):
                    raw = [raw]
                if patterns is not None and entry_id in patterns:
                    pattern = patterns[entry_id]
                else:
                    pattern = self.pattern(grid, entry_id) if grid is not None else None
                rank = 0
                for value in raw:
                    answer = normalize_answer(str(value))
                    if len(answer) != self.entries[entry_id].length:
                        continue
                    if (
                        enforce_pattern
                        and pattern
                        and any(known != "." and known != letter for known, letter in zip(pattern, answer))
                    ):
                        continue
                    rank += 1
                    self._propose(entry_id, answer, RANK_PRIOR.get(rank, DEFAULT_PRIOR) + bonus)

    def _propose(self, entry_id: str, answer: str, prior: float) -> None:
        """Score = best prior ever seen for this answer + a small, capped bonus per repeat proposal."""

        best = max(self._best_prior[entry_id].get(answer, 0.0), prior)
        votes = self._votes[entry_id].get(answer, 0) + 1
        self._best_prior[entry_id][answer] = best
        self._votes[entry_id][answer] = votes
        self.cands[entry_id][answer] = best + VOTE_BONUS * min(votes - 1, MAX_VOTE_BONUSES)

    # ------------------------------------------------------------------ scoring and fill
    def support(self, entry_id: str, answer: str) -> float:
        """One-step message passing: how well neighbouring pools agree with each letter (0..1)."""

        total = agreement = 0.0
        for own_index, other_id, other_index in self.crossings[entry_id]:
            pool = self.cands[other_id]
            total += 1.0
            if not pool:
                agreement += NEUTRAL_SUPPORT
                continue
            best = max(pool.values())
            best_agreeing = max(
                (score for other, score in pool.items() if other[other_index] == answer[own_index]),
                default=0.0,
            )
            agreement += best_agreeing / best
        return agreement / total if total else 0.0

    def fill(self) -> tuple[Grid, dict[str, Assignment]]:
        """Weighted CSP by beam search: maximise the sum of candidate scores subject to crossing
        agreement; an entry may stay blank (abstain) at a fixed penalty so an unsupported candidate is
        never forced in. Entries are processed most-confident-first."""

        beam_width = max(1, self.config.beam_width)
        abstain_penalty = self.config.abstain_penalty
        pools: dict[str, list[tuple[float, str, float]]] = {}
        for entry_id, pool in self.cands.items():
            scored = [
                (prior + SUPPORT_WEIGHT * self.support(entry_id, answer), answer, self.support(entry_id, answer))
                for answer, prior in pool.items()
            ]
            scored.sort(reverse=True)
            pools[entry_id] = scored[:POOL_LIMIT]
        order = sorted(
            self.ids,
            key=lambda entry_id: (
                -(pools[entry_id][0][0] if pools[entry_id] else 0.0),
                -self.entries[entry_id].length,
            ),
        )
        blank = tuple(self.puzzle.grid)
        states: list[tuple[float, tuple[str, ...], dict[str, Assignment]]] = [(0.0, blank, {})]
        for entry_id in order:
            cells = self.entries[entry_id].cells
            successors: dict[tuple[str, ...], tuple[float, tuple[str, ...], dict[str, Assignment]]] = {}
            for score, grid, assigned in states:
                children = [(score - abstain_penalty, grid, assigned)]
                for candidate_score, answer, candidate_support in pools[entry_id]:
                    if all(grid[row][col] in (".", letter) for (row, col), letter in zip(cells, answer)):
                        rows = [list(row) for row in grid]
                        for (row, col), letter in zip(cells, answer):
                            rows[row][col] = letter
                        new_grid = tuple("".join(row) for row in rows)
                        new_assigned = dict(assigned)
                        new_assigned[entry_id] = (answer, candidate_support, candidate_score)
                        children.append((score + candidate_score, new_grid, new_assigned))
                for child in children:
                    key = child[1]
                    if key not in successors or child[0] > successors[key][0]:
                        successors[key] = child
            states = sorted(successors.values(), key=lambda state: -state[0])[:beam_width]
        _, best_grid, assigned = states[0]
        grid = [list(row) for row in best_grid]
        for entry_id in self.ids:  # entries fully implied by crossings; unverified unless proposed
            if entry_id not in assigned:
                pattern = self.pattern(grid, entry_id)
                if "." not in pattern:
                    support = self.support(entry_id, pattern) if pattern in self.cands[entry_id] else 0.0
                    assigned[entry_id] = (pattern, support, 0.0)
        return grid, assigned

    # ------------------------------------------------------------------ verification and policy
    def pattern(self, grid: Grid, entry_id: str) -> str:
        return "".join(grid[row][col] for row, col in self.entries[entry_id].cells)

    def verified(self, entry_id: str, assigned: dict[str, Assignment]) -> bool:
        if entry_id not in assigned:
            return False
        answer, support, _ = assigned[entry_id]
        return answer in self.cands[entry_id] and support >= self.config.verify_support

    def complete(self, grid: Grid, assigned: dict[str, Assignment]) -> bool:
        return all(cell != "." for row in grid for cell in row) and all(
            self.verified(entry_id, assigned) for entry_id in self.ids
        )

    def unverified(self, assigned: dict[str, Assignment]) -> list[str]:
        return [entry_id for entry_id in self.ids if not self.verified(entry_id, assigned)]

    def trusted_ids(self, assigned: dict[str, Assignment], min_support: float) -> set[str]:
        return {
            entry_id
            for entry_id, (answer, support, _) in assigned.items()
            if support >= min_support and answer in self.cands[entry_id]
        }

    def trusted_grid(self, assigned: dict[str, Assignment], min_support: float) -> Grid:
        """Letters from verified, well-supported entries only: the pattern source for prompts."""

        grid = [list(row) for row in self.puzzle.grid]
        for entry_id, (answer, support, _) in sorted(assigned.items(), key=lambda item: -item[1][2]):
            if support < min_support or answer not in self.cands[entry_id]:
                continue
            cells = self.entries[entry_id].cells
            if all(grid[row][col] in (".", letter) for (row, col), letter in zip(cells, answer)):
                for (row, col), letter in zip(cells, answer):
                    grid[row][col] = letter
        return grid

    def prompt_patterns(
        self, assigned: dict[str, Assignment], weak: list[str], min_support: float
    ) -> tuple[Grid, dict[str, str]]:
        """The letters each weak entry is shown when it is re-asked.

        A weak entry never sees its own current letters, only letters contributed by trusted
        crossing entries. If that still determines every cell, the cells at its most doubtful
        crossings (support below DOUBT_SUPPORT, or the single weakest) become dots, so the model
        has something to decide instead of a fill to echo."""

        trusted = self.trusted_grid(assigned, min_support)
        trusted_ids = self.trusted_ids(assigned, min_support)
        patterns: dict[str, str] = {}
        for entry_id in weak:
            cells = self.entries[entry_id].cells
            letters = [
                self.puzzle.grid[row][col] if self.puzzle.grid[row][col] not in (".", "#") else "."
                for row, col in cells
            ]
            crossing_support: dict[int, float] = {}
            for own_index, other_id, other_index in self.crossings[entry_id]:
                if other_id in trusted_ids and other_id != entry_id:
                    letters[own_index] = assigned[other_id][0][other_index]
                    crossing_support[own_index] = assigned[other_id][1]
            if "." not in letters and crossing_support:
                doubtful = [index for index, support in crossing_support.items() if support < DOUBT_SUPPORT]
                if not doubtful:
                    doubtful = [min(crossing_support, key=lambda index: (crossing_support[index], index))]
                for index in doubtful:
                    letters[index] = "."
            patterns[entry_id] = "".join(letters)
        return trusted, patterns

    def weak_set(self, assigned: dict[str, Assignment], min_support: float) -> list[str]:
        """Entries to re-ask: unverified or weak, plus doubtful neighbours of unverified entries."""

        weak = {
            entry_id
            for entry_id in self.ids
            if not self.verified(entry_id, assigned) or assigned[entry_id][1] < min_support
        }
        for entry_id in list(weak):
            for _, other_id, _ in self.crossings[entry_id]:
                if other_id in assigned and assigned[other_id][1] < 0.8:
                    weak.add(other_id)
        return [entry_id for entry_id in self.ids if entry_id in weak]

    # ------------------------------------------------------------------ clue-fit audit
    def _audit_call(self, system: str, user: str) -> dict:
        model = self.config.audit_model
        remaining = self.config.max_model_calls - self.usage.calls
        if not model or remaining <= 0:
            if remaining <= 0:
                self.budget_reason = f"model-call budget of {self.config.max_model_calls} exhausted"
            return {}
        self.usage.calls += 1
        reply = self.model.complete_json(system, user, thinking=True, max_tokens=THINK_MAX_TOKENS, model=model)
        self.usage.prompt += reply.input_tokens
        self.usage.completion += reply.output_tokens
        self.usage.reasoning += reply.reasoning_tokens
        if reply.estimated_cost_usd is None:
            self.usage.unknown_cost_calls += 1
        else:
            self.usage.cost += reply.estimated_cost_usd
        if reply.error:
            self.usage.errors.append(f"{reply.model or model}: {reply.error}")
        return reply.data or {}

    def audit(self, grid: Grid, assigned: dict[str, Assignment]) -> list[str]:
        """Ask the audit model which fills do not answer their clues and what small edit would; check
        each edit's effect on the crossings; ask again with that context; adopt confirmed answers and
        return the entries (flagged plus touched crossings) that must be re-asked."""

        filled = [entry_id for entry_id in self.ids if "." not in self.pattern(grid, entry_id)]
        if not filled:
            return []
        lines = [f"{entry_id} ({self.entries[entry_id].length}): {self.entries[entry_id].clue} -> {self.pattern(grid, entry_id)}" for entry_id in filled]
        data = self._audit_call(AUDIT_SYSTEM, "Entries (id (length): clue -> current fill):\n" + "\n".join(lines))
        flags: dict[str, str] = {}
        for flag in data.get("flags") or []:
            if not isinstance(flag, dict):
                continue
            entry_id = str(flag.get("id", "")).upper()
            answer = normalize_answer(str(flag.get("answer", "")))
            current = self.pattern(grid, entry_id) if entry_id in self.entries else ""
            if (
                entry_id in self.entries
                and len(answer) == self.entries[entry_id].length
                and answer != current
                and sum(a != b for a, b in zip(answer, current)) <= MAX_AUDIT_EDIT
            ):
                flags[entry_id] = answer
        if not flags:
            return []

        # Effect of each proposed edit on its crossings, for the joint check.
        def crossing_effects(entry_id: str, answer: str) -> list[tuple[str, str, str]]:
            current = self.pattern(grid, entry_id)
            effects = []
            for own_index, other_id, other_index in self.crossings[entry_id]:
                if current[own_index] == answer[own_index]:
                    continue
                before = self.pattern(grid, other_id)
                after = before[:other_index] + answer[own_index] + before[other_index + 1 :]
                effects.append((other_id, before, after))
            return effects

        blocks = []
        for entry_id, answer in flags.items():
            effects = crossing_effects(entry_id, answer)
            detail = "; ".join(
                f"{other_id} '{self.entries[other_id].clue}' {before} -> {after}" for other_id, before, after in effects
            ) or "no crossing letters change"
            blocks.append(
                f"{entry_id} '{self.entries[entry_id].clue}': {self.pattern(grid, entry_id)} -> {answer} | affected crossings: {detail}"
            )
        decision = self._audit_call(JOINT_SYSTEM, "Proposed corrections:\n" + "\n".join(blocks))
        accepted = {str(value).upper() for value in (decision.get("accept") or []) if str(value).upper() in flags}
        for alternative in decision.get("alternatives") or []:
            if isinstance(alternative, dict):
                entry_id = str(alternative.get("id", "")).upper()
                answer = normalize_answer(str(alternative.get("answer", "")))
                if entry_id in flags and len(answer) == self.entries[entry_id].length:
                    flags[entry_id] = answer
                    accepted.add(entry_id)

        # Adopt only the flagged answers themselves. The crossings they touch are discounted and
        # returned so the caller re-asks them with the contested cell blanked: a crossing counts as
        # verified only if the sweep model proposes its new fill independently, never on the audit
        # model's word alone, so one bad correction cannot cascade through the grid.
        affected: list[str] = []
        for entry_id in accepted:
            answer = flags[entry_id]
            current = self.pattern(grid, entry_id)
            self._propose(entry_id, answer, AUDIT_PRIOR)
            if current in self.cands[entry_id]:
                self.cands[entry_id][current] *= AUDIT_DOUBT
            affected.append(entry_id)
            for other_id, before, after in crossing_effects(entry_id, answer):
                if before in self.cands[other_id]:
                    self.cands[other_id][before] *= AUDIT_DOUBT
                if other_id not in affected:
                    affected.append(other_id)
        return affected

    def audit_round(self, grid: Grid, assigned: dict[str, Assignment]) -> tuple[int, Grid, dict[str, Assignment]]:
        """One audit: corrections, then a repair re-ask of every affected entry, then a refill."""

        affected = self.audit(grid, assigned)
        if not affected:
            return 0, grid, assigned
        trusted, patterns = self.prompt_patterns(assigned, affected, self.config.repair_trust)
        self.query(affected, trusted, thinking=False, bonus=0.15, clues_per_call=self.config.clues_per_call, patterns=patterns)
        grid, assigned = self.fill()
        return len(affected), grid, assigned

    # ------------------------------------------------------------------ events and results
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started

    def _mark_stage(self) -> None:
        self._stage_mark = (
            self.usage.prompt,
            self.usage.completion,
            self.usage.cost,
            self.usage.unknown_cost_calls,
        )

    def emit(self, kind: str, label: str, grid: Grid, assigned: dict[str, Assignment]) -> None:
        previous = self._previous_assigned
        added = sorted(f"{entry_id}={value[0]}" for entry_id, value in assigned.items() if entry_id not in previous)
        revised = sorted(
            f"{entry_id}:{previous[entry_id][0]}->{value[0]}"
            for entry_id, value in assigned.items()
            if entry_id in previous and previous[entry_id][0] != value[0]
        )
        removed = sorted(f"{entry_id}={value[0]}" for entry_id, value in previous.items() if entry_id not in assigned)
        unverified = self.unverified(assigned)
        verified_count = len(self.ids) - len(unverified)
        open_cells = sum(cell != "#" for row in grid for cell in row)
        filled = sum(cell not in ("#", ".") for row in grid for cell in row)
        prompt_before, completion_before, cost_before, unknown_before = self._stage_mark
        unknown_delta = self.usage.unknown_cost_calls - unknown_before
        rationale_parts: list[str] = []
        if revised:
            rationale_parts.append("revised: " + ", ".join(revised))
        if removed:
            rationale_parts.append("removed: " + ", ".join(removed))
        if unverified and kind != "start":
            rationale_parts.append("unverified: " + ", ".join(unverified))
        event = TraceEvent(
            turn=len(self.events) + 1,
            action=kind,  # type: ignore[arg-type]
            accepted=kind in ("start", "finish") or bool(added or revised) or verified_count > self._previous_verified,
            message=(
                f"{label}: {verified_count}/{len(self.ids)} verified, "
                f"{100 * filled // max(open_cells, 1)}% filled, {len(unverified)} unverified"
            ),
            grid=tuple("".join(row) for row in grid),
            alternatives=tuple(added),
            rationale="; ".join(rationale_parts),
            input_tokens=self.usage.prompt - prompt_before,
            output_tokens=self.usage.completion - completion_before,
            estimated_cost_usd=None if unknown_delta else self.usage.cost - cost_before,
            revisions=len(revised) + len(removed),
        )
        self.events.append(event)
        if self.on_event:
            self.on_event(event)
        self._previous_assigned = dict(assigned)
        self._previous_verified = verified_count
        self._mark_stage()

    def result(self, status: str, grid: Grid, assigned: dict[str, Assignment], error: str | None = None) -> SolveResult:
        usage = self.usage
        cost: float | None
        if usage.calls and usage.unknown_cost_calls == usage.calls:
            cost = None
        else:
            cost = usage.cost
        return SolveResult(
            puzzle_id=self.puzzle.id,
            status=status,  # type: ignore[arg-type]
            grid=tuple("".join(row) for row in grid),
            assignments={entry_id: value[0] for entry_id, value in assigned.items()},
            events=tuple(self.events),
            model_calls=usage.calls,
            input_tokens=usage.prompt,
            output_tokens=usage.completion,
            estimated_cost_usd=cost,
            duration_ms=self.elapsed_seconds() * 1000,
            unknown_cost_calls=usage.unknown_cost_calls,
            error=error,
            reasoning_tokens=usage.reasoning,
            unverified_entries=tuple(self.unverified(assigned)),
        )

    # ------------------------------------------------------------------ main loop
    def solve(self) -> SolveResult:
        config = self.config
        grid: Grid = [list(row) for row in self.puzzle.grid]
        assigned: dict[str, Assignment] = {}
        self._mark_stage()
        self.emit("start", f"{len(self.ids)} clues, empty grid", grid, assigned)
        try:
            self.query(self.ids, None, thinking=False, bonus=0.0, clues_per_call=config.clues_per_call)
            grid, assigned = self.fill()
            self.emit("sweep", f"sweep ({len(self.ids)} clues)", grid, assigned)

            for round_number in range(1, config.repair_rounds + 1):
                if self.complete(grid, assigned) or self.budget_reason:
                    break
                weak = self.weak_set(assigned, config.repair_trust)
                trusted, patterns = self.prompt_patterns(assigned, weak, config.repair_trust)
                self.query(
                    weak, trusted, thinking=False, bonus=0.15, clues_per_call=config.clues_per_call, patterns=patterns
                )
                grid, assigned = self.fill()
                self.emit("repair", f"repair {round_number} ({len(weak)} clues)", grid, assigned)

            for round_number in range(1, config.escalation_rounds + 1):
                if self.complete(grid, assigned) or self.budget_reason:
                    break
                if config.strong_model:
                    weak = self.weak_set(assigned, config.escalate_trust)
                    trusted, patterns = self.prompt_patterns(assigned, weak, config.escalate_trust)
                    self.query(
                        weak,
                        trusted,
                        thinking=False,
                        bonus=0.3,
                        clues_per_call=STRONG_CLUES_PER_CALL,
                        with_crossings=True,
                        model=config.strong_model,
                        enforce_pattern=False,
                        patterns=patterns,
                    )
                    grid, assigned = self.fill()
                    self.emit(
                        "escalate",
                        f"escalate {round_number}: {config.strong_model} without reasoning ({len(weak)} clues)",
                        grid,
                        assigned,
                    )
                if self.complete(grid, assigned) or self.budget_reason:
                    break
                if config.reasoning_model:
                    weak = self.weak_set(assigned, config.escalate_trust)
                    trusted, patterns = self.prompt_patterns(assigned, weak, config.escalate_trust)
                    self.query(
                        weak,
                        trusted,
                        thinking=True,
                        bonus=0.35,
                        clues_per_call=REASONING_CLUES_PER_CALL,
                        with_crossings=True,
                        model=config.reasoning_model,
                        enforce_pattern=False,
                        patterns=patterns,
                    )
                    grid, assigned = self.fill()
                    self.emit(
                        "escalate",
                        f"escalate {round_number}: {config.reasoning_model} with bounded reasoning ({len(weak)} clues)",
                        grid,
                        assigned,
                    )
            for round_number in range(1, config.audit_rounds + 1):
                if not config.audit_model or self.budget_reason:
                    break
                touched, grid, assigned = self.audit_round(grid, assigned)
                if not touched:
                    break
                self.emit("audit", f"audit {round_number}: {config.audit_model} flagged fills; {touched} entries re-asked", grid, assigned)
        except ModelError as exc:
            return self.result("error", grid, assigned, str(exc))

        if self.complete(grid, assigned):
            self.emit("finish", "complete: every entry proposed by the model and supported by its crossings", grid, assigned)
            return self.result("complete", grid, assigned)
        unverified = self.unverified(assigned)
        note = f"; {len(self.usage.errors)} model call(s) failed, first: {self.usage.errors[0]}" if self.usage.errors else ""
        if self.budget_reason:
            return self.result(
                "budget_exhausted", grid, assigned, f"{self.budget_reason}; {len(unverified)} entries unverified{note}"
            )
        return self.result(
            "incomplete",
            grid,
            assigned,
            f"{len(unverified)} entries still unverified after all repair and escalation rounds{note}",
        )


def solve_with_sweep_fill(
    puzzle: Puzzle,
    model: AnswerModel,
    config: SolveConfig | None = None,
    on_event: ProgressCallback | None = None,
) -> SolveResult:
    """Solve ``puzzle`` with the sweep-fill loop. Gold answers are never part of this call."""

    return SweepFillSolver(puzzle, model, config, on_event).solve()
