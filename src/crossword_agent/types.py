from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Coordinate = tuple[int, int]
Direction = Literal["across", "down"]
ActionKind = Literal["start", "sweep", "repair", "escalate", "audit", "finish"]


@dataclass(frozen=True)
class Entry:
    id: str
    number: int
    direction: Direction
    clue: str
    cells: tuple[Coordinate, ...]

    @property
    def length(self) -> int:
        return len(self.cells)


@dataclass(frozen=True)
class Puzzle:
    id: str
    title: str
    grid: tuple[str, ...]
    clues: dict[str, str]


@dataclass(frozen=True)
class PuzzleFixture:
    puzzle: Puzzle
    solution: tuple[str, ...]
    candidate_bank: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class JsonReply:
    """One structured model completion: the parsed JSON object plus usage and outcome."""

    data: dict[str, Any] | None
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost_usd: float | None = 0.0
    finish_reason: str | None = None
    error: str | None = None  # recoverable failure (timeout, length stop, malformed JSON)


@dataclass(frozen=True)
class TraceEvent:
    turn: int
    action: ActionKind
    accepted: bool
    message: str
    grid: tuple[str, ...]
    alternatives: tuple[str, ...] = ()  # entries placed in this stage
    rationale: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float | None = None
    revisions: int = 0


@dataclass(frozen=True)
class SolveConfig:
    """Configuration of the sweep-fill solver. Defaults reproduce the measured configuration."""

    # Model roles: the sweep model is the adapter's model; None disables an escalation stage.
    strong_model: str | None = "deepseek-ai/DeepSeek-V4-Pro"
    reasoning_model: str | None = "zai-org/GLM-5.3-Flash"
    # Candidate acquisition.
    candidates_per_entry: int = 5
    clues_per_call: int = 26
    repair_rounds: int = 3
    escalation_rounds: int = 2
    audit_model: str | None = "zai-org/GLM-5.3-Flash"  # clue-fit audit with bounded reasoning; None disables
    audit_rounds: int = 2
    # Deterministic fill and verification.
    beam_width: int = 300
    abstain_penalty: float = 0.6
    verify_support: float = 0.5
    repair_trust: float = 0.55
    escalate_trust: float = 0.6
    # Budgets.
    max_model_calls: int = 40
    max_cost_usd: float | None = None
    max_seconds: float | None = None


@dataclass(frozen=True)
class SolveResult:
    puzzle_id: str
    status: Literal["complete", "incomplete", "budget_exhausted", "error"]
    grid: tuple[str, ...]
    assignments: dict[str, str]
    events: tuple[TraceEvent, ...]
    model_calls: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    duration_ms: float
    unknown_cost_calls: int = 0
    error: str | None = None
    reasoning_tokens: int = 0
    unverified_entries: tuple[str, ...] = ()
