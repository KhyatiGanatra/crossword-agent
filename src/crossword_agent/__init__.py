"""Crossword-solving agent: sweep -> fill -> verify -> repair -> escalate."""

from .sweep_solver import SweepFillSolver, solve_with_sweep_fill
from .types import SolveConfig, SolveResult

solve = solve_with_sweep_fill

__all__ = [
    "SolveConfig",
    "SolveResult",
    "SweepFillSolver",
    "solve",
    "solve_with_sweep_fill",
]
