from __future__ import annotations

import json
from pathlib import Path

from .types import Direction, Entry, Puzzle, PuzzleFixture


class PuzzleFormatError(ValueError):
    pass


def _validate_grid(rows: tuple[str, ...], *, allow_empty: bool) -> None:
    if not rows or not rows[0]:
        raise PuzzleFormatError("The grid must not be empty")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise PuzzleFormatError("All grid rows must have the same width")

    allowed = {"#", "."} if allow_empty else {"#"}
    for row in rows:
        for cell in row:
            if cell in allowed or ("A" <= cell <= "Z"):
                continue
            expected = "#, ., or A-Z" if allow_empty else "# or A-Z"
            raise PuzzleFormatError(f"Grid cells must contain {expected}")

def _run_cells(
    rows: tuple[str, ...], row: int, col: int, direction: Direction
) -> tuple[tuple[int, int], ...]:
    height, width = len(rows), len(rows[0])
    delta_row, delta_col = (0, 1) if direction == "across" else (1, 0)
    cells: list[tuple[int, int]] = []
    while row < height and col < width and rows[row][col] != "#":
        cells.append((row, col))
        row += delta_row
        col += delta_col
    return tuple(cells)


def derive_entries(puzzle: Puzzle) -> dict[str, Entry]:
    rows = puzzle.grid
    height, width = len(rows), len(rows[0])
    entries: dict[str, Entry] = {}
    number = 0

    for row in range(height):
        for col in range(width):
            if rows[row][col] == "#":
                continue

            across_cells = ()
            down_cells = ()
            if col == 0 or rows[row][col - 1] == "#":
                across_cells = _run_cells(rows, row, col, "across")
            if row == 0 or rows[row - 1][col] == "#":
                down_cells = _run_cells(rows, row, col, "down")

            starts_across = len(across_cells) >= 2
            starts_down = len(down_cells) >= 2
            if not (starts_across or starts_down):
                continue

            number += 1
            for direction, suffix, cells in (
                ("across", "A", across_cells),
                ("down", "D", down_cells),
            ):
                if len(cells) < 2:
                    continue
                entry_id = f"{number}{suffix}"
                if entry_id not in puzzle.clues:
                    raise PuzzleFormatError(f"Missing clue for {entry_id}")
                entries[entry_id] = Entry(
                    id=entry_id,
                    number=number,
                    direction=direction,
                    clue=puzzle.clues[entry_id],
                    cells=cells,
                )

    unused_clues = set(puzzle.clues) - set(entries)
    if unused_clues:
        raise PuzzleFormatError(f"Clues do not match grid entries: {sorted(unused_clues)}")
    return entries


def entry_pattern(entry: Entry, grid: tuple[str, ...]) -> str:
    """The letters a grid currently holds for an entry, "." for unknown."""

    return "".join(grid[row][col] for row, col in entry.cells)


def load_fixture(path: str | Path) -> PuzzleFixture:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    grid = tuple(row.upper() for row in data["grid"])
    solution = tuple(row.upper() for row in data["solution"])
    _validate_grid(grid, allow_empty=True)
    _validate_grid(solution, allow_empty=False)
    if len(grid) != len(solution) or any(
        len(left) != len(right) for left, right in zip(grid, solution)
    ):
        raise PuzzleFormatError("Grid and solution dimensions must match")
    for row_index, (puzzle_row, solution_row) in enumerate(zip(grid, solution)):
        for col_index, (puzzle_cell, solution_cell) in enumerate(
            zip(puzzle_row, solution_row)
        ):
            if (puzzle_cell == "#") != (solution_cell == "#"):
                raise PuzzleFormatError(
                    f"Block mismatch at row {row_index + 1}, column {col_index + 1}"
                )
            if puzzle_cell not in {"#", "."} and puzzle_cell != solution_cell:
                raise PuzzleFormatError(
                    f"Prefilled letter mismatch at row {row_index + 1}, column {col_index + 1}"
                )

    puzzle = Puzzle(
        id=data["id"],
        title=data.get("title", data["id"]),
        grid=grid,
        clues={key.upper(): value for key, value in data["clues"].items()},
    )
    derive_entries(puzzle)
    candidate_bank = {
        key.upper(): tuple(value.upper() for value in values)
        for key, values in data.get("candidate_bank", {}).items()
    }
    return PuzzleFixture(
        puzzle=puzzle,
        solution=solution,
        candidate_bank=candidate_bank,
    )
