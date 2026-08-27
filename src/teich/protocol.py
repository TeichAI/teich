from __future__ import annotations

import re
from collections.abc import Iterator


_CHATML_ROLES = ("system", "developer", "user", "assistant", "tool")
_CHATML_STRUCTURAL_TURN_BOUNDARY_PATTERN = re.compile(
    r"<\|im_end\|>(?:\r?\n)?"
    r"(?=<\|im_start\|>(?P<role>" + "|".join(_CHATML_ROLES) + r")(?:\r?\n))"
)


def iter_chatml_structural_turn_boundaries(text: str, start: int = 0) -> Iterator[re.Match[str]]:
    """Yield ChatML turn boundaries, excluding quoted or standalone token names."""
    return _CHATML_STRUCTURAL_TURN_BOUNDARY_PATTERN.finditer(text, max(start, 0))


def first_chatml_structural_turn_boundary_end(text: str, start: int = 0) -> int | None:
    """Return the end of the first terminator/new-turn boundary at or after start."""
    match = next(iter_chatml_structural_turn_boundaries(text, start), None)
    return match.end() if match is not None else None
