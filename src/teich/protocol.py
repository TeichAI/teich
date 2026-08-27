from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from typing import Any


_CHATML_ROLES = ("system", "developer", "user", "assistant", "tool", "ipython")
_CHATML_STRUCTURAL_TURN_BOUNDARY_PATTERN = re.compile(
    r"<\|im_end\|>(?:\r?\n)?"
    r"(?=<\|im_start\|>(?P<role>" + "|".join(_CHATML_ROLES) + r")(?:\r?\n))"
)
_CHATML_TURN_HEADER_PATTERN = re.compile(
    r"<\|im_start\|>(?P<role>" + "|".join(_CHATML_ROLES) + r")(?:\r?\n)"
)


def _span_range(span: Mapping[str, Any]) -> tuple[int, int] | None:
    start = span.get("source_start", span.get("start"))
    end = span.get("source_end", span.get("end"))
    if isinstance(start, int) and isinstance(end, int) and start < end:
        return start, end
    return None


def _roles_match(rendered_role: str, span_role: object) -> bool:
    normalized = "assistant" if span_role == "model" else span_role
    if rendered_role == "ipython":
        return normalized in {"ipython", "tool"}
    return normalized == rendered_role


def chatml_header_is_source_anchored(
    text: str,
    header_start: int,
    role: str,
    source_spans: Sequence[Mapping[str, Any]],
) -> bool:
    """Reject protocol-looking transcript snippets inside known message content."""
    has_context_spans = any(
        span.get("role") not in {"assistant", "model"}
        for span in source_spans
    )
    if not has_context_spans:
        # External/legacy metadata may contain only the model span. Retain the
        # fail-closed raw protocol boundary in that case because there is no
        # marker-derived context with which to disambiguate transcript text.
        return True
    containing_ranges = [
        span_range
        for span in source_spans
        if (span_range := _span_range(span)) is not None
        and span_range[0] < header_start < span_range[1]
    ]
    if not containing_ranges:
        return True

    header = _CHATML_TURN_HEADER_PATTERN.match(text, header_start)
    if header is None or header.group("role") != role:
        return False
    turn_end = text.find("<|im_end|>", header.end())
    if turn_end < 0:
        return False
    return any(
        _roles_match(role, span.get("role"))
        and (span_range := _span_range(span)) is not None
        and header.end() <= span_range[0] < turn_end
        for span in source_spans
    )


def iter_chatml_structural_turn_boundaries(
    text: str,
    start: int = 0,
    source_spans: Sequence[Mapping[str, Any]] | None = None,
) -> Iterator[re.Match[str]]:
    """Yield ChatML turn boundaries, excluding quoted or standalone token names."""
    for match in _CHATML_STRUCTURAL_TURN_BOUNDARY_PATTERN.finditer(text, max(start, 0)):
        if source_spans is not None and not chatml_header_is_source_anchored(
            text,
            match.end(),
            match.group("role"),
            source_spans,
        ):
            continue
        yield match


def first_chatml_structural_turn_boundary_end(
    text: str,
    start: int = 0,
    source_spans: Sequence[Mapping[str, Any]] | None = None,
) -> int | None:
    """Return the end of the first terminator/new-turn boundary at or after start."""
    match = next(iter_chatml_structural_turn_boundaries(text, start, source_spans), None)
    return match.end() if match is not None else None


def chatml_turn_role_at_start(text: str) -> str | None:
    """Return a ChatML role only when its header begins the supplied text."""
    match = _CHATML_TURN_HEADER_PATTERN.match(text)
    return match.group("role") if match is not None else None
