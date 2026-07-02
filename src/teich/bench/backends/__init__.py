"""Bench backends: a registry mapping a source ``type`` to its backend implementation."""

from __future__ import annotations

from .base import (
    BENCH_SPLITS,
    BenchBackend,
    BenchRun,
    BenchTask,
    bench_root,
    bench_stem,
    existing_output,
    harvest,
    primary_score,
    route_split,
    source_id,
)
from .harbor import HarborBackend
from .swebench import SweBenchBackend

# type -> backend factory.
_BACKENDS: dict[str, type] = {
    HarborBackend.type: HarborBackend,
    SweBenchBackend.type: SweBenchBackend,
}


def get_backend(source_type: str) -> BenchBackend:
    """Instantiate the backend for a source ``type``; clear error for an unknown type."""
    cls = _BACKENDS.get(source_type)
    if cls is None:
        supported = ", ".join(sorted(_BACKENDS))
        raise RuntimeError(f"Unknown bench source type {source_type!r}; supported: {supported}.")
    return cls()


__all__ = [
    "BENCH_SPLITS",
    "BenchBackend",
    "BenchRun",
    "BenchTask",
    "bench_root",
    "bench_stem",
    "existing_output",
    "get_backend",
    "harvest",
    "primary_score",
    "route_split",
    "source_id",
]
