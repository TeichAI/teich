from __future__ import annotations

import sys
from importlib import import_module

from agentic_datagen import (
    Config,
    TrainingExample,
    __version__,
    convert_trace_to_training_example,
    convert_traces_to_training_data,
    format_and_mask,
    load_config,
    load_traces,
)

for _module_name in (
    "cli",
    "config",
    "converter",
    "formatter",
    "loader",
    "runner",
    "trace_readme",
):
    sys.modules[f"{__name__}.{_module_name}"] = import_module(f"agentic_datagen.{_module_name}")

__all__ = [
    "Config",
    "TrainingExample",
    "convert_trace_to_training_example",
    "convert_traces_to_training_data",
    "format_and_mask",
    "load_traces",
    "load_config",
]
