"""Agentic Datagen v2 - Generate training data from Codex and Pi traces."""

__version__ = "0.1.1a1"

from .config import Config, load_config
from .converter import TrainingExample, convert_trace_to_training_example, convert_traces_to_training_data
from .formatter import format_and_mask
from .loader import load_traces

__all__ = [
    "Config",
    "TrainingExample",
    "convert_trace_to_training_example",
    "convert_traces_to_training_data",
    "format_and_mask",
    "load_traces",
    "load_config",
]
