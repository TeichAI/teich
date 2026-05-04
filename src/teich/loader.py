from __future__ import annotations

from pathlib import Path

from datasets import Dataset, Features, Json, List, Value
from huggingface_hub import snapshot_download

from .converter import convert_traces_to_training_data


def _trace_directory(root: Path, split: str | None) -> Path:
    if split:
        candidate = root / split
        if candidate.is_dir():
            return candidate
    return root


def _dataset_from_rows(rows: list[dict]) -> Dataset:
    try:
        return Dataset.from_list(rows, on_mixed_types="use_json")
    except TypeError as exc:
        if "on_mixed_types" not in str(exc):
            raise
    features = Features(
        {
            "prompt": Value("string"),
            "messages": List(Json()),
            "tools": List(Json()),
            "metadata": Json(),
        }
    )
    return Dataset.from_list(rows, features=features)


def load_traces(
    source: str | Path,
    split: str | None = "train",
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> Dataset:
    source_path = Path(source)
    if source_path.exists():
        root = source_path
    else:
        root = Path(
            snapshot_download(
                repo_id=str(source),
                repo_type="dataset",
                revision=revision,
                token=token,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_dir=str(local_dir) if local_dir is not None else None,
                allow_patterns=["*.jsonl", "**/*.jsonl"],
            )
        )
    traces_dir = _trace_directory(root, split)
    rows = convert_traces_to_training_data(traces_dir)
    if not rows:
        location = traces_dir if traces_dir != root else root
        if split and traces_dir == root:
            raise ValueError(f"No trace files found in {location} for split '{split}'.")
        raise ValueError(f"No trace files found in {location}.")
    return _dataset_from_rows(rows)
