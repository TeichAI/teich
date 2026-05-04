from __future__ import annotations

import csv
from pathlib import Path


PromptRow = dict[str, str | None]


def load_prompt_rows(path: Path) -> list[PromptRow]:
    if path.suffix.lower() == ".csv":
        return _load_prompt_rows_from_csv(path)
    return _load_prompt_rows_from_text(path)


def _load_prompt_rows_from_text(path: Path) -> list[PromptRow]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            {"prompt": line.strip()}
            for line in handle
            if line.strip() and not line.startswith("#")
        ]


def _load_prompt_rows_from_csv(path: Path) -> list[PromptRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [name.strip().lower() for name in reader.fieldnames or [] if isinstance(name, str)]
        if "prompt" not in fieldnames:
            raise ValueError("Prompt CSV must include a 'prompt' column")

        rows: list[PromptRow] = []
        for row in reader:
            normalized_row: PromptRow = {
                key.strip().lower(): value
                for key, value in row.items()
                if isinstance(key, str)
            }
            if not any(isinstance(value, str) and value.strip() for value in normalized_row.values()):
                continue
            rows.append(normalized_row)
    return rows
