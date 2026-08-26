from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset


@dataclass
class SFTAuditReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    samples: list[dict[str, Any]] = field(default_factory=list)

    def raise_for_errors(self) -> None:
        if not self.ok:
            raise ValueError("SFT audit failed:\n" + "\n".join(f"- {error}" for error in self.errors))


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)


def _as_list(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _audit_training_row(row: dict[str, Any], tokenizer: Any, row_index: int) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    sample: dict[str, Any] = {"row_index": row_index}

    for column_name in ("input_ids", "labels"):
        if column_name not in row:
            errors.append(f"row {row_index}: missing required column '{column_name}'")
            return errors, warnings, sample

    input_ids = _as_list(row["input_ids"])
    attention_mask = _as_list(row["attention_mask"]) if "attention_mask" in row else [1] * len(input_ids)
    labels = _as_list(row["labels"])
    sample["tokens"] = len(input_ids)

    if not (len(input_ids) == len(attention_mask) == len(labels)):
        errors.append(
            f"row {row_index}: input_ids, attention_mask, and labels lengths differ "
            f"({len(input_ids)}, {len(attention_mask)}, {len(labels)})"
        )
        return errors, warnings, sample

    supervised_positions = [index for index, label in enumerate(labels) if label != -100]
    sample["supervised_tokens"] = len(supervised_positions)
    sample["supervised_ratio"] = round(len(supervised_positions) / len(labels), 4) if labels else 0.0

    if not supervised_positions:
        errors.append(f"row {row_index}: labels are fully masked")
        return errors, warnings, sample

    mismatches = [index for index in supervised_positions if labels[index] != input_ids[index]]
    if mismatches:
        errors.append(f"row {row_index}: labels differ from input_ids at supervised positions, first mismatch {mismatches[0]}")

    supervised_ids = [labels[index] for index in supervised_positions]
    supervised_text = _decode(tokenizer, supervised_ids)
    sample["supervised_preview"] = supervised_text[:500]

    masked_ids = [token_id for token_id, label in zip(input_ids, labels) if label == -100]
    masked_text = _decode(tokenizer, masked_ids[-200:]) if masked_ids else ""
    sample["masked_suffix_preview"] = masked_text[-500:]

    suspicious_masked_markers = (
        "<|im_start|>user",
        "<|start_header_id|>user<|end_header_id|>",
        "<start_of_turn>user",
        "<|start_of_role|>user<|end_of_role|>",
        "<|turn>user",
        "<tool_response>",
        "</tool_response>",
        "<|tool_response>",
        "<tool_response|>",
        "# Tools",
    )
    for marker in suspicious_masked_markers:
        if marker in supervised_text:
            errors.append(f"row {row_index}: supervised text contains masked-context marker {marker!r}")

    useful_targets = ("<tool_call>", "<|tool_call>", "<tool_call|>", "</think>", "<|channel>thought", "<|im_end|>", "<turn|>")
    if not any(target in supervised_text for target in useful_targets):
        warnings.append(f"row {row_index}: supervised text lacks common assistant/tool/reasoning delimiters")

    return errors, warnings, sample


def audit_sft_dataset(dataset: Dataset, tokenizer: Any, *, sample_size: int | None = None) -> SFTAuditReport:
    if not isinstance(dataset, Dataset):
        return SFTAuditReport(ok=False, errors=["dataset must be a datasets.Dataset instance"])
    errors: list[str] = []
    warnings: list[str] = []
    samples: list[dict[str, Any]] = []

    required_columns = {"input_ids", "labels"}
    missing_columns = sorted(required_columns.difference(dataset.column_names))
    if missing_columns:
        return SFTAuditReport(ok=False, errors=[f"dataset missing required columns: {', '.join(missing_columns)}"])

    if dataset.num_rows == 0:
        return SFTAuditReport(ok=False, errors=["dataset contains no rows"])

    preview_limit = dataset.num_rows if sample_size is None else min(max(sample_size, 0), dataset.num_rows)
    if preview_limit == 0:
        warnings.append("sample_size is 0; no row previews retained (all rows were still audited)")

    # Sampling controls report size only. Correctness checks are deliberately
    # exhaustive so an audit that returns ok=True is a dataset-wide gate.
    for row_index in range(dataset.num_rows):
        row_errors, row_warnings, sample = _audit_training_row(dataset[row_index], tokenizer, row_index)
        errors.extend(row_errors)
        warnings.extend(row_warnings)
        if row_index < preview_limit:
            samples.append(sample)

    return SFTAuditReport(ok=not errors, errors=errors, warnings=warnings, samples=samples)
