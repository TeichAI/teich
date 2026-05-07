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

    for column_name in ("input_ids", "attention_mask", "labels"):
        if column_name not in row:
            errors.append(f"row {row_index}: missing required column '{column_name}'")
            return errors, warnings, sample

    input_ids = _as_list(row["input_ids"])
    attention_mask = _as_list(row["attention_mask"])
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

    suspicious_masked_markers = ("<|im_start|>user", "<tool_response>", "# Tools")
    for marker in suspicious_masked_markers:
        if marker in supervised_text:
            errors.append(f"row {row_index}: supervised text contains masked-context marker {marker!r}")

    useful_targets = ("<tool_call>", "</think>", "<|im_end|>")
    if not any(target in supervised_text for target in useful_targets):
        warnings.append(f"row {row_index}: supervised text lacks common assistant/tool/reasoning delimiters")

    return errors, warnings, sample


def audit_sft_dataset(dataset: Dataset, tokenizer: Any, *, sample_size: int = 8) -> SFTAuditReport:
    if not isinstance(dataset, Dataset):
        return SFTAuditReport(ok=False, errors=["dataset must be a datasets.Dataset instance"])
    errors: list[str] = []
    warnings: list[str] = []
    samples: list[dict[str, Any]] = []

    required_columns = {"input_ids", "attention_mask", "labels"}
    missing_columns = sorted(required_columns.difference(dataset.column_names))
    if missing_columns:
        return SFTAuditReport(ok=False, errors=[f"dataset missing required columns: {', '.join(missing_columns)}"])

    if dataset.num_rows == 0:
        return SFTAuditReport(ok=False, errors=["dataset contains no rows"])

    limit = min(max(sample_size, 0), dataset.num_rows)
    if limit == 0:
        warnings.append("sample_size is 0; no rows audited")

    for row_index in range(limit):
        row_errors, row_warnings, sample = _audit_training_row(dataset[row_index], tokenizer, row_index)
        errors.extend(row_errors)
        warnings.extend(row_warnings)
        samples.append(sample)

    return SFTAuditReport(ok=not errors, errors=errors, warnings=warnings, samples=samples)


def audit_sft_trainer_batch(
    dataset: Dataset,
    tokenizer: Any,
    *,
    data_collator: Any | None = None,
    sample_size: int = 2,
) -> SFTAuditReport:
    dataset_report = audit_sft_dataset(dataset, tokenizer, sample_size=sample_size)
    errors = list(dataset_report.errors)
    warnings = list(dataset_report.warnings)
    samples = list(dataset_report.samples)
    if errors:
        return SFTAuditReport(ok=False, errors=errors, warnings=warnings, samples=samples)

    if data_collator is None:
        try:
            from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
        except Exception as exc:
            return SFTAuditReport(ok=False, errors=[f"unable to import TRL DataCollatorForLanguageModeling: {exc}"])
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            return SFTAuditReport(ok=False, errors=["tokenizer must define pad_token_id or eos_token_id for default collator"])
        data_collator = DataCollatorForLanguageModeling(pad_token_id=pad_token_id)

    limit = min(max(sample_size, 0), dataset.num_rows)
    examples = [
        {column_name: dataset[row_index][column_name] for column_name in ("input_ids", "attention_mask", "labels")}
        for row_index in range(limit)
    ]
    if not examples:
        return SFTAuditReport(ok=not errors, errors=errors, warnings=warnings, samples=samples)

    try:
        batch = data_collator(examples)
    except Exception as exc:
        return SFTAuditReport(
            ok=False,
            errors=errors
            + [
                "data collator failed while batching precomputed input_ids/labels. "
                "For Teich pre-tokenized SFT data, pass a collator that pads labels with -100, such as "
                f"trl.trainer.sft_trainer.DataCollatorForLanguageModeling. Original error: {exc}"
            ],
            warnings=warnings,
            samples=samples,
        )

    batch_labels = batch.get("labels") if isinstance(batch, dict) else None
    if batch_labels is None:
        errors.append("data collator output is missing labels")
        return SFTAuditReport(ok=False, errors=errors, warnings=warnings, samples=samples)

    for row_index, example in enumerate(examples):
        collated_labels = batch_labels[row_index]
        if hasattr(collated_labels, "tolist"):
            collated_labels = collated_labels.tolist()
        original_labels = list(example["labels"])
        if list(collated_labels[: len(original_labels)]) != original_labels:
            errors.append(f"collated labels differ from dataset labels for sample {row_index}")
        if any(label != -100 for label in collated_labels[len(original_labels) :]):
            errors.append(f"collated padding labels are not masked for sample {row_index}")

    return SFTAuditReport(ok=not errors, errors=errors, warnings=warnings, samples=samples)
