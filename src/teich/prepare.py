from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset

from .audit import SFTAuditReport, audit_sft_dataset, audit_sft_trainer_batch
from .collator import TeichDataCollator
from .formatter import format_and_mask, format_data, preview_sft_example
from .loader import load_traces


@dataclass(slots=True)
class PreparedSFTDataset:
    dataset: Dataset
    collator: TeichDataCollator
    dataset_report: SFTAuditReport | None
    batch_report: SFTAuditReport | None
    sft_config_kwargs: dict[str, Any]
    tokenizer: Any | None = None

    def preview(self, index: int = 0) -> str:
        return preview_sft_example(self.dataset, self.tokenizer, index=index)


def prepare_sft_dataset(
    source_or_dataset: str | Path | Dataset | Sequence[str | Path | Dataset],
    tokenizer: Any,
    *,
    split: str | None = "train",
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    max_examples: int | None = None,
    messages_column: str = "messages",
    tools_column: str = "tools",
    chat_template_kwargs: dict[str, Any] | None = None,
    train_on_reasoning: bool = True,
    max_length: int | None = None,
    include_debug_columns: bool = False,
    drop_oversized_examples: bool = True,
    strict: bool = True,
    audit: bool = True,
    audit_sample_size: int = 8,
    batch_audit_sample_size: int = 2,
    verbose: bool = True,
    collator: TeichDataCollator | None = None,
) -> PreparedSFTDataset:
    dataset = _resolve_source_dataset(
        source_or_dataset,
        split=split,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_dir=local_dir,
        max_examples=max_examples,
    )
    training_data = format_and_mask(
        dataset,
        tokenizer,
        messages_column=messages_column,
        tools_column=tools_column,
        chat_template_kwargs=chat_template_kwargs,
        train_on_reasoning=train_on_reasoning,
        max_length=max_length,
        include_debug_columns=include_debug_columns,
        drop_oversized_examples=drop_oversized_examples,
        strict=strict,
        verbose=verbose,
    )
    data_collator = collator or TeichDataCollator(tokenizer=tokenizer)
    dataset_report: SFTAuditReport | None = None
    batch_report: SFTAuditReport | None = None
    if audit:
        dataset_report = audit_sft_dataset(training_data, tokenizer, sample_size=audit_sample_size)
        dataset_report.raise_for_errors()
        batch_report = audit_sft_trainer_batch(
            training_data,
            tokenizer,
            data_collator=data_collator,
            sample_size=batch_audit_sample_size,
        )
        batch_report.raise_for_errors()
    return PreparedSFTDataset(
        dataset=training_data,
        collator=data_collator,
        dataset_report=dataset_report,
        batch_report=batch_report,
        sft_config_kwargs={"dataset_kwargs": {"skip_prepare_dataset": True}, "dataset_num_proc": 1},
        tokenizer=tokenizer,
    )


def prepare_data(
    source_or_dataset: str | Path | Dataset | Sequence[str | Path | Dataset],
    tokenizer: Any,
    *,
    split: str | None = "train",
    revision: str | None = None,
    token: str | None = None,
    cache_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
    max_examples: int | None = None,
    messages_column: str = "messages",
    tools_column: str = "tools",
    text_column: str = "text",
    chat_template_kwargs: dict[str, Any] | None = None,
    train_on_reasoning: bool = True,
    max_length: int | None = None,
    drop_oversized_examples: bool = True,
    strict: bool = True,
    verbose: bool = True,
) -> Dataset:
    dataset = _resolve_source_dataset(
        source_or_dataset,
        split=split,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_dir=local_dir,
        max_examples=max_examples,
    )
    return format_data(
        dataset,
        tokenizer,
        messages_column=messages_column,
        tools_column=tools_column,
        text_column=text_column,
        chat_template_kwargs=chat_template_kwargs,
        train_on_reasoning=train_on_reasoning,
        max_length=max_length,
        drop_oversized_examples=drop_oversized_examples,
        strict=strict,
        verbose=verbose,
    )


def _resolve_source_dataset(
    source_or_dataset: str | Path | Dataset | Sequence[str | Path | Dataset],
    *,
    split: str | None,
    revision: str | None,
    token: str | None,
    cache_dir: str | Path | None,
    local_dir: str | Path | None,
    max_examples: int | None,
) -> Dataset | Sequence[Dataset]:
    if isinstance(source_or_dataset, Dataset):
        return _resolve_single_source_dataset(
            source_or_dataset,
            split=split,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            local_dir=local_dir,
            max_examples=max_examples,
        )
    if isinstance(source_or_dataset, Sequence) and not isinstance(source_or_dataset, (str, bytes, bytearray)):
        sources = list(source_or_dataset)
        if not sources:
            raise ValueError("At least one dataset must be provided.")
        return [
            _resolve_single_source_dataset(
                source,
                split=split,
                revision=revision,
                token=token,
                cache_dir=cache_dir,
                local_dir=local_dir,
                max_examples=max_examples,
            )
            for source in sources
        ]
    return _resolve_single_source_dataset(
        source_or_dataset,
        split=split,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_dir=local_dir,
        max_examples=max_examples,
    )


def _resolve_single_source_dataset(
    source_or_dataset: str | Path | Dataset,
    *,
    split: str | None,
    revision: str | None,
    token: str | None,
    cache_dir: str | Path | None,
    local_dir: str | Path | None,
    max_examples: int | None,
) -> Dataset:
    if isinstance(source_or_dataset, Dataset):
        if max_examples is None:
            return source_or_dataset
        if max_examples < 0:
            raise ValueError("max_examples must be non-negative.")
        limit = min(max_examples, source_or_dataset.num_rows)
        return source_or_dataset.shuffle(seed=3407).select(range(limit))
    if not isinstance(source_or_dataset, (str, Path)):
        raise TypeError(
            "A sequence source must contain only dataset paths, Hugging Face dataset IDs, or datasets.Dataset objects."
        )
    return load_traces(
        source_or_dataset,
        split=split,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_dir=local_dir,
        max_examples=max_examples,
    )
