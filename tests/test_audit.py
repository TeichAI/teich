from __future__ import annotations

import ast
from pathlib import Path

from datasets import Dataset

from teich import audit_sft_dataset, audit_sft_trainer_batch


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self._reverse_vocab = {1: "a", 2: "b", 3: "<|im_start|>user", 4: "<tool_call>", 5: "</think>"}

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._reverse_vocab[token_id] for token_id in token_ids)


class PreservingCollator:
    def __call__(self, examples):
        return {"labels": [example["labels"] for example in examples]}


class CorruptingCollator:
    def __call__(self, examples):
        return {"labels": [[-100] * len(example["labels"]) for example in examples]}


def test_audit_sft_dataset_accepts_valid_precomputed_labels():
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [1, 2, 4],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 4],
            }
        ]
    )

    report = audit_sft_dataset(dataset, TinyTokenizer())

    assert report.ok
    assert report.errors == []
    assert report.samples[0]["supervised_tokens"] == 2


def test_audit_sft_dataset_rejects_label_input_mismatch():
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "labels": [-100, 1],
            }
        ]
    )

    report = audit_sft_dataset(dataset, TinyTokenizer())

    assert not report.ok
    assert "labels differ from input_ids" in report.errors[0]


def test_audit_sft_dataset_rejects_supervised_user_marker():
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [3, 2],
                "attention_mask": [1, 1],
                "labels": [3, 2],
            }
        ]
    )

    report = audit_sft_dataset(dataset, TinyTokenizer())

    assert not report.ok
    assert "<|im_start|>user" in report.errors[0]


def test_audit_sft_trainer_batch_detects_collator_label_corruption():
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [1, 2, 4],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 4],
            }
        ]
    )

    report = audit_sft_trainer_batch(dataset, TinyTokenizer(), data_collator=CorruptingCollator())

    assert not report.ok
    assert "collated labels differ" in report.errors[0]


def test_audit_sft_trainer_batch_accepts_preserved_labels():
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [1, 2, 4],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 4],
            }
        ]
    )

    report = audit_sft_trainer_batch(dataset, TinyTokenizer(), data_collator=PreservingCollator())

    assert report.ok


def test_teich_example_has_single_safe_training_flow():
    source = Path("teich_example.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert source.count("FastLanguageModel.from_pretrained") == 1
    assert source.count("DataCollatorForLanguageModeling") >= 2
    assert "hf_" not in source
    assert "strict=True" in source
    assert 'optim="muon"' in source
    assert 'optim_target_modules="all-linear"' in source
    assert "audit_sft_dataset" in source
    assert "audit_sft_trainer_batch" in source
    assert "data_collator=data_collator" in source
    assert sum(isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "train" for node in ast.walk(tree)) == 1
