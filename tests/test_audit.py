from __future__ import annotations

import ast
from pathlib import Path

from datasets import Dataset

from teich import audit_sft_dataset


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def __init__(self):
        self._reverse_vocab = {
            1: "a",
            2: "b",
            3: "<|im_start|>user",
            4: "<tool_call>",
            5: "</think>",
            6: "<|turn>user",
            7: "<|tool_response>",
            8: "<|im_end|>\n",
            9: "<|im_start|>user\n",
            10: '["<|im_start|>", "<|im_end|>"]',
            11: "EXAMPLE<|im_end|>\n<|im_start|>user\nLITERAL<|im_end|>",
        }

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(self._reverse_vocab[token_id] for token_id in token_ids)


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


def test_audit_sft_dataset_rejects_run_starting_at_chatml_user_header():
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [1, 8, 9, 2],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, -100, 9, 2],
            }
        ]
    )

    report = audit_sft_dataset(dataset, TinyTokenizer())

    assert not report.ok
    assert "begins at masked-context ChatML 'user' header" in report.errors[0]


def test_audit_sft_dataset_allows_quoted_chatml_token_names():
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [1, 10, 4],
                "attention_mask": [1, 1, 1],
                "labels": [1, 10, 4],
            }
        ]
    )

    report = audit_sft_dataset(dataset, TinyTokenizer())

    assert report.ok
    assert report.errors == []


def test_audit_sft_dataset_allows_literal_full_chatml_transcript_in_assistant_output():
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [1, 11, 4],
                "attention_mask": [1, 1, 1],
                "labels": [1, 11, 4],
            }
        ]
    )

    report = audit_sft_dataset(dataset, TinyTokenizer())

    assert report.ok
    assert report.errors == []


def test_audit_sft_dataset_rejects_gemma_context_markers():
    dataset = Dataset.from_list(
        [
            {
                "input_ids": [6, 7, 2],
                "attention_mask": [1, 1, 1],
                "labels": [6, 7, 2],
            }
        ]
    )

    report = audit_sft_dataset(dataset, TinyTokenizer())

    assert not report.ok
    assert "<|turn>user" in report.errors[0]
    assert "<|tool_response>" in report.errors[1]


def test_audit_sft_dataset_checks_all_rows_by_default():
    safe = {"input_ids": [1, 2, 4], "attention_mask": [1, 1, 1], "labels": [-100, 2, 4]}
    leaked = {"input_ids": [6, 2], "attention_mask": [1, 1], "labels": [6, 2]}
    dataset = Dataset.from_list([safe] * 8 + [leaked])

    report = audit_sft_dataset(dataset, TinyTokenizer())

    assert not report.ok
    assert any("row 8" in error and "<|turn>user" in error for error in report.errors)


def test_audit_sample_size_limits_previews_not_correctness_checks():
    safe = {"input_ids": [1, 2, 4], "attention_mask": [1, 1, 1], "labels": [-100, 2, 4]}
    leaked = {"input_ids": [6, 2], "attention_mask": [1, 1], "labels": [6, 2]}
    dataset = Dataset.from_list([safe] * 8 + [leaked])

    report = audit_sft_dataset(dataset, TinyTokenizer(), sample_size=1)

    assert not report.ok
    assert len(report.samples) == 1
    assert any("row 8" in error and "<|turn>user" in error for error in report.errors)


def test_teich_example_has_single_safe_training_flow():
    source = Path("teich_example.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert source.count("FastLanguageModel.from_pretrained") == 1
    assert "DataCollatorForLanguageModeling" not in source
    assert '"hf_' not in source
    assert "strict=True" in source
    assert 'optim="adamw_8bit"' in source
    assert 'lr_scheduler_type="cosine"' in source
    assert "prepare_data" in source
    assert "mask_data" in source
    assert 'dataset_text_field="text"' in source
    assert "data_collator=" not in source
    assert "prepare_sft_dataset" not in source
    assert sum(isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "train" for node in ast.walk(tree)) == 1


def test_gemma4_example_uses_live_remote_template_and_safe_masks():
    source = Path("gemma4_example.py").read_text(encoding="utf-8")

    assert 'os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")' in source
    assert 'MODEL_REVISION = os.environ.get("MODEL_REVISION", "main")' in source
    assert 'CHAT_TEMPLATE_PATH = os.environ.get("CHAT_TEMPLATE_PATH")' in source
    assert 'os.environ.get("GEMMA4_THINKING_MODE")' in source
    assert 'os.environ.get("GEMMA4_ENABLE_THINKING")' in source
    assert 'GEMMA4_THINKING_MODE not in {"auto", "thinking", "nonthinking"}' in source
    assert "chat_template_kwargs=CHAT_TEMPLATE_KWARGS" in source
    assert '"reasoning_policy": AGENT_REASONING_POLICY' in source
    assert '"reasoning_policy": CHAT_REASONING_POLICY' in source
    assert 'AGENT_REASONING_POLICY = os.environ.get("AGENT_REASONING_POLICY", "keep")' in source
    assert 'CHAT_REASONING_POLICY = os.environ.get("CHAT_REASONING_POLICY", "strip")' in source
    assert "train_dataset, prep_report = prepare_data(" in source
    assert "return_report=True" in source
    assert "prep_report.gemma4_modes" in source
    assert "Do not append a" in source
    assert 'or "gemma-template.jinja"' not in source
    assert "token=HF_TOKEN or None" in source
    assert 'oversized_policy="trim_followups"' in source
    assert "processing_class=tokenizer" in source


def test_model_family_examples_use_live_templates_and_source_level_reasoning_contracts():
    examples = {
        "granite42_example.py": [
            '"ibm-granite/granite-4.2-8b"',
            "prep_report.granite42_modes",
            'os.environ.get("GRANITE42_LOW_EFFORT", "0")',
            'agent_source["chat_template_kwargs"] = {"low_effort": True}',
        ],
        "qwen38_example.py": [
            '"Qwen/Qwen3.8-27B"',
            'QWEN38_REASONING_EFFORT", "medium"',
            'REASONING_EFFORT not in {"low", "medium", "xhigh"}',
            '"reasoning_effort": REASONING_EFFORT',
            '"preserve_thinking": True',
        ],
        "qwen36_example.py": [
            '"Qwen/Qwen3.6-35B-A3B"',
            '"preserve_thinking": True',
            "has no reasoning_effort switch",
        ],
    }

    for path, expected_fragments in examples.items():
        source = Path(path).read_text(encoding="utf-8")
        tree = ast.parse(source)

        assert '"hf_' not in source
        assert "CHAT_TEMPLATE_PATH" not in source
        assert "prepare_data" in source
        assert "mask_data" in source
        assert '"reasoning_policy": AGENT_REASONING_POLICY' in source
        assert '"reasoning_policy": CHAT_REASONING_POLICY' in source
        assert '"enable_thinking": False' in source or path == "granite42_example.py"
        assert 'oversized_policy="trim_followups"' in source
        assert "strict=True" in source
        assert "return_report=True" in source
        assert 'dataset_text_field="text"' in source
        assert "packing=False" in source
        assert "processing_class=tokenizer" in source
        assert "DataCollatorForLanguageModeling" not in source
        assert "token=HF_TOKEN or None" in source
        assert sum(
            isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "train"
            for node in ast.walk(tree)
        ) == 1
        for fragment in expected_fragments:
            assert fragment in source
