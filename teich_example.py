# -*- coding: utf-8 -*-
import os

from unsloth import FastLanguageModel
import torch
from trl import SFTConfig, SFTTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

from teich import audit_sft_dataset, audit_sft_trainer_batch, format_and_mask, load_traces


MAX_SEQ_LEN = 32768
MODEL_NAME = "unsloth/Qwen3.5-4B"
TRAIN_ON_REASONING = True
CHAT_TEMPLATE_KWARGS = {"enable_thinking": True}
PUSH_TO_HUB_REPO_ID = "armand0e/traces-test"
HF_TOKEN = os.environ.get("HF_TOKEN") or ""


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LEN,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "out_proj"],
    lora_alpha=64,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

datasets = [
    load_traces("TeichAI/lordx64-claude-opus-4.7-max-cleaned", split="train", max_examples=500),
]

training_data = format_and_mask(
    datasets,
    tokenizer,
    chat_template_kwargs=CHAT_TEMPLATE_KWARGS,
    train_on_reasoning=TRAIN_ON_REASONING,
    max_length=MAX_SEQ_LEN,
    strict=True,
)

dataset_audit = audit_sft_dataset(training_data, tokenizer)
dataset_audit.raise_for_errors()
print(training_data.preview())

data_collator = DataCollatorForLanguageModeling(pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=training_data,
    eval_dataset=None,
    data_collator=data_collator,
    args=SFTConfig(
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=1,
        optim="muon",
        optim_target_modules="all-linear",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        output_dir="outputs",
        seed=3407,
        report_to="none",
    ),
)

batch_audit = audit_sft_trainer_batch(training_data, tokenizer, data_collator=data_collator)
batch_audit.raise_for_errors()

gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer_stats = trainer.train(resume_from_checkpoint=False)

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.push_to_hub_merged(PUSH_TO_HUB_REPO_ID, tokenizer, save_method="merged_16bit", token=HF_TOKEN)
