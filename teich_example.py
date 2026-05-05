# -*- coding: utf-8 -*-
from unsloth import FastLanguageModel
import torch

MAX_SEQ_LEN = 65536

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen3.5-4B",
    max_seq_length = MAX_SEQ_LEN, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",
                      "out_proj",],
    lora_alpha = 32,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


from teich import format_and_mask, load_traces
datasets = [
    load_traces("armand0e/ag-datagen-v2-test", split = "train"),
    load_traces("./output"),
]
#dataset = dataset.filter(lambda row: isinstance(row["messages"], list) and len(row["messages"]) > 0)


training_data = format_and_mask(
    datasets,
    tokenizer,
    chat_template_kwargs = {"enable_thinking": True, "preserve_thinking": True},
    max_length = MAX_SEQ_LEN,
)

# This step might take ~3m on this A100 notebook
from trl import SFTTrainer, SFTConfig
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = training_data,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 1, # Increasing "might" throw error on Colab/other envs.
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 3, # Set this for 1 full training run.
        learning_rate = 2e-5, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        output_dir= "outputs",
        seed = 3407,
        report_to = "none", # Use TrackIO/WandB etc
    ),
)

# lets do a quick masking verification preview

print(training_data.preview())

# Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

"""Let's train the model! To resume a training run, set `trainer.train(resume_from_checkpoint = True)`"""

trainer_stats = trainer.train( resume_from_checkpoint = False)

# Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(
    f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
)
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

model.push_to_hub_merged("armand0e/traces-test", tokenizer, save_method = "merged_16bit", token = "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")
