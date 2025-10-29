# train_and_merge.py
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import torch
import gc
import os
import shutil


def safe_remove_directory(path):
    try:
        if os.path.exists(path) and os.path.isdir(path):
            shutil.rmtree(path)
            return True
        else:
            print(f"Path {path} is not a valid directory")
            return False
    except Exception as e:
        print(f"Failed to remove directory {path}: {e}")
        return False


# This tokenizer will be used by the mapping function
tokenizer = None


def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        )
        for convo in convos
    ]
    return {"text": texts}


# --- Load 4-bit Model and Train ---
print("Loading 4-bit Mxfp4 gpt-oss model for training...")
max_seq_length = 1024
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/gpt-oss-20b", max_seq_length = max_seq_length, load_in_4bit = True
)

dataset = load_dataset("HuggingFaceH4/Multilingual-Thinking", split = "train[:50]").map(
    formatting_prompts_func, batched = True
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha = 16,
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        max_steps = 10,
        learning_rate = 2e-4,
        output_dir = "outputs",
        report_to = "none",
    ),
)

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# --- Merge and Save ---
print("\n💾 Merging and saving the 16-bit model to './gpt-oss-finetuned-merged'...")
model.save_pretrained_merged(
    save_directory = "./gpt-oss-finetuned-merged", tokenizer = tokenizer
)
print("✅ Model merged and saved.")

# --- Cleanup ---
print("\n🧹 Cleaning up training artifacts...")
del model, trainer, tokenizer, dataset
torch.cuda.empty_cache()
gc.collect()

safe_remove_directory("./outputs")
safe_remove_directory(
    "./unsloth_compiled_cache"
)  # Clean up the cache created by this process
print("✅ Cleanup complete. Exiting training script.")
