from unsloth import FastLanguageModel, FastVisionModel, UnslothVisionDataCollator
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTConfig
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    TrainingArguments,
)
from datasets import load_dataset, Dataset
import torch
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from multiprocessing import Process, Queue
import gc
import os
from huggingface_hub import HfFileSystem, hf_hub_download

# ruff: noqa
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.utils.cleanup_utils import safe_remove_directory
from tests.utils.perplexity_eval import (
    ppl_model,
    add_to_comparison,
    print_model_comparison,
)


# Define helper functions outside of main
def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize = False, add_generation_prompt = False
        )
        for convo in convos
    ]
    return {"text": texts}


if torch.cuda.is_bf16_supported():
    compute_dtype = torch.bfloat16
    attn_implementation = "flash_attention_2"
else:
    compute_dtype = torch.float16
    attn_implementation = "sdpa"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-Instruct",
    max_seq_length = 2048,
    dtype = compute_dtype,
    load_in_4bit = True,
    load_in_8bit = False,
    full_finetuning = False,
    attn_implementation = attn_implementation,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

from unsloth.chat_templates import standardize_sharegpt

dataset_train = load_dataset("allenai/openassistant-guanaco-reformatted", split = "train")
dataset_ppl = load_dataset("allenai/openassistant-guanaco-reformatted", split = "eval")

dataset_train = dataset_train.map(formatting_prompts_func, batched = True)
dataset_ppl = dataset_ppl.map(formatting_prompts_func, batched = True)

add_to_comparison("Base model 4 bits", ppl_model(model, tokenizer, dataset_ppl))

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = [
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "down_proj",
        "up_proj",
    ],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset_train,
    dataset_text_field = "text",
    max_seq_length = 2048,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        max_steps = 30,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 50,
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

from unsloth.chat_templates import train_on_responses_only

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# run training
trainer_stats = trainer.train()


# saving and merging the model to local disk
hf_username = os.environ.get("HF_USER", "")
if not hf_username:
    hf_username = input("Please enter your Hugging Face username: ").strip()
    os.environ["HF_USER"] = hf_username

hf_token = os.environ.get("HF_TOKEN", "")
if not hf_token:
    hf_token = input("Please enter your Hugging Face token: ").strip()
    os.environ["HF_TOKEN"] = hf_token


repo_name = f"{hf_username}/merged_llama_text_model"
success = {
    "upload": False,
    "safetensors_check": False,
    "download": False,
}

# Stage 1: Upload model to Hub
try:
    print("\n" + "=" * 80)
    print("=== UPLOADING MODEL TO HUB ===".center(80))
    print("=" * 80 + "\n")
    model.push_to_hub_merged(repo_name, tokenizer = tokenizer, token = hf_token)
    success["upload"] = True
    print("✅ Model uploaded successfully!")
except Exception as e:
    print(f"❌ Failed to upload model: {e}")
    raise Exception("Model upload failed.")

# Stage 2: Verify safetensors.index.json exists
try:
    print("\n" + "=" * 80)
    print("=== VERIFYING REPO CONTENTS ===".center(80))
    print("=" * 80 + "\n")
    fs = HfFileSystem(token = hf_token)
    file_list = fs.ls(repo_name, detail = True)
    safetensors_found = any(
        file["name"].endswith("model.safetensors.index.json") for file in file_list
    )
    if safetensors_found:
        success["safetensors_check"] = True
        print("✅ model.safetensors.index.json found in repo!")
    else:
        raise Exception("model.safetensors.index.json not found in repo.")
except Exception as e:
    print(f"❌ Verification failed: {e}")
    raise Exception("Repo verification failed.")

# Stage 3: Test downloading the model (even if cached)
safe_remove_directory("./RTannous")

try:
    print("\n" + "=" * 80)
    print("=== TESTING MODEL DOWNLOAD ===".center(80))
    print("=" * 80 + "\n")
    # Force download even if cached
    model, tokenizer = FastLanguageModel.from_pretrained(
        f"{hf_username}/merged_llama_text_model"
    )
    success["download"] = True
    print("✅ Model downloaded successfully!")
except Exception as e:
    print(f"❌ Download failed: {e}")
    raise Exception("Model download failed.")

# Final report
print("\n" + "=" * 80)
print("=== VALIDATION REPORT ===".center(80))
print("=" * 80 + "\n")
for stage, passed in success.items():
    status = "✓" if passed else "✗"
    print(f"{status} {stage.replace('_', ' ').title()}")
print("\n" + "=" * 80)

if all(success.values()):
    print("\n🎉 All stages completed successfully!")
else:
    raise Exception("Validation failed for one or more stages.")

# final cleanup
safe_remove_directory("./outputs")
safe_remove_directory("./unsloth_compiled_cache")
