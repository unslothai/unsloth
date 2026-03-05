# -*- coding: utf-8 -*-

from unsloth import FastVisionModel

import torch
from qwen_vl_utils import process_vision_info
import os
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(REPO_ROOT))

from tests.utils.cleanup_utils import safe_remove_directory
from tests.utils.ocr_eval import OCRModelEvaluator


## Dataset Preparation
from datasets import load_dataset

dataset = load_dataset("lbourdois/OCR-liboaccn-OPUS-MIT-5M-clean", "en", split = "train")
# To select the first 2000 examples
train_dataset = dataset.select(range(2000))

# To select the next 200 examples for evaluation
eval_dataset = dataset.select(range(2000, 2200))


# Convert dataset to OAI messages
def format_data(sample):
    return {
        "messages": [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": sample["question"],
                    },
                    {
                        "type": "image",
                        "image": sample["image"],
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}],
            },
        ],
    }


system_message = "You are an expert french ocr system."
# Convert dataset to OAI messages
# need to use list comprehension to keep Pil.Image type, .mape convert image to bytes
train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]

## Setup OCR main evaluation function and helpers
import os
import torch
from tqdm import tqdm
import pandas as pd
from jiwer import wer, cer
from qwen_vl_utils import process_vision_info

#
ocr_evaluator = OCRModelEvaluator()
model_comparison_results = {}

## Finetuning Setup and Run
# Load Base Model

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen2-VL-7B-Instruct",
    max_seq_length = 2048,  # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False,  # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False,  # [NEW!] We have full finetuning now!
)

# benchmark base model performance
model_name = "Unsloth Base model"
FastVisionModel.for_inference(model)
avg_wer, avg_cer = ocr_evaluator.evaluate_model(
    model, tokenizer, eval_dataset, output_dir = "unsloth_base_model_results"
)
ocr_evaluator.add_to_comparison(model_name, avg_wer, avg_cer)

## Lora Finetuning
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = True,  # Turn off for just text!
    finetune_language_layers = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules = True,  # SHould leave on always!
    r = 16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    # "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 32,
    lora_dropout = 0,  # Supports any, but = 0 is optimized
    bias = "none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth",  # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None,  # And LoftQ
)

from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

FastVisionModel.for_training(model)  # Enable for training!
model.config.use_cache = False


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = train_dataset,
    args = SFTConfig(
        # per_device_train_batch_size = 4,
        # gradient_accumulation_steps = 8,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,
        gradient_checkpointing_kwargs = {
            "use_reentrant": False
        },  # use reentrant checkpointing
        max_grad_norm = 0.3,  # max gradient norm based on QLoRA paper
        warmup_ratio = 0.03,
        # num_train_epochs = 2, # Set this instead of max_steps for full training runs
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 5,
        save_strategy = "epoch",
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "unsloth-qwen2-7vl-french-ocr-checkpoints",
        report_to = "none",  # For Weights and Biases
        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

# run training
trainer_stats = trainer.train()

model.save_pretrained("unsloth-qwen2-7vl-french-ocr-adapter", tokenizer)
tokenizer.save_pretrained("unsloth-qwen2-7vl-french-ocr-adapter")

## Measure Adapter Performance

# benchmark lora model performance
model_name = "Unsloth lora adapter model"
FastVisionModel.for_inference(model)
avg_wer, avg_cer = ocr_evaluator.evaluate_model(
    model, tokenizer, eval_dataset, output_dir = "unsloth_lora_model_results"
)
ocr_evaluator.add_to_comparison(model_name, avg_wer, avg_cer)

## Merge Model


def find_lora_base_model(model_to_inspect):
    current = model_to_inspect
    if hasattr(current, "base_model"):
        current = current.base_model
    if hasattr(current, "model"):
        current = current.model
    return current


base = find_lora_base_model(model)

print((base.__class__.__name__))

# merge default 16 bits
model.save_pretrained_merged(
    save_directory = "qwen2-ocr-merged-finetune-merge-16bit", tokenizer = tokenizer
)


## Benchmark merged model performance

### 16 bits merged model

model, tokenizer = FastVisionModel.from_pretrained(
    "./qwen2-ocr-merged-finetune-merge-16bit", load_in_4bit = False, load_in_8bit = False
)

# benchmark 4bit loaded, 16bits merged model performance
model_name = "Unsloth 16bits-merged model load-16bits"
model.config.use_cache = True

avg_wer, avg_cer = ocr_evaluator.evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    output_dir = "unsloth_16bits_merged_model_load_16bits_results",
)
ocr_evaluator.add_to_comparison(model_name, avg_wer, avg_cer)

# load 16bits-merged model in 4 bits
model, tokenizer = FastVisionModel.from_pretrained(
    "./qwen2-ocr-merged-finetune-merge-16bit", load_in_4bit = True, load_in_8bit = False
)

# benchmark 4bit loaded, 16bits merged model performance
model_name = "Unsloth 16bits-merged model load-4bits"
model.config.use_cache = True

avg_wer, avg_cer = ocr_evaluator.evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    output_dir = "unsloth_16bits_merged_model_load_4bits_results",
)
ocr_evaluator.add_to_comparison(model_name, avg_wer, avg_cer)

# load model in 8 bits
model, tokenizer = FastVisionModel.from_pretrained(
    "./qwen2-ocr-merged-finetune-merge-16bit", load_in_4bit = False, load_in_8bit = True
)

# benchmark 4bit loaded, 16bits merged model performance
model_name = "Unsloth 16bits-merged model load-8bits"
avg_wer, avg_cer = ocr_evaluator.evaluate_model(
    model,
    tokenizer,
    eval_dataset,
    output_dir = "unsloth_16bits_merged_model_load_8bits_results",
)
ocr_evaluator.add_to_comparison(model_name, avg_wer, avg_cer)

# """### 4 bits merged model"""
#
# # load 4bits-merged model in 4 bits
# model, tokenizer = FastVisionModel.from_pretrained("./qwen2-ocr-merged-finetune-merge-4bit",load_in_4bit=True, load_in_8bit=False)
#
# # benchmark 4bit loaded, 4bits merged model performance
# model_name = "Unsloth 4bits-merged model load-4bits"
#
# avg_wer, avg_cer = ocr_evaluator.evaluate_model(model, tokenizer, eval_dataset, output_dir="unsloth_4bits_merged_model_load_4bits_results")
# ocr_evaluator.add_to_comparison(model_name, avg_wer, avg_cer)
#
# # load model in 8 bits
# model, tokenizer = FastVisionModel.from_pretrained("./qwen2-ocr-merged-finetune-merge-4bit",load_in_4bit=False, load_in_8bit=True)
#
# # benchmark 8bit loaded, 4bits merged model performance
# model_name = "Unsloth 4bits-merged model load-8bits"
#
# avg_wer, avg_cer = ocr_evaluator.evaluate_model(model, tokenizer, eval_dataset, output_dir="unsloth_4bits_merged_model_load_8bits_results")
# ocr_evaluator.add_to_comparison(model_name, avg_wer, avg_cer)

# Model comparison report
# print model comparison
ocr_evaluator.print_model_comparison()


# Final cleanup
print("\n🧹 Cleaning up temporary files...")
safe_remove_directory("./unsloth-qwen2-7vl-french-ocr-adapter")
safe_remove_directory("./unsloth-qwen2-7vl-french-ocr-checkpoints")
safe_remove_directory("./unsloth_compiled_cache")
safe_remove_directory("./qwen2-ocr-merged-finetune-merge-16bit")

print("\n🎯 Pipeline completed successfully!")
print("=" * 80)
