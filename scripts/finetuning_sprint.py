#!/usr/bin/env python
"""
100-Step Finetuning Sprint: Unsloth (MLX) vs Pure PyTorch
=========================================================
This script runs a head-to-head comparison of:
1. Pure PyTorch/HuggingFace training (baseline)
2. Unsloth-optimized training (MLX-accelerated on Apple Silicon)

Both use the same model, dataset, and hyperparameters.
"""

import os
import time
import gc
import torch

# Disable tokenizer parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Colors for terminal output
class colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def log_header(text):
    print(f"\n{colors.HEADER}{colors.BOLD}{'='*80}")
    print(f" {text}".center(80))
    print(f"{'='*80}{colors.ENDC}")


def log_result(name, time_sec, steps, tokens_per_sec=None, extra=""):
    tps_str = f" | {tokens_per_sec:.1f} tok/s" if tokens_per_sec else ""
    print(f" {name:<35} | Time: {time_sec:>7.2f}s | Steps: {steps}{tps_str} {extra}")


def clean_gpu_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def get_dataset():
    """Create a simple synthetic dataset for the sprint."""
    from datasets import Dataset
    
    # Create synthetic training data (simple instruction-following)
    data = []
    templates = [
        "Explain the concept of {}.",
        "What is {}?",
        "Describe {} in simple terms.",
        "How does {} work?",
        "Write a short paragraph about {}.",
    ]
    topics = [
        "machine learning", "neural networks", "transformers", "attention mechanisms",
        "gradient descent", "backpropagation", "tokenization", "embeddings",
        "fine-tuning", "LoRA adapters", "quantization", "mixed precision",
        "optimization", "loss functions", "activation functions", "batch normalization",
    ]
    
    for i, topic in enumerate(topics):
        for template in templates:
            prompt = template.format(topic)
            response = f"Here is an explanation of {topic}: " + \
                       f"{topic.capitalize()} is a fundamental concept in AI. " * 5
            data.append({
                "instruction": prompt,
                "output": response,
            })
    
    return Dataset.from_list(data)


def run_pytorch_baseline(model_name, num_steps=100, batch_size=2, max_seq_len=256):
    """Run pure PyTorch/HuggingFace training without Unsloth."""
    log_header(f"PyTorch Baseline: {model_name}")
    
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from peft import get_peft_model, LoraConfig, TaskType
    
    # Load model without Unsloth
    print(f" Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Determine device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device if device != "mps" else None,  # MPS doesn't support device_map
    )
    if device == "mps":
        model = model.to(device)
    
    # Apply LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Get dataset
    dataset = get_dataset()
    
    # Format dataset
    def formatting_func(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"text": text}
    
    dataset = dataset.map(formatting_func)
    
    # Tokenize
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=["instruction", "output", "text"])
    dataset = dataset.with_format("torch")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./pytorch_baseline_output",
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        logging_steps=10,
        save_steps=1000,  # Don't save during sprint
        fp16=True if device == "cuda" else False,
        bf16=False,
        optim="adamw_torch",
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=5,
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # Time the training
    clean_gpu_cache()
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    elapsed = end_time - start_time
    tokens_processed = num_steps * batch_size * max_seq_len
    tokens_per_sec = tokens_processed / elapsed
    
    log_result("PyTorch + PEFT Baseline", elapsed, num_steps, tokens_per_sec)
    
    # Cleanup
    del model, trainer
    clean_gpu_cache()
    
    return elapsed, tokens_per_sec


def run_unsloth_optimized(model_name, num_steps=100, batch_size=2, max_seq_len=256):
    """Run Unsloth-optimized training (MLX-accelerated on Apple Silicon)."""
    log_header(f"Unsloth Optimized: {model_name}")
    
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    # Load model with Unsloth
    print(f" Loading model with Unsloth: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_len,
        dtype=None,  # Auto-detect
        load_in_4bit=True,  # 4-bit quantization for speed
    )
    
    # Apply LoRA with Unsloth's optimized adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's memory optimization
    )
    
    # Get dataset
    dataset = get_dataset()
    
    # Format dataset for Unsloth
    def formatting_func(examples):
        texts = []
        for instruction, output in zip(examples["instruction"], examples["output"]):
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            texts.append(text)
        return {"text": texts}
    
    dataset = dataset.map(formatting_func, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./unsloth_output",
        max_steps=num_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        logging_steps=10,
        save_steps=1000,  # Don't save during sprint
        fp16=False,  # Unsloth handles precision
        bf16=False,
        optim="adamw_8bit",  # Unsloth's 8-bit optimizer
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=5,
        report_to="none",
    )
    
    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=False,  # Disable packing for fair comparison
    )
    
    # Time the training
    clean_gpu_cache()
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    
    elapsed = end_time - start_time
    tokens_processed = num_steps * batch_size * max_seq_len
    tokens_per_sec = tokens_processed / elapsed
    
    log_result("Unsloth Optimized", elapsed, num_steps, tokens_per_sec, 
               f"{colors.GREEN}[4-bit + MLX]{colors.ENDC}")
    
    # Cleanup
    del model, trainer
    clean_gpu_cache()
    
    return elapsed, tokens_per_sec


def main():
    log_header("100-Step Finetuning Sprint: Unsloth vs PyTorch")
    
    # Configuration
    MODEL_NAME = "unsloth/Llama-3.2-1B"  # Small model for the sprint
    NUM_STEPS = 100
    BATCH_SIZE = 2
    MAX_SEQ_LEN = 256
    
    print(f"\n{colors.YELLOW}Configuration:{colors.ENDC}")
    print(f" Model:      {MODEL_NAME}")
    print(f" Steps:      {NUM_STEPS}")
    print(f" Batch Size: {BATCH_SIZE}")
    print(f" Max Seq:    {MAX_SEQ_LEN}")
    
    # Check device
    if torch.backends.mps.is_available():
        print(f" Device:     {colors.GREEN}Apple Silicon (MPS){colors.ENDC}")
    elif torch.cuda.is_available():
        print(f" Device:     {colors.GREEN}CUDA{colors.ENDC}")
    else:
        print(f" Device:     {colors.RED}CPU (Warning: Very Slow){colors.ENDC}")
    
    # Run sprints
    results = {}
    
    # 1. PyTorch Baseline
    try:
        pytorch_time, pytorch_tps = run_pytorch_baseline(
            MODEL_NAME, NUM_STEPS, BATCH_SIZE, MAX_SEQ_LEN
        )
        results["pytorch"] = {"time": pytorch_time, "tps": pytorch_tps}
    except Exception as e:
        print(f"{colors.RED}PyTorch baseline failed: {e}{colors.ENDC}")
        results["pytorch"] = None
    
    # 2. Unsloth Optimized
    try:
        unsloth_time, unsloth_tps = run_unsloth_optimized(
            MODEL_NAME, NUM_STEPS, BATCH_SIZE, MAX_SEQ_LEN
        )
        results["unsloth"] = {"time": unsloth_time, "tps": unsloth_tps}
    except Exception as e:
        print(f"{colors.RED}Unsloth optimized failed: {e}{colors.ENDC}")
        results["unsloth"] = None
    
    # Summary
    log_header("Sprint Results Summary")
    
    if results.get("pytorch") and results.get("unsloth"):
        pytorch_time = results["pytorch"]["time"]
        unsloth_time = results["unsloth"]["time"]
        speedup = pytorch_time / unsloth_time
        
        print(f"\n{colors.BOLD}  PyTorch Baseline:{colors.ENDC}")
        print(f"    Time: {pytorch_time:.2f}s | {results['pytorch']['tps']:.1f} tok/s")
        
        print(f"\n{colors.BOLD}  Unsloth Optimized:{colors.ENDC}")
        print(f"    Time: {unsloth_time:.2f}s | {results['unsloth']['tps']:.1f} tok/s")
        
        if speedup > 1:
            print(f"\n{colors.GREEN}{colors.BOLD}  🏆 WINNER: Unsloth ({speedup:.2f}x faster!){colors.ENDC}")
        else:
            print(f"\n{colors.YELLOW}{colors.BOLD}  🏆 WINNER: PyTorch ({1/speedup:.2f}x faster){colors.ENDC}")
    elif results.get("unsloth"):
        print(f"\n{colors.YELLOW}Only Unsloth completed successfully.{colors.ENDC}")
        print(f"Time: {results['unsloth']['time']:.2f}s | {results['unsloth']['tps']:.1f} tok/s")
    elif results.get("pytorch"):
        print(f"\n{colors.YELLOW}Only PyTorch completed successfully.{colors.ENDC}")
        print(f"Time: {results['pytorch']['time']:.2f}s | {results['pytorch']['tps']:.1f} tok/s")
    else:
        print(f"\n{colors.RED}Both sprints failed!{colors.ENDC}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
