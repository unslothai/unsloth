#!/usr/bin/env python3
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Comprehensive Unsloth Fine-Tuning Script
==========================================

A complete training script supporting:
- Platform detection (CUDA/MPS/CPU)
- LoRA fine-tuning with configurable parameters
- Synthetic and HuggingFace dataset support
- Model saving (adapter, merged, GGUF)

Usage:
    # Quick test with synthetic dataset
    python train.py --use_synthetic_data --max_steps 10

    # Fine-tune with HuggingFace dataset
    python train.py --dataset yahma/alpaca-cleaned --max_steps 100

    # Full training with GGUF export
    python train.py --dataset yahma/alpaca-cleaned --save_gguf --quantization_method q4_k_m

For all options:
    python train.py --help
"""

from __future__ import annotations

import unsloth

import argparse
import os
import platform
import sys
import shutil
from typing import Optional, List, Any

import torch


def detect_platform() -> dict:
    """
    Detect the current platform and available hardware.
    
    Returns:
        Dictionary with platform information including:
        - device_type: 'cuda', 'mps', or 'cpu'
        - is_mps: bool
        - is_cuda: bool
        - is_cpu: bool
        - supports_4bit: bool
        - supports_8bit: bool
        - recommended_dtype: torch.dtype
        - recommended_optimizer: str
    """
    info = {
        "device_type": "cpu",
        "is_mps": False,
        "is_cuda": False,
        "is_cpu": True,
        "supports_4bit": False,
        "supports_8bit": False,
        "recommended_dtype": torch.float32,
        "recommended_optimizer": "adamw_torch",
        "gpu_name": None,
        "gpu_memory_gb": 0.0,
    }
    
    if torch.cuda.is_available():
        info["device_type"] = "cuda"
        info["is_cuda"] = True
        info["is_cpu"] = False
        info["supports_4bit"] = True
        info["supports_8bit"] = True
        
        if torch.cuda.is_bf16_supported():
            info["recommended_dtype"] = torch.bfloat16
        else:
            info["recommended_dtype"] = torch.float16
        
        info["recommended_optimizer"] = "adamw_8bit"
        
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_gb"] = props.total_memory / (1024**3)
        
    elif platform.system() == "Darwin" and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            info["device_type"] = "mps"
            info["is_mps"] = True
            info["is_cpu"] = False
            info["supports_4bit"] = False
            info["supports_8bit"] = False
            info["recommended_dtype"] = torch.bfloat16
            info["recommended_optimizer"] = "adamw_torch"
            
            try:
                import subprocess
                result = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    capture_output=True, text=True, check=True
                )
                info["gpu_name"] = result.stdout.strip()
                
                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True, text=True, check=True
                )
                info["gpu_memory_gb"] = int(result.stdout.strip()) / (1024**3)
            except Exception:
                info["gpu_name"] = "Apple Silicon"
    
    return info


def apply_mps_patches() -> None:
    """
    Apply Mac compatibility patches before importing unsloth.
    Must be called BEFORE importing unsloth or related libraries.
    """
    if platform.system() != "Darwin":
        return
    
    try:
        from patcher import patch_for_mac
        print("\n🍎 Apple Silicon detected - applying MPS compatibility patches...")
        patch_for_mac(verbose=False)
    except ImportError:
        pass


def create_synthetic_dataset(n_samples: int = 20) -> "Dataset":
    """
    Create a small synthetic dataset for quick testing.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        HuggingFace Dataset with 'text' field
    """
    from datasets import Dataset
    
    conversation_templates = [
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{answer}<|eot_id|>",
    ]
    
    qa_pairs = [
        ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."),
        ("Explain neural networks.", "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information in layers."),
        ("What is deep learning?", "Deep learning is a subset of machine learning using neural networks with many layers to analyze various factors of data."),
        ("Define overfitting.", "Overfitting occurs when a model learns the training data too well, including noise, leading to poor generalization on new data."),
        ("What is gradient descent?", "Gradient descent is an optimization algorithm that iteratively adjusts parameters to minimize a loss function."),
        ("Explain backpropagation.", "Backpropagation is an algorithm for training neural networks by calculating gradients and propagating errors backward through the network."),
        ("What are hyperparameters?", "Hyperparameters are parameters set before training that control the learning process, like learning rate and batch size."),
        ("What is regularization?", "Regularization techniques prevent overfitting by adding constraints or penalties to the model during training."),
        ("Explain batch normalization.", "Batch normalization is a technique that normalizes layer inputs to improve training speed and stability."),
        ("What is transfer learning?", "Transfer learning uses a pre-trained model on a new task, leveraging learned features to improve performance with less data."),
    ]
    
    data = []
    for i in range(n_samples):
        qa = qa_pairs[i % len(qa_pairs)]
        text = conversation_templates[0].format(question=qa[0], answer=qa[1])
        data.append({"text": text})
    
    return Dataset.from_list(data)


def load_huggingface_dataset(
    dataset_name: str,
    dataset_split: str = "train",
    text_field: Optional[str] = None,
    prompt_template: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> "Dataset":
    """
    Load and optionally format a HuggingFace dataset.
    
    Args:
        dataset_name: Name of the dataset on HuggingFace Hub
        dataset_split: Dataset split to use
        text_field: Field containing text (if already formatted)
        prompt_template: Template for formatting prompts
        max_samples: Maximum number of samples to use
        
    Returns:
        Formatted HuggingFace Dataset with 'text' field
    """
    from datasets import load_dataset
    
    print(f"\n📂 Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split=dataset_split)
    
    if max_samples is not None and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"   Limited to {max_samples} samples")
    
    if text_field is not None and text_field in dataset.column_names:
        dataset = dataset.rename_column(text_field, "text")
        return dataset
    
    if prompt_template:
        def format_example(example):
            return {"text": prompt_template.format(**example)}
        dataset = dataset.map(format_example)
    elif "text" not in dataset.column_names:
        if "instruction" in dataset.column_names:
            def format_alpaca(example):
                instruction = example.get("instruction", "")
                input_text = example.get("input", "")
                output = example.get("output", "")
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                return {"text": text}
            dataset = dataset.map(format_alpaca)
    
    print(f"   Loaded {len(dataset)} samples")
    return dataset


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the training script.
    
    Returns:
        Namespace containing all parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Unsloth Fine-Tuning Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test with synthetic data
    python train.py --use_synthetic_data --max_steps 10
    
    # Fine-tune on Alpaca dataset
    python train.py --dataset yahma/alpaca-cleaned --max_steps 100
    
    # Full training with GGUF export
    python train.py --dataset yahma/alpaca-cleaned --save_gguf --quantization_method q4_k_m
    
    # Use 8-bit quantization on CUDA
    python train.py --load_in_8bit --dataset yahma/alpaca-cleaned
        """
    )
    
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model_name", type=str, default="unsloth/Llama-3.2-1B-Instruct",
        help="Model name or path (default: unsloth/Llama-3.2-1B-Instruct)"
    )
    model_group.add_argument(
        "--max_seq_length", type=int, default=2048,
        help="Maximum sequence length (default: 2048)"
    )
    model_group.add_argument(
        "--dtype", type=str, default=None,
        help="Data type: 'float16', 'bfloat16', or None for auto (default: None)"
    )
    model_group.add_argument(
        "--load_in_4bit", action="store_true",
        help="Load model in 4-bit quantization (CUDA only)"
    )
    model_group.add_argument(
        "--load_in_8bit", action="store_true",
        help="Load model in 8-bit quantization (CUDA only)"
    )
    model_group.add_argument(
        "--full_finetuning", action="store_true",
        help="Enable full fine-tuning (no LoRA)"
    )
    
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument(
        "--lora_r", type=int, default=16,
        help="LoRA rank (default: 16)"
    )
    lora_group.add_argument(
        "--lora_alpha", type=int, default=16,
        help="LoRA alpha (default: 16)"
    )
    lora_group.add_argument(
        "--lora_dropout", type=float, default=0.0,
        help="LoRA dropout (default: 0.0)"
    )
    lora_group.add_argument(
        "--target_modules", type=str, nargs="+", default=None,
        help="Target modules for LoRA (default: auto-detect)"
    )
    lora_group.add_argument(
        "--use_rslora", action="store_true",
        help="Use rank-stabilized LoRA"
    )
    
    dataset_group = parser.add_argument_group("Dataset Configuration")
    dataset_group.add_argument(
        "--dataset", type=str, default=None,
        help="HuggingFace dataset name (e.g., yahma/alpaca-cleaned)"
    )
    dataset_group.add_argument(
        "--dataset_split", type=str, default="train",
        help="Dataset split to use (default: train)"
    )
    dataset_group.add_argument(
        "--use_synthetic_data", action="store_true",
        help="Use synthetic dataset for testing"
    )
    dataset_group.add_argument(
        "--synthetic_samples", type=int, default=20,
        help="Number of synthetic samples (default: 20)"
    )
    dataset_group.add_argument(
        "--max_samples", type=int, default=None,
        help="Maximum samples from dataset (default: all)"
    )
    dataset_group.add_argument(
        "--text_field", type=str, default=None,
        help="Text field name in dataset (default: auto-detect)"
    )
    
    training_group = parser.add_argument_group("Training Configuration")
    training_group.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Output directory for model (default: outputs)"
    )
    training_group.add_argument(
        "--per_device_train_batch_size", type=int, default=2,
        help="Batch size per device (default: 2)"
    )
    training_group.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Gradient accumulation steps (default: 4)"
    )
    training_group.add_argument(
        "--max_steps", type=int, default=-1,
        help="Maximum training steps (-1 for full epochs) (default: -1)"
    )
    training_group.add_argument(
        "--num_train_epochs", type=int, default=1,
        help="Number of training epochs (default: 1)"
    )
    training_group.add_argument(
        "--learning_rate", type=float, default=2e-4,
        help="Learning rate (default: 2e-4)"
    )
    training_group.add_argument(
        "--warmup_steps", type=int, default=10,
        help="Warmup steps (default: 10)"
    )
    training_group.add_argument(
        "--warmup_ratio", type=float, default=None,
        help="Warmup ratio (overrides warmup_steps)"
    )
    training_group.add_argument(
        "--logging_steps", type=int, default=1,
        help="Logging steps (default: 1)"
    )
    training_group.add_argument(
        "--save_steps", type=int, default=100,
        help="Save steps (default: 100)"
    )
    training_group.add_argument(
        "--save_total_limit", type=int, default=2,
        help="Total checkpoint limit (default: 2)"
    )
    training_group.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay (default: 0.01)"
    )
    training_group.add_argument(
        "--lr_scheduler_type", type=str, default="linear",
        help="Learning rate scheduler type (default: linear)"
    )
    training_group.add_argument(
        "--optim", type=str, default=None,
        help="Optimizer (default: auto-detect based on platform)"
    )
    training_group.add_argument(
        "--seed", type=int, default=3407,
        help="Random seed (default: 3407)"
    )
    training_group.add_argument(
        "--report_to", type=str, default="none",
        help="Reporting tool: 'none', 'tensorboard', 'wandb' (default: none)"
    )
    training_group.add_argument(
        "--gradient_checkpointing", type=str, default="unsloth",
        help="Gradient checkpointing: 'unsloth', True, False (default: unsloth)"
    )
    
    save_group = parser.add_argument_group("Saving Configuration")
    save_group.add_argument(
        "--save_model", action="store_true",
        help="Save the trained model (LoRA adapter)"
    )
    save_group.add_argument(
        "--save_merged", action="store_true",
        help="Save merged model (LoRA + base)"
    )
    save_group.add_argument(
        "--save_gguf", action="store_true",
        help="Export model to GGUF format"
    )
    save_group.add_argument(
        "--quantization_method", type=str, default="q8_0",
        help="GGUF quantization method (default: q8_0)"
    )
    save_group.add_argument(
        "--push_to_hub", action="store_true",
        help="Push model to HuggingFace Hub"
    )
    save_group.add_argument(
        "--hub_model_id", type=str, default=None,
        help="HuggingFace Hub model ID for pushing"
    )
    save_group.add_argument(
        "--hub_token", type=str, default=None,
        help="HuggingFace Hub token"
    )
    
    args = parser.parse_args()
    
    if not args.use_synthetic_data and args.dataset is None:
        print("⚠️  No dataset specified, using synthetic data for demonstration")
        args.use_synthetic_data = True
    
    return args


def main():
    """Main training function."""
    args = parse_arguments()
    
    print("=" * 70)
    print("           🦥 Unsloth Fine-Tuning Script")
    print("=" * 70)
    
    platform_info = detect_platform()
    
    print(f"\n🖥️  Platform: {platform_info['device_type'].upper()}")
    if platform_info["gpu_name"]:
        print(f"   Device: {platform_info['gpu_name']}")
        if platform_info["gpu_memory_gb"] > 0:
            print(f"   Memory: {platform_info['gpu_memory_gb']:.1f} GB")
    
    if platform_info["is_mps"]:
        print("   🍎 Apple Silicon detected - MPS mode enabled")
        print("   Note: 4-bit/8-bit quantization not supported on MPS")
        print("   Using bfloat16 and adamw_torch optimizer")
        apply_mps_patches()
    
    print(f"\n📦 Loading model: {args.model_name}")
    
    if platform_info["is_mps"]:
        load_in_4bit = False
        load_in_8bit = False
    else:
        load_in_4bit = args.load_in_4bit
        load_in_8bit = args.load_in_8bit
    
    if args.dtype is not None:
        dtype = getattr(torch, args.dtype, None)
    else:
        dtype = None
    
    from unsloth import FastLanguageModel
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit and not args.load_in_8bit,
        load_in_8bit=load_in_8bit,
        full_finetuning=args.full_finetuning,
    )
    
    print("   ✓ Model loaded successfully")
    
    if not args.full_finetuning:
        print(f"\n🔧 Configuring LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
        
        target_modules = args.target_modules
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            target_modules=target_modules,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing=args.gradient_checkpointing,
            random_state=args.seed,
            use_rslora=args.use_rslora,
            loftq_config=None,
        )
        print("   ✓ LoRA adapters configured")
    
    print("\n📊 Preparing dataset...")
    if args.use_synthetic_data:
        print(f"   Using synthetic dataset ({args.synthetic_samples} samples)")
        dataset = create_synthetic_dataset(args.synthetic_samples)
    else:
        dataset = load_huggingface_dataset(
            dataset_name=args.dataset,
            dataset_split=args.dataset_split,
            text_field=args.text_field,
            max_samples=args.max_samples,
        )
    print(f"   Dataset size: {len(dataset)} samples")
    
    model_dtype = getattr(model.config, "torch_dtype", None)
    if model_dtype is None:
        model_dtype = model.dtype
    
    is_bf16 = model_dtype == torch.bfloat16
    is_fp16 = model_dtype == torch.float16
    
    optim = args.optim
    if optim is None:
        optim = platform_info["recommended_optimizer"]
    
    print(f"\n⚙️  Training configuration:")
    print(f"   Optimizer: {optim}")
    print(f"   Precision: bf16={is_bf16}, fp16={is_fp16}")
    print(f"   Batch size: {args.per_device_train_batch_size}")
    print(f"   Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"   Learning rate: {args.learning_rate}")
    
    from transformers import TrainingArguments
    from trl import SFTTrainer
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        num_train_epochs=args.num_train_epochs if args.max_steps < 0 else 1,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps if args.warmup_ratio is None else 0,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=optim,
        seed=args.seed,
        fp16=is_fp16,
        bf16=is_bf16,
        report_to=args.report_to,
        gradient_checkpointing=args.gradient_checkpointing == True,
    )
    
    print("\n🚀 Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        args=training_args,
    )
    
    print("\n🔥 Starting training...")
    print("=" * 70)
    
    try:
        trainer.train()
        print("\n" + "=" * 70)
        print("   ✅ Training completed successfully!")
        print("=" * 70)
    except Exception as e:
        print(f"\n   ❌ Training failed: {e}")
        raise
    
    if args.save_model or args.save_merged or args.save_gguf or args.push_to_hub:
        print("\n💾 Saving model...")
        
        if args.save_model:
            adapter_path = os.path.join(args.output_dir, "lora_adapter")
            print(f"   Saving LoRA adapter to: {adapter_path}")
            model.save_pretrained(adapter_path)
            tokenizer.save_pretrained(adapter_path)
            print("   ✓ LoRA adapter saved")
        
        if args.save_merged:
            merged_path = os.path.join(args.output_dir, "merged_model")
            print(f"   Saving merged model to: {merged_path}")
            model.save_pretrained_merged(
                merged_path,
                tokenizer,
                save_method="merged_16bit",
            )
            print("   ✓ Merged model saved")
        
        if args.save_gguf:
            gguf_path = os.path.join(args.output_dir, "gguf")
            print(f"   Exporting to GGUF: {gguf_path}")
            print(f"   Quantization method: {args.quantization_method}")
            model.save_pretrained_gguf(
                gguf_path,
                tokenizer,
                quantization_method=args.quantization_method,
            )
            print("   ✓ GGUF model exported")
        
        if args.push_to_hub:
            if args.hub_model_id is None:
                args.hub_model_id = args.model_name.split("/")[-1] + "-finetuned"
            print(f"   Pushing to HuggingFace Hub: {args.hub_model_id}")
            model.push_to_hub(
                args.hub_model_id,
                token=args.hub_token,
            )
            tokenizer.push_to_hub(
                args.hub_model_id,
                token=args.hub_token,
            )
            print("   ✓ Model pushed to Hub")
    
    print("\n" + "=" * 70)
    print("   🎉 All operations completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
