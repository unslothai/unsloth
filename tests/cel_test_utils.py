import argparse
import logging
from pathlib import Path

import torch
from cel_analysis import load_log_diffs
from peft import LoraConfig
from tabulate import tabulate
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from transformers.models.llama import LlamaConfig
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_utils import enable_full_determinism
from transformers.trainer_utils import set_seed as hf_set_seed
from trl import SFTTrainer

import unsloth.utils.data as data_utils
from unsloth.kernels.fused_cel import LlamaForCausalLMFusedCEL
from unsloth.kernels.fused_cel import patch_model as patch_model_fused_cel
from unsloth.models._utils import patch_tokenizer, prepare_model_for_kbit_training
from unsloth.utils.data import get_data_loader
from unsloth.utils.profiling import MetricsCallBack

parent_dir = Path(__file__).parent.absolute()
# SEED = 3407
# enable_full_determinism(SEED)
torch.autograd.set_detect_anomaly(True)


def get_quant_config(load_in_4bit, dtype):
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=dtype,
    )


def get_model(model_id, dtype, use_fused_cel_layer=True, quant_config=None):
    model_cls = (
        LlamaForCausalLMFusedCEL if use_fused_cel_layer else AutoModelForCausalLM
    )

    model = model_cls.from_pretrained(
        model_id,
        quantization_config=quant_config,
        torch_dtype=dtype,
    )
    return model


def get_tokenizer(model_id, max_seq_len):
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        model_max_length=max_seq_len,
        padding_side="right",
    )
    return tokenizer


def get_trainer_args(batch_size, grad_accum_steps, max_steps, dtype, seed, output_dir):
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        warmup_steps=1,
        max_steps=max_steps,
        learning_rate=2e-4,
        fp16=dtype == "float16",
        bf16=dtype == "bfloat16",
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=seed,
        data_seed=seed,
        output_dir=output_dir,
        overwrite_output_dir=True,
        # Metrics
        skip_memory_metrics=False,
    )
    return training_args


def get_peft_config(
    target_modules="all-linear",
    lora_alpha=8,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
):
    # accepted_modules = frozenset(
    #     (
    #         "q_proj",
    #         "k_proj",
    #         "v_proj",
    #         "o_proj",
    #         "gate_proj",
    #         "up_proj",
    #         "down_proj",
    #     ),
    # )

    peft_config = LoraConfig(
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        task_type=task_type,
    )
    return peft_config


def get_sft_trainer(
    model,
    tokenizer,
    dataset,
    peft_config,
    trainer_args,
    max_seq_len,
    file_prefix,
):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=trainer_args,
    )

    # Remove default callbacks, make less verbose
    trainer.remove_callback(ProgressCallback)
    trainer.model.enable_input_require_grads()
    # file_prefix = "fused_cel" if use_fused_cel else ""
    # file_prefix += "_" + args.dtype
    _ = trainer.add_callback(MetricsCallBack(name=file_prefix, verbose=False))
    return trainer
