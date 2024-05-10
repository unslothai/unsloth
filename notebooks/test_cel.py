import dataclasses
import json
import logging
import os
from pathlib import Path

import torch
from IPython.core.interactiveshell import InteractiveShell
from llama_head import CEL_only_forward
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.models.llama import LlamaConfig, LlamaForCausalLM
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_pt_utils import _secs2timedelta
from trl import SFTTrainer

import unsloth.utils.data as data_utils
import unsloth.utils.memory as memory_utils
import unsloth.utils.testing as test_utils
from unsloth.kernels import fused_cel
from unsloth.kernels.fused_cel import patch_model as patch_model_fused_cel
from unsloth.models._utils import patch_tokenizer, prepare_model_for_kbit_training
from unsloth.models.llama import FastLlamaModel
from unsloth.utils.profiling import MetricsCallBack

parent_dir = Path(__file__).parent.absolute()


logging.basicConfig(
    format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
)
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
)


model_config = LlamaConfig.from_pretrained(parent_dir / "llama-10m.json")
model = AutoModelForCausalLM.from_pretrained(
    parent_dir / "llama-10m",
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
)
# model = LlamaForCausalLM(model_config).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", model_max_length=4096, padding_side="right"
)
model, tokenizer = patch_tokenizer(model, tokenizer)

max_seq_length = 256

training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    warmup_steps=1,
    max_steps=1,
    learning_rate=2e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir="outputs",
    overwrite_output_dir=True,
    # Metrics
    skip_memory_metrics=False,
    include_num_input_tokens_seen=True,
    include_tokens_per_second=True,
)

accepted_modules = frozenset(
    (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ),
)

dataset = data_utils.get_alpaca(tokenizer)

peft_config = LoraConfig(
    target_modules=accepted_modules,
    lora_alpha=8,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)
# Need to set inputs require_grad (i.e., output of embeddings requires grad for fused_cel to work)
model = prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=training_args.gradient_checkpointing
)
patched_model = patch_model_fused_cel(model, use_fused_cel=True)

trainer = SFTTrainer(
    model=patched_model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=training_args,
)
# trainer.remove_callback(ProgressCallback)
# trainer.model.enable_input_require_grads()

# _ = trainer.add_callback(MetricsCallBack())
train_stats = trainer.train()
