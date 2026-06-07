from unsloth import FastLanguageModel
from unsloth import FastVisionModel
import torch
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig

import dataP1


max_seq_length = 16384  # 可根据你的显存和需求调整
dtype = None  # None 表示自动检测（T4 上为 Float16，A100 上为 BFloat16）
load_in_4bit = False  # 使用 4bit 量化大幅降低显存占用
batch_size = 4

Seed = 3407


ModelPath = "/home/git/Qwen3.5-2B.new"  # it is a unsloth version of Qwen3.5-2b

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = ModelPath,
    load_in_4bit = load_in_4bit,
    use_gradient_checkpointing = "unsloth",
)

print(tokenizer.__dict__)

model = FastLanguageModel.get_peft_model(
    model,
    finetune_vision_layers = True,  # False if not finetuning vision layers
    finetune_language_layers = True,  # False if not finetuning language layers
    finetune_attention_modules = True,  # False if not finetuning attention layers
    finetune_mlp_modules = True,  # False if not finetuning mlp modules
)

data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)
