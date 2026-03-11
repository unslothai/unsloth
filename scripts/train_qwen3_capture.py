"""
Standalone training script: Qwen3-4B-Thinking with activation capture.
Mirrors the Qwen3_(4B)_Thinking.ipynb workflow via CLI.
"""
import sys, os
# RTX 5060 Ti (Blackwell sm_100): disable inductor JIT to avoid compilation errors
os.environ["TORCHDYNAMO_DISABLE"] = "1"
sys.path.insert(0, "/mnt/m/Unsloth_Work/unsloth")

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from unsloth import ActivationCaptureConfig, ActivationCapture, ActivationCaptureCallback
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig
import torch

# ── Model ──────────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Qwen3-4B-Thinking-2507",
    max_seq_length = 2048,
    load_in_4bit   = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r                        = 32,
    target_modules           = ["q_proj", "k_proj", "v_proj", "o_proj",
                                 "gate_proj", "up_proj", "down_proj"],
    lora_alpha               = 32,
    lora_dropout             = 0,
    bias                     = "none",
    use_gradient_checkpointing = "unsloth",
    random_state             = 3407,
)

# ── Tokenizer / chat template ──────────────────────────────────────────────────
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-thinking")

# ── Dataset ────────────────────────────────────────────────────────────────────
dataset = load_dataset("unsloth/OpenMathReasoning-mini", split="cot")

def build_conversations(examples):
    conversations = []
    for problem, solution in zip(examples["problem"], examples["generated_solution"]):
        conversations.append([
            {"role": "user",      "content": problem},
            {"role": "assistant", "content": solution},
        ])
    return {"conversations": conversations}

dataset = dataset.map(build_conversations, batched=True)

def apply_template(examples):
    texts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=False)
        for c in examples["conversations"]
    ]
    return {"text": texts}

dataset = dataset.map(apply_template, batched=True)

# ── Activation capture ────────────────────────────────────────────────────────
capture_cfg = ActivationCaptureConfig(
    output_dir       = "/home/leo/qwen3_activation_logs",
    capture_interval = 5,
    max_channels     = 64,
)
capture          = ActivationCapture(model, capture_cfg)
capture_callback = ActivationCaptureCallback(capture)

# ── Trainer ────────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model             = model,
    tokenizer         = tokenizer,
    train_dataset     = dataset,
    eval_dataset      = None,
    callbacks         = [capture_callback],
    args              = SFTConfig(
        dataset_text_field          = "text",
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps                = 5,
        max_steps                   = 60,
        learning_rate               = 2e-4,
        logging_steps               = 1,
        optim                       = "adamw_8bit",
        weight_decay                = 0.001,
        lr_scheduler_type           = "linear",
        seed                        = 3407,
        report_to                   = "none",
        output_dir                  = "/home/leo/qwen3_output",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part    = "<|im_start|>assistant\n",
)

# ── Train ──────────────────────────────────────────────────────────────────────
gpu = torch.cuda.get_device_properties(0)
print(f"GPU: {gpu.name} — {round(gpu.total_memory/1024**3,1)} GB")
trainer_stats = trainer.train()
print(f"Training done in {round(trainer_stats.metrics['train_runtime']/60,2)} min")
print(f"Activation logs → /home/leo/qwen3_activation_logs/")

# ── Generate viz ──────────────────────────────────────────────────────────────
import subprocess
result = subprocess.run(
    [sys.executable,
     "/mnt/m/Unsloth_Work/unsloth/visualize_activations.py",
     "/home/leo/qwen3_activation_logs",
     "--output", "/home/leo/qwen3_activation_logs/viz.html"],
    capture_output=True, text=True,
)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
else:
    print("Visualization → /home/leo/qwen3_activation_logs/viz.html")
