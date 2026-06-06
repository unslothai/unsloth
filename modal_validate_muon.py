"""Modal GPU validation for Muon optimizer integration.
Run: modal run modal_validate_muon.py
"""

import modal
from pathlib import Path

LOCAL_UNSLOTH = Path("/home/keypa/dev/unsloth")

image = (
    modal.Image.debian_slim(python_version = "3.12")
    .pip_install(
        "torch==2.12.0",
        "transformers==5.5.0",
        "datasets==4.3.0",
        "trl==0.24.0",
        "peft==0.19.1",
        "accelerate==1.13.0",
        "huggingface_hub",
        "bitsandbytes==0.49.2",
        "sentencepiece",
        "unsloth_zoo==2026.5.4",
        extra_index_url = "https://download.pytorch.org/whl/cu124",
    )
    # Copy local unsloth source into image, then install it
    .add_local_dir(LOCAL_UNSLOTH, remote_path = "/root/unsloth", copy = True)
    .run_commands("pip install /root/unsloth --no-deps")
)

app = modal.App("unsloth-muon-validation", image = image)

VALIDATION_SCRIPT = r"""
import os, sys
os.environ["UNSLOTH_ALLOW_CPU"] = "0"

from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments, MuonConfig
from datasets import Dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen2.5-0.5B",
    max_seq_length=512,
    full_finetuning=True,
)

texts = [
    "Explain the concept of gravity.",
    "What is the capital of France?",
    "How do computers work?",
    "Write a short poem about nature.",
    "Describe the process of photosynthesis.",
    "What are the benefits of exercise?",
    "Explain how neural networks function.",
    "Describe the solar system.",
    "How does machine learning work?",
    "What is recursion in programming?",
] * 20
dataset = Dataset.from_dict({"text": texts})

def formatting_func(example):
    return [tokenizer.apply_chat_template(
        [{"role": "user", "content": t}],
        tokenize=False,
        add_generation_prompt=True,
    ) for t in example["text"]]

trainer = UnslothTrainer(
    model=model,
    tokenizer=tokenizer,
    args=UnslothTrainingArguments(
        muon_config=MuonConfig(momentum=0.95, ns_steps=5),
        output_dir="/output/muon_test",
        num_train_epochs=1,
        max_steps=50,
        per_device_train_batch_size=2,
        learning_rate=1e-4,
        logging_steps=10,
        report_to="none",
    ),
    train_dataset=dataset,
    formatting_func=formatting_func,
)

trainer.train()

losses = [log["loss"] for log in trainer.state.log_history if "loss" in log]
print(f"\n{'='*60}")
print(f"Muon validation completed: {len(losses)} logged losses")
for i, loss in enumerate(losses):
    print(f"  Step {i*10}: loss = {loss:.4f}")
if len(losses) >= 2:
    trend = "descending" if losses[-1] < losses[0] else "ASCENDING"
    print(f"Loss trend: {trend}")
    if losses[-1] < losses[0]:
        print("PASS: Loss decreased over training steps")
    else:
        print("WARNING: Loss did not decrease — check Muon hyperparameters")
print(f"{'='*60}")
"""


@app.function(
    gpu = "a100-40gb:1",
    timeout = 1800,
    volumes = {"/output": modal.Volume.from_name("muon-output", create_if_missing = True)},
)
def validate_muon():
    import subprocess, sys

    subprocess.run(
        [sys.executable, "-c", VALIDATION_SCRIPT],
        check = True,
        cwd = "/root",
    )


@app.local_entrypoint()
def main():
    validate_muon.remote()
    print("Done — check /output volume for checkpoints and logs")
