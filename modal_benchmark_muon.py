"""Muon vs AdamW benchmark on Modal (A100-40GB).
Run: modal run modal_benchmark_muon.py
"""
import modal
from pathlib import Path

LOCAL_UNSLOTH = Path("/home/keypa/dev/unsloth")

image = (
    modal.Image.debian_slim(python_version="3.12")
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
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(LOCAL_UNSLOTH, remote_path="/root/unsloth", copy=True)
    .run_commands("pip install /root/unsloth --no-deps")
)

app = modal.App("unsloth-muon-benchmark", image=image)

BENCHMARK_SCRIPT = r"""
import os, sys, time, torch
os.environ["UNSLOTH_ALLOW_CPU"] = "0"

from unsloth import FastLanguageModel
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments, MuonConfig
from datasets import Dataset

# ---- helpers ----
def reset_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

def peak_memory_gib():
    return torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0

# ---- parameters ----
MODEL = "unsloth/Qwen2.5-0.5B"
STEPS = 200
LR = 2e-4
SEQ_LEN = 512
BATCH_SIZE = 2
GRAD_ACCUM = 4

# ---- dataset ----
text = "The quick brown fox jumps over the lazy dog. " * 20
texts = [text] * (STEPS * BATCH_SIZE)
dataset = Dataset.from_dict({"text": texts})

tokenizer_kwargs = dict(truncation=True, max_length=SEQ_LEN, padding="max_length",
                        return_tensors="pt")

def formatting_func(example):
    return example["text"]

# ---- configs ----
configs = [
    ("AdamW", None),
    ("Muon (ns=5)",  MuonConfig(ns_steps=5, muon_lr_scale=1.0)),
    ("Muon (ns=10)", MuonConfig(ns_steps=10, muon_lr_scale=1.0)),
    ("Muon (ns=20)", MuonConfig(ns_steps=20, muon_lr_scale=1.0)),
]

results = []

for label, muon_cfg in configs:
    print(f"\n{'='*72}")
    print(f"  Benchmark: {label}")
    print(f"{'='*72}")
    reset_memory()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=SEQ_LEN,
        full_finetuning=True,
    )

    training_args = UnslothTrainingArguments(
        output_dir="/tmp/muon_bench",
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=STEPS,
        logging_steps=STEPS,
        save_strategy="no",
        report_to="none",
        weight_decay=0.0,
        bf16=True,
        fp16=False,
        muon_config=muon_cfg,
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
    )

    t0 = time.time()
    train_result = trainer.train()
    t1 = time.time()

    mem = peak_memory_gib()
    elapsed = t1 - t0
    loss = train_result.training_loss if train_result else None

    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * SEQ_LEN
    total_tokens = STEPS * tokens_per_step
    throughput = total_tokens / elapsed

    print(f"  RESULT\toptimizer={label}\t"
          f"throughput_tok_s={throughput:.1f}\t"
          f"peak_mem_gib={mem:.2f}\t"
          f"loss={loss:.6f}\t"
          f"elapsed_s={elapsed:.1f}")

    results.append(dict(label=label, throughput=throughput, peak_mem=mem,
                        loss=loss, elapsed=elapsed))
    del trainer, model, tokenizer
    reset_memory()

# ---- summary ----
print(f"\n{'='*72}")
print(f"  Summary")
print(f"{'='*72}")
header = f"{'Optimizer':<24} {'Throughput':>14} {'Peak Mem':>10} {'Loss':>12} {'Time':>8}"
print(header)
print("-" * 68)
for r in results:
    loss_s = f"{r['loss']:.6f}" if r.get('loss') else "N/A"
    print(f"{r['label']:<24} {r['throughput']:>8.1f} tok/s  {r['peak_mem']:>6.2f} GiB  {loss_s:>12}  {r['elapsed']:>6.1f}s")

speedup = results[1]["throughput"] / results[0]["throughput"] if results[0]["throughput"] else 0
print(f"\nMuon(ns=5) vs AdamW throughput ratio: {speedup:.2f}x")
if results[0].get("loss") and results[1].get("loss"):
    print(f"Muon(ns=5) loss delta: {results[1]['loss'] - results[0]['loss']:.6f}")
print(f"{'='*72}")
"""


@app.function(
    gpu="a100-40gb:1",
    timeout=3600,
)
def benchmark_muon():
    import subprocess, sys
    subprocess.run(
        [sys.executable, "-c", BENCHMARK_SCRIPT],
        check=True,
        cwd="/root",
    )


@app.local_entrypoint()
def main():
    benchmark_muon.remote()
    print("Done.")
