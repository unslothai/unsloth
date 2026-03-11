#!/usr/bin/env python3
"""
Generate synthetic activation_log.jsonl + metadata.json to demo
the visualize_activations.py animation without running any training.

Generates all supported data types:
  - Layer activations (mean_abs, mean per channel)
  - Gradient norms per layer
  - LoRA adapter norms (simulated q_proj, v_proj per layer)

Usage:
    python scripts/gen_fake_activations.py [output_dir]
    python visualize_activations.py [output_dir] --open
"""
import json
import math
import os
import random
import sys

OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "activation_logs_demo"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_LAYERS   = 22
NUM_CHANNELS = 64
NUM_STEPS    = 80
CAPTURE_EVERY = 1            # every step so the animation is smooth
LORA_TARGETS = ["q_proj", "v_proj"]  # Simulated LoRA targets

rng = random.Random(42)
captured_channels = list(range(NUM_CHANNELS))

# ----- metadata -----------------------------------------------------------
metadata = {
    "model_name":        "unsloth/tinyllama (simulated)",
    "num_layers":        NUM_LAYERS,
    "hidden_size":       2048,
    "intermediate_size": 5632,
    "captured_channels": captured_channels,
    "capture_interval":  CAPTURE_EVERY,
    "max_channels":      NUM_CHANNELS,
    "capture_mlp_out":   False,
    "capture_gradients": True,
    "capture_lora_norms": True,
    "lora_targets":      LORA_TARGETS,
    "created_at":        "2026-03-10T00:00:00Z",
}
with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

# ----- baseline activation state (step 0 = pre-finetune) ------------------
# Each channel starts with a smooth per-layer profile
def baseline(layer_idx, ch_idx):
    depth  = layer_idx / NUM_LAYERS           # 0..1 deeper = cooler
    period = 1.0 + ch_idx / NUM_CHANNELS * 3  # slow wave across channels
    return 0.3 + 0.25 * math.sin(ch_idx / period) * (1 - depth * 0.4)

base = [
    [baseline(l, c) for c in range(NUM_CHANNELS)]
    for l in range(NUM_LAYERS)
]

# ----- simulate finetune effect -------------------------------------------
# The finetuning "lights up" a band of channels in mid-to-late layers
# and gradually increases activity, peaking around step 50 then stabilising.
def finetune_delta(layer_idx, ch_idx, step):
    progress  = step / NUM_STEPS              # 0..1
    # Gaussian bump across channels centred at ch=28, width=14
    ch_center = 28
    ch_sigma  = 14
    ch_weight = math.exp(-((ch_idx - ch_center)**2) / (2 * ch_sigma**2))
    # Layers 10-18 are most affected
    l_center  = 14
    l_sigma   = 5
    l_weight  = math.exp(-((layer_idx - l_center)**2) / (2 * l_sigma**2))
    # Activity peaks at step ~60 then decays slightly
    time_curve = math.sin(progress * math.pi) * 0.9 + progress * 0.1
    noise      = rng.gauss(0, 0.015)
    return ch_weight * l_weight * time_curve * 0.55 + noise

# ----- write JSONL --------------------------------------------------------
log_path = os.path.join(OUT_DIR, "activation_log.jsonl")
fake_loss_start = 2.4
with open(log_path, "w") as f:
    for step in range(0, NUM_STEPS + 1, CAPTURE_EVERY):
        progress = step / NUM_STEPS
        loss     = round(fake_loss_start * math.exp(-progress * 1.6) + 0.3 + rng.gauss(0, 0.03), 4)
        layers   = {}
        grad_norms = {}
        lora_norms = {}
        
        for l in range(NUM_LAYERS):
            mean_abs = []
            mean     = []
            for c in range(NUM_CHANNELS):
                delta = finetune_delta(l, c, step)
                val   = max(0.0, base[l][c] + delta)
                mean_abs.append(round(val, 5))
                # polarity: mostly positive, shifts with layer depth
                pol = val * (0.6 + 0.4 * math.sin(l / NUM_LAYERS * math.pi))
                mean.append(round(pol, 5))
            layers[str(l)] = {"mean_abs": mean_abs, "mean": mean}
            
            # Gradient norms: high early in training, decay as loss stabilizes
            # Middle layers tend to have higher gradients
            layer_factor = math.exp(-((l - NUM_LAYERS/2)**2) / (2 * (NUM_LAYERS/3)**2))
            time_factor = math.exp(-progress * 2) + 0.1  # decays over training
            grad_norm = (0.5 + layer_factor * 0.8) * time_factor + rng.gauss(0, 0.02)
            grad_norms[str(l)] = round(max(0.01, grad_norm), 5)
            
            # LoRA norms: grow over training as adapters accumulate signal
            for target in LORA_TARGETS:
                # Different targets grow at slightly different rates
                target_offset = 0.1 if target == "q_proj" else 0.0
                lora_growth = progress * (0.3 + layer_factor * 0.4) + target_offset
                lora_norm = lora_growth + rng.gauss(0, 0.01)
                lora_norms[f"{l}.{target}"] = round(max(0.001, lora_norm), 5)
        
        record = {
            "step": step,
            "loss": loss,
            "layers": layers,
            "grad_norms": grad_norms,
            "lora_norms": lora_norms,
        }
        f.write(json.dumps(record, separators=(",", ":")) + "\n")

print(f"✅  Wrote {NUM_STEPS // CAPTURE_EVERY + 1} steps to '{OUT_DIR}/'")
print(f"    metadata.json + activation_log.jsonl")
print(f"    (includes gradient norms + LoRA norms)")
print(f"\nNow run:")
print(f"    python visualize_activations.py {OUT_DIR} --open")
