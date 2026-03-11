# Neuron Activation Capture & Visualization

Unsloth includes a built-in system for capturing neuron activations, gradient norms, and LoRA adapter statistics during finetuning. This data can be visualized as an animated heatmap to understand *where* and *how* finetuning reshapes your model.

## Quick Start

### 1. Enable capture during training (CLI)

```bash
python unsloth-cli.py \
    --model_name unsloth/Qwen3-4B \
    --dataset your/dataset \
    --capture_activations \
    --capture_output_dir ./my_activations \
    --capture_interval 5 \
    --max_steps 100
```

### 2. Generate the visualization

```bash
python visualize_activations.py ./my_activations --open
```

This creates a self-contained HTML file you can open in any browser — no server required.

---

## What Gets Captured

At each capture interval, the system records:

| Metric | Description | Visual |
|--------|-------------|--------|
| **Activations** | Mean absolute value per channel × layer | Main heatmap grid |
| **Gradient norms** | L2 norm of gradients at each layer | Green sparklines |
| **LoRA norms** | Frobenius norm ‖B @ A‖ per adapter | Purple bars |
| **Loss** | Training loss at that step | Top-left stat panel |

### Output Files

```
my_activations/
├── metadata.json          # Model config, capture settings
└── activation_log.jsonl   # One JSON record per captured step
```

**metadata.json** example:
```json
{
  "model_name": "unsloth/Qwen3-4B",
  "num_layers": 40,
  "hidden_size": 4096,
  "captured_channels": [0, 1, 2, ...],
  "capture_interval": 5,
  "capture_gradients": true,
  "capture_lora_norms": true,
  "lora_targets": ["q_proj", "v_proj", "o_proj", ...]
}
```

**activation_log.jsonl** record example:
```json
{
  "step": 50,
  "loss": 1.234,
  "layers": {
    "0": {"mean_abs": [0.32, 0.41, ...], "mean": [0.12, -0.08, ...]},
    "1": {...}
  },
  "grad_norms": {"0": 0.0234, "1": 0.0189, ...},
  "lora_norms": {"0.q_proj": 0.145, "0.v_proj": 0.132, ...}
}
```

---

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--capture_activations` | `false` | Enable the capture system |
| `--capture_output_dir` | `activation_logs` | Where to write output files |
| `--capture_interval` | `10` | Capture every N optimizer steps |
| `--capture_max_channels` | `64` | Number of channels tracked per layer |

Lower intervals → smoother animations but larger files.

---

## Python API

For more control, use the classes directly:

```python
from unsloth import FastLanguageModel
from unsloth import (
    ActivationCaptureConfig,
    ActivationCapture,
    ActivationCaptureCallback,
)
from trl import SFTTrainer

# Load model
model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen3-4B", ...)
model = FastLanguageModel.get_peft_model(model, ...)

# Configure capture
config = ActivationCaptureConfig(
    output_dir="./activations",
    capture_interval=5,      # every 5 steps
    max_channels=64,         # 64 channels per layer
    capture_mlp_out=False,   # also hook MLP outputs?
    capture_gradients=True,  # capture gradient norms
    capture_lora_norms=True, # capture LoRA adapter norms
    seed=42,                 # reproducible channel sampling
)

# Create capture + callback
capture = ActivationCapture(model, config)
callback = ActivationCaptureCallback(capture)

# Train with callback
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    callbacks=[callback],
    ...
)
trainer.train()
```

The callback automatically:
- Attaches hooks at training start
- Arms capture on scheduled steps
- Flushes data after each step
- Detaches hooks at training end

---

## Understanding the Visualization

### Heatmap Grid

- **Columns** = transformer layers (left = early, right = late)
- **Rows** = sampled channels (dimension indices)
- **Color** = activation intensity

| Color | Meaning |
|-------|---------|
| Deep blue | Low / dormant |
| Cyan | Moderate |
| Yellow | High |
| Red | Very high / "hot" |

As training progresses, you'll typically see:
1. Early layers stay relatively cool (frozen features)
2. Middle/late layers light up as task-specific patterns emerge
3. Specific channel bands activate (domain knowledge encoding)

### Gradient Sparklines (Green)

One sparkline per layer showing gradient norm over time.
- **High** = layer is actively learning
- **Low** = layer is stable / converged
- **Spike** = sudden gradient (bad batch? learning rate issue?)

### LoRA Panel (Purple)

Shows ‖B @ A‖_F for each LoRA target across layers.
- **Growing** = adapter accumulating task signal
- **Flat** = adapter saturated or not needed
- **Different targets** = see which projections (Q, K, V, etc.) are most important

---

## Inspecting JSONL Data

Quick check that data was captured correctly:

```bash
python scripts/check_jsonl.py ./my_activations/activation_log.jsonl
```

Output:
```
Step 0: keys=['step', 'loss', 'layers', 'grad_norms', 'lora_norms']
  grad_norms: 40 layers, sample: layer 0 = 0.0312
  lora_norms: 240 entries, sample: ['0.q_proj', '0.v_proj', '0.o_proj', '0.gate_proj']
```

---

## Performance Notes

- **Overhead**: Near-zero on uncaptured steps (single bool check)
- **Memory**: Stats computed in-place, no activation storage
- **Gradient checkpointing**: Handled automatically (deduplication)
- **File size**: ~30KB per 100 steps with default settings

For very long runs, increase `capture_interval` to reduce file size.

---

## Troubleshooting

### "Could not locate .layers on the model"

The capture system expects a standard HuggingFace causal LM structure:
- `model.model.layers` (standard LlamaForCausalLM)
- `model.base_model.model.model.layers` (PEFT-wrapped)

If your model has a different structure, it may not be supported.

### No gradient data in output

Ensure `capture_gradients=True` (default) and that the model is actually training (not just evaluating).

### LoRA norms all zero

- Model might not have LoRA adapters (non-PEFT model)
- LoRA modules might use different naming conventions

---

## Example Training Script

See [scripts/train_qwen3_capture.py](../scripts/train_qwen3_capture.py) for a complete example training Qwen3-4B with activation capture enabled.

---

## Visualization Script Options

```bash
python visualize_activations.py <activation_dir> [options]

Options:
  --output PATH   Output HTML path (default: <dir>/viz.html)
  --open          Open in browser after generating
```
