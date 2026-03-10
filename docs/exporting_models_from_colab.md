# Exporting Fine-Tuned Models from Google Colab

This guide covers how to save, export, and download your fine-tuned models when
working in Google Colab. Because Colab instances are ephemeral (your files are
lost when the runtime disconnects), it is critical to persist your model before
the session ends.

**Table of Contents**

- [Overview of Save Methods](#overview-of-save-methods)
- [1. Save LoRA Adapters (Fastest)](#1-save-lora-adapters-fastest)
- [2. Save a Merged Model (16-bit)](#2-save-a-merged-model-16-bit)
- [3. Export to GGUF Format](#3-export-to-gguf-format)
- [4. Push to Hugging Face Hub](#4-push-to-hugging-face-hub)
- [5. Save to Google Drive](#5-save-to-google-drive)
- [6. Download Large Files from Colab](#6-download-large-files-from-colab)
- [Common Issues and Troubleshooting](#common-issues-and-troubleshooting)
- [Quick Reference](#quick-reference)

---

## Overview of Save Methods

After fine-tuning with Unsloth, you have several options for saving your model.
Each method is exposed directly on the model object:

| Method | Description | Use Case |
|--------|-------------|----------|
| `model.save_pretrained()` | Save LoRA adapters only | Resuming training, lightweight sharing |
| `model.save_pretrained_merged()` | Merge LoRA into base weights and save | Full model for HF inference, vLLM, SGLang |
| `model.save_pretrained_gguf()` | Merge and convert to GGUF format | Ollama, llama.cpp, local inference |
| `model.push_to_hub()` | Push LoRA adapters to Hugging Face Hub | Sharing adapters on HF |
| `model.push_to_hub_merged()` | Merge and push full model to HF Hub | Sharing full model on HF |
| `model.push_to_hub_gguf()` | Merge, convert to GGUF, and push to HF Hub | Sharing GGUF on HF |

---

## 1. Save LoRA Adapters (Fastest)

This is the fastest and smallest save option. It only saves the LoRA adapter
weights (typically a few hundred MB), not the full model. You will need the
original base model to load it again later.

```python
# Save LoRA adapters locally
model.save_pretrained("my_lora_model")
tokenizer.save_pretrained("my_lora_model")
```

To load the LoRA model later:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="my_lora_model",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

---

## 2. Save a Merged Model (16-bit)

This merges the LoRA adapter weights back into the base model and saves the
result as a full model in float16. This is needed if you want to convert to GGUF
later, serve with vLLM/SGLang, or share a standalone model.

```python
# Merge LoRA into the base model and save as 16-bit
model.save_pretrained_merged(
    "my_merged_model",
    tokenizer=tokenizer,
    save_method="merged_16bit",
)
```

### Available `save_method` options:

- `"merged_16bit"` -- Merge LoRA into float16 weights. Recommended for GGUF
  conversion, vLLM, and SGLang.
- `"merged_4bit_forced"` -- Merge LoRA into 4-bit weights. Smaller but lossy.
  Use `"merged_4bit_forced"` (not `"merged_4bit"`) to confirm you accept the
  accuracy trade-off.
- `"lora"` -- Same as `save_pretrained()`; saves only the adapter.

---

## 3. Export to GGUF Format

GGUF is the format used by [llama.cpp](https://github.com/ggml-org/llama.cpp)
and [Ollama](https://ollama.ai) for local inference. Unsloth handles the full
pipeline: merging LoRA weights, converting to GGUF, and quantizing.

### Save GGUF locally

```python
# Save as GGUF with q4_k_m quantization (recommended for most use cases)
model.save_pretrained_gguf(
    "my_model",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",
)
```

### Save multiple quantizations at once

You can pass a list of quantization methods to generate multiple GGUF files in
one call:

```python
model.save_pretrained_gguf(
    "my_model",
    tokenizer=tokenizer,
    quantization_method=["q4_k_m", "q5_k_m", "q8_0"],
)
```

### Push GGUF directly to Hugging Face Hub

```python
model.push_to_hub_gguf(
    "your-username/my-model-gguf",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",
    token="hf_...",
)
```

### Available quantization methods

| Method | Description |
|--------|-------------|
| `"not_quantized"` | No quantization (f16/bf16). Largest file, highest accuracy. |
| `"fast_quantized"` | Alias for `q8_0`. Fast conversion, good balance. |
| `"quantized"` | Alias for `q4_k_m`. Slow conversion, small files, fast inference. |
| `"q8_0"` | 8-bit quantization. High quality, larger files. |
| `"q5_k_m"` | 5-bit quantization. Good balance of quality and size. |
| `"q4_k_m"` | 4-bit quantization. Recommended for most use cases. |
| `"q3_k_m"` | 3-bit quantization. Smaller but lower quality. |
| `"q2_k"` | 2-bit quantization. Smallest files, notable quality loss. |
| `"f16"` | Float16. No quantization, retains full accuracy. |
| `"bf16"` | Bfloat16. No quantization, retains full accuracy. |

For a complete list, see the
[Unsloth GGUF docs](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf).

### Use your GGUF model with Ollama

After exporting, you can create an Ollama model:

```bash
# Create a Modelfile
echo 'FROM ./my_model_gguf/<base-model-name>.Q4_K_M.gguf' > Modelfile

# Create the Ollama model
ollama create my-model -f Modelfile

# Run it
ollama run my-model
```

---

## 4. Push to Hugging Face Hub

Pushing to Hugging Face Hub is the most reliable way to persist your model from
Colab, since it uploads directly to cloud storage that survives runtime
disconnections.

### Prerequisites

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co).
2. Create an access token at
   [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with
   **Write** permission.
3. Log in from your Colab notebook:

```python
from huggingface_hub import login
login(token="hf_...")

# Or use notebook_login() for an interactive widget:
# from huggingface_hub import notebook_login
# notebook_login()
```

### Push LoRA adapters

```python
model.push_to_hub("your-username/my-lora-model", token="hf_...")
tokenizer.push_to_hub("your-username/my-lora-model", token="hf_...")
```

### Push merged model

```python
model.push_to_hub_merged(
    "your-username/my-merged-model",
    tokenizer=tokenizer,
    save_method="merged_16bit",
    token="hf_...",
)
```

### Push GGUF to Hub

```python
model.push_to_hub_gguf(
    "your-username/my-model-gguf",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",
    token="hf_...",
)
```

### Make the repo private

Add `private=True` to any push call:

```python
model.push_to_hub_merged(
    "your-username/my-model",
    tokenizer=tokenizer,
    save_method="merged_16bit",
    token="hf_...",
    private=True,
)
```

---

## 5. Save to Google Drive

Google Drive is another way to persist files from Colab. However, it has
limitations (15 GB free storage, slow transfers for large files, occasional
credential issues).

### Mount Google Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### Save directly to Google Drive

```python
# Save LoRA adapters to Drive (fast, small files)
model.save_pretrained("/content/drive/MyDrive/my_lora_model")
tokenizer.save_pretrained("/content/drive/MyDrive/my_lora_model")

# Save merged model to Drive (slower, larger files)
model.save_pretrained_merged(
    "/content/drive/MyDrive/my_merged_model",
    tokenizer=tokenizer,
    save_method="merged_16bit",
)

# Save GGUF to Drive
model.save_pretrained_gguf(
    "/content/drive/MyDrive/my_model_gguf",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",
)
```

### Important notes about Google Drive

- **Storage limit**: Free Google Drive accounts have 15 GB. A merged 7B model in
  float16 is about 14 GB. Use GGUF quantization (e.g., `q4_k_m`) to reduce size
  to around 4 GB.
- **Credential propagation error**: If you see
  `"credential propagation was unsuccessful"`, try:
  1. Disconnect and remount Google Drive:
     ```python
     drive.flush_and_unmount()
     drive.mount("/content/drive", force_remount=True)
     ```
  2. If that fails, use `push_to_hub` instead (see section 4).

---

## 6. Download Large Files from Colab

### Direct download (small files only)

For files under ~1-2 GB, you can download directly:

```python
from google.colab import files
files.download("my_model_gguf/<model-name>.Q4_K_M.gguf")
```

This often fails or times out for larger files.

### Split large files for download

For GGUF files that are too large to download directly, split them first:

```python
import subprocess

# Split into 2 GB chunks
subprocess.run([
    "split", "-b", "2G",
    "my_model_gguf/<model-name>.Q4_K_M.gguf",
    "my_model_gguf/model_part_"
])

# Download each part
from google.colab import files
import glob
for part in sorted(glob.glob("my_model_gguf/model_part_*")):
    files.download(part)
```

Reassemble on your local machine:

```bash
# Linux / macOS
cat model_part_* > model.gguf

# Windows (PowerShell)
Get-Content model_part_* -Encoding Byte -ReadCount 0 | Set-Content model.gguf -Encoding Byte
```

### Use `gdown` to download from Google Drive

If you saved to Google Drive, you can use `gdown` from your local machine:

```bash
pip install gdown

# Get the file ID from the Google Drive share link
gdown "https://drive.google.com/uc?id=YOUR_FILE_ID"
```

### Use `rclone` for reliable transfers

[rclone](https://rclone.org/) can handle large files reliably:

```bash
# In Colab, install and configure rclone
!curl https://rclone.org/install.sh | sudo bash
!rclone config  # Follow prompts to set up your remote

# Copy model to your remote storage
!rclone copy my_model_gguf/ remote:my-models/
```

### Push to Hugging Face Hub then download locally (recommended)

The most reliable approach for large files is to push to Hugging Face Hub first,
then download using the `huggingface-cli`:

```bash
# On your local machine
pip install huggingface_hub
huggingface-cli download your-username/my-model-gguf --local-dir ./my-model-gguf
```

---

## Common Issues and Troubleshooting

### Disk space errors in Colab

Colab free tier provides roughly 78 GB of disk space, but some is used by the
system and cached model weights. Saving a merged 16-bit model can temporarily
require 2x the model size in disk space.

**Solutions:**

```python
# 1. Free cached model weights to reclaim disk space
#    Unsloth does this automatically in most cases, but you can also:
import shutil, os
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)

# 2. Check available disk space
!df -h /

# 3. Use GGUF with quantization instead of merged_16bit to reduce file size
model.save_pretrained_gguf(
    "my_model",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",  # ~4 GB for a 7B model vs ~14 GB for 16-bit
)

# 4. Push directly to Hub instead of saving locally
model.push_to_hub_gguf(
    "your-username/my-model",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",
    token="hf_...",
)
```

### Colab runtime disconnects during save

Large model exports can take 10-30 minutes. Colab may disconnect during this
time.

**Solutions:**

- **Keep the browser tab active and in the foreground.** Colab throttles
  background tabs.
- **Save LoRA adapters first** (takes seconds), then save the merged/GGUF
  version. This way you have a backup even if the larger save fails.
- **Push to Hugging Face Hub** instead of saving locally. Hub uploads can resume
  from where they left off if interrupted.

```python
# Save LoRA first as a safety net (very fast)
model.save_pretrained("my_lora_backup")
tokenizer.save_pretrained("my_lora_backup")

# Then do the longer merged/GGUF export
model.push_to_hub_gguf(
    "your-username/my-model",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",
    token="hf_...",
)
```

### Hugging Face authentication errors

If you get `"Please supply a token!"` or `401 Unauthorized`:

```python
# Option 1: Pass token directly
model.push_to_hub_merged(
    "your-username/my-model",
    tokenizer=tokenizer,
    save_method="merged_16bit",
    token="hf_YOUR_TOKEN_HERE",
)

# Option 2: Log in first (token is then used automatically)
from huggingface_hub import login
login(token="hf_YOUR_TOKEN_HERE")

# Option 3: Use notebook_login for an interactive widget
from huggingface_hub import notebook_login
notebook_login()
```

Make sure your token has **Write** permissions. Create one at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### Google Drive credential propagation error

If you see `"credential propagation was unsuccessful"` when mounting Google
Drive:

1. Try force remounting:
   ```python
   from google.colab import drive
   drive.flush_and_unmount()
   drive.mount("/content/drive", force_remount=True)
   ```
2. If that fails, try a different browser or clear cookies.
3. As a fallback, use `push_to_hub` methods instead of Google Drive.

### GGUF conversion fails or runs out of memory

GGUF conversion requires merging LoRA into 16-bit first, which temporarily uses
significant memory.

**Solutions:**

```python
# Use the maximum_memory_usage parameter to control memory
model.save_pretrained_gguf(
    "my_model",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",
    maximum_memory_usage=0.75,  # Use at most 75% of available GPU memory
)
```

If you still run out of memory, try saving the merged 16-bit model first, then
restart the runtime and convert the saved model separately using llama.cpp
directly.

### Model files are too large for Colab's direct download

Colab's `files.download()` often fails for files over 1-2 GB. See
[section 6](#6-download-large-files-from-colab) for alternatives including file
splitting, gdown, rclone, and the recommended Hub-based workflow.

---

## Quick Reference

### Complete Colab workflow (recommended)

This is the recommended end-to-end workflow for saving a model from Colab:

```python
from unsloth import FastLanguageModel
from huggingface_hub import login

# -- After training is complete --

# Step 1: Log in to Hugging Face
login(token="hf_YOUR_TOKEN")

# Step 2: Save LoRA adapters as a quick backup
model.save_pretrained("lora_backup")
tokenizer.save_pretrained("lora_backup")

# Step 3a: Push merged model to Hub (for vLLM, SGLang, HF inference)
model.push_to_hub_merged(
    "your-username/my-model",
    tokenizer=tokenizer,
    save_method="merged_16bit",
    token="hf_YOUR_TOKEN",
)

# Step 3b: Push GGUF to Hub (for Ollama, llama.cpp)
model.push_to_hub_gguf(
    "your-username/my-model-gguf",
    tokenizer=tokenizer,
    quantization_method="q4_k_m",
    token="hf_YOUR_TOKEN",
)
```

Then download on your local machine:

```bash
pip install huggingface_hub

# Download the merged model
huggingface-cli download your-username/my-model --local-dir ./my-model

# Download the GGUF file
huggingface-cli download your-username/my-model-gguf --local-dir ./my-model-gguf
```

### Save method comparison

| Goal | Method | Approx. Size (7B) | Speed |
|------|--------|--------------------|-------|
| Quick backup, resume training | `save_pretrained()` (LoRA) | ~100-500 MB | Seconds |
| Full model for HF inference | `save_pretrained_merged()` 16-bit | ~14 GB | 5-10 min |
| Local inference with Ollama | `save_pretrained_gguf()` q4_k_m | ~4 GB | 10-20 min |
| Share on Hugging Face | `push_to_hub_merged()` | Upload ~14 GB | 10-20 min |
| Share GGUF on Hugging Face | `push_to_hub_gguf()` | Upload ~4 GB | 10-20 min |
