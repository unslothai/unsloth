# 🦥 Exporting Models from Google Colab

After fine-tuning your model with Unsloth on Google Colab, you'll need to transfer the resulting model files (GGUF, LoRA adapters, merged weights, etc.) to your local machine or a cloud server. This guide covers several reliable methods — especially useful for **large files (>5 GB)** where Colab's built-in download often fails.

## Quick Comparison

| Method | Best For | Max Size | Difficulty |
|--------|----------|----------|------------|
| [Hugging Face Hub](#1-push-to-hugging-face-hub-recommended) | All file sizes | Unlimited | ⭐ Easy |
| [Google Drive](#2-save-to-google-drive) | Files < 15 GB (free tier) | Drive quota | ⭐ Easy |
| [Direct download (small files)](#3-direct-download-small-files-only) | Files < 1–2 GB | ~2 GB | ⭐ Easy |
| [Cloud storage via rclone](#4-upload-to-cloud-storage-via-rclone) | S3, GCS, Azure users | Unlimited | ⚙️ Moderate |

---

## 1. Push to Hugging Face Hub (Recommended)

The most reliable method. Push directly to [Hugging Face](https://huggingface.co/) and download from there. Works with any file size and gives you a permanent, shareable link.

### Push GGUF files

```python
# Push GGUF to your Hugging Face account
model.push_to_hub_gguf(
    "your-hf-username/my-model-gguf",  # Replace with your HF username
    tokenizer,
    quantization_method = "q4_k_m",
    token = "hf_...",  # Your HF write token
)
```

### Push merged weights (16-bit or 4-bit)

```python
# Push merged 16-bit model
model.push_to_hub_merged(
    "your-hf-username/my-model-16bit",
    tokenizer,
    save_method = "merged_16bit",
    token = "hf_...",
)

# Or push just the LoRA adapter
model.push_to_hub(
    "your-hf-username/my-model-lora",
    tokenizer,
    token = "hf_...",
)
```

### Download to your local machine

Once pushed, download using the `huggingface_hub` CLI on your local machine:

```bash
# Install if needed
pip install huggingface_hub

# Download the entire repo
huggingface-cli download your-hf-username/my-model-gguf --local-dir ./my-model

# Or download a specific GGUF file
huggingface-cli download your-hf-username/my-model-gguf my-model-Q4_K_M.gguf --local-dir ./my-model
```

> **💡 Tip:** You can get a free Hugging Face token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). Make sure to create a **write** token.

---

## 2. Save to Google Drive

Mount your Google Drive and copy the model files there. Once synced, download from [drive.google.com](https://drive.google.com) on your local machine.

```python
from google.colab import drive
drive.mount('/content/drive')

# Save GGUF directly to Google Drive
model.save_pretrained_gguf(
    "/content/drive/MyDrive/my-model",
    tokenizer,
    quantization_method = "q4_k_m",
)
```

Alternatively, **copy files** if you've already saved to local Colab storage:

```python
import shutil
shutil.copytree("local_directory", "/content/drive/MyDrive/my-model")
```

### Troubleshooting Google Drive

If you see `MessageError: credential propagation was unsuccessful`:

1. **Try a different browser** — Some browser extensions block the OAuth popup
2. **Use Incognito/Private mode** — Avoids conflicts with cached credentials
3. **Clear cookies** for `accounts.google.com` and `colab.research.google.com`
4. **Check storage quota** — Free Google Drive accounts have 15 GB; make sure you have enough space
5. **Re-authorize** — Click "Runtime → Disconnect and delete runtime", then reconnect and try `drive.mount()` again

> **⚠️ Note:** Google Drive's free tier is 15 GB. Larger models (e.g., Q8_0 for 7B parameter models) may not fit. Consider using Hugging Face Hub or a smaller quantization like `q4_k_m` instead.

---

## 3. Direct Download (Small Files Only)

For files under ~1–2 GB, you can download directly from Colab:

```python
from google.colab import files
files.download("path/to/your-model.gguf")
```

> **⚠️ Warning:** This method is unreliable for files over 1–2 GB. Browser downloads from Colab frequently time out or fail silently for large files. Use Hugging Face Hub or Google Drive instead.

---

## 4. Upload to Cloud Storage via `rclone`

If you use cloud storage (AWS S3, Google Cloud Storage, Azure Blob, etc.), you can install `rclone` in Colab and upload directly.

```bash
# Install rclone in Colab
!curl https://rclone.org/install.sh | sudo bash

# Configure (interactive — or provide a pre-made rclone.conf)
!rclone config

# Example: copy to S3
!rclone copy local_directory/ myremote:mybucket/my-model/
```

> **💡 Tip:** To avoid interactive configuration, you can create an `rclone.conf` file in advance and upload it to Colab, or set environment variables like `RCLONE_CONFIG_MYREMOTE_TYPE=s3`.

---

## Summary

For most users, **pushing to Hugging Face Hub** is the easiest and most reliable option — it works with any file size, gives you a permanent link, and integrates directly with Unsloth's save methods. If you prefer keeping files private, use Google Drive (for models under 15 GB) or `rclone` for cloud storage.

## Need Help?

- 📚 [Unsloth Documentation](https://docs.unsloth.ai/)
- 💬 [Discord](https://discord.com/invite/unsloth)
- 🦥 [Reddit r/unsloth](https://reddit.com/r/unsloth)
