# Windows Installation Guide for NVIDIA Blackwell (RTX 50-Series)

This guide covers installing Unsloth on Windows with an NVIDIA Blackwell GPU (`sm_120`), such as the RTX 5060, 5070, 5080, or 5090.

Blackwell needs a PyTorch build compiled for CUDA 12.8 or newer. Stable PyTorch has shipped Blackwell-ready `cu128` wheels since 2.7.0, so a nightly build is not required. The command below lets Unsloth select a matching PyTorch wheel for your GPU.

## 1. Recommended install

```powershell
winget install -e --id Python.Python.3.13
winget install -e --id astral-sh.uv
uv venv unsloth_env --python 3.13
.\unsloth_env\Scripts\activate
uv pip install unsloth --torch-backend=auto
```

`--torch-backend=auto` detects your GPU and driver and picks the right PyTorch index (stable `cu128` for Blackwell). See the [Windows guide](https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation) and [Blackwell guide](https://unsloth.ai/docs/blog/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth) for more.

## 2. C++ build tools (only if a kernel compiles)

Some kernels (for example Triton or `bitsandbytes`) may compile on the fly and need the Microsoft C++ toolchain. If you hit `RuntimeError: Failed to find C compiler`:

1. Install **Visual Studio Build Tools 2022** and select **Desktop development with C++** (includes MSVC v143 and the Windows SDK), then restart.
2. Run your training from the **x64 Native Tools Command Prompt for VS 2022** so the compiler is on `PATH`.

## 3. Verify the install

```python
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("name:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))  # (12, 0) on Blackwell
```

A Blackwell GPU should report a `+cu128` (or newer) torch build with capability `(12, 0)`. A `+cpu` build means the wrong wheel was installed: recreate the environment and reinstall with `--torch-backend=auto`.

## 4. Troubleshooting

### `CUDA error: no kernel image is available for execution on the device`

Your PyTorch build predates Blackwell (for example `cu121` or `cu124`). Reinstall with `--torch-backend=auto` to get a `cu128`+ build, clear the Triton cache at `%USERPROFILE%\.triton\cache`, then retry.

### `PicklingError: Can't pickle ...` during training

On Windows, DataLoader workers start with `spawn`, so anything passed to them must be picklable. This usually comes from a lambda or locally defined object (for example an inline collator or `formatting_func`), not from `SFTConfig` itself.

1. Set `dataloader_num_workers = 0` in your `SFTConfig`.
2. Guard your entry point with `if __name__ == "__main__":`.
3. Move any collator or `formatting_func` to module scope instead of defining it inline.

### Response-only training

`DataCollatorForCompletionOnlyLM` was removed from recent `trl`. Use `train_on_responses_only`, matching the instruction and response markers in your chat template:

```python
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    args = SFTConfig(dataloader_num_workers = 0, output_dir = "checkpoints"),
)
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|user|>\n",
    response_part = "<|assistant|>\n",
)
trainer.train()
```

## 5. Capture a known good environment

Once everything works, snapshot your versions so you can reproduce the setup:

```powershell
pip freeze > requirements-blackwell.txt
```

The key packages are a `cu128`+ build of `torch` plus current `unsloth`, `unsloth_zoo`, `bitsandbytes`, `trl`, `transformers`, and `datasets`.
