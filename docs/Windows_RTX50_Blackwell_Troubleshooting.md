# 🚀 Windows Installation Guide for NVIDIA Blackwell (RTX 50-Series)

This guide provides a step-by-step workflow for setting up Unsloth locally on a Windows machine using NVIDIA's new Blackwell architecture (`sm_120`), such as the RTX 5060, 5070, 5080, or 5090.

Because standard stable releases of PyTorch currently max out at Hopper/Ada Lovelace (`sm_90`), Blackwell GPUs require a specific combination of PyTorch Nightly builds (`cu128`) and C++ compiler configurations to successfully compile Triton kernels on Windows.

## 🛠️ 1. System Prerequisites (The C++ Compiler)

Before creating your Python environment, you **must** have the Microsoft C++ compiler installed. Triton will attempt to compile hardware-specific kernels on the fly, and it will crash without this.

1. Download the **Visual Studio Build Tools 2022** from the official Microsoft website. *(Note: This is different from the Visual C++ Redistributable).*
2. Run the installer and check the box for **Desktop development with C++**. 
3. Ensure the *Windows SDK* and *MSVC v143* are checked in the right-hand installation details panel.
4. Click Install and **restart your computer**.
5. **CRITICAL:** Do not use the standard Command Prompt or PowerShell. Search your Windows Start menu for **`x64 Native Tools Command Prompt for VS 2022`** and use this terminal for all subsequent steps. This natively links the C++ compiler to your session.

## 📦 2. The "No-Dependency" Installation Protocol

Because Unsloth's standard `pip` requirements enforce a maximum PyTorch version, standard installations will automatically uninstall the cutting-edge PyTorch Nightly build and downgrade you to a CPU-only version. You must install the libraries in this exact order to lock in the `cu128` binaries.

First, create a fresh virtual environment:
```cmd
python -m venv blackwell-env
blackwell-env\\Scripts\\activate
```
**Step A: Install PyTorch Nightly (CUDA 12.8)**

```cmd
pip install --pre torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/nightly/cu128](https://download.pytorch.org/whl/nightly/cu128)

```

**Step B: Resolve minor dependency conflicts**

```cmd
pip install "fsspec<=2025.9.0"

```

**Step C: Install Unsloth bypassing PyTorch version checks**
By using `--no-deps`, we force pip to accept the PyTorch 2.12.0 Nightly build without downgrading it.

```cmd
pip install --no-deps unsloth unsloth_zoo

```

**Step D: Install remaining core libraries**

```cmd
pip install bitsandbytes trl transformers datasets jupyter ipykernel

```

**Step E: Register your kernel**

```cmd
python -m ipykernel install --user --name=blackwell-env --display-name "Python (Blackwell RTX 50-Series)"

```

---

## 🛑 3. Troubleshooting Common Errors

### Error: `RuntimeError: Failed to find C compiler`

* **Symptom:** Crashes during `bitsandbytes` or `triton` initialization.
* **Cause:** Windows lacks a native C compiler, or the terminal cannot find it.
* **Solution:** Install the MSVC Build Tools as detailed in Step 1. Launch Jupyter Notebook exclusively from the **`x64 Native Tools Command Prompt`**.

### Error: `CUDA error: no kernel image is available for execution on the device`

* **Symptom:** Crashes on mathematical operations (e.g., `emb.cos()`) when first initializing the model.
* **Cause:** The environment is using a standard PyTorch build (like `cu121` or `cu124`). These binaries are compiled for older architectures (`sm_89` and below) and cannot communicate with Blackwell (`sm_120`).
* **Solution:** 1. Clear the Triton cache: Delete the contents of `%USERPROFILE%\\.triton\\cache`.
2. Run `import torch; print(torch.__version__)` in Python. If it says `+cpu` or anything other than `+cu128`, pip downgraded your installation. Repeat the "No-Dependency" installation protocol in Step 2.

### Error: `PicklingError: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>`

* **Symptom:** Crashes at the very end of a training run or when attempting to save a checkpoint.
* **Cause:** Fragmented memory in Jupyter after re-running cells, combined with recent updates in the `trl` library migrating away from standard `TrainingArguments`.
* **Solution:** 1. Go to **Kernel -> Restart & Clear Output** in Jupyter Notebook.
2. Update your training script to use `SFTConfig` instead of `TrainingArguments`.
3. Remove `DataCollatorForCompletionOnlyLM` (which has been deprecated) and wrap the trainer in Unsloth's native `train_on_responses_only` function.

**Updated Trainer Implementation:**

```python
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        output_dir = "checkpoints",
        # ... your other arguments ...
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|user|>\\n",
    response_part = "<|assistant|>\\n",
)

trainer.train()

```

## 📋 4. Verified Working Environment (Requirements)

If you want to perfectly replicate this working environment, here is the complete snapshot of all packages and versions used for the Blackwell RTX 5060 setup. 

<details>
<summary><b>Click to expand the full requirements list</b></summary>

```text
accelerate==1.14.0
aiohappyeyeballs==2.6.2
aiohttp==3.14.1
aiosignal==1.4.0
annotated-doc==0.0.4
annotated-types==0.7.0
anyio==4.13.0
argon2-cffi==25.1.0
argon2-cffi-bindings==25.1.0
arrow==1.4.0
asttokens==3.0.1
async-lru==2.3.0
attrs==26.1.0
babel==2.18.0
beautifulsoup4==4.15.0
bitsandbytes==0.49.2
bleach==6.4.0
certifi==2026.5.20
cffi==2.0.0
charset-normalizer==3.4.7
click==8.4.1
colorama==0.4.6
comm==0.2.3
cut-cross-entropy==25.1.1
datasets==4.3.0
debugpy==1.8.21
decorator==5.3.1
defusedxml==0.7.1
diffusers==0.38.0
dill==0.4.0
docstring-parser==0.18.0
executing==2.2.1
fastjsonschema==2.21.2
filelock==3.29.1
fqdn==1.5.1
frozenlist==1.8.0
fsspec==2025.9.0
h11==0.16.0
hf-transfer==0.1.9
hf-xet==1.5.1
httpcore==1.0.9
httpx==0.28.1
huggingface-hub==1.19.0
idna==3.18
importlib-metadata==9.0.0
ipykernel==7.3.0
ipython==9.14.1
ipython-pygments-lexers==1.1.1
ipywidgets==8.1.8
isoduration==20.11.0
jedi==0.20.0
jinja2==3.1.6
json5==0.14.0
jsonpointer==3.1.1
jsonschema==4.26.0
jsonschema-specifications==2025.9.1
jupyter==1.1.1
jupyter-client==8.9.1
jupyter-console==6.6.3
jupyter-core==5.9.1
jupyter-events==0.12.1
