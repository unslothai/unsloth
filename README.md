<div align="center">

  <a href="https://unsloth.ai"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png">
    <img alt="unsloth logo" src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png" height="110" style="max-width: 100%;">
  </picture></a>
  
<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing"><img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/start free finetune button.png" height="48"></a>
<a href="https://discord.gg/u54VK8m8tk"><img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/Discord button.png" height="48"></a>
<a href="https://ko-fi.com/unsloth"><img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/buy me a coffee button.png" height="48"></a>

### Finetune Mistral, Gemma, Llama 2-5x faster with 80% less memory!

![](https://i.ibb.co/sJ7RhGG/image-41.png)

</div>

## ✨ Finetune for Free

All notebooks are **beginner friendly**! Add your dataset, click "Run All", and you'll get a 2x faster finetuned model which can be exported to GGUF, vLLM or uploaded to Hugging Face.

| Unsloth supports          |    Free Notebooks                                                                                           | Performance | Memory use |
|-----------------|--------------------------------------------------------------------------------------------------------------------------|-------------|----------|
| **Llama-3 8b**      | [▶️ Start on Colab](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)               | 2x faster | 60% less |
| **Gemma 7b**      | [▶️ Start on Colab](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing)               | 2.4x faster | 71% less |
| **Mistral 7b**    | [▶️ Start on Colab](https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing)               | 2.2x faster | 73% less |
| **TinyLlama**  | [▶️ Start on Colab](https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing)              | 3.9x faster | 82% less |
| **CodeLlama 34b** A100   | [▶️ Start on Colab](https://colab.research.google.com/drive/1y7A0AxE3y8gdj4AVkl2aZX47Xu3P1wJT?usp=sharing)              | 1.9x faster | 49% less |
| **Mistral 7b** 1xT4  | [▶️ Start on Kaggle](https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook) | 5x faster\* | 73% less |
| **DPO - Zephyr**     | [▶️ Start on Colab](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing)               | 1.9x faster | 43% less |

- Benchmarking compared to FA2 + Hugging Face combined.
- This [conversational notebook](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing) is useful for ShareGPT ChatML / Vicuna templates.
- This [text completion notebook](https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing) is for raw text. This [DPO notebook](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing) replicates Zephyr.
- \* Kaggle has 2x T4s, but we use 1. Due to overhead, 1x T4 is 5x faster.

## 🦥 Unsloth.ai News
- 📣 NEW! [Llama-3 8b](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing) now works! Llama-3 70b also works (just change the model name in the notebook).
- 📣 NEW! We cut memory usage by a [further 30%](https://unsloth.ai/blog/long-context) and now support fine-tuning of LLMs with [4x longer context windows](https://unsloth.ai/blog/long-context)! No change required if you're using our notebooks. To enable, simply change 1 line:
```python
model = FastLanguageModel.get_peft_model(
    model,
    use_gradient_checkpointing = "unsloth", # <<<<<<<
)
```
- 📣 [CodeGemma](https://colab.research.google.com/drive/19lwcRk_ZQ_ZtX-qzFP3qZBBHZNcMD1hh?usp=sharing) now works along with [Gemma 7b](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing) and [Gemma 2b](https://colab.research.google.com/drive/15gGm7x_jTm017_Ic8e317tdIpDG53Mtu?usp=sharing)
- 📣 [2x faster inference](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing) added for all our models
- 📣 [DPO support](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing) is now included. [More info](#DPO) on DPO
- 📣 We did a [blog](https://huggingface.co/blog/unsloth-trl) with 🤗Hugging Face and are in their official docs! Check out the [SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth) and [DPO docs](https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth)

## 🔗 Links and Resources
| Type                            | Links                               |
| ------------------------------- | --------------------------------------- |
| 📚 **Wiki & FAQ**              | [Read Our Wiki](https://github.com/unslothai/unsloth/wiki) |
| <img height="14" src="https://upload.wikimedia.org/wikipedia/commons/6/6f/Logo_of_Twitter.svg" />&nbsp; **Twitter (aka X)**              |  [Follow us on X](https://twitter.com/unslothai)|
| 📜 **Documentation**              | [Read The Doc](https://github.com/unslothai/unsloth/tree/main#-documentation) |
| 💾 **Installation**               | [unsloth/README.md](https://github.com/unslothai/unsloth/tree/main#installation-instructions)|
| 🥇 **Benchmarking**                   | [Performance Tables](https://github.com/unslothai/unsloth/tree/main#-performance-benchmarking)
| 🌐 **Released Models**            | [Unsloth Releases](https://huggingface.co/unsloth)|
| ✍️ **Blog**                    | [Read our Blogs](https://unsloth.ai/blog)|

## ⭐ Key Features
- All kernels written in [OpenAI's Triton](https://openai.com/research/triton) language. **Manual backprop engine**.
- **0% loss in accuracy** - no approximation methods - all exact.
- No change of hardware. Supports NVIDIA GPUs since 2018+. Minimum CUDA Capability 7.0 (V100, T4, Titan V, RTX 20, 30, 40x, A100, H100, L40 etc) [Check your GPU!](https://developer.nvidia.com/cuda-gpus) GTX 1070, 1080 works, but is slow.
- Works on **Linux** and **Windows** via WSL.
- Supports 4bit and 16bit QLoRA / LoRA finetuning via [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
- Open source trains 5x faster - see [Unsloth Pro](https://unsloth.ai/) for up to **30x faster training**!
- If you trained a model with 🦥Unsloth, you can use this cool sticker! &nbsp; <img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made with unsloth.png" height="50" align="center" />


## 🥇 Performance Benchmarking
- For the full list of **reproducable** benchmarking tables, [go to our website](https://unsloth.ai/blog/mistral-benchmark#Benchmark%20tables)

| 1 A100 40GB  | 🤗Hugging Face | Flash Attention | 🦥Unsloth Open Source | 🦥[Unsloth Pro](https://unsloth.ai/pricing) |
|--------------|--------------|-----------------|---------------------|-----------------|
| Alpaca       | 1x           | 1.04x           | 1.98x               | **15.64x**      |
| LAION Chip2  | 1x           | 0.92x           | 1.61x               | **20.73x**      |
| OASST        | 1x           | 1.19x           | 2.17x               | **14.83x**      |
| Slim Orca    | 1x           | 1.18x           | 2.22x               | **14.82x**      |

- Benchmarking table below was conducted by [🤗Hugging Face](https://huggingface.co/blog/unsloth-trl).

| Free Colab T4 | Dataset | 🤗Hugging Face | Pytorch 2.1.1 | 🦥Unsloth | 🦥 VRAM reduction |
| --- | --- | --- | --- | --- | --- |
| Llama-2 7b | OASST | 1x | 1.19x | 1.95x | -43.3% |
| Mistral 7b | Alpaca | 1x | 1.07x | 1.56x | -13.7% |
| Tiny Llama 1.1b | Alpaca | 1x | 2.06x | 3.87x | -73.8% |
| DPO with Zephyr | Ultra Chat | 1x | 1.09x | 1.55x | -18.6% |

![](https://i.ibb.co/sJ7RhGG/image-41.png)

## 💾 Installation Instructions
### Conda Installation
Select either `pytorch-cuda=11.8` for CUDA 11.8 or `pytorch-cuda=12.1` for CUDA 12.1. If you have `mamba`, use `mamba` instead of `conda` for faster solving. See this [Github issue](https://github.com/unslothai/unsloth/issues/73) for help on debugging Conda installs.
```bash
conda create --name unsloth_env python=3.10
conda activate unsloth_env

conda install pytorch-cuda=<12.1/11.8> pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

pip install --no-deps trl peft accelerate bitsandbytes
```

### Pip Installation
Do **NOT** use this if you have Anaconda. You must use the Conda install method, or else stuff will BREAK.

1. Find your CUDA version via
```python
import torch; torch.version.cuda
```
2. For Pytorch 2.1.0: You can update Pytorch via Pip (interchange `cu121` / `cu118`). Go to https://pytorch.org/ to learn more. Select either `cu118` for CUDA 11.8 or `cu121` for CUDA 12.1. If you have a RTX 3060 or higher (A100, H100 etc), use the `"ampere"` path. For Pytorch 2.1.1: go to step 3. For Pytorch 2.2.0: go to step 4.
```bash
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton \
  --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere] @ git+https://github.com/unslothai/unsloth.git"
```
3. For Pytorch 2.1.1: Use the `"ampere"` path for newer RTX 30xx GPUs or higher.
```bash
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.1 triton \
  --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install "unsloth[cu118-torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere-torch211] @ git+https://github.com/unslothai/unsloth.git"
```
4. For Pytorch 2.2.0: Use the `"ampere"` path for newer RTX 30xx GPUs or higher.
```bash
pip install --upgrade --force-reinstall --no-cache-dir torch==2.2.0 triton \
  --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install "unsloth[cu118-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere-torch220] @ git+https://github.com/unslothai/unsloth.git"
```
5. If you get errors, try the below first, then go back to step 1:
```bash
pip install --upgrade pip
```
6. For Pytorch 2.2.1:
```bash
# RTX 3090, 4090 Ampere GPUs:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps packaging ninja einops flash-attn xformers trl peft accelerate bitsandbytes

# Pre Ampere RTX 2080, T4, GTX 1080 GPUs:
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps xformers trl peft accelerate bitsandbytes
```
7. To troubleshoot installs try the below (all must succeed). Xformers should mostly all be available.
```bash
nvcc
python -m xformers.info
python -m bitsandbytes
```

## 📜 Documentation
- Go to our [Wiki page](https://github.com/unslothai/unsloth/wiki) for saving to GGUF, checkpointing, evaluation and more!
- We support Huggingface's TRL, Trainer, Seq2SeqTrainer or even Pytorch code!
- We're in 🤗Hugging Face's official docs! Check out the [SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth) and [DPO docs](https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth)!

```python
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
max_seq_length = 2048 # Supports RoPE Scaling interally, so choose any!
# Get LAION dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files = {"train" : url}, split = "train")

# 4bit pre quantized models we support - 4x faster downloading!
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
] # Go to https://huggingface.co/unsloth for more 4-bit models!

# Load Llama model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-bnb-4bit", # Supports Llama, Mistral - replace this!
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)
trainer.train()

# Go to https://github.com/unslothai/unsloth/wiki for advanced tips like
# (1) Saving to GGUF / merging to 16bit for vLLM
# (2) Continued training from a saved LoRA adapter
# (3) Adding an evaluation loop / OOMs
# (4) Cutomized chat templates
```

<a name="DPO"></a>
## DPO Support
DPO (Direct Preference Optimization), PPO, Reward Modelling all seem to work as per 3rd party independent testing from [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory). We have a preliminary Google Colab notebook for reproducing Zephyr on Tesla T4 here: [notebook](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing).

We're in 🤗Hugging Face's official docs! We're on the [SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth) and the [DPO docs](https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth)!

```python
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
import torch
from transformers import TrainingArguments
from trl import DPOTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/zephyr-sft-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Do model patching and add fast LoRA weights
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 64,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
)

dpo_trainer = DPOTrainer(
    model = model,
    ref_model = None,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        seed = 42,
        output_dir = "outputs",
    ),
    beta = 0.1,
    train_dataset = YOUR_DATASET_HERE,
    # eval_dataset = YOUR_DATASET_HERE,
    tokenizer = tokenizer,
    max_length = 1024,
    max_prompt_length = 512,
)
dpo_trainer.train()
```

## 🥇 Detailed Benchmarking Tables
- Click "Code" for fully reproducible examples
- "Unsloth Equal" is a preview of our PRO version, with code stripped out. All settings and the loss curve remains identical.
- For the full list of benchmarking tables, [go to our website](https://unsloth.ai/blog/mistral-benchmark#Benchmark%20tables)
  
| 1 A100 40GB | 🤗Hugging Face | Flash Attention 2 | 🦥Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|-------------|-------------|-----------------|--------------|---------------|-------------|
| Alpaca       | 1x          | 1.04x       | 1.98x           | 2.48x        | 5.32x         | **15.64x**      |
| code | [Code](https://colab.research.google.com/drive/1u4dBeM-0vGNVmmO6X7cScAut-Hyt4KDF?usp=sharing) |    [Code](https://colab.research.google.com/drive/1fgTOxpMbVjloQBvZyz4lF4BacKSZOB2A?usp=sharing) |    [Code](https://colab.research.google.com/drive/1YIPY_18xm-K0iJDgvNkRoJsgkPMPAO3G?usp=sharing) |    [Code](https://colab.research.google.com/drive/1ANW8EFL3LVyTD7Gq4TkheC1Z7Rxw-rHp?usp=sharing) | | |
| seconds| 1040 | 1001 | 525 | 419 | 196 | 67  |
| memory MB| 18235 | 15365 | 9631 | 8525 | | |
| % saved| | 15.74 | 47.18 | 53.25 | | | |

### Llama-Factory 3rd party benchmarking
- [Link to performance table.](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-Comparison) TGS: tokens per GPU per second. Model: LLaMA2-7B. GPU: NVIDIA A100 * 1. Batch size: 4. Gradient accumulation: 2. LoRA rank: 8. Max length: 1024.

| Method | Bits | TGS | GRAM | Speed |
| --- | --- | --- | --- | --- |
| HF | 16 | 2392 | 18GB | 100% |
| HF+FA2 | 16 | 2954 | 17GB | 123% |
| Unsloth+FA2 | 16 | 4007 | 16GB | **168%** |
| HF | 4 | 2415 | 9GB | 101% |
| Unsloth+FA2 | 4 | 3726 | 7GB | **160%** |

### Performance comparisons between popular models
<details>
  <summary>Click for specific model benchmarking tables (Mistral 7b, CodeLlama 34b etc.)</summary>
  
### Mistral 7b
| 1 A100 40GB | Hugging Face | Flash Attention 2 | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|-------------|-------------|-----------------|--------------|---------------|-------------|
| Mistral 7B Slim Orca  | 1x | 1.15x        | 2.15x        | 2.53x            | 4.61x         | **13.69x**         |
| code | [Code](https://colab.research.google.com/drive/1mePk3KzwTD81hr5mcNcs_AX3Kbg_Ha0x?usp=sharing) | [Code](https://colab.research.google.com/drive/1dgHxjvTmX6hb0bPcLp26RXSE6_n9DKj7?usp=sharing) | [Code](https://colab.research.google.com/drive/1SKrKGV-BZoU4kv5q3g0jtE_OhRgPtrrQ?usp=sharing) | [Code](https://colab.research.google.com/drive/18yOiyX0T81mTwZqOALFSCX_tSAqju6aD?usp=sharing) | |
| seconds      | 1813        | 1571        | 842             | 718          | 393           | 132         |
| memory MB    | 32853       | 19385       | 12465           | 10271        |          |        |
| % saved| | 40.99      | 62.06       | 68.74           |         |          |

### CodeLlama 34b
| 1 A100 40GB | Hugging Face | Flash Attention 2 | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|-------------|-------------|-----------------|--------------|---------------|-------------|
| Code Llama 34B   | OOM ❌         | 0.99x        | 1.87x           | 2.61x        | 4.27x      | 12.82x      |
| code | [▶️ Code](https://colab.research.google.com/drive/1ykfz3BqrtC_AUFegCzUQjjfUNlxp6Otc?usp=sharing) | [Code](https://colab.research.google.com/drive/12ZypxQh7OC6kBXvWZI-5d05I4m-B_hoR?usp=sharing) | [Code](https://colab.research.google.com/drive/1gdHyAx8XJsz2yNV-DHvbHjR1iCef5Qmh?usp=sharing) | [Code](https://colab.research.google.com/drive/1fm7wqx9MJ0kRrwKOfmLkK1Rmw-pySahB?usp=sharing) | |
| seconds      | 1953  | 1982  | 1043  | 748   | 458   | 152   |
| memory MB    | 40000 | 33217 | 27413 | 22161 |       | |
| % saved|    | 16.96| 31.47 | 44.60 |       | | |

### 1 Tesla T4

| 1 T4 16GB  | Hugging Face | Flash Attention | Unsloth Open    | Unsloth Pro Equal | Unsloth Pro   | Unsloth Max |
|--------------|-------------|-----------------|-----------------|---------------|---------------|-------------|
| Alpaca       | 1x          | 1.09x           | 1.69x           | 1.79x         | 2.93x          | **8.3x**        |
| code | [▶️ Code](https://colab.research.google.com/drive/1XpLIV4s8Bj5uryB-X2gqM88oRGHEGdaB?usp=sharing) |    [Code](https://colab.research.google.com/drive/1LyXu6CjuymQg6ddHX8g1dpUvrMa1nn4L?usp=sharing) |    [Code](https://colab.research.google.com/drive/1gsv4LpY7C32otl1rgRo5wXTk4HIitXoM?usp=sharing) |    [Code](https://colab.research.google.com/drive/1VtULwRQwhEnVdNryjm27zXfdSM1tNfFK?usp=sharing) | | |
| seconds       | 1599        | 1468        | 942             | 894          | 545           | 193         |
| memory MB       | 7199        | 7059        | 6459            | 5443         |               |             |
| % saved        |         | 1.94        | 10.28           | 24.39        |               | |

### 2 Tesla T4s via DDP

 | 2 T4 DDP | Hugging Face | Flash Attention | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|----------|-------------|-----------------|--------------|---------------|-------------|
| Alpaca       | 1x       | 0.99x       | 4.95x           | 4.44x        | 7.28x         | **20.61x**      |
| code | [▶️ Code](https://www.kaggle.com/danielhanchen/hf-original-alpaca-t4-ddp) |   [Code](https://www.kaggle.com/danielhanchen/hf-sdpa-alpaca-t4-ddp) |   [Code](https://www.kaggle.com/danielhanchen/unsloth-alpaca-t4-ddp) | | |
| seconds       | 9882     | 9946        | 1996            | 2227         | 1357          | 480         |
| memory MB| 9176 | 9128 | 6904 | 6782 |  | |
| % saved |     | 0.52 | 24.76 | 26.09 |  | | |
</details>

### Performance comparisons on 1 Tesla T4 GPU:
<details>
  <summary>Click for Time taken for 1 epoch</summary>

One Tesla T4 on Google Colab
`bsz = 2, ga = 4, max_grad_norm = 0.3, num_train_epochs = 1, seed = 3047, lr = 2e-4, wd = 0.01, optim = "adamw_8bit", schedule = "linear", schedule_steps = 10`

| System | GPU | Alpaca (52K) | LAION OIG (210K) | Open Assistant (10K) | SlimOrca (518K) |
| --- | --- | --- | --- | --- | --- |
| Huggingface | 1 T4 | 23h 15m | 56h 28m | 8h 38m | 391h 41m |
| Unsloth Open | 1 T4 | 13h 7m (1.8x) | 31h 47m (1.8x) | 4h 27m (1.9x) | 240h 4m (1.6x) |
| Unsloth Pro | 1 T4 | 3h 6m (7.5x) | 5h 17m (10.7x) | 1h 7m (7.7x) | 59h 53m (6.5x) |
| Unsloth Max | 1 T4 | 2h 39m (8.8x) | 4h 31m (12.5x) | 0h 58m (8.9x) | 51h 30m (7.6x) |

**Peak Memory Usage**

| System | GPU | Alpaca (52K) | LAION OIG (210K) | Open Assistant (10K) | SlimOrca (518K) |
| --- | --- | --- | --- | --- | --- |
| Huggingface | 1 T4 | 7.3GB | 5.9GB | 14.0GB | 13.3GB |
| Unsloth Open | 1 T4 | 6.8GB | 5.7GB | 7.8GB | 7.7GB |
| Unsloth Pro | 1 T4 | 6.4GB | 6.4GB | 6.4GB | 6.4GB |
| Unsloth Max | 1 T4 | 11.4GB | 12.4GB | 11.9GB | 14.4GB |
</details>

<details>
  <summary>Click for Performance Comparisons on 2 Tesla T4 GPUs via DDP:</summary>
**Time taken for 1 epoch**

Two Tesla T4s on Kaggle
`bsz = 2, ga = 4, max_grad_norm = 0.3, num_train_epochs = 1, seed = 3047, lr = 2e-4, wd = 0.01, optim = "adamw_8bit", schedule = "linear", schedule_steps = 10`

| System | GPU | Alpaca (52K) | LAION OIG (210K) | Open Assistant (10K) | SlimOrca (518K) * |
| --- | --- | --- | --- | --- | --- |
| Huggingface | 2 T4 | 84h 47m | 163h 48m | 30h 51m | 1301h 24m * |
| Unsloth Pro | 2 T4 | 3h 20m (25.4x) | 5h 43m (28.7x) | 1h 12m (25.7x) | 71h 40m (18.1x) * |
| Unsloth Max | 2 T4 | 3h 4m (27.6x) | 5h 14m (31.3x) | 1h 6m (28.1x) | 54h 20m (23.9x) * |

**Peak Memory Usage on a Multi GPU System (2 GPUs)**

| System | GPU | Alpaca (52K) | LAION OIG (210K) | Open Assistant (10K) | SlimOrca (518K) * |
| --- | --- | --- | --- | --- | --- |
| Huggingface | 2 T4 | 8.4GB \| 6GB | 7.2GB \| 5.3GB | 14.3GB \| 6.6GB | 10.9GB \| 5.9GB * |
| Unsloth Pro | 2 T4 | 7.7GB \| 4.9GB | 7.5GB \| 4.9GB | 8.5GB \| 4.9GB | 6.2GB \| 4.7GB * |
| Unsloth Max | 2 T4 | 10.5GB \| 5GB | 10.6GB \| 5GB | 10.6GB \| 5GB | 10.5GB \| 5GB * |

* Slim Orca `bsz=1` for all benchmarks since `bsz=2` OOMs. We can handle `bsz=2`, but we benchmark it with `bsz=1` for consistency.
</details>

![](https://i.ibb.co/sJ7RhGG/image-41.png)
<br>

### Thank You to
- [HuyNguyen-hust](https://github.com/HuyNguyen-hust) for making [RoPE Embeddings 28% faster](https://github.com/unslothai/unsloth/pull/238)
- [RandomInternetPreson](https://github.com/RandomInternetPreson) for confirming WSL support
- [152334H](https://github.com/152334H) for experimental DPO support
- [atgctg](https://github.com/atgctg) for syntax highlighting
