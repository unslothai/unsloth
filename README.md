<div class="align-center">
  <a href="https://unsloth.ai/"><img src="./images/unsloth new logo.png" width="350"></a>
  <a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing"><img src="./images/try live demo green.png" height="50"></a>
  <a href="https://discord.gg/u54VK8m8tk"><img src="./images/Discord.png" height="50"></a>
</div>

## Finetune Mistral, Llama 2-5x faster with 50% less memory!
| Llama 7b                    | Mistral 7b                  | CodeLlama 34b           | Llama 7b Kaggle 2x T4  |
|-----------------------------|-----------------------------|-------------------------|------------------------|
| **2.2x faster 43% less VRAM**     | **2.2x faster 62% less VRAM**     | **1.9x faster 27% less VRAM**  | **5.5x faster 44% less VRAM** |
| [⭐Llama **free** Colab 2x faster](https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing") | [⭐Mistral **free** Colab 2x faster](https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing) | [CodeLlama A100 Colab notebook](https://colab.research.google.com/drive/1y7A0AxE3y8gdj4AVkl2aZX47Xu3P1wJT?usp=sharing) | [⭐Kaggle **free** Alpaca notebook](https://www.kaggle.com/danielhanchen/unsloth-alpaca-t4-ddp)
| [Llama A100 Colab notebook](https://colab.research.google.com/drive/1YIPY_18xm-K0iJDgvNkRoJsgkPMPAO3G?usp=sharing) | [Mistral A100 Colab notebook](https://colab.research.google.com/drive/1SKrKGV-BZoU4kv5q3g0jtE_OhRgPtrrQ?usp=sharing) | 50+ more examples below! | [⭐Kaggle **free** Slim Orca notebook](https://www.kaggle.com/danielhanchen/unsloth-slimorca-t4-ddp) |

* **NEW!** [DPO](https://arxiv.org/abs/2305.18290) support. ⭐**Free!** DPO Zephyr, Mistral example! <a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing"><img src="./images/Colab.png" height="20">  [More info](#DPO) on DPO
* **NEW!** [TinyLlama 1.1b](https://github.com/jzhang38/TinyLlama) on 3T tokens! ⭐**Free!** example <a href="https://colab.research.google.com/drive/1AZghoNBQaMDgWJpi4RbffGM1h6raLUj9?usp=sharing"><img src="./images/Colab.png" height="20">
* **NEW!** We're in 🤗 Huggingface's official docs! We're on the [SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth) and the [DPO docs](https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth)!
* Supports Llama, Yi, Mistral, CodeLlama, Qwen (llamafied), Deepseek and their derived models (Open Hermes etc).
* All kernels written in [OpenAI's Triton](https://openai.com/research/triton) language. **Manual backprop engine**.
* **0% loss in accuracy** - no approximation methods - all exact.
* No change of hardware. Supports NVIDIA GPUs since 2018+. Minimum CUDA Capability 7.0 (V100, T4, Titan V, RTX 20, 30, 40x, A100, H100, L40 etc) [Check your GPU!](https://developer.nvidia.com/cuda-gpus) GTX 1070, 1080 works, but is slow.
* Works on **Linux** and **Windows** via WSL.
* **NEW!** Download 4 bit models 4x faster from 🤗 Huggingface! Eg: `unsloth/mistral-7b-bnb-4bit`
* Supports 4bit and 16bit QLoRA / LoRA finetuning via [bitsandbytes](https://github.com/TimDettmers/bitsandbytes).
* **NEW!** Want a UI for finetuning? Try [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory) and use `--use_unsloth`!
* Open source trains 5x faster - see [Unsloth Pro](https://unsloth.ai/) for **30x faster training**!

| 1 A100 40GB  | 🤗 Hugging Face | Flash Attention | 🦥 Unsloth Open Source | [🦥 Unsloth Pro](https://unsloth.ai/pricing) |
|--------------|--------------|-----------------|---------------------|-----------------|
| Alpaca       | 1x           | 1.04x           | 1.98x               | **15.64x**      |
| LAION Chip2  | 1x           | 0.92x           | 1.61x               | **20.73x**      |
| OASST        | 1x           | 1.19x           | 2.17x               | **14.83x**      |
| Slim Orca    | 1x           | 1.18x           | 2.22x               | **14.82x**      |

Join our [Discord](https://discord.gg/nsS4V5Z6ge)!

<img src="./images/unsloth made with love.png" width="200" />
If you trained a model with 🦥 Unsloth, we made a cool sticker if you want to use it!

# Installation Instructions - Conda
Select either `pytorch-cuda=11.8` for CUDA 11.8 or `pytorch-cuda=12.1` for CUDA 12.1.
```bash
conda install cudatoolkit xformers bitsandbytes pytorch pytorch-cuda=12.1 \
  -c pytorch -c nvidia -c xformers -c conda-forge -y
pip install "unsloth[conda] @ git+https://github.com/unslothai/unsloth.git"
```

# Installation Instructions - Pip
Do **NOT** use this if you have Anaconda. You must use the Conda install method, or else stuff will BREAK.

1. Find your CUDA version via
```python
import torch; torch.version.cuda
```
2. For Pytorch 2.1.0: You can update Pytorch via Pip (interchange `cu121` / `cu118`). Go to https://pytorch.org/ to learn more. Select either `cu118` for CUDA 11.8 or `cu121` for CUDA 12.1. If you have a RTX 3060 or higher (A100, H100 etc), use the `"ampere"` path. For Pytorch 2.1.1: got to step 3.
```bash
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.0 triton \
  --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118_ampere] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121_ampere] @ git+https://github.com/unslothai/unsloth.git"
```
3. For Pytorch 2.1.1: Use the `"ampere"` path for newer RTX 30xx GPUs or higher.
```bash
pip install --upgrade --force-reinstall --no-cache-dir torch==2.1.1 triton \
  --index-url https://download.pytorch.org/whl/cu121
```
```bash
pip install "unsloth[cu118_torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121_torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118_ampere_torch211] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121_ampere_torch211] @ git+https://github.com/unslothai/unsloth.git"
```
4. We're working on Pytorch 2.1.2 support.
5. If you get errors, try the below first, then go back to step 1:
```bash
pip install --upgrade pip
```

# Documentation
We support Huggingface's TRL, Trainer, Seq2SeqTrainer or even Pytorch code!

We're in 🤗 Huggingface's official docs! We're on the [SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth) and the [DPO docs](https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth)!

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
]
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
```

<a name="DPO"></a>
# DPO (Direct Preference Optimization) Support
DPO, PPO, Reward Modelling all seem to work as per 3rd party independent testing from [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory). We have a preliminary Google Colab notebook for reproducing Zephyr on Tesla T4 here: [notebook](https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing).

We're in 🤗 Huggingface's official docs! We're on the [SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth) and the [DPO docs](https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth)!

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


# Future Milestones and limitations
1. Support Mixtral.
2. Supports all Mistral, Llama type models, but some are unoptimized (Qwen with biases)
3. Dropout, bias in LoRA matrices are supported, just not optimized.

# Performance comparisons on 1 Tesla T4 GPU:
**Time taken for 1 epoch**

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

# Performance comparisons on 2 Tesla T4 GPUs via DDP:
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

# Llama-Factory 3rd party benchmarking

| Method | Bits | TGS | GRAM | Speed |
| --- | --- | --- | --- | --- |
| HF | 16 | 2392 | 18GB | 100% |
| HF+FA2 | 16 | 2954 | 17GB | 123% |
| Unsloth+FA2 | 16 | 4007 | 16GB | **168%** |
| HF | 4 | 2415 | 9GB | 101% |
| Unsloth+FA2 | 4 | 3726 | 7GB | **160%** |

[Link](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-Comparison) to performance table. TGS: tokens per GPU per second. Model: LLaMA2-7B. GPU: NVIDIA A100 * 1. Batch size: 4. Gradient accumulation: 2. LoRA rank: 8. Max length: 1024.

# How did we make it faster?
Manual autograd, Triton kernels etc. See our [Benchmark Breakdown](https://unsloth.ai/blog/mistral-benchmark) for more info!

# Troubleshooting
1. Sometimes `bitsandbytes` or `xformers` does not link properly. Try running:
```bash
!ldconfig /usr/lib64-nvidia
```
2. Windows is not supported as of yet - we rely on Xformers and Triton support, so until both packages support Windows officially, Unsloth will then support Windows.

3. If it doesn't install - maybe try updating `pip`.


# Full benchmarking tables
Click  "Code" for a fully reproducible example.
"Unsloth Equal" is a preview of our PRO version, with code stripped out. All settings and the loss curve remains identical.
| 1 A100 40GB | Hugging Face | Flash Attention 2 | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|-------------|-------------|-----------------|--------------|---------------|-------------|
| Alpaca       | 1x          | 1.04x       | 1.98x           | 2.48x        | 5.32x         | **15.64x**      |
| code | [Code](https://colab.research.google.com/drive/1u4dBeM-0vGNVmmO6X7cScAut-Hyt4KDF?usp=sharing) |    [Code](https://colab.research.google.com/drive/1fgTOxpMbVjloQBvZyz4lF4BacKSZOB2A?usp=sharing) |    [Code](https://colab.research.google.com/drive/1YIPY_18xm-K0iJDgvNkRoJsgkPMPAO3G?usp=sharing) |    [Code](https://colab.research.google.com/drive/1ANW8EFL3LVyTD7Gq4TkheC1Z7Rxw-rHp?usp=sharing) | | |
| seconds| 1040 | 1001 | 525 | 419 | 196 | 67  |
| memory MB| 18235 | 15365 | 9631 | 8525 | | |
| % saved| | 15.74 | 47.18 | 53.25 | | | |


| 1 A100 40GB | Hugging Face | Flash Attention 2 | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|-------------|-------------|-----------------|--------------|---------------|-------------|
| LAION Chip2  | 1x          | 0.92x       | 1.61x           | 1.84x        | 7.05x         | **20.73x**      |
| code |[Code](https://colab.research.google.com/drive/1gjL1TaKwc_xv2TcxJC8QWEWBG1msh3g2?usp=sharing) |    [Code](https://colab.research.google.com/drive/15vlPjMr8xDj5BFhGdqunGaOQSMqXPEXU?usp=sharing) |    [Code](https://colab.research.google.com/drive/1zPwvf-BmHyHlPMBxDsY8zS0BnQ-KKbCc?usp=sharing) |    [Code](https://colab.research.google.com/drive/1X2uHy-arRsZxqWHvKHwwW102JaMwChD2?usp=sharing) | | |
| seconds| 581  | 631  | 361 | 315 | 82  | 28  |
| memory MB| 7763  | 8047  | 7763 | 6441 | | |
| % saved| | -3.66 | 0.00  | 17.03 | | | |


| 1 A100 40GB | Hugging Face | Flash Attention 2 | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|-------------|-------------|-----------------|--------------|---------------|-------------|
| OASST        | 1x          | 1.19x       | 2.17x           | 2.66x        | 5.04x         | **14.83x**      |
| code |[Code](https://colab.research.google.com/drive/10NzDreFbuWELGUuBv0MOoC7y3MBewaNx?usp=sharing) |    [Code](https://colab.research.google.com/drive/1TwdkJ1sHsuEH-kgeCPqSFeCpOnCfz6Ou?usp=sharing) |    [Code](https://colab.research.google.com/drive/1AkwjUkOF0XeRBMT_S8Uhh74kitEsZHla?usp=sharing) |    [Code](https://colab.research.google.com/drive/1roMkp2UjbeK2t3DkNz50cRs1MT92RPFT?usp=sharing) | | |
| seconds| 1852 | 1558 | 852 | 696 | 367 | 125 |
| memory MB| 26431 | 16565 | 12267| 11223| | |
| % saved| | 37.33 | 53.59 | 57.54 | | |

| 1 A100 40GB | Hugging Face | Flash Attention 2 | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|-------------|-------------|-----------------|--------------|---------------|-------------|
| Slim Orca    | 1x          | 1.18x       | 2.22x           | 2.64x        | 5.04x         | **14.82x**      |
| code |[Code](https://colab.research.google.com/drive/1UNo1xsMl8YH7xnWnIVjDFnCAPfc0RGgu?usp=sharing) |    [Code](https://colab.research.google.com/drive/1zbphER-SKhbSWGjHTfnBLPFyTgIVvaeH?usp=sharing) |    [Code](https://colab.research.google.com/drive/156si33585iv4Uh-VILFglUmIMrNCNuc2?usp=sharing) |    [Code](https://colab.research.google.com/drive/1_mhZy7dfl9jEnJRuJBZJ5y3OwW06jgQA?usp=sharing) | | |
| seconds| 1824 | 1545 | 821 | 691 | 362 | 123 |
| memory MB| 24557 | 15681 | 10595| 9007 | | |
| % saved| | 36.14 | 56.86 | 63.32 | | |

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
| code | [Code](https://colab.research.google.com/drive/1ykfz3BqrtC_AUFegCzUQjjfUNlxp6Otc?usp=sharing) | [Code](https://colab.research.google.com/drive/12ZypxQh7OC6kBXvWZI-5d05I4m-B_hoR?usp=sharing) | [Code](https://colab.research.google.com/drive/1gdHyAx8XJsz2yNV-DHvbHjR1iCef5Qmh?usp=sharing) | [Code](https://colab.research.google.com/drive/1fm7wqx9MJ0kRrwKOfmLkK1Rmw-pySahB?usp=sharing) | |
| seconds      | 1953  | 1982  | 1043  | 748   | 458   | 152   |
| memory MB    | 40000 | 33217 | 27413 | 22161 |       | |
| % saved|    | 16.96| 31.47 | 44.60 |       | | |

### 1 Tesla T4

| 1 T4 16GB  | Hugging Face | Flash Attention | Unsloth Open    | Unsloth Pro Equal | Unsloth Pro   | Unsloth Max |
|--------------|-------------|-----------------|-----------------|---------------|---------------|-------------|
| Alpaca       | 1x          | 1.09x           | 1.69x           | 1.79x         | 2.93x          | **8.3x**        |
| code | [Code](https://colab.research.google.com/drive/1XpLIV4s8Bj5uryB-X2gqM88oRGHEGdaB?usp=sharing) |    [Code](https://colab.research.google.com/drive/1LyXu6CjuymQg6ddHX8g1dpUvrMa1nn4L?usp=sharing) |    [Code](https://colab.research.google.com/drive/1gsv4LpY7C32otl1rgRo5wXTk4HIitXoM?usp=sharing) |    [Code](https://colab.research.google.com/drive/1VtULwRQwhEnVdNryjm27zXfdSM1tNfFK?usp=sharing) | | |
| seconds       | 1599        | 1468        | 942             | 894          | 545           | 193         |
| memory MB       | 7199        | 7059        | 6459            | 5443         |               |             |
| % saved        |         | 1.94        | 10.28           | 24.39        |               | |

| 1 T4 16GB  | Hugging Face | Flash Attention | Unsloth Open    | Unsloth Pro Equal | Unsloth Pro   | Unsloth Max |
|--------------|-------------|-----------------|-----------------|---------------|---------------|-------------|
| LAION Chip2  | 1x          | 0.99x           | 1.80x           | 1.75x         | 4.15x         | **11.75x**      |
| code | [Code](https://colab.research.google.com/drive/1EtdStADehE4FVJnU2Cu6O8p9jDYdqG2L?usp=sharing) |    [Code](https://colab.research.google.com/drive/1Ik4jO68odUiQIJ_szZ3xok5fk58WpA5Q?usp=sharing) |    [Code](https://colab.research.google.com/drive/1E2nR4V3bXIWBQIUE7uR39lYPr3UikzqH?usp=sharing) |    [Code](https://colab.research.google.com/drive/13jbj8D8FOt9KyXwZt9Yf2MsYkD8CyCVR?usp=sharing) | | |
| seconds  | 952         | 955         | 529             | 543          | 229           | 81          | 
| memory MB  | 6037        | 6033        | 5797            | 4855         |               | |
| % saved   |         | 0.07        | 3.98            | 19.58        |               | |

| 1 T4 16GB  | Hugging Face | Flash Attention | Unsloth Open    | Unsloth Pro Equal | Unsloth Pro   | Unsloth Max |
|--------------|-------------|-----------------|-----------------|---------------|---------------|-------------|
| OASST        | 1x          | 1.19x           | 1.95x           | 1.86x         | 2.58x         | **7.3x**        |
| code | [Code](https://colab.research.google.com/drive/1aXzGgEM3yYB6SWy_XR81nQFWME40ksSy?usp=sharing) |    [Code](https://colab.research.google.com/drive/1-5MdIOp0cM0scC-CdRZhh8OYhnGHqct4?usp=sharing) |    [Code](https://colab.research.google.com/drive/1n-fgduZhRUsSjgpqNtVkXA3rSfE7iBdg?usp=sharing) |    [Code](https://colab.research.google.com/drive/1z_GlHr2M_bB4lQrPhdWC7dseZv23cBIy?usp=sharing) | | |
| seconds        | 2640        | 2222        | 1355            | 1421         | 1024          | 362         |
| memory MB        | 14827       | 10391       | 8413            | 7031         |               | |
| % saved         |         | 29.92       | 43.26           | 52.58        |               | |

| 1 T4 16GB  | Hugging Face | Flash Attention | Unsloth Open    | Unsloth Pro Equal | Unsloth Pro   | Unsloth Max |
|--------------|-------------|-----------------|-----------------|---------------|---------------|-------------|
| Slim Orca    | 1x          | 1.21x           | 1.77x           | 1.85x         | 2.71x         | **7.67x**       |
| code | [Code](https://colab.research.google.com/drive/15yLlJx9IE84kzx7ikky45pRcarPyUtEs?usp=sharing) |    [Code](https://colab.research.google.com/drive/16IShIBmjKULWy87I-xURpj4nztTkAF13?usp=sharing) |    [Code](https://colab.research.google.com/drive/1CJG3XLg_OQpCz71eB7Uqx7wuK_n2b-a8?usp=sharing) |    [Code](https://colab.research.google.com/drive/1UmwuWHtlrC6MAfl9mX7A_TRfo5iSHDa-?usp=sharing) | | |
| seconds    | 2735        | 2262        | 1545            | 1478         | 1009          | 356         |
| memory MB    | 13933       | 10489       | 7661            | 6563         |               | |
| % saved    |         | 24.72       | 45.02           | 52.90        |               | |

### 2 Tesla T4s via DDP

 | 2 T4 DDP | Hugging Face | Flash Attention | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|----------|-------------|-----------------|--------------|---------------|-------------|
| Alpaca       | 1x       | 0.99x       | 4.95x           | 4.44x        | 7.28x         | **20.61x**      |
| code | [Code](https://www.kaggle.com/danielhanchen/hf-original-alpaca-t4-ddp) |   [Code](https://www.kaggle.com/danielhanchen/hf-sdpa-alpaca-t4-ddp) |   [Code](https://www.kaggle.com/danielhanchen/unsloth-alpaca-t4-ddp) | | |
| seconds       | 9882     | 9946        | 1996            | 2227         | 1357          | 480         |
| memory MB| 9176 | 9128 | 6904 | 6782 |  | |
| % saved |     | 0.52 | 24.76 | 26.09 |  | | |

 | 2 T4 DDP | Hugging Face | Flash Attention | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|----------|-------------|-----------------|--------------|---------------|-------------|
| LAION Chip2  | 1x       | 1.12x       | 5.28x           | 4.21x        | 10.01x        | **28.32x**      |
| code | [Code](https://www.kaggle.com/danielhanchen/hf-original-laion-t4-ddp) |    [Code](https://www.kaggle.com/danielhanchen/hf-sdpa-laion-t4-ddp) |    [Code](https://www.kaggle.com/danielhanchen/unsloth-laion-t4-ddp) | | |
| seconds  | 5418     | 4854        | 1027            | 1286         | 541           | 191         |
| memory MB| 7316 | 7316 | 5732 | 5934 |  | |
| % saved |     | 0.00 | 21.65 | 18.89 |  | |

 | 2 T4 DDP | Hugging Face | Flash Attention | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|----------|-------------|-----------------|--------------|---------------|-------------|
| OASST (bsz=1)        | 1x       | 1.14x       | 5.56x           | 5.09x        | 5.64x         | **15.97x**      |
| code | [Code](https://www.kaggle.com/danielhanchen/hf-original-oasst-bsz1-t4-ddp) |   [Code](https://www.kaggle.com/danielhanchen/hf-sdpa-oasst-bsz1-t4-ddp) |   [Code](https://www.kaggle.com/danielhanchen/unsloth-oasst-bsz1-t4-ddp) | | | |
| seconds        | 4503 | 3955 | 811 | 885 | 798 | 282 |
| memory MB | 11896 | 11628 | 6616 | 7105 |  | |
| % saved |     | 2.25 | 44.38 | 40.27 |  | |

 | 2 T4 DDP | Hugging Face | Flash Attention | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|----------|-------------|-----------------|--------------|---------------|-------------|
| Slim Orca (bsz=1)    | 1x       | 0.97x       | 5.54x           | 4.68x        | 6.88x         | **19.46x**       |
| code | [Code](https://www.kaggle.com/danielhanchen/hf-original-slimorca-bsz1-t4-ddp) |    [Code](https://www.kaggle.com/danielhanchen/hf-sdpa-slimorca-bsz1-t4-ddp) |    [Code](https://www.kaggle.com/danielhanchen/unsloth-slimorca-bsz1-t4-ddp) | | |
| seconds | 4042 | 4158 | 729 | 863 | 588 | 208 |
| memory MB| 11010 | 11042 | 6492 | 7410 |  | |
| % saved |     | -0.29| 41.04 | 32.70 |  | | |

 | 2 T4 DDP | Hugging Face | Flash Attention | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|----------|-------------|-----------------|--------------|---------------|-------------|
| OASST (bsz=2)        | OOM ❌      | OOM ❌       |  ✓          | ✓         | ✓         | ✓ |
| code | [Code](https://www.kaggle.com/danielhanchen/hf-original-oasst-t4-ddp) |    [Code](https://www.kaggle.com/danielhanchen/hf-sdpa-oasst-t4-ddp) |    [Code](https://www.kaggle.com/danielhanchen/unsloth-oasst-t4-ddp) | | | |
| seconds        | OOM      | OOM         | 2719            | 3391         | 2794          | 987         |
| memory MB| OOM  | OOM  | 8134 | 9600 |  | |
| % saved | OOM  | OOM  |       |       |  | |

 | 2 T4 DDP | Hugging Face | Flash Attention | Unsloth Open | Unsloth Equal | Unsloth Pro | Unsloth Max |
|--------------|----------|-------------|-----------------|--------------|---------------|-------------|
| Slim Orca (bsz=2)    | OOM ❌       | OOM ❌       |  ✓          | ✓        | ✓         |✓ |
| code  | [Code](https://www.kaggle.com/danielhanchen/hf-original-slimorca-t4-ddp) |     [Code](https://www.kaggle.com/danielhanchen/hf-sdpa-slimorca-t4-ddp) |     [Code](https://www.kaggle.com/danielhanchen/unsloth-slimorca-t4-ddp) | | |
| seconds    | OOM      | OOM         | 2990            | 3444         | 2351          | 831         |
| memory MB| OOM  | OOM  | 7594 | 8881 | | |
| % saved | OOM  | OOM  |       |       |  | |

# Credits
1. [RandomInternetPreson](https://github.com/RandomInternetPreson) for confirming WSL support
2. [152334H](https://github.com/152334H) for experimental DPO support
3. [atgctg](https://github.com/atgctg) for syntax highlighting
<img src="./images/unsloth loading page render.png" width="300" />
