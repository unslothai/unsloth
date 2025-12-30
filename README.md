<div align="center">

  <a href="https://unsloth.ai/docs"><picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png">
    <img alt="unsloth logo" src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20black%20text.png" height="110" style="max-width: 100%;">
  </picture></a>
  
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb"><img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/start free finetune button.png" width="154"></a>
<a href="https://discord.com/invite/unsloth"><img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/Discord button.png" width="165"></a>
<a href="https://unsloth.ai/docs"><img src="https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/images/Documentation%20Button.png" width="137"></a>

### Train gpt-oss, DeepSeek, Gemma, Qwen & Llama 2x faster with 70% less VRAM!

![](https://i.ibb.co/sJ7RhGG/image-41.png)

</div>

## ✨ Train for Free

Notebooks are beginner friendly. Read our [guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide). Add dataset, run, then deploy your trained model.

| Model | Free Notebooks | Performance | Memory use |
|-----------|---------|--------|----------|
| **gpt-oss (20B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-Fine-tuning.ipynb)               | 1.5x faster | 70% less |
| **Mistral Ministral 3 (3B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Ministral_3_VL_(3B)_Vision.ipynb)               | 1.5x faster | 60% less |
| **gpt-oss (20B): GRPO**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb)               | 2x faster | 80% less |
| **Qwen3: Advanced GRPO**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)               | 2x faster | 50% less |
| **Qwen3-VL (8B): GSPO**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb)               | 1.5x faster | 80% less |
| **Gemma 3 (270M)** | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(270M).ipynb)               | 1.7x faster | 60% less |
| **Gemma 3n (4B)**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3N_(4B)-Conversational.ipynb)               | 1.5x faster | 50% less |
| **DeepSeek-OCR (3B)**    | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Deepseek_OCR_(3B).ipynb)               | 1.5x faster | 30% less |
| **Llama 3.1 (8B) Alpaca**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb)               | 2x faster | 70% less |
| **Llama 3.2 Conversational**      | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb)               | 2x faster | 70% less |
| **Orpheus-TTS (3B)**     | [▶️ Start for free](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Orpheus_(3B)-TTS.ipynb)               | 1.5x faster | 50% less |

- See all our notebooks for: [Kaggle](https://github.com/unslothai/notebooks?tab=readme-ov-file#-kaggle-notebooks), [GRPO](https://unsloth.ai/docs/get-started/unsloth-notebooks#grpo-reasoning-rl-notebooks), [TTS](https://unsloth.ai/docs/get-started/unsloth-notebooks#text-to-speech-tts-notebooks) & [Vision](https://unsloth.ai/docs/get-started/unsloth-notebooks#vision-multimodal-notebooks)
- See [all our models](https://unsloth.ai/docs/get-started/unsloth-model-catalog) and [all our notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)
- See detailed documentation for Unsloth [here](https://unsloth.ai/docs)

## ⚡ Quickstart
### Linux or WSL
```bash
pip install unsloth
```
### Windows
For Windows, `pip install unsloth` works only if you have Pytorch installed. Read our [Windows Guide](https://unsloth.ai/docs/get-started/install-and-update/windows-installation).

### Docker
Use our official [Unsloth Docker image](https://hub.docker.com/r/unsloth/unsloth) ```unsloth/unsloth``` container. Read our [Docker Guide](https://unsloth.ai/docs/get-started/install-and-update/docker).

### Blackwell & DGX Spark
For RTX 50x, B200, 6000 GPUs: `pip install unsloth`. Read our [Blackwell Guide](https://unsloth.ai/docs/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth) and [DGX Spark Guide](https://unsloth.ai/docs/basics/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth) for more details.

## 🦥 Unsloth News
- New RoPE & MLP **Triton Kernels** & **Padding Free + Packing**: 3x faster training & 30% less VRAM. [Blog](https://unsloth.ai/docs/new/3x-faster-training-packing)
- **New Mistral**: Run Ministral 3 or Devstral 2 and fine-tune with vision/RL sodoku notebooks. [Guide](https://unsloth.ai/docs/models/ministral-3) • [Notebooks](https://unsloth.ai/docs/models/ministral-3#fine-tuning-ministral-3)
- **500K Context**: Training a 20B model with >500K context is now possible on an 80GB GPU. [Blog](https://unsloth.ai/docs/new/500k-context-length-fine-tuning)
- **FP8 Reinforcement Learning**: You can now do FP8 GRPO on consumer GPUs. [Blog](https://unsloth.ai/docs/new/fp8-reinforcement-learning) • [Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_8B_FP8_GRPO.ipynb)
- **DeepSeek-OCR**: Fine-tune to improve language understanding by 89%. [Guide](https://unsloth.ai/docs/models/deepseek-ocr-how-to-run-and-fine-tune) • [Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Deepseek_OCR_(3B).ipynb)
- **Docker**: Use Unsloth with no setup & environment issues with our new image. [Guide](https://unsloth.ai/docs/new/how-to-fine-tune-llms-with-unsloth-and-docker) • [Docker image](https://hub.docker.com/r/unsloth/unsloth)
- **gpt-oss RL**: Introducing the fastest possible inference for gpt-oss RL! [Read blog](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/gpt-oss-reinforcement-learning)
- **Vision RL**: You can now train VLMs with GRPO or GSPO in Unsloth! [Read guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/vision-reinforcement-learning-vlm-rl)
- **gpt-oss** by OpenAI: Read our [Unsloth Flex Attention](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training) blog and [gpt-oss Guide](https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune). 20B works on 14GB VRAM. 120B on 65GB.

<details>
  <summary>Click for more news</summary>

- **Quantization-Aware Training**: We collabed with Pytorch, recovering ~70% accuracy. [Read blog](https://unsloth.ai/docs/basics/quantization-aware-training-qat)
- **Memory-efficient RL**: We're introducing even better RL. Our new kernels & algos allows faster RL with 50% less VRAM & 10× more context. [Read blog](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/memory-efficient-rl)
- **Gemma 3n** by Google: [Read Blog](https://unsloth.ai/docs/models/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune). We [uploaded GGUFs, 4-bit models](https://huggingface.co/collections/unsloth/gemma-3n-685d3874830e49e1c93f9339).
- **[Text-to-Speech (TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning)** is now supported, including `sesame/csm-1b` and STT `openai/whisper-large-v3`.
- **[Qwen3](https://unsloth.ai/docs/models/qwen3-how-to-run-and-fine-tune)** is now supported. Qwen3-30B-A3B fits on 17.5GB VRAM.
- Introducing **[Dynamic 2.0](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)** quants that set new benchmarks on 5-shot MMLU & Aider Polyglot.
- [**EVERYTHING** is now supported](https://unsloth.ai/blog/gemma3#everything) - all models (TTS, BERT, Mamba), FFT, etc. [MultiGPU](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) coming soon. Enable FFT with `full_finetuning = True`, 8-bit with `load_in_8bit = True`.
- 📣 [DeepSeek-R1](https://unsloth.ai/blog/deepseek-r1) - run or fine-tune them [with our guide](https://unsloth.ai/blog/deepseek-r1). All model uploads: [here](https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5).
- 📣 Introducing Long-context [Reasoning (GRPO)](https://unsloth.ai/blog/grpo) in Unsloth. Train your own reasoning model with just 5GB VRAM. Transform Llama, Phi, Mistral etc. into reasoning LLMs!
- 📣 Introducing Unsloth [Dynamic 4-bit Quantization](https://unsloth.ai/blog/dynamic-4bit)! We dynamically opt not to quantize certain parameters and this greatly increases accuracy while only using <10% more VRAM than BnB 4-bit. See our collection on [Hugging Face here.](https://huggingface.co/collections/unsloth/unsloth-4-bit-dynamic-quants-67503bb873f89e15276c44e7)
- 📣 **[Llama 4](https://unsloth.ai/blog/llama4)** by Meta, including Scout & Maverick are now supported.
- 📣 [Phi-4](https://unsloth.ai/blog/phi4) by Microsoft: We also [fixed bugs](https://unsloth.ai/blog/phi4) in Phi-4 and [uploaded GGUFs, 4-bit](https://huggingface.co/collections/unsloth/phi-4-all-versions-677eecf93784e61afe762afa).
- 📣 [Vision models](https://unsloth.ai/blog/vision) now supported! [Llama 3.2 Vision (11B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(11B)-Vision.ipynb), [Qwen 2.5 VL (7B)](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2_VL_(7B)-Vision.ipynb) and [Pixtral (12B) 2409](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Pixtral_(12B)-Vision.ipynb)
- 📣 [Llama 3.3 (70B)](https://huggingface.co/collections/unsloth/llama-33-all-versions-67535d7d994794b9d7cf5e9f), Meta's latest model is supported.
- 📣 We worked with Apple to add [Cut Cross Entropy](https://arxiv.org/abs/2411.09009). Unsloth now supports 89K context for Meta's Llama 3.3 (70B) on a 80GB GPU - 13x longer than HF+FA2. For Llama 3.1 (8B), Unsloth enables 342K context, surpassing its native 128K support.
- 📣 We found and helped fix a [gradient accumulation bug](https://unsloth.ai/blog/gradient)! Please update Unsloth and transformers.
- 📣 We cut memory usage by a [further 30%](https://unsloth.ai/blog/long-context) and now support [4x longer context windows](https://unsloth.ai/blog/long-context)!
</details>

## 🔗 Links and Resources
| Type                                                                                                                                      | Links                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| <img width="15" src="https://redditinc.com/hs-fs/hubfs/Reddit%20Inc/Brand/Reddit_Logo.png" />  **r/unsloth Reddit**                       | [Join Reddit community](https://reddit.com/r/unsloth)                          |
| 📚 **Documentation & Wiki**                                                                                                               | [Read Our Docs](https://unsloth.ai/docs)                                       |
| <img width="13" src="https://upload.wikimedia.org/wikipedia/commons/0/09/X_(formerly_Twitter)_logo_late_2025.svg" />  **Twitter (aka X)** | [Follow us on X](https://twitter.com/unslothai)                                |
| 💾 **Installation**                                                                                                                       | [Pip & Docker Install](https://unsloth.ai/docs/get-started/install-and-update) |
| 🔮 **Our Models**                                                                                                                         | [Unsloth Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog)   |
| ✍️ **Blog**                                                                                                                               | [Read our Blogs](https://unsloth.ai/blog)                                      |

## ⭐ Key Features

* Supports **full-finetuning**, pretraining, 4b-bit, 16-bit and **FP8** training
* Supports **all models** including [TTS](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning), multimodal, [BERT](https://unsloth.ai/docs/get-started/unsloth-notebooks#other-important-notebooks) and more! Any model that works in transformers, works in Unsloth.
* The most efficient library for [Reinforcement Learning (RL)](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), using 80% less VRAM. Supports GRPO, GSPO, DrGRPO, DAPO etc.
* **0% loss in accuracy** - no approximation methods - all exact.
* Export and [deploy your model](https://unsloth.ai/docs/basics/inference-and-deployment) to GGUF, llama.cpp, vLLM, SGLang and Hugging Face.
* Supports NVIDIA (since 2018), [AMD](https://unsloth.ai/docs/get-started/install-and-update/amd) and Intel GPUs. Minimum CUDA Capability 7.0 (V100, T4, Titan V, RTX 20, 30, 40x, A100, H100, L40 etc)
* Works on **Linux**, WSL and **Windows**
* All kernels written in OpenAI's Triton language. Manual backprop engine.
* If you trained a model with 🦥Unsloth, you can use this cool sticker!   <img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/made with unsloth.png" width="200" align="center" />

## 💾 Install Unsloth
You can also see our docs for more detailed installation and updating instructions [here](https://unsloth.ai/docs/get-started/install-and-update).

Unsloth supports Python 3.13 or lower.

### Pip Installation
**Install with pip (recommended) for Linux devices:**
```
pip install unsloth
```
**To update Unsloth:**
```
pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
```
See [here](#advanced-pip-installation) for advanced pip install instructions.
### Windows Installation

1. **Install NVIDIA Video Driver:**
  You should install the latest driver for your GPU. Download drivers here: [NVIDIA GPU Driver](https://www.nvidia.com/Download/index.aspx).

3. **Install Visual Studio C++:**
   You will need Visual Studio, with C++ installed. By default, C++ is not installed with [Visual Studio](https://visualstudio.microsoft.com/vs/community/), so make sure you select all of the C++ options. Also select options for Windows 10/11 SDK. For detailed instructions with options, see [here](https://unsloth.ai/docs/get-started/install-and-update/windows-installation#method-3-windows-directly).

5. **Install CUDA Toolkit:**
   Follow the instructions to install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive).

6. **Install PyTorch:**
   You will need the correct version of PyTorch that is compatible with your CUDA drivers, so make sure to select them carefully.
   [Install PyTorch](https://pytorch.org/get-started/locally/).

7. **Install Unsloth:**
   
```python
pip install unsloth
```

#### Advanced/Troubleshooting
For **advanced installation instructions** or if you see weird errors during installations:

First try using an isolated environment via then `pip install unsloth`
```bash
python -m venv unsloth
source unsloth/bin/activate
pip install unsloth
```

1. Install `torch` and `triton`. Go to https://pytorch.org to install it. For example `pip install torch torchvision torchaudio triton`
2. Confirm if CUDA is installed correctly. Try `nvcc`. If that fails, you need to install `cudatoolkit` or CUDA drivers.
3. Install `xformers` manually via:
  ```bash
  pip install ninja
  pip install -v --no-build-isolation -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
  ```
    Check if `xformers` succeeded with `python -m xformers.info` Go to https://github.com/facebookresearch/xformers. Another option is to install `flash-attn` for Ampere GPUs and ignore `xformers`

5. For GRPO runs, you can try installing `vllm` and seeing if `pip install vllm` succeeds.
6. Double check that your versions of Python, CUDA, CUDNN, `torch`, `triton`, and `xformers` are compatible with one another. The [PyTorch Compatibility Matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix) may be useful. 
5. Finally, install `bitsandbytes` and check it with `python -m bitsandbytes`

### Conda Installation (Optional)
`⚠️Only use Conda if you have it. If not, use Pip`. Select either `pytorch-cuda=11.8,12.1` for CUDA 11.8 or CUDA 12.1. We support `python=3.10,3.11,3.12`.
```bash
conda create --name unsloth_env \
    python=3.11 \
    pytorch-cuda=12.1 \
    pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers \
    -y
conda activate unsloth_env

pip install unsloth
```

<details>
  <summary>If you're looking to install Conda in a Linux environment, <a href="https://docs.anaconda.com/miniconda/">read here</a>, or run the below 🔽</summary>
  
  ```bash
  mkdir -p ~/miniconda3
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
  bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
  rm -rf ~/miniconda3/miniconda.sh
  ~/miniconda3/bin/conda init bash
  ~/miniconda3/bin/conda init zsh
  ```
</details>

### Advanced Pip Installation
`⚠️Do **NOT** use this if you have Conda.` Pip is a bit more complex since there are dependency issues. The pip command is different for `torch 2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9` and CUDA versions.

For other torch versions, we support `torch211`, `torch212`, `torch220`, `torch230`, `torch240`, `torch250`, `torch260`, `torch270`, `torch280`, `torch290` and for CUDA versions, we support `cu118` and `cu121` and `cu124`. For Ampere devices (A100, H100, RTX3090) and above, use `cu118-ampere` or `cu121-ampere` or `cu124-ampere`.

For example, if you have `torch 2.4` and `CUDA 12.1`, use:
```bash
pip install --upgrade pip
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
```

Another example, if you have `torch 2.9` and `CUDA 13.0`, use:
```bash
pip install --upgrade pip
pip install "unsloth[cu130-torch290] @ git+https://github.com/unslothai/unsloth.git"
```

And other examples:
```bash
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu118-torch240] @ git+https://github.com/unslothai/unsloth.git"

pip install "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"

pip install "unsloth[cu121-torch250] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu124-ampere-torch250] @ git+https://github.com/unslothai/unsloth.git"
```

Or, run the below in a terminal to get the **optimal** pip installation command:
```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```

Or, run the below manually in a Python REPL:
```python
try: import torch
except: raise ImportError('Install torch via `pip install torch`')
from packaging.version import Version as V
import re
v = V(re.match(r"[0-9\.]{3,}", torch.__version__).group(0))
cuda = str(torch.version.cuda)
is_ampere = torch.cuda.get_device_capability()[0] >= 8
USE_ABI = torch._C._GLIBCXX_USE_CXX11_ABI
if cuda not in ("11.8", "12.1", "12.4", "12.6", "12.8", "13.0"): raise RuntimeError(f"CUDA = {cuda} not supported!")
if   v <= V('2.1.0'): raise RuntimeError(f"Torch = {v} too old!")
elif v <= V('2.1.1'): x = 'cu{}{}-torch211'
elif v <= V('2.1.2'): x = 'cu{}{}-torch212'
elif v  < V('2.3.0'): x = 'cu{}{}-torch220'
elif v  < V('2.4.0'): x = 'cu{}{}-torch230'
elif v  < V('2.5.0'): x = 'cu{}{}-torch240'
elif v  < V('2.5.1'): x = 'cu{}{}-torch250'
elif v <= V('2.5.1'): x = 'cu{}{}-torch251'
elif v  < V('2.7.0'): x = 'cu{}{}-torch260'
elif v  < V('2.7.9'): x = 'cu{}{}-torch270'
elif v  < V('2.8.0'): x = 'cu{}{}-torch271'
elif v  < V('2.8.9'): x = 'cu{}{}-torch280'
elif v  < V('2.9.1'): x = 'cu{}{}-torch290'
elif v  < V('2.9.2'): x = 'cu{}{}-torch291'
else: raise RuntimeError(f"Torch = {v} too new!")
if v > V('2.6.9') and cuda not in ("11.8", "12.6", "12.8", "13.0"): raise RuntimeError(f"CUDA = {cuda} not supported!")
x = x.format(cuda.replace(".", ""), "-ampere" if False else "") # is_ampere is broken due to flash-attn
print(f'pip install --upgrade pip && pip install --no-deps git+https://github.com/unslothai/unsloth-zoo.git && pip install "unsloth[{x}] @ git+https://github.com/unslothai/unsloth.git" --no-build-isolation')
```
### Docker Installation
You can use our pre-built Docker container with all dependencies to use Unsloth instantly with no setup required.
[Read our guide](https://unsloth.ai/docs/get-started/install-and-update/docker).

This container requires installing [NVIDIA's Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker run -d -e JUPYTER_PASSWORD="mypassword" \
  -p 8888:8888 -p 2222:22 \
  -v $(pwd)/work:/workspace/work \
  --gpus all \
  unsloth/unsloth
```

Access Jupyter Lab at `http://localhost:8888` and start fine-tuning!

## 📜 Documentation
* Go to our official [Documentation](https://unsloth.ai/docs) for [running models](https://unsloth.ai/docs/basics/inference-and-deployment), [saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf), [checkpointing](https://unsloth.ai/docs/basics/finetuning-from-last-checkpoint), [evaluation](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide#evaluation) and more!
* Read our Guides for: [Fine-tuning](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide), [Reinforcement Learning](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide), [Text-to-Speech (TTS)](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning), [Vision](https://unsloth.ai/docs/basics/vision-fine-tuning) and [any model](https://unsloth.ai/docs/models/tutorials-how-to-fine-tune-and-run-llms).
* We support Huggingface's transformers, TRL, Trainer, Seq2SeqTrainer and Pytorch code.

Unsloth example code to fine-tune gpt-oss-20b:

```python
from unsloth import FastLanguageModel, FastModel
import torch
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
max_seq_length = 2048 # Supports RoPE Scaling internally, so choose any!
# Get LAION dataset
url = "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"
dataset = load_dataset("json", data_files = {"train" : url}, split = "train")

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/gpt-oss-20b-unsloth-bnb-4bit", #or choose any model

] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gpt-oss-20b",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4-bit quantization. False = 16-bit LoRA.
    load_in_8bit = False, # 8-bit quantization
    load_in_16bit = False, # 16-bit LoRA
    full_finetuning = False, # Use for full fine-tuning.
    trust_remote_code = False, # Enable to support new models
    # token = "hf_...", # use one if using gated models
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
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    tokenizer = tokenizer,
    args = SFTConfig(
        max_seq_length = max_seq_length,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,
        logging_steps = 1,
        output_dir = "outputs",
        optim = "adamw_8bit",
        seed = 3407,
    ),
)
trainer.train()

# Go to https://unsloth.ai/docs for advanced tips like
# (1) Saving to GGUF / merging to 16bit for vLLM or SGLang
# (2) Continued training from a saved LoRA adapter
# (3) Adding an evaluation loop / OOMs
# (4) Customized chat templates
```

<a name="RL"></a>
## 💡 Reinforcement Learning
[RL](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) including [GRPO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide#training-with-grpo), [GSPO](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/gspo-reinforcement-learning), [**FP8** training](https://unsloth.ai/docs/new/fp8-reinforcement-learning), DrGRPO, DAPO, PPO, Reward Modelling, Online DPO all work with Unsloth.

Read our [Reinforcement Learning Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) or our [advanced RL docs](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide/advanced-rl-documentation) for batching, generation & training parameters.

List of RL notebooks:
- gpt-oss GSPO notebook: [Link](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/gpt-oss-(20B)-GRPO.ipynb)
- - ***FP8*** Qwen3-8B GRPO notebook (L4): [Link](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_8B_FP8_GRPO.ipynb)
- Qwen2.3-VL GSPO notebook: [Link](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_VL_(8B)-Vision-GRPO.ipynb)
- Advanced Qwen3 GRPO notebook: [Link](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
- ORPO notebook: [Link](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3_(8B)-ORPO.ipynb)
- DPO Zephyr notebook: [Link](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Zephyr_(7B)-DPO.ipynb)
- KTO notebook: [Link](https://colab.research.google.com/drive/1MRgGtLWuZX4ypSfGguFgC-IblTvO2ivM?usp=sharing)
- SimPO notebook: [Link](https://colab.research.google.com/drive/1Hs5oQDovOay4mFA6Y9lQhVJ8TnbFLFh2?usp=sharing)

## 🥇 Performance Benchmarking
- For our most detailed benchmarks, read our [Llama 3.3 Blog](https://unsloth.ai/blog/llama3-3).
- Benchmarking of Unsloth was also conducted by [🤗Hugging Face](https://huggingface.co/blog/unsloth-trl).

We tested using the Alpaca  Dataset, a batch size of 2, gradient accumulation steps of 4, rank = 32, and applied QLoRA on all linear layers (q, k, v, o, gate, up, down):
  
| Model          | VRAM  | 🦥 Unsloth speed | 🦥 VRAM reduction | 🦥 Longer context | 😊 Hugging Face + FA2 |
|----------------|-------|-----------------|----------------|----------------|--------------------|
| Llama 3.3 (70B)| 80GB  | 2x              | >75%           | 13x longer     | 1x                 |
| Llama 3.1 (8B) | 80GB  | 2x              | >70%           | 12x longer     | 1x                 |

### Context length benchmarks

#### Llama 3.1 (8B) max. context length
We tested Llama 3.1 (8B) Instruct and did 4bit QLoRA on all linear layers (Q, K, V, O, gate, up and down) with rank = 32 with a batch size of 1. We padded all sequences to a certain maximum sequence length to mimic long context finetuning workloads.
| GPU VRAM | 🦥Unsloth context length | Hugging Face + FA2 |
|----------|-----------------------|-----------------|
| 8 GB     | 2,972                 | OOM             |
| 12 GB    | 21,848                | 932             |
| 16 GB    | 40,724                | 2,551           |
| 24 GB    | 78,475                | 5,789           |
| 40 GB    | 153,977               | 12,264          |
| 48 GB    | 191,728               | 15,502          |
| 80 GB    | 342,733               | 28,454          |

#### Llama 3.3 (70B) max. context length
We tested Llama 3.3 (70B) Instruct on a 80GB A100 and did 4bit QLoRA on all linear layers (Q, K, V, O, gate, up and down) with rank = 32 with a batch size of 1. We padded all sequences to a certain maximum sequence length to mimic long context finetuning workloads.

| GPU VRAM | 🦥Unsloth context length | Hugging Face + FA2 |
|----------|------------------------|------------------|
| 48 GB    | 12,106                | OOM              |
| 80 GB    | 89,389                | 6,916            |

<br>

![](https://i.ibb.co/sJ7RhGG/image-41.png)
<br>

### Citation

You can cite the Unsloth repo as follows:
```bibtex
@software{unsloth,
  author = {Daniel Han, Michael Han and Unsloth team},
  title = {Unsloth},
  url = {http://github.com/unslothai/unsloth},
  year = {2023}
}
```

### Thank You to
- The [llama.cpp library](https://github.com/ggml-org/llama.cpp) that lets users save models with Unsloth
- The Hugging Face team and their libraries: [transformers](https://github.com/huggingface/transformers) and [TRL](https://github.com/huggingface/trl)
- The Pytorch and [Torch AO](https://github.com/unslothai/unsloth/pull/3391) team for their contributions
- And of course for every single person who has contributed or has used Unsloth!
