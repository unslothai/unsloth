<img src="./images/unsloth new logo.png" width="400" />

## 2x faster 50% less memory LLM finetuning
* Manual autograd engine.
* All kernels written in OpenAI's Triton language.
* 0% loss in accuracy.
* No change of hardware necessary.

<div class="align-center">
  <img src="./images/SlimOrca%201GPU.svg" width="400" />
  <img src="./images/LAION%202GPU.svg" width="400" />
</div>

1. Try our Colab examples for [the Alpaca 52K dataset](https://colab.research.google.com/drive/1oW55fBmwzCOrBVX66RcpptL3a99qWBxb?usp=sharing) or [the Slim Orca 518K dataset](https://colab.research.google.com/drive/1VNqLARpE8N8eYwNrUSDoHVjtbR9W0_c7?usp=sharing).
2. Try our Kaggle example for [the LAION OIG Chip2 dataset](https://www.kaggle.com/danielhanchen/unsloth-laion-chip2-kaggle)
3. Join our [Discord](https://discord.gg/nsS4V5Z6ge)!

# Installation Instructions
Unsloth currently only supports Linux distros and Pytorch >= 2.1.

You must first update Pytorch to 2.1 before using pip. If you have Conda, you MUST first upgrade your Pytorch installation with the command we provided, since it also installs xformers and bitsandbytes.

1. Find your CUDA version via
```
import torch; torch.version.cuda
```
2. For CUDA 11.8 or CUDA 12.1. If you are using Kaggle or Colab notebooks, we also provide a distro: DO NOT run this first if you have Conda - do step 3 then 2.
```
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[colab] @ git+https://github.com/unslothai/unsloth.git"
pip install "unsloth[kaggle] @ git+https://github.com/unslothai/unsloth.git"
```
3. To update Pytorch to 2.1: (You MUST run this if you have Conda FIRST)
```
conda install cudatoolkit xformers bitsandbytes pytorch pytorch-cuda=12.1 \
  -c pytorch -c nvidia -c xformers -c conda-forge -y
```
or
```
pip install --upgrade --force-reinstall --no-cache-dir torch triton \
  --index-url https://download.pytorch.org/whl/cu121
```
Change `cu121` to `cu118` for CUDA version 11.8 or 12.1. Go to https://pytorch.org/ to learn more.

# Future Milestones and limitations
1. Support sqrt gradient checkpointing which further slashes memory usage by 25%.
2. Does not support non Llama models - we do so in the future.

# Performance comparisons on 1 Tesla T4 GPU:
**Time taken for 1 epoch**

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

| System | GPU | Alpaca (52K) | LAION OIG (210K) | Open Assistant (10K) | SlimOrca (518K) |
| --- | --- | --- | --- | --- | --- |
| Huggingface | 2 T4 | 84h 47m | 163h 48m | 30h 51m | 1301h 24m |
| Unsloth Pro | 2 T4 | 3h 20m (25.4x) | 5h 43m (28.7x) | 1h 12m (25.7x) | 71h 40m (18.1x) |
| Unsloth Max | 2 T4 | 3h 4m (27.6x) | 5h 14m (31.3x) | 1h 6m (28.1x) | 54h 20m (23.9x) |

**Peak Memory Usage on a Multi GPU System (2 GPUs)**

| System | GPU | Alpaca (52K) | LAION OIG (210K) | Open Assistant (10K) | SlimOrca (518K) |
| --- | --- | --- | --- | --- | --- |
| Huggingface | 2 T4 | 8.4GB \| 6GB | 7.2GB \| 5.3GB | 14.3GB \| 6.6GB | 10.9GB \| 5.9GB |
| Unsloth Pro | 2 T4 | 7.7GB \| 4.9GB | 7.5GB \| 4.9GB | 8.5GB \| 4.9GB | 6.2GB \| 4.7GB |
| Unsloth Max | 2 T4 | 10.5GB \| 5GB | 10.6GB \| 5GB | 10.6GB \| 5GB | 10.5GB \| 5GB |

# Troubleshooting
1. Sometimes `bitsandbytes` or `xformers` does not link properly. Try running:
```
!ldconfig /usr/lib64-nvidia
```
2. Windows is not supported as of yet - we rely on Xformers and Triton support, so until both packages support Windows officially, Unsloth will then support Windows.

<img src="./images/unsloth loading page render.png" width="300" />
