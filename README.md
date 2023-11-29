# Unsloth
2x faster 50% less memory LLM finetuning on a single GPU.

`!pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"`
`!pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"`


### Google Colab examples
1. [Unsloth fast finetuning example](https://colab.research.google.com/drive/1oW55fBmwzCOrBVX66RcpptL3a99qWBxb?usp=sharing)
2. [Original slow finetuning example](https://colab.research.google.com/drive/1c7zxdLHaLJ9R9YTZ74y4tUERvS-kySyA?usp=sharing)

### Installation instructions
In Google Colab:
```
!ldconfig /usr/lib64-nvidia
!pip install xformers --index-url https://download.pytorch.org/whl/cu118
!pip install git+https://github.com/danielhanchen/unsloth.git
```
`!ldconfig /usr/lib64-nvidia` is necessary (for now) to link CUDA with Python. Possibly a Google Colab linking bug.

For general installations:
1. Install Xformers *OR* Flash Attention. Choose 1. Old GPUs use Xformers. New use Flash Attention.
2. For Xformers, find your Pytorch CUDA version via `torch.version.cuda` or `nvidia-smi`.
   * If you have Conda, `conda install xformers -c xformers`
   * If you have CUDA 11.8, `pip install xformers --index-url https://download.pytorch.org/whl/cu118`
   * If you have CUDA 12.1, `pip install xformers --index-url https://download.pytorch.org/whl/cu121`
   * Go to https://github.com/facebookresearch/xformers for other issues.
   * You must have Pytorch 2.1 installed for Xformers. If not, try Flash Attention.
   * Xformers supports all GPUs (Tesla T4 etc).
3. For Flash Attention, you must have a Ampere, Ada, Hopper GPU (A100, RTX 3090, RTX 4090, H100).
   * Install Flash Attention via `pip uninstall -y ninja && pip install ninja` then `pip install flash-attn --no-build-isolation`.
   * Xformers has native support for Flash Attention, so technically installing Xformers is enough.
4. Then install Unsloth:
   `pip install git+https://github.com/danielhanchen/unsloth.git`
