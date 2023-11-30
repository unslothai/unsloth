# Unsloth
2x faster 50% less memory LLM finetuning on a single GPU.

# Installation Instructions
Unsloth currently only supports Linux* and Pytorch >= 2.1.

1. Find your CUDA version via
```
import torch; torch.version.cuda
```
2. For CUDA 11.8:
```
pip install "unsloth[cu118] @ git+https://github.com/unslothai/unsloth.git"
```
3. For CUDA 12.1:
```
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
```

To update Pytorch to 2.1:
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

Then install Unsloth.

For Google Colab and Kaggle instances:
1. Try our Colab example:
2. Try our Kaggle example:

# Future Milestones

# Troubleshooting
1. Sometimes `bitsandbytes` or `xformers` does not link properly. Try running:
```
!ldconfig /usr/lib64-nvidia
```
2. Windows is not supported as of yet - we rely on Xformers and Triton support, so until both packages support Windows officially, Unsloth will then support Windows.
