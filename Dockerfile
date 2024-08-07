
ARG CUDA_VERSION=12.1.1
FROM nvcr.io/nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select Los_Angeles' | debconf-set-selections \
    && apt-get update -y \
    && apt-get install -y software-properties-common python3 python3-pip  python-is-python3 git curl sudo
RUN pip install --upgrade pip
RUN pip install --upgrade torch triton packaging
RUN pip install --upgrade --force-reinstall --no-cache-dir \
    ninja einops flash-attn xformers trl peft accelerate bitsandbytes \
    "unsloth[cu121-torch230] @ git+https://github.com/unslothai/unsloth.git"
