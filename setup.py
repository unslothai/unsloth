import unsloth
from setuptools import setup, find_packages
import os

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

version = unsloth.models._utils.__version__

setup(
    name="unsloth",
    version=version,
    description="2-5X faster LLM finetuning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Unsloth AI team",
    author_email="info@unsloth.ai",
    maintainer="Daniel Han, Michael Han",
    maintainer_email="danielhanchen@gmail.com, info@unsloth.ai",
    url="https://github.com/unslothai/unsloth",
    license="Apache 2.0",
    python_requires=">=3.9",
    packages=find_packages(exclude=["images*"]),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
    ],
    install_requires=[
        "unsloth_zoo>=2025.1.2",
        "packaging",
        "tyro",
        "transformers>=4.46.1,!=4.47.0",
        "datasets>=2.16.0",
        "sentencepiece>=0.2.0",
        "tqdm",
        "psutil",
        "wheel>=0.42.0",
        "numpy",
        "accelerate>=0.34.1",
        "trl>=0.7.9,!=0.9.0,!=0.9.1,!=0.9.2,!=0.9.3",
        "peft>=0.7.1,!=0.11.0",
        "protobuf<4.0.0",
        "huggingface_hub",
        "hf_transfer",
        "unsloth[triton]",
    ],
    extras_require={
        "triton": [
            "triton @ https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
            "triton @ https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
            "triton @ https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
            "triton @ https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",

        ],
        "huggingface": [
            "unsloth_zoo>=2025.1.2",
            "packaging",
            "tyro",
            "transformers>=4.46.1,!=4.47.0",
            "datasets>=2.16.0",
            "sentencepiece>=0.2.0",
            "tqdm",
            "psutil",
            "wheel>=0.42.0",
            "numpy",
            "accelerate>=0.34.1",
            "trl>=0.7.9",
            "peft>=0.7.1",
            "protobuf<4.0.0",
            "huggingface_hub",
            "hf_transfer",
            "unsloth[triton]",
        ],
        "cu118only": [
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.22.post7%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.22.post7%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.22.post7%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",

        ],
        "cu121only":[
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.22.post7-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.22.post7-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.22.post7-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
        ],
        "cu118onlytorch211":[
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",

        ],
        "cu121onlytorch211":[
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",

        ],
        "cu118onlytorch212":[
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23.post1%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23.post1%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.23.post1%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",

        ],
        "cu121onlytorch212":[
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23.post1-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23.post1-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.23.post1-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",

        ],
        "cu118onlytorch220":[
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.24%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.24%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.24%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",

        ],
        "cu121onlytorch220":[
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.24-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.24-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.24-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",

        ],
        "cu118onlytorch230":[
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27%2Bcu118-cp312-cp312-manylinux2014_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",

        ],
        "cu121onlytorch230":[
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.27-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.27-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.27-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.27-cp312-cp312-manylinux2014_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
        ],
        "cu118onlytorch240":[
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27.post2%2Bcu118-cp39-cp39-manylinux2014_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27.post2%2Bcu118-cp310-cp310-manylinux2014_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27.post2%2Bcu118-cp311-cp311-manylinux2014_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu118/xformers-0.0.27.post2%2Bcu118-cp312-cp312-manylinux2014_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
        ],
        "cu121onlytorch240":[
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post1-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post1-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post1-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post1-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",

        ],
        "cu124onlytorch240":[
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post1-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",

        ],
        "cu121onlytorch250":[
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post2-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post2-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post2-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.28.post2-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
        ],
        "cu124onlytorch250":[
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.28.post2-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",
        ],
        "cu121onlytorch251":[
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.29.post1-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.29.post1-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.29.post1-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu121/xformers-0.0.29.post1-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",

        ],
        "cu124onlytorch251":[
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp39-cp39-manylinux_2_28_x86_64.whl ; python_version=='3.9' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp310-cp310-manylinux_2_28_x86_64.whl ; python_version=='3.10' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp311-cp311-manylinux_2_28_x86_64.whl ; python_version=='3.11' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp312-cp312-manylinux_2_28_x86_64.whl ; python_version=='3.12' and platform_system == 'Linux'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp39-cp39-win_amd64.whl ; python_version=='3.9' and platform_system == 'Windows'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp310-cp310-win_amd64.whl ; python_version=='3.10' and platform_system == 'Windows'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp311-cp311-win_amd64.whl ; python_version=='3.11' and platform_system == 'Windows'",
            "xformers @ https://download.pytorch.org/whl/cu124/xformers-0.0.29.post1-cp312-cp312-win_amd64.whl ; python_version=='3.12' and platform_system == 'Windows'",
        ],
        "cu118":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu118only]",
        ],
        "cu121":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu121only]",
        ],
        "cu118-torch211":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu118onlytorch211]",
        ],
        "cu121-torch211":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu121onlytorch211]",
        ],
        "cu118-torch212":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu118onlytorch212]",
        ],
        "cu121-torch212":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu121onlytorch212]",
        ],
        "cu118-torch220":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu118onlytorch220]",
        ],
        "cu121-torch220":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu121onlytorch220]",
        ],
        "cu118-torch230":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu118onlytorch230]",
        ],
        "cu121-torch230":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu121onlytorch230]",
        ],
        "cu118-torch240":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu118onlytorch240]",
        ],
        "cu121-torch240":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu121onlytorch240]",
        ],
        "cu121-torch250":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu121onlytorch250]",
        ],
        "cu124-torch240":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu124onlytorch240]",
        ],
        "cu124-torch250":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu124onlytorch250]",
        ],
        "cu121-torch251":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu121onlytorch251]",
        ],
        "cu124-torch251":[
            "unsloth[huggingface]",
            "bitsandbytes>=0.43.3",
            "unsloth[cu124onlytorch251]",
        ],

"kaggle": [
    "unsloth[huggingface]",
],
"kaggle-new": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
],
"conda": [
    "unsloth[huggingface]",
],
"colab-torch211": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121onlytorch211]",
],
"colab-ampere-torch211": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121onlytorch211]",
    "packaging",
    "ninja",
    "flash-attn>=2.6.3",
],
"colab-torch220": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121onlytorch220]",
],
"colab-ampere-torch220": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121onlytorch220]",
    "packaging",
    "ninja",
    "flash-attn>=2.6.3",
],
"colab-new": [
    "unsloth_zoo>=2025.1.2",
    "packaging",
    "tyro",
    "transformers>=4.46.1,!=4.47.0",
    "datasets>=2.16.0",
    "sentencepiece>=0.2.0",
    "tqdm",
    "psutil",
    "wheel>=0.42.0",
    "numpy",
    "protobuf<4.0.0",
    "huggingface_hub",
    "hf_transfer",
    "bitsandbytes>=0.43.3",
    "unsloth[triton]",
],
"colab-no-deps": [
    "accelerate>=0.34.1",
    "trl>=0.7.9,!=0.9.0,!=0.9.1,!=0.9.2,!=0.9.3",
    "peft>=0.7.1",
    "xformers",
    "bitsandbytes>=0.46.1",
    "protobuf<4.0.0",
],
"colab": [
    "unsloth[cu121]",
],
"flashattention": [
    "packaging ; platform_system == 'Linux'",
    "ninja ; platform_system == 'Linux'",
    "flash-attn>=2.6.3 ; platform_system == 'Linux'",
],
"colab-ampere": [
    "unsloth[colab-ampere-torch220]",
    "unsloth[flashattention]",
],
"cu118-ampere": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu118only]",
    "unsloth[flashattention]",
],
"cu121-ampere": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121only]",
    "unsloth[flashattention]",
],
"cu118-ampere-torch211": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu118onlytorch211]",
    "unsloth[flashattention]",
],
"cu121-ampere-torch211": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121onlytorch211]",
    "unsloth[flashattention]",
],
"cu118-ampere-torch220": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu118onlytorch220]",
    "unsloth[flashattention]",
],
"cu121-ampere-torch220": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121onlytorch220]",
    "unsloth[flashattention]",
],
"cu118-ampere-torch230": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu118onlytorch230]",
    "unsloth[flashattention]",
],
"cu121-ampere-torch230": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121onlytorch230]",
    "unsloth[flashattention]",
],
"cu118-ampere-torch240": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu118onlytorch240]",
    "unsloth[flashattention]",
],
"cu121-ampere-torch240": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121onlytorch240]",
    "unsloth[flashattention]",
],
"cu121-ampere-torch250": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121onlytorch250]",
    "unsloth[flashattention]",
],
"cu124-ampere-torch240": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu124onlytorch240]",
    "unsloth[flashattention]",
],
"cu124-ampere-torch250": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu124onlytorch250]",
    "unsloth[flashattention]",
],
"cu121-ampere-torch251": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu121onlytorch251]",
    "unsloth[flashattention]",
],
"cu124-ampere-torch251": [
    "unsloth[huggingface]",
    "bitsandbytes>=0.43.3",
    "unsloth[cu124onlytorch251]",
    "unsloth[flashattention]",
],

    },
    include_package_data=False,
    keywords=["ai", "llm"],
)
