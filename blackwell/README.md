## Unsloth Blackwell Compatibility

### Overview

`Blackwell` (`sm100+`) requires all dependent libraries to be compiled with `cuda 12.8`.  

The core libs for running unsloth which have dependencies on `CUDA` version are:
- `bitsandbytes` - already has wheels built with `CUDA 12.8` so `pip install` should work out of the box
- `triton` - requires `triton>=3.3.1`
- `torch` - requires installing with `pip install torch --extra-index-url https://download.pytorch.org/whl/cu128` 
- `vllm` - safest is to use the nightly build: `uv pip install -U vllm --torch-backend=cu128 --extra-index-url https://wheels.vllm.ai/nightly`
- `xformers` - as of 6/26, `xformers` wheels are not yet built with `sm100+` enabled as support was only recently [added](https://github.com/facebookresearch/xformers/commit/d9b3b6e2b38ca485c89507ef8ac1fbef2723cdfa) so will require a source build (see below).
  
### Installation

The installation order is important, since we want the overwrite bundled dependencies with specific versions (namely, `xformers` and `triton`).

1) I prefer to use `uv` over `pip` as it's faster and better for resolving dependencies, especially for libraries which depend on `torch` but for which a specific `CUDA` version is required per this scenario.  
    
    Install `uv`
    
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
    ```
    
    Create a project dir and venv:
    
    ```bash
    mkdir `unsloth-blackwell` && cd `unsloth-blackwell`
    uv venv .venv --python=3.12 --seed
    source .venv/bin/activate
    ```

2) Install `vllm`
    
    ```bash
    uv pip install -U vllm --torch-backend=cu128 --extra-index-url https://wheels.vllm.ai/nightly
    ```

    Note that we have to specify `cu128`, otherwise `vllm` will install `torch==2.7.0` but with `cu126`.

3) Install `unsloth` dependencies
    
    ```bash
    uv pip install unsloth unsloth_zoo bitsandbytes
    ```

4) Download and build `xformers`
    
    ```bash
    # First uninstall xformers installed by previous libraries
    uv pip uninstall xformers
    
    # Clone and build
    git clone --depth=1 https://github.com/facebookresearch/xformers --recursive
    cd xformers
    export TORCH_CUDA_ARCH_LIST="12.0"
    python setup.py install
    ```
    
    Note that we have to explicitly set `TORCH_CUDA_ARCH_LIST=12.0`.

5) Update `triton`
    
    ```bash
    uv pip install -U triton>=3.3.1
    ```
    
    `triton>=3.3.1` is required for `Blackwell` support.

6) `transformers`  
    `transformers >= 4.53.0` breaks `unsloth` inference.  Specifically, `transformers` with `gradient_checkpointing` enabled will automatically [switch off caching](https://github.com/huggingface/transformers/blob/67ddc82fbc7e52c6f42a395b4a6d278c55b77a39/src/transformers/modeling_layers.py#L52-L59).  
     
    When using `unsloth` `FastLanguageModel` to `generate` directly after training with `use_cache=True`, this will result in mismatch between expected and actual outputs [here](https://github.com/unslothai/unsloth/blob/bfa6a3678e2fb8097c5ece41d095a8051f099db3/unsloth/models/llama.py#L939).
    
    Temporary solution is to switch off `gradient_checkpointing` (e.g., `model.disable_gradient_checkpointing()`) before generation if using `4.53.0` or stick with `4.52.4` for now:
    
    ```bash
    uv pip install -U transformers==4.52.4
    ```

After installation, your environment should look similar to `blackwell.requirements.txt`.

Note, might need to downgrade `numpy<=2.2` after all the installs.

### Test
Both `test_llama32_sft.py` and `test_qwen3_grpo.py` should run without issue if correct install. If not, check diff between your installed env and `blackwell.requirements.txt`.
