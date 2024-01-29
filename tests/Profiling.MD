## Profiling

Script for running simple profiling of `unsloth` models and default `HuggingFace` peft training.

Operates in 2 modes:

- Default: Runs `SFTTrainer` for small number of training steps and prints resulting metrics. Good sanity check of training time and loss.

- Profiling: Consists of running a sample batch of data through a given model instrumented with `torch.profiler` and also `cudaProfilerAPI` for `nsys` analysis.

  - Every forward / backward pass of every module of the model are appropriately marked (using `torch.profiler.record_function` or `torch.cuda.nvtx`), which helps when viewing in `torch.profiler` and `nsys` traces.

### Usage

Below command will run the Huggingface `SFTTrainer` for 20 steps and print the resulting training metrics to `stdout`

```
python test_unsloth.py --model_name llama --model_type unsloth-bnb --dtype float16 --dataset guanaco --output_dir ./results

```

Alternatively, the following will run the profiler using `torch.profiler`:

```
python test_unsloth.py --model_name llama --model_type unsloth-bnb --dtype float16 --profile --output_dir ./results
```

Submitting the above command to `nsys` will also work.

### Script args

#### Model names:

- llama

  - maps to either of the following based on the model type (see below):
    - "TheBloke/Llama-2-7B-GPTQ",
    - "unsloth/llama-2-7b-bnb-4bit",

- mistral
  - maps to either of the following based on the model type (see below):
    - "TheBloke/Mistral-7B-v0.1-GPTQ"
    - "unsloth/mistral-7b-bnb-4bit"

#### Model types

- unsloth-gptq-triton
  - Fast LoRA implementation that uses `auto_gptq` `triton` quantized matmul kernels fused with LoRA adapters for custom autograd
- hf-gptq-default
  - Default `HuggingFace` GPTQ peft implementation. Note that HF uses auto_gptq `cuda` quant linear layers under the hood.  
    However, this `auto_gptq` layer automatically disables the cuda kernel when the layer is trainable and falls back to a pure torch implementation (see https://github.com/AutoGPTQ/AutoGPTQ/blob/d2662b18bb91e1864b29e4e05862712382b8a076/auto_gptq/nn_modules/qlinear/qlinear_cuda.py#L40-L41)
- hf-gptq-triton-patch

  - This is a patch of `HuggingFace` GPTQ peft model that replaces the default `cuda` `qlinear` layers with `auto_gptq` `triton` `qlinear` layers.

- unsloth-bnb
  - Fast LoRA implementation with fused bitsandbytes quantization and LoRA adapters for custom autograd.

#### Additional

For additional options:

```python
python test_unsloth.py --help
```

**NOTE**: If profiling huggingface default models, need to change the following lines for backward profiling hooks to work
see [here](https://github.com/huggingface/peft/blob/bfc102c0c095dc9094cdd3523b729583bfad4688/src/peft/tuners/lora/gptq.py#L70)

_Original_

```python
    result += output
    return result
```

_New_

```python
    return result + output
```

Also need to patch the following [lines](https://github.com/huggingface/peft/blob/bfc102c0c095dc9094cdd3523b729583bfad4688/src/peft/tuners/lora/layer.py#L320) to

```python
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    previous_dtype = x.dtype

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            # result += lora_B(lora_A(dropout(x))) * scaling
            out = result + lora_B(lora_A(dropout(x))) * scaling
            return out.to(previous_dtype)
    result = result.to(previous_dtype)
    return result
```
