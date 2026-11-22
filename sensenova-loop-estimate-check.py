"""CPU-only reproduction of the transformers 4-bit placement estimate for this checkpoint.

Usage: python sensenova-loop-estimate-check.py <converted-dir> [free_gib]

Hijacks transformers' _get_device_map (the same call modeling_utils:4319 makes, after the
bnb module swap, before any weight is read, no dispatch, no CUDA context) and prints the
module size map: total estimate, five largest leaf modules, bytes-per-element buckets and
the 90%-derated budget for the given free figure. Pure diagnostic; exits before loading.
"""

import sys

from unsloth.models.sensenova_loop import prepare_sensenova_loop_int8_load
import transformers.modeling_utils as modeling_utils
from transformers.integrations.accelerate import compute_module_sizes, get_max_memory


class _Stop(Exception):
    pass


def main():
    checkpoint = sys.argv[1]
    free_gib = float(sys.argv[2]) if len(sys.argv) > 2 else 6.76

    report = {}

    def shim(model, device_map, max_memory, hf_quantizer):
        module_sizes, leaves = compute_module_sizes(model, hf_quantizer)
        report["total_estimate_gib"] = round(module_sizes.get("", 0) / 1024**3, 3)
        report["top_leaf_modules"] = [
            [name, round(size / 1024**3, 3)]
            for name, size in sorted(leaves.items(), key = lambda kv: kv[1], reverse = True)[:5]
        ]
        buckets = {}
        for name, param in model.named_parameters():
            size = (
                hf_quantizer.param_element_size(model, name, param)
                if hf_quantizer is not None
                else param.element_size()
            )
            buckets[size] = buckets.get(size, 0) + param.numel() * size
        report["bytes_per_element_gib"] = {
            str(key): round(value / 1024**3, 3) for key, value in sorted(buckets.items())
        }
        if hf_quantizer is not None:
            budget = hf_quantizer.adjust_max_memory(
                get_max_memory(max_memory or {0: int(free_gib * 1024**3)})
            )
            report["budget_gib"] = {
                str(key): round(value / 1024**3, 3)
                for key, value in budget.items()
                if isinstance(key, int)
            }
        raise _Stop()

    modeling_utils._get_device_map = shim

    import torch
    from transformers import AutoConfig, AutoModelForImageTextToText, BitsAndBytesConfig

    from unsloth.models.vision import _architecture_skip_modules
    from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES

    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code = True)
    prepare_sensenova_loop_int8_load(checkpoint, config)
    # Mirror the loader's bnb config for this model: the skip list minus lm_head,
    # which the loader quantizes (see loader.py's _unsloth_sensenova_loop_quantize_lm_head).
    skip_modules = SKIP_QUANTIZATION_MODULES.copy() + _architecture_skip_modules(
        ["qwen3_5", "sensenova_loop"]
    )
    skip_modules = [module for module in skip_modules if module != "lm_head"]
    bnb = BitsAndBytesConfig(
        load_in_4bit = True,
        bnb_4bit_use_double_quant = True,
        bnb_4bit_quant_type = "nf4",
        bnb_4bit_compute_dtype = torch.bfloat16,
        llm_int8_skip_modules = skip_modules,
    )
    try:
        AutoModelForImageTextToText.from_pretrained(
            checkpoint,
            config = config,
            trust_remote_code = True,
            device_map = "sequential",
            quantization_config = bnb,
            dtype = torch.bfloat16,
        )
    except _Stop:
        pass
    print("estimate:", report)


if __name__ == "__main__":
    main()
