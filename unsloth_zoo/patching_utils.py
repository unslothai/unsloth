# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch

__all__ = [
    "patch_compiling_bitsandbytes",
    "patch_layernorm",
    "patch_torch_compile",
    "patch_regional_compilation",
    "patch_model_and_tokenizer",
]

# Also disable compiling on bitsandbytes
def patch_compiling_bitsandbytes():
    # import peft.tuners.lora.bnb
    # peft.tuners.lora.bnb.Linear4bit.forward = \
    #     torch._disable_dynamo(peft.tuners.lora.bnb.Linear4bit.forward)
    # peft.tuners.lora.bnb.Linear8bitLt.forward = \
    #     torch._disable_dynamo(peft.tuners.lora.bnb.Linear8bitLt.forward)
    # return
    import bitsandbytes.nn.modules
    bitsandbytes.nn.modules.Linear4bit.forward = \
        torch._disable_dynamo(bitsandbytes.nn.modules.Linear4bit.forward)
    return
pass


def patch_layernorm(fast_layernorm):
    import torch.nn
    if torch.nn.LayerNorm.__name__ != "Unsloth_LayerNorm":

        from torch.nn import LayerNorm
        class Unsloth_LayerNorm(LayerNorm):
            def forward(self, X):
                return fast_layernorm(self, X)
            pass
        pass

        torch.nn.LayerNorm = Unsloth_LayerNorm
    return
pass


def patch_torch_compile(debug = True, O3 = False):
    assert(type(debug) is bool)
    assert(type(O3)    is bool)
    import os
    if debug:
        os.environ["TORCHDYNAMO_VERBOSE"] = "1"
        os.environ["TORCH_LOGS"] = "+dynamo"
    else:
        os.environ.pop("TORCHDYNAMO_VERBOSE", None)
        os.environ.pop("TORCH_LOGS", None)
    pass

    # Torch compile arguments
    torch_compile_arguments = [
        f"config.debug = {debug}",
        "config.dce = True",
        "config.memory_planning = True",
        "config.memory_pool = 'combined'",
        "config.efficient_conv_bn_eval_fx_passes = True", # Reduces stability a little bit
        "config.dynamic_scale_rblock = True", # Scale down RBLOCK for better occupancy
        # Disable reorder_for_compute_comm_overlap since it errors for non multi GPU systems
        # "config.reorder_for_compute_comm_overlap = True", # # enable reordering pass for increasing overlap between compute and communication
        f"config.max_autotune = {O3}", # enable slow autotuning passes to select algorithms
        f"config.max_autotune_pointwise = {O3}", # enable slow autotuning passes to select pointwise/reductions algorithms
        f"config.max_autotune_gemm = {O3}", # GEMM is unnecessary
        "config.max_autotune_gemm_backends = 'TRITON,ATEN,CPP'", # Not much faster
        "config.autotune_fallback_to_aten = True", # Fallback to ATEN backend
        "config.autotune_multi_device = True", # If autotuning in subprocess, whether to use multiple devices
        "config.coordinate_descent_tuning = True",
        f"config.aggressive_fusion = {O3}", # Careful changes results!
        "config.combo_kernels = True", # Experimental - enable the combo kernel that combines data-independent kernels
        "config.combo_kernel_foreach_dynamic_shapes = True",
        "config.freezing = False", # Freezes weights --> ** only useful for inference **
        "config.triton.multi_kernel = True", # use tuning to pick between different subkernels
        "config.cuda.enable_cuda_lto = True",
        "config.cuda.use_fast_math = True",
        "config.cuda.compile_opt_level = '-O2'",
    ]
    # Torch dynamo arguments
    torch_dynamo_arguments = [
        "config.accumulated_cache_size_limit = 1024", # Bump up a bit from 256
        f"config.suppress_errors = {not debug}", # Supress errors for now
        f"config.do_not_emit_runtime_asserts = {not debug}",
        "config.cache_size_limit = 1024", # Flex Attention
        "config.inline_inbuilt_nn_modules = True", # Torch 2.5 Regional recompilation
        "config.numpy_default_float = 'float32'",
        "config.compiled_autograd = True", # New Torch 2.4 feature which can compile backwards passes
        # https://pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html
    ]
    import torch._inductor.config as config
    for _try_compile_argument in torch_compile_arguments:
        try:    exec(_try_compile_argument)
        except: pass
    pass
    import torch._dynamo.config as config
    for _try_dynamo_argument in torch_dynamo_arguments:
        try:    exec(_try_dynamo_argument)
        except: pass
    pass
pass


def patch_regional_compilation():
    # Regional torch 2.5 Recompilation - weirdly very slow??
    if torch.nn.ModuleList.__name__ == "UnslothModuleList": return
    # Only works for torch 2.5
    if Version(torch.__version__) < Version("2.5.0"): return

    old_module_list = torch.nn.ModuleList

    def UnslothModuleList(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and type(args[0]) is list:
            args = [old_module_list([torch.compile(x, dynamic = True, options = torch_compile_options, fullgraph = False) for x in args[0]])]
        return old_module_list(*args, **kwargs)
    pass
    UnslothModuleList.__doc__ = old_module_list.__doc__

    torch.nn.ModuleList = UnslothModuleList
    return
pass


def patch_model_and_tokenizer(model, tokenizer, downcast_rope = True):
    assert(type(downcast_rope) is bool)
    import gc

    # Torch.compile fails on embedding matrix??
    try: old_input_embedding  = model.get_input_embeddings ().weight
    except: return model, tokenizer

    # Maybe not all models have a lm_head?
    try: old_output_embedding = model.get_output_embeddings().weight
    except: old_output_embedding = torch.zeros(0)

    # Check for tied weights as well
    is_tied = (old_input_embedding.data_ptr() == old_output_embedding.data_ptr()) \
        or (model.config.tie_word_embeddings)

    # Check pad token's id -> we need to expand the embedding
    if len(tokenizer) > old_input_embedding.shape[0]:
        # Workaround randomnly fixes it for torch versions < 2.
        requires_grad = old_input_embedding.requires_grad
        old_input_embedding.requires_grad_(False)
        old_input_embedding.resize_(len(tokenizer), old_input_embedding.shape[1])
        old_input_embedding.requires_grad_(requires_grad)

        # Fix up all vocab sizes
        current_model = model
        while hasattr(current_model, "model") and hasattr(current_model, "config"):
            if hasattr(current_model.config, "vocab_size"):
                current_model.config.update({"vocab_size" : len(tokenizer)})
            current_model.update({"unsloth_optimized" : True})
            current_model = current_model.model
        if hasattr(current_model, "model") and hasattr(current_model, "config"):
            if hasattr(current_model.config, "vocab_size"):
                current_model.config.update({"vocab_size" : len(tokenizer)})
            current_model.update({"unsloth_optimized" : True})
        pass
    pass

    model.set_input_embeddings(
        torch.nn.Embedding.from_pretrained(
            old_input_embedding,
            padding_idx = getattr(model.config, "pad_token_id", None),
        )
    )

    # We also do this for the lm_head
    if old_output_embedding.numel() != 0:

        requires_grad = old_output_embedding.requires_grad
        lm_head = torch.nn.Linear(1, 1, bias = None)
        del lm_head.weight

        lm_head.weight = old_output_embedding if not is_tied else old_input_embedding
        lm_head.in_features  = lm_head.weight.shape[1]
        lm_head.out_features = lm_head.weight.shape[0]
        
        lm_head.weight.requires_grad_(requires_grad)
        model.set_output_embeddings(lm_head)
        if hasattr(model, "lm_head"): model.lm_head = lm_head
        
        correct_dtype = lm_head.weight.dtype
    else:
        correct_dtype = old_input_embedding.dtype
    pass

    # Must tie lm_head and embed_tokens if they are tied!
    # Otherwise error will occur on saving models ie use save_model
    if is_tied: model.tie_weights()

    # Also fix torch_dtype
    internal_model = model
    while hasattr(internal_model, "model"):
        if hasattr(internal_model, "config"):
            if   internal_model.config.torch_dtype ==  "float32":
                internal_model.config.torch_dtype = torch.float32
            elif internal_model.config.torch_dtype == "bfloat16":
                internal_model.config.torch_dtype = torch.bfloat16
            elif internal_model.config.torch_dtype ==  "float16":
                internal_model.config.torch_dtype = torch.float16
            pass
        pass
        internal_model = internal_model.model
    pass
    if hasattr(internal_model, "config"):
        if   internal_model.config.torch_dtype ==  "float32":
            internal_model.config.torch_dtype = torch.float32
        elif internal_model.config.torch_dtype == "bfloat16":
            internal_model.config.torch_dtype = torch.bfloat16
        elif internal_model.config.torch_dtype ==  "float16":
            internal_model.config.torch_dtype = torch.float16
        pass
    pass
    
    # Also patch all dtypes - BnB seems to not allocate the correct type?
    # BnB default dtype seems to be float16!
    try:
        from bitsandbytes.nn  import Linear4bit as Bnb_Linear4bit
    except:
        raise ImportError("Unsloth: Please install bitsandbytes via `pip install bitsandbytes`")
    try:
        from peft.tuners.lora import Linear4bit as Peft_Linear4bit
    except:
        raise ImportError("Unsloth: Please install peft via `pip install peft`")
    pass

    for name, module in model.named_modules():
        if isinstance(module, (Bnb_Linear4bit, Peft_Linear4bit)):
            weight = module.weight
            quant_state = weight.quant_state

            if type(quant_state) is list:
                # BnB seems to have float16 as default!
                module.weight.quant_state[2] = correct_dtype # Cast to correct dtype
            else:
                # https://github.com/TimDettmers/bitsandbytes/pull/763/files
                quant_state.dtype = correct_dtype
            pass
        pass
        # Downcast RoPE embedding to correct data type
        if downcast_rope and ((name.endswith("rotary_emb") or hasattr(module, "cos_cached"))):

            if hasattr(module, "cos_cached") and \
                (module.cos_cached.dtype != correct_dtype):

                module.cos_cached = module.cos_cached.to(correct_dtype)
                module.sin_cached = module.sin_cached.to(correct_dtype)

            elif hasattr(module, "short_cos_cached") and \
                (module.short_cos_cached.dtype != correct_dtype):
                
                module.short_cos_cached = module.short_cos_cached.to(correct_dtype)
                module.short_sin_cached = module.short_sin_cached.to(correct_dtype)
            pass
        pass
    pass

    # Clear deleted GPU items
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    return model, tokenizer
pass
