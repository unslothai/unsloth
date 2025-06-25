# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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
import os
import re
import ast

__all__ = [
    "patch_compiling_bitsandbytes",
    "patch_layernorm",
    "patch_torch_compile",
    "patch_model_and_tokenizer",
    "patch_compiled_autograd",
]

from .compiler import UNSLOTH_COMPILE_LOCATION
from .utils import _get_dtype, Version

# Also disable compiling on bitsandbytes
def patch_compiling_bitsandbytes():
    # All Unsloth Zoo code licensed under LGPLv3
    os.environ["UNSLOTH_PATCHED"] = "1"

    import bitsandbytes
    if Version(bitsandbytes.__version__) >= Version("0.46.0"):
        if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
            print("Unsloth: Bitsandbytes >= 0.46.0 supports torch.compile - enabling.")
    else:
        # Disable dynamo on Linear4bit, Linear8bit and other future modules
        if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
            print("Unsloth: Bitsandbytes < 0.46.0 does not support torch.compile - disabling.")
        for x in ["bitsandbytes.nn.modules", "peft.tuners.lora.bnb",]:
            exec(f"import {x}", globals(), locals())
            layers = dir(eval(x))
            for fx in layers:
                try: layer = eval(f"{x}.{fx}")
                except: continue
                if not hasattr(layer, "forward"): continue
                if hasattr(eval(f"{x}.{fx}.forward"), "__wrapped__"): continue
                exec(f"{x}.{fx}.forward = torch._disable_dynamo({x}.{fx}.forward)", globals(), locals())
            pass
        pass
    pass

    # import bitsandbytes.autograd._functions
    # bitsandbytes.autograd._functions.matmul_4bit = torch._disable_dynamo(
    #     bitsandbytes.autograd._functions.matmul_4bit
    # )
    return
pass


def patch_layernorm(fast_layernorm):
    # All Unsloth Zoo code licensed under LGPLv3
    import torch.nn
    if torch.nn.LayerNorm.__name__ != "Unsloth_LayerNorm":
        os.environ["UNSLOTH_PATCHED"] = "1"

        from torch.nn import LayerNorm
        class Unsloth_LayerNorm(LayerNorm):
            def forward(self, X):
                return fast_layernorm(self, X)
            pass
        pass

        torch.nn.LayerNorm = Unsloth_LayerNorm
    return
pass


def patch_torch_compile(debug = False, O3 = False, ignore_errors = True):
    # All Unsloth Zoo code licensed under LGPLv3
    assert(type(debug) is bool)
    assert(type(O3)    is bool)
    import os, logging

    if debug:
        DEBUGGING = " with debugging"
        os.environ["TORCHDYNAMO_VERBOSE"] = "1"
        os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
        # os.environ["TORCH_LOGS"] = "dynamo,graph_breaks,recompiles,graph_code,aot_joint_graph,aot_graphs,compiled_autograd_verbose"
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"
        torch._logging.set_logs(
            dynamo = logging.WARN,
            inductor = logging.WARN,
            graph_breaks = True,
            recompiles = True,
            recompiles_verbose = True,
            compiled_autograd_verbose = False, # Produces too much code
            aot_joint_graph = False, # Produces too much code
            aot_graphs = False,  # Produces too much code
        )
        torch._dynamo.config.verbose = True
    else:
        DEBUGGING = ""
        os.environ.pop("TORCHDYNAMO_VERBOSE", None)
        os.environ.pop("TORCHINDUCTOR_COMPILE_THREADS", None)
        os.environ.pop("TORCHINDUCTOR_FORCE_DISABLE_CACHES", None)
        os.environ.pop("TORCH_LOGS", None)
        torch._logging.set_logs(all = logging.CRITICAL)
        torch._dynamo.config.verbose = False
    pass
    try:
        print(f"🦥 Unsloth Zoo will now patch everything{DEBUGGING} to make training faster!")
    except:
        print(f"Unsloth Zoo will now patch everything{DEBUGGING} to make training faster!")
    pass

    os.environ["UNSLOTH_PATCHED"] = "1"
    # See https://pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
    # Caches kernel generations for faster restarts
    # https://dev-discuss.pytorch.org/t/impact-of-multithreading-and-local-caching-on-torch-compile/2498/3
    os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    os.environ["TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE"] = "1"
    os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)

    # Duplicate functions will cause hashing issues
    # os.environ["TORCHINDUCTOR_CACHE_DIR"] = UNSLOTH_COMPILE_LOCATION

    # https://github.com/sayakpaul/diffusers-torchao?tab=readme-ov-file#things-to-keep-in-mind-when-benchmarking
    os.environ["ENABLE_AOT_AUTOGRAD_CACHE"] = "1"

    # Torch compile arguments
    torch_compile_arguments = [
        f"config.debug = {debug}",
        "config.dce = True",
        "config.memory_planning = True",
        # Using 'combined' memory pool will cause re-compiles for dynamic shapres. We just re-use already allocated memory pools
        "config.memory_pool = 'none'",
        "config.efficient_conv_bn_eval_fx_passes = True", # Reduces stability a little bit
        "config.dynamic_scale_rblock = True", # Scale down RBLOCK for better occupancy
        # Disable reorder_for_compute_comm_overlap since it errors for non multi GPU systems
        # "config.reorder_for_compute_comm_overlap = True", # # enable reordering pass for increasing overlap between compute and communication
        f"config.max_autotune = {O3}", # enable slow autotuning passes to select algorithms
        f"config.max_autotune_pointwise = {O3}", # enable slow autotuning passes to select pointwise/reductions algorithms
        f"config.max_autotune_gemm = False", # GEMM is unnecessary
        "config.max_autotune_gemm_backends = 'ATEN,TRITON,CPP'", # Not much faster
        "config.autotune_fallback_to_aten = True", # Fallback to ATEN backend
        "config.autotune_multi_device = True", # If autotuning in subprocess, whether to use multiple devices
        f"config.coordinate_descent_tuning = {O3}",
        f"config.aggressive_fusion = {O3}", # Careful changes results!
        # [TODO] COMBO KERNELS makes everything slower!
        # "config.combo_kernels = True", # Experimental - enable the combo kernel that combines data-independent kernels
        # "config.combo_kernel_foreach_dynamic_shapes = True",
        "config.freezing = False", # Freezes weights --> ** only useful for inference **
        # f"config.triton.multi_kernel = {O3}", # use tuning to pick between different subkernels
        "config.cuda.enable_cuda_lto = True",
        "config.cuda.use_fast_math = True",
        f"config.cuda.compile_opt_level = {'-O2' if O3 else '-O1'}",
        # Capture torch.arange(...), torch.zeros(...)
        "config.capture_dynamic_output_shape_ops = True",
    ]
    # Torch dynamo arguments
    torch_dynamo_arguments = [
        "config.accumulated_cache_size_limit = 1024", # Bump up a bit from 256
        f"config.suppress_errors = {not debug and ignore_errors}", # Supress errors for now
        f"config.do_not_emit_runtime_asserts = {not debug}",
        "config.cache_size_limit = 1024", # Flex Attention
        "config.inline_inbuilt_nn_modules = True", # Torch 2.5 Regional recompilation
        "config.numpy_default_float = 'float32'",
        # FAILS for Gemma!
        "config.compiled_autograd = False", # New Torch 2.4 feature which can compile backwards passes
        # https://pytorch.org/tutorials/intermediate/compiled_autograd_tutorial.html
    ]
    if not debug and ignore_errors:
        # Have to explicitly set it!
        torch._dynamo.config.suppress_errors = True
    pass
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


def patch_model_and_tokenizer(
    model,
    tokenizer,
    downcast_rope = True,
    fix_embeddings = True,
    do_forced_float32 = False,
    correct_dtype = None,
):
    # All Unsloth Zoo code licensed under LGPLv3
    assert(type(downcast_rope) is bool)
    import gc

    # Fix torch_dtype
    m = model
    while hasattr(m, "model"):
        if hasattr(m, "config"):
            if   m.config.torch_dtype ==  "float32": m.config.torch_dtype = torch.float32
            elif m.config.torch_dtype == "bfloat16": m.config.torch_dtype = torch.bfloat16
            elif m.config.torch_dtype ==  "float16": m.config.torch_dtype = torch.float16
        pass
        m = m.model
    pass
    if hasattr(m, "config"):
        if   m.config.torch_dtype ==  "float32": m.config.torch_dtype = torch.float32
        elif m.config.torch_dtype == "bfloat16": m.config.torch_dtype = torch.bfloat16
        elif m.config.torch_dtype ==  "float16": m.config.torch_dtype = torch.float16
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

    # Get most likely the correct data-type of the model
    if correct_dtype is None:
        try:
            correct_dtype = _get_dtype(model.config.torch_dtype)
        except:
            correct_dtype = model.get_input_embeddings().weight.dtype
    pass
    # If we force float32, we first use bfloat16, then downcast to float16
    if do_forced_float32:
      correct_dtype = torch.float16
      for name, module in model.named_modules():
          if "down_proj" in name or "up_proj" in name or "gate_proj" in name or "fc1" in name or "fc2" in name:
              module.to(torch.float16)
          if "q_proj" in name or "k_proj" in name or "v_proj" in name or "o_proj" in name or "out_proj" in name:
              module.to(torch.float16)
          if "lm_head" in name or "embed_tokens" in name:
              module.to(torch.float16)
          if "embed_tokens" in name or "patch_embedding" in name:
              module.to(torch.float16)
          if "norm" in name:
              module.to(torch.float16)
          torch.cuda.empty_cache()

      # Convert any remaining bfloat16 parameters
      for name, param in model.named_parameters():
          if param.dtype == torch.bfloat16:
              param.data = param.data.to(torch.float16)

      # Also convert buffers (like position embeddings)
      for name, buffer in model.named_buffers():
          if buffer.dtype == torch.bfloat16:
              buffer.data = buffer.data.to(torch.float16)
      pass
    pass

    # Correct torch_dtype
    def __fix_dtype(config):
        if not hasattr(config, "to_dict"): return
        dicts = config.to_dict()
        for key, value in dicts.items():
            if key == "torch_dtype":
                setattr(config, "torch_dtype", correct_dtype)
            else:
                __fix_dtype(getattr(config, key))
    m = model
    while hasattr(m, "model"):
        if hasattr(m, "dtype"):
            try: setattr(m, "dtype", correct_dtype)
            except: pass
        if hasattr(m, "config"): __fix_dtype(m.config)
        m = m.model
    pass
    if hasattr(m, "config"): __fix_dtype(m.config)
    if hasattr(m, "dtype"):
        try: setattr(m, "dtype", correct_dtype)
        except: pass
    pass

    # Check all params and patch!
    for name, module in model.named_modules():
        if isinstance(module, (Bnb_Linear4bit, Peft_Linear4bit)):
            weight = module.weight
            # Check if quant_state exists for vision models like unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit, unsloth/granite-vision-3.2-2b
            if not hasattr(weight, 'quant_state'):
                print(f"Skipping {name}: no quant_state found")
                continue

            quant_state = weight.quant_state

            if type(quant_state) is list:
                # BnB seems to have float16 as default!
                module.weight.quant_state[2] = correct_dtype # Cast to correct dtype
            else:
                # https://github.com/TimDettmers/bitsandbytes/pull/763/files
                quant_state.dtype = correct_dtype
            pass

            if hasattr(module, "compute_dtype"):
                module.compute_dtype = correct_dtype
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

    if not fix_embeddings: return model, tokenizer

    # Torch.compile fails on embedding matrix??
    try: old_input_embedding = model.get_input_embeddings ().weight
    except: return model, tokenizer

    # Maybe not all models have a lm_head?
    try: old_output_embedding = model.get_output_embeddings().weight
    except: old_output_embedding = torch.zeros(0)

    # Check for tied weights as well
    is_tied = (old_input_embedding.data_ptr() == old_output_embedding.data_ptr()) \
        or (model.config.tie_word_embeddings)

    # Check pad token's id -> we need to expand the embedding
    if tokenizer is not None and len(tokenizer) > old_input_embedding.shape[0]:
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
            current_model.config.update({"unsloth_optimized" : True})
            current_model = current_model.model
        if hasattr(current_model, "model") and hasattr(current_model, "config"):
            if hasattr(current_model.config, "vocab_size"):
                current_model.config.update({"vocab_size" : len(tokenizer)})
            current_model.config.update({"unsloth_optimized" : True})
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
    pass

    # Must tie lm_head and embed_tokens if they are tied!
    # Otherwise error will occur on saving models ie use save_model
    if is_tied: model.tie_weights()

    # Clear deleted GPU items
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()
    return model, tokenizer
pass


def patch_compiled_autograd():
    # Fixes double compilation of functions during gradient checkpointing
    # See https://github.com/pytorch/pytorch/issues/135298
    # All Unsloth Zoo code licensed under LGPLv3
    import inspect, re

    # From https://github.com/pytorch/pytorch/pull/135795/files
    import torch._dynamo.compiled_autograd
    fx = torch._dynamo.compiled_autograd.AutogradCompilerInstance.end_capture
    if fx.__name__ == "unsloth_end_capture": return
    source = inspect.getsource(fx)
    if "with disable()" in source: return
    spaces = source.find("def")
    source = source.split("\n")
    source = "\n".join(x[spaces:] for x in source)
    old = "return compiled_fn(inputs, sizes, scalars, hooks)"
    match = re.search(r"\n([ ]{1,})return compiled_fn", source)
    n = len(match.group(1)) if match else 0
    source = source.replace(old, f"with disable():\n{' '*(n + 4)}{old}")
    source = source.replace("def end_capture", "def unsloth_end_capture", 1)

    # Import items to make the function executable
    all_items = dir(torch._dynamo.compiled_autograd)
    good_items = [x for x in all_items if x in source]
    exec("from torch._dynamo.compiled_autograd import (" + ", ".join(x for x in good_items) + ")", globals())
    exec(source, globals())
    torch._dynamo.compiled_autograd.AutogradCompilerInstance.end_capture = unsloth_end_capture

    # From https://github.com/pytorch/pytorch/pull/135795/files
    try:
        import torch._dynamo.variables.misc
        fx = torch._dynamo.variables.misc.AutogradEngineVariable.call_method
    except:
        return
    if fx.__name__ == "unsloth_call_method": return
    source = inspect.getsource(fx)
    if "in_compiled_autograd_region" in source: return
    spaces = source.find("def")
    source = source.split("\n")
    source = "\n".join(x[spaces:] for x in source)
    source = source.replace(
        "torch._dynamo.compiled_autograd.compiled_autograd_enabled",
        "torch._dynamo.compiled_autograd.in_compiled_autograd_region",
        1,
    )
    source = source.replace("def call_method", "def unsloth_call_method", 1)

    # Import items to make the function executable
    all_items = dir(torch._dynamo.variables.misc)
    good_items = [x for x in all_items if x in source]
    exec("from torch._dynamo.variables.misc import (" + ", ".join(x for x in good_items) + ")", globals())
    exec(source, globals())
    torch._dynamo.variables.misc.AutogradEngineVariable.call_method = unsloth_call_method
    return
pass


# utility function to help BC with old module hierarchy for transformers >=4.52.0
def check_conversion_mappings(model, current_key_name_str, skip_modules):
    # model_root_cls is None if there are no conversion_mappings or no _root_cls
    model_root_cls = getattr(model, "_root_cls", model if hasattr(model, "_checkpoint_conversion_mapping") else None)
    if model_root_cls is None:
        return False
    if hasattr(model_root_cls, "_checkpoint_conversion_mapping") and len(model_root_cls._checkpoint_conversion_mapping) > 0:
        # if this is true, then it means that we must be on transformers >=4.52.0 because conversion_mappings was added in 4.52.0
        # we cant know if the skip module naming convention is new or old
        # but if we are supposed to skip this current_key_name_str, and it didn't pass
        # (current_key_name_str in quantization_config.llm_int8_skip_modules)
        # then new transformers + new module hierarchy means it should not be skipped, ie no BC check needed
        # and new transformers + old module hierarchy means we still need to check to skip
        # old transformers + old module hierarchy means no BC needed
        # old transformers + new module hierarchy is problematic since we don't have the conversion_mappings to reverse
        # follow the logic from save_pretrained in transformers.modeling_utils
        reverse_conversion_mappings = {v: k for k, v in model_root_cls._checkpoint_conversion_mapping.items()}
        new_current_key_names_str = current_key_name_str
        for pattern, replacement in reverse_conversion_mappings.items():
            try:
                replacement = replacement.lstrip("^")  # strip off un-needed chars and patterns
                replacement = re.sub(r"\(.*?\)", "", replacement)
                key, n_replace = re.subn(pattern, replacement, current_key_name_str)
                # Early exit of the loop
                if n_replace > 0:
                    new_current_key_names_str = key
                    break
            except Exception as e:
                # skip this pattern but log
                do_logging = os.environ.get('UNSLOTH_ENABLE_LOGGING', '0') == '1'
                if do_logging:
                    print(f"Unsloth: Replace bnb issue: {str(e)}")
                break
        return any([(skip_key + "." in new_current_key_names_str) or (skip_key == new_current_key_names_str) for skip_key in skip_modules])
    return False


def _mark_parent(child, parent_type):
    """Attach the parent’s class so the child can inspect it later."""
    child._root_cls = parent_type


def _unmark_parent(child):
    """Remove the temporary attribute if it is present."""
    if hasattr(child, "_root_cls"):
        delattr(child, "_root_cls")


def parsed_statement(code: str) -> ast.stmt:
    """Return the statement parsed from a one-liner."""
    return ast.parse(code).body[0]


class WrapRecursiveCall(ast.NodeTransformer):
    function_name = "_replace_with_bnb_linear"
    mark_statement = parsed_statement(
        '_mark_parent(module, model._root_cls '
        'if hasattr(model, "_root_cls") else type(model))'
    )
    unmark_statement = parsed_statement('_unmark_parent(module)')

    def visit_Assign(self, node: ast.Assign):
        """
        Replace
            _, has_been_replaced = _replace_with_bnb_linear(...)
        with
            try:
                _mark_parent(module,
                             model._root_cls
                             if hasattr(model, "_root_cls")
                             else type(model))
                _, has_been_replaced = _replace_with_bnb_linear(...)
            finally:
                _unmark_parent(module)
        """
        if (
            isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == self.function_name
        ):
            wrapped = ast.Try(
                body      =[self.mark_statement, node],
                handlers  =[],
                orelse    =[],
                finalbody =[self.unmark_statement],
            )
            return ast.copy_location(wrapped, node)
        return node


# Patch for dynamic 4bit quantization
import inspect
import transformers.integrations.bitsandbytes
if hasattr(transformers.integrations.bitsandbytes, "_replace_with_bnb_linear") and \
    (transformers.integrations.bitsandbytes._replace_with_bnb_linear.__name__ != "_unsloth_replace_with_bnb_linear"):

    # All Unsloth Zoo code licensed under LGPLv3
    source = inspect.getsource(transformers.integrations.bitsandbytes._replace_with_bnb_linear)
    functions = dir(transformers.integrations.bitsandbytes)
    functions = [x for x in functions if f" {x}" in source or f"{x}." in source or f"{x}(" in source]
    functions = [x for x in functions if x != "_replace_with_bnb_linear"]
    x = ", ".join(functions)
    exec(f"from transformers.integrations.bitsandbytes import ({x})", globals())
    if "current_key_name_str" not in source:
        raise RuntimeError("Unsloth: Patch for dynamic quantization failed since current_key_name_str does not exist.")

    # First patch recursive calls to mark the parent class
    # we need it to access the parent class to check for conversion_mappings
    try:
        mark_parent_error = False
        new_source = source.replace(
            "name in quantization_config.llm_int8_skip_modules\n",
            "((name in quantization_config.llm_int8_skip_modules) or (current_key_name_str in quantization_config.llm_int8_skip_modules) or (check_conversion_mappings(model, current_key_name_str, quantization_config.llm_int8_skip_modules)))\n",
            1,
        )

        source_tree = ast.parse(new_source)
        source_tree = WrapRecursiveCall().visit(source_tree)
        ast.fix_missing_locations(source_tree)
        new_source = ast.unparse(source_tree)

        # will raise error if patch fails
        compile(new_source, '<temp_patched>', 'exec')
        if '_mark_parent' not in new_source and '_unmark_parent' not in new_source:
            do_logging = os.environ.get('UNSLOTH_ENABLE_LOGGING', '0') == '1'
            if do_logging:
                print(f"Unsloth: Could not wrap replace_with_bnb_linear but may not be an issue")
            mark_parent_error = True
        else:
            source = new_source

    except Exception as e:
        do_logging = os.environ.get('UNSLOTH_ENABLE_LOGGING', '0') == '1'
        if do_logging:
            print(f"Unsloth: Could not wrap replace_with_bnb_linear but may not be an issue. {str(e)}")
        mark_parent_error = True

    if mark_parent_error:
        # we sitll have the original source without the mark_parent and unmark_parent patches
        source = source.replace(
            "name in quantization_config.llm_int8_skip_modules\n",
            "((name in quantization_config.llm_int8_skip_modules) or (current_key_name_str in quantization_config.llm_int8_skip_modules))\n",
            1,
        )

    source = source.replace(
        "_replace_with_bnb_linear",
        "_unsloth_replace_with_bnb_linear",
    )

    score_code = """if name == 'score':
    modules_to_not_convert.append("score")"""

    pattern = r"(^\s*)(current_key_name\.append\(name\))"

    def add_score_code(match):
        indentation = match.group(1)  # Captured indentation
        line_content = match.group(2) # The line 'current_key_name.append(name)'

        indented_breakpoint_code = "\n".join([f"{indentation}{line}" for line in score_code.splitlines()])

        return f"{indentation}{line_content}\n{indented_breakpoint_code}"

    source = re.sub(pattern, add_score_code, source, flags=re.MULTILINE)

    exec(source, globals())
    transformers.integrations.bitsandbytes._replace_with_bnb_linear = _unsloth_replace_with_bnb_linear
pass

# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
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
