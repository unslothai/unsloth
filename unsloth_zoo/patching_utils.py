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
    # Torch compile arguments
    torch_compile_arguments = [
        "config.dce = True",
        "config.memory_planning = True",
        "config.memory_pool = 'combined'",
        "config.coordinate_descent_tuning = True",
        f"config.max_autotune_gemm = {O3}", # GEMM is unnecessary
        "config.autotune_multi_device = False",
        "config.max_autotune_gemm_backends = 'TRITON,ATEN,CPP'", # Not much faster
        f"config.aggressive_fusion = {O3}", # Careful changes results!
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
