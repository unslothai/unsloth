# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""One torch.compile region per transformer block for training.

Each decoder-layer class gets its stock forward compiled whole (attention, MLP or MoE, norms,
residuals, LoRA and NF4 projections), so a step pays one compiled-region call per layer instead
of dozens of small regions and eager ops. Pieces that would break or slow the graph:

- Attention marked `torch_compiler_disable_unless_decode` is inlined (UNSLOTH_DECODE_COMPILE set
  while a block runs); Gemma-4's sliding-window router gets its band verdict precomputed.
- fla's chunk_gated_delta_rule (Qwen3.5 / Qwen3.6) runs as the opaque op from `_gdn_op`, the
  causal conv as shifted multiply-adds; bitsandbytes NF4 projections use Unsloth's dequantize.
- Methods that are themselves torch.compile'd are inlined through their original, and
  functions whose globals no registered module owns are rebound to one that does.

Sequence dims are marked dynamic before the first compile, so a block compiles once per (layer
variant, grad mode). Each layer's first compile raises instead of falling back silently: a
failing layer retries with flex_attention outside its graph, then keeps the stock forward.
Calls with a KV cache keep the stock forward. UNSLOTH_BLOCK_COMPILE=0 or UNSLOTH_COMPILE_DISABLE
turns it off.
"""

import functools
import os
import sys
import types

import torch

__all__ = ["patch_block_compile"]

_MARK = "_unsloth_block_compiled"
# No max_autotune: one region per layer class already gives Inductor the large fusions, and
# autotuning every block multiplies first-step compile time.
_COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "shape_padding": True,
    "triton.cudagraphs": False,
}


# gpt-oss: unsloth_zoo builds its flex block mask inside attention (graph breaks on
# torch.is_inference_mode_enabled) and the block recompiled per sequence length (4 lengths: 27
# breaks, 53 graphs, 210 s of extra compiles); its per-module compile stays.
_SKIP = {"GptOssDecoderLayer"}


def _enabled():
    return os.environ.get("UNSLOTH_BLOCK_COMPILE", "1") != "0" and os.environ.get(
        "UNSLOTH_COMPILE_DISABLE", "0"
    ) not in ("1", "partial")


def _writes_globals(code):
    import dis
    if any(i.opname in ("STORE_GLOBAL", "DELETE_GLOBAL") for i in dis.get_instructions(code)):
        return True
    return any(_writes_globals(c) for c in code.co_consts if isinstance(c, types.CodeType))


def _rebind_globals(fn):
    """dynamo resolves a function's globals through sys.modules[__globals__["__name__"]]; Unsloth's
    generated modules exec code into a dict that module does not own (Gemma-4: `module
    'transformers...modeling_gemma4' has no attribute 'Gemma4RMSNorm_forward'`). Rebuild such a
    function on a registered module holding a copy of that dict. A copy cannot follow later
    writes, so functions that assign globals are left alone."""
    g = getattr(fn, "__globals__", None)
    if g is None or _writes_globals(fn.__code__):
        return fn
    mod = sys.modules.get(g.get("__name__"))
    if mod is not None and mod.__dict__ is g:
        return fn
    name = f"unsloth_block_globals_{id(g)}"
    holder = sys.modules.get(name)
    if holder is None:
        holder = types.ModuleType(name)
        holder.__dict__.update(g)
        holder.__dict__["__name__"] = name
        sys.modules[name] = holder
    new = types.FunctionType(
        fn.__code__, holder.__dict__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__dict__.update(fn.__dict__)
    new.__qualname__ = fn.__qualname__
    new.__module__ = name
    return new


def _inline_compiled(fn):
    """A method that is itself torch.compile'd (zoo's patch_function): inside a block, call the
    original so dynamo inlines it. Inlining through the compiled wrapper guards on the wrapper's
    closure as if it were the original's (torch 2.13 / 2.14, Gemma-4 fp16 MLP: `Unexpected type
    CatchErrorsWrapper`). Outside a block the compiled wrapper runs as before."""
    original = getattr(fn, "_torchdynamo_orig_callable", None)
    if (
        not isinstance(original, types.FunctionType)
        or getattr(fn, "_torchdynamo_disable", False)
        or getattr(fn, "_unsloth_block_inline", False)
    ):
        return fn
    compiled = fn
    original = _rebind_globals(original)

    @functools.wraps(original, updated = ())
    def dispatch(*args, **kwargs):
        if torch.compiler.is_compiling():
            return original(*args, **kwargs)
        return compiled(*args, **kwargs)

    dispatch._unsloth_block_inline = True
    return dispatch


def _rebind_class(cls):
    for key, value in list(cls.__dict__.items()):
        new = _inline_compiled(value) if callable(value) else value
        if isinstance(new, types.FunctionType):
            new = _rebind_globals(new)
        if new is not value:
            setattr(cls, key, new)


def _mark_seq_dynamic(seq_len, obj):
    if isinstance(obj, torch.Tensor):
        # The last dim of a float (batch, seq, features) tensor is a width, even when it equals
        # seq_len; integer ids (mrope position_ids: 3, batch, seq) keep it.
        last = obj.dim() - 1 if obj.dim() == 3 and obj.is_floating_point() else None
        for dim, size in enumerate(obj.shape):
            if size == seq_len and dim != last:
                torch._dynamo.maybe_mark_dynamic(obj, dim)
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _mark_seq_dynamic(seq_len, o)
    elif isinstance(obj, dict):
        for o in obj.values():
            _mark_seq_dynamic(seq_len, o)


# fla's opaque op state: None until a GatedDeltaNet layer ran its self-check, then True / False.
_GDN_OP = [None]
_GDN_STATE_KWARGS = (
    "initial_state",
    "cu_seqlens",
    "cu_seq_lens_q",
    "cu_seq_lens_k",
    "seq_idx",
    "scale",
)


def _gdn_chunk_adapter(original):
    # updated = (): the original's __dict__ carries torch.compiler.disable's marker.
    @functools.wraps(original, updated = ())
    def chunk_gated_delta_rule(
        q,
        k,
        v,
        g = None,
        beta = None,
        **kwargs,
    ):
        # Training case only (no recurrent state, no packing, q/k l2norm in kernel); the rest
        # keeps fla's own Function.
        if (
            _GDN_OP[0]
            and torch.compiler.is_compiling()
            and all(kwargs.get(key) is None for key in _GDN_STATE_KWARGS)
            and not kwargs.get("output_final_state")
            and kwargs.get("use_qk_l2norm_in_kernel")
        ):
            # Contiguous in the graph: Inductor pins a custom op's input strides to the example's
            # (a sliced projection keeps a size-specialized stride and fails at the next length).
            q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
            out = torch.ops.unsloth.qwen3_5_gdn_chunk(q, k, v, g, beta, q.shape[-1] ** -0.5)
            return out[0], None
        return original(q, k, v, g, beta, **kwargs)

    chunk_gated_delta_rule._unsloth_block_adapter = True
    return chunk_gated_delta_rule


def _route_fla_kernel():
    """Point modeling code's references to fla's chunk_gated_delta_rule at the adapter. Callers
    reach it through module globals, transformers' hub-kernel wrapper (a closure over it) and
    Unsloth's compiled-module copies, so callers are found from the function object (gc
    referrers), not by name. Closures are left alone: only namespace entries are swapped."""
    import gc

    kernels = []
    for name, module in list(sys.modules.items()):
        if module is not None and name.endswith("fla.ops.gated_delta_rule.chunk"):
            fn = getattr(module, "chunk_gated_delta_rule", None)
            if callable(fn) and fn not in kernels:
                kernels.append(fn)
    # Wrappers of wrappers (hub kernel, then Unsloth's compile-disable shim) all close over the
    # next one in: follow closures outwards a few levels.
    callers, frontier = list(kernels), list(kernels)
    for _ in range(4):
        found = []
        for inner in frontier:
            for cell in gc.get_referrers(inner):
                if type(cell).__name__ != "cell":
                    continue
                for closure in gc.get_referrers(cell):
                    if not isinstance(closure, tuple):
                        continue
                    for fn in gc.get_referrers(closure):
                        if (
                            isinstance(fn, types.FunctionType)
                            and fn.__closure__ is closure
                            and fn not in callers
                            and not getattr(fn, "_unsloth_block_adapter", False)
                        ):
                            callers.append(fn)
                            found.append(fn)
        frontier = found
        if not frontier:
            break
    routed = 0
    for caller in callers:
        if getattr(caller, "_unsloth_block_adapter", False):
            continue
        adapter = _gdn_chunk_adapter(caller)
        for space in gc.get_referrers(caller):
            if not isinstance(space, dict) or space is adapter.__dict__:
                continue
            for key, value in list(space.items()):
                if (
                    value is caller
                    and isinstance(key, str)
                    and key.endswith("chunk_gated_delta_rule")
                ):
                    space[key] = adapter
                    routed += 1
    return routed


def _causal_conv1d_adapter(original, depthwise_conv):
    # Inductor's convolution_backward lowering guards on the exact sequence length, so a conv
    # inside the block recompiles it for every new length. In the training case (no cache, no
    # packing) the depthwise causal conv runs as K shifted multiply-adds in fp32 instead.
    @functools.wraps(original, updated = ())
    def causal_conv1d_fn(
        hidden_states,
        weight,
        bias = None,
        activation = None,
        **kwargs,
    ):
        if (
            _IN_BLOCK[0]
            and torch.compiler.is_compiling()
            and weight.dim() == 2
            and activation in (None, "silu", "swish")
            and all(kwargs.get(key) is None for key in ("seq_idx", "cu_seqlens", "initial_states"))
            and not kwargs.get("return_final_states")
        ):
            out = depthwise_conv(hidden_states.to(weight.dtype), weight)
            if bias is not None:
                out = out + bias[:, None]
            if activation is not None:
                out = torch.nn.functional.silu(out)
            return out.to(hidden_states.dtype)
        return original(hidden_states, weight, bias, activation = activation, **kwargs)

    causal_conv1d_fn._unsloth_block_adapter = True
    return causal_conv1d_fn


def _namespaces(fn, depth = 3):
    """Global namespaces a forward reaches through wrappers (closures) and module-level helpers."""
    seen, out, todo = set(), [], [(fn, 0)]
    while todo:
        f, d = todo.pop()
        if not isinstance(f, types.FunctionType) or id(f) in seen:
            continue
        seen.add(id(f))
        if all(f.__globals__ is not g for g in out):
            out.append(f.__globals__)
        if d >= depth:
            continue
        for cell in f.__closure__ or ():
            try:
                todo.append((cell.cell_contents, d + 1))
            except ValueError:
                pass
        for name in f.__code__.co_names:
            todo.append((f.__globals__.get(name), d + 1))
    return out


def _route_causal_conv(gdn):
    from ._gdn_op import _causal_depthwise_conv
    for space in _namespaces(type(gdn).forward):
        fn = space.get("causal_conv1d_fn")
        if callable(fn) and not getattr(fn, "_unsloth_block_adapter", False):
            space["causal_conv1d_fn"] = _causal_conv1d_adapter(fn, _causal_depthwise_conv)


def _prepare_gdn(layer, hidden_states):
    """Self-check fla's opaque op on this layer's shapes, then point the GatedDeltaNet forward's
    chunk kernel at the adapter. Runs eagerly, before the first trace."""
    gdn = getattr(layer, "linear_attn", None)
    if gdn is None or _GDN_OP[0] is not None:
        return
    _GDN_OP[0] = False
    try:
        from ._gdn_op import _fla_ops

        heads, dk, dv = gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        dev, dt = hidden_states.device, hidden_states.dtype

        def make_inputs():
            gen = torch.Generator(device = dev).manual_seed(0)
            r = lambda *s, dtype = dt: torch.randn(*s, device = dev, dtype = dtype, generator = gen)
            g = -torch.rand(B, T, heads, device = dev, generator = gen) * 0.1
            return (
                r(B, T, heads, dk),
                r(B, T, heads, dk),
                r(B, T, heads, dv),
                g,
                r(B, T, heads).sigmoid(),
            )

        if not _fla_ops(make_inputs):
            return
        _GDN_OP[0] = _route_fla_kernel() > 0
        if _GDN_OP[0]:
            _route_causal_conv(gdn)
    except Exception:
        _GDN_OP[0] = False


# unsloth_zoo's Gemma-4 sliding-window router scans the mask (`_mask_is_plain_band`, data
# dependent) before taking is_causal SDPA (flash); while compiling it declines, so inside a block
# every sliding layer fell back to masked, mem-efficient SDPA (Gemma-4 12B on A100: 3.5x the
# attention backward time, the block 10% slower than eager). The block runs the scan eagerly,
# stores the verdict on the mask, and a router wrapper reads it while tracing.
_BAND_MODULE = "unsloth_zoo.temporary_patches.gemma4_banded_attention"
_BAND_CHECK = [None]


def _gemma4_router_hook():
    """Inside a block, a sliding layer whose mask is a plain causal band no longer than the
    window runs is_causal SDPA, as the stock router does; longer bands run masked SDPA. The
    router itself is not traced: its engagement counters drive a log-once branch dynamo guards
    on, and its banded kernel traced in a checkpointed block recomputes different saved tensors
    than its forward (CheckpointError)."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return
    router = ALL_ATTENTION_FUNCTIONS.get("sdpa")
    if not getattr(router, "_unsloth_gemma4_flash", False) or getattr(
        router, "_unsloth_block_band", False
    ):
        return
    zoo = sys.modules.get(router.__module__)
    plain = getattr(zoo, "_ORIG_SDPA", [None])[0]
    check = getattr(sys.modules.get(_BAND_MODULE), "_mask_is_plain_band", None)
    if plain is None or not callable(check):
        return
    _BAND_CHECK[0] = check

    # wraps() copies the router's markers: zoo's sdpa patches skip a function carrying theirs, so
    # a later model load does not wrap this again with it as the fallback (an endless loop).
    @functools.wraps(router)
    def sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout = 0.0,
        scaling = None,
        is_causal = None,
        **kwargs,
    ):
        if (
            _IN_BLOCK[0]
            and torch.compiler.is_compiling()
            and getattr(module, "is_sliding", False)
            and type(module).__name__.startswith("Gemma4")
            and query.dim() == 4
            and query.shape[-1] <= 256
        ):
            window = kwargs.get("sliding_window", None) or getattr(module, "sliding_window", None)
            causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
            if (
                window
                and query.shape[2] == key.shape[2]
                and query.shape[2] <= window
                and (
                    causal
                    if attention_mask is None
                    else getattr(attention_mask, "_unsloth_block_band", False) is True
                )
            ):
                out = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p = dropout if module.training else 0.0,
                    is_causal = True,
                    scale = scaling,
                    enable_gqa = True,
                )
                return out.transpose(1, 2).contiguous(), None
            return plain(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout = dropout,
                scaling = scaling,
                is_causal = is_causal,
                **kwargs,
            )
        return router(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout = dropout,
            scaling = scaling,
            is_causal = is_causal,
            **kwargs,
        )

    sdpa_attention_forward._unsloth_block_band = True
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward


def _sliding_window(layer):
    window = layer.__dict__.get("_unsloth_block_window", 0)
    if window == 0:
        window = None
        for module in layer.modules():
            if getattr(module, "is_sliding", False) and getattr(module, "sliding_window", None):
                window = module.sliding_window
                break
        layer.__dict__["_unsloth_block_window"] = window
    return window


def _precheck_band(layer, seq_len, args, kwargs):
    # Every 4D mask entering a block carries the verdict, so the traced read never misses (a
    # missing attribute would recompile).
    check = _BAND_CHECK[0]
    if check is None:
        return
    window = _sliding_window(layer)
    for obj in (*args, *kwargs.values()):
        if isinstance(obj, torch.Tensor) and obj.dim() == 4 and not obj.requires_grad:
            if window is not None and seq_len <= window:
                try:
                    obj._unsloth_block_band = bool(check(obj, seq_len, window))
                except Exception:
                    obj._unsloth_block_band = False
            elif not hasattr(obj, "_unsloth_block_band"):
                obj._unsloth_block_band = False


# Mode of the block being traced: 0 none, _FULL, or _SPLIT (flex_attention left outside the
# block). Read inside the trace, so dynamo guards on it.
_IN_BLOCK = [0]
_OFF, _FULL, _SPLIT = 0, 1, 2
# UNSLOTH_BLOCK_FLEX=outside starts every layer with flex_attention outside its block.
_FIRST_MODE = _SPLIT if os.environ.get("UNSLOTH_BLOCK_FLEX", "inside") == "outside" else _FULL


def _flex_split_hook():
    """Lets a block keep flex_attention out of its graph. Some masks do not lower inside a larger
    graph (torch 2.13, Gemma-4 12B blockwise mask: `unsupported operand type(s) for &:
    'FloorDiv' and 'Add'`); flex then runs as its own compiled region, one break per layer."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return
    flex = ALL_ATTENTION_FUNCTIONS.get("flex_attention")
    if flex is None or getattr(flex, "_unsloth_block_split", False):
        return
    outside = torch.compiler.disable(flex)

    @functools.wraps(flex)
    def flex_attention_forward(*args, **kwargs):
        if _IN_BLOCK[0] == _SPLIT and torch.compiler.is_compiling():
            return outside(*args, **kwargs)
        return flex(*args, **kwargs)

    flex_attention_forward._unsloth_block_split = True
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = flex_attention_forward


def _nf4_in_blocks():
    """bitsandbytes NF4 Linear4bit inside a block: Unsloth's Triton dequantize + F.linear (bit-exact
    to bitsandbytes' dequantize; kernels/bnb_override.py), on every GPU. Outside blocks the
    override keeps its own per-GPU default."""
    try:
        import bitsandbytes as bnb

        from ..kernels import bnb_override as O
    except Exception:
        return
    if not O._TRACE:
        return  # torch < 2.11 miscompiles the traced backward
    Linear4bit = bnb.nn.Linear4bit
    current = Linear4bit.forward
    if getattr(current, "_unsloth_block_nf4", False):
        return

    # wraps() copies bnb_override's marker, so a later install does not take this as its fallback.
    @functools.wraps(current)
    def forward(self, x, *args, **kwargs):
        if _IN_BLOCK[0] and torch.compiler.is_compiling() and not args and not kwargs:
            weight = self.weight
            quant_state = getattr(weight, "quant_state", None)
            if O._eligible(self, x, weight, quant_state) and (
                quant_state.dtype is not torch.bfloat16 or O._TRACE_BF16
            ):
                dtype = quant_state.dtype
                bias = None if self.bias is None else self.bias.to(dtype)
                return O._linear(x.to(dtype), weight, quant_state, bias).to(x.dtype)
        return current(self, x, *args, **kwargs)

    forward._unsloth_block_nf4 = True
    Linear4bit.forward = forward


def _wrap(cls, compile_options):
    original = cls.forward
    compiled = torch.compile(original, fullgraph = False, dynamic = None, options = compile_options)
    try:
        from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE as inline_flag
    except Exception:
        inline_flag = [False]
    # Each layer's first call per grad mode compiles with errors raised, not swallowed (layers of
    # one class differ: sliding / full attention, widths, quantized or not). A layer that does not
    # compile retries with flex_attention outside, then keeps the stock forward. Per layer, never
    # per class: a checkpointed layer must recompute in the mode its forward ran in.
    state = {"reported": set()}

    def run(self, hidden_states, args, kwargs, mode):
        previous = inline_flag[0]
        inline_flag[0] = True
        _IN_BLOCK[0] = mode
        try:
            return compiled(self, hidden_states, *args, **kwargs)
        finally:
            inline_flag[0] = previous
            _IN_BLOCK[0] = _OFF

    @functools.wraps(original)
    def forward(self, hidden_states, *args, **kwargs):
        mode = self.__dict__.get("_unsloth_block_mode", _FIRST_MODE)
        # Training only: eval / generation keep the stock forward, so their logits match
        # transformers bit for bit (Inductor's bf16 fusions round differently).
        if (
            mode == _OFF
            or not self.training
            or kwargs.get("past_key_values") is not None
            or not _enabled()
        ):
            return original(self, hidden_states, *args, **kwargs)
        if _GDN_OP[0] is None and hasattr(self, "linear_attn"):
            _prepare_gdn(self, hidden_states)
        if hidden_states.dim() == 3 and hidden_states.shape[1] > 1:
            _mark_seq_dynamic(hidden_states.shape[1], (hidden_states, args, kwargs))
            _precheck_band(self, hidden_states.shape[1], args, kwargs)
        bit = 2 if torch.is_grad_enabled() else 1
        checked = self.__dict__.get("_unsloth_block_checked", 0)
        if checked & bit:
            return run(self, hidden_states, args, kwargs, mode)
        while True:
            try:
                with torch._dynamo.config.patch(suppress_errors = False):
                    out = run(self, hidden_states, args, kwargs, mode)
                self.__dict__["_unsloth_block_checked"] = checked | bit
                return out
            except torch.OutOfMemoryError:
                raise
            except Exception as e:
                # Compile errors, and dynamo internal errors (torch 2.14 T4: an AssertionError
                # building guards) that the default suppress_errors would have swallowed.
                if os.environ.get("UNSLOTH_BLOCK_COMPILE_DEBUG") == "1":
                    import traceback
                    traceback.print_exc()
                mode = _SPLIT if mode == _FULL else _OFF
                self.__dict__["_unsloth_block_mode"] = mode
                first = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
                if (first, mode) not in state["reported"]:
                    state["reported"].add((first, mode))
                    print(
                        f"Unsloth: {cls.__name__} block compile failed ({first}); "
                        + (
                            "retrying with flex_attention outside the block."
                            if mode == _SPLIT
                            else "keeping the stock forward for this layer."
                        )
                    )
                if mode == _OFF:
                    return original(self, hidden_states, *args, **kwargs)

    forward._unsloth_block_original = original
    forward._unsloth_block_state = state
    setattr(forward, _MARK, True)
    cls.forward = forward


def _layer_lists(model):
    """ModuleLists of decoder layers (one class each): the text stack, not vision towers."""
    return [
        module
        for module in model.modules()
        if isinstance(module, torch.nn.ModuleList)
        and len(module) > 0
        and type(module[0]).__name__.endswith("DecoderLayer")
        and all(type(m) is type(module[0]) for m in module)
    ]


def _decoder_layer_classes(model):
    classes = []
    for layer_list in _layer_lists(model):
        if type(layer_list[0]) not in classes:
            classes.append(type(layer_list[0]))
    return classes


def patch_block_compile(model):
    """Compile every decoder-layer class of `model` as one region. Returns the classes wrapped."""
    if not _enabled() or not torch.cuda.is_available():
        return []
    classes = _decoder_layer_classes(model)
    if not classes:
        return []
    # Only what runs inside a block: the model's top-level forward keeps its own globals.
    inner = {
        type(m)
        for layer_list in _layer_lists(model)
        for layer in layer_list
        for m in layer.modules()
    }
    for cls in inner:
        _rebind_class(cls)
    _flex_split_hook()
    _gemma4_router_hook()
    if any(type(m).__name__ == "Linear4bit" for m in model.modules()):
        _nf4_in_blocks()
    wrapped = []
    for cls in classes:
        if getattr(cls.forward, _MARK, False) or cls.__name__ in _SKIP:
            continue
        _wrap(cls, _COMPILE_OPTIONS)
        wrapped.append(cls.__name__)
    if wrapped:
        print(f"Unsloth: compiling each transformer block as one region ({', '.join(wrapped)}).")
    return wrapped
