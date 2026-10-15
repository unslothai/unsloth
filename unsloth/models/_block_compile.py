# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""One torch.compile region per decoder layer for training (stock forward: attention, MLP or MoE,
norms, LoRA, NF4). UNSLOTH_BLOCK_COMPILE=0 or UNSLOTH_COMPILE_DISABLE turns it off."""

import functools
import os
import sys
import types

import torch

__all__ = ["patch_block_compile"]

_MARK = "_unsloth_block_compiled"
# No max_autotune: it multiplies first-step compile time for every block.
_COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "shape_padding": True,
    "triton.cudagraphs": False,
}


# gpt-oss builds its flex mask inside attention: the block broke and recompiled per length.
_SKIP = {"GptOssDecoderLayer"}


def _enabled():
    return os.environ.get("UNSLOTH_BLOCK_COMPILE", "1") != "0" and os.environ.get(
        "UNSLOTH_COMPILE_DISABLE", "0"
    ) not in ("1", "partial")


def _forced_float32(module):
    # Zoo's float32-forced patches round to fp16 on purpose; Inductor drops those roundings.
    # Stamped per layer at load: the env var only describes the latest load.
    forced = module.__dict__.get("_unsloth_forced_float32")
    if forced is None:
        forced = os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1"
    return forced


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
        # A float tensor's last dim is a width even when it equals seq_len (mrope ids are int).
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
        if (
            _GDN_OP[0]
            and torch.compiler.is_compiling()
            and kwargs.get("use_qk_l2norm_in_kernel") is True
            # The op is the self-checked default: no state, varlen, scale, layout or gating flags.
            and all(
                value is None or value is False
                for key, value in kwargs.items()
                if key != "use_qk_l2norm_in_kernel"
            )
        ):
            # Inductor pins a custom op's input strides; a sliced input fails at the next length.
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
    # Inductor's conv backward guards on the exact sequence length: recompile per length.
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


def _route_sm_count(modules = None):
    # ROCm: Triton's hip_utils.get_device_properties is untraceable; the SM count is per device.
    for name, module in list((sys.modules if modules is None else modules).items()):
        if module is None or "fla." not in name:
            continue
        fn = getattr(module, "get_multiprocessor_count", None)
        if not callable(fn) or getattr(fn, "_unsloth_block_adapter", False):
            continue
        try:
            counts = {i: int(fn(i)) for i in range(torch.cuda.device_count())}
        except Exception:
            continue

        @functools.wraps(fn, updated = ())
        def get_multiprocessor_count(
            tensor_idx = 0,
            _original = fn,
            _counts = counts,
        ):
            count = _counts.get(tensor_idx)
            return _original(tensor_idx) if count is None else count

        get_multiprocessor_count._unsloth_block_adapter = True
        module.get_multiprocessor_count = get_multiprocessor_count


def _prepare_gdn(layer, hidden_states):
    """Self-check fla's opaque op on this layer's shapes, then point the GatedDeltaNet forward's
    chunk kernel at the adapter. Runs eagerly, before the first trace."""
    gdn = getattr(layer, "linear_attn", None)
    if gdn is None or _GDN_OP[0] is not None:
        return
    _GDN_OP[0] = False
    try:
        _route_sm_count()
    except Exception:
        pass
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


# zoo's Gemma-4 sliding router scans the mask (data dependent) and declines while compiling, so
# the block runs the scan eagerly and a router wrapper reads the verdict stored on the mask.
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

    # wraps() keeps zoo's marker, else a later load wraps this as its own fallback (endless loop).
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
    # Every 4D mask gets a verdict: a missing attribute would recompile.
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


# Mode of the block being traced (_SPLIT: flex_attention outside); dynamo guards on it.
_IN_BLOCK = [0]
_OFF, _FULL, _SPLIT = 0, 1, 2
_DEMOTION = (_FULL, _SPLIT, _OFF)
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
    if not O._TRACE or O._LINEAR == "0":
        return  # torch < 2.11 miscompiles the traced backward; "0" = the user's opt-out
    Linear4bit = bnb.nn.Linear4bit
    current = Linear4bit.forward
    if getattr(current, "_unsloth_block_nf4", False):
        return

    @functools.wraps(current)
    def forward(self, x, *args, **kwargs):
        # Both checkpoint halves must take one kernel: a trainable bias fails _eligible only under
        # grad, and one row picks fast_gemv only without grad (O._linear).
        bias = self.bias
        rows = x.numel() // max(x.shape[-1], 1)
        if (
            _IN_BLOCK[0]
            and torch.compiler.is_compiling()
            and not args
            and not kwargs
            and (bias is None or not bias.requires_grad)
            and rows > 1
        ):
            weight = self.weight
            quant_state = getattr(weight, "quant_state", None)
            if (
                O._eligible(self, x, weight, quant_state)
                and (quant_state.dtype is not torch.bfloat16 or O._TRACE_BF16)
                # bitsandbytes' fp32 small-row kernel is more accurate (O._forward keeps it too).
                and not (quant_state.dtype is torch.float32 and rows < 8)
            ):
                dtype = quant_state.dtype
                bias = None if self.bias is None else self.bias.to(dtype)
                return O._linear(x.to(dtype), weight, quant_state, bias).to(x.dtype)
        return current(self, x, *args, **kwargs)

    forward._unsloth_block_nf4 = True
    Linear4bit.forward = forward


_CACHE_KWARGS = ("past_key_values", "past_key_value", "cache_params", "layer_past")


def _has_cache(args, kwargs):
    # transformers 4.x layers take the singular `past_key_value`, sometimes positionally.
    if any(kwargs.get(key) is not None for key in _CACHE_KWARGS):
        return True
    try:
        from transformers.cache_utils import Cache
    except Exception:
        return False
    return any(isinstance(arg, Cache) for arg in args)


def _layer_kind(layer):
    attn = getattr(layer, "self_attn", None) or getattr(layer, "linear_attn", None)
    return (
        type(attn).__name__,
        getattr(layer, "attention_type", None),
        getattr(attn, "is_sliding", None),
        getattr(attn, "sliding_window", None),
        getattr(attn, "head_dim", None),
    )


def _wrap(cls, compile_options):
    original = cls.forward
    compiled = torch.compile(original, fullgraph = False, dynamic = None, options = compile_options)
    try:
        from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE as inline_flag
    except Exception:
        inline_flag = [False]
    # First call per grad mode compiles with errors raised: on failure retry with flex outside,
    # then the stock forward. The mode is per layer (a checkpoint recompute must not switch graph);
    # a failure is also the starting mode for later layers of the same kind.
    state = {"reported": set(), "start": {}}

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
        mode = self.__dict__.get("_unsloth_block_mode")
        if mode is None:
            mode = state["start"].get(_layer_kind(self), _FIRST_MODE)
            self.__dict__["_unsloth_block_mode"] = mode
        # Training only: eval logits stay bit-equal to transformers.
        if (
            mode == _OFF
            or not self.training
            or _has_cache(args, kwargs)
            or not _enabled()
            or _forced_float32(self)
        ):
            return original(self, hidden_states, *args, **kwargs)
        if not self.__dict__.get("_unsloth_block_moe_done", False):
            _experts_outside(self)
        if _GDN_OP[0] is None and hasattr(self, "linear_attn"):
            _prepare_gdn(self, hidden_states)
        if hidden_states.dim() == 3 and hidden_states.shape[1] > 1:
            _mark_seq_dynamic(hidden_states.shape[1], (hidden_states, args, kwargs))
            _precheck_band(self, hidden_states.shape[1], args, kwargs)
        bit = 2 if torch.is_grad_enabled() else 1
        checked = self.__dict__.get("_unsloth_block_checked", 0)
        if bit == 1 and not checked & 2:
            # Settle the grad-mode graph first: a checkpoint recompute must not demote below the
            # mode this no-grad forward ran in. fork_rng: the trial leaves dropout draws alone.
            devices = [hidden_states.device.index] if hidden_states.is_cuda else []
            with torch.random.fork_rng(devices = devices), torch.enable_grad():
                forward(self, hidden_states.detach().requires_grad_(), *args, **kwargs)
            mode = self.__dict__["_unsloth_block_mode"]
            if mode == _OFF:
                return original(self, hidden_states, *args, **kwargs)
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
                if os.environ.get("UNSLOTH_BLOCK_COMPILE_DEBUG") == "1":
                    import traceback
                    traceback.print_exc()
                mode = _SPLIT if mode == _FULL else _OFF
                self.__dict__["_unsloth_block_mode"] = mode
                kind = _layer_kind(self)
                state["start"][kind] = max(
                    state["start"].get(kind, _FIRST_MODE), mode, key = _DEMOTION.index
                )
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


def _torch_supported():
    # torch 2.6: blocks fragmented and QLoRA grads moved vs the stock forward.
    from packaging.version import Version
    return Version(torch.__version__.split("+")[0]) >= Version("2.7")


def _experts_of(module):
    while module is not None:
        if type(module).__name__.endswith("Experts"):
            return module
        module = getattr(module, "base_layer", None)
    return None


def _lora_would_fold(experts):
    # Traced, zoo folds the LoRA delta into every expert's weights when the forward ignores the stash.
    try:
        from unsloth_zoo.temporary_patches import moe_utils as zoo_moe
        reads = zoo_moe._forward_statically_reads_stash(experts)
        if not reads and zoo_moe._interface_route_reads_stash(experts):
            reads = True
    except Exception:
        return False
    return reads is False


def _experts_outside(layer):
    # At the layer's first call, after get_peft_model wrapped the experts.
    layer.__dict__["_unsloth_block_moe_done"] = True
    mode = os.environ.get("UNSLOTH_BLOCK_MOE", "auto")
    if mode == "inside":
        return
    for _, module in layer.named_modules():
        if getattr(module, "_unsloth_block_outside", False):
            continue
        experts = _experts_of(module)
        if experts is None:  # named_modules() is pre-order: the outermost wrapper comes first
            continue
        if mode != "outside" and not _lora_would_fold(experts):
            continue
        module.forward = torch.compiler.disable(module.forward)
        for inner in module.modules():
            inner._unsloth_block_outside = True


def patch_block_compile(model):
    """Compile every decoder-layer class of `model` as one region. Returns the classes wrapped."""
    # getattr: PeftModel delegates to the model it wraps. Without a flag, the env describes this
    # load only for unstamped layers; a re-patch (get_peft_model) keeps the stamp.
    forced = getattr(model, "_unsloth_forced_float32", None)
    env = os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1"
    for layer_list in _layer_lists(model):
        for layer in layer_list:
            if forced is None:
                layer.__dict__.setdefault("_unsloth_forced_float32", env)
            else:
                layer.__dict__["_unsloth_forced_float32"] = bool(forced)
    forced = env if forced is None else forced
    if forced or not _enabled() or not torch.cuda.is_available() or not _torch_supported():
        return []
    classes = _decoder_layer_classes(model)
    if not classes:
        return []
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
