# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Prefix-shared preference forward (DPO / ORPO / CPO / SimPO).

The preference trainers score ``[chosen_0 .. chosen_{B-1}, rejected_0 .. rejected_{B-1}]`` rows, and
row ``i`` and row ``i + B`` start with the same prompt. Inside those trainers' forward methods the
decoder backbone is hooked: the longest common prefix of each pair is run once, both completions
attend to it through the PrefixGrouper FlexAttention mask (``prefix_seg_info``), and the final hidden
states are gathered back to the padded ``[2B, L, H]`` layout. Everything after the backbone (TRL's
fused head, Unsloth's fused preference loss, full logits) is unchanged, and the shared prefix gets
the sum of both rows' gradients through the gather. Tokens per pair go from ``2 (P + C)`` to
``P + 2 C``.

Env ``UNSLOTH_DPO_PREFIX_SHARE``: ``auto`` (default: on for CUDA sm80+ when the forward applies
``prefix_seg_info``), ``1`` (also below sm80), ``0`` (off). A batch is shared only when it saves at
least ``_MIN_SAVED`` (default 2048) prompt tokens and ``unshared / shared`` tokens is at least ``_TOKR``
(default 1.3). ``_VERIFY`` (default 1) compares the first shared forward of each backbone with the
unshared one and turns the feature off for that backbone if they disagree.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import os
from dataclasses import dataclass
from typing import List, Optional

import torch

__all__ = [
    "PairLayout",
    "build_pair_layout",
    "preference_prefix_share",
    "wrap_preference_trainer",
    "prefix_share_mode",
]

# Engagement counters, read by tests and benchmarks.
ENGAGED_COUNT = 0
DECLINED_COUNT = 0
LAST_VERIFY = None

_STATE = "_unsloth_dpo_prefix_share_state"
_UNSAFE = "_unsloth_dpo_prefix_share_unsafe"
_VERIFIED = "_unsloth_dpo_prefix_share_verified"
_ACTIVE = "_unsloth_dpo_prefix_share_active"

# Hidden-state disagreement (relative L2 per token) that marks a backbone unsafe. FlexAttention vs
# the padded forward on Qwen3-0.6B: mean 1.6e-2 / max 3.3e-2 in bf16, ~1e-6 in fp32; a completion
# that sees its sibling: mean 8e-2 / max 0.66 (Qwen3-0.6B fp32) and 1.4 (tiny random model).
_VERIFY_MEAN_TOL = 0.05
_VERIFY_MAX_TOL = 0.3


def prefix_share_mode() -> str:
    mode = os.environ.get("UNSLOTH_DPO_PREFIX_SHARE", "auto").strip().lower()
    if mode in ("0", "false", "no", "off"):
        return "off"
    if mode in ("1", "true", "yes", "on", "force"):
        return "on"
    return "auto"


def _env_number(
    name,
    default,
    cast = float,
):
    try:
        return cast(os.environ.get(name, default))
    except ValueError:
        return cast(default)


# Auto gate, from Qwen3-1.7B LoRA on an A100 (bs 4, C = 300): the shared path costs a fixed host
# overhead per forward (more, smaller kernels), so it pays only once enough prompt tokens are saved.
# 800 saved (P = 200): +14% step time; 2048 (P = 512): -8%; 4096 (P = 1024): -24%; 8192: -27%.
def _tokr_threshold() -> float:
    return _env_number("UNSLOTH_DPO_PREFIX_SHARE_TOKR", "1.3")


def _min_saved() -> int:
    return _env_number("UNSLOTH_DPO_PREFIX_SHARE_MIN_SAVED", "2048", int)


def _verify_on() -> bool:
    return os.environ.get("UNSLOTH_DPO_PREFIX_SHARE_VERIFY", "1").lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _log(message):
    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") in ("1", "True", "true"):
        print(f"Unsloth: DPO prefix share: {message}", flush = True)


@dataclass
class PairLayout:
    """Flat shared-prefix stream for ``2B`` paired rows, and the map back to ``[2B, L]``."""

    flat_ids: torch.Tensor  # [1, T]
    position_ids: torch.Tensor  # [1, T]
    prefix_seg_info: object  # PrefixSegInfo
    gather: torch.Tensor  # [2B * L] flat index each padded (row, col) reads its hidden state from
    rows: int
    L: int
    tok_r: float
    shared: List[int]  # shared prefix length per pair


def build_pair_layout(
    input_ids,
    attention_mask,
    *,
    min_tok_r = None,
    min_saved = None,
    max_segment_cap = None,
):
    """PairLayout for ``input_ids`` ``[2B, L]`` (right padded, row ``i`` paired with ``i + B``), or
    None when the batch is not that shape or sharing would not pay.

    The shared prefix of a pair is the longest common prefix of the two real token runs, capped so
    each row keeps at least one token of its own. Every real token keeps its own position (its
    column), so with right padding the result is the unshared forward's, token for token.
    """
    from ..utils.prefix_grouper_kernel import build_seg_info_multigroup

    if not (isinstance(input_ids, torch.Tensor) and isinstance(attention_mask, torch.Tensor)):
        return None
    if input_ids.dim() != 2 or attention_mask.shape != input_ids.shape:
        return None
    if input_ids.dtype not in (torch.int64, torch.int32) or input_ids.is_floating_point():
        return None
    rows, L = input_ids.shape
    if rows < 2 or rows % 2 != 0 or L < 2:
        return None
    B = rows // 2
    device = input_ids.device
    keep = attention_mask.to(torch.bool)
    n_real = keep.sum(dim = 1)
    ar = torch.arange(L, device = device)
    right_padded = (keep == (ar < n_real.unsqueeze(1))).all()
    same = (input_ids[:B] == input_ids[B:]) & keep[:B] & keep[B:]
    lcp = same.to(torch.int32).cumprod(dim = 1).sum(dim = 1)
    # One device-to-host read.
    host = torch.cat(
        [n_real.to(torch.long), lcp.to(torch.long), right_padded.to(torch.long).view(1)]
    )
    host = host.tolist()
    if not host[-1]:
        return None
    n_list = host[:rows]
    lcp_list = host[rows : rows + B]

    shared = []
    specs = []
    for i in range(B):
        n_c, n_r = n_list[i], n_list[B + i]
        s = min(lcp_list[i], n_c - 1, n_r - 1)
        if s < 1:
            return None
        shared.append(s)
        specs.append((s, [n_c - s, n_r - s]))
    if max_segment_cap is not None and max(n_list) > max_segment_cap:
        return None
    unshared = sum(n_list)
    flat_len = unshared - sum(shared)
    tok_r = unshared / flat_len
    if min_tok_r is None:
        min_tok_r = _tokr_threshold()
    if min_saved is None:
        min_saved = _min_saved()
    if tok_r < min_tok_r or sum(shared) < min_saved:
        return None

    seg, meta = build_seg_info_multigroup(specs, device)
    T = seg.T
    assert T == flat_len, (T, flat_len)

    src_rows = []
    src_cols = []
    positions = []
    gather = []
    for i in range(B):
        s = shared[i]
        n_c, n_r = n_list[i], n_list[B + i]
        base = meta[i]["base"]
        (c0, _), (r0, _) = meta[i]["suffix_slices"]
        # Flat order: prefix (from the chosen row), chosen suffix, rejected suffix.
        src_rows.append(torch.full((n_c,), i, dtype = torch.long))
        src_cols.append(torch.arange(n_c, dtype = torch.long))
        src_rows.append(torch.full((n_r - s,), B + i, dtype = torch.long))
        src_cols.append(torch.arange(s, n_r, dtype = torch.long))
        positions.append(torch.arange(n_c, dtype = torch.long))
        positions.append(torch.arange(s, n_r, dtype = torch.long))
        # Chosen row: prefix then its own suffix, contiguous from base. Pad columns read the last token.
        g_c = torch.full((L,), base + n_c - 1, dtype = torch.long)
        g_c[:n_c] = torch.arange(base, base + n_c, dtype = torch.long)
        assert c0 == base + s
        g_r = torch.full((L,), r0 + (n_r - s) - 1, dtype = torch.long)
        g_r[:s] = torch.arange(base, base + s, dtype = torch.long)
        g_r[s:n_r] = torch.arange(r0, r0 + (n_r - s), dtype = torch.long)
        gather.append((i, g_c))
        gather.append((B + i, g_r))
    gather.sort(key = lambda x: x[0])
    table = torch.cat(
        [torch.cat(src_rows), torch.cat(src_cols), torch.cat(positions)] + [g for _, g in gather]
    ).to(device)
    fr, fc, pos, g = torch.split(table, [T, T, T, rows * L])
    flat_ids = input_ids[fr, fc].unsqueeze(0)
    return PairLayout(
        flat_ids = flat_ids,
        position_ids = pos.unsqueeze(0),
        prefix_seg_info = seg,
        gather = g,
        rows = rows,
        L = L,
        tok_r = tok_r,
        shared = shared,
    )


# ---------------------------------------------------------------------------------------------
# Eligibility


_SSM_MOE_ATTRS = (
    "mamba_d_ssm",
    "mamba_d_state",
    "mamba_expand",
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_intermediate_size",
)


def _unwrap(trainer, model):
    seen = 0
    while seen < 6 and model is not None:
        inner = getattr(model, "module", None)  # DDP
        if isinstance(model, torch.nn.parallel.DistributedDataParallel) and inner is not None:
            model = inner
        elif getattr(model, "_orig_mod", None) is not None:
            model = model._orig_mod
        else:
            break
        seen += 1
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        try:
            model = get_base()
        except Exception:
            return None
    return model


def _backbone_of(causal_lm):
    prefix = getattr(causal_lm, "base_model_prefix", None)
    backbone = getattr(causal_lm, prefix, None) if isinstance(prefix, str) else None
    if not isinstance(backbone, torch.nn.Module) or backbone is causal_lm:
        return None
    if not isinstance(getattr(backbone, "layers", None), torch.nn.ModuleList):
        return None
    return backbone


def _model_eligible(causal_lm, device):
    """Structural checks, cached per model; the device capability check runs per call."""
    cached = getattr(causal_lm, "_unsloth_dpo_prefix_share_eligible", None)
    if cached is None:
        cached = False
        try:
            from ..utils.prefix_grouper import prefix_seg_info_applied
            from ..utils.prefix_grouper_kernel import TORCH_SUPPORTS_PREFIX_GROUPER

            config = getattr(causal_lm, "config", None)
            text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
            cached = bool(
                TORCH_SUPPORTS_PREFIX_GROUPER
                and config is not None
                and text_config is config  # text-only decoders
                and getattr(config, "vision_config", None) is None
                and not getattr(config, "is_encoder_decoder", False)
                and not getattr(config, "attn_logit_softcapping", None)
                and not getattr(config, "attention_dropout", 0)
                and not any(getattr(config, a, None) is not None for a in _SSM_MOE_ATTRS)
                and _backbone_of(causal_lm) is not None
                and not any(type(p).__name__ == "DTensor" for p in causal_lm.parameters())
                and prefix_seg_info_applied(causal_lm)
            )
        except Exception:
            cached = False
        try:
            causal_lm._unsloth_dpo_prefix_share_eligible = cached
        except Exception:
            pass
    if not cached:
        return False
    from ..utils.prefix_grouper_kernel import compile_check_failed

    if compile_check_failed() is not None:
        return False
    if device is None or device.type != "cuda":
        return False
    if prefix_share_mode() == "auto":
        try:
            if torch.cuda.get_device_capability(device)[0] < 8:
                return False
        except Exception:
            return False
    return True


def _segment_cap(causal_lm):
    window = getattr(getattr(causal_lm, "config", None), "sliding_window", None)
    return window if isinstance(window, int) and window > 0 else None


def _distributed_world():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _cp_active():
    try:
        from ..context_parallel import get_cp_manager
    except Exception:
        return False
    try:
        return get_cp_manager() is not None
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------
# Backbone hooks

_PASSTHROUGH_NONE = (
    "position_ids",
    "past_key_values",
    "inputs_embeds",
    "cache_position",
    "causal_mask",
)
_ALLOWED_KWARGS = frozenset(
    (
        "input_ids",
        "attention_mask",
        "use_cache",
        "return_dict",
        "output_attentions",
        "output_hidden_states",
    )
    + _PASSTHROUGH_NONE
)


def _unknown_kwarg_set(kwargs):
    """Any kwarg outside the known set that carries a value (tensor, True, non-zero)."""
    for name, value in kwargs.items():
        if name in _ALLOWED_KWARGS or value is None or value is False:
            continue
        if isinstance(value, int) and not isinstance(value, bool) and value == 0:
            continue
        return True
    return False


class _HookState:
    def __init__(self, causal_lm):
        self.causal_lm = causal_lm
        self.pending = []


def _pre_hook(module, args, kwargs):
    global DECLINED_COUNT
    state = getattr(module, _STATE, None)
    if state is None or args or getattr(module, _UNSAFE, False):
        return None
    if _unknown_kwarg_set(kwargs):
        DECLINED_COUNT += 1
        return None
    if any(kwargs.get(name) is not None for name in _PASSTHROUGH_NONE if name != "causal_mask"):
        return None
    if kwargs.get("use_cache") or kwargs.get("output_attentions"):
        return None
    # TRL < 1.0 DPO always asks for every layer's hidden states; those are gathered back too, which
    # needs a ModelOutput to put them in.
    if kwargs.get("output_hidden_states") and kwargs.get("return_dict") is False:
        return None
    input_ids = kwargs.get("input_ids")
    attention_mask = kwargs.get("attention_mask")
    if attention_mask is None or not isinstance(input_ids, torch.Tensor):
        return None
    try:
        layout = build_pair_layout(
            input_ids, attention_mask, max_segment_cap = _segment_cap(state.causal_lm)
        )
    except Exception as exc:
        _log(f"layout failed ({exc!r}); unshared forward")
        layout = None
    if layout is None:
        DECLINED_COUNT += 1
        return None
    new_kwargs = dict(kwargs)
    new_kwargs["input_ids"] = layout.flat_ids
    new_kwargs["position_ids"] = layout.position_ids
    new_kwargs["attention_mask"] = None
    new_kwargs["prefix_seg_info"] = layout.prefix_seg_info
    if "causal_mask" in new_kwargs:
        new_kwargs["causal_mask"] = None
    state.pending.append((layout, kwargs))
    return args, new_kwargs


def _hidden_of(output):
    if isinstance(output, tuple):
        return output[0]
    return output.last_hidden_state


def _with_hidden(
    output,
    hidden,
    layout = None,
):
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    output.last_hidden_state = hidden
    every = getattr(output, "hidden_states", None)
    if layout is not None and isinstance(every, (tuple, list)) and every:
        output.hidden_states = tuple(_gather(h, layout) for h in every)
    return output


def _gather(hidden, layout):
    """[1, T, H] shared stream -> [2B, L, H] padded rows."""
    if not isinstance(hidden, torch.Tensor) or hidden.dim() != 3 or hidden.shape[0] != 1:
        return hidden
    if hidden.shape[1] != layout.flat_ids.shape[1]:
        raise RuntimeError(
            f"Unsloth: DPO prefix share got hidden states {tuple(hidden.shape)} for a "
            f"{tuple(layout.flat_ids.shape)} stream."
        )
    full = hidden[0].index_select(0, layout.gather.to(hidden.device))
    return full.view(layout.rows, layout.L, hidden.shape[-1])


def _token_rel_err(shared, unshared, keep):
    a = shared.float()[keep]
    b = unshared.float()[keep]
    num = (a - b).norm(dim = -1)
    den = b.norm(dim = -1).clamp_min(1e-6)
    rel = num / den
    return float(rel.mean()), float(rel.max())


def _post_hook(module, args, kwargs, output):
    global ENGAGED_COUNT, LAST_VERIFY
    state = getattr(module, _STATE, None)
    if state is None or kwargs.get("prefix_seg_info") is None or not state.pending:
        return None
    layout, original_kwargs = state.pending.pop()
    hidden = _hidden_of(output)  # [1, T, H]
    if not (isinstance(hidden, torch.Tensor) and hidden.dim() == 3 and hidden.shape[0] == 1):
        raise RuntimeError(
            f"Unsloth: DPO prefix share expected [1, T, H] hidden states, got {type(hidden).__name__}."
        )
    full = _gather(hidden, layout)

    if _verify_on() and not getattr(module, _VERIFIED, False):
        # First shared forward of this backbone: the unshared forward must agree on every real token.
        setattr(module, _STATE, None)  # run the plain forward through the hooks untouched
        try:
            with torch.no_grad():
                reference = _hidden_of(module(**original_kwargs))
        finally:
            setattr(module, _STATE, state)
        keep = original_kwargs["attention_mask"].to(torch.bool)
        mean_err, max_err = _token_rel_err(full.detach(), reference, keep)
        LAST_VERIFY = {"mean_rel": mean_err, "max_rel": max_err, "tok_r": layout.tok_r}
        _log(f"verify mean_rel={mean_err:.3e} max_rel={max_err:.3e} tok_r={layout.tok_r:.3f}")
        if not (mean_err <= _VERIFY_MEAN_TOL and max_err <= _VERIFY_MAX_TOL):
            setattr(module, _UNSAFE, True)
            print(
                "Unsloth: DPO prefix share is off for this model: the shared-prefix forward "
                f"disagrees with the unshared one (mean {mean_err:.3e}, max {max_err:.3e}).",
                flush = True,
            )
            # The unshared forward again, in the caller's grad mode, is the result.
            setattr(module, _STATE, None)
            try:
                return module(**original_kwargs)
            finally:
                setattr(module, _STATE, state)
        setattr(module, _VERIFIED, True)
    ENGAGED_COUNT += 1
    return _with_hidden(output, full, layout)


@contextlib.contextmanager
def preference_prefix_share(trainer, *models):
    """Hook the backbones of ``models`` (and the trainer's policy / reference) for the duration."""
    if prefix_share_mode() == "off" or _cp_active():
        yield
        return
    if getattr(trainer, _ACTIVE, False):
        yield  # nested: the outer call already hooked everything
        return
    if getattr(trainer, "is_encoder_decoder", False) or getattr(trainer, "is_vision_model", False):
        yield
        return
    if getattr(getattr(trainer, "args", None), "padding_free", False):
        yield
        return
    handles = []
    seen = set()
    candidates = list(models) + [
        getattr(trainer, "model", None),
        getattr(trainer, "ref_model", None),
    ]
    try:
        for model in candidates:
            if not isinstance(model, torch.nn.Module):
                continue
            causal_lm = _unwrap(trainer, model)
            if not isinstance(causal_lm, torch.nn.Module) or id(causal_lm) in seen:
                continue
            seen.add(id(causal_lm))
            try:
                device = next(causal_lm.parameters()).device
            except StopIteration:
                continue
            if not _model_eligible(causal_lm, device):
                continue
            backbone = _backbone_of(causal_lm)
            if getattr(backbone, _UNSAFE, False) or getattr(backbone, _STATE, None) is not None:
                continue
            if _distributed_world() > 1 and not getattr(backbone, _VERIFIED, False):
                # A rank-local verify forward would desynchronise sharded collectives.
                if _verify_on():
                    setattr(backbone, _VERIFIED, True)
            setattr(backbone, _STATE, _HookState(causal_lm))
            handles.append(
                (
                    backbone,
                    backbone.register_forward_pre_hook(_pre_hook, with_kwargs = True),
                    backbone.register_forward_hook(_post_hook, with_kwargs = True),
                )
            )
        setattr(trainer, _ACTIVE, True)
        yield
    finally:
        try:
            setattr(trainer, _ACTIVE, False)
        except Exception:
            pass
        for backbone, pre, post in handles:
            pre.remove()
            post.remove()
            try:
                setattr(backbone, _STATE, None)
            except Exception:
                pass


def _wrap_method(function):
    if getattr(function, "_unsloth_dpo_prefix_share", False):
        return function

    @functools.wraps(function)
    def wrapped(self, model, *args, **kwargs):
        from ..utils.prefix_grouper_kernel import PrefixGrouperCompileError
        try:
            with preference_prefix_share(self, model):
                return function(self, model, *args, **kwargs)
        except PrefixGrouperCompileError:
            # The flex kernel failed its first-call check inside the forward (nothing backpropagated
            # yet); compile_check_failed() now declines every backbone, so this rerun is unshared.
            return function(self, model, *args, **kwargs)

    wrapped._unsloth_dpo_prefix_share = True
    return wrapped


def wrap_preference_trainer(trainer_cls):
    """Scope the shared-prefix forward to the preference forwards of ``trainer_cls``:
    ``_compute_loss`` / ``compute_ref_log_probs`` (TRL >= 1.0 DPO) and ``concatenated_forward``
    (TRL < 1.0 DPO, ORPO, CPO). Returns the names it wrapped."""
    wrapped = []
    for name in ("_compute_loss", "compute_ref_log_probs", "concatenated_forward"):
        function = trainer_cls.__dict__.get(name)
        if function is None:
            for klass in trainer_cls.__mro__[1:]:
                if name in klass.__dict__:
                    function = klass.__dict__[name]
                    break
        if function is None or not callable(function):
            continue
        try:
            params = list(inspect.signature(function).parameters)
        except (TypeError, ValueError):
            continue
        # Only methods whose first argument is the model being run.
        if len(params) < 2 or params[1] != "model":
            continue
        setattr(trainer_cls, name, _wrap_method(function))
        wrapped.append(name)
    return wrapped
