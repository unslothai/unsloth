# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""For one generate() call, fold each 16-bit LoRA delta into its base weight in place (unsloth_zoo
merged_rollout_exact: bit-exact restore, checksummed), so decode runs one GEMM per projection; undone in a
finally block. fold_refusal() lists when it stays off. UNSLOTH_LORA_FOLD_GENERATE=0 disables it."""

import contextlib
import functools
import os
import threading
import time

import torch

__all__ = [
    "lora_fold_generate",
    "requested_new_tokens",
    "fold_refusal",
    "STATS",
    "stats",
    "reset_stats",
]

STATS = {"engaged": 0, "restored": 0, "refused": {}, "last": None}
_LOCK = threading.RLock()
# generate() calls in flight: only a call that starts alone folds.
_ACTIVE = 0
# Set after a failed restore (weights may be corrupted): never fold again.
_POISONED = None
_STREAMS = {}
# Thread whose call may fold, until it refuses or restores: other threads wait (weights in flux).
_OWNER = None
_RESTORED = threading.Condition(_LOCK)
_POISON_ATTR = "_unsloth_lora_fold_poisoned"


def stats():
    with _LOCK:
        return {**STATS, "refused": dict(STATS["refused"])}


def reset_stats():
    with _LOCK:
        STATS["engaged"] = STATS["restored"] = 0
        STATS["refused"] = {}
        STATS["last"] = None


def _refuse(reason):
    with _LOCK:
        STATS["refused"][reason] = STATS["refused"].get(reason, 0) + 1
    return None


# Break-even on L4 1B LoRA: 5-16 tokens eager, 43-216 with decode CUDA graphs.
_DEFAULT_MIN_TOKENS = 128
_DEFAULT_MIN_TOKENS_GRAPHS = 512


def _min_tokens():
    value = os.environ.get("UNSLOTH_LORA_FOLD_GENERATE_MIN_TOKENS")
    if value is not None:
        try:
            return int(value)
        except ValueError:
            pass
    return _DEFAULT_MIN_TOKENS_GRAPHS if _graphs_enabled() else _DEFAULT_MIN_TOKENS


def _graphs_enabled():
    try:
        from ._decode_graphs import enabled
        return bool(enabled())
    except Exception:
        return False


def _exact_api():
    from unsloth_zoo import merged_rollout_exact
    return merged_rollout_exact


def requested_new_tokens(
    model,
    args = (),
    kwargs = None,
):
    """Upper bound on the tokens this generate() call decodes, as transformers resolves it."""
    kwargs = kwargs or {}
    config = kwargs.get("generation_config") or getattr(model, "generation_config", None)
    n = kwargs.get("max_new_tokens", None)
    if n is None and "max_length" not in kwargs:
        n = getattr(config, "max_new_tokens", None)
    if n is not None:
        return int(n)
    max_length = kwargs.get("max_length", getattr(config, "max_length", None))
    if max_length is None:
        return 0
    ids = kwargs.get("input_ids", kwargs.get("inputs", args[0] if len(args) else None))
    prompt = ids.shape[-1] if isinstance(ids, torch.Tensor) and ids.dim() >= 1 else 0
    return int(max_length) - prompt


def _model_chain(model):
    seen, m = [], model
    while m is not None and all(m is not x for x in seen) and len(seen) < 8:
        seen.append(m)
        m = getattr(m, "base_model", None) if getattr(m, "base_model", None) is not m else None
        if m is None:
            m = getattr(seen[-1], "model", None)
    return seen


def _is_dtensor(t):
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        return False
    return isinstance(t, DTensor)


def _distributed(model):
    if os.environ.get("ACCELERATE_USE_FSDP", "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    if os.environ.get("FSDP_VERSION", "").strip() not in ("", "0"):
        return True
    for m in _model_chain(model):
        if getattr(m, "_unsloth_tp", None) or getattr(m, "_unsloth_pp", None):
            return True
    for m in model.modules():
        for cls in type(m).__mro__:
            if cls.__name__ in (
                "FullyShardedDataParallel",
                "FSDPModule",
                "DistributedDataParallel",
            ):
                return True
    return False


def fold_refusal(model):
    """(None, targets) when the model can be folded, else (reason, None). targets: (module, adapter, W)."""
    if _POISONED is not None:
        return "poisoned", None
    for m in _model_chain(model):
        if getattr(m, "vllm_engine", None) is not None:
            return "vllm", None
    if _distributed(model):
        return "distributed", None
    # Merged experts push grouped MoE onto its per-expert loop.
    for m in _model_chain(model):
        config = getattr(m, "config", None)
        text = getattr(config, "text_config", None) or config
        if any(
            getattr(text, k, None) for k in ("num_experts", "num_local_experts", "n_routed_experts")
        ):
            return "moe", None
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.tuners.lora import LoraLayer, Linear as LoraLinear
    except Exception:
        return "no_peft", None
    devices = set()
    for m in _model_chain(model):
        device_map = getattr(m, "hf_device_map", None)
        if isinstance(device_map, dict) and len(set(map(str, device_map.values()))) > 1:
            return "multi_device", None
    targets = []
    for module in model.modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        if not isinstance(module, LoraLayer):
            return "non_lora_adapter", None
        if module.disable_adapters:
            return "adapters_disabled", None
        if len(getattr(module, "merged_adapters", None) or ()) != 0:
            return "already_merged", None
        active = module.active_adapters
        if len(active) != 1:
            return "multiple_adapters", None
        adapter = active[0]
        if adapter in getattr(module, "lora_embedding_A", {}):
            return "unsupported_layer", None
        if adapter not in module.lora_A:
            # Not on this layer: PEFT runs its base layer alone, folded or not.
            continue
        if type(module) is not LoraLinear:
            return "unsupported_layer", None
        if getattr(module, "use_dora", {}).get(adapter, False) or adapter in (
            getattr(module, "lora_variant", None) or {}
        ):
            return "dora", None
        if getattr(module, "lora_bias", {}).get(adapter, False):
            return "lora_bias", None
        if getattr(module, "fan_in_fan_out", False):
            return "unsupported_layer", None
        dropout = module.lora_dropout[adapter]
        if (
            not isinstance(dropout, torch.nn.Identity)
            and dropout.training
            and getattr(dropout, "p", 0) > 0
        ):
            return "dropout_active", None
        scale = module.scaling.get(adapter, None)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            return "scaling", None
        base = module.get_base_layer()
        W = getattr(base, "weight", None)
        A = module.lora_A[adapter]
        B = module.lora_B[adapter]
        # Exact nn.Linear: subclasses (QAT FakeQuantizedLinear, custom forwards) compute something else.
        if any(type(m) is not torch.nn.Linear for m in (base, A, B)):
            return "unsupported_layer", None
        if A.bias is not None or B.bias is not None:
            return "lora_bias", None
        # QAT computes fake_quant(W) + B @ A; a fold would give fake_quant(W + B @ A).
        if any(
            getattr(m, "weight_fake_quantizer", None) is not None
            or getattr(m, "activation_fake_quantizer", None) is not None
            for m in (base, A, B)
        ):
            return "qat", None
        # PEFT generate(adapter_names=...) hooks: PEFT refuses them on merged layers.
        if any(
            isinstance(h, functools.partial)
            and getattr(h.func, "__name__", "") == "_adapter_names_pre_forward_hook"
            for h in module._forward_pre_hooks.values()
        ):
            return "mixed_adapters", None
        if any(_is_dtensor(t) or hasattr(t, "ds_id") for t in (W, A.weight, B.weight)):
            return "distributed", None
        # Params4bit / Int8Params / torchao / fp8 tensors are Parameter subclasses or other dtypes.
        if (
            type(W) is not torch.nn.Parameter
            or hasattr(W, "quant_state")
            or hasattr(base, "quant_state")
        ):
            return "quantized", None
        if W.dtype not in (torch.bfloat16, torch.float16):
            return (
                "quantized" if not W.is_floating_point() or W.element_size() < 2 else "not_16bit",
                None,
            )
        if any(type(t) is not torch.nn.Parameter for t in (A.weight, B.weight)):
            return "quantized", None
        if any(t.device.type == "meta" for t in (W, A.weight, B.weight)):
            return "offloaded", None
        if A.weight.device != W.device or B.weight.device != W.device:
            return "multi_device", None
        if W.requires_grad:
            return "base_trainable", None
        if (
            W.dim() != 2
            or A.weight.dim() != 2
            or B.weight.dim() != 2
            or A.weight.shape[1] != W.shape[1]
            or B.weight.shape[0] != W.shape[0]
            or B.weight.shape[1] != A.weight.shape[0]
        ):
            return "unsupported_layer", None
        devices.add(W.device)
        targets.append((module, adapter, W))
    if not targets:
        return "no_lora", None
    if len(devices) != 1:
        return "multi_device", None
    # A weight tied to another parameter (lm_head / embeddings) would change the other user too.
    owners = {}
    for p in _all_params(model):
        key = p.untyped_storage().data_ptr()
        owners[key] = owners.get(key, 0) + 1
    seen = set()
    for _, _, W in targets:
        key = W.untyped_storage().data_ptr()
        if owners.get(key, 0) > 1 or key in seen:
            return "tied_weight", None
        seen.add(key)
    api = _exact_api()
    for device in devices:
        for dtype in {W.dtype for _, _, W in targets}:
            if not api.exact_merge_available(device, dtype):
                return "exact_unavailable", None
    return None, targets


def _all_params(model):
    for _, p in model.named_parameters(remove_duplicate = False):
        yield p
    for _, b in model.named_buffers(remove_duplicate = False):
        if b is not None and b.numel() > 0:
            yield b


def _side_stream(device):
    if device.type != "cuda":
        return None
    key = device.index if device.index is not None else torch.cuda.current_device()
    if key not in _STREAMS:
        _STREAMS[key] = torch.cuda.Stream(device = key)
    return _STREAMS[key]


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextlib.contextmanager
def lora_fold_generate(
    model,
    args = (),
    kwargs = None,
):
    """Fold every LoRA adapter into its 16-bit base weight for the body, then restore it bit for bit."""
    global _ACTIVE
    me = threading.get_ident()
    with _RESTORED:
        while _OWNER is not None and _OWNER != me:
            _RESTORED.wait()
        first = _ACTIVE == 0
        _ACTIVE += 1
        if first:
            _set_owner(me)
    try:
        poisoned = getattr(model, _POISON_ATTR, None)
        if poisoned is not None:
            raise RuntimeError(
                "Unsloth: an earlier LoRA-folded generate() could not restore this model's base weights "
                f"({poisoned}). Reload the model."
            )
        reason, targets = None, None
        if os.environ.get("UNSLOTH_LORA_FOLD_GENERATE", "1") == "0":
            reason = "kill_switch"
        elif not first:
            reason = "nested"
        elif torch.is_grad_enabled():
            reason = "grad_enabled"
        elif requested_new_tokens(model, args, kwargs) < _min_tokens():
            reason = "short"
        elif (kwargs or {}).get("adapter_names", None) is not None:
            reason = "mixed_adapters"
        else:
            try:
                reason, targets = fold_refusal(model)
            except Exception:
                reason = "check_failed"
        if reason is not None:
            if first:
                _release(me)
            _refuse(reason)
            yield False
            return
        with _scope(model, targets) as engaged:
            if not engaged:
                _release(me)
            yield engaged
    finally:
        with _RESTORED:
            _ACTIVE -= 1
        # Only the owning call releases (a nested call must not end the outer fold's).
        if first:
            _release(me)


def _set_owner(owner):
    global _OWNER
    _OWNER = owner


def _release(me):
    with _RESTORED:
        if _OWNER == me:
            _set_owner(None)
            _RESTORED.notify_all()


@contextlib.contextmanager
def _scope(model, entries):
    api = _exact_api()
    device = entries[0][2].device
    targets, flagged, stream = [], [], None
    t0 = time.perf_counter()
    try:
        stream = _side_stream(device)
        # A / B cast once and held, so the unfold recomputes the same delta. W.data: no version bump,
        # so an autograd graph saved before the call still backpropagates through the restored bits.
        for module, adapter, W in entries:
            targets.append(
                api.ExactFoldTarget(
                    W.data.unsqueeze(0),
                    module.lora_A[adapter].weight.data.to(W.dtype).unsqueeze(0),
                    module.lora_B[adapter].weight.data.to(W.dtype).unsqueeze(0),
                    module.scaling[adapter],
                )
            )
        info = api.exact_fold(targets, stream = stream)
        for module, adapter, _ in entries:
            # Merged = base layer only. Mutated in place so compiled guards see the same list.
            module.merged_adapters.append(adapter)
            flagged.append(module)
        _sync(device)
        t1 = time.perf_counter()
        _record_fold(info, targets, t1 - t0)
    except Exception:
        # Nothing ran on folded weights yet: undo what was folded and take the normal LoRA path.
        _restore(model, api, targets, flagged, stream)
        _refuse("fold_failed")
        yield False
        return
    except BaseException:
        _restore(model, api, targets, flagged, stream)
        raise
    t2 = None
    try:
        yield True
        t2 = time.perf_counter()
    finally:
        _restore(model, api, targets, flagged, stream)
        _sync(device)
        with _LOCK:
            STATS["restored"] += 1
            if t2 is not None:
                STATS["last"]["unfold_s"] = time.perf_counter() - t2


def _record_fold(info, targets, fold_s):
    # Pinned host RAM of the fix-up store, held for the whole call.
    host_bytes, pinned = 0, None
    for t in targets:
        for buf in (getattr(t, "host_idx", None), getattr(t, "host_val", None)):
            if isinstance(buf, torch.Tensor):
                host_bytes += buf.numel() * buf.element_size()
                pinned = buf.is_pinned() if pinned is None else (pinned and buf.is_pinned())
    with _LOCK:
        STATS["engaged"] += 1
        STATS["last"] = {
            **info,
            "fold_s": fold_s,
            "unfold_s": None,
            "targets": len(targets),
            "host_store_bytes": host_bytes,
            "host_store_pinned": pinned,
        }


def _restore(model, api, targets, flagged, stream):
    global _POISONED
    for module in flagged:
        module.merged_adapters.clear()
    if not targets:
        return
    try:
        api.exact_unfold(targets, stream = stream)
    except BaseException as error:
        # No more folds; this model refuses to generate on its unknown weights.
        _POISONED = repr(error)
        try:
            setattr(model, _POISON_ATTR, _POISONED)
        except Exception:
            pass
        raise RuntimeError(
            "Unsloth: restoring the base weights after a LoRA-folded generate() failed; the weights may "
            "still hold the folded adapter. Reload the model. Set UNSLOTH_LORA_FOLD_GENERATE=0 to avoid it."
        ) from error
