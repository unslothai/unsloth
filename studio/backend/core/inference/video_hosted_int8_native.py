# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Serve a hosted int8 video denoiser to an OFFLOADED load as torchao-free native W8A8 layers.

torchao tensors do not survive the offload hooks' ``Module.to()``, but the codes and scales are plain tensors
that ``NativeWeightOnlyLinear`` runs with the same ``torch._int_mm`` maths. NOT the runtime offload build
(ConvRot g256, ``amax / 127``): its codes differ. Kill switch: ``UNSLOTH_VIDEO_HOSTED_INT8_OFFLOAD=0``.
"""

from __future__ import annotations

import os
from typing import Any, Optional

HOSTED_INT8_OFFLOAD_ENV = "UNSLOTH_VIDEO_HOSTED_INT8_OFFLOAD"
# Families whose hosted int8 pair is proven to load this way; the resident path serves the same files.
HOSTED_INT8_OFFLOAD_FAMILIES = frozenset({"wan2.2-t2v-a14b"})
_ACT_QUANT_NAME = "_int8_symm_per_token_reduced_range_quant"


def hosted_int8_offload_enabled() -> bool:
    raw = (os.environ.get(HOSTED_INT8_OFFLOAD_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def hosted_int8_offload_eligible(fam: Any, scheme: Optional[str], target: Any) -> bool:
    """Whether an offloaded load of ``fam`` on ``target`` may serve the hosted ``scheme`` file natively. Never raises."""
    try:
        if not hosted_int8_offload_enabled():
            return False
        if str(scheme or "").strip().lower() != "int8":
            return False
        name = getattr(fam, "name", None)
        if name not in HOSTED_INT8_OFFLOAD_FAMILIES or getattr(fam, "modular_workflow", None):
            return False
        from .diffusion_transformer_quant import TQ_INT8, native_int8_act, native_quant_scheme

        # W8A8 only: the file's activations are int8, a weight-only native layer would compute something else.
        return native_quant_scheme(target, TQ_INT8, family = name, offload = True) == TQ_INT8 and bool(
            native_int8_act(target)
        )
    except Exception:  # noqa: BLE001 -- unanswerable: keep declining the file
        return False


def native_cache_covers(fam: Any, target: Any, base: Optional[str]) -> bool:
    """Whether this machine's native cache holds the runtime offload build of EVERY expert. Never raises."""
    try:
        from .diffusion_native_cache import lookup, native_spec
        from .diffusion_transformer_quant import TQ_INT8, native_int8_act
        from .video_denoiser_prequant import denoiser_components

        spec = native_spec(
            TQ_INT8, getattr(fam, "name", None), act_int8 = bool(native_int8_act(target))
        )
        return all(
            lookup(fam, spec, base_repo = base, source_base = base, component = c) is not None
            for c in denoiser_components(fam)
        )
    except Exception:  # noqa: BLE001
        return False


def hosted_int8_offload_ok(
    fam: Any, scheme: Optional[str], target: Any, base: Optional[str]
) -> bool:
    return hosted_int8_offload_eligible(fam, scheme, target) and not native_cache_covers(
        fam, target, base
    )


def _is_zero(t: Any) -> bool:
    import torch
    return t is None or (type(t) is torch.Tensor and not bool(torch.any(t != 0)))


def int8_weight_parts(name: str, value: Any) -> Optional[tuple]:
    """``(codes [n, k] int8, scale [n], dtype)`` of a torchao v1/v2 W8A8 weight, None for a plain tensor.
    RAISES on any other layout so the load falls back to dense."""
    import torch

    kind = type(value).__name__
    if type(value) in (torch.Tensor, torch.nn.Parameter):
        return None

    def bad(why: str) -> ValueError:
        return ValueError(f"hosted int8 weight {name!r} ({kind}): {why}")

    if kind == "Int8Tensor":
        qdata, scale = getattr(value, "qdata", None), getattr(value, "scale", None)
        if not _is_zero(getattr(value, "zero_point", None)):
            raise bad("asymmetric")
        if getattr(value, "act_quant_kwargs", None) is None:
            raise bad("weight-only (no activation quant)")
        granularity = getattr(value.act_quant_kwargs, "granularity", None)
        if type(granularity).__name__ != "PerRow":
            raise bad(f"activation granularity {granularity!r} is not per-token")
        for attr in ("act_quant_scale", "act_quant_zero_point", "act_pre_scale"):
            if getattr(value, attr, None) is not None:
                raise bad(f"{attr} is set")
        block = list(getattr(value, "block_size", ()) or ())
    elif kind == "LinearActivationQuantizedTensor":
        act = getattr(value, "input_quant_func", None)
        if getattr(act, "__name__", None) != _ACT_QUANT_NAME or getattr(
            value, "quant_kwargs", None
        ):
            raise bad("activation quant is not the per-token int8 one")
        aqt = getattr(value, "original_weight_tensor", None)
        impl = getattr(aqt, "tensor_impl", None)
        if type(getattr(impl, "_layout", None)).__name__ != "PlainLayout":
            raise bad("inner layout is not PlainLayout")
        qdata, scale = getattr(impl, "int_data", None), getattr(impl, "scale", None)
        if not _is_zero(getattr(impl, "zero_point", None)):
            raise bad("asymmetric")
        block = list(getattr(aqt, "block_size", ()) or ())
    else:
        raise bad("not an int8 W8A8 weight")
    if type(qdata) is not torch.Tensor or qdata.dtype != torch.int8 or qdata.dim() != 2:
        raise bad("codes are not a 2-D int8 tensor")
    n, k = qdata.shape
    if tuple(value.shape) != (n, k) or block != [1, k]:
        raise bad(f"block size {block} is not per-row for {(n, k)}")
    if type(scale) is not torch.Tensor or not scale.is_floating_point() or scale.numel() != n:
        raise bad("scale is not one float per output row")
    return qdata, scale.reshape(n), value.dtype


def native_state_from_hosted(model: Any, state_dict: dict) -> tuple[dict, int]:
    """Swap every int8 Linear of the (meta) ``model`` for a native W8A8 twin holding the file's codes; returns the
    state dict ``model`` now loads strictly, and the number of layers swapped. Raises on any mismatch."""
    import torch

    from .diffusion_native_quant import NATIVE_INT8, native_linear_class

    cls = native_linear_class()
    out: dict = {}
    swapped = 0
    for key, value in state_dict.items():
        parts = int8_weight_parts(key, value)
        if parts is None:
            out[key] = value
            continue
        if not key.endswith(".weight"):
            raise ValueError(f"hosted int8 tensor {key!r} is not a Linear weight")
        fqn = key[: -len(".weight")]
        codes, scale, dtype = parts
        parent_name, _, leaf = fqn.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        linear = getattr(parent, leaf)
        if not isinstance(linear, torch.nn.Linear) or tuple(linear.weight.shape) != tuple(
            codes.shape
        ):
            raise ValueError(f"hosted int8 weight {key!r} does not match the model's {fqn}")
        stand_in = torch.nn.Linear(
            linear.in_features,
            linear.out_features,
            bias = linear.bias is not None,
            device = "meta",
            dtype = dtype,
        )
        layer = cls(stand_in, NATIVE_INT8, act_int8 = True, rot_group = 0, codes = codes, scale = scale)
        setattr(parent, leaf, layer)
        out[f"{fqn}.weight_q"] = layer.weight_q
        out[f"{fqn}.weight_scale"] = layer.weight_scale
        swapped += 1
    return out, swapped


def load_hosted_int8_native(
    transformer_cls: Any,
    base: str,
    source: Any,
    *,
    component: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_files_only: bool = False,
    logger: Any = None,
) -> Optional[Any]:
    """The hosted int8 ``component`` as native W8A8 layers on the CPU, or None (logged). Same checks as the resident
    reader (header refusal, metadata, payload fingerprint); a rotated artifact is refused. Never raises."""
    try:
        from . import diffusion_prequant as pq
        from .diffusion_convrot import declares_rotation
        from .diffusion_transformer_quant import DEFAULT_MIN_LINEAR_FEATURES

        scheme = "int8"
        # Hosted files only: a local path source has its own allowlist and digest checks in the torchao reader.
        if getattr(source, "kind", None) != "repo":
            return None
        refusal = pq.hosted_prequant_refusal(
            source,
            scheme,
            base = base,
            min_features = DEFAULT_MIN_LINEAR_FEATURES,
            component = component,
            hf_token = hf_token,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
            logger = logger,
        )
        if refusal is not None:
            pq._warn(logger, f"{scheme}:native", ValueError(refusal))
            return None
        path = pq._resolve_checkpoint_path(
            source,
            hf_token,
            cache_dir,
            local_files_only = local_files_only,
            scheme = scheme,
            logger = logger,
        )
        if path is None:
            return None
        ckpt = pq._read_prequant_for(path, "cpu", logger)
        if not pq._validate_checkpoint(
            ckpt,
            scheme,
            base,
            logger,
            min_features = DEFAULT_MIN_LINEAR_FEATURES,
            component = component,
        ):
            return None
        metadata = ckpt.get("metadata") or {}
        if declares_rotation(metadata):
            pq._warn(
                logger,
                f"{scheme}:native",
                ValueError("a rotated checkpoint is not served natively"),
            )
            return None
        state_dict = ckpt["state_dict"]
        if not pq._verify_packed_fingerprint(state_dict, metadata, logger = logger, path = path):
            return None
        config = pq._load_transformer_config(
            transformer_cls,
            base,
            hf_token,
            cache_dir,
            path,
            component,
            local_files_only = local_files_only,
        )
        from accelerate import init_empty_weights

        # Buffers stay real: non-persistent ones (rope tables) are built by __init__, as the native cache does.
        with init_empty_weights(include_buffers = False):
            model = transformer_cls.from_config(config)
        native_state, swapped = native_state_from_hosted(model, state_dict)
        if not swapped:
            raise ValueError("the checkpoint holds no int8 weight")
        model.load_state_dict(native_state, strict = True, assign = True)
        del native_state, state_dict, ckpt
        if any(t.device.type == "meta" for t in (*model.parameters(), *model.buffers())):
            raise ValueError("the native build left meta tensors")
        model.eval()
        model._unsloth_runtime_quant = scheme
        model._unsloth_runtime_quant_info = {"scheme": scheme, "native": True, "hosted": str(path)}
        model._unsloth_prequant_path = path
        if logger is not None:
            logger.info(
                "diffusion.prequant: loaded int8 checkpoint (%s) for %s as %d torchao-free W8A8 "
                "layers on cpu (offloaded DiT)",
                getattr(source, "kind", None),
                component,
                swapped,
            )
        return model
    except Exception as exc:  # noqa: BLE001 - the caller falls back to the dense experts
        if logger is not None:
            logger.warning("video.denoiser_prequant: native %s read failed: %s", component, exc)
        return None
