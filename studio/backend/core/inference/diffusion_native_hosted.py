# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An offloaded int8 denoiser read from the hosted ConvRot int8 checkpoint as native Linears.

Offload hooks cannot move torchao weights, so an offloaded load used to fetch the dense bf16 shards and quantise them
at load (LTX-2.5: 36 GB staged, ~100 s of build, then a 23 GB cache write). The hosted file already holds per-row int8
codes and float32 scales of the same ConvRot-rotated weights, which is exactly what ``NativeWeightOnlyLinear`` stores:
this reads them into native Linears with the same layer set, rotation and activation path as the runtime build."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

NATIVE_HOSTED_ENV = "UNSLOTH_VIDEO_NATIVE_HOSTED"
_QDATA = "._weight_qdata"
_SCALE = "._weight_scale"
_ZERO_POINT = "._weight_zero_point"
# Int8Tensor payloads that would change the activation math (static / pre-scaled activations): refused.
_ACT_PAYLOADS = (
    "._weight_act_quant_scale",
    "._weight_act_quant_zero_point",
    "._weight_act_pre_scale",
)


def native_hosted_enabled() -> bool:
    raw = (os.environ.get(NATIVE_HOSTED_ENV) or "").strip().lower()
    return raw not in ("0", "off", "false", "no")


def native_hosted_family(fam: Any) -> bool:
    """Whether ``fam`` opted in (its hosted int8 file is ConvRot safetensors) and the switch is on."""
    return bool(getattr(fam, "native_offload_from_hosted_int8", False)) and native_hosted_enabled()


def _log(logger: Any, level: str, msg: str) -> None:
    if logger is not None:
        getattr(logger, level)("video.native_hosted: %s", msg)


def hosted_metadata(path: str) -> Optional[dict]:
    """The checkpoint's ``unsloth_metadata``, or None. Header only."""
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework = "pt") as handle:
            raw = (handle.metadata() or {}).get("unsloth_metadata")
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001 - unreadable: not usable
        return None


def metadata_refusal(
    meta: Any, spec: dict, *, family: Optional[str], component: str
) -> Optional[str]:
    """Why the hosted file is not the native build ``spec`` describes, or None."""
    from .diffusion_convrot import CONVROT_KIND, ROTATION_FQNS_KEY, ROTATION_GROUP_KEY, ROTATION_KEY

    if not isinstance(meta, dict):
        return "no unsloth metadata"
    if spec.get("scheme") != "int8" or not spec.get("act_int8"):
        return f"the native build is {spec.get('scheme')} (act_int8={spec.get('act_int8')}), not W8A8 int8"
    if meta.get("scheme") != "int8" or meta.get("quant_backend") != "torchao":
        return f"checkpoint scheme {meta.get('scheme')!r} / backend {meta.get('quant_backend')!r}"
    if family is not None and meta.get("family") != family:
        return f"checkpoint family {meta.get('family')!r}, not {family!r}"
    if (meta.get("component") or "transformer") != component:
        return f"checkpoint component {meta.get('component')!r}, not {component!r}"
    if str(meta.get("torch_dtype") or "bfloat16") != "bfloat16":
        return f"checkpoint dtype {meta.get('torch_dtype')!r}"
    if meta.get(ROTATION_KEY) != CONVROT_KIND:
        return f"activation rotation {meta.get(ROTATION_KEY)!r}"
    if int(meta.get(ROTATION_GROUP_KEY) or 0) != int(spec.get("rot_group") or 0):
        return f"rotation group {meta.get(ROTATION_GROUP_KEY)!r}, the native build uses {spec.get('rot_group')!r}"
    if not isinstance(meta.get(ROTATION_FQNS_KEY), (list, tuple)):
        return "no rotation fqns"
    if int(meta.get("min_features") or -1) != int(spec.get("min_features") or -2):
        return f"min_features {meta.get('min_features')!r}"
    excludes = meta.get("exclude_name_tokens")
    if not isinstance(excludes, (list, tuple)):
        return "no exclude tokens"
    if sorted(set(excludes) | {"lora_"}) != list(spec.get("exclude_name_tokens") or ()):
        return "its excluded layers differ from the native build's"
    return None


def load_native_from_hosted(
    path: str,
    transformer_cls: Any,
    config_source: str,
    *,
    spec: dict,
    family: Optional[str],
    component: str = "transformer",
    config_kwargs: Optional[dict] = None,
    logger: Any = None,
) -> Any:
    """The native ``component`` built from hosted ``path``, or None (the caller falls back to the dense build). Never
    raises. Refuses unless the file holds exactly the layers, rotation and activation path the runtime build makes."""
    try:
        import time

        import torch

        t0 = time.perf_counter()
        meta = hosted_metadata(path)
        problem = metadata_refusal(meta, spec, family = family, component = component)
        if problem:
            _log(logger, "info", f"hosted int8 checkpoint not used ({problem})")
            return None
        from accelerate import init_empty_weights
        from safetensors.torch import load_file

        from .diffusion_convrot import ROTATION_FQNS_KEY
        from .diffusion_native_quant import native_linear_class
        from .diffusion_prequant import HostedInt8Parts, _verify_packed_fingerprint
        from .diffusion_transformer_quant import make_filter_fn

        config = transformer_cls.load_config(
            config_source, subfolder = component, **(config_kwargs or {})
        )
        with init_empty_weights(include_buffers = False):
            model = transformer_cls.from_config(config)
        group = int(spec["rot_group"])
        # The runtime build's layer set (its require_bf16 holds: the whole load is bf16, checked above).
        filter_fn = make_filter_fn(int(spec["min_features"]), tuple(spec["exclude_name_tokens"]))
        wanted = {name for name, mod in model.named_modules() if name and filter_fn(mod, name)}
        from .prequant_safetensors import UNSLOTH_ROOT_PREFIX

        state = load_file(str(path))
        # Root-level tensors are stored prefixed (the torchao unflatten splits keys on dots).
        for key in [k for k in state if k.startswith(UNSLOTH_ROOT_PREFIX)]:
            state[key[len(UNSLOTH_ROOT_PREFIX) :]] = state.pop(key)
        quantised = {key[: -len(_QDATA)] for key in state if key.endswith(_QDATA)}
        if quantised != wanted:
            _log(
                logger,
                "info",
                f"hosted int8 checkpoint not used ({len(quantised)} quantised layers, the native build makes "
                f"{len(wanted)}; e.g. {sorted(quantised ^ wanted)[:2]})",
            )
            return None
        if any(key.endswith(_ACT_PAYLOADS) for key in state):
            _log(
                logger,
                "info",
                "hosted int8 checkpoint not used (it carries static activation scales)",
            )
            return None
        modules = dict(model.named_modules())
        rotated = set(meta[ROTATION_FQNS_KEY])
        expect_rotated = {fqn for fqn in wanted if int(modules[fqn].in_features) % group == 0}
        if rotated != expect_rotated:
            _log(
                logger,
                "info",
                "hosted int8 checkpoint not used (its rotated layers differ from the native build's)",
            )
            return None
        parts: dict = {}
        for fqn in quantised:
            codes = state[fqn + _QDATA]
            scale = state[fqn + _SCALE]
            zero = state.get(fqn + _ZERO_POINT)
            lin = modules[fqn]
            shape = (int(lin.out_features), int(lin.in_features))
            if (
                codes.dtype != torch.int8
                or tuple(codes.shape) != shape
                or scale.numel() != shape[0]
                or not scale.is_floating_point()
                or (zero is not None and bool(torch.any(zero != 0)))
            ):
                _log(
                    logger,
                    "info",
                    f"hosted int8 checkpoint not used ({fqn}: unexpected payload layout)",
                )
                return None
            parts[fqn + ".weight"] = HostedInt8Parts(codes, scale, zero)
        # The torchao load checks the same recorded digests: corruption after build must not render.
        if not _verify_packed_fingerprint(parts, meta, logger = logger, path = path):
            return None
        del parts
        cls = native_linear_class()
        for fqn in sorted(quantised):
            lin = modules[fqn]
            parent_name, _, leaf = fqn.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            stand_in = torch.nn.Linear(
                int(lin.in_features),
                int(lin.out_features),
                bias = lin.bias is not None,
                device = "meta",
                dtype = torch.bfloat16,
            )
            codes = state.pop(fqn + _QDATA)
            scale = state.pop(fqn + _SCALE).to(torch.float32).reshape(-1).contiguous()
            state.pop(fqn + _ZERO_POINT, None)
            setattr(
                parent,
                leaf,
                cls(
                    stand_in,
                    "int8",
                    act_int8 = True,
                    rot_group = group if fqn in rotated else 0,
                    codes = codes,
                    scale = scale,
                ),
            )
            state[fqn + ".weight_q"] = codes
            state[fqn + ".weight_scale"] = scale.view(torch.int32)
        from .prequant_safetensors import _unviewed

        # No-copy aliases: a view's base survives offload's ``.data =`` re-point and keeps the whole file mapped.
        state = {key: _unviewed(value) for key, value in state.items()}
        model.load_state_dict(state, strict = True, assign = True)
        del state
        if any(t.device.type == "meta" for t in (*model.parameters(), *model.buffers())):
            _log(logger, "warning", "hosted int8 checkpoint left meta tensors; not used")
            return None
        if any(p.dtype != torch.bfloat16 for p in model.parameters() if p.is_floating_point()):
            _log(logger, "info", "hosted int8 checkpoint not used (a non-bf16 dense tensor)")
            return None
        model.eval()
        try:
            model._unsloth_runtime_quant = "int8"
            model._unsloth_runtime_quant_info = {
                "scheme": "int8",
                "native": True,
                "hosted_path": str(path),
            }
        except Exception:  # noqa: BLE001
            pass
        _log(
            logger,
            "info",
            f"read {len(quantised)} int8 ConvRot layers of {component} from the hosted checkpoint as native "
            f"Linears in {time.perf_counter() - t0:.1f} s (no dense shards, no runtime quantise)",
        )
        return model
    except Exception as exc:  # noqa: BLE001 - an unusable file falls back to the dense build
        _log(logger, "warning", f"hosted int8 read failed: {type(exc).__name__}: {str(exc)[:200]}")
        return None
