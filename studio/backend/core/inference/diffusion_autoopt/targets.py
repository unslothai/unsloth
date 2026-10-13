# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which modules of a pipeline are denoisers, and which pipeline actually runs a request.

Two resolvers with different contracts:

* :func:`denoiser_dits` / :func:`denoiser_unet` / :func:`guarded_dits` / :func:`hooked_denoisers` are the exact
  attribute contracts the speed, attention, graph and offload layers have always used (one copy here instead of one
  per module). Every primitive keeps reaching precisely the modules it reached before.
* :func:`denoisers` resolves BY TYPE for the optimizer's profile / audit and for a family with no data: diffusers
  ``ModelMixin`` components that declare or expose repeated transformer blocks, honouring ``family.denoiser_attr``,
  an MoE ``transformer_2`` and named partitions (``transformer_ref``). It never feeds a primitive of a registered
  family, so it cannot widen what a known load touches.

:func:`runner_for` names the pipeline that executes a call (from_pipe workflows, the I2V class swap, LTX-2.5's
condition pipe) so call-time checks (``runtime.py``) judge the runner, not the base pipe.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

# The DiT attributes every denoise loop may run, in the order the speed / attention layers have always visited them.
DIT_ATTRS: tuple[str, ...] = ("transformer", "transformer_2", "unconditional_transformer")
# Attributes an offload hook may move (the graph layer's stale-pointer check).
HOOKED_ATTRS: tuple[str, ...] = DIT_ATTRS + ("unet",)
# Named partitions a modular workflow compiles through a view (MiniMax-H3's reference DiT).
PARTITION_ATTRS: tuple[str, ...] = ("transformer_ref",)

# U-Net denoisers ship no ``_repeated_blocks``, so the regional compile cannot reach them; these classes get a
# WHOLE-module static compile (diffusion_speed).
UNET_WHOLE_COMPILE: frozenset[str] = frozenset({"UNet2DConditionModel"})

# Component names that are never a denoiser even when they are a ModelMixin with repeated blocks.
_NOT_DENOISERS = (
    "vae",
    "controlnet",
    "text_encoder",
    "image_encoder",
    "audio",
    "vocoder",
    "upsampler",
    "connector",
)


def _unique(modules: Iterable[Any]) -> list:
    out: list = []
    for m in modules:
        if m is not None and not any(m is seen for seen in out):
            out.append(m)
    return out


def denoiser_dits(pipe: Any) -> list:
    """Every DiT the denoise loop runs: ``transformer`` plus a second expert some families carry (Ideogram's
    ``unconditional_transformer``, an MoE ``transformer_2``)."""
    dits: list = []
    for attr in DIT_ATTRS:
        m = getattr(pipe, attr, None)
        if m is not None and m not in dits:
            dits.append(m)
    return dits


def denoiser_unet(pipe: Any) -> Any:
    """The pipe's U-Net when its class is on the whole-compile list, else None."""
    unet = getattr(pipe, "unet", None)
    if unet is not None and type(unet).__name__ in UNET_WHOLE_COMPILE:
        return unet
    return None


def guarded_dits(pipe: Any) -> list:
    """:func:`denoiser_dits` plus a modular workflow's named partition (compiled through a view)."""
    dits = denoiser_dits(pipe)
    for attr in PARTITION_ATTRS:
        extra = getattr(pipe, attr, None)
        if extra is not None and extra not in dits:
            dits.append(extra)
    return dits


def hooked_denoisers(pipe: Any) -> list:
    """The modules whose offload hooks make a captured graph replay stale pointers."""
    return [m for m in (getattr(pipe, a, None) for a in HOOKED_ATTRS) if m is not None]


# ----------------------------------------------------------------------------------------------- by-type resolver
def _is_model_mixin(module: Any) -> bool:
    try:
        from diffusers.models.modeling_utils import ModelMixin  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - no diffusers: duck-type below
        ModelMixin = None  # type: ignore[assignment]  # noqa: N806
    if ModelMixin is not None and isinstance(module, ModelMixin):
        return True
    return callable(getattr(module, "compile_repeated_blocks", None))


def _declares_blocks(module: Any) -> bool:
    if getattr(type(module), "_repeated_blocks", None):
        return True
    if getattr(module, "_repeated_blocks", None):
        return True
    return bool(getattr(module, "_no_split_modules", None))


def _components(pipe: Any) -> dict:
    comps = getattr(pipe, "components", None)
    if isinstance(comps, dict):
        return comps
    return {}


def denoisers(pipe: Any, family: Any = None) -> list:
    """Denoiser modules of ``pipe`` resolved by type, primary first.

    Order: ``family.denoiser_attr`` (``unet`` for SDXL), the standard DiT attributes, named partitions, then any other
    ``ModelMixin`` component that declares repeated blocks and is not a VAE / encoder / ControlNet."""
    found: list = []
    attr = getattr(family, "denoiser_attr", None)
    if isinstance(attr, str) and attr:
        found.append(getattr(pipe, attr, None))
    found.extend(getattr(pipe, a, None) for a in DIT_ATTRS + PARTITION_ATTRS)
    if getattr(pipe, "unet", None) is not None:
        found.append(pipe.unet)
    for name, module in _components(pipe).items():
        if module is None or any(tag in str(name).lower() for tag in _NOT_DENOISERS):
            continue
        if _is_model_mixin(module) and _declares_blocks(module):
            found.append(module)
    return _unique(found)


def discovered_only(pipe: Any, family: Any = None) -> list:
    """Denoisers the by-type resolver finds that the attribute contracts above miss (what the generic path adds)."""
    known = guarded_dits(pipe) + [m for m in [getattr(pipe, "unet", None)] if m is not None]
    return [m for m in denoisers(pipe, family) if not any(m is k for k in known)]


# ------------------------------------------------------------------------------------------------------- runners
def runner_for(base: Any, runner: Optional[Any] = None) -> Any:
    """The pipeline that executes a call: ``runner`` when a workflow swapped one in (from_pipe img2img / inpaint,
    the I2V class swap, LTX-2.5's condition pipe), else ``base``."""
    return runner if runner is not None else base
