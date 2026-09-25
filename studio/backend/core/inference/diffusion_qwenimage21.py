# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qwen-Image-2.1 denoiser forward with the per-step token layout built once (bit-identical).

Kill switch: ``UNSLOTH_DIFFUSION_Q21_FAST_STEP=0``."""

from __future__ import annotations

import ast
import functools
import hashlib
import inspect
import io
import math
import os
import textwrap
import threading
import tokenize
import weakref
from typing import Any, Optional

FAST_STEP_ENV = "UNSLOTH_DIFFUSION_Q21_FAST_STEP"

_MODULE = "diffusers.models.transformers.transformer_qwenimage21"
_CLASS = "QwenImage21Transformer2DModel"

# Stock diffusers sources the fast forward relies on; any drift keeps the stock forward.
_FINGERPRINTS: dict[str, frozenset] = {
    "forward": frozenset({"7c1fd48f75efbe41"}),
    "build_token_metadata": frozenset({"51818c1674e0d406"}),
    "rope_forward": frozenset({"f9e064940341e3ee"}),
    "prefix_segments": frozenset({"27729cffc22b0aa5"}),
}

_LAYOUTS_PER_MODULE = 8
_RECENT_PER_MODULE = 4

_LOCK = threading.Lock()
_INSTALLED: dict = {}


def fast_step_disabled() -> bool:
    return (os.environ.get(FAST_STEP_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _digest(fn: Any) -> Optional[str]:
    """Source text, not ``ast.dump``, whose output changes between Python versions."""
    try:
        src = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
        tree = ast.parse(src)
        comments = [
            tok.start
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type == tokenize.COMMENT
        ]
    except (OSError, TypeError, SyntaxError, ValueError, tokenize.TokenError):
        return None
    lines = src.splitlines()
    for row, col in comments:
        lines[row - 1] = lines[row - 1][:col]
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            for row in range(body[0].lineno, body[0].end_lineno + 1):
                lines[row - 1] = ""
    text = "\n".join(line.rstrip() for line in lines if line.strip())
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def stock_digests(module: Any) -> dict[str, Optional[str]]:
    cls = getattr(module, _CLASS, None)
    rope = getattr(module, "QwenImage21Rope", None)
    return {
        "forward": _digest(vars(cls)["forward"])
        if cls is not None and "forward" in vars(cls)
        else None,
        "build_token_metadata": _digest(getattr(cls, "build_token_metadata", None))
        if cls is not None
        else None,
        "rope_forward": _digest(vars(rope)["forward"])
        if rope is not None and "forward" in vars(rope)
        else None,
        "prefix_segments": _digest(getattr(module, "_qwenimage21_prefix_segments", None)),
    }


def why_unsupported(module: Any) -> Optional[str]:
    got = stock_digests(module)
    for name, want in _FINGERPRINTS.items():
        if got.get(name) not in want:
            return f"{name} differs from the version this was written against ({got.get(name)})"
    return None


class _Layout:
    __slots__ = (
        "repeats",
        "image_pad_mask",
        "total",
        "image_positions",
        "text_positions",
        "rotary_emb",
        "image_ids",
        "target_token_mask",
        "prefix_len",
        "tail_is_image",
        "vlm_row",
        "_segments",
        "_vlm_text",
        "__weakref__",
    )


def _build_layout(model: Any, mod: Any, img_mask: Any, shapes: list, device: Any) -> _Layout:
    import torch

    lay = _Layout()
    lay.repeats = torch.where(img_mask, mod._IMG_TOKENS_PER_SLOT, 1)[0]
    lay.image_pad_mask = torch.repeat_interleave(img_mask[0], lay.repeats)
    lay.total = int(lay.image_pad_mask.shape[0])
    lay.image_positions = lay.image_pad_mask.nonzero(as_tuple = True)[0]
    lay.text_positions = (~lay.image_pad_mask).nonzero(as_tuple = True)[0]
    lay.rotary_emb = model.pos_embed(shapes, lay.image_pad_mask, device = device)
    lay.image_ids, lay.target_token_mask = model.build_token_metadata(lay.image_pad_mask, shapes)
    lay.prefix_len = int((~lay.target_token_mask).sum())
    # Decode steps read rows [prefix_len:] only; all-image tails never need the text projection.
    lay.tail_is_image = bool(lay.image_pad_mask[lay.prefix_len :].all())
    lay.vlm_row = img_mask[0].clone()  # the caller may reuse and rewrite its mask
    lay._segments = None
    lay._vlm_text = {}
    return lay


def _segments(lay: _Layout, mod: Any) -> list:
    if lay._segments is None:
        lay._segments = mod._qwenimage21_prefix_segments(lay.image_ids, lay.prefix_len)
    return lay._segments


def _vlm_text(lay: _Layout, length: int) -> Any:
    idx = lay._vlm_text.get(length)
    if idx is None:
        idx = (~lay.vlm_row[:length]).nonzero(as_tuple = True)[0]
        lay._vlm_text[length] = idx
    return idx


def _mask_version(img_mask: Any) -> Any:
    try:
        return img_mask._version
    except Exception:  # noqa: BLE001 - inference tensors do not track versions
        return None


def _layout_for(
    model: Any,
    mod: Any,
    img_mask: Any,
    img_shapes: Any,
    device: Any,
    *,
    reuse_identity: bool = False,
) -> _Layout:
    shapes = img_shapes[0]
    shapes_key = tuple(tuple(int(v) for v in s) for s in shapes)
    state = model.__dict__.get("_unsloth_q21_layouts")
    if state is None:
        state = {"recent": [], "by_content": {}}
        model.__dict__["_unsloth_q21_layouts"] = state
    # Only a cached step may trust img_mask identity (its KV cache fixes the prefix); others check content.
    version = _mask_version(img_mask)
    if reuse_identity:
        for ref, key, lay in state["recent"]:
            if ref() is img_mask and key == (shapes_key, device, version):
                return lay
    content = (
        tuple(img_mask.shape),
        str(img_mask.dtype),
        img_mask[0].detach().to("cpu").numpy().tobytes(),
        shapes_key,
        str(device),
    )
    lay = state["by_content"].pop(content, None)
    if lay is None:
        lay = _build_layout(model, mod, img_mask, shapes, device)
    state["by_content"][content] = lay
    while len(state["by_content"]) > _LAYOUTS_PER_MODULE:
        state["by_content"].pop(next(iter(state["by_content"])))
    state["recent"].insert(0, (weakref.ref(img_mask), (shapes_key, device, version), lay))
    del state["recent"][_RECENT_PER_MODULE:]
    return lay


def forget_layouts(model: Any) -> None:
    try:
        model.__dict__.pop("_unsloth_q21_layouts", None)
    except Exception:  # noqa: BLE001
        pass


def _cached_step(
    model: Any,
    lay: _Layout,
    text_dtype: Any,
    hidden_states: Any,
    timestep: Any,
    encoder_hidden_states_mask: Any,
    kv_cache: Any,
) -> Any:
    import torch

    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    prefix_len = lay.prefix_len
    hidden_states = model.img_in(hidden_states)
    # Keep the stock (batch, total, dim) layout so the blocks see the same strides.
    full = torch.empty(
        (batch_size, lay.total, hidden_states.shape[2]), dtype = text_dtype, device = device
    )
    full[:, prefix_len:] = hidden_states[:, hidden_states.shape[1] - (lay.total - prefix_len) :]

    timestep = timestep.to(hidden_states.dtype)
    timestep = torch.cat([timestep, timestep.new_zeros(1)], dim = 0)
    temb = model.time_text_embed(timestep, hidden_states)
    modulation = model.modulation(temb)

    attention_mask = None
    if encoder_hidden_states_mask is not None:
        joint_key_valid = torch.ones(batch_size, lay.total, dtype = torch.bool, device = device)
        joint_key_valid[:, lay.text_positions] = encoder_hidden_states_mask.bool()[
            :, _vlm_text(lay, encoder_hidden_states_mask.shape[1])
        ]
        attention_mask = joint_key_valid[:, None, None, :]

    joint_hidden_states = full[:, prefix_len:]
    rotary_emb = lay.rotary_emb[prefix_len:]
    modulation_mask = lay.target_token_mask[prefix_len:]
    for index_block, block in enumerate(model.transformer_blocks):
        joint_hidden_states = block(
            hidden_states = joint_hidden_states,
            modulation = modulation,
            rotary_emb = rotary_emb,
            attention_mask = attention_mask,
            target_token_mask = modulation_mask,
            layer_cache = kv_cache.get_layer(index_block),
            kv_cache_mode = "cached",
            cache_write_slice = None,
            segments = None,
            key_valid = None,
        )
    joint_hidden_states = model.norm_out(joint_hidden_states, temb, modulation_mask)
    return model.proj_out(joint_hidden_states)


def _autocast_state(device_type: str) -> tuple:
    """Autocast decides the text projection's output dtype as much as the weights do."""
    import torch
    try:
        return torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type)
    except Exception:  # noqa: BLE001 - older signature: CUDA state only
        return torch.is_autocast_enabled(), None


def _text_dtype(model: Any, encoder_hidden_states: Any) -> tuple:
    weight = getattr(getattr(model.txt_in, "out_layer", None), "weight", None)
    key = (
        encoder_hidden_states.dtype,
        id(weight),
        getattr(weight, "dtype", None),
        _autocast_state(encoder_hidden_states.device.type),
    )
    return key, model.__dict__.setdefault("_unsloth_q21_text_dtype", {}).get(key)


def _make_forward(mod: Any, stock: Any) -> Any:
    import torch

    stock_inner = inspect.unwrap(stock)
    lora_scale = getattr(mod, "apply_lora_scale", None)
    flex_cls = getattr(mod, "QwenImage21FlexAttnProcessor")
    build_block_mask = getattr(mod, "build_qwenimage21_block_causal_mask")

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_shapes,
        img_mask,
        encoder_hidden_states_mask = None,
        attention_kwargs = None,
        kv_cache = None,
        kv_cache_mode = None,
        return_dict = True,
    ):
        if torch.is_grad_enabled() or torch.compiler.is_compiling() or fast_step_disabled():
            return stock_inner(
                self,
                hidden_states,
                encoder_hidden_states,
                timestep,
                img_shapes,
                img_mask,
                encoder_hidden_states_mask = encoder_hidden_states_mask,
                attention_kwargs = attention_kwargs,
                kv_cache = kv_cache,
                kv_cache_mode = kv_cache_mode,
                return_dict = return_dict,
            )

        batch_size = hidden_states.shape[0]
        if kv_cache is not None and not self.config.causal_condition:
            raise ValueError(
                "kv_cache requires `causal_condition=True`. The cache is only valid because text and condition-image "
                "tokens modulate from t=0, which makes their activations independent of the denoising step."
            )
        if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
            raise ValueError(
                f"kv_cache_mode must be 'extract' or 'cached' when kv_cache is provided, got {kv_cache_mode!r}."
            )
        if kv_cache is None and kv_cache_mode is not None:
            raise ValueError(
                f"kv_cache_mode is {kv_cache_mode!r} but no kv_cache was passed to hold the prefix."
            )

        device = hidden_states.device
        lay = _layout_for(
            self, mod, img_mask, img_shapes, device, reuse_identity = kv_cache_mode == "cached"
        )
        dtype_key, text_dtype = _text_dtype(self, encoder_hidden_states)
        if (
            kv_cache_mode == "cached"
            and lay.tail_is_image
            and text_dtype is not None
            and self.config.causal_condition
        ):
            output = _cached_step(
                self, lay, text_dtype, hidden_states, timestep, encoder_hidden_states_mask, kv_cache
            )
            return (output,) if not return_dict else mod.Transformer2DModelOutput(sample = output)

        hidden_states = self.img_in(hidden_states)
        prefix_len = lay.prefix_len
        encoder_hidden_states = self.txt_in(encoder_hidden_states)
        self.__dict__["_unsloth_q21_text_dtype"][dtype_key] = encoder_hidden_states.dtype
        target_tokens = math.prod(img_shapes[0][-1])
        joint_hidden_states = torch.cat(
            [
                encoder_hidden_states,
                encoder_hidden_states.new_zeros(
                    batch_size, target_tokens // 4, encoder_hidden_states.shape[2]
                ),
            ],
            dim = 1,
        )
        joint_hidden_states = joint_hidden_states.repeat_interleave(
            lay.repeats, dim = 1, output_size = lay.total
        )
        joint_hidden_states[:, lay.image_positions] = hidden_states

        rotary_emb = lay.rotary_emb
        target_token_mask = lay.target_token_mask

        timestep = timestep.to(hidden_states.dtype)
        if self.config.causal_condition:
            timestep = torch.cat([timestep, timestep.new_zeros(1)], dim = 0)
            modulation_mask = target_token_mask
        else:
            modulation_mask = None
        temb = self.time_text_embed(timestep, hidden_states)
        modulation = self.modulation(temb)

        joint_key_valid = None
        if encoder_hidden_states_mask is not None:
            joint_key_valid = torch.ones(batch_size, lay.total, dtype = torch.bool, device = device)
            joint_key_valid[:, lay.text_positions] = encoder_hidden_states_mask.bool()[
                :, _vlm_text(lay, encoder_hidden_states_mask.shape[1])
            ]

        if kv_cache_mode == "cached":
            joint_hidden_states = joint_hidden_states[:, prefix_len:]
            rotary_emb = rotary_emb[prefix_len:]
            modulation_mask = modulation_mask[prefix_len:]
            attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
            cache_write_slice = None
            block_segments, block_key_valid = None, None
        else:
            processors = [block.attn.processor for block in self.transformer_blocks]
            needs_block_mask = any(isinstance(processor, flex_cls) for processor in processors)
            attention_mask = (
                build_block_mask(lay.image_ids, joint_key_valid, batch_size, device)
                if needs_block_mask
                else None
            )
            block_segments = (
                None
                if all(isinstance(processor, flex_cls) for processor in processors)
                else _segments(lay, mod)
            )
            cache_write_slice = slice(0, prefix_len) if kv_cache_mode == "extract" else None
            block_key_valid = joint_key_valid

        for index_block, block in enumerate(self.transformer_blocks):
            layer_cache = kv_cache.get_layer(index_block) if kv_cache is not None else None
            joint_hidden_states = block(
                hidden_states = joint_hidden_states,
                modulation = modulation,
                rotary_emb = rotary_emb,
                attention_mask = attention_mask,
                target_token_mask = modulation_mask,
                layer_cache = layer_cache,
                kv_cache_mode = kv_cache_mode,
                cache_write_slice = cache_write_slice,
                segments = block_segments,
                key_valid = block_key_valid,
            )

        joint_hidden_states = self.norm_out(joint_hidden_states, temb, modulation_mask)
        output = self.proj_out(joint_hidden_states)

        if not return_dict:
            return (output,)
        return mod.Transformer2DModelOutput(sample = output)

    functools.update_wrapper(forward, stock_inner, assigned = ("__name__", "__doc__"), updated = ())
    wrapped = lora_scale("attention_kwargs")(forward) if callable(lora_scale) else forward
    wrapped.__unsloth_q21_fast_step__ = True
    wrapped.__unsloth_stock_forward__ = stock
    return wrapped


def install(logger: Any = None) -> bool:
    if fast_step_disabled():
        return False
    try:
        import importlib
        mod = importlib.import_module(_MODULE)
    except Exception:  # noqa: BLE001 - diffusers without Qwen-Image 2.1
        return False
    cls = getattr(mod, _CLASS, None)
    if cls is None:
        return False
    with _LOCK:
        current = vars(cls).get("forward")
        if getattr(current, "__unsloth_q21_fast_step__", False):
            return True
        why = why_unsupported(mod)
        if why is not None:
            if logger is not None:
                logger.info("diffusion.qwenimage21: stock forward kept: %s", why)
            return False
        try:
            fast = _make_forward(mod, current)
        except Exception as exc:  # noqa: BLE001 - optimisation only
            if logger is not None:
                logger.warning("diffusion.qwenimage21: fast step unavailable: %s", exc)
            return False
        _INSTALLED[cls] = current
        cls.forward = fast
    if logger is not None:
        logger.info("diffusion.qwenimage21: denoiser step layout built once per render")
    return True


def install_for_pipe(pipe: Any, logger: Any = None) -> bool:
    if type(getattr(pipe, "transformer", None)).__name__ != _CLASS:
        return False
    try:
        return install(logger)
    except Exception as exc:  # noqa: BLE001 - optimisation only: the stock forward still runs
        if logger is not None:
            logger.warning("diffusion.qwenimage21: fast step unavailable: %s", exc)
        return False


def uninstall() -> None:
    with _LOCK:
        for cls, stock in list(_INSTALLED.items()):
            if getattr(vars(cls).get("forward"), "__unsloth_q21_fast_step__", False):
                cls.forward = stock
            _INSTALLED.pop(cls, None)


def is_installed() -> bool:
    return bool(_INSTALLED)
