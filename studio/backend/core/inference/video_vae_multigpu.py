# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan VAE tiled decode with its spatial tiles shared by two CUDA cards of a multi-GPU selection.

Each spatial tile is an independent causal decode over every frame (its own feature cache), so the tiles run on the
load card and a per-decode copy of the VAE on the partner card at the same time, then blend on the load card exactly
as diffusers does: bitwise the stock tiled decode (Kaggle 2x T4, Wan 2.2 TI2V 5B fp32: 832x480x33 129 s -> 65 s).
Only the stock ``AutoencoderKLWan.tiled_decode`` body is mirrored (checked by AST fingerprint); untiled decodes are
untouched. ``UNSLOTH_VIDEO_VAE_MULTIGPU=0`` keeps the decode on one card."""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import inspect
import os
import textwrap
import threading
import types
from typing import Any, Optional

ENV = "UNSLOTH_VIDEO_VAE_MULTIGPU"
_ATTR = "_unsloth_vae_multigpu"
# ast.unparse of AutoencoderKLWan.tiled_decode's body without its docstring (diffusers 0.42.1 and 0.43.0).
_KNOWN_BODIES = frozenset({"3b78672ea534d444"})
# Partner-card working memory above the copy's weights for one 256 x 256 px tile over every frame (fp32 Wan 2.2 VAE
# measured 1.4 - 1.5 GiB on a T4, plus slack); scaled by tile area for wider tiles.
_WORKING_MIB = 1792
_WORKING_TILE_PX = 256 * 256


def enabled() -> bool:
    return (os.environ.get(ENV) or "").strip().lower() not in ("0", "off", "false", "no")


def body_fingerprint(fn: Any) -> Optional[str]:
    """Hash of a function's statements (docstring, signature and annotations ignored); None when unreadable."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except Exception:  # noqa: BLE001
        return None
    node = tree.body[0] if tree.body else None
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    body = list(node.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
    ):
        body = body[1:]
    dumped = "\n".join(ast.unparse(stmt) for stmt in body)
    return hashlib.sha1(dumped.encode()).hexdigest()[:16]


def _stock_tiled_decode(vae: Any) -> Any:
    return getattr(type(vae), "tiled_decode", None)


def eligible(vae: Any) -> Optional[str]:
    """None when the split can mirror this VAE's tiled decode, else why not."""
    if type(vae).__name__ != "AutoencoderKLWan":
        return f"{type(vae).__name__} is not the Wan VAE"
    stock = _stock_tiled_decode(vae)
    if stock is None or body_fingerprint(stock) not in _KNOWN_BODIES:
        return "AutoencoderKLWan.tiled_decode differs from the mirrored diffusers body"
    for name in ("clear_cache", "blend_v", "blend_h", "post_quant_conv", "decoder"):
        if not hasattr(vae, name):
            return f"the VAE has no {name}"
    return None


def _hooked(vae: Any) -> Optional[str]:
    """A VAE whose weights move under hooks (offload) cannot be copied as it is."""
    for name, sub in vae.named_modules():
        if (
            getattr(sub, "_hf_hook", None) is not None
            or getattr(sub, "_diffusers_hook", None) is not None
        ):
            return f"{name or '<root>'} is offloaded"
    return None


def install(
    pipe: Any,
    partner: Optional[int],
    logger: Any = None,
) -> bool:
    """Route ``pipe.vae.tiled_decode`` through the two-card split; partner = the selection's secondary ordinal."""
    vae = getattr(pipe, "vae", None)
    if vae is None or partner is None or not enabled():
        return False
    why = eligible(vae)
    if why is not None:
        if logger is not None:
            logger.info("video.vae_multigpu: not installed (%s)", why)
        return False
    if _ATTR in vae.__dict__:
        vae.__dict__[_ATTR]["partner"] = int(partner)
        return True
    state = {"partner": int(partner), "split": 0, "one_card": 0, "last": None}
    vae.__dict__[_ATTR] = state
    stock = _stock_tiled_decode(vae)

    @functools.wraps(stock)
    def tiled_decode(
        self: Any,
        z: Any,
        return_dict: bool = True,
    ) -> Any:
        dec, why = _try_split(self, z, state, logger)
        if dec is None:
            state["one_card"] += 1
            state["last"] = why
            return stock(self, z, return_dict = return_dict)
        state["split"] += 1
        state["last"] = "split"
        if not return_dict:
            return (dec,)
        from diffusers.models.autoencoders.vae import DecoderOutput

        return DecoderOutput(sample = dec)

    vae.tiled_decode = types.MethodType(tiled_decode, vae)
    if logger is not None:
        logger.info(
            "video.vae_multigpu: tiled decodes share their tiles with cuda:%d", int(partner)
        )
    return True


def uninstall(pipe: Any) -> None:
    vae = getattr(pipe, "vae", None)
    if vae is not None and vae.__dict__.pop(_ATTR, None) is not None:
        vae.__dict__.pop("tiled_decode", None)


def stats(pipe: Any) -> Optional[dict]:
    vae = getattr(pipe, "vae", None)
    state = getattr(vae, "__dict__", {}).get(_ATTR) if vae is not None else None
    return (
        None if state is None else {k: state[k] for k in ("partner", "split", "one_card", "last")}
    )


def _module_mib(module: Any) -> int:
    return (
        sum(t.numel() * t.element_size() for t in (*module.parameters(), *module.buffers())) >> 20
    )


@contextlib.contextmanager
def _without_hooks(vae: Any):
    """Forward hooks (e.g. the MP4 stream encoder's decoder hook) stay with the source: the copy is built without
    them, and they are restored at once."""
    saved = []
    for sub in vae.modules():
        for attr in (
            "_forward_hooks",
            "_forward_pre_hooks",
            "_forward_hooks_with_kwargs",
            "_forward_pre_hooks_with_kwargs",
            "_forward_hooks_always_called",
        ):
            d = sub.__dict__.get(attr)
            if d:
                saved.append((d, dict(d)))
                d.clear()
    try:
        yield
    finally:
        for d, items in saved:
            d.update(items)


def _partner_free_mib(torch: Any, partner: int) -> int:
    """Free MiB on the partner card, counting the caching allocator's reusable blocks (the CFG copy's activations)."""
    free, _total = torch.cuda.mem_get_info(partner)
    try:
        from .diffusion_memory import cuda_reusable_cache_bytes
        free += cuda_reusable_cache_bytes(torch.device("cuda", partner))
    except Exception:  # noqa: BLE001 - driver-free only
        pass
    return int(free) >> 20


def need_mib(vae: Any) -> int:
    tile_px = int(getattr(vae, "tile_sample_min_height", 256)) * int(
        getattr(vae, "tile_sample_min_width", 256)
    )
    return _module_mib(vae) + max(1024, _WORKING_MIB * tile_px // _WORKING_TILE_PX)


def _stray_tensors(module: Any, device: Any) -> list:
    """Tensors the copy holds on another CUDA card (parameters, buffers and plain tensor attributes)."""
    import torch

    out = []
    for name, sub in module.named_modules():
        for key, value in list(sub.__dict__.items()):
            if (
                isinstance(value, torch.Tensor)
                and value.device.type == "cuda"
                and value.device != device
            ):
                out.append(f"{name}.{key}")
    out += [
        n
        for n, t in (*module.named_parameters(), *module.named_buffers())
        if t.device.type == "cuda" and t.device != device
    ]
    return out


def _try_split(vae: Any, z: Any, state: dict, logger: Any) -> tuple[Optional[Any], str]:
    import torch

    partner = state["partner"]
    if not enabled():
        return None, "disabled"
    if getattr(z, "device", None) is None or z.device.type != "cuda" or z.device.index == partner:
        return None, "latents not on another CUDA card"
    if any(p.device != z.device for p in vae.post_quant_conv.parameters()):
        return None, "the VAE is not resident beside the latents"
    hooked = _hooked(vae)
    if hooked is not None:
        return None, hooked
    try:
        free = _partner_free_mib(torch, partner)
    except Exception as exc:  # noqa: BLE001
        return None, f"cuda:{partner} unreadable ({exc})"
    need = need_mib(vae)
    if free < need:
        return None, f"cuda:{partner} has {free} MiB free, the copy needs ~{need}"
    try:
        return split_tiled_decode(vae, z, partner), "split"
    except Exception as exc:  # noqa: BLE001 - the stock decode reruns every tile on the load card
        if logger is not None:
            logger.warning("video.vae_multigpu: split decode failed, one card: %s", exc)
        return None, f"split failed ({type(exc).__name__}: {exc})"


def split_tiled_decode(
    vae: Any,
    z: Any,
    partner: Any,
    replica: Any = None,
) -> Any:
    """``AutoencoderKLWan.tiled_decode`` (diffusers 0.42 / 0.43 body) with tiles alternating between ``vae`` (the
    latents' card, on the caller's current stream) and a copy on ``partner`` (its own stream); returns the clamped
    sample. Tile math, order and blend are the stock ones.

    Stream discipline (every cross-card copy is explicit): the copy is built and its tiles decoded on a dedicated
    partner stream that first waits for the caller's stream (the latents and weights are ready); copies from the load
    card run with that caller stream current for the load card, so PyTorch orders them against it; the decoded tiles
    come back on the caller's stream after it waits for the partner stream, with the partner stream current for the
    source side; nothing on the partner card is freed before that stream has finished."""
    import torch

    from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

    _, _, num_frames, height, width = z.shape
    sample_height = height * vae.spatial_compression_ratio
    sample_width = width * vae.spatial_compression_ratio
    tile_latent_min_height = vae.tile_sample_min_height // vae.spatial_compression_ratio
    tile_latent_min_width = vae.tile_sample_min_width // vae.spatial_compression_ratio
    tile_latent_stride_height = vae.tile_sample_stride_height // vae.spatial_compression_ratio
    tile_latent_stride_width = vae.tile_sample_stride_width // vae.spatial_compression_ratio
    tile_sample_stride_height = vae.tile_sample_stride_height
    tile_sample_stride_width = vae.tile_sample_stride_width
    if vae.config.patch_size is not None:
        sample_height = sample_height // vae.config.patch_size
        sample_width = sample_width // vae.config.patch_size
        tile_sample_stride_height = tile_sample_stride_height // vae.config.patch_size
        tile_sample_stride_width = tile_sample_stride_width // vae.config.patch_size
        blend_height = (
            vae.tile_sample_min_height // vae.config.patch_size - tile_sample_stride_height
        )
        blend_width = vae.tile_sample_min_width // vae.config.patch_size - tile_sample_stride_width
    else:
        blend_height = vae.tile_sample_min_height - tile_sample_stride_height
        blend_width = vae.tile_sample_min_width - tile_sample_stride_width

    row_starts = list(range(0, height, tile_latent_stride_height))
    col_starts = list(range(0, width, tile_latent_stride_width))
    coords = [(i, j) for i in row_starts for j in col_starts]
    decoded: dict = {}
    grad, inference = torch.is_grad_enabled(), torch.is_inference_mode_enabled()

    def run(module: Any, latents: Any, todo: list) -> None:
        for i, j in todo:
            module.clear_cache()
            time = []
            for k in range(num_frames):
                module._conv_idx = [0]
                tile = latents[
                    :, :, k : k + 1, i : i + tile_latent_min_height, j : j + tile_latent_min_width
                ]
                tile = module.post_quant_conv(tile)
                time.append(
                    module.decoder(
                        tile,
                        feat_cache = module._feat_map,
                        feat_idx = module._conv_idx,
                        first_chunk = (k == 0),
                    )
                )
            decoded[(i, j)] = torch.cat(time, dim = 2)
        module.clear_cache()

    home = z.device
    partner = torch.device(partner) if not isinstance(partner, torch.device) else partner
    on_cuda = home.type == "cuda" and partner.type == "cuda"
    caller = torch.cuda.current_stream(home) if on_cuda else None
    side = torch.cuda.Stream(device = partner) if on_cuda else None
    ready = None
    if on_cuda:
        ready = torch.cuda.Event()
        ready.record(caller)
    box: dict = {}
    errors: list = []

    def streams():
        if not on_cuda:
            return contextlib.nullcontext()
        stack = contextlib.ExitStack()
        stack.enter_context(
            torch.cuda.stream(caller)
        )  # current for the load card in this thread too
        stack.enter_context(torch.cuda.stream(side))  # last: the partner card is the current device
        return stack

    def partner_run(todo: list) -> None:
        try:
            with streams(), torch.inference_mode(inference), torch.set_grad_enabled(grad):
                if on_cuda:
                    side.wait_event(ready)
                copy = replica
                if copy is None:
                    from .diffusion_multigpu import build_replica
                    with _without_hooks(vae):
                        copy = build_replica(vae, partner)
                stray = _stray_tensors(copy, partner)
                if stray:
                    raise RuntimeError(
                        f"the copy holds tensors on another card: {', '.join(stray[:4])}"
                    )
                box["copy"] = copy
                run(copy, z.to(partner), todo)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            errors.append(exc)

    worker = threading.Thread(
        target = partner_run, args = (coords[1::2],), name = "unsloth-vae-split", daemon = True
    )
    worker.start()
    try:
        run(vae, z, coords[0::2])
    finally:
        worker.join()
    try:
        if errors:
            raise errors[0]
        if on_cuda:
            caller.wait_stream(side)
            with torch.cuda.device(partner), torch.cuda.stream(side):
                # source side current = the partner stream: PyTorch makes the copy wait for it, and makes it wait
                # for the copy before any partner block is reused
                rows = [[decoded[(i, j)].to(home) for j in col_starts] for i in row_starts]
        else:
            rows = [[decoded[(i, j)].to(home) for j in col_starts] for i in row_starts]
    finally:
        if on_cuda:
            side.synchronize()
        decoded.clear()
        box.clear()
        if on_cuda:
            import gc

            # the copy sits in reference cycles (bound hooks / patch closures): collect it now, or the next decode
            # finds its ~2.7 GiB still held on the partner card (Kaggle 2x T4: 2478 MiB free on the second render)
            gc.collect()
            with contextlib.suppress(Exception), torch.cuda.device(partner):
                torch.cuda.empty_cache()
    result_rows = []
    for i, row in enumerate(rows):
        result_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = vae.blend_v(rows[i - 1][j], tile, blend_height)
            if j > 0:
                tile = vae.blend_h(row[j - 1], tile, blend_width)
            result_row.append(tile[:, :, :, :tile_sample_stride_height, :tile_sample_stride_width])
        result_rows.append(torch.cat(result_row, dim = -1))
    dec = torch.cat(result_rows, dim = 3)[:, :, :, :sample_height, :sample_width]
    if vae.config.patch_size is not None:
        dec = unpatchify(dec, patch_size = vae.config.patch_size)
    return torch.clamp(dec, min = -1.0, max = 1.0)
