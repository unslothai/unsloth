# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Wan VAE tiled decode with its spatial tiles split across two CUDA cards; bitwise the stock tiled decode.

Only the stock ``AutoencoderKLWan.tiled_decode`` body is mirrored (AST fingerprint). Automatic placement lends only an
idle same-model card. ``UNSLOTH_VIDEO_VAE_MULTIGPU=0`` disables (``force``: any selected card under Automatic)."""

from __future__ import annotations

import ast
import contextlib
import functools
import hashlib
import inspect
import math
import os
import textwrap
import threading
import traceback
import types
from typing import Any, Optional

ENV = "UNSLOTH_VIDEO_VAE_MULTIGPU"
_ATTR = "_unsloth_vae_multigpu"
# ast.unparse of AutoencoderKLWan.tiled_decode's body without its docstring (diffusers 0.42.1 and 0.43.0).
_KNOWN_BODIES = frozenset({"3b78672ea534d444"})
# Partner-card memory the split takes (allocator reserved, which is what the card loses): the copy's weights plus a
# fixed part plus a part per tile pixel, times a margin. Fitted on Kaggle 2x T4 (Wan 2.2 TI2V 5B, fp16 decode,
# 1280x704x49, 1629 MiB of VAE weights), reserved peak above the split's start: 256x256 tiles 2656 MiB, 704x496
# 6370 MiB (allocated 2422 / 5307 MiB); 704x704 allocated 6371 MiB (its reserved peak sat on cache the render had
# already reserved), which the line predicts at 8288 MiB reserved.
_WORKING_FIXED_MIB = 169
_WORKING_MIB_PER_TILE_PX = 3714 / 283648  # (6370 - 2656) MiB over (704 * 496 - 256 * 256) px
_NEED_MARGIN = 1.25
_FITTED_FRAMES = 49  # the clip length the line was fitted at


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


def _automatic_partner(vae: Any, partner: int) -> tuple[Optional[int], str]:
    """Automatic (no gpu_ids) lends only an idle same-model card, sized via nvidia-smi so no CUDA context opens on it."""
    from .diffusion_multigpu import DeviceProbe, automatic_partner, current_selection, probe_devices

    selection = current_selection()
    if (
        selection is None
        or not selection.automatic
        or (os.environ.get(ENV) or "").strip().lower() == "force"
    ):
        return int(partner), "selected"
    probes = probe_devices((selection.primary, *selection.secondaries))
    cards = [
        probes[o]
        for o in selection.secondaries
        if probes.get(o) is not None and probes[o].free_mib is not None
    ]
    primary = probes.get(int(selection.primary)) or DeviceProbe(
        int(selection.primary), "", None, None
    )
    pick, why = automatic_partner(primary, cards, need_mib(vae))
    return pick, f"automatic GPU placement: {why}"


def install(
    pipe: Any,
    partner: Optional[int],
    logger: Any = None,
) -> bool:
    vae = getattr(pipe, "vae", None)
    if vae is None or partner is None or not enabled():
        return False
    why = eligible(vae)
    if why is not None:
        if logger is not None:
            logger.info("video.vae_multigpu: not installed (%s)", why)
        return False
    partner, why = _automatic_partner(vae, partner)
    if partner is None:
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
    """Forward hooks stay with the source: the copy is built without them."""
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


def _extra_frames_mib(vae: Any, z: Any, tile_px: int) -> float:
    """Decoded frames the partner keeps beyond the fitted 49: every partner tile holds its whole clip (the per-frame
    list plus its concatenation) until the blend."""
    try:
        _, _, latent_frames, height, width = z.shape
        ratio = int(getattr(vae.config, "scale_factor_temporal", None) or 4)
        frames = 1 + (int(latent_frames) - 1) * ratio
        spatial = int(vae.spatial_compression_ratio)
        rows = len(range(0, int(height), max(1, int(vae.tile_sample_stride_height) // spatial)))
        cols = len(range(0, int(width), max(1, int(vae.tile_sample_stride_width) // spatial)))
        itemsize = next(vae.parameters()).element_size()
    except Exception:  # noqa: BLE001 - shape unknown: the fitted size alone
        return 0.0
    partner_tiles = (rows * cols) // 2
    return max(0, frames - _FITTED_FRAMES) * 3 * tile_px * itemsize * 2 * partner_tiles / 2**20


def need_mib(vae: Any, z: Any = None) -> int:
    tile_px = int(getattr(vae, "tile_sample_min_height", 256)) * int(
        getattr(vae, "tile_sample_min_width", 256)
    )
    working = _WORKING_FIXED_MIB + _WORKING_MIB_PER_TILE_PX * tile_px
    if z is not None:
        working += _extra_frames_mib(vae, z, tile_px)
    return int(math.ceil(_NEED_MARGIN * (_module_mib(vae) + working)))


def _cuda_autocast_state() -> tuple:
    """(enabled, dtype) of the calling thread's CUDA autocast."""
    import torch
    try:
        return torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda")
    except (AttributeError, TypeError):
        return torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()


def _source_map(src: Any, copy: Any) -> dict:
    out = {}
    copies = dict(copy.named_modules())
    for name, module in src.named_modules():
        if name in copies:
            out[id(module)] = copies[name]
    for named in ("named_parameters", "named_buffers"):
        copies = dict(getattr(copy, named)())
        for name, tensor in getattr(src, named)():
            if name in copies:
                out[id(tensor)] = copies[name]
    return out


def _rebound(value: Any, mapping: dict, seen: dict) -> Any:
    """``value`` with captured source modules / tensors swapped for the copy's, recursively."""
    import functools as _ft

    key = id(value)
    if key in mapping:
        return mapping[key]
    if key in seen:
        return seen[key]
    seen[key] = value
    if isinstance(value, types.MethodType):
        self_, func = (
            _rebound(value.__self__, mapping, seen),
            _rebound(value.__func__, mapping, seen),
        )
        out = (
            value
            if (self_ is value.__self__ and func is value.__func__)
            else types.MethodType(func, self_)
        )
    elif isinstance(value, _ft.partial):
        func = _rebound(value.func, mapping, seen)
        args = tuple(_rebound(a, mapping, seen) for a in value.args)
        kw = {k: _rebound(v, mapping, seen) for k, v in value.keywords.items()}
        same = (
            func is value.func
            and all(a is b for a, b in zip(args, value.args))
            and all(kw[k] is value.keywords[k] for k in kw)
        )
        out = value if same else _ft.partial(func, *args, **kw)
    elif isinstance(value, types.FunctionType):
        if not any(id(item) in mapping for item in _captured(value, set())):
            return value
        # Built before rebinding captures, so self-referencing closures get the new function; shared cells stay shared.
        cells = value.__closure__ or ()
        closure = []
        filled = []
        for cell in cells:
            new_cell = seen.get(("cell", id(cell)))
            if new_cell is None:
                new_cell = types.CellType()
                seen[("cell", id(cell))] = new_cell
                try:
                    filled.append((new_cell, cell.cell_contents))
                except ValueError:
                    pass
            closure.append(new_cell)
        out = types.FunctionType(
            value.__code__, value.__globals__, value.__name__, None, tuple(closure) or None
        )
        seen[key] = out
        for new_cell, contents in filled:
            new_cell.cell_contents = _rebound(contents, mapping, seen)
        defaults = tuple(_rebound(d, mapping, seen) for d in (value.__defaults__ or ()))
        out.__defaults__ = defaults or None
        kwdefaults = {
            k: _rebound(v, mapping, seen) for k, v in (value.__kwdefaults__ or {}).items()
        }
        out.__kwdefaults__ = kwdefaults or None
        out.__dict__.update(value.__dict__)
        out.__qualname__ = value.__qualname__
    else:
        out = value
    seen[key] = out
    return out


def _rebind_forwards(src: Any, copy: Any) -> int:
    """Rebuild speed-path instance forwards (closures over source modules) against the copy's modules."""
    mapping = _source_map(src, copy)
    seen: dict = {}
    moved = 0
    for module in copy.modules():
        for key, value in list(module.__dict__.items()):
            if callable(value) and not isinstance(value, (type, torch_module_type())):
                new = _rebound(value, mapping, seen)
                if new is not value:
                    module.__dict__[key] = new
                    moved += 1
    return moved


def torch_module_type() -> Any:
    import torch
    return torch.nn.Module


def _captured(value: Any, seen: set) -> list:
    import functools as _ft

    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, types.MethodType):
        return [value.__self__] + _captured(value.__func__, seen)
    if isinstance(value, _ft.partial):
        inner = [value.func, *value.args, *value.keywords.values()]
    elif isinstance(value, types.FunctionType):
        inner = [*(value.__defaults__ or ()), *(value.__kwdefaults__ or {}).values()]
        for cell in value.__closure__ or ():
            try:
                inner.append(cell.cell_contents)
            except ValueError:
                pass
    else:
        return []
    out = []
    for item in inner:
        out.append(item)
        out += _captured(item, seen)
    return out


def _stray_tensors(module: Any, device: Any) -> list:
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
    own = {id(m) for m in module.modules()}
    for name, sub in module.named_modules():
        for key, value in list(sub.__dict__.items()):
            if not callable(value) or isinstance(value, (type, torch.nn.Module)):
                continue
            for item in _captured(value, set()):
                if (isinstance(item, torch.nn.Module) and id(item) not in own) or (
                    isinstance(item, torch.Tensor)
                    and item.device.type == "cuda"
                    and item.device != device
                ):
                    out.append(f"{name}.{key} (captured {type(item).__name__})")
                    break
    return out


def _partner_reserve_mib(torch: Any, partner: int, free_mib: int) -> int:
    """Reserve kept free on the partner, re-read at decode time: headroom can shrink after the load."""
    from .diffusion_multigpu import _replica_reserve_mib

    try:
        total = int(torch.cuda.mem_get_info(partner)[1]) >> 20
    except Exception:  # noqa: BLE001 - the budget share is then taken of the free memory
        total = None
    return int(_replica_reserve_mib(free_mib, total))


_STICKY_CUDA_ERRORS = (
    "illegal memory access",
    "illegal address",
    "unspecified launch failure",
    "device-side assert",
    "misaligned address",
    "illegal instruction",
    "hardware stack error",
    "launch timed out",
)


def sticky_cuda_error(exc: Optional[BaseException]) -> bool:
    """True for a CUDA error that poisons the context (an illegal address, a device-side assert): every later call on
    it fails too, so a one-card retry cannot succeed. Out-of-memory and ordinary failures are not sticky."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        text = str(exc).lower()
        if any(m in text for m in _STICKY_CUDA_ERRORS):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


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
    need = need_mib(vae, z)
    reserve = _partner_reserve_mib(torch, partner, free)
    if free - reserve < need:
        return None, (
            f"cuda:{partner} has {free} MiB free ({reserve} MiB kept free), the copy needs ~{need}"
        )
    try:
        return split_tiled_decode(vae, z, partner), "split"
    except Exception as exc:  # noqa: BLE001 - the stock decode reruns every tile on the load card
        if sticky_cuda_error(exc):
            raise  # the context is lost: a one-card retry would fail on its first kernel
        if logger is not None:
            logger.warning("video.vae_multigpu: split decode failed, one card: %s", exc)
        return None, f"split failed ({type(exc).__name__}: {exc})"


def _cleanup(
    torch: Any, on_cuda: bool, caller: Any, side: Any, partner: Any, decoded: dict, box: dict
) -> Optional[BaseException]:
    """Every cleanup step runs even if an earlier one raised; returns the first error.

    Both streams are waited for before freeing (Kaggle 2x T4: without it the next decode hit an illegal memory
    access); after a failed wait nothing is released to the driver."""
    failures: list = []

    def step(fn: Any) -> bool:
        try:
            fn()
            return True
        except BaseException as exc:  # noqa: BLE001 - recorded, the remaining steps still run
            failures.append(exc)
            return False

    waited = True
    if on_cuda:
        waited = step(side.synchronize)
        waited = step(caller.synchronize) and waited
    step(decoded.clear)
    step(box.clear)
    if on_cuda:
        import gc

        # the copy sits in reference cycles; without collect ~2.7 GiB stays on the partner (Kaggle 2x T4)
        step(gc.collect)
        if waited:

            def release() -> None:
                with contextlib.suppress(Exception), torch.cuda.device(partner):
                    torch.cuda.empty_cache()

            step(release)
    return failures[0] if failures else None


def split_tiled_decode(
    vae: Any,
    z: Any,
    partner: Any,
    replica: Any = None,
) -> Any:
    """Stock ``AutoencoderKLWan.tiled_decode`` (diffusers 0.42 / 0.43) with tiles alternating between ``vae`` and a copy
    on ``partner`` (own stream); every cross-card copy is ordered by explicit stream waits."""
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
    autocast = (
        _cuda_autocast_state()
    )  # thread-local too: the partner must decode at the caller's precision

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
        stack.enter_context(torch.cuda.stream(caller))
        stack.enter_context(torch.cuda.stream(side))
        return stack

    def partner_run(todo: list) -> None:
        try:
            with (
                streams(),
                torch.inference_mode(inference),
                torch.set_grad_enabled(grad),
                (
                    torch.autocast("cuda", dtype = autocast[1], enabled = autocast[0])
                    if on_cuda
                    else contextlib.nullcontext()
                ),
            ):
                if on_cuda:
                    side.wait_event(ready)
                copy = replica
                try:
                    if copy is None:
                        from .diffusion_multigpu import build_replica
                        with _without_hooks(vae):
                            copy = build_replica(vae, partner)
                    _rebind_forwards(vae, copy)
                    # checked before releasing the caller, so no source tiles are wasted
                    stray = _stray_tensors(copy, partner)
                    if stray:
                        raise RuntimeError(
                            f"the copy holds tensors on another card: {', '.join(stray[:4])}"
                        )
                except BaseException as exc:  # noqa: BLE001 - recorded before the caller is released
                    errors.append(exc)
                    return
                finally:
                    copied.set()
                box["copy"] = copy
                run(copy, z.to(partner), todo)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            errors.append(exc)
        finally:
            copied.set()

    worker = threading.Thread(
        target = partner_run, args = (coords[1::2],), name = "unsloth-vae-split", daemon = True
    )
    # Source decode starts after the copy exists: otherwise the copy clones live causal caches (_feat_map).
    copied = threading.Event()
    worker.start()
    raised: Optional[BaseException] = None
    try:
        try:
            copied.wait()
            if errors:
                raise errors[0]
            run(vae, z, coords[0::2])
        finally:
            worker.join()
            for exc in errors:
                traceback.clear_frames(exc.__traceback__)
        if errors:
            raise errors[0]
        if on_cuda:
            caller.wait_stream(side)
            with torch.cuda.device(partner), torch.cuda.stream(side):
                # partner stream current for the source side orders frees against the copy
                rows = [[decoded[(i, j)].to(home) for j in col_starts] for i in row_starts]
        else:
            rows = [[decoded[(i, j)].to(home) for j in col_starts] for i in row_starts]
    except BaseException as exc:
        raised = exc
        raise
    finally:
        failure = _cleanup(torch, on_cuda, caller, side, partner, decoded, box)
        if failure is not None and (raised is None or sticky_cuda_error(failure)):
            # a sticky (context-lost) cleanup error must not be retried on one card
            raise failure
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
