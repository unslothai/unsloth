# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3 PDD (parallel decoding distillation) checkpoints: the stacked output-head bank.

A PDD ``with_heads`` file stores ``pdd_steps`` (32) video and audio output heads stacked along the rows of
``final_layer.video_out`` / ``final_layer.audio_out`` (``[32 * 96, 5376]`` and ``[32 * 32, 5376]``): head 0 is a full
head and every later head is an offset from it. One sampler step from ``sigma`` to ``sigma_next`` covers a contiguous
run of fine intervals on the unshifted grid, and consumes head 0 plus the dt-weighted mean of the offsets it spans.
This matches ComfyUI's PDD head selection (``FinalLayer.forward`` / ``_pdd_head``) term for term.

Graph safety: the ``(sigma, sigma_next)`` pair is a device tensor on each bank (``pdd_state``), rewritten in place by
a forward pre-hook on the denoiser before every call, and the head selection is pure tensor math on it. Nothing about
the step is a Python constant, so a CUDA graph replays it with new sigmas and torch.compile never specializes on it.
Without device float64 (Intel Arc A-series) the float64 math stays on the host; only the cast weights go to the device.

A ``without_heads`` bake keeps the plain single heads and needs an external head bank applied by a custom node, which
Studio does not have, so it is refused rather than run as a plain trunk (that renders wrong output with no error).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Optional

# ComfyUI's recipe for the baked 8-step PDD checkpoints: Euler, simple schedule, 8 steps, CFG 1, shifts 12 / 3.
H3_PDD_DEFAULT_STEPS = 8
H3_PDD_GUIDANCE = 1.0

VIDEO_HEAD_KEY = "final_layer.video_out.weight"
AUDIO_HEAD_KEY = "final_layer.audio_out.weight"
_BANKS_ATTR = "_unsloth_h3_pdd_banks"
_SCHEDULER_ATTR = "_unsloth_h3_pdd_scheduler"
_HOOK_ATTR = "_unsloth_h3_pdd_hook"
HEADS_ATTR = "_unsloth_h3_pdd_heads"

_PDD_TOKEN = re.compile(r"(?<![a-z0-9])pdd(?![a-z])")


def _name(filename: Optional[str]) -> str:
    return os.path.basename(str(filename or "")).lower()


def h3_pdd_name(filename: Optional[str]) -> bool:
    """The file name says PDD (``..._pdd_acc_8step_...``)."""
    return bool(_PDD_TOKEN.search(_name(filename)))


def _without_heads_message(name: str) -> str:
    from .video_families import H3_PDD_WITHOUT_HEADS_REFUSAL
    return f"'{name}': {H3_PDD_WITHOUT_HEADS_REFUSAL}"


def _metadata_pdd_steps(metadata: dict) -> Optional[int]:
    raw = metadata.get("config")
    if not raw:
        return None
    try:
        value = (json.loads(raw).get("transformer") or {}).get("pdd_steps")
    except (TypeError, ValueError, AttributeError):
        return None
    return int(value) if value is not None else None


def _in_dim(header: dict, key: str, default: int) -> int:
    shape = (header.get(key) or {}).get("shape") or ()
    return int(shape[1]) if len(shape) == 2 else default


def h3_pdd_heads_from_header(
    header: dict,
    *,
    video_dim: Optional[int] = None,
    audio_dim: Optional[int] = None,
    filename: Optional[str] = None,
) -> int:
    """Heads per stream of a ComfyUI-layout H3 file (1 for a plain denoiser). Raises ``ValueError`` for a PDD file
    Studio cannot run correctly: a trunk without its bank, or a bank whose shapes disagree."""
    name = os.path.basename(str(filename or "")) or "this checkpoint"
    metadata = header.get("__metadata__") or {}
    video = header.get(VIDEO_HEAD_KEY) or {}
    audio = header.get(AUDIO_HEAD_KEY) or {}
    if not video or not audio:
        return 1
    video_dim = int(video_dim or _in_dim(header, "video_patch_proj.weight", 96))
    audio_dim = int(audio_dim or _in_dim(header, "audio_patch_proj.weight", 32))
    rows_v = int((video.get("shape") or [0])[0])
    rows_a = int((audio.get("shape") or [0])[0])
    if rows_v % video_dim or rows_a % audio_dim or rows_v // video_dim != rows_a // audio_dim:
        raise ValueError(
            f"'{name}': output heads of {rows_v} video / {rows_a} audio rows are not one stacked bank of "
            f"{video_dim} / {audio_dim}-row heads."
        )
    heads = rows_v // video_dim
    declared = _metadata_pdd_steps(metadata)
    if heads == 1:
        baked = str(metadata.get("pdd_acc_baked", "")).strip().lower() == "true"
        if baked or (declared or 0) > 1 or h3_pdd_name(filename):
            raise ValueError(_without_heads_message(name))
    elif declared is not None and declared > 1 and declared != heads:
        raise ValueError(
            f"'{name}' declares {declared} PDD heads but stores {heads}: the file is inconsistent."
        )
    return heads


def h3_pdd_heads_in_file(
    path: str,
    *,
    video_dim: Optional[int] = None,
    audio_dim: Optional[int] = None,
    filename: Optional[str] = None,
) -> int:
    from .diffusion_comfy_quant import _read_header
    header, _ = _read_header(path)
    return h3_pdd_heads_from_header(
        header, video_dim = video_dim, audio_dim = audio_dim, filename = filename or path
    )


# ── the head-bank math (shared by every bank and by the tests' oracle comparisons) ──

_FP64_BY_DEVICE: dict = {}


def device_has_fp64(device: Any) -> bool:
    """CPU / CUDA / ROCm always; XPU per ``has_fp64`` (Arc A-series raises "Required aspect fp64"); else (MPS) never.
    Unknown counts as without: the host path is exact everywhere."""
    kind = getattr(device, "type", None)
    if kind is None:
        import torch
        device = torch.device(device)
        kind = device.type
    if kind in ("cpu", "cuda", "meta"):
        return True
    key = (kind, getattr(device, "index", None))
    known = _FP64_BY_DEVICE.get(key)
    if known is None:
        known = False
        if kind == "xpu":
            try:
                import torch
                index = key[1] if key[1] is not None else torch.xpu.current_device()
                known = bool(torch.xpu.get_device_properties(index).has_fp64)
            except Exception:  # noqa: BLE001
                known = False
        _FP64_BY_DEVICE[key] = known
    return known


def pdd_head_weights(state: Any, heads: int) -> Any:
    """Per-head blend weights ``(heads,)`` float64 for one step, from ``state = [sigma, sigma_next, select_shift,
    stream_shift]`` (float64). Head 0 gets 0 here: it is always added whole.

    ComfyUI: ``start, stop = round((1 - unshift(sigma)) * n), round((1 - unshift(sigma_next)) * n)`` with the unshift
    in float32 (the sampler's sigma dtype), ``start = min(start, n - 1)``, ``stop = max(stop, start + 1)``, then
    ``dt`` of the stream's shifted ``n``-interval grid over ``[start, stop)`` normalised to sum 1, applied to heads
    ``max(start, 1) .. stop - 1``. Same here, as masks instead of slices so no index ever reaches the host."""
    import torch

    sig = state[0:2].to(torch.float32)
    sel = state[2]
    # time_shift_sigma(s, shift, 1.0): the Python-float shift and (1 - shift) become float32 operands.
    base = sig / (sel.to(torch.float32) + sig * (1.0 - sel).to(torch.float32))
    pos = torch.round((1.0 - base).to(torch.float64) * heads)
    start = pos[0].clamp(min = 0, max = heads - 1)
    stop = torch.maximum(pos[1], start + 1)
    shift = state[3]
    grid = torch.linspace(1.0, 0.0, heads + 1, dtype = torch.float64, device = state.device)
    dt = (1.0 - shift * grid / (1.0 + (shift - 1.0) * grid)).diff()
    k = torch.arange(heads, dtype = torch.float64, device = state.device)
    inside = (k >= start) & (k < stop)
    dt = torch.where(inside, dt, torch.zeros_like(dt))
    w = dt / dt.sum()
    return torch.where(k >= 1, w, torch.zeros_like(w))


def _bank_class():
    import torch
    from torch import nn

    class H3PDDHeadBank(nn.Module):
        """``heads`` stacked output heads (``weight [heads * out, in]``, the checkpoint's own layout and key), blended
        per step from ``pdd_state``. Exposes ``weight`` / ``in_features`` / ``out_features`` like the Linear it
        replaces (the denoiser casts its activation to ``proj_out.weight.dtype``)."""

        def __init__(
            self,
            in_features: int,
            out_features: int,
            heads: int,
            *,
            dtype: Any = None,
        ):
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.heads = int(heads)
            dtype = dtype or torch.float32
            self.weight = nn.Parameter(
                torch.empty(self.heads * self.out_features, self.in_features, dtype = dtype)
            )
            self.bias = nn.Parameter(torch.empty(self.heads * self.out_features, dtype = dtype))
            # [sigma, sigma_next, select (video) shift, this stream's shift], float64. A plain tensor attribute, not a
            # buffer: a module-wide .to(bfloat16) must not round the sigmas, and offload must not move it.
            self.pdd_state = torch.tensor([1.0, 0.0, 12.0, 12.0], dtype = torch.float64)

        def extra_repr(self) -> str:
            return f"in_features={self.in_features}, out_features={self.out_features}, heads={self.heads}"

        def _own_state(self, device: Any = None) -> Any:
            """``pdd_state`` as a normal (non-inference) tensor on ``device``. The first render moves it to the
            GPU inside the pipeline's inference_mode; a tensor made there is an inference tensor, and the next
            render's set_shifts (outside inference_mode) could not write it."""
            state = self.pdd_state
            if state.is_inference() or (device is not None and state.device != device):
                with torch.inference_mode(False):
                    state = state.to(device or state.device, copy = True)
                self.pdd_state = state
            return state

        def set_state(self, pair: Any) -> None:
            self._own_state(pair.device)[0:2].copy_(pair)

        def set_shifts(self, select_shift: float, stream_shift: float) -> None:
            state = self._own_state()
            state[2] = float(select_shift)
            state[3] = float(stream_shift)

        def forward(self, hidden_states: Any) -> Any:
            state = self.pdd_state
            device = hidden_states.device
            if device_has_fp64(device):
                if state.device != device:
                    state = state.to(device)
                w = pdd_head_weights(state, self.heads).to(hidden_states.dtype)
            else:
                if state.device.type != "cpu":
                    state = state.cpu()
                w = pdd_head_weights(state, self.heads).to(hidden_states.dtype).to(device)
            weight = self.weight.to(hidden_states.dtype)
            bias = self.bias.to(hidden_states.dtype)
            rows = weight.view(self.heads, self.out_features, self.in_features)
            brows = bias.view(self.heads, self.out_features)
            return nn.functional.linear(
                hidden_states,
                rows[0] + torch.einsum("n,noi->oi", w, rows),
                brows[0] + torch.einsum("n,no->o", w, brows),
            )

    return H3PDDHeadBank


_BANK_CLASS = None


def head_bank_class():
    global _BANK_CLASS
    if _BANK_CLASS is None:
        _BANK_CLASS = _bank_class()
    return _BANK_CLASS


# ── the per-step sigma pair ──

_SIGMA_ARGS = ("timestep", "timestep_indices", "video_indices")
# Their MiniMaxH3Transformer3DModel.forward positions.
_SIGMA_POSITIONS = (3, 4, 7)


def h3_pdd_sigma_pair(
    timestep: Any, timestep_indices: Any, video_indices: Any, sample_sigmas: Any
) -> Any:
    """``[sigma, sigma_next]`` float64 on the device (host if it lacks float64), without a host sync. ``sigma`` is
    the generated video rows' sigma (conditioning rows sit at a larger t, so it is the largest video-row sigma),
    clamped like ComfyUI's; the next one is the schedule entry after the nearest, the last entry for the last step."""
    import torch

    rows = timestep_indices.index_select(0, video_indices.reshape(-1).to(timestep_indices.device))
    t = timestep.reshape(-1).index_select(0, rows).to(torch.float32).min()
    sigma = (1.0 - t).clamp(min = 1e-6)
    sigmas = sample_sigmas.reshape(-1).to(device = sigma.device, dtype = torch.float32)
    i = (sigmas - sigma).abs().argmin()
    nxt = sigmas.index_select(0, (i + 1).clamp(max = sigmas.shape[0] - 1).reshape(1))[0]
    pair = torch.stack([sigma, nxt])
    if not device_has_fp64(pair.device):
        pair = pair.cpu()
    return pair.to(torch.float64)


class _Schedule:
    """Holds the scheduler whose ``sigmas`` drive the bank (bound per render)."""

    __slots__ = ("scheduler",)

    def __init__(self) -> None:
        self.scheduler = None


def _pre_hook(module: Any, args: tuple, kwargs: dict) -> None:
    banks = module.__dict__.get(_BANKS_ATTR) or ()
    if not banks:
        return None
    values = []
    for name, position in zip(_SIGMA_ARGS, _SIGMA_POSITIONS):
        value = kwargs.get(name)
        if value is None and len(args) > position:
            value = args[position]
        values.append(value)
    holder = module.__dict__.get(_SCHEDULER_ATTR)
    scheduler = getattr(holder, "scheduler", None)
    sigmas = getattr(scheduler, "sigmas", None)
    if sigmas is None or any(v is None for v in values):
        raise ValueError(
            "MiniMax-H3 PDD heads need the sampler's sigma schedule: bind the render's scheduler with "
            "bind_h3_pdd_schedule before calling the denoiser."
        )
    pair = h3_pdd_sigma_pair(*values, sigmas)
    for bank in banks:
        bank.set_state(pair)
    return None


def install_h3_pdd_heads(model: Any, heads: int) -> bool:
    """Swap a freshly built (meta) H3 denoiser's ``proj_out`` / ``audio_proj_out`` for ``heads``-head banks and hook
    the per-step sigma pair. No-op (False) for ``heads <= 1``."""
    if int(heads) <= 1:
        return False
    import torch

    bank_cls = head_bank_class()
    banks = []
    for attr in ("proj_out", "audio_proj_out"):
        linear = getattr(model, attr)
        bank = bank_cls(linear.in_features, linear.out_features, heads, dtype = torch.float32)
        setattr(model, attr, bank)
        banks.append(bank)
    model.__dict__[_BANKS_ATTR] = tuple(banks)
    model.__dict__[_SCHEDULER_ATTR] = _Schedule()
    model.__dict__[HEADS_ATTR] = int(heads)
    model.__dict__[_HOOK_ATTR] = model.register_forward_pre_hook(_pre_hook, with_kwargs = True)
    return True


def h3_pdd_heads(module: Any) -> int:
    """Heads per stream of an installed bank, 0 otherwise."""
    try:
        return int((getattr(module, "__dict__", None) or {}).get(HEADS_ATTR) or 0)
    except (TypeError, ValueError):
        return 0


def bind_h3_pdd_schedule(
    module: Any, scheduler: Any, video_shift: Optional[float], audio_shift: Optional[float]
) -> bool:
    """Point one PDD denoiser at the scheduler whose sigmas it reads, and set the per-stream shifts (head selection
    runs on the video shift; each stream's dt grid on its own). False for a plain denoiser."""
    if not h3_pdd_heads(module):
        return False
    video_shift = 12.0 if video_shift is None else float(video_shift)
    audio_shift = 3.0 if audio_shift is None else float(audio_shift)
    module.__dict__[_SCHEDULER_ATTR].scheduler = scheduler
    video_bank, audio_bank = module.__dict__[_BANKS_ATTR]
    video_bank.set_shifts(video_shift, video_shift)
    audio_bank.set_shifts(video_shift, audio_shift)
    return True


H3_DENOISER_COMPONENTS = ("transformer", "transformer_ref")


def _components(pipe: Any) -> list:
    out = []
    for name in H3_DENOISER_COMPONENTS:
        try:
            module = getattr(pipe, name, None)
        except Exception:  # noqa: BLE001 - a modular pipeline may raise for an unloaded component
            module = None
        if module is not None:
            out.append(module)
    return out


def pipe_h3_pdd_heads(pipe: Any) -> int:
    return max((h3_pdd_heads(m) for m in _components(pipe)), default = 0)


def bind_h3_pdd_render(
    pipe: Any, video_shift: Optional[float], audio_shift: Optional[float]
) -> int:
    """Bind every PDD denoiser of an H3 pipeline to its video scheduler for this render; returns the head count
    (0: no PDD denoiser, nothing changed)."""
    scheduler = getattr(pipe, "scheduler", None)
    heads = 0
    for module in _components(pipe):
        if bind_h3_pdd_schedule(module, scheduler, video_shift, audio_shift):
            heads = max(heads, h3_pdd_heads(module))
    return heads


def h3_pdd_num_inference_steps(steps: int) -> int:
    """The MiniMax-H3 scheduler's grid size for ``steps`` model evaluations: its grid holds the terminal 0
    (``linspace(1, 0, n)``), so ComfyUI's 8-step simple schedule (``1, 7/8, ..., 1/8, 0`` before the shift) is 9."""
    return int(steps) + 1
