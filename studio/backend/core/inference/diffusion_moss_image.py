# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
#
# The transformer, scheduler and pipeline below are a port of OpenMOSS-Team's ``pipeline_moss_image.py``
# (Apache-2.0, https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT), keeping its state-dict names and numerics.

"""MOSS-Image (OpenMOSS-Team/MOSS-Image-SFT and -Base) as an in-tree diffusers pipeline.

The official repos ship the transformer and scheduler as remote code (``model_index.json`` points both at the repo's
``pipeline_moss_image.py``), and Studio never runs repo code: no ``trust_remote_code``, no ``custom_pipeline``. So the
model lives here and the pipeline is assembled per component from safetensors and JSON only: ``MOSSImageTransformer``
(this file) from ``transformer/``, the Qwen3.5 text encoder (``transformers.Qwen3_5Model``, transformers >= 5.3) from
``text_encoder/``, the FLUX.2 small-decoder VAE (``diffusers.AutoencoderKLFlux2``) from ``vae/`` and the tokenizer.

Numerics follow the reference op for op (fp32 RMSNorm with the weight applied after the cast, fp64 timestep sinusoid,
fp32 RoPE phases cast to the activation dtype, fp32 noise, CFG and Euler update, fp32 VAE decode); the tests check the
port bit for bit against the reference file on a tiny random model. What differs is only how the same values are
reached:

- RoPE tables are built once per (height, width, text length) and reused (the reference rebuilt them per forward).
- The text refiner (two unmodulated text blocks) does not read the latent or the timestep, so the pipeline runs it
  once per prompt per render (``refine_text``) and hands the refined tokens to every step (``refined_text``), unless
  the transformer carries offload hooks (then every step refines, as the reference does).
- One sample at its full text length takes no boolean-mask indexing and no ``torch.all`` mask check, so the forward
  is free of host syncs (CUDA-graph capturable); the pipeline passes no mask when every row has the same length.
- Attention goes through diffusers' dispatcher (``set_attention_backend`` reaches it), whose native backend is the
  same ``scaled_dot_product_attention`` call.

Sampling defaults follow ComfyUI's MOSS-Image workflow (the baseline): 1024x1024, 50 steps, CFG 3, shift 5, Euler
over ``sgm_uniform`` sigmas, skip-layer guidance on block 22 for the negative pass, empty negative prompt. The
reference pipeline snaps its sigmas to the 1000-point training grid instead; ``sigma_schedule = "training_grid"`` on
the scheduler (or ``UNSLOTH_MOSS_SIGMAS=training_grid``) selects that.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F

from loggers import get_logger

logger = get_logger(__name__)

MOSS_IMAGE_FAMILY_NAME = "moss-image"
MOSS_IMAGE_PIPELINE_CLASS = "MOSSImagePipeline"
MOSS_IMAGE_TRANSFORMER_CLASS = "MOSSImageTransformer"
MOSS_IMAGE_SCHEDULER_CLASS = "MOSSImageScheduler"
# The Qwen3.5 text encoder class (``transformers.Qwen3_5Model``) first shipped in transformers 5.3.
MIN_TRANSFORMERS = "5.3.0"
SIGMAS_ENV = "UNSLOTH_MOSS_SIGMAS"
SIGMA_SCHEDULES = ("sgm_uniform", "training_grid")
# Set by diffusion_moss_fused, read only inside a compiled block.
FUSED_QKV_ATTR = "_unsloth_moss_fused_qkv"
LINEAR_RUNNER_ATTR = "_unsloth_moss_linear"
QK_ROPE_ATTR = "_unsloth_moss_qk_rope"
# Kill switch for the per-render text-refiner cache (``refine_text``): 0 refines on every step like the reference.
TEXT_CACHE_ENV = "UNSLOTH_MOSS_TEXT_CACHE"
DEFAULT_UNCOND_SKIP_LAYERS = (22,)


def _env_off(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in ("0", "off", "false", "no")


def _compute_dtype(linear: nn.Module) -> torch.dtype:
    """The dtype a Linear's input is cast to: the reference casts to ``weight.dtype``, which is the storage dtype under
    diffusers' ``GGUFLinear`` (uint8), so a ``compute_dtype`` the layer declares wins. Same dtype for a plain Linear."""
    declared = getattr(linear, "compute_dtype", None)
    return declared if isinstance(declared, torch.dtype) else linear.weight.dtype


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim = True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _validate_skip_layers(
    layers,
    depth,
    name = "skip_layers",
):
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f"{name} must contain integer block indices in [0, {depth}).")
    return frozenset(layers)


def _length_dynamic(*tensors) -> None:
    """Hint dim 1 (the text length) of the refiner block inputs as dynamic. The pipeline refines the text once per render
    OUTSIDE ``forward`` (``refine_text``), where Studio's per-forward prompt-length arming does not reach, so the
    compiled text block otherwise specialised on the first length and recompiled on the uncond one. A no-op eagerly."""
    try:
        if torch.compiler.is_compiling():
            return
        from torch._dynamo import maybe_mark_dynamic
    except Exception:  # noqa: BLE001 - no dynamo, nothing to hint
        return
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.dim() > 1:
            maybe_mark_dynamic(tensor, 1)


def _rope_tables(rope):
    """``(cos, sin)`` duplicated to the full head dim, shaped to broadcast over heads: the split-half layout the
    reference's ``_rotate`` builds on every call, built once."""
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    return cos, sin


def _rotate(x, tables):
    cos, sin = tables
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


class _AttnProcessor:
    """Holds the attention backend diffusers' ``set_attention_backend`` assigns (None = the dispatcher's active one)."""

    _attention_backend = None
    _parallel_config = None


def _dispatch_attention():
    try:
        from diffusers.models.attention_dispatch import dispatch_attention_fn
    except Exception:  # noqa: BLE001 - an older diffusers without the dispatcher
        return None
    return dispatch_attention_fn


_DISPATCH = _dispatch_attention()


def _attention(
    q,
    k,
    v,
    tables,
    heads,
    backend = None,
):
    shape = (*q.shape[:2], heads, q.shape[-1] // heads)
    q = _rotate(q.reshape(shape), tables)
    k = _rotate(k.reshape(shape), tables)
    v = v.reshape(shape)
    if _DISPATCH is None:
        out = F.scaled_dot_product_attention(
            q.to(v.dtype).transpose(1, 2), k.to(v.dtype).transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
    else:
        out = _DISPATCH(q.to(v.dtype), k.to(v.dtype), v, backend = backend)
    return out.flatten(2)


def _attention_heads_first(
    q,
    k,
    v,
    backend = None,
):
    """Attention over already normalised and rotated [B, H, S, D] q / k / v (the dispatcher takes [B, S, H, D] views)."""
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    if _DISPATCH is None:
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
    else:
        out = _DISPATCH(q, k, v, backend = backend)
    return out.flatten(2)


try:  # the mixin is what ``ModelMixin.set_attention_backend`` looks for
    from diffusers.models.attention import AttentionModuleMixin as _AttentionMixin
except Exception:  # noqa: BLE001 - older diffusers: attention stays on SDPA

    class _AttentionMixin:  # type: ignore[no-redef]
        pass


class _TextBlock(nn.Module, _AttentionMixin):
    _supports_qkv_fusion = False

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.processor = _AttnProcessor()
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)

    def forward(self, x, rope):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(_attention(q, k, v, rope, self.heads, self.processor._attention_backend))
        y = self.norm2(x)
        return x + self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([_TextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope):
        x = self.proj(x.to(_compute_dtype(self.proj)))
        _length_dynamic(x, *rope)
        for block in self.blocks:
            x = block(x, rope)
        return x


class _Block(nn.Module, _AttentionMixin):
    _supports_qkv_fusion = False

    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.processor = _AttnProcessor()
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict(
            {name: nn.Linear(dim, dim, bias = False) for name in ("q", "k", "v", "o")}
        )
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias = False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias = False)

    def _linear(self, name, x):
        run = self.__dict__.get(LINEAR_RUNNER_ATTR) if torch.compiler.is_compiling() else None
        return self.proj[name](x) if run is None else run(self.proj[name], x)

    def _qkv(self, y):
        fused = self.__dict__.get(FUSED_QKV_ATTR) if torch.compiler.is_compiling() else None
        if fused is not None:
            linear, q, k, v, run = fused
            # only while q / k / v are still the modules it was built from (a LoRA wrapper replaces them)
            if self.proj["q"] is q and self.proj["k"] is k and self.proj["v"] is v:
                return (linear(y) if run is None else run(linear, y)).chunk(3, -1)
        return self._linear("q", y), self._linear("k", y), self._linear("v", y)

    def forward(self, x, rope, modulation):
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a[:, None]) + shift_a[:, None]
        dtype = _compute_dtype(self.proj["q"])
        y = y.to(dtype)
        q, k, v = self._qkv(y)
        fused = self.__dict__.get(QK_ROPE_ATTR) if torch.compiler.is_compiling() else None
        if fused is not None and q.dtype == torch.bfloat16:
            q, k, v = fused(
                q, k, v, self.q_norm.weight, self.k_norm.weight, rope[0], rope[1], self.heads
            )
            attn = _attention_heads_first(q, k, v, self.processor._attention_backend)
        else:
            q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
            k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
            attn = _attention(q, k, v, rope, self.heads, self.processor._attention_backend)
        x = x + self._linear("o", attn) * gate_a[:, None]
        y = self.norm2(x) * (1 + scale_f[:, None]) + shift_f[:, None]
        hidden, gate = self._linear("ffn_up_gate", y.to(dtype)).chunk(2, -1)
        return x + self._linear("ffn_down", _situ(hidden, gate)) * gate_f[:, None]


class _RoPE(nn.Module):
    _MAX_CACHED = 8

    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))
        self._tables: dict = {}

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = torch.arange(self.num_bands, device = "cpu", dtype = torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def _apply(
        self,
        fn,
        recurse = True,
    ):
        super()._apply(fn, recurse = recurse)
        # .float() cannot recover frequencies already rounded by .to(bf16).
        self.bands = self._default_bands(self.bands.device)
        self._tables = {}
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Frequencies are derived constants; the key stays for checkpoint compatibility.
        self.bands = self._default_bands(self.bands.device)
        self._tables = {}

    def forward(self, height, width, text_length, device):
        """The reference's phase table ``[sin | cos]`` in fp32, [1, H*W + L, head_dim]."""
        if self.bands.dtype != torch.float32 or self.bands.device.type == "meta":
            self.bands = self._default_bands(device)
        bands = self.bands.to(device)
        h, w = torch.meshgrid(
            torch.arange(height, device = device), torch.arange(width, device = device), indexing = "ij"
        )
        image = (
            torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1)
            .flatten(0, 1)
            .float()
        )
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device = device, dtype = torch.float32)
        text[:, 1] = torch.arange(text_length, device = device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)

    def tables(self, height, width, text_length, device, dtype):
        """``((cos, sin) for image + text, (cos, sin) for the text alone)`` in ``dtype``, cached per key."""
        device = torch.device(device)
        key = (int(height), int(width), int(text_length), device.type, device.index, dtype)
        hit = self._tables.get(key)
        if hit is not None:
            return hit
        rope = self.forward(height, width, text_length, device).to(dtype)
        full = _rope_tables(rope)
        text = _rope_tables(rope[:, height * width :])
        if len(self._tables) >= self._MAX_CACHED:
            self._tables.pop(next(iter(self._tables)))
        entry = self._tables[key] = (full, text)
        return entry


def _model_bases():
    from diffusers.configuration_utils import ConfigMixin
    from diffusers.models.modeling_utils import ModelMixin
    return ModelMixin, ConfigMixin


_ModelMixin, _ConfigMixin = _model_bases()
from diffusers.configuration_utils import register_to_config  # noqa: E402


class MOSSImageTransformer(_ModelMixin, _ConfigMixin):
    """MOSS-Image DiT: a text refiner, then 42 single-stream joint blocks over [image ; text] sharing one AdaLN.

    Each sample runs at its true text length, text refinement included, so padded tokens never reach attention."""

    _repeated_blocks = ["_Block", "_TextBlock"]
    _no_split_modules = ["_Block", "_TextBlock"]
    _skip_layerwise_casting_patterns = ["rope", "norm"]

    @register_to_config
    def __init__(
        self,
        dim = 3072,
        ffn_dim = 8192,
        num_heads = 24,
        num_layers = 42,
        text_dim = 4096,
        freq_dim = 256,
        latent_channels = 128,
        rope_base = 10000.0,
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError(
                "Require dim divisible by heads, head_dim divisible by 8, and even freq_dim."
            )
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([_Block(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias = False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)

    def load_training_state_dict(
        self,
        state_dict,
        *,
        prefix = "",
    ):
        """Load a consolidated raw/EMA model mapping, strictly; no silent key drops."""
        if prefix:
            if any(not key.startswith(prefix) for key in state_dict):
                raise ValueError("All keys must start with the explicit prefix.")
            state_dict = {key[len(prefix) :]: value for key, value in state_dict.items()}
        return self.load_state_dict(state_dict, strict = True)

    def _activation_dtype(self):
        return _compute_dtype(self.video_embedder)

    def refine_text(self, encoder_hidden_states):
        """The text refiner over [B, L, text_dim] at full length L (no padding): [B, L, dim]. Depends on neither the
        latent nor the timestep, so it is step-invariant; the RoPE text positions depend only on L."""
        length = encoder_hidden_states.shape[1]
        device = encoder_hidden_states.device
        _, text_tables = self.rope.tables(1, 1, length, device, self._activation_dtype())
        return self.text_embedder(encoder_hidden_states, text_tables)

    def _time_modulation(self, timestep, batch, device):
        times = torch.as_tensor(timestep, device = device, dtype = torch.float32).reshape(-1)
        times = times.expand(batch)
        # Training computes the sinusoid in float64 and feeds float32 into its MLP.
        half = self.config.freq_dim // 2
        freq = 10000 ** (-torch.arange(half, device = times.device, dtype = torch.float64) / half)
        phase = times.double()[:, None] * freq
        embedding = torch.cat((phase.cos(), phase.sin()), -1).float()
        time = self.time_embedding(embedding.to(_compute_dtype(self.time_embedding[0]))).float()
        mod = self.shared_block_modulation(
            time.to(_compute_dtype(self.shared_block_modulation))
        ).view(batch, 6, -1)
        head = self.video_head_modulation(time.to(_compute_dtype(self.video_head_modulation))).view(
            batch, 2, -1
        )
        return mod, head

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states = None,
        encoder_attention_mask = None,
        return_dict = True,
        skip_layers = None,
        refined_text = None,
    ):
        """``refined_text`` ([B, L, dim], from ``refine_text``) replaces ``encoder_hidden_states`` and skips the text
        refiner; give exactly one. A mask applies to ``encoder_hidden_states`` only (refined text is unpadded)."""
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if (encoder_hidden_states is None) == (refined_text is None):
            raise ValueError("Give exactly one of encoder_hidden_states and refined_text.")
        text_in = encoder_hidden_states if refined_text is None else refined_text
        if text_in.ndim != 3 or text_in.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        if mask is not None:
            if refined_text is not None:
                raise ValueError(
                    "refined_text is unpadded; pass no encoder_attention_mask with it."
                )
            # The reference checks a mask on every call; it costs a host sync, so only a mask that is given is checked.
            if mask.shape != text_in.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        device = hidden_states.device
        mod, head = self._time_modulation(timestep, batch, device)
        dtype = self._activation_dtype()
        outputs = []
        for index in range(batch):
            if mask is None:
                context = text_in[index : index + 1]
            else:
                context = text_in[index : index + 1, mask[index].bool()]
            length = context.shape[1]
            if length == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            video = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            video = self.video_embedder(video.to(_compute_dtype(self.video_embedder)).contiguous())
            # The reference casts RoPE to the video activation dtype.
            full_tables, text_tables = self.rope.tables(height, width, length, device, video.dtype)
            text = self.text_embedder(context, text_tables) if refined_text is None else context
            x = torch.cat((video, text.to(video.dtype)), 1)
            sample_mod = mod[index : index + 1]
            for layer_index, layer in enumerate(self.layers):
                if layer_index not in skip_layers:
                    x = layer(x, full_tables, sample_mod)
            x = self.final_norm_video(x[:, : height * width].to(self.final_norm_video.weight.dtype))
            x = x * (1 + head[index : index + 1, 1, None]) + head[index : index + 1, 0, None]
            x = self.final_linear_video(x.to(_compute_dtype(self.final_linear_video)))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        sample = outputs[0] if batch == 1 else torch.cat(outputs)
        if not return_dict:
            return (sample,)
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample = sample)


from diffusers import FlowMatchEulerDiscreteScheduler  # noqa: E402


def _time_snr_shift(shift: float, t):
    if shift == 1.0:
        return t
    return shift * t / (1 + (shift - 1) * t)


def sgm_uniform_sigmas(
    steps: int,
    shift: float,
    num_train_timesteps: int = 1000,
) -> list:
    """ComfyUI ``sgm_uniform`` over ``ModelSamplingDiscreteFlow`` (multiplier 1), in its fp32 ops, without the
    terminal zero: ``linspace(sigma_max, sigma_min, steps + 1)[:-1]`` re-shifted, as ``comfy.samplers.normal_scheduler``
    with ``sgm=True``."""
    grid = _time_snr_shift(shift, torch.arange(1, num_train_timesteps + 1, 1) / num_train_timesteps)
    sigma_min, sigma_max = grid[0], grid[-1]
    timesteps = torch.linspace(sigma_max, sigma_min, steps + 1)[:-1]
    return [float(_time_snr_shift(shift, ts)) for ts in timesteps]


def training_grid_sigmas(
    steps: int,
    shift: float,
    training_shift: float,
    total: int = 1000,
) -> list:
    """The reference pipeline's schedule: shifted linspace snapped to the nearest of the training grid's sigmas."""

    def grid(count, s):
        lower = s / (total + s - 1)
        values = torch.linspace(1, lower, count)
        return s * values / (1 + (s - 1) * values)

    sigmas = grid(steps, shift)
    training = grid(total, training_shift).flip(0)
    upper = torch.searchsorted(training, sigmas).clamp(max = total - 1)
    lower = (upper - 1).clamp(min = 0)
    nearest = torch.where(
        (sigmas - training[lower]).abs() <= (training[upper] - sigmas).abs(), lower, upper
    )
    return torch.unique_consecutive(training[nearest]).tolist()


def default_sigma_schedule() -> str:
    raw = (os.environ.get(SIGMAS_ENV) or "").strip().lower()
    return raw if raw in SIGMA_SCHEDULES else "sgm_uniform"


class MOSSImageScheduler(FlowMatchEulerDiscreteScheduler):
    """Flow-matching Euler at shift 5. ``sigma_schedule``: ``sgm_uniform`` (ComfyUI, the Studio default) or
    ``training_grid`` (the reference pipeline's nearest-training-sigma snapping); None reads ``UNSLOTH_MOSS_SIGMAS``."""

    @register_to_config
    def __init__(
        self,
        num_train_timesteps = 1000,
        shift = 5.0,
        training_shift = 1.0,
        sigma_schedule = None,
    ):
        super().__init__(num_train_timesteps = num_train_timesteps, shift = 1.0)
        self.register_to_config(
            shift = shift, training_shift = training_shift, sigma_schedule = sigma_schedule
        )

    def resolved_sigma_schedule(self) -> str:
        chosen = getattr(self.config, "sigma_schedule", None)
        return chosen if chosen in SIGMA_SCHEDULES else default_sigma_schedule()

    def set_timesteps(
        self,
        num_inference_steps = 50,
        device = None,
        sigmas = None,
        **kwargs,
    ):
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 1:
                raise ValueError("Steps and shifts must be positive.")
            if self.config.shift <= 0 or self.config.training_shift <= 0:
                raise ValueError("Steps and shifts must be positive.")
            if self.resolved_sigma_schedule() == "training_grid":
                sigmas = training_grid_sigmas(
                    num_inference_steps,
                    self.config.shift,
                    self.config.training_shift,
                    self.config.num_train_timesteps,
                )
            else:
                sigmas = sgm_uniform_sigmas(
                    num_inference_steps, self.config.shift, self.config.num_train_timesteps
                )
        # Already shifted: no second shift.
        self._shift = 1.0
        super().set_timesteps(device = device, sigmas = list(sigmas))


from diffusers import DiffusionPipeline  # noqa: E402


def _offloaded(module: Any) -> bool:
    """Accelerate / diffusers offload hooks move weights per call: the pre-refined text path calls a submodule outside
    the forward, so an offloaded transformer keeps the reference per-step refine."""
    if module is None:
        return True
    for sub in module.modules():
        if getattr(sub, "_hf_hook", None) is not None:
            return True
        registry = getattr(sub, "_diffusers_hook", None)
        hooks = getattr(registry, "hooks", None) or {}
        if hooks:
            return True
    return False


def comfy_noise(shape, generator, device):
    """Initial fp32 noise as ComfyUI draws it: ``torch.randn`` on the CPU from the seed (``comfy.sample.prepare_noise``).

    A CPU generator is used as given (the reference pipeline's ``randn_tensor`` draws a CPU generator's noise on the
    CPU too, so the two agree bit for bit); an accelerator generator contributes only its seed, so a Studio seed renders
    what ComfyUI renders for it. A list holds one generator per image; None draws from the global CPU stream."""
    gens = list(generator) if isinstance(generator, (list, tuple)) else [generator]

    def _cpu(g):
        if g is None or g.device.type == "cpu":
            return g
        return torch.Generator("cpu").manual_seed(int(g.initial_seed()))

    if len(gens) == 1:
        noise = torch.randn(shape, generator = _cpu(gens[0]), dtype = torch.float32)
    else:
        if len(gens) != shape[0]:
            raise ValueError(
                f"Got {len(gens)} generators for {shape[0]} images; pass one per image."
            )
        row = (1, *shape[1:])
        noise = torch.cat([torch.randn(row, generator = _cpu(g), dtype = torch.float32) for g in gens])
    return noise.to(device)


class MOSSImagePipeline(DiffusionPipeline):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = ["text_encoder", "tokenizer", "vae"]
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer,
        scheduler,
        text_encoder = None,
        tokenizer = None,
        vae = None,
    ):
        super().__init__()
        self.register_modules(
            transformer = transformer,
            scheduler = scheduler,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            vae = vae,
        )
        from diffusers.image_processor import VaeImageProcessor

        self.image_processor = VaeImageProcessor(vae_scale_factor = 16)
        self._interrupt = False

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt,
        max_sequence_length = 4096,
        device = None,
    ):
        """``(embeds [B, L, C], mask [B, L] bool)``: each prompt through the chat template, encoded unpadded on its own
        (Qwen3.5's gated-delta layers see no padding), then right-padded into one batch."""
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError("Text encoder/tokenizer required for string prompts.")
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or not all(isinstance(p, str) for p in prompts):
            raise ValueError("prompt must be a string or a nonempty list of strings.")
        device = device or self._execution_device
        rows = []
        cache: dict = {}
        for value in prompts:
            if value in cache:
                rows.append(cache[value])
                continue
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": value}],
                tokenize = False,
                add_generation_prompt = True,
                enable_thinking = True,
            )
            inputs = self.tokenizer(
                formatted, return_tensors = "pt", truncation = True, max_length = max_sequence_length
            ).to(device)
            output = self.text_encoder(**inputs, use_cache = False, return_dict = True)
            cache[value] = output.last_hidden_state[0]
            rows.append(cache[value])
        length = max(row.shape[0] for row in rows)
        embeds = rows[0].new_zeros(len(rows), length, rows[0].shape[-1])
        mask = torch.zeros(len(rows), length, device = embeds.device, dtype = torch.bool)
        for index, row in enumerate(rows):
            embeds[index, : row.shape[0]] = row
            mask[index, : row.shape[0]] = True
        return embeds, mask

    def _branch(self, embeds, mask, refine: bool) -> dict:
        """Transformer kwargs for one CFG branch. Equal-length rows need no mask (no host sync in the forward); with
        ``refine`` the text refiner runs here, once for the whole render."""
        full = mask is None or bool(mask.all())
        kwargs: dict = {}
        if full and refine:
            kwargs["refined_text"] = self.transformer.refine_text(embeds)
        elif full:
            kwargs["encoder_hidden_states"] = embeds
        else:
            kwargs["encoder_hidden_states"] = embeds
            kwargs["encoder_attention_mask"] = mask
        return kwargs

    @torch.no_grad()
    def __call__(
        self,
        prompt = None,
        *,
        height = 1024,
        width = 1024,
        num_inference_steps = 50,
        guidance_scale = 3.0,
        negative_prompt = "",
        num_images_per_prompt = 1,
        generator = None,
        latents = None,
        prompt_embeds = None,
        prompt_attention_mask = None,
        negative_prompt_embeds = None,
        negative_prompt_attention_mask = None,
        max_sequence_length = 4096,
        output_type = "pil",
        return_dict = True,
        uncond_skip_layers = DEFAULT_UNCOND_SKIP_LAYERS,
        slg_layers = None,
        slg_scale = 0.0,
        slg_start = 0.0,
        slg_stop = 1.0,
        sigmas = None,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs = ("latents",),
    ):
        """Generate images. ``uncond_skip_layers`` (default block 22) skips those zero-based blocks in the negative
        branch only (two forwards per step); None / () disables it. ``slg_layers`` with ``slg_scale > 0`` adds a third,
        skipped positive forward and the fp32 correction ``slg_scale * (cond - cond_skip)``. Both are active where
        ``slg_start <= step_index / num_steps < slg_stop``."""
        self._interrupt = False
        depth = len(self.transformer.layers)
        uncond_skip = _validate_skip_layers(uncond_skip_layers, depth, "uncond_skip_layers")
        slg = _validate_skip_layers(slg_layers, depth, "slg_layers")
        # Tuples, not frozensets: a CUDA graph keys plain values, not objects.
        uncond_skip_kw = tuple(sorted(uncond_skip))
        slg_kw = tuple(sorted(slg))
        if not math.isfinite(slg_scale) or slg_scale < 0:
            raise ValueError("slg_scale must be finite and nonnegative.")
        if not 0 <= slg_start <= slg_stop <= 1:
            raise ValueError("Require 0 <= slg_start <= slg_stop <= 1.")
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError("height and width must be positive multiples of 16.")
        if output_type not in {"pil", "np", "pt", "latent"}:
            raise ValueError("output_type must be pil, np, pt or latent.")
        if num_images_per_prompt < 1 or not math.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("Invalid image count or guidance scale.")
        if not 1 <= max_sequence_length <= 4096:
            raise ValueError("max_sequence_length must be between 1 and 4096.")
        if (prompt is None) == (prompt_embeds is None):
            raise ValueError("Provide exactly one of prompt and prompt_embeds.")
        device = self._execution_device
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self.encode_prompt(
                prompt, max_sequence_length, device
            )
        batch = prompt_embeds.shape[0]

        def prepare(embeds, mask):
            if embeds.ndim != 3 or embeds.shape[0] != batch:
                raise ValueError("Prompt embeddings must be [B, S, C] with matching batch size.")
            if mask is None:
                mask = torch.ones(embeds.shape[:2], dtype = torch.bool, device = embeds.device)
            return (
                embeds.to(device).repeat_interleave(num_images_per_prompt, 0),
                mask.to(device).repeat_interleave(num_images_per_prompt, 0),
            )

        positive, positive_mask = prepare(prompt_embeds, prompt_attention_mask)
        do_cfg = guidance_scale != 1.0
        if do_cfg:
            if negative_prompt_embeds is None:
                negative_prompt = "" if negative_prompt is None else negative_prompt
                negatives = (
                    [negative_prompt] * batch
                    if isinstance(negative_prompt, str)
                    else list(negative_prompt)
                )
                if len(negatives) != batch:
                    raise ValueError("negative_prompt must match prompt batch size.")
                negative_prompt_embeds, negative_prompt_attention_mask = self.encode_prompt(
                    negatives, max_sequence_length, device
                )
            negative, negative_mask = prepare(
                negative_prompt_embeds, negative_prompt_attention_mask
            )
        shape = (
            batch * num_images_per_prompt,
            self.transformer.config.latent_channels,
            height // 16,
            width // 16,
        )
        if latents is None:
            latents = comfy_noise(shape, generator, device)
        elif tuple(latents.shape) != shape:
            raise ValueError(f"latents must have shape {shape}.")
        else:
            latents = latents.to(device = device, dtype = torch.float32)

        refine = not _env_off(TEXT_CACHE_ENV) and not _offloaded(self.transformer)
        cond_kw = self._branch(positive, positive_mask, refine)
        uncond_kw = self._branch(negative, negative_mask, refine) if do_cfg else None

        if sigmas is not None:
            self.scheduler.set_timesteps(device = device, sigmas = list(sigmas))
        else:
            self.scheduler.set_timesteps(num_inference_steps, device = device)
        timesteps = self.scheduler.timesteps
        num_steps = len(timesteps)
        for step_index, timestep in enumerate(self.progress_bar(timesteps)):
            if self._interrupt:
                continue
            skip_active = slg_start <= step_index / num_steps < slg_stop
            # CFG combines fp32 predictions, even when the DiT runs in bf16.
            # return_dict=False: a CUDA-graph replay hands back a tuple, never a dataclass.
            prediction = self.transformer(latents, timestep, **cond_kw, return_dict = False)[
                0
            ].float()
            conditional = prediction
            if do_cfg:
                skip_kwargs = (
                    {"skip_layers": uncond_skip_kw} if uncond_skip_kw and skip_active else {}
                )
                uncond = self.transformer(
                    latents, timestep, **uncond_kw, **skip_kwargs, return_dict = False
                )[0].float()
                prediction = uncond + guidance_scale * (prediction - uncond)
            if slg_kw and slg_scale > 0 and skip_active:
                skipped = self.transformer(
                    latents, timestep, **cond_kw, skip_layers = slg_kw, return_dict = False
                )[0].float()
                if not do_cfg:
                    prediction = skipped
                prediction = prediction + slg_scale * (conditional - skipped)
            latents = self.scheduler.step(prediction, timestep, latents, return_dict = False)[0]
            if callback_on_step_end is not None:
                # A loop, not a comprehension: before Python 3.12 a comprehension's locals() is its own scope.
                callback_kwargs = {}
                for name in callback_on_step_end_tensor_inputs:
                    callback_kwargs[name] = locals()[name]
                callback_outputs = callback_on_step_end(self, step_index, timestep, callback_kwargs)
                if callback_outputs is not None:
                    latents = callback_outputs.pop("latents", latents)
        if output_type == "latent":
            images = latents
        else:
            if self.vae is None:
                raise ValueError("A FLUX.2 VAE is required for decoded output.")
            images = self.vae_decode(latents)
            images = self.image_processor.postprocess(images, output_type = output_type)
        self.maybe_free_model_hooks()
        from diffusers.pipelines.pipeline_utils import ImagePipelineOutput

        return ImagePipelineOutput(images = images) if return_dict else (images,)

    def vae_decode(self, latents):
        """[B, 128, h, w] transformer latents -> [B, 3, 16h, 16w] in [-1, 1]: undo the VAE BatchNorm statistics, unpack
        the 2x2 patches to 32 channels, decode in the VAE's dtype (fp32 by default)."""
        z = latents.to(dtype = self.vae.dtype)
        bn = self.vae.bn
        eps = self.vae.config.batch_norm_eps
        z = z * (bn.running_var + eps).sqrt().to(z).view(1, -1, 1, 1) + bn.running_mean.to(z).view(
            1, -1, 1, 1
        )
        b, c, h, w = z.shape
        z = (
            z.reshape(b, c // 4, 2, 2, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(b, c // 4, 2 * h, 2 * w)
        )
        return self.vae.decode(z, return_dict = False)[0].clamp(-1, 1)


def transformers_version_ok() -> tuple[bool, Optional[str]]:
    """``(new enough, installed version)`` for the Qwen3.5 text encoder."""
    try:
        from importlib.metadata import version
        installed = version("transformers")
    except Exception:  # noqa: BLE001 - unreadable metadata: let the import decide
        return True, None
    try:
        from packaging.version import Version
        return Version(Version(installed).base_version) >= Version(MIN_TRANSFORMERS), installed
    except Exception:  # noqa: BLE001
        return True, installed


def transformers_refusal(installed: Optional[str]) -> str:
    return (
        f"MOSS-Image needs transformers >= {MIN_TRANSFORMERS} for its Qwen3.5 text encoder "
        f"(transformers.Qwen3_5Model), but this environment has transformers {installed or 'unknown'}. "
        "Update Unsloth Studio (or run: pip install -U 'transformers>=5.3') and restart it."
    )


def assert_transformers_supported() -> None:
    ok, installed = transformers_version_ok()
    if not ok:
        raise ValueError(transformers_refusal(installed))
    try:
        from transformers import Qwen3_5Model  # noqa: F401
    except Exception:  # noqa: BLE001
        raise ValueError(transformers_refusal(installed)) from None


def _live_cache_dir() -> Optional[str]:
    try:
        from utils.hf_cache_settings import active_hf_hub_cache
        return active_hf_hub_cache()
    except Exception:  # noqa: BLE001 - outside the Studio tree the hub default applies
        return None


def _tokenizer_class(repo_id: str, common: dict[str, Any]) -> Any:
    """The tokenizer class ``tokenizer/tokenizer_config.json`` names (Qwen2Tokenizer), else ``AutoTokenizer``.
    transformers 5.x's ``AutoTokenizer`` on a repo id first reads a ROOT ``config.json``, which these repos do not
    ship, so an offline load from the cache failed before reaching the tokenizer files."""
    import json

    import transformers

    try:
        if os.path.isdir(str(repo_id)):
            path = os.path.join(str(repo_id), "tokenizer", "tokenizer_config.json")
        else:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id,
                "tokenizer_config.json",
                subfolder = "tokenizer",
                cache_dir = common.get("cache_dir"),
                token = common.get("token"),
                local_files_only = bool(common.get("local_files_only")),
            )
        with open(path, encoding = "utf-8") as fh:
            name = json.load(fh).get("tokenizer_class")
    except Exception:  # noqa: BLE001 - AutoTokenizer reports the real error
        name = None
    cls = getattr(transformers, str(name), None) if name else None
    return cls if isinstance(cls, type) else transformers.AutoTokenizer


def drop_vision_tower(text_encoder: Any) -> bool:
    """Free Qwen3.5's vision tower (~0.4 B params): MOSS feeds the encoder text only, and the text forward never touches
    it (images enter only through ``pixel_values``). True when it was dropped."""
    visual = getattr(text_encoder, "visual", None)
    if not isinstance(visual, nn.Module):
        return False
    try:
        text_encoder.visual = None
    except Exception:  # noqa: BLE001 - a frozen module keeps its tower
        return False
    return True


def _conv_offload_hook(conv: Any) -> Any:
    registry = getattr(conv, "_diffusers_hook", None)
    get = getattr(registry, "get_hook", None)
    return get("group_offloading") if callable(get) else None


def _gdn_conv_onload(module: Any, args: Any) -> None:
    hook = _conv_offload_hook(module.conv1d)
    if hook is None:
        return
    hook.pre_forward(module.conv1d)
    stream = getattr(getattr(hook, "group", None), "stream", None)
    if stream is not None:
        stream.synchronize()


def _gdn_conv_offload(module: Any, args: Any, output: Any) -> None:
    hook = _conv_offload_hook(module.conv1d)
    if hook is not None:
        hook.post_forward(module.conv1d, None)


def install_gdn_offload_guard(text_encoder: Any) -> int:
    """Qwen3.5's gated-delta layers read ``self.conv1d.weight`` straight into the causal-conv kernel and never call
    ``conv1d``, so a leaf-level group offload (Studio's text-encoder streaming) never onloads that weight. Hooks on each
    gated-delta layer onload its conv for the layer's forward and offload it after; without a group-offload hook on the
    conv they do nothing. Returns the number of layers guarded."""
    count = 0
    for module in text_encoder.modules():
        if not type(module).__name__.endswith("GatedDeltaNet") or not isinstance(
            getattr(module, "conv1d", None), nn.Module
        ):
            continue
        if getattr(module, "_unsloth_conv_guard", False):
            continue
        module.register_forward_pre_hook(_gdn_conv_onload)
        module.register_forward_hook(_gdn_conv_offload)
        module._unsloth_conv_guard = True
        count += 1
    return count


def load_moss_image_text_encoder(source: str, dtype, **common):
    from transformers import Qwen3_5Model

    text_encoder = Qwen3_5Model.from_pretrained(
        source, subfolder = "text_encoder", dtype = dtype, **common
    )
    drop_vision_tower(text_encoder)
    install_gdn_offload_guard(text_encoder)
    return text_encoder.eval()


def load_moss_image_pipeline(
    repo_id: str,
    dtype,
    hf_token: Optional[str] = None,
    transformer: Any = None,
    text_encoder: Any = None,
    local_files_only: bool = False,
    check_cancelled: Optional[Callable[[], None]] = None,
    vae_dtype: Any = None,
) -> MOSSImagePipeline:
    """A ready ``MOSSImagePipeline`` for ``repo_id`` (a repo id or a local diffusers folder), on CPU.

    Loads safetensors and JSON only, per subfolder; the repo's ``pipeline_moss_image.py`` is never imported.
    ``transformer`` / ``text_encoder`` let a quantised or pre-cast build hand in a component. The VAE decodes in fp32
    unless ``vae_dtype`` says otherwise (the reference and ComfyUI's MOSS launcher both decode in fp32)."""
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    assert_transformers_supported()
    import diffusers

    common: dict[str, Any] = {"local_files_only": local_files_only}
    cache_dir = _live_cache_dir()
    if cache_dir and not os.path.isdir(str(repo_id)):
        common["cache_dir"] = cache_dir
    if hf_token:
        common["token"] = hf_token
    tokenizer = _tokenizer_class(repo_id, common).from_pretrained(
        repo_id, subfolder = "tokenizer", **common
    )
    check_cancelled()
    if text_encoder is None:
        text_encoder = load_moss_image_text_encoder(repo_id, dtype, **common)
        check_cancelled()
    else:
        # A pre-cast build (the hosted int8 ConvRot encoder) carries the vision tower and needs the same guard.
        drop_vision_tower(text_encoder)
        install_gdn_offload_guard(text_encoder)
    if transformer is None:
        transformer = MOSSImageTransformer.from_pretrained(
            repo_id, subfolder = "transformer", torch_dtype = dtype, **common
        )
        check_cancelled()
    transformer.eval()
    vae = diffusers.AutoencoderKLFlux2.from_pretrained(
        repo_id, subfolder = "vae", torch_dtype = vae_dtype or torch.float32, **common
    )
    check_cancelled()
    scheduler = MOSSImageScheduler.from_pretrained(repo_id, subfolder = "scheduler", **common)
    logger.info("diffusion.moss_image: assembled pipeline from %s per component", repo_id)
    return MOSSImagePipeline(
        transformer = transformer,
        scheduler = scheduler,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        vae = vae,
    )


def in_tree_class(name: str) -> Optional[type]:
    """The in-tree class for a diffusers-style class name this module defines, else None."""
    return {
        MOSS_IMAGE_PIPELINE_CLASS: MOSSImagePipeline,
        MOSS_IMAGE_TRANSFORMER_CLASS: MOSSImageTransformer,
        MOSS_IMAGE_SCHEDULER_CLASS: MOSSImageScheduler,
    }.get(name)
