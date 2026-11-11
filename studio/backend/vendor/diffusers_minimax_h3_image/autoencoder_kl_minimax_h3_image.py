# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations  # Unsloth patch: PEP 604 annotations must not evaluate on Python 3.9.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin, get_parameter_dtype
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.MiniMaxH3VideoCausalConv3d with MiniMaxH3Video->MiniMaxH3Image
class MiniMaxH3ImageCausalConv3d(nn.Conv3d):
    r"""
    3D convolution used throughout the MiniMax-H3 video encoder.

    Spatial padding is symmetric and uses `spatial_padding_mode` (`"reflect"` in the released checkpoint); temporal
    padding is causal, i.e. `kernel_size_t - 1` zero frames are prepended and nothing is appended.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding > 0:
            padding = self.spatial_padding
            hidden_states = F.pad(
                hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode
            )
        if self.temporal_padding > 0:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        return F.conv3d(hidden_states, self.weight, self.bias, stride=self.stride, padding=0, dilation=self.dilation)


# Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.MiniMaxH3VideoGroupNorm with MiniMaxH3Video->MiniMaxH3Image
class MiniMaxH3ImageGroupNorm(nn.GroupNorm):
    r"""
    Group normalization applied to each latent frame in isolation (`use_t_isolated_gn` in the original config): the
    temporal axis is folded into the batch axis so statistics never mix across frames.
    """

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).contiguous()
        hidden_states = hidden_states.view(batch_size * num_frames, num_channels, 1, height, width)
        hidden_states = super().forward(hidden_states)
        hidden_states = hidden_states.view(batch_size, num_frames, num_channels, height, width)
        return hidden_states.permute(0, 2, 1, 3, 4).contiguous()


# Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.MiniMaxH3VideoResnetBlock3d with MiniMaxH3Video->MiniMaxH3Image
class MiniMaxH3ImageResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = MiniMaxH3ImageGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3ImageCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3ImageGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3ImageCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3ImageCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = F.silu(self.norm1(hidden_states))
        hidden_states = self.conv1(hidden_states)
        hidden_states = F.silu(self.norm2(hidden_states))
        hidden_states = self.conv2(hidden_states)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return residual + hidden_states


# Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.MiniMaxH3VideoDownsample3d with MiniMaxH3Video->MiniMaxH3Image
class MiniMaxH3ImageDownsample3d(nn.Module):
    r"""
    Strided 3x3x3 downsampling convolution. A spatial stride of 2 is preceded by an asymmetric bottom/right pad of 1
    (the convolution itself carries no spatial padding), so the output is exactly `ceil(size / 2)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3ImageCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.spatial_stride == 2:
            hidden_states = F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode)
        return self.conv(hidden_states)


# Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.MiniMaxH3VideoDownBlock3d with MiniMaxH3Video->MiniMaxH3Image
class MiniMaxH3ImageDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                MiniMaxH3ImageResnetBlock3d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(num_layers)
            ]
        )
        self.downsamplers = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsamplers = nn.ModuleList(
                [
                    MiniMaxH3ImageDownsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                        spatial_padding_mode=spatial_padding_mode,
                    )
                ]
            )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
            else:
                hidden_states = resnet(hidden_states)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)
        return hidden_states


# Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.MiniMaxH3VideoEncoder3d with MiniMaxH3Video->MiniMaxH3Image
class MiniMaxH3ImageEncoder3d(nn.Module):
    r"""
    Causal 3D CNN encoder. `block_out_channels` gives the channel count of every level; the per-level
    `spatial_downsample_factors` / `temporal_downsample_factors` multiply out to the total compression ratios.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.conv_in = MiniMaxH3ImageCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down_blocks = nn.ModuleList(
            [
                MiniMaxH3ImageDownBlock3d(
                    in_channels=block_in_channels[i],
                    out_channels=block_out_channels[i],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(len(block_out_channels))
            ]
        )

        self.norm_out = MiniMaxH3ImageGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3ImageCausalConv3d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)
        hidden_states = F.silu(self.norm_out(hidden_states))
        return self.conv_out(hidden_states)


# Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.MiniMaxH3VideoRotaryPosEmbed with MiniMaxH3Video->MiniMaxH3Image
class MiniMaxH3ImageRotaryPosEmbed(nn.Module):
    r"""
    3-axis rotary embedding for the ViT decoder. Coordinates are length-normalized to `[-1, 1)` per axis and scaled by
    `2 * pi`, and the resulting `(t, h, w)` angles are concatenated and then duplicated, so the first `rope_dim_ratio *
    attention_head_dim` channels of every head are rotated.
    """

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"`dim` {dim} must be divisible by `2 * num_axes` {2 * num_axes}.")
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return angles.cos(), angles.sin()


# Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.MiniMaxH3VideoAttnProcessor with MiniMaxH3Video->MiniMaxH3Image
class MiniMaxH3ImageAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "MiniMaxH3ImageAttention",
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))

        # The reference normalizes Q/K in float32 regardless of the compute dtype.
        query = attn.norm_q(query.float()).to(query.dtype)
        key = attn.norm_k(key.float()).to(key.dtype)

        if rotary_emb is not None:
            cos, sin = rotary_emb
            cos = cos.to(query.dtype)
            sin = sin.to(query.dtype)
            rotary_dim = cos.shape[-1]
            query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
            key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
            query_first, query_second = query_rotary.chunk(2, dim=-1)
            key_first, key_second = key_rotary.chunk(2, dim=-1)
            query_rotated = torch.cat([-query_second, query_first], dim=-1)
            key_rotated = torch.cat([-key_second, key_first], dim=-1)
            query = torch.cat([query_rotary * cos + query_rotated * sin, query_pass], dim=-1)
            key = torch.cat([key_rotary * cos + key_rotated * sin, key_pass], dim=-1)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        return attn.to_out[0](hidden_states)


# Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.MiniMaxH3VideoAttention with MiniMaxH3Video->MiniMaxH3Image
class MiniMaxH3ImageAttention(nn.Module, AttentionModuleMixin):
    _default_processor_cls = MiniMaxH3ImageAttnProcessor
    _available_processors = [MiniMaxH3ImageAttnProcessor]

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.use_bias = bias
        inner_dim = heads * dim_head

        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(dim, inner_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(inner_dim, dim, bias=bias), nn.Dropout(0.0)])

        self.set_processor(MiniMaxH3ImageAttnProcessor())

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb)


# Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.MiniMaxH3VideoTransformerBlock with MiniMaxH3Video->MiniMaxH3Image
class MiniMaxH3ImageTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3ImageAttention(dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = FeedForward(dim, mult=ffn_mult, activation_fn="swiglu", bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self, hidden_states: torch.Tensor, rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        # The reference normalizes in float32 regardless of the compute dtype. The residual stream is float32 too
        # because the anchors are, so the normed activation follows the projections instead.
        compute_dtype = get_parameter_dtype(self.attn.to_q)
        norm_hidden_states = self.norm1(hidden_states.float()).to(compute_dtype)
        hidden_states = hidden_states + self.attn(norm_hidden_states, rotary_emb) * self.scale1
        norm_hidden_states = self.norm2(hidden_states.float()).to(compute_dtype)
        hidden_states = hidden_states + self.ff(norm_hidden_states) * self.scale2
        return hidden_states


class MiniMaxH3ImageViTDecoder2d(nn.Module):
    r"""
    Non-causal 2D ViT decoder. Every latent pixel becomes one token; `num_register_tokens` learned register tokens plus
    a single all-zero token are appended (all at position `0`), attended over with full self-attention, and dropped
    again before the patch projection expands each token into a `patch_size x patch_size` pixel block. The rotary
    embedding carries `(h, w)` coordinates only.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        num_layers: int = 24,
        num_attention_heads: int = 16,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.5,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.rope = MiniMaxH3ImageRotaryPosEmbed(
            int(attention_head_dim * rope_dim_ratio), theta=rope_theta, num_axes=2
        )
        self.proj_in = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3ImageTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size * patch_size)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_size, height * width, num_channels)
        hidden_states = self.proj_in(hidden_states)
        num_patches = hidden_states.shape[1]

        # `proj_in` is pinned to float32 and `register_tokens` is not, so align rather than rely on promotion.
        register_tokens = self.register_tokens.expand(batch_size, -1, -1).to(hidden_states.dtype)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 1)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 2))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(block, hidden_states, rotary_emb)
            else:
                hidden_states = block(hidden_states, rotary_emb)

        # `norm_out` is pinned to float32 and `proj_out` is not.
        hidden_states = self.norm_out(hidden_states).to(get_parameter_dtype(self.proj_out))
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]

        patch_size = self.patch_size
        hidden_states = hidden_states.view(batch_size, height, width, self.out_channels, patch_size, patch_size)
        hidden_states = hidden_states.permute(0, 3, 1, 4, 2, 5).contiguous()
        return hidden_states.reshape(batch_size, self.out_channels, height * patch_size, width * patch_size)


class AutoencoderKLMiniMaxH3Image(ModelMixin, ConfigMixin, AttentionMixin, AutoencoderMixin):
    r"""
    A VAE model with a causal 3D CNN encoder and a non-causal 2D ViT decoder, used in
    [MiniMax-H3-Image](https://huggingface.co/MiniMaxAI). The encoder is the MiniMax-H3 video encoder run over a single
    frame; the decoder is an image-only ViT.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Latents are normalized with per-channel `latents_mean` / `latents_std` rather than a `scaling_factor`; a pipeline
    encodes with `(latent - latents_mean) / latents_std` and decodes with `latent * latents_std + latents_mean`.

    The pixel convention is ImageNet-normalized RGB over a `[0, 1]` base range, not the usual `[-1, 1]`: `encode`
    expects `(pixel - imagenet_mean) / imagenet_std` and `decode` returns values in that same space, so a pipeline has
    to apply `sample * imagenet_std + imagenet_mean` (mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`) and
    clamp to `[0, 1]` before postprocessing.

    Unlike most autoencoders in the library, spatial tiling is **on by default**: MiniMax-H3-Image was released with
    tiling enabled for both encoding and decoding, and the released images are the blended-tile ones, so disabling
    tiling changes the output. Use `enable_tiling` to change the tile geometry, `disable_tiling` to turn it off.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxH3ImageResnetBlock3d", "MiniMaxH3ImageTransformerBlock"]
    _repeated_blocks = ["MiniMaxH3ImageTransformerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    # The released checkpoint is float32; only the cast-sensitive modules are pinned so `torch_dtype` still reaches
    # the decoder block stack.
    _keep_in_fp32_modules = [
        "encoder",
        "quant_conv",
        "norm1",
        "norm2",
        "norm_out",
        "scale1",
        "scale2",
    ]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
        decoder_num_layers: int = 24,
        decoder_num_attention_heads: int = 16,
        decoder_attention_head_dim: int = 64,
        decoder_num_register_tokens: int = 4,
        decoder_ffn_mult: int = 4,
        decoder_rope_theta: float = 100.0,
        decoder_rope_dim_ratio: float = 0.5,
        decoder_norm_eps: float = 1e-5,
        latents_mean: tuple[float, ...] = (0.0,) * 24,
        latents_std: tuple[float, ...] = (1.0,) * 24,
    ) -> None:
        super().__init__()

        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)

        self.encoder = MiniMaxH3ImageEncoder3d(
            in_channels=in_channels,
            out_channels=2 * latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            spatial_downsample_factors=spatial_downsample_factors,
            temporal_downsample_factors=temporal_downsample_factors,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3ImageViTDecoder2d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=self.spatial_compression_ratio,
            num_layers=decoder_num_layers,
            num_attention_heads=decoder_num_attention_heads,
            attention_head_dim=decoder_attention_head_dim,
            num_register_tokens=decoder_num_register_tokens,
            ffn_mult=decoder_ffn_mult,
            rope_theta=decoder_rope_theta,
            rope_dim_ratio=decoder_rope_dim_ratio,
            norm_eps=decoder_norm_eps,
        )

        # When decoding a batch of latents at a time, one can save memory by slicing across the batch dimension to
        # perform decoding of a single latent at a time.
        self.use_slicing = False

        # Large images are encoded and decoded in overlapping tiles that are linearly blended back together.
        # MiniMax-H3-Image ships with tiling enabled.
        self.use_tiling = True

        # The tile size in pixel space, and the minimum overlap between two neighbouring tiles. The actual overlaps are
        # widened (in multiples of `spatial_compression_ratio`) so that the tiles cover the image exactly.
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    # Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3.enable_tiling
    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_overlap_height: int | None = None,
        tile_sample_min_overlap_width: int | None = None,
    ) -> None:
        r"""
        Enable tiled VAE encoding/decoding. When this option is enabled, the VAE splits the frames into tiles, encodes
        or decodes each tile separately and linearly blends the overlaps back together. This lowers the memory
        requirement and allows processing larger frames.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The tile height in pixel space. Frames taller than this are split along the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The tile width in pixel space. Frames wider than this are split along the width dimension.
            tile_sample_min_overlap_height (`int`, *optional*):
                The minimum overlap, in pixels, between two consecutive vertical tiles.
            tile_sample_min_overlap_width (`int`, *optional*):
                The minimum overlap, in pixels, between two consecutive horizontal tiles.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_overlap_height = tile_sample_min_overlap_height or self.tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = tile_sample_min_overlap_width or self.tile_sample_min_overlap_width

    # Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3._split_tiles
    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        r"""
        Lay `tile_size`-wide tiles over `length` pixels. The number of tiles is the smallest one whose union can cover
        `length` while keeping every overlap at least `min_overlap`; the slack is then distributed round-robin over the
        overlaps in whole `spatial_compression_ratio` steps so that every tile boundary stays latent-aligned.
        """
        if tile_size >= length:
            return [0], [length], []

        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1

        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio

        tile_start_indices = [0]
        for i in range(num_tiles - 1):
            tile_start_indices.append(tile_start_indices[-1] + tile_size - overlaps[i])
        return tile_start_indices, [tile_size] * num_tiles, overlaps

    # Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3._blend
    def _blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    # Copied from diffusers.models.autoencoders.autoencoder_kl_minimax_h3.AutoencoderKLMiniMaxH3._stitch_tiles
    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # The encoder is a causal video encoder; an image is a single frame, which maps to a single latent frame.
        if not self.use_tiling:
            return self.quant_conv(self.encoder(x[:, :, None]))[:, :, -1]

        height, width = x.shape[-2], x.shape[-1]
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                tile = x[..., i_pos : i_pos + i_len, j_pos : j_pos + j_len]
                row.append(self.quant_conv(self.encoder(tile[:, :, None]))[:, :, -1])
            rows.append(row)

        latent_y_overlaps = [overlap // self.spatial_compression_ratio for overlap in y_overlaps]
        latent_x_overlaps = [overlap // self.spatial_compression_ratio for overlap in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))

        # Tiles are laid out in pixel space and then mapped back onto the latent grid.
        height = z.shape[-2] * self.spatial_compression_ratio
        width = z.shape[-1] * self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        ratio = self.spatial_compression_ratio
        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                tile = z[
                    ...,
                    i_pos // ratio : i_pos // ratio + i_len // ratio,
                    j_pos // ratio : j_pos // ratio + j_len // ratio,
                ]
                row.append(self.decoder(self.post_quant_conv(tile)))
            rows.append(row)

        return self._stitch_tiles(rows, y_overlaps, x_overlaps)

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True) -> AutoencoderKLOutput | tuple[torch.Tensor]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`):
                Input batch of images, shape `(batch_size, in_channels, height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.autoencoder_kl.AutoencoderKLOutput`] instead of a plain
                tuple.

        Returns:
            The latent distribution of the encoded images, over latents of shape `(batch_size, latent_channels, height
            // spatial_compression_ratio, width // spatial_compression_ratio)`. Note that MiniMax-H3-Image normalizes
            the sampled latents with `latents_mean` / `latents_std` afterwards.
        """
        # The encoder is pinned to float32 by `_keep_in_fp32_modules`; align lower-precision pixels with the weights.
        x = x.to(get_parameter_dtype(self.encoder))
        if self.use_slicing and x.shape[0] > 1:
            moments = torch.cat([self._encode(x_slice) for x_slice in x.split(1)])
        else:
            moments = self._encode(x)
        posterior = DiagonalGaussianDistribution(moments)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Decode a batch of latents.

        Args:
            z (`torch.Tensor`):
                Input batch of latents, shape `(batch_size, latent_channels, height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                The decoded images, shape `(batch_size, out_channels, height * spatial_compression_ratio, width *
                spatial_compression_ratio)`.
        """
        # The decoder is mixed precision, so align with the first module that consumes the latents.
        z = z.to(get_parameter_dtype(self.post_quant_conv))
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice) for z_slice in z.split(1)])
        else:
            decoded = self._decode(z)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Encode then decode a batch of images.

        Args:
            sample (`torch.Tensor`):
                Input batch of images, shape `(batch_size, in_channels, height, width)`.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample the posterior instead of taking its mode.
            generator (`torch.Generator`, *optional*):
                Generator used when `sample_posterior=True`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                The round-tripped images, shape `(batch_size, out_channels, height, width)`.
        """
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
