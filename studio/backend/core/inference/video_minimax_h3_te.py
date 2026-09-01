# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Load MiniMax-H3's Qwen3-VL conditioner from a hosted QUANTIZED checkpoint.

The H3 Diffusers path runs every component under ``ComponentsManager.enable_auto_cpu_offload``,
so its VRAM floor is the LARGEST SINGLE RESIDENT COMPONENT, not their sum. The released bfloat16
conditioner is 66.7 GB and the released denoiser is 66.3 GB, which is why seeding a 20 GB
pre-quantized denoiser moved that floor by nothing at all: the encoder simply became the largest
component instead. The encoder is the only remaining lever.

``diffusion_te_prequant`` cannot serve this one. Its artifacts are layerwise-fp8 STORAGE casts of
a dense encoder saved as ``torch.load``-able state dicts, which halve the bytes but keep every
released tensor. The hosted H3 conditioner is a different artifact in two independent ways, and
both savings stack:

  1. It carries 50 of the released 64 decoder layers, and no ``lm_head`` and no final ``norm``
     (47.97 GiB against 62.14 GiB for ``MiniMaxAI/MiniMax-H3`` ``text_encoder/``).
  2. Those 50 layers' four attention and three MLP projections are ConvRot INT8
     (25.28 GiB, i.e. 27.14 GB resident).

Dropping the tail is LOSSLESS here, not an approximation. MiniMax-H3 conditions the transformer on
``hidden_states[50]`` of the conditioner and never touches the language-model head -- see
``diffusers/modular_pipelines/minimax_h3/modular_pipeline.py`` (``text_encoder_layer`` returns 50,
"MiniMax-H3 reads `hidden_states[50]`, not the final one") and ``.../encoders.py``
``get_qwen3vl_prompt_embeds``, which calls ``text_encoder.model(..., output_hidden_states=True)``
and returns ``outputs.hidden_states[text_encoder_layer]``. Decoder layers 51-64 and ``lm_head``
therefore cannot influence the conditioning by construction. Comparing the hosted file's tensor
names against ``MiniMaxAI/MiniMax-H3``'s own shard index confirms the artifact drops EXACTLY that
set and nothing else: 902 names that map 1:1 onto the released ones, the remainder being decoder
layers 50-63 (0-based), ``model.language_model.norm.weight`` and ``lm_head.weight``.

There is one mechanical catch, and it is the reason this module builds 51 layers rather than 50.
``output_hidden_states`` yields ``[embeddings, layer_0_out, ..., layer_{N-1}_out]`` where the LAST
entry is post-``norm``. A stack truncated to exactly 50 layers therefore returns a NORMALIZED
``hidden_states[50]``, which is not the conditioning the released weights were trained against;
diffusers refuses that case outright (``encoders.py``: "The last hidden state of a stack truncated
to exactly 50 layers is post-norm"). Building a 51st slot that passes its input through unchanged
puts ``hidden_states[50]`` back where it belongs -- the raw output after 50 real layers, bit-identical
to what the full 64-layer conditioner produces there -- for no weights and no compute, and keeps
``len(layers) == config.num_hidden_layers`` honest so the diffusers guard passes on a true statement.

Best-effort throughout, exactly like ``diffusion_prequant``: any missing / unreadable / mismatched
artifact returns None and the caller loads the released bfloat16 encoder instead. Inert with
nothing configured.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Optional

# The ConvRot primitives, shared with the denoiser's hosted INT8 checkpoint. Re-exported below
# under the names this module has always used.
from .diffusion_convrot import (  # noqa: F401  (re-export: callers and tests name them here)
    build_convrot_hadamard,
    rotate_convrot_activation,
)

# The repo the prequantized denoisers already come from, so this adds no new dependency. It is no
# longer H3_COMPONENT_REPO: the VAEs moved to the GGUF mirror and the conditioner did not, so the
# two are now separate repos and aliasing them would send this file to the wrong one.
H3_TE_QUANT_REPO = "unsloth/MiniMax-H3-FP8"
# The community repack these quants were mirrored from, for an install whose cache predates the
# move. Same reasoning as H3_LEGACY_COMPONENT_REPO, and the same owner: the pairing itself lives
# in diffusion_families' _SD_CPP_LEGACY_SOURCES and is read through h3_te_quant_source below.
H3_LEGACY_TE_QUANT_REPO = "Comfy-Org/MiniMax-H3"

# Hosted quantized conditioners, by ``text_encoder_quant`` scheme.
#
# nvfp4 (``qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors``, 14.61 GiB) is deliberately absent: it is
# an AWQ NVFP4 layout with two levels of scale and a per-tensor global scale, a different loader
# and a different kernel, and shipping it needs its own numerical verification. The table is the
# one place to add it.
H3_TE_QUANT_FILES: dict[str, str] = {
    "int8": "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
}

# The scheme an UNSET ``text_encoder_quant`` resolves to on a device that supports it.
#
# The released conditioner is a 66.7 GB dense bfloat16 Qwen3-VL of 64 decoder layers and H3 reads
# ``hidden_states[50]``; transformers has no early exit, so the default runs all 64 layers and
# streams 66.7 GB across the CPU-offload boundary every generation. The hosted artifact is 27.1 GB
# over 50 layers with the same read, which is why it is the default rather than an opt-in. NOT an
# INT8 GEMM: the ConvRot forward dequantizes to the compute dtype and runs an ordinary
# ``F.linear``, so the win is bytes moved and layers executed, not faster math.
H3_TE_QUANT_DEFAULT = "int8"

# Resident bytes of each conditioner, decimal GB, measured from the safetensors headers on the Hub
# (2026-08-09) as the end of the last tensor's data. The quantized load is storage-faithful -- INT8
# stays INT8, the per-output-channel scales stay float32, the vision tower and the embedding table
# stay bfloat16 -- so the file size IS the resident size, unlike a cast-on-load path whose peak is
# the dense encoder.
#
#   qwen3vl_32b_minimax_h3_bf16.safetensors           51.51 GB  (47.97 GiB)
#   qwen3vl_32b_minimax_h3_int8_convrot.safetensors   27.14 GB  (25.28 GiB)
#
# The bfloat16 number here is the HOSTED 50-layer file. The budget default elsewhere is the
# RELEASED 64-layer encoder (66.7 GB), which is what a load without this path actually materialises.
H3_TE_QUANT_RESIDENT_GB: dict[str, float] = {
    "int8": 27.2,
}

# Which Qwen3-VL hidden state conditions the transformer. Must stay equal to
# ``MiniMaxH3ModularPipeline.text_encoder_layer``; ``h3_te_layer_budget`` below is the only reader.
H3_TE_READ_LAYER = 50

# ConvRot group size baked into the hosted checkpoint's per-tensor ``comfy_quant`` blob. Validated
# per tensor at load rather than assumed, so a re-upload under a different group is refused instead
# of silently decoding to noise.
H3_TE_CONVROT_GROUP = 256

# The quantization format the loader below implements. Any other value in a tensor's ``comfy_quant``
# blob is a format this code has not been verified against, and is refused.
_H3_TE_EXPECTED_QUANT = {"format": "int8_tensorwise", "convrot": True}

# Tensor-name suffixes that make up one quantized linear in the hosted layout.
_SCALE_SUFFIX = ".weight_scale"
_QUANT_SUFFIX = ".comfy_quant"


def h3_te_quant_scheme(mode: Optional[str]) -> Optional[str]:
    """The hosted-conditioner scheme for a requested ``text_encoder_quant``, or None.

    Pure and non-raising: the request has already been validated by ``normalize_te_quant`` at the
    route, and a scheme with no hosted artifact simply keeps the released bfloat16 encoder."""
    if mode is None:
        return None
    normalized = str(mode).strip().lower().replace("-", "_")
    return normalized if normalized in H3_TE_QUANT_FILES else None


def h3_te_quant_filename(scheme: Optional[str]) -> Optional[str]:
    """The hosted checkpoint's path inside ``H3_TE_QUANT_REPO`` for ``scheme``, or None."""
    return H3_TE_QUANT_FILES.get(scheme or "")


def h3_te_quant_source(scheme: Optional[str]) -> str:
    """The repo to fetch the hosted quantized conditioner for ``scheme`` from: our mirror, or the
    repack it was mirrored from when this install already holds that exact artifact under the old
    id.

    The conditioner's half of ``h3_component_source``, and it matters more than the VAEs do: the
    artifact is ~27 GB, and the load that wants it has already dropped the dense encoder shards
    from its pull. So on an upgraded install a live re-download is not merely slow -- offline it
    leaves the pipeline with no encoder at all, because the base snapshot it would fall back to
    was never staged either.

    PURE, and the same shared owner (``prefer_cached_legacy_source``, both cache roots) the VAEs
    use, so planning, prefetching and loading cannot name different repos.
    """
    filename = h3_te_quant_filename(scheme)
    if filename is None:
        return H3_TE_QUANT_REPO
    try:
        from .diffusion_families import prefer_cached_legacy_source
        return prefer_cached_legacy_source(H3_TE_QUANT_REPO, (filename,))
    except Exception:  # noqa: BLE001 -- an unreadable cache just means "not cached"
        return H3_TE_QUANT_REPO


def h3_te_resident_gb(scheme: Optional[str], *, bf16_gb: float) -> float:
    """Resident decimal GB of the conditioner this pick loads: the hosted quantized size when one
    exists for ``scheme``, else the released bfloat16 ``bf16_gb``."""
    resolved = h3_te_quant_scheme(scheme)
    return H3_TE_QUANT_RESIDENT_GB[resolved] if resolved else bf16_gb


# ── ConvRot INT8 ──────────────────────────────────────────────────────────────
# W_rot = W @ H_block^T offline, x_rot = x @ H_block online, and H is a NORMALIZED REGULAR
# Hadamard, which is symmetric and orthogonal, so H @ H == I exactly and
# x_rot @ W_rot^T == x @ H @ H @ W^T == x @ W^T. The rotation is what lets INT8 survive Qwen3-VL's
# per-channel activation outliers; dequantizing WITHOUT it is not an approximation, it is noise
# (measured against the hosted bfloat16 file for one projection: 0.9% relative error with the
# rotation, 137% without).
#
# This mirrors comfy-kitchen's ``_build_hadamard`` / ``_rotate_activation`` / ``_rotate_weight``
# exactly, in a few lines of torch, rather than taking a dependency on a wheel Unsloth does not ship.
#
# ``build_convrot_hadamard`` and ``rotate_convrot_activation`` are imported at the top of this
# module from ``diffusion_convrot``, where they now live. The DENOISER runs the same ConvRot on its
# own hosted INT8 checkpoint, and the two rotations have to agree with the same comfy-kitchen
# definition down to the normalizer -- two copies of a matrix nobody re-derives at review time is
# exactly how they would stop agreeing. Both stay importable from here.


@lru_cache(maxsize = None)
def _int8_convrot_linear_class() -> Any:
    """The ConvRot INT8 ``nn.Linear`` stand-in, built lazily so importing this module never imports
    torch, and built exactly ONCE.

    One load already shares a single class across every projection, so the cache is about the
    SECOND load in a process: a fresh class there is a fresh ``___check_type_id`` guard, which
    retraces every compiled block that survived the first one. Same reason the denoiser's
    ``convrot_linear_class`` is cached."""
    import torch
    from torch import nn

    class Int8ConvRotLinear(nn.Module):
        """A Linear whose weight stays INT8 in the rotated basis for its whole residency.

        The parameter names are the hosted checkpoint's own (``weight``, ``weight_scale``, ``bias``)
        so the remapped state dict loads straight in under ``strict=True``.

        The forward keeps the weight quantized and dequantizes a bfloat16 view per call (at most
        262 MB, for the 25600x5120 MLP projections) rather than holding one. That is deliberate:
        the whole point of this path is that the encoder's RESIDENT footprint is 27 GB, and a
        cached dense view would put the 51 GB back. The per-output-channel scale is applied to the
        OUTPUT, not the weight, because it factors straight out of the matmul --
        ``sum_i x_i q_oi s_o == s_o * sum_i x_i q_oi`` -- which keeps it exact in float32 over a
        tiny tensor instead of approximate in bfloat16 over a huge one. INT8 values are integers
        below 256, so the ``.to(compute dtype)`` is exact in bfloat16.

        Weight-only, unlike comfy-kitchen's W8A8 kernel, which also quantizes the activation
        per row. Same weights and the same rotation, strictly less error, and the conditioner runs
        once per generation so the dequantize is not on any hot path.
        """

        def __init__(
            self, in_features: int, out_features: int, bias: bool, group_size: int
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.group_size = group_size
            self.register_buffer(
                "weight",
                torch.empty(out_features, in_features, dtype = torch.int8, device = "meta"),
                persistent = True,
            )
            self.register_buffer(
                "weight_scale",
                torch.empty(out_features, 1, dtype = torch.float32, device = "meta"),
                persistent = True,
            )
            if bias:
                self.register_buffer(
                    "bias",
                    torch.empty(out_features, dtype = torch.bfloat16, device = "meta"),
                    persistent = True,
                )
            else:
                self.bias = None

        def forward(self, x: Any) -> Any:  # noqa: D102
            h = build_convrot_hadamard(self.group_size, device = x.device, dtype = x.dtype)
            rotated = rotate_convrot_activation(x, h, self.group_size)
            out = torch.nn.functional.linear(rotated, self.weight.to(x.dtype))
            out = out * self.weight_scale.reshape(1, -1).to(dtype = out.dtype, device = out.device)
            if self.bias is not None:
                out = out + self.bias.to(out.dtype)
            return out

        def extra_repr(self) -> str:  # noqa: D102
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, int8_convrot(group={self.group_size})"
            )

    return Int8ConvRotLinear


def _terminator_layer_class() -> Any:
    """The 51st decoder slot: passes its input through untouched. See the module docstring."""
    from torch import nn

    class H3TextEncoderTerminatorLayer(nn.Module):
        """Zero-parameter stand-in for decoder layer 50 (0-based).

        MiniMax-H3 reads ``hidden_states[50]``, the input to this slot, so nothing this returns is
        ever consumed -- it exists only so the final ``norm`` lands on an entry no one reads and
        ``hidden_states[50]`` stays the raw post-layer-50 state. Holding real weights here would
        cost ~0.5 GB and change nothing."""

        def forward(self, hidden_states: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: D102
            return hidden_states

    return H3TextEncoderTerminatorLayer


# ── name mapping ──────────────────────────────────────────────────────────────
# The hosted checkpoint uses the ComfyUI flattening of the Qwen3-VL tree; transformers nests the
# language model and the vision tower under ``model.``. Verified exhaustive on 2026-08-09: all 902
# hosted names map into ``MiniMaxAI/MiniMax-H3`` ``text_encoder/model.safetensors.index.json``, with
# nothing left over on this side.


def h3_te_remap_key(key: str) -> str:
    """A hosted checkpoint tensor name in transformers' ``Qwen3VLForConditionalGeneration`` naming."""
    if key.startswith("visual."):
        return "model." + key
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    return key


def _validate_comfy_quant(blob: Any, name: str) -> None:
    """Raise unless a tensor's ``comfy_quant`` metadata is the format this loader implements."""
    import json

    meta = json.loads(blob.cpu().numpy().tobytes().decode("utf-8").rstrip("\x00"))
    for field, expected in _H3_TE_EXPECTED_QUANT.items():
        if meta.get(field) != expected:
            raise ValueError(
                f"{name}: unsupported quant metadata {field}={meta.get(field)!r} "
                f"(this loader implements {field}={expected!r})"
            )
    group = int(meta.get("convrot_groupsize", H3_TE_CONVROT_GROUP))
    if group != H3_TE_CONVROT_GROUP:
        raise ValueError(
            f"{name}: ConvRot group {group} != the {H3_TE_CONVROT_GROUP} this loader implements"
        )


def load_h3_quantized_text_encoder(
    base: str,
    scheme: str,
    *,
    dtype: Any,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    local_base: Optional[str] = None,
    local_files_only: bool = False,
    logger: Any = None,
) -> Optional[Any]:
    """The hosted quantized Qwen3-VL conditioner for ``scheme``, on CPU, ready to seed into the
    modular pipeline; None on any problem so the caller loads the released bfloat16 encoder.

    ``base`` supplies the component CONFIG only (``<base>/text_encoder/config.json``); every weight
    comes from the hosted artifact, so the 62 GB dense encoder is never fetched. ``local_base`` is
    the already-staged snapshot of ``base`` when there is one, and it is preferred: the scoped
    pre-download keeps every component config precisely so the meta-init loaders can read them
    locally, and reading the config back out of the snapshot cannot go to the network at all.
    ``cache_dir`` pins the config resolution to the live cache root for the hub-id case, exactly as
    the artifact download above and every other loader call in this backend do -- unset, it
    resolves through huggingface_hub's import-time constant instead and can re-download into a root
    Unsloth no longer reads (or fail outright on an offline host that has already staged it).

    ``local_files_only`` is a load nobody asked for, which may not fetch anything. The artifact is
    ~27 GB, and the caller's staging phase (``_fetch_h3_te_quant``) has already accepted it -- so
    without the flag this is where that promise is broken, after the resident pipeline was evicted.
    It rides with the same other-root reuse the stager uses: the stager accepts a copy living only
    under huggingface_hub's import-time root, so a lookup pinned to ``cache_dir`` alone would refuse
    an artifact the load was cleared on and drop to the dense encoder the base pull already left
    behind. A genuine miss still returns None through the handler below.

    CPU on purpose: ``enable_auto_cpu_offload`` owns placement for every component, and a
    pre-placed encoder would only be moved again."""
    try:
        filename = h3_te_quant_filename(scheme)
        if filename is None:
            return None

        import torch
        import transformers
        from accelerate import init_empty_weights
        from safetensors import safe_open

        from utils.hf_xet_fallback import hf_hub_download_with_xet_fallback

        # The repack when this install pulled the artifact before the move, else the mirror. The
        # stager resolved the same way, so a load that was cleared on a cached copy reads it from
        # the id it is actually cached under rather than 401ing or re-pulling 27 GB.
        source_repo = h3_te_quant_source(scheme)
        path = hf_hub_download_with_xet_fallback(
            source_repo,
            filename,
            hf_token,
            cache_dir = cache_dir,
            # Resolve the artifact through whichever root holds it, exactly as the stager that
            # cleared this load did; pinned to cache_dir alone a moved cache folder re-pulls 27 GB.
            reuse_other_cache_root = True,
            local_files_only = local_files_only,
        )

        config = transformers.AutoConfig.from_pretrained(
            local_base or base,
            subfolder = "text_encoder",
            token = hf_token,
            cache_dir = cache_dir,
            # ``local_base`` is None on an offline load (the scoped base predownload stands down),
            # so this reads the hub id and would go out for the config without the flag.
            local_files_only = local_files_only,
        )
        text_config = getattr(config, "text_config", config)
        released_layers = int(getattr(text_config, "num_hidden_layers", 0))
        if released_layers <= H3_TE_READ_LAYER:
            # A base repo whose conditioner is already at or below the read layer is not the model
            # this artifact was cut from; refuse rather than build something that reads a post-norm
            # state. Cannot happen for MiniMaxAI/MiniMax-H3 (64), but the pairing is not enforced
            # anywhere else.
            raise ValueError(
                f"{base} text_encoder has {released_layers} layers; MiniMax-H3 reads "
                f"hidden_states[{H3_TE_READ_LAYER}] and needs more than that"
            )
        # See the module docstring: one slot past the read layer, so the read lands on the raw
        # state rather than the post-norm one.
        text_config.num_hidden_layers = H3_TE_READ_LAYER + 1

        # include_buffers=False on purpose (it is also accelerate's default, but the whole load
        # turns on it): PARAMETERS go to meta and are replaced wholesale by ``assign=True`` below,
        # while BUFFERS -- the rotary inverse frequencies, which no state dict carries because they
        # are non-persistent -- are built for real on CPU. They are a few KB. Putting them on meta
        # instead would leave the encoder with meta tensors after the load and force a dense CPU
        # rebuild, i.e. the 51 GB allocation this path exists to avoid.
        with init_empty_weights(include_buffers = False):
            encoder = transformers.Qwen3VLForConditionalGeneration(config)

        language_model = encoder.model.language_model
        language_model.layers[H3_TE_READ_LAYER] = _terminator_layer_class()()
        # Neither is in the artifact and neither can reach the conditioning: ``norm`` is only ever
        # applied to the terminator's output, and the head is never called (the pipeline invokes
        # ``text_encoder.model``, not ``text_encoder``). Dropping the head alone saves 1.56 GB.
        language_model.norm = torch.nn.Identity()
        encoder.lm_head = torch.nn.Identity()

        int8_linear_cls = _int8_convrot_linear_class()
        with safe_open(path, framework = "pt", device = "cpu") as handle:
            names = set(handle.keys())
            quantized = {
                name[: -len(_SCALE_SUFFIX)] for name in names if name.endswith(_SCALE_SUFFIX)
            }
            # Swap every quantized projection for the INT8 module BEFORE loading, so the state dict
            # lands on buffers of the right dtype instead of being cast into bfloat16 Linears.
            for prefix in sorted(quantized):
                quant_key = prefix + _QUANT_SUFFIX
                if quant_key not in names:
                    raise ValueError(f"{prefix}: quantized weight with no {_QUANT_SUFFIX} metadata")
                _validate_comfy_quant(handle.get_tensor(quant_key), prefix)
                target = h3_te_remap_key(prefix)
                parent_path, _, leaf = target.rpartition(".")
                parent = encoder.get_submodule(parent_path)
                existing = getattr(parent, leaf)
                slice_ = handle.get_slice(prefix + ".weight")
                out_features, in_features = slice_.get_shape()
                if (existing.out_features, existing.in_features) != (out_features, in_features):
                    raise ValueError(
                        f"{target}: checkpoint shape {(out_features, in_features)} != model "
                        f"{(existing.out_features, existing.in_features)}"
                    )
                setattr(
                    parent,
                    leaf,
                    int8_linear_cls(
                        in_features,
                        out_features,
                        bias = (prefix + ".bias") in names,
                        group_size = H3_TE_CONVROT_GROUP,
                    ),
                )

            # The INT8 payload and its float32 scales are STORAGE and keep their dtypes; everything
            # still dense (the vision tower, the embedding table, the norms) follows the pipeline's
            # compute dtype, exactly as ``load_components(dtype=...)`` would have set it had this
            # component been loaded the ordinary way. A no-op for the bfloat16 H3 runs.
            state_dict = {}
            for name in names:
                if name.endswith(_QUANT_SUFFIX):
                    continue
                tensor = handle.get_tensor(name)
                if (
                    not name.endswith(_SCALE_SUFFIX)
                    and tensor.is_floating_point()
                    and dtype is not None
                ):
                    tensor = tensor.to(dtype)
                state_dict[h3_te_remap_key(name)] = tensor

        # strict=True proves the artifact and the meta-initialised skeleton describe the same 902
        # tensors, so a re-upload that renames, adds or drops one fails the load instead of
        # silently conditioning on partly random weights. It does NOT prove they are quantized:
        # a projection re-uploaded as a plain dense ``weight`` with its scale and metadata removed
        # simply drops out of the set above, leaves the original bfloat16 Linear in place, and
        # loads cleanly under strict. The load would then be recorded as engaged int8 while the
        # resident encoder crept back toward the dense 51 GB, and the VRAM preflight -- which
        # sizes the floor from the ENGAGED scheme -- would clear a generation that cannot fit.
        #
        # Checked structurally rather than against a count, so it stays true if the layer or
        # projection set ever changes: no ordinary Linear may survive inside the decoder stack.
        # The vision tower keeps its own dense Linears and is not in scope.
        dense = [
            name
            for name, module in language_model.layers.named_modules()
            if isinstance(module, torch.nn.Linear)
        ]
        if dense:
            raise ValueError(
                f"{len(dense)} decoder projection(s) are not quantized in this artifact, "
                f"e.g. {', '.join(sorted(dense)[:3])}; it is not the {scheme} checkpoint this "
                f"path budgets for"
            )
        encoder.load_state_dict(state_dict, strict = True, assign = True)
        # Nothing may be left on meta. A dense CPU rebuild is NOT an acceptable repair here (it is
        # the 51 GB allocation this path exists to avoid), so a leftover is a refusal and the caller
        # falls back to the released encoder, which at least loads correctly.
        stranded = _meta_tensor_names(encoder)
        if stranded:
            raise ValueError(
                f"{len(stranded)} tensor(s) still on the meta device after loading, "
                f"e.g. {', '.join(stranded[:3])}"
            )
        encoder.eval()
        if logger is not None:
            logger.info(
                "video.h3_te_quant: loaded the hosted %s conditioner (%s, %s), "
                "%d ConvRot INT8 projections over %d decoder layers",
                scheme,
                source_repo,
                filename,
                len(quantized),
                H3_TE_READ_LAYER,
            )
        return encoder
    except Exception as exc:  # noqa: BLE001 -- fall back to the released bfloat16 encoder
        if logger is not None:
            logger.warning(
                "video.h3_te_quant: hosted %s conditioner unusable (%s); "
                "loading the released bfloat16 encoder instead",
                scheme,
                exc,
            )
        return None


def _meta_tensor_names(module: Any) -> list[str]:
    """Names of every parameter and buffer still on the meta device."""
    from itertools import chain
    return [
        name
        for name, tensor in chain(module.named_parameters(), module.named_buffers())
        if getattr(tensor, "is_meta", False)
    ]
