# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Separate text-encoder / VAE files (ComfyUI ``models/text_encoders``, ``models/vae``) beside a single-file DiT.

Each file is classified from its header, assigned to the base pipeline slot whose class it fits, renamed under a
rule that loads strictly, and built from the base repo's config (which still supplies configs and tokenizers).
fp8 dequantizes, int8 ConvRot stays int8, every other ComfyUI quant format is refused by name.
"""

from __future__ import annotations

import contextvars
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

COMPONENT_VAE = "vae"
TEXT_ENCODER_COMPONENTS = ("text_encoder", "text_encoder_2", "text_encoder_3", "text_encoder_4")
MAX_TEXT_ENCODER_FILES = len(TEXT_ENCODER_COMPONENTS)
COMPONENT_FILE_SUFFIX = ".safetensors"
# Header metadata keys / blobs ComfyUI files carry beside the weights (tokenizer models, the legacy fp8 marker).
_NON_WEIGHT_KEYS = frozenset({"scaled_fp8", "tekken_model", "spiece_model"})
_QUANT_COMPANION_SUFFIXES = (
    ".comfy_quant",
    ".weight_scale",
    ".weight_scale_2",
    ".input_scale",
    ".pre_quant_scale",
    ".weight_s_rel",
    ".weight_s_channel",
    ".weight_codebook",
    ".scale_weight",
    ".scale_input",
)

TE_KIND_CLASSES: dict[str, frozenset] = {
    "clip_l": frozenset({"CLIPTextModel", "CLIPTextModelWithProjection"}),
    "clip_g": frozenset({"CLIPTextModelWithProjection", "CLIPTextModel"}),
    "t5": frozenset({"T5EncoderModel"}),
    "umt5": frozenset({"UMT5EncoderModel"}),
    "qwen3": frozenset({"Qwen3Model", "Qwen3ForCausalLM"}),
    "qwen2_5_vl": frozenset({"Qwen2_5_VLForConditionalGeneration", "Qwen2_5_VLModel"}),
    "qwen3_vl": frozenset({"Qwen3VLForConditionalGeneration", "Qwen3VLModel"}),
    "mistral3": frozenset({"Mistral3ForConditionalGeneration", "Mistral3Model"}),
    "llama": frozenset({"LlamaForCausalLM", "LlamaModel"}),
    "gemma2": frozenset({"Gemma2Model", "Gemma2ForCausalLM"}),
    "gemma3": frozenset({"Gemma3ForConditionalGeneration", "Gemma3Model", "Gemma3ForCausalLM"}),
}
# File kind -> diffusers VAE classes it can fill. Qwen-Image's VAE is the Wan 2.1 layout.
VAE_KIND_CLASSES: dict[str, frozenset] = {
    "ldm_kl": frozenset({"AutoencoderKL", "AutoencoderKLFlux2"}),
    "wan": frozenset({"AutoencoderKLWan", "AutoencoderKLQwenImage"}),
    # Wan 2.2 / Qwen-Image-2.1: residual blocks and the resampler nested under one ``downsamples.<i>`` stage.
    "wan_nested": frozenset({"AutoencoderKLWan", "AutoencoderKLQwenImage21"}),
    "diffusers": frozenset(
        {
            "AutoencoderKL",
            "AutoencoderKLFlux2",
            "AutoencoderKLWan",
            "AutoencoderKLQwenImage",
            "AutoencoderKLQwenImage21",
        }
    ),
}

# Tensors a ComfyUI file may carry that the diffusers-side class never reads. Everything else must match.
_DEAD_UNEXPECTED = (
    re.compile(r"(^|\.)position_ids$"),
    re.compile(r"^logit_scale$"),
    re.compile(r"^text_projection\.weight$"),  # full-CLIP export into CLIPTextModel
    re.compile(r"^lm_head\.weight$"),  # class without (or a trimmed) lm_head
    re.compile(r"(^|\.)num_batches_tracked$"),
)
_DEAD_MISSING = (
    re.compile(r"(^|\.)position_ids$"),
    re.compile(r"^encoder\.embed_tokens\.weight$"),  # T5 / UMT5: tied to shared.weight
)


class ComponentFileError(ValueError):
    """A supplied text-encoder / VAE file this load cannot use; the message names the file and why."""


@dataclass(frozen = True)
class ComponentFileRef:
    """One request-supplied file: a local path, or ``owner/repo/path/in/repo.safetensors`` on the Hub."""

    spec: str
    local_path: Optional[str] = None
    repo_id: Optional[str] = None
    filename: Optional[str] = None

    @property
    def name(self) -> str:
        return os.path.basename(self.local_path or self.filename or self.spec)

    @property
    def is_hub(self) -> bool:
        return self.repo_id is not None


def normalize_text_encoder_files(value: Any) -> list[str]:
    """The request's ``text_encoder_file`` (one string or a list) as a deduplicated list of specs."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise ComponentFileError("text_encoder_file entries must be strings")
        item = item.strip()
        if item and item not in out:
            out.append(item)
    if len(out) > MAX_TEXT_ENCODER_FILES:
        raise ComponentFileError(
            f"at most {MAX_TEXT_ENCODER_FILES} text-encoder files can be supplied; got {len(out)}"
        )
    return out


def _path_shaped(spec: str) -> bool:
    return (
        spec.startswith(("/", "\\", "~", "."))
        or "\\" in spec
        or bool(re.match(r"^[A-Za-z]:[\\/]", spec))
        or Path(spec).is_absolute()
    )


def parse_component_file(
    spec: str,
    *,
    model_path: Optional[str] = None,
    trusted_repo: Optional[Callable[[str], bool]] = None,
    what: str = "text_encoder_file",
) -> ComponentFileRef:
    """Validate one spec without touching the network. Local: an existing ``.safetensors`` file (a relative or
    bare name resolves against a local ``model_path`` directory, so a ComfyUI ``models/`` tree works with
    ``../text_encoders/x.safetensors``). Hub: ``owner/repo/path.safetensors`` held to ``trusted_repo``, the
    same bar a single-file DiT clears."""
    spec = (spec or "").strip()
    if not spec:
        raise ComponentFileError(f"{what} is empty")
    if not spec.lower().endswith(COMPONENT_FILE_SUFFIX):
        raise ComponentFileError(
            f"{what} '{os.path.basename(spec)}' is not a .safetensors file; only .safetensors "
            "text-encoder / VAE files are supported"
        )
    root: Optional[Path] = None
    if model_path:
        try:
            candidate = Path(model_path).expanduser()
            if candidate.is_dir():
                root = candidate
            elif candidate.is_file():
                root = candidate.parent
        except OSError:
            root = None
    local: Optional[Path] = None
    if root is not None and not _path_shaped(spec) and (root / spec).is_file():
        local = (
            root / spec
        )  # an existing file under model_path wins over the owner/repo/file reading
    elif _path_shaped(spec):
        local = Path(spec).expanduser()
        if not local.is_absolute():
            if root is None:
                raise ComponentFileError(
                    f"{what} '{spec}' is a relative path, but model_path is not a local directory "
                    "to resolve it against; pass an absolute path"
                )
            local = root / local
    elif "/" not in spec.replace("\\", "/"):
        if root is None:
            raise ComponentFileError(
                f"{what} '{spec}' names no directory; pass an absolute path or owner/repo/file.safetensors"
            )
        local = root / spec
    if local is not None:
        try:
            resolved = local.resolve()
        except OSError as exc:
            raise ComponentFileError(f"{what} '{spec}' cannot be resolved: {exc}") from exc
        if not resolved.is_file():
            raise ComponentFileError(f"{what} '{spec}' does not exist or is not a file")
        return ComponentFileRef(spec = spec, local_path = str(resolved))
    parts = spec.replace("\\", "/").split("/")
    if len(parts) < 3 or not all(parts):
        raise ComponentFileError(
            f"{what} '{spec}' is neither a local file nor owner/repo/path/to/file.safetensors"
        )
    repo_id, filename = "/".join(parts[:2]), "/".join(parts[2:])
    if trusted_repo is not None and not trusted_repo(repo_id):
        raise ComponentFileError(
            f"{what} from the Hub is restricted to the same repos as a single-file transformer "
            f"(unsloth/* or a local path); got '{repo_id}'"
        )
    return ComponentFileRef(spec = spec, repo_id = repo_id, filename = filename)


def read_safetensors_header(path: str) -> tuple[dict[str, dict], dict]:
    """``({tensor: {"dtype", "shape", "data_offsets"}}, metadata)`` from the file header alone."""
    from .diffusion_comfy_quant import _read_header

    try:
        header, _ = _read_header(str(path))
    except Exception as exc:  # noqa: BLE001
        raise ComponentFileError(
            f"'{os.path.basename(path)}' is not a readable safetensors file: {exc}"
        )
    metadata = header.pop("__metadata__", None) or {}
    return header, metadata if isinstance(metadata, dict) else {}


def weight_keys(header: dict[str, Any]) -> list[str]:
    """The header's weight tensors: quant companions, tokenizer blobs and markers left out."""
    return [
        k for k in header if k not in _NON_WEIGHT_KEYS and not k.endswith(_QUANT_COMPANION_SUFFIXES)
    ]


def classify_text_encoder(
    keys: Iterable[str], shapes: Optional[dict[str, Sequence[int]]] = None
) -> Optional[str]:
    """The text-encoder kind of a ComfyUI / original-layout header, or None when it is not one."""
    keyset = set(keys)

    def has(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in keyset)

    if has("text_model.encoder.layers."):
        width = None
        shape = (shapes or {}).get("text_model.embeddings.token_embedding.weight")
        if shape and len(shape) == 2:
            width = int(shape[1])
        return "clip_g" if width == 1280 else "clip_l"
    if has("encoder.block.") and ("shared.weight" in keyset or has("encoder.embed_tokens.")):
        umt5 = any(
            re.match(
                r"^encoder\.block\.[1-9]\d*\.layer\.0\.SelfAttention\.relative_attention_bias\.", k
            )
            for k in keyset
        )
        return "umt5" if umt5 else "t5"
    lm_prefix = "model.language_model." if has("model.language_model.layers.") else "model."
    if not has(lm_prefix + "layers."):
        return None
    q_norm = has(lm_prefix + "layers.0.self_attn.q_norm.")
    if has(lm_prefix + "layers.0.pre_feedforward_layernorm."):
        return (
            "gemma3" if (q_norm or has("vision_model.") or has("model.vision_tower.")) else "gemma2"
        )
    if has("visual.") or has("model.visual."):
        return "qwen3_vl" if q_norm else "qwen2_5_vl"
    if (
        has("vision_tower.")
        or has("model.vision_tower.")
        or has("multi_modal_projector.")
        or "tekken_model" in keyset
    ):
        return "mistral3"
    if q_norm:
        return "qwen3"
    return "llama"


def classify_vae(keys: Iterable[str]) -> Optional[str]:
    keyset = set(keys)

    def has(prefix: str) -> bool:
        return any(k.startswith(prefix) for k in keyset)

    if any(
        re.match(r"^(encoder\.downsamples|decoder\.upsamples)\.\d+\.(downsamples|upsamples)\.", k)
        for k in keyset
    ):
        return "wan_nested"
    if has("decoder.upsamples.") or has("encoder.downsamples."):
        return "wan"
    if has("decoder.up.") or has("encoder.down.") or has("first_stage_model.decoder.up."):
        return "ldm_kl"
    if has("decoder.up_blocks.") or has("encoder.down_blocks."):
        return "diffusers"
    return None


def _rule_identity(key: str) -> str:
    return key


def _rule_strip_model(key: str) -> Optional[str]:
    # ``Qwen3Model`` / ``LlamaModel`` / ``Gemma2Model`` are the decoder without the causal-LM wrapper.
    if key.startswith("model."):
        return key[len("model.") :]
    if key.startswith("lm_head."):
        return key  # left unexpected; dead for a headless class
    return key


_VL_KEEP = (
    "model.visual.",
    "model.language_model.",
    "model.vision_tower.",
    "model.multi_modal_projector.",
)


def _rule_nest_language_model(key: str) -> str:
    # transformers >= 4.52 (and 5.x) nest the decoder under ``model.language_model`` and the vision tower under
    # ``model.visual`` / ``model.vision_tower``; ComfyUI keeps the flat pre-4.52 layout.
    if key.startswith(_VL_KEEP) or key.startswith("lm_head."):
        return key
    if key.startswith("visual."):
        return "model." + key
    if key.startswith("vision_tower."):
        return "model." + key
    if key.startswith("vision_model."):
        return "model.vision_tower." + key
    if key.startswith("multi_modal_projector."):
        return "model." + key
    if key.startswith("model."):
        return "model.language_model." + key[len("model.") :]
    return key


KEY_RULES: tuple[tuple[str, Callable[[str], Optional[str]]], ...] = (
    ("identity", _rule_identity),
    ("strip_model_prefix", _rule_strip_model),
    ("nest_language_model", _rule_nest_language_model),
    # The headless VL classes (``Qwen3VLModel``): nested, then without the ``model.`` wrapper.
    ("nest_language_model_headless", lambda k: _rule_strip_model(_rule_nest_language_model(k))),
)


@dataclass
class KeyMapping:
    rule: str
    mapping: dict[str, str]  # file key -> model key
    dead_unexpected: list[str] = field(default_factory = list)
    dead_missing: list[str] = field(default_factory = list)


def _is_dead(key: str, patterns: Sequence[Any]) -> bool:
    return any(p.search(key) for p in patterns)


def match_keys(
    file_shapes: dict[str, Sequence[int]],
    expected: dict[str, Sequence[int]],
    *,
    tied_missing: Iterable[str] = (),
    rules: Sequence[tuple[str, Callable[[str], Optional[str]]]] = KEY_RULES,
) -> KeyMapping:
    """The first rename rule under which ``file_shapes`` loads STRICTLY into ``expected``: every expected
    tensor present with the same shape, nothing extra except the known-dead tensors. ``tied_missing``
    names model tensors the class re-ties after loading (an ``lm_head`` tied to the embedding).
    Raises ``ComponentFileError`` with the closest rule's diagnostics."""
    tied = set(tied_missing)
    best: Optional[tuple[int, str, list, list, list]] = None
    for name, rule in rules:
        mapping: dict[str, str] = {}
        collisions = False
        for key in file_shapes:
            new = rule(key)
            if new is None:
                continue
            if new in mapping.values():
                collisions = True
                break
            mapping[key] = new
        if collisions:
            continue
        reverse = {v: k for k, v in mapping.items()}
        missing = [
            k
            for k in expected
            if k not in reverse and k not in tied and not _is_dead(k, _DEAD_MISSING)
        ]
        unexpected = [
            k for k, v in mapping.items() if v not in expected and not _is_dead(v, _DEAD_UNEXPECTED)
        ]
        mismatched = [
            f"{k} {list(file_shapes[k])} != {list(expected[v])}"
            for k, v in mapping.items()
            if v in expected
            and tuple(int(d) for d in file_shapes[k]) != tuple(int(d) for d in expected[v])
        ]
        score = len(missing) + len(unexpected) + len(mismatched)
        if score == 0:
            return KeyMapping(
                rule = name,
                mapping = {k: v for k, v in mapping.items() if v in expected},
                dead_unexpected = sorted(k for k, v in mapping.items() if v not in expected),
                dead_missing = sorted(k for k in expected if k not in reverse),
            )
        if best is None or score < best[0]:
            best = (score, name, missing, unexpected, mismatched)
    if best is None:
        raise ComponentFileError("no key rule applies")
    _, name, missing, unexpected, mismatched = best
    parts = []
    if missing:
        parts.append(f"{len(missing)} missing (e.g. {', '.join(sorted(missing)[:3])})")
    if unexpected:
        parts.append(f"{len(unexpected)} unexpected (e.g. {', '.join(sorted(unexpected)[:3])})")
    if mismatched:
        parts.append(f"{len(mismatched)} shape mismatches (e.g. {'; '.join(mismatched[:2])})")
    raise ComponentFileError("; ".join(parts) + f" under the closest key rule '{name}'")


def quant_layers(path: str) -> dict[str, Any]:
    """``{layer name (file naming, no .weight): ComfyQuantLayer}``; raises for a format Studio cannot run."""
    from .diffusion_comfy_quant import scan_comfy_quant

    scan = scan_comfy_quant(path)
    if scan is None:
        return {}
    if scan.problems:
        shown = "; ".join(scan.problems[:3])
        more = f" (and {len(scan.problems) - 3} more)" if len(scan.problems) > 3 else ""
        raise ComponentFileError(
            f"'{os.path.basename(path)}' uses a ComfyUI quantization Studio cannot load as a text encoder "
            f"or VAE: {shown}{more}. Supported: unquantized, scaled fp8 (float8_e4m3fn / float8_e5m2) and "
            "int8_tensorwise (with or without ConvRot); use the bf16 or fp8 file instead."
        )
    return dict(scan.layers)


_HEADER_DTYPE_BYTES = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def resident_bytes(
    path: str,
    *,
    dtype_itemsize: int = 2,
    kind: str = "text_encoder",
) -> int:
    """What the loaded component holds: int8 ConvRot Linears at stored size (+ scales), every other weight
    (dequantized fp8 included) at the compute dtype. Header-only."""
    header, _ = read_safetensors_header(path)
    layers = quant_layers(path) if kind != COMPONENT_VAE else {}
    total = 0
    for key in weight_keys(header):
        entry = header[key]
        shape = entry.get("shape") or ()
        dtype = str(entry.get("dtype", ""))
        layer = layers.get(key[: -len(".weight")]) if key.endswith(".weight") else None
        if layer is not None and layer.convrot and layer.format == "int8_tensorwise":
            total += _numel(shape) + int(shape[0]) * 4
        elif dtype in ("F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "I8", "U8") or layer is not None:
            total += _numel(shape) * dtype_itemsize
        else:
            total += _numel(shape) * _HEADER_DTYPE_BYTES.get(dtype, dtype_itemsize)
    return total


def _fp8_view(codes: Any, fmt: str) -> Any:
    import torch
    if codes.dtype == torch.uint8:
        return codes.view(torch.float8_e5m2 if fmt == "float8_e5m2" else torch.float8_e4m3fn)
    return codes


def _dequant_fp8(codes: Any, scale: Optional[Any], fmt: str, dtype: Any) -> Any:
    import torch

    weight = _fp8_view(codes, fmt).to(torch.float32)
    if scale is not None:
        s = scale.to(torch.float32)
        weight = weight * (s.reshape(-1, 1) if s.numel() > 1 else s.reshape(()))
    return weight.to(dtype)


def _expected_shapes(module: Any) -> dict[str, tuple]:
    return {k: tuple(v.shape) for k, v in module.state_dict().items()}


def _file_logical_shapes(header: dict) -> dict[str, tuple]:
    return {k: tuple(int(d) for d in (header[k].get("shape") or ())) for k in weight_keys(header)}


def _strip_meta(module: Any) -> list[str]:
    stranded = [n for n, p in module.named_parameters() if p.is_meta]
    stranded += [n for n, b in module.named_buffers() if b is not None and b.is_meta]
    return stranded


def _fit_text_encoder(
    path: str, header: dict, encoder_cls: Any, config: Any, trim_lm_head: bool
) -> tuple[Any, KeyMapping, tuple]:
    """``encoder_cls`` on the meta device plus the strict key mapping of ``header`` onto it."""
    from accelerate import init_empty_weights

    from .diffusion_text_encoder_trim import config_ties_lm_head, trim_text_encoder

    with init_empty_weights(include_buffers = False):
        encoder = encoder_cls(config)
    if trim_lm_head:
        trim_text_encoder(encoder)
    expected = _expected_shapes(encoder)
    tied = ()
    if config_ties_lm_head(config) and "lm_head.weight" in expected:
        tied = ("lm_head.weight",)
    try:
        mapping = match_keys(_file_logical_shapes(header), expected, tied_missing = tied)
    except ComponentFileError as exc:
        raise ComponentFileError(
            f"'{os.path.basename(path)}' does not fit this pipeline's {encoder_cls.__name__}: {exc}"
        ) from None
    return encoder, mapping, tied


def build_text_encoder(
    path: str,
    *,
    encoder_cls: Any,
    config: Any,
    dtype: Any,
    trim_lm_head: bool = False,
    logger: Any = None,
) -> Any:
    """``encoder_cls`` (a transformers class) built from ``config`` and loaded from the ComfyUI file at
    ``path``, strictly. CPU; the pipeline assembly places it."""
    import torch
    from safetensors import safe_open

    from .diffusion_te_prequant import TE_PREQUANT_SCHEME_ATTR

    name = os.path.basename(path)
    header, _ = read_safetensors_header(path)
    layers = quant_layers(path)
    encoder, mapping, tied = _fit_text_encoder(path, header, encoder_cls, config, trim_lm_head)

    from .diffusion_comfy_quant import _dequant

    convrot_cls = None
    state: dict[str, Any] = {}
    kept_int8 = 0
    with safe_open(path, framework = "pt", device = "cpu") as handle:
        for src, dst in mapping.mapping.items():
            layer = layers.get(src[: -len(".weight")]) if src.endswith(".weight") else None
            if layer is None:
                tensor = handle.get_tensor(src)
                if tensor.is_floating_point() and dtype is not None and tensor.dtype != dtype:
                    tensor = tensor.to(dtype)
                state[dst] = tensor
                continue
            stem = src[: -len(".weight")]
            scale_key = next(
                (stem + s for s in (".weight_scale", ".scale_weight") if stem + s in header), None
            )
            scale = handle.get_tensor(scale_key) if scale_key else None
            codes = handle.get_tensor(src)
            if layer.format == "int8_tensorwise":
                if scale is None:
                    raise ComponentFileError(f"'{name}': {stem} is int8 with no weight_scale")
                parent_path, _, leaf = dst[: -len(".weight")].rpartition(".")
                target = encoder.get_submodule(parent_path) if parent_path else encoder
                existing = getattr(target, leaf, None)
                if layer.convrot and isinstance(existing, torch.nn.Linear):
                    if convrot_cls is None:
                        from .video_minimax_h3_te import _int8_convrot_linear_class
                        convrot_cls = _int8_convrot_linear_class()
                    has_bias = (stem + ".bias") in mapping.mapping
                    setattr(
                        target,
                        leaf,
                        convrot_cls(
                            existing.in_features,
                            existing.out_features,
                            bias = has_bias,
                            group_size = int(layer.group),
                        ),
                    )
                    state[dst] = codes.to(torch.int8)
                    state[dst[: -len(".weight")] + ".weight_scale"] = (
                        scale.to(torch.float32)
                        .reshape(-1, 1)
                        .expand(codes.shape[0], 1)
                        .contiguous()
                    )
                    kept_int8 += 1
                else:
                    state[dst] = _dequant(codes, scale, int(layer.group or 0), dtype)
            else:
                state[dst] = _dequant_fp8(codes, scale, layer.format, dtype)
    if convrot_cls is not None:
        # Int8ConvRotLinear keeps its bias in bf16 storage.
        for key, tensor in list(state.items()):
            if key.endswith(".bias") and tensor.is_floating_point():
                state[key] = tensor.to(dtype)
    missing, unexpected = encoder.load_state_dict(state, strict = False, assign = True)
    bad_missing = [k for k in missing if k not in tied and not _is_dead(k, _DEAD_MISSING)]
    if bad_missing or unexpected:
        raise ComponentFileError(
            f"'{name}' did not load strictly into {encoder_cls.__name__}: missing {bad_missing[:3]}, "
            f"unexpected {list(unexpected)[:3]}"
        )
    tie = getattr(encoder, "tie_weights", None)
    if callable(tie):
        tie()
    # T5 / UMT5: ``encoder.embed_tokens`` IS ``shared``; re-point a meta leftover rather than trust tie_weights.
    inner = getattr(encoder, "encoder", None)
    shared = getattr(encoder, "shared", None)
    if (
        inner is not None
        and shared is not None
        and getattr(inner, "embed_tokens", None) is not None
    ):
        if inner.embed_tokens.weight.is_meta:
            inner.embed_tokens = shared
    stranded = _strip_meta(encoder)
    if stranded:
        raise ComponentFileError(
            f"'{name}': {len(stranded)} tensor(s) of {encoder_cls.__name__} were not in the file, "
            f"e.g. {stranded[0]}"
        )
    encoder.requires_grad_(False)
    encoder.eval()
    if kept_int8:
        # Already quantized: the runtime text-encoder cast must leave it alone and report it as int8.
        setattr(encoder, TE_PREQUANT_SCHEME_ATTR, "int8")
    if logger is not None:
        logger.info(
            "diffusion.comfy_components: %s -> %s (rule %s, %d int8 ConvRot linears kept, %d dead tensors "
            "skipped)",
            name,
            encoder_cls.__name__,
            mapping.rule,
            kept_int8,
            len(mapping.dead_unexpected),
        )
    return encoder


_WAN_RESIDUAL = {
    "residual.0.gamma": "norm1.gamma",
    "residual.2.weight": "conv1.weight",
    "residual.2.bias": "conv1.bias",
    "residual.3.gamma": "norm2.gamma",
    "residual.6.weight": "conv2.weight",
    "residual.6.bias": "conv2.bias",
    "shortcut.weight": "conv_shortcut.weight",
    "shortcut.bias": "conv_shortcut.bias",
}


def _convert_wan_nested_vae(state: dict) -> dict:
    """Wan 2.2-layout VAE (Qwen-Image-2.1's too) -> diffusers ``AutoencoderKLWan`` / ``AutoencoderKLQwenImage21``
    naming. Each ``downsamples.<i>`` stage nests its residual blocks and its resampler as numbered children; the
    residual children become ``resnets.<n>`` in order and the resampler ``downsampler`` / ``upsampler``."""
    out: dict = {}
    stage = re.compile(
        r"^(encoder|decoder)\.(downsamples|upsamples)\.(\d+)\.(downsamples|upsamples)\.(\d+)\.(.+)$"
    )
    residual_children: dict[tuple[str, int], list[int]] = {}
    for key in state:
        m = stage.match(key)
        if m and (m.group(6).startswith("residual.") or m.group(6).startswith("shortcut.")):
            residual_children.setdefault((m.group(1), int(m.group(3))), [])
            if int(m.group(5)) not in residual_children[(m.group(1), int(m.group(3)))]:
                residual_children[(m.group(1), int(m.group(3)))].append(int(m.group(5)))
    for children in residual_children.values():
        children.sort()
    for key, value in state.items():
        new = None
        m = stage.match(key)
        if m:
            side, _, i, _, j, rest = m.groups()
            block = "down_blocks" if side == "encoder" else "up_blocks"
            children = residual_children.get((side, int(i)), [])
            if int(j) in children and rest in _WAN_RESIDUAL:
                new = f"{side}.{block}.{i}.resnets.{children.index(int(j))}.{_WAN_RESIDUAL[rest]}"
            elif rest.startswith(("resample.", "time_conv.")):
                sampler = "downsampler" if side == "encoder" else "upsampler"
                new = f"{side}.{block}.{i}.{sampler}.{rest}"
        else:
            m2 = re.match(r"^(encoder|decoder)\.middle\.(\d+)\.(.+)$", key)
            if m2:
                side, idx, rest = m2.groups()
                if idx == "1":
                    new = f"{side}.mid_block.attentions.0.{rest}"
                elif rest in _WAN_RESIDUAL:
                    new = f"{side}.mid_block.resnets.{int(idx) // 2}.{_WAN_RESIDUAL[rest]}"
            else:
                m3 = re.match(r"^(encoder|decoder)\.(conv1|head\.0|head\.2)\.(.+)$", key)
                if m3:
                    side, part, rest = m3.groups()
                    new = (
                        f"{side}."
                        + {"conv1": "conv_in", "head.0": "norm_out", "head.2": "conv_out"}[part]
                        + f".{rest}"
                    )
                elif key.startswith("conv1."):
                    new = "quant_conv." + key[len("conv1.") :]
                elif key.startswith("conv2."):
                    new = "post_quant_conv." + key[len("conv2.") :]
        out[new or key] = value
    return out


def fit_conv_shapes(state: dict, expected: dict) -> dict:
    """A causal-3D conv stored with a singleton time axis (``[out, in, 1, k, k]``) loads into a 2-D conv
    (``[out, in, k, k]``): squeeze exactly that axis, nothing else."""
    out = {}
    for key, value in state.items():
        want = expected.get(key)
        shape = tuple(value.shape)
        if want is not None and len(shape) == len(want) + 1 and len(shape) == 5 and shape[2] == 1:
            if (shape[:2] + shape[3:]) == tuple(want):
                value = value.squeeze(2)
        out[key] = value
    return out


def convert_vae_state_dict(state: dict, kind: str, config: dict) -> dict:
    """ComfyUI / original VAE layout -> diffusers naming, through diffusers' own single-file converters."""
    if kind == "wan_nested":
        return _convert_wan_nested_vae(state)
    if kind == "ldm_kl":
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

        converted = convert_ldm_vae_checkpoint(state, dict(config))
        # FLUX.2's VAE adds a latent BatchNorm the LDM converter does not know; it keeps its name.
        for key, value in state.items():
            if key.startswith("bn."):
                converted.setdefault(key, value)
        return converted
    if kind == "wan":
        from diffusers.loaders.single_file_utils import convert_wan_vae_to_diffusers
        return convert_wan_vae_to_diffusers(state)
    return dict(state)


def _fit_vae(name: str, raw: dict, kind: str, vae_cls: Any, config: dict) -> tuple[Any, dict, dict]:
    """``vae_cls`` on the meta device and ``raw`` converted to its naming, checked strictly. ``raw`` may hold meta
    tensors (a header-only check)."""
    from accelerate import init_empty_weights

    converted = convert_vae_state_dict(raw, kind, config)
    with init_empty_weights():
        vae = vae_cls.from_config(config)
    expected = _expected_shapes(vae)
    converted = fit_conv_shapes(converted, expected)
    try:
        match_keys(
            {k: tuple(v.shape) for k, v in converted.items()},
            expected,
            rules = (("identity", _rule_identity),),
        )
    except ComponentFileError as exc:
        raise ComponentFileError(
            f"'{name}' does not fit this pipeline's {vae_cls.__name__}: {exc}"
        ) from None
    return vae, converted, expected


def build_vae(
    path: str,
    *,
    vae_cls: Any,
    config: dict,
    dtype: Any,
    logger: Any = None,
) -> Any:
    """``vae_cls`` from ``config``, loaded strictly from the supplied VAE file."""
    from safetensors.torch import load_file

    name = os.path.basename(path)
    header, _ = read_safetensors_header(path)
    if quant_layers(path):
        raise ComponentFileError(
            f"'{name}': quantized VAE files are not supported; use the bf16/fp32 VAE"
        )
    kind = classify_vae(weight_keys(header))
    if kind is None or vae_cls.__name__ not in VAE_KIND_CLASSES.get(kind, ()):
        raise ComponentFileError(
            f"'{name}' is not a VAE this pipeline's {vae_cls.__name__} can load (detected {kind or 'unknown'})"
        )
    raw = {k: v for k, v in load_file(path).items() if k not in _NON_WEIGHT_KEYS}
    vae, converted, expected = _fit_vae(name, raw, kind, vae_cls, config)
    state = {}
    for key, tensor in converted.items():
        if key not in expected:
            continue
        if tensor.is_floating_point() and dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype)
        state[key] = tensor.contiguous()
    vae.load_state_dict(state, strict = True, assign = True)
    stranded = _strip_meta(vae)
    if stranded:
        raise ComponentFileError(
            f"'{name}': {len(stranded)} VAE tensor(s) not in the file, e.g. {stranded[0]}"
        )
    vae.requires_grad_(False)
    vae.eval()
    if logger is not None:
        logger.info(
            "diffusion.comfy_components: %s -> %s (%s layout)", name, vae_cls.__name__, kind
        )
    return vae


@dataclass
class ComponentOverrides:
    """The supplied files of one load, assigned to base pipeline components, plus the built modules."""

    files: dict[str, ComponentFileRef] = field(default_factory = dict)  # component -> file
    paths: dict[str, str] = field(default_factory = dict)  # component -> resolved local path
    classes: dict[str, str] = field(default_factory = dict)  # component -> class name
    kinds: dict[str, str] = field(default_factory = dict)  # component -> detected file kind
    modules: dict[str, Any] = field(default_factory = dict)

    @property
    def components(self) -> tuple[str, ...]:
        return tuple(self.files)

    @property
    def text_encoder_components(self) -> tuple[str, ...]:
        return tuple(c for c in self.files if c.startswith("text_encoder"))

    def summary(self) -> dict[str, str]:
        return {c: ref.name for c, ref in self.files.items()}

    def resident_mib(self, dtype_itemsize: int = 2) -> dict[str, int]:
        out = {}
        for component, path in self.paths.items():
            try:
                out[component] = resident_bytes(
                    path,
                    dtype_itemsize = dtype_itemsize,
                    kind = COMPONENT_VAE if component == COMPONENT_VAE else "text_encoder",
                ) // (1024 * 1024)
            except ComponentFileError:
                out[component] = 0
        return out


_ACTIVE: contextvars.ContextVar[Optional[ComponentOverrides]] = contextvars.ContextVar(
    "unsloth_diffusion_component_overrides", default = None
)


def active_component_overrides() -> Optional[ComponentOverrides]:
    return _ACTIVE.get()


def set_active_component_overrides(value: Optional[ComponentOverrides]) -> Any:
    return _ACTIVE.set(value)


def reset_active_component_overrides(token: Any) -> None:
    try:
        _ACTIVE.reset(token)
    except (ValueError, RuntimeError):
        _ACTIVE.set(None)


def component_classes_from_index(index: dict) -> dict[str, str]:
    """``{component: class name}`` for the text encoders and VAE a ``model_index.json`` declares."""
    out: dict[str, str] = {}
    for component, value in (index or {}).items():
        if component != COMPONENT_VAE and component not in TEXT_ENCODER_COMPONENTS:
            continue
        if isinstance(value, (list, tuple)) and len(value) == 2 and value[1]:
            out[component] = str(value[1])
    return out


def assign_components(
    te_headers: Sequence[tuple[ComponentFileRef, dict]],
    vae_header: Optional[tuple[ComponentFileRef, dict]],
    classes: dict[str, str],
    *,
    family: str = "",
) -> tuple[dict[str, ComponentFileRef], dict[str, str]]:
    """Each supplied file onto the one base component whose class it can fill (header-only). Raises
    when a file fits nothing, or two files want the same slot. A CLIP-L vs CLIP-G pair is told apart by
    width, so HiDream's two CLIP slots resolve in index order."""
    assigned: dict[str, ComponentFileRef] = {}
    kinds: dict[str, str] = {}
    for ref, header in te_headers:
        keys = weight_keys(header)
        shapes = {k: header[k].get("shape") for k in keys}
        # Tokenizer blobs (``tekken_model``) are evidence too, so classify on every non-companion name.
        kind = classify_text_encoder(keys + [k for k in header if k in _NON_WEIGHT_KEYS], shapes)
        if kind is None:
            raise ComponentFileError(f"'{ref.name}' is not a recognised text-encoder file")
        fits = [
            c
            for c in TEXT_ENCODER_COMPONENTS
            if c in classes and classes[c] in TE_KIND_CLASSES.get(kind, ()) and c not in assigned
        ]
        if kind in ("clip_l", "clip_g") and len(fits) > 1:
            # HiDream: text_encoder = CLIP-L, text_encoder_2 = CLIP-G (both WithProjection).
            fits = [fits[0]] if kind == "clip_l" else [fits[-1]]
        if not fits:
            wanted = ", ".join(f"{c}={classes[c]}" for c in TEXT_ENCODER_COMPONENTS if c in classes)
            taken = [c for c in assigned if classes.get(c) in TE_KIND_CLASSES.get(kind, ())]
            why = (
                f"its slot ({', '.join(taken)}) is already taken by another supplied file"
                if taken
                else f"the {family or 'pipeline'} text encoders are {wanted or 'none'}"
            )
            raise ComponentFileError(f"'{ref.name}' is a {kind} text encoder, but {why}")
        assigned[fits[0]] = ref
        kinds[fits[0]] = kind
    if vae_header is not None:
        ref, header = vae_header
        kind = classify_vae(weight_keys(header))
        vae_class = classes.get(COMPONENT_VAE)
        if kind is None:
            raise ComponentFileError(f"'{ref.name}' is not a recognised VAE file")
        if vae_class is None or vae_class not in VAE_KIND_CLASSES.get(kind, ()):
            raise ComponentFileError(
                f"'{ref.name}' is a {kind}-layout VAE, but the {family or 'pipeline'} VAE is "
                f"{vae_class or 'absent'}"
            )
        assigned[COMPONENT_VAE] = ref
        kinds[COMPONENT_VAE] = kind
    return assigned, kinds


def hub_header(ref: ComponentFileRef, hf_token: Optional[str]) -> dict:
    """A Hub file's safetensors header without downloading its weights."""
    from huggingface_hub import HfApi

    try:
        info = HfApi().parse_safetensors_file_metadata(ref.repo_id, ref.filename, token = hf_token)
    except Exception as exc:  # noqa: BLE001
        raise ComponentFileError(
            f"'{ref.name}': cannot read its safetensors header on the Hub: {exc}"
        )
    return {
        name: {"dtype": t.dtype, "shape": list(t.shape)} for name, t in (info.tensors or {}).items()
    }


def plan_component_overrides(
    *,
    text_encoder_files: Sequence[str],
    vae_file: Optional[str],
    model_path: Optional[str],
    model_index: dict,
    family: str = "",
    trusted_repo: Optional[Callable[[str], bool]] = None,
    hf_token: Optional[str] = None,
    resolve_hub: Optional[Callable[[ComponentFileRef], str]] = None,
) -> Optional[ComponentOverrides]:
    """Parse, classify and assign. ``resolve_hub`` turns a Hub ref into a local path (the load's download);
    None plans from the Hub header alone (the download plan)."""
    te_specs = list(text_encoder_files or ())
    if not te_specs and not vae_file:
        return None
    classes = component_classes_from_index(model_index)
    refs = [
        parse_component_file(s, model_path = model_path, trusted_repo = trusted_repo) for s in te_specs
    ]
    vae_ref = (
        parse_component_file(
            vae_file, model_path = model_path, trusted_repo = trusted_repo, what = "vae_file"
        )
        if vae_file
        else None
    )
    paths: dict[int, str] = {}

    def header_of(ref: ComponentFileRef) -> dict:
        if ref.local_path:
            return read_safetensors_header(ref.local_path)[0]
        if resolve_hub is not None:
            local = resolve_hub(ref)
            paths[id(ref)] = local
            return read_safetensors_header(local)[0]
        return hub_header(ref, hf_token)

    te_headers = [(ref, header_of(ref)) for ref in refs]
    vae_header = (vae_ref, header_of(vae_ref)) if vae_ref is not None else None
    assigned, kinds = assign_components(te_headers, vae_header, classes, family = family)
    out = ComponentOverrides(files = assigned, kinds = kinds)
    for component, ref in assigned.items():
        out.classes[component] = classes[component]
        local = ref.local_path or paths.get(id(ref))
        if local:
            # Refused before anything is evicted (a Hub file is only on disk from here).
            if component != COMPONENT_VAE:
                quant_layers(local)
            elif quant_layers(local):
                raise ComponentFileError(
                    f"'{ref.name}': quantized VAE files are not supported; use the bf16/fp32 VAE"
                )
            out.paths[component] = local
    return out


def _component_class_and_config(
    component: str,
    class_name: str,
    *,
    base: str,
    hf_token: Optional[str],
    local_files_only: bool,
    cache_dir: Optional[str],
) -> tuple[Any, Any]:
    import diffusers
    import transformers

    if component == COMPONENT_VAE:
        vae_cls = getattr(diffusers, class_name, None)
        if vae_cls is None:
            raise ComponentFileError(f"diffusers has no {class_name}; update diffusers")
        return vae_cls, vae_cls.load_config(
            base,
            subfolder = COMPONENT_VAE,
            token = hf_token,
            cache_dir = cache_dir,
            local_files_only = local_files_only,
        )
    encoder_cls = getattr(transformers, class_name, None)
    if encoder_cls is None:
        raise ComponentFileError(f"transformers has no {class_name}; update transformers")
    config = transformers.AutoConfig.from_pretrained(
        base,
        subfolder = component,
        token = hf_token,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
    )
    from .diffusion_krea2 import remap_rope_parameters

    remap_rope_parameters(getattr(config, "text_config", config))
    return encoder_cls, config


def _resolved_path(overrides: ComponentOverrides, component: str) -> str:
    path = overrides.paths.get(component)
    if not path:
        raise ComponentFileError(
            f"'{overrides.files[component].name}' was not resolved to a local file before assembly"
        )
    return path


def verify_override_fit(
    overrides: ComponentOverrides,
    *,
    base: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    family: Optional[str] = None,
    cache_dir: Optional[str] = None,
    text_encoder_quant: Optional[str] = None,
) -> None:
    """Header-only strict fit of every supplied file against the base repo's real class, plus the precision
    request, so a wrong-size or un-castable file is refused before the resident pipeline is unloaded."""
    import torch

    from .diffusion_precision import TE_QUANT_INT8, normalize_te_quant
    from .diffusion_text_encoder_trim import family_trims_lm_head

    requested = normalize_te_quant(text_encoder_quant)
    for component, class_name in overrides.classes.items():
        path = _resolved_path(overrides, component)
        cls, config = _component_class_and_config(
            component,
            class_name,
            base = base,
            hf_token = hf_token,
            local_files_only = local_files_only,
            cache_dir = cache_dir,
        )
        header, _ = read_safetensors_header(path)
        name = os.path.basename(path)
        if component == COMPONENT_VAE:
            meta = {
                k: torch.empty(tuple(int(d) for d in (header[k].get("shape") or ())), device = "meta")
                for k in weight_keys(header)
            }
            _fit_vae(
                name, meta, overrides.kinds.get(component) or classify_vae(meta) or "", cls, config
            )
            continue
        trim = component == "text_encoder" and family_trims_lm_head(family)
        _fit_text_encoder(path, header, cls, config, trim)
        if requested not in (None, TE_QUANT_INT8) and any(
            layer.convrot and layer.format == "int8_tensorwise"
            for layer in quant_layers(path).values()
        ):
            raise ComponentFileError(
                f"'{name}' is stored as int8 ConvRot and cannot be re-cast to {requested}; leave "
                "text_encoder_quant at auto or pick the bf16 file"
            )


def load_override_modules(
    overrides: ComponentOverrides,
    *,
    base: str,
    dtype: Any,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    family: Optional[str] = None,
    cache_dir: Optional[str] = None,
    logger: Any = None,
) -> dict[str, Any]:
    """Build every supplied component once per load (cached on ``overrides``) and return the pipe kwargs."""
    missing = [c for c in overrides.files if c not in overrides.modules]
    if not missing:
        return dict(overrides.modules)
    from .diffusion_text_encoder_trim import family_trims_lm_head

    for component in missing:
        path = _resolved_path(overrides, component)
        cls, config = _component_class_and_config(
            component,
            overrides.classes[component],
            base = base,
            hf_token = hf_token,
            local_files_only = local_files_only,
            cache_dir = cache_dir,
        )
        if component == COMPONENT_VAE:
            overrides.modules[component] = build_vae(
                path, vae_cls = cls, config = config, dtype = dtype, logger = logger
            )
            continue
        trim = component == "text_encoder" and family_trims_lm_head(family)
        overrides.modules[component] = build_text_encoder(
            path,
            encoder_cls = cls,
            config = config,
            dtype = dtype,
            trim_lm_head = trim,
            logger = logger,
        )
    return dict(overrides.modules)


def read_model_index(
    base: str,
    *,
    base_local_dir: Optional[str] = None,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
) -> dict:
    for root in (base_local_dir, base):
        if not root:
            continue
        candidate = Path(root).expanduser() / "model_index.json"
        try:
            if candidate.is_file():
                return json.loads(candidate.read_text(encoding = "utf-8"))
        except (OSError, ValueError):
            continue
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        base,
        "model_index.json",
        token = hf_token,
        cache_dir = cache_dir,
        local_files_only = local_files_only,
    )
    return json.loads(Path(local).read_text(encoding = "utf-8"))


def scanned_component_bytes(sizes: dict[str, int], components: Iterable[str]) -> dict[str, int]:
    """The ``{relative path: bytes}`` entries of a base-repo scan that ``components`` replace."""
    wanted = set(components)
    return {rel: size for rel, size in sizes.items() if rel.split("/", 1)[0] in wanted}


def hub_repo_ids(
    text_encoder_files: Sequence[str], vae_file: Optional[str], *, model_path: Optional[str]
) -> tuple[str, ...]:
    """Hub repos the supplied specs name; unparseable specs are left to validation."""
    out: list[str] = []
    for spec in [*(text_encoder_files or ()), *([vae_file] if vae_file else [])]:
        try:
            ref = parse_component_file(spec, model_path = model_path)
        except ComponentFileError:
            continue
        if ref.is_hub and ref.repo_id not in out:
            out.append(ref.repo_id)
    return tuple(out)


def validate_component_specs(
    text_encoder_files: Sequence[str],
    vae_file: Optional[str],
    *,
    model_path: Optional[str],
    trusted_repo: Optional[Callable[[str], bool]] = None,
) -> list[ComponentFileRef]:
    """Network-free request validation, run before anything is evicted: every spec parses and clears the trust
    bar, and every LOCAL file is a text encoder / VAE Studio recognises in a quantization it can load. The slot
    assignment needs the base repo's ``model_index.json`` and happens at load time."""
    refs: list[ComponentFileRef] = []
    for spec in text_encoder_files or ():
        ref = parse_component_file(spec, model_path = model_path, trusted_repo = trusted_repo)
        if ref.local_path:
            header, _ = read_safetensors_header(ref.local_path)
            keys = weight_keys(header)
            kind = classify_text_encoder(
                keys + [k for k in header if k in _NON_WEIGHT_KEYS],
                {k: header[k].get("shape") for k in keys},
            )
            if kind is None:
                vae_kind = classify_vae(keys)
                hint = " (it looks like a VAE; pass it as vae_file)" if vae_kind else ""
                raise ComponentFileError(
                    f"'{ref.name}' is not a recognised text-encoder file{hint}"
                )
            quant_layers(ref.local_path)
        refs.append(ref)
    if vae_file:
        ref = parse_component_file(
            vae_file, model_path = model_path, trusted_repo = trusted_repo, what = "vae_file"
        )
        if ref.local_path:
            header, _ = read_safetensors_header(ref.local_path)
            if classify_vae(weight_keys(header)) is None:
                raise ComponentFileError(f"'{ref.name}' is not a recognised VAE file")
            if quant_layers(ref.local_path):
                raise ComponentFileError(
                    f"'{ref.name}': quantized VAE files are not supported; use the bf16/fp32 VAE"
                )
        refs.append(ref)
    names = [r.local_path or f"{r.repo_id}/{r.filename}" for r in refs]
    if len(set(names)) != len(names):
        raise ComponentFileError("the same file was supplied twice")
    return refs
