# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a single-file diffusion checkpoint IS (DiT of which family / text encoder / VAE / LoRA /
ControlNet), from the safetensors or GGUF header only: no torch, no weight bytes, cached per
(path, size, mtime). ComfyUI and diffusers layouts, bare or under a container prefix.

Same-architecture variants (FLUX.1 dev / Krea / Kontext, Qwen-Image / 2512 / Edit / Edit-2509,
klein distilled / base, Z-Image base / Turbo, HiDream full / dev / fast, LTX-2 / 2.3,
HunyuanVideo-1.5 480p / 720p, Wan A14B high / low noise) have identical keys: the file name decides.
"""

from __future__ import annotations

import json
import os
import re
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

ROLE_DIT = "dit"
ROLE_TEXT_ENCODER = "text_encoder"
ROLE_VAE = "vae"
ROLE_LORA = "lora"
ROLE_CONTROLNET = "controlnet"
ROLE_UNKNOWN = "unknown"

PAGE_IMAGE = "image"
PAGE_VIDEO = "video"

NON_DIT_ROLES = frozenset({ROLE_TEXT_ENCODER, ROLE_VAE, ROLE_LORA, ROLE_CONTROLNET})

SAME_ARCH_VARIANTS: dict[str, frozenset[str]] = {
    "flux.1": frozenset({"flux.1", "flux.1-kontext"}),
    "qwen-image": frozenset({"qwen-image", "qwen-image-edit"}),
    "qwen-image-edit": frozenset({"qwen-image-edit"}),
    "hunyuanvideo-1.5": frozenset({"hunyuanvideo-1.5", "hunyuanvideo-1.5-720p"}),
}

_VIDEO_FAMILIES = frozenset(
    {
        "ltx-2",
        "wan2.2-ti2v-5b",
        "wan2.2-t2v-a14b",
        "hunyuanvideo-1.5",
        "hunyuanvideo-1.5-720p",
        "minimax-h3",
    }
)

_MAX_SAFETENSORS_HEADER = 128 * 1024 * 1024
_MAX_GGUF_TENSORS = 1 << 20
_MAX_GGUF_KV = 1 << 20


@dataclass(frozen = True)
class CheckpointInfo:
    role: str
    family: Optional[str] = None
    page: Optional[str] = None
    what: str = ""  # e.g. "a VAE", used in refusals
    layout: str = ""
    variant: Optional[str] = None  # flux.1-schnell vs flux.1-dev


def family_page(family: Optional[str]) -> Optional[str]:
    if not family:
        return None
    return PAGE_VIDEO if family in _VIDEO_FAMILIES else PAGE_IMAGE


def _read_safetensors_header(path: str) -> Optional[tuple[dict[str, list[int]], dict]]:
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) < 8:
            return None
        (n,) = struct.unpack("<Q", head)
        if n < 2 or n > _MAX_SAFETENSORS_HEADER:
            return None
        raw = f.read(n)
    if len(raw) < n:
        return None
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None
    if not isinstance(payload, dict):
        return None
    meta = payload.pop("__metadata__", None)
    shapes: dict[str, list[int]] = {}
    for key, entry in payload.items():
        shape = entry.get("shape") if isinstance(entry, dict) else None
        shapes[str(key)] = [int(d) for d in shape] if isinstance(shape, list) else []
    return shapes, (meta if isinstance(meta, dict) else {})


_GGUF_MAGIC = 0x46554747  # "GGUF" little-endian
_GGUF_FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def _gguf_read_str(f) -> Optional[bytes]:
    b = f.read(8)
    if len(b) < 8:
        return None
    (n,) = struct.unpack("<Q", b)
    if n > 1 << 24:
        return None
    s = f.read(n)
    return s if len(s) == n else None


def _gguf_skip(f, vtype: int) -> bool:
    if vtype == 8:
        b = f.read(8)
        if len(b) < 8:
            return False
        (n,) = struct.unpack("<Q", b)
        if n > 1 << 30:
            return False
        f.seek(n, 1)
        return True
    if vtype == 9:
        b = f.read(12)
        if len(b) < 12:
            return False
        atype, alen = struct.unpack("<IQ", b)
        if alen > 1 << 30:
            return False
        if atype == 8:
            for _ in range(alen):
                if not _gguf_skip(f, 8):
                    return False
            return True
        size = _GGUF_FIXED.get(atype)
        if size is None:
            return False
        f.seek(size * alen, 1)
        return True
    size = _GGUF_FIXED.get(vtype)
    if size is None:
        return False
    f.seek(size, 1)
    return True


def _read_gguf_header(path: str) -> Optional[tuple[dict[str, list[int]], dict]]:
    with open(path, "rb") as f:
        head = f.read(24)
        if len(head) < 24:
            return None
        magic, version, n_tensors, n_kv = struct.unpack("<IIQQ", head)
        if (
            magic != _GGUF_MAGIC
            or version < 2
            or n_tensors > _MAX_GGUF_TENSORS
            or n_kv > _MAX_GGUF_KV
        ):
            return None
        meta: dict[str, str] = {}
        for _ in range(n_kv):
            key = _gguf_read_str(f)
            vt = f.read(4)
            if key is None or len(vt) < 4:
                return None
            (vtype,) = struct.unpack("<I", vt)
            if vtype == 8 and key == b"general.architecture":
                value = _gguf_read_str(f)
                if value is None:
                    return None
                meta["general.architecture"] = value.decode("utf-8", "replace")
            elif not _gguf_skip(f, vtype):
                return None
        shapes: dict[str, list[int]] = {}
        for _ in range(n_tensors):
            name = _gguf_read_str(f)
            nd = f.read(4)
            if name is None or len(nd) < 4:
                break  # a truncated table still classifies on what was read
            (n_dims,) = struct.unpack("<I", nd)
            if n_dims > 8:
                break
            dims_raw = f.read(8 * n_dims + 12)
            if len(dims_raw) < 8 * n_dims + 12:
                break
            dims = list(struct.unpack(f"<{n_dims}Q", dims_raw[: 8 * n_dims]))
            # GGML lists the innermost dimension first; torch order is the reverse.
            shapes[name.decode("utf-8", "replace")] = [int(d) for d in reversed(dims)]
    return shapes, meta


_DIT_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.model.",
    "model.",
    "transformer.",
    "",
)

_LORA_RE = re.compile(
    r"(?:^|\.)(?:lora_[AB]|lora_down|lora_up|lora\.(?:up|down)|lokr_w\d|hada_w\d_[ab])(?:\.|$)|^lora_unet_|^lora_te\d?_"
)


def _strip(shapes: dict[str, list[int]], prefix: str) -> dict[str, list[int]]:
    if not prefix:
        return shapes
    n = len(prefix)
    return {k[n:]: v for k, v in shapes.items() if k.startswith(prefix)}


def _tops(keys) -> set[str]:
    return {k.split(".", 1)[0] for k in keys}


def _dim(
    shapes: dict[str, list[int]],
    key: str,
    axis: int = 0,
) -> Optional[int]:
    shape = shapes.get(key)
    if not shape or len(shape) <= axis:
        return None
    return shape[axis]


def _any(keys, pattern: str) -> bool:
    rx = re.compile(pattern)
    return any(rx.search(k) for k in keys)


def _dit(family: Optional[str], what: str, layout: str) -> CheckpointInfo:
    return CheckpointInfo(ROLE_DIT, family, family_page(family), what, layout)


def _match_dit(s: dict[str, list[int]]) -> Optional[CheckpointInfo]:
    """Most specific signature first."""
    keys = s.keys()
    tops = _tops(keys)

    if "single_stream_modulation" in tops and "double_stream_modulation_img" in tops:
        layout = "diffusers" if "x_embedder" in tops else "comfy"
        width = _dim(s, "img_in.weight") or _dim(s, "x_embedder.weight")
        if width == 6144 or "guidance_in" in tops:
            return _dit("flux.2-dev", "a FLUX.2 [dev] diffusion transformer", layout)
        return _dit("flux.2-klein", "a FLUX.2 [klein] diffusion transformer", layout)

    if "double_blocks" in tops and "single_blocks" in tops and "vector_in" in tops:
        in_ch = _dim(s, "img_in.weight", 1)
        if in_ch not in (None, 64):
            return _dit(
                None,
                "a FLUX.1 Fill/Depth/Canny-style diffusion transformer (not supported)",
                "comfy",
            )
        info = _dit("flux.1", "a FLUX.1 diffusion transformer", "comfy")
        # schnell is the only FLUX.1 without guidance_in
        return CheckpointInfo(
            **{
                **info.__dict__,
                "variant": "flux.1-dev" if "guidance_in" in tops else "flux.1-schnell",
            }
        )

    if "byt5_in" in tops and "double_blocks" in tops:
        if "single_blocks" in tops:
            return _dit("hunyuanimage-2.1", "a HunyuanImage-2.1 diffusion transformer", "comfy")
        return _dit("hunyuanvideo-1.5", "a HunyuanVideo-1.5 diffusion transformer", "comfy")

    if "double_stream_blocks" in tops and "single_stream_blocks" in tops:
        return _dit("hidream-i1", "a HiDream-I1 diffusion transformer", "comfy")

    if "noise_refiner" in tops and "context_refiner" in tops and "layers" in tops:
        if "cap_pad_token" in tops or "all_x_embedder" in tops or "x_pad_token" in tops:
            return _dit(
                "z-image",
                "a Z-Image diffusion transformer",
                "diffusers" if "all_x_embedder" in tops else "comfy",
            )
        return _dit(
            "lumina-2",
            "a Lumina-Image-2.0 diffusion transformer",
            "diffusers" if "time_caption_embed" in tops else "comfy",
        )

    if {"txtfusion", "tmlp", "first"} <= tops:
        return _dit("krea-2", "a Krea-2 diffusion transformer", "comfy")
    if {"text_fusion", "time_mod_proj", "transformer_blocks"} <= tops:
        return _dit("krea-2", "a Krea-2 diffusion transformer", "diffusers")

    if "token_refiner" in tops and (
        {"video_patch_proj", "audio_patch_proj"} <= tops or "audio_proj_in" in tops
    ):
        return _dit(
            "minimax-h3",
            "a MiniMax-H3 audio-video diffusion transformer",
            "comfy" if "video_patch_proj" in tops else "diffusers",
        )

    if "transformer_blocks" in tops and (
        "audio_adaln_single" in tops
        or "av_ca_a2v_gate_adaln_single" in tops
        or "av_cross_attn_audio_scale_shift" in tops
        or "audio_caption_projection" in tops
    ):
        return _dit(
            "ltx-2",
            "an LTX-2 audio-video diffusion transformer",
            "diffusers" if "proj_in" in tops else "comfy",
        )
    if "patchify_proj" in tops and "transformer_blocks" in tops:
        return _dit(None, "an LTX-Video (v0.9) diffusion transformer (not supported)", "comfy")

    if (
        "patch_embedding" in tops
        and "blocks" in tops
        and ("head" in tops or "condition_embedder" in tops)
    ):
        layout = "diffusers" if "condition_embedder" in tops else "comfy"
        pe = s.get("patch_embedding.weight") or []
        width = pe[0] if pe else None
        in_ch = pe[1] if len(pe) > 1 else None
        if any(k.startswith(("vace_", "motion_encoder", "face_")) for k in tops):
            return _dit(None, "a Wan VACE/Animate diffusion transformer (not supported)", layout)
        if in_ch == 48 and width == 3072:
            return _dit("wan2.2-ti2v-5b", "a Wan2.2 TI2V-5B diffusion transformer", layout)
        if in_ch == 16 and width == 5120:
            return _dit(
                "wan2.2-t2v-a14b",
                "a Wan 14B text-to-video diffusion transformer (one A14B expert)",
                layout,
            )
        if in_ch == 36:
            return _dit(None, "a Wan image-to-video diffusion transformer (not supported)", layout)
        return _dit(None, "a Wan diffusion transformer of an unsupported size", layout)

    if (
        "transformer_blocks" in tops
        and ("cond_type_embed" in tops or "image_embedder" in tops)
        and ("context_embedder_2" in tops or "x_embedder" in tops)
    ):
        return _dit("hunyuanvideo-1.5", "a HunyuanVideo-1.5 diffusion transformer", "diffusers")
    if {
        "transformer_blocks",
        "single_transformer_blocks",
        "x_embedder",
        "context_embedder",
    } <= tops and ("time_guidance_embed" in tops or "context_embedder_2" in tops):
        return _dit("hunyuanimage-2.1", "a HunyuanImage-2.1 diffusion transformer", "diffusers")

    if "transformer_blocks" in tops and "img_in" in tops and "txt_in" in tops:
        if "txt_norm" in tops:
            if "time_text_embed.addition_t_embedding.weight" in s:
                return _dit(
                    "qwen-image-layered", "a Qwen-Image-Layered diffusion transformer", "comfy"
                )
            if "__index_timestep_zero__" in s:
                return _dit(
                    "qwen-image-edit", "a Qwen-Image-Edit-2511 diffusion transformer", "comfy"
                )
            return _dit("qwen-image", "a Qwen-Image diffusion transformer", "comfy")
        if "modulation" in tops or _any(keys, r"^txt_in\.(in_layer|text_norm)"):
            return _dit("qwen-image-2.1", "a Qwen-Image-2.1 diffusion transformer", "comfy")

    if {
        "transformer_blocks",
        "single_transformer_blocks",
        "x_embedder",
        "context_embedder",
    } <= tops and ("time_text_embed" in tops):
        in_ch = _dim(s, "x_embedder.weight", 1)
        if in_ch not in (None, 64):
            return _dit(
                None,
                "a FLUX.1 Fill/Depth/Canny-style diffusion transformer (not supported)",
                "diffusers",
            )
        info = _dit("flux.1", "a FLUX.1 diffusion transformer", "diffusers")
        guided = _any(keys, r"^time_text_embed\.guidance_embedder\.")
        return CheckpointInfo(
            **{**info.__dict__, "variant": "flux.1-dev" if guided else "flux.1-schnell"}
        )

    if {"input_blocks", "middle_block", "output_blocks"} <= tops:
        if "label_emb" in tops:
            return _dit("sdxl", "an SDXL UNet", "comfy")
        return _dit(None, "a Stable Diffusion 1.x/2.x UNet (not supported)", "comfy")
    if {"down_blocks", "mid_block", "up_blocks", "conv_in"} <= tops and "time_embedding" in tops:
        if "add_embedding" in tops:
            return _dit("sdxl", "an SDXL UNet", "diffusers")
        return _dit(None, "a Stable Diffusion 1.x/2.x UNet (not supported)", "diffusers")

    if "joint_blocks" in tops:
        return _dit(None, "a Stable Diffusion 3 diffusion transformer (not supported)", "comfy")
    if "double_layers" in tops and "cond_seq_linear" in tops:
        return _dit(None, "an AuraFlow diffusion transformer (not supported)", "comfy")
    return None


def _match_controlnet(keys) -> bool:
    tops = _tops(keys)
    return bool(
        {t for t in tops if t.startswith(("controlnet_", "control_"))}
        or "control_model" in tops
        or "controlnet_cond_embedding" in tops
        or "vace_blocks" in tops
    )


def _match_text_encoder(keys) -> Optional[str]:
    tops = _tops(keys)
    joined_tail = {k.split(".", 2)[1] if k.count(".") >= 1 else "" for k in keys}
    if (
        "text_model" in tops
        or "text_projection" in tops
        or "transformer" in tops
        and "text_model" in joined_tail
    ):
        return "a CLIP text encoder"
    if "encoder" in tops and "shared" in tops:
        return "a T5 text encoder"
    if "spiece_model" in tops or "tekken_model" in tops:
        return "a text encoder (LLM)"
    if "token_embd" in tops or "blk" in tops and "output_norm" in tops:
        return "a text encoder (LLM, GGUF)"
    if "lm_head" in tops or "visual" in tops or "vision_tower" in tops or "vision_model" in tops:
        return "a text encoder (LLM / vision-language model)"
    if _any(keys, r"^model\.(language_model\.)?(embed_tokens|layers\.\d+\.self_attn)"):
        return "a text encoder (LLM)"
    if tops == {"text_embedding_projection"}:
        return "an LTX-2 text-embedding projection (a text-encoder companion)"
    if "conditioner" in tops or "cond_stage_model" in tops or "text_encoders" in tops:
        return "a text encoder bundle"
    return None


def _match_vae(keys) -> Optional[str]:
    tops = _tops(keys)
    if "audio_vae" in tops or "vocoder" in tops:
        return "an audio VAE / vocoder"
    if "first_stage_model" in tops or "vae" in tops:
        return "a VAE"
    if "encoder" in tops and "decoder" in tops:
        return "a VAE"
    if "decoder" in tops and (
        "post_quant_conv" in tops or "latents_mean" in tops or "per_channel_statistics" in tops
    ):
        return "a VAE (decoder)"
    return None


_TE_GGUF_ARCHS = frozenset(
    {
        "t5",
        "t5encoder",
        "llama",
        "qwen2",
        "qwen2vl",
        "qwen25vl",
        "qwen3",
        "qwen3vl",
        "qwen35",
        "gemma",
        "gemma2",
        "gemma3",
        "mistral3",
        "clip",
        "bert",
    }
)


def classify_tensors(shapes: dict[str, list[int]], meta: Optional[dict] = None) -> CheckpointInfo:
    if not shapes:
        return CheckpointInfo(ROLE_UNKNOWN, what = "an empty or unreadable checkpoint")
    keys = list(shapes)
    if any(_LORA_RE.search(k) for k in keys):
        return CheckpointInfo(ROLE_LORA, what = "a LoRA adapter")
    arch = str((meta or {}).get("general.architecture") or "").lower()
    if arch in _TE_GGUF_ARCHS:
        return CheckpointInfo(
            ROLE_TEXT_ENCODER, what = f"a text encoder ({arch} GGUF)", layout = "gguf"
        )
    if _match_controlnet(keys):
        return CheckpointInfo(ROLE_CONTROLNET, what = "a ControlNet")

    tops = _tops(keys)
    bundled = bool(
        tops
        & {
            "vae",
            "first_stage_model",
            "text_encoders",
            "conditioner",
            "cond_stage_model",
            "audio_vae",
        }
    )
    for prefix in _DIT_PREFIXES:
        sub = _strip(shapes, prefix)
        if not sub:
            continue
        info = _match_dit(sub)
        if info is not None:
            layout = "gguf" if arch else ("checkpoint" if bundled else info.layout)
            return CheckpointInfo(
                info.role, info.family, info.page, info.what, layout, info.variant
            )

    te = _match_text_encoder(keys)
    if te:
        return CheckpointInfo(ROLE_TEXT_ENCODER, what = te)
    vae = _match_vae(keys)
    if vae:
        return CheckpointInfo(ROLE_VAE, what = vae)
    return CheckpointInfo(ROLE_UNKNOWN, what = "an unrecognised checkpoint layout")


_CACHE: dict[tuple, CheckpointInfo] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 4096


def inspect_checkpoint(path: str) -> CheckpointInfo:
    """Header-only classification of a local ``.safetensors`` / ``.gguf``; never raises."""
    try:
        st = os.stat(path)
    except (OSError, TypeError, ValueError):
        return CheckpointInfo(ROLE_UNKNOWN, what = "an unreadable file")
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
    if hit is not None:
        return hit
    reader: Optional[Callable] = None
    low = str(path).lower()
    if low.endswith(".gguf"):
        reader = _read_gguf_header
    elif low.endswith((".safetensors", ".sft")):
        reader = _read_safetensors_header
    info = CheckpointInfo(ROLE_UNKNOWN, what = "an unsupported file type")
    if reader is not None:
        try:
            parsed = reader(str(path))
        except (OSError, ValueError, TypeError, struct.error, MemoryError, RecursionError):
            parsed = None
        info = (
            classify_tensors(*parsed)
            if parsed
            else CheckpointInfo(ROLE_UNKNOWN, what = "an unreadable checkpoint header")
        )
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[key] = info
    return info


def offer_as_dit(path: str) -> bool:
    """False for TE / VAE / LoRA / ControlNet; an unclassified header is left to the caller's name check."""
    return inspect_checkpoint(path).role not in NON_DIT_ROLES


def local_pick_file(repo_id: Optional[str], filename: Optional[str]) -> Optional[str]:
    try:
        if not repo_id:
            return None
        root = Path(str(repo_id)).expanduser()
        if filename:
            name = str(filename)
            if os.path.isabs(name) or ".." in Path(name).parts:
                return None
            candidate = root / name
            return str(candidate) if root.is_dir() and candidate.is_file() else None
        if root.is_file() and root.suffix.lower() in (".safetensors", ".gguf", ".sft"):
            return str(root)
    except (OSError, ValueError):
        return None
    return None


def refusal_for(path: str, page: str) -> Optional[str]:
    """Why ``path`` cannot be this page's diffusion model (non-DiT, or other page's DiT), else None."""
    info = inspect_checkpoint(path)
    name = os.path.basename(str(path))
    if info.role in NON_DIT_ROLES:
        kind = {
            ROLE_TEXT_ENCODER: "text_encoders/ or clip/",
            ROLE_VAE: "vae/",
            ROLE_LORA: "loras/",
            ROLE_CONTROLNET: "controlnet/",
        }[info.role]
        return (
            f"'{name}' is {info.what}, not a diffusion model (DiT/UNet), so it cannot be loaded as the "
            f"model. In a ComfyUI layout it belongs in {kind}; pick the diffusion model file "
            f"(diffusion_models/, unet/ or checkpoints/) instead."
        )
    if info.role == ROLE_DIT and info.family and info.page and info.page != page:
        other = "Video" if info.page == PAGE_VIDEO else "Image"
        return f"'{name}' is {info.what} ({info.family}); load it from the {other} page."
    return None


def resolve_family_with_content(
    name_family: Optional[str],
    path: Optional[str],
    page: str,
    name_vetoed: bool = False,
) -> tuple[Optional[str], bool]:
    """(family, decided_by_content). Non-DiT / other page -> None; a same-architecture name variant
    keeps the name, else the header wins; no supported header family -> the name. A name the variant
    guard refused (``...-inpaint``) stays refused."""
    if not path:
        return name_family, False
    info = inspect_checkpoint(path)
    if info.role in NON_DIT_ROLES:
        return None, True
    if info.role != ROLE_DIT or not info.family:
        return name_family, False
    if info.page != page:
        return None, True
    if name_family and name_family in SAME_ARCH_VARIANTS.get(info.family, frozenset({info.family})):
        return name_family, False
    if name_vetoed and name_family is None:
        return None, False
    return info.family, name_family != info.family


def assert_local_pick_is_dit(repo_id: Optional[str], filename: Optional[str], page: str) -> None:
    """ValueError when a LOCAL pick is not this page's diffusion model. Header-only, before any load."""
    path = local_pick_file(repo_id, filename)
    if not path:
        return
    message = refusal_for(path, page)
    if message:
        raise ValueError(message)


def content_variant_hint(repo_id: Optional[str], filename: Optional[str]) -> Optional[str]:
    """Defaults / flow-shift key a local pick's header implies (``flux.1-dev`` / ``-schnell``), or None.
    Callers put it AFTER the file name and BEFORE the folder path and base repo (FLUX.1's base is
    schnell, which would hand a renamed dev file 4 steps)."""
    path = local_pick_file(repo_id, filename)
    return inspect_checkpoint(path).variant if path else None
