# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3-Image prompt enhancer: Qwen3.5-9B on the FLUX.3 upsampler's llama-server sidecar.

Mirrors the reference workflow templates (image_minimax_h3_image_t2i / _edit): the system prompt and the task prompt
with the user's text in place of ``{input text}``, the edit's reference images ahead of it, thinking on, temperature 1,
top_k 20, top_p 0.95, seed 0, and the caption read from the reply's first JSON object under ``expanded_caption``.
The templates leave ``aspect_ratio`` unwired (the canvas comes from the size widget), so it is not read here either.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

from core.inference import flux3_prompt_upsampler as pup
from core.inference.h3_image_prompt_enhancer_prompts import (
    EDIT_SYSTEM_PROMPT,
    EDIT_TASK_PROMPT,
    INPUT_PLACEHOLDER,
    T2I_SYSTEM_PROMPT,
    T2I_TASK_PROMPT,
)

logger = get_logger(__name__)

H3_IMAGE_FAMILY = "minimax-h3-image"

# ── weights ─────────────────────────────────────────────────────────────────────────────────────────────────────────
# The templates run Qwen3.5-9B (a w4a8 GPTQ safetensors). Studio runs the same model as Unsloth's GGUF on llama.cpp,
# the FLUX.3 upsampler's runtime (that upsampler is a Qwen3.5-9B fine-tune, so the sidecar's sizing holds as is).
ENHANCER_ID = "minimax-h3-image-pe"
ENHANCER_REPO = "unsloth/Qwen3.5-9B-GGUF"
ENHANCER_MODEL_FILES = {
    "UD-Q4_K_XL": "Qwen3.5-9B-UD-Q4_K_XL.gguf",
    "Q8_0": "Qwen3.5-9B-Q8_0.gguf",
    "UD-Q6_K_XL": "Qwen3.5-9B-UD-Q6_K_XL.gguf",
    "Q6_K": "Qwen3.5-9B-Q6_K.gguf",
    "UD-Q5_K_XL": "Qwen3.5-9B-UD-Q5_K_XL.gguf",
    "Q5_K_M": "Qwen3.5-9B-Q5_K_M.gguf",
    "Q4_K_M": "Qwen3.5-9B-Q4_K_M.gguf",
}
ENHANCER_DEFAULT_QUANT = "UD-Q4_K_XL"
ENHANCER_MODEL_FILE = ENHANCER_MODEL_FILES[ENHANCER_DEFAULT_QUANT]
# What a run reads, best first: the default download, else a larger quant already cached (a chat download).
ENHANCER_MODEL_FILE_PREFERENCE = (ENHANCER_MODEL_FILE,) + tuple(
    file for quant, file in ENHANCER_MODEL_FILES.items() if quant != ENHANCER_DEFAULT_QUANT
)
# Vision projector, needed only for edits (the reference pictures).
ENHANCER_MMPROJ_FILE = "mmproj-F16.gguf"
ENHANCER_FILE_SIZES = {
    "Qwen3.5-9B-UD-Q4_K_XL.gguf": 5_966_095_584,
    "Qwen3.5-9B-Q8_0.gguf": 9_527_502_048,
    "Qwen3.5-9B-UD-Q6_K_XL.gguf": 8_756_929_760,
    "Qwen3.5-9B-Q6_K.gguf": 7_458_301_152,
    "Qwen3.5-9B-UD-Q5_K_XL.gguf": 6_743_680_224,
    "Qwen3.5-9B-Q5_K_M.gguf": 6_577_841_376,
    "Qwen3.5-9B-Q4_K_M.gguf": 5_680_522_464,
    "mmproj-F16.gguf": 918_166_080,
}
ENV_MODEL_OVERRIDE = "UNSLOTH_H3_IMAGE_ENHANCER_GGUF"
ENV_MMPROJ_OVERRIDE = "UNSLOTH_H3_IMAGE_ENHANCER_MMPROJ"
LABEL = "MiniMax-H3-Image prompt enhancer"

# ── sampling (the templates' TextGenerateV2 widgets) ───────────────────────────────────────────────────────────────
# t2i ["", 8192, "on", 1, 20, 0.95, 0, 1, 0, 0, true, true, "auto"]; edit the same with max_length 8196: max_length,
# sampling on, temperature, top_k, top_p, min_p, repetition_penalty, presence_penalty, seed, thinking,
# use_default_template, mtp. (No multi-token-prediction head in this GGUF, so "auto" has nothing to run.)
SAMPLING = {
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "presence_penalty": 0.0,
    "seed": 0,
}
MAX_NEW_TOKENS = {"t2i": 8192, "edit": 8196}
THINKING = True
JSON_CAPTION_KEY = "expanded_caption"

# ── reference pictures (the edit template's two ResizeImageMaskNode steps) ─────────────────────────────────────────
REFERENCE_MEGAPIXELS = 4
REFERENCE_MULTIPLE = 32
# Image tokens a 4 MP reference costs (32 px per token after the 2x2 merge), for the server's context.
_TOKENS_PER_REFERENCE = 4096

# ── context ────────────────────────────────────────────────────────────────────────────────────────────────────────
# t2i: ~3.5k prompt tokens + 8192 generated fit the sidecar's 16k default. An edit's task prompt is ~7.4k tokens,
# plus 4096 per reference and 8196 generated, so its server starts with the context it needs.
_CONTEXT_STEP = 4096
_EDIT_PROMPT_TOKENS = 8192


def supports_family(family_name: Optional[str]) -> bool:
    return family_name == H3_IMAGE_FAMILY


def default_enhance(edit: bool = False) -> bool:
    """The templates' default: "Enable PE?" is true on the edit template, and the t2i template's subgraph instance
    sets its promoted ``enable_PE`` input to true (its inner switch widget's false is overridden)."""
    return True


def context_tokens_for(prompt: str, num_references: int) -> int:
    if not num_references:
        return pup._CONTEXT_TOKENS
    # A generous token bound for the typed prompt: about 3 bytes a token in English.
    need = (
        _EDIT_PROMPT_TOKENS
        + MAX_NEW_TOKENS["edit"]
        + _TOKENS_PER_REFERENCE * num_references
        + len(prompt.encode("utf-8")) // 3
        + 256
    )
    return max(pup._CONTEXT_TOKENS, int(math.ceil(need / _CONTEXT_STEP)) * _CONTEXT_STEP)


# ── request building ────────────────────────────────────────────────────────────────────────────────────────────────
def task_prompt(prompt: str, edit: bool) -> str:
    """The template's StringReplace: the task prompt with the user's text in place of ``{input text}``."""
    return (EDIT_TASK_PROMPT if edit else T2I_TASK_PROMPT).replace(INPUT_PLACEHOLDER, prompt)


def build_chat_payload(
    prompt: str,
    references: Optional[list[bytes]] = None,
    *,
    edit: Optional[bool] = None,
) -> dict:
    """The /v1/chat/completions body. ``references`` are prepared PNGs, ahead of the text as the template's
    tokenizer places them; ``edit`` forces the edit prompts without images (a missing projector)."""
    import base64

    references = list(references or [])
    edit = bool(references) if edit is None else bool(edit)
    mode = "edit" if edit else "t2i"
    content: list[dict] = [
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(ref).decode("ascii")},
        }
        for ref in references
    ]
    content.append({"type": "text", "text": task_prompt(prompt, edit)})
    return {
        "messages": [
            {"role": "system", "content": EDIT_SYSTEM_PROMPT if edit else T2I_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        **SAMPLING,
        "max_tokens": MAX_NEW_TOKENS[mode],
        "chat_template_kwargs": {"enable_thinking": THINKING},
        "stream": False,
    }


def visible_text(text: Optional[str]) -> str:
    """The reply after the reasoning, as the template's text node returns it: what follows ``</think>``, and nothing
    when the thinking never closed (the generation budget ran out inside it)."""
    text = text or ""
    reasoning, sep, rest = text.partition("</think>")
    if sep:
        return rest.strip()
    # llama-server already splits the reasoning off; an unclosed block left in the content is a truncated one.
    return "" if reasoning.lstrip().startswith("<think>") else text.strip()


def extract_json_string(text: str, key: str = JSON_CAPTION_KEY) -> str:
    """The template's JsonExtractString: ``key`` of the first JSON object that decodes, starting at each ``{`` in
    turn (Markdown fences around it included); "" when none decodes or the key is missing."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        value = data.get(key)
        return "" if value is None else str(value)
    return ""


def caption_from_reply(raw: Optional[str]) -> str:
    return extract_json_string(visible_text(raw)).strip()


# ── reference preparation ───────────────────────────────────────────────────────────────────────────────────────────
def reference_size(
    width: int, height: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """(scaled, covered, final) (w, h) of the edit template's two resizes: "scale total pixels" to 4 MP, then "scale
    to multiple" 32, which covers the floored multiple and centre-crops to it (covered == final: no second resize)."""
    total = int(REFERENCE_MEGAPIXELS * 1024 * 1024)
    scale = math.sqrt(total / (width * height))
    w, h = round(width * scale), round(height * scale)
    target_w = (w // REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE
    target_h = (h // REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE
    if target_w == 0 or target_h == 0 or (target_w, target_h) == (w, h):
        return (w, h), (w, h), (w, h)
    s_w, s_h = target_w / w, target_h / h
    if s_w >= s_h:
        cover = (target_w, max(target_h, int(math.ceil(h * s_w))))
    else:
        cover = (max(target_w, int(math.ceil(w * s_h))), target_h)
    return (w, h), cover, (target_w, target_h)


def prepare_reference(image: Any) -> bytes:
    """One reference as the template feeds its text generator: EXIF-upright RGB (an alpha channel dropped, as its
    image loader does), area-resized to 4 MP, covered and centre-cropped to a multiple of 32, as PNG."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image).convert("RGB")
    pixels = torch.from_numpy(np.asarray(image, dtype = np.float32) / 255.0).permute(2, 0, 1)[None]
    (w, h), (cover_w, cover_h), (target_w, target_h) = reference_size(image.width, image.height)
    pixels = F.interpolate(pixels, size = (h, w), mode = "area")
    if (cover_w, cover_h) != (w, h) or (target_w, target_h) != (w, h):
        pixels = F.interpolate(pixels, size = (cover_h, cover_w), mode = "area")
        x0, y0 = (cover_w - target_w) // 2, (cover_h - target_h) // 2
        pixels = pixels[:, :, y0 : y0 + target_h, x0 : x0 + target_w]
    # Back to 8-bit for the PNG the server decodes.
    array = (pixels[0].permute(1, 2, 0).clamp(0, 1) * 255.0).numpy().astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format = "PNG", compress_level = 1)
    return buffer.getvalue()


def raw_references_hash(images: Optional[list]) -> str:
    """A digest of the references as uploaded (data URLs, or PIL images), in order: the cache key, computed without
    decoding or resizing anything."""
    import hashlib

    digest = hashlib.sha256()
    for item in list(images or [])[: pup.MAX_REFERENCES]:
        if isinstance(item, str):
            part = item.encode("utf-8")
        else:
            part = f"{item.mode}:{item.size}:".encode() + item.tobytes()
        digest.update(hashlib.sha256(part).digest())
    return digest.hexdigest()


def prepare_references(images: Optional[list]) -> list[bytes]:
    items = list(images or [])[: pup.MAX_REFERENCES]
    prepared = []
    with pup._DECODE_SLOTS:
        for item in items:
            image = pup.decode_reference(item) if isinstance(item, str) else item
            prepared.append(prepare_reference(image))
            del image
    return prepared


# ── weights on disk ─────────────────────────────────────────────────────────────────────────────────────────────────
SPEC = pup._UpsamplerSpec(
    repo = ENHANCER_REPO,
    model_file = ENHANCER_MODEL_FILE,
    mmproj_file = ENHANCER_MMPROJ_FILE,
    label = LABEL,
)


def _cached_file(
    filename: str,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
):
    try:
        from huggingface_hub import hf_hub_download

        from core.inference.stt_sidecar import _active_hf_hub_cache

        root = hub_cache if hub_cache is not None else _active_hf_hub_cache()
        return hf_hub_download(
            repo_id = ENHANCER_REPO,
            filename = filename,
            revision = revision,
            local_files_only = True,
            cache_dir = str(root),
        )
    except Exception:
        return None


def _candidate_revisions(hub_cache: Optional[Path] = None) -> list:
    from core.inference.stt_sidecar import _fallback_revisions, _read_revision_record
    recorded = _read_revision_record(ENHANCER_REPO)
    return ([recorded] if recorded else []) + [
        rev for rev in _fallback_revisions(ENHANCER_REPO, hub_cache = hub_cache) if rev != recorded
    ]


def _cached_at_any_revision(filename: str, hub_cache: Optional[Path] = None) -> Optional[str]:
    for revision in _candidate_revisions(hub_cache):
        path = _cached_file(filename, hub_cache = hub_cache, revision = revision)
        if path is not None:
            return path
    return None


def _pair_at(hub_cache: Optional[Path], model_file: str, *, need_mmproj: bool):
    model_only = None
    for revision in _candidate_revisions(hub_cache):
        model = _cached_file(model_file, hub_cache = hub_cache, revision = revision)
        if model is None:
            continue
        mmproj = _cached_file(ENHANCER_MMPROJ_FILE, hub_cache = hub_cache, revision = revision)
        if mmproj is not None:
            return model, mmproj
        model_only = model_only or model
    return (model_only, None) if model_only is not None and not need_mmproj else None


def _cached_pair(
    hub_cache: Optional[Path] = None, model_files: tuple = ENHANCER_MODEL_FILE_PREFERENCE
) -> tuple[Optional[str], Optional[str]]:
    """The model and projector from one revision, in ``model_files`` order; a complete pair beats a model alone."""
    from core.inference.model_cache_search import find_in_hub_caches

    for need_mmproj in (True, False):
        for model_file in model_files:
            if hub_cache is not None:
                found = _pair_at(hub_cache, model_file, need_mmproj = need_mmproj)
            else:
                found = find_in_hub_caches(
                    ENHANCER_REPO,
                    lambda root, model_file = model_file, need_mmproj = need_mmproj: _pair_at(
                        root, model_file, need_mmproj = need_mmproj
                    ),
                )
            if found is not None:
                return found
    return None, None


def _cached_download_paths(
    model_id: str,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    if revision is None:
        model, mmproj = _cached_pair(hub_cache)
    else:
        model = _cached_file(ENHANCER_MODEL_FILE, hub_cache = hub_cache, revision = revision)
        mmproj = _cached_file(ENHANCER_MMPROJ_FILE, hub_cache = hub_cache, revision = revision)
    if model is None or mmproj is None:
        return None
    return model, mmproj


def _env_path(name: str) -> Optional[str]:
    value = (os.environ.get(name) or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path) if path.is_file() else None


def overridden() -> bool:
    return any(
        (os.environ.get(name) or "").strip() for name in (ENV_MODEL_OVERRIDE, ENV_MMPROJ_OVERRIDE)
    )


def model_path() -> Optional[str]:
    if (os.environ.get(ENV_MODEL_OVERRIDE) or "").strip():
        return _env_path(ENV_MODEL_OVERRIDE)
    return _cached_pair()[0]


def mmproj_path() -> Optional[str]:
    if (os.environ.get(ENV_MMPROJ_OVERRIDE) or "").strip():
        return _env_path(ENV_MMPROJ_OVERRIDE)
    if overridden():
        from core.inference.model_cache_search import find_in_hub_caches
        return find_in_hub_caches(
            ENHANCER_REPO, lambda root: _cached_at_any_revision(ENHANCER_MMPROJ_FILE, root)
        )
    return _cached_pair()[1]


def _resolve_id(model_id: Optional[str]) -> str:
    if (model_id or ENHANCER_ID) != ENHANCER_ID:
        raise ValueError(f"Unknown prompt enhancer '{model_id}'.")
    return ENHANCER_ID


_download_state_lock = threading.Lock()
_download_state = None


def _get_download_state():
    global _download_state
    with _download_state_lock:
        if _download_state is None:
            from core.inference.stt_mtmd_sidecar import _MtmdDownloadState
            _download_state = _MtmdDownloadState(
                specs = {ENHANCER_ID: SPEC},
                cached_paths = _cached_download_paths,
                resolve = _resolve_id,
                label = LABEL,
            )
        return _download_state


def start_download(hf_token: Optional[str] = None) -> str:
    return _get_download_state().start(ENHANCER_ID, hf_token)


def download_status() -> dict:
    return _get_download_state().status()


def cancel_download(download_id: Optional[str] = None) -> bool:
    return _get_download_state().cancel(ENHANCER_ID, download_id)


def status() -> dict:
    """The prompt-upsampler status route body for the H3-Image enhancer (the FLUX.3 body's shape)."""
    model = model_path()
    mmproj = mmproj_path()
    binary = pup.find_llama_server_binary()
    model_name = Path(model).name if model else ENHANCER_MODEL_FILE
    size = None
    if model:
        with suppress(OSError):
            size = os.path.getsize(model)
    is_overridden = overridden()
    # Any cached quant with its projector serves: only a missing pair is offered as a download.
    current_downloaded = is_overridden or (model is not None and mmproj is not None)
    missing = (
        []
        if current_downloaded
        else [
            name
            for name, present in ((ENHANCER_MODEL_FILE, model), (ENHANCER_MMPROJ_FILE, mmproj))
            if present is None
        ]
    )
    return {
        "supported": binary is not None,
        "downloaded": model is not None,
        "loaded": bool(binary and model) and pup.get_sidecar().serves(binary, model, None),
        "repo": ENHANCER_REPO,
        "filename": model_name,
        "size_bytes": size
        or ENHANCER_FILE_SIZES.get(model_name)
        or ENHANCER_FILE_SIZES[ENHANCER_MODEL_FILE],
        "mmproj_filename": ENHANCER_MMPROJ_FILE,
        "mmproj_size_bytes": ENHANCER_FILE_SIZES[ENHANCER_MMPROJ_FILE],
        "mmproj_downloaded": mmproj is not None,
        "current_filename": ENHANCER_MODEL_FILE,
        "current_downloaded": current_downloaded,
        "current_missing_bytes": sum(ENHANCER_FILE_SIZES.get(name, 0) for name in missing),
        "override": is_overridden,
        "download": download_status(),
    }


# ── enhancing ───────────────────────────────────────────────────────────────────────────────────────────────────────
def upsample(
    prompt: str,
    *,
    reference_images: Optional[list] = None,
    cancel_event: Optional[threading.Event] = None,
    sidecar: Optional[pup.PromptUpsamplerSidecar] = None,
) -> str:
    """The expanded caption for ``prompt`` (cached). Raises as ``pup.upsample`` does; a reply with no caption (no
    JSON object, no ``expanded_caption``, thinking that never closed) returns ``prompt``."""
    binary = pup.find_llama_server_binary()
    if binary is None:
        raise pup.UpsamplerUnavailableError("llama.cpp (llama-server) is not installed.")
    model = model_path()
    if model is None:
        raise pup.UpsamplerNotDownloadedError(f"The {LABEL} is not downloaded.")
    edit = bool(reference_images)
    mmproj = mmproj_path() if edit else None
    with_images = edit and mmproj is not None
    # Keyed on the uploads as sent, so a repeat (another seed, a re-render) skips the 4 MP resizes as well as the
    # generation. The render seed is not part of it: the templates fix the enhancer's own seed at 0.
    key = (
        ENHANCER_ID,
        prompt,
        edit,
        raw_references_hash(reference_images) if with_images else "",
        model,
    )
    cached = pup._cache_get(key)
    if cached is not None:
        if cancel_event is not None and cancel_event.is_set():
            raise pup.UpsamplerCancelledError("Cancelled.")
        return cached
    references: list[bytes] = []
    if with_images:
        references = prepare_references(reference_images)
    elif edit:
        logger.warning(
            "h3_image.prompt_enhancer: %s is not downloaded; enhancing the edit text-only",
            ENHANCER_MMPROJ_FILE,
        )
    payload = build_chat_payload(prompt, references, edit = edit)
    sidecar = sidecar or pup.get_sidecar()
    raw = sidecar.generate(
        payload,
        binary = binary,
        model = model,
        mmproj = mmproj,
        cancel_event = cancel_event,
        context_tokens = context_tokens_for(prompt, len(references)),
    )
    caption = caption_from_reply(raw)
    if not caption:
        logger.warning(
            "h3_image.prompt_enhancer: no %s in the reply (%d chars); using the prompt as typed",
            JSON_CAPTION_KEY,
            len(raw or ""),
        )
        return prompt
    pup._cache_put(key, caption)
    return caption


def warm(*, expected_epoch: Optional[int] = None, resident: Optional[tuple] = None) -> str:
    """``pup.warm_for_family`` for this enhancer: the t2i server (16k context, the projector when downloaded)."""
    binary = pup.find_llama_server_binary()
    model = model_path()
    if binary is None:
        return "unavailable"
    if model is None:
        return "not_downloaded"
    sidecar = pup.get_sidecar()
    mmproj = mmproj_path()
    if sidecar.is_loading() or sidecar.serves(binary, model, mmproj):
        return sidecar.warm(
            binary = binary, model = model, mmproj = mmproj, expected_epoch = expected_epoch
        )

    def _run() -> None:
        if resident is not None and pup._resident_identity() != resident:
            outcome = "stale"
        else:
            outcome = sidecar.warm(
                binary = binary, model = model, mmproj = mmproj, expected_epoch = expected_epoch
            )
        logger.info("h3_image.prompt_enhancer: warm start %s", outcome)

    threading.Thread(target = _run, name = "h3-image-enhancer-warm", daemon = True).start()
    return "scheduled"
