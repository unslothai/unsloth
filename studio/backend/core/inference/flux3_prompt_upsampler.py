# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 Image Klein prompt upsampling through a llama.cpp sidecar.

FLUX.3 was trained on ~250-word captions, and a short request ("a gymnast doing a backflip") renders broken
anatomy. BFL ships a prompt upsampler (a Qwen3.5-9B fine-tune) that rewrites the request into such a caption;
ComfyUI's FLUX.3 templates run it by default. Studio runs a GGUF of it in one ``llama-server`` on an ephemeral
127.0.0.1 port, loaded on demand and unloaded after the dictation keep-alive (or right after the call when VRAM is
tight, so the DiT and text encoder never denoise next to it). The diffusers graph's own ``prompt_upsampler`` step
stays dropped (``diffusion_flux3.flux3_blocks``); this replaces it.

Prompts and sampling follow ComfyUI's FLUX.3 templates. The system prompts are BFL's ``system_prompts.json``
(black-forest-labs/flux-3-image-klein-pup at commit 8b069751), byte-identical to ``SYSTEM_PROMPTS`` in diffusers'
``modular_pipelines/flux3/prompt_upsampler.py`` at d42ebf9c6e (huggingface/diffusers#15058). The reference
preparation is adapted from that file's ``get_reference_size`` / ``prepare_reference`` (Apache-2.0).

Weights only ever come from the unsloth GGUF repo below or a local override; nothing is fetched implicitly by a
generate call, the UI drives the download. Any failure leaves the prompt as typed.
"""

from __future__ import annotations

import collections
import hashlib
import http.client
import io
import json
import math
import os
import re
import secrets
import subprocess
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from loggers import get_logger

logger = get_logger(__name__)

# ── weights: one place for repo, files and sizes ────────────────────────────────────────────────────────────────────
UPSAMPLER_ID = "flux-3-image-klein-pup"
UPSAMPLER_REPO = "unsloth/flux-3-image-klein-pup-GGUF"
UPSAMPLER_MODEL_FILES = {
    "Q4_K_M": "flux-3-image-klein-pup-Q4_K_M.gguf",
    "Q8_0": "flux-3-image-klein-pup-Q8_0.gguf",
}
UPSAMPLER_DEFAULT_QUANT = "Q4_K_M"
UPSAMPLER_MODEL_FILE = UPSAMPLER_MODEL_FILES[UPSAMPLER_DEFAULT_QUANT]
# Vision projector, needed only for edits (reference images).
UPSAMPLER_MMPROJ_FILE = "mmproj-BF16.gguf"
# Expected sizes of the files as built, for the UI before the download starts (the download reads the real ones).
UPSAMPLER_FILE_SIZES = {
    "flux-3-image-klein-pup-Q4_K_M.gguf": 5_629_109_248,
    "flux-3-image-klein-pup-Q8_0.gguf": 9_527_501_824,
    "mmproj-BF16.gguf": 921_704_928,
}
# Local files instead of the repo, for testing a build before upload.
ENV_MODEL_OVERRIDE = "UNSLOTH_FLUX3_UPSAMPLER_GGUF"
ENV_MMPROJ_OVERRIDE = "UNSLOTH_FLUX3_UPSAMPLER_MMPROJ"
# Append llama-server's output to this file (debugging only).
ENV_SERVER_LOG = "UNSLOTH_FLUX3_UPSAMPLER_SERVER_LOG"

FLUX3_UPSAMPLER_FAMILIES = {"flux.3-klein": "distilled", "flux.3-klein-base": "base"}
VARIANTS = ("distilled", "base")

# ── prompts (ComfyUI FLUX.3 templates; BFL system_prompts.json, verbatim) ───────────────────────────────────────────
SYSTEM_PROMPTS = {
    "t2i": (
        "You are the prompt upsampler of a text-to-image model. Expand the user's short request into ONE detailed, "
        "self-contained caption: the exact text the image model receives. Output only the caption, with no greetings, "
        "explanations, refusals, questions, Markdown, headings, JSON, coordinates, ids or angle-bracket tokens, and no "
        "quotes around it. If the input is not clearly an image request, caption a visualization of its concept. The "
        "request may begin with a \"TARGET CANVAS: W:H\" line: compose for that canvas (wide spreads content sideways, "
        "tall stacks it) and never mention the ratio or the canvas.\n"
        "\n"
        "Fidelity and planning: keep every explicit subject, attribute, count, spatial constraint, style cue and piece "
        "of wording; never contradict, substitute, drop or narrow the request, and keep an abstract concept abstract. "
        "Decide everything left open as ONE concrete, consistent plan; never offer alternatives or hedge. Never "
        "re-express a stated constraint through another mechanism; add camera optics (depth of field, bokeh, blur) only "
        "when asked for or implied. Without a stated style, plan an actual photograph and open with the medium "
        "convention (\"Full-color outdoor photograph, ...\", \"Full-color indoor photograph, ...\", \"Studio "
        "photograph, ...\", \"Close-up photograph, ...\"); keep \"Photorealistic\" for renders, and turn "
        "\"photorealistic\" or \"realistic\" requests for photos into the photograph convention. Give a photograph a "
        "few ordinary, non-ideal specifics drawn from the scene's own objects instead of the stock-photo ideal. Do not "
        "invent names, brands, logos or celebrities; keep user-named ones. A long request keeps its content but not "
        "its wording: drop appraisal and sales vocabulary, turn moods into observable physical facts, and add only new "
        "concrete content with positions. Time of day is lighting, not rendered text.\n"
        "\n"
        "Caption format: one continuous prose block in the generative voice, never \"this image shows\", \"we can "
        "see\" or \"is visible\". Lead with medium and style, then scene, lighting and camera, then the subjects with "
        "concrete colors, materials and state. Every person in focus gets an observable profile: ethnicity, gender, "
        "age range, skin tone (\"fair skin\", \"brown skin\"), hair and clothing. Make the layout reconstructable from "
        "words: frame-relative positions (frame-left, frame-right, upper, lower, foreground, midground, background), "
        "sentence-initial anchors (\"On the left, ...\", \"In the background, ...\"), relations (holds, faces, "
        "behind) and a consistent order; describe up to about ten instances one by one, beyond that give the exact "
        "count and arrangement. Out-of-focus or distant planes are shapes, colors and silhouettes, never identities or "
        "readable text, and the background takes about a sixth of the caption. Be definite: no hedging adverbs, no "
        "quality or emotion judgments, no closing sentence about mood; state negations as what is present. Rendered "
        "text: transcribe every wording verbatim inside double quotes in its original language and script, keep the "
        "line breaks of multi-line text, and give its typography, color and placement; author concrete, domain-correct "
        "wording when the request needs text it does not supply, and name canonical structures (calendar, keyboard, "
        "periodic table) exactly with their full layout. All prose is English; only quoted text keeps its language. "
        "Typical length is 100 to 350 words, more for complex scenes.\n"
        "\n"
        "Output only the caption."
    ),
    "i2i": (
        "You are the prompt upsampler of an image-editing model. You see the reference image(s), attached in order "
        "(the first image, the second image, ...), and the user's short request. Expand it into ONE detailed, "
        "self-contained instruction: the caption the editing model receives together with the same images. Output only "
        "the instruction as one continuous English prose block, with no greetings, explanations, Markdown, JSON, "
        "coordinates, ids or angle-bracket tokens. Read a vague or conversational request as the most plausible edit "
        "of the supplied images.\n"
        "\n"
        "Pick the voice by task shape. When one reference stays and gets bounded changes, write an imperative editing "
        "instruction on it (\"Remove the ...\", \"Change the ... to ...\", \"Zoom in on ...\"); a whole-image "
        "transformation opens with its scope (\"Convert the entire image into ...\", \"Apply a ... filter over the "
        "entire scene\", \"Horizontally flip the entire image so that ...\", \"Shift the camera viewpoint to ...\", "
        "then where the major elements should now be). When the references are drawn into a newly composed frame "
        "(combining subjects, a new scene, a garment, object or style donor), write a generation request (\"Create a "
        "photograph of ...\", \"Create an anime-style illustration of ...\", \"Generate a new image featuring ...\") "
        "that says what to take from each image, by ordinal, and what is newly generated.\n"
        "\n"
        "Keep every explicit edit, constraint, count, position, style cue and piece of wording, and decide what is left "
        "open as one concrete plan. Ground everything in what is actually visible: identify each referent "
        "unambiguously by its real appearance, never mention content that is not there, and use every image the user "
        "attached. State every end state concretely (which color, how large, which pose, which material) and give new "
        "content enough appearance to be generated. Cover the knock-on effects of the change (shadows, reflections, "
        "occlusion, perspective, scale, lighting) and say what fills a removed region. A zoom-in names what stays in "
        "frame and which neighbors are cropped out; an outpainting continues the scene in the same medium. Describe "
        "the change, not the whole scene, and close an editing instruction with one compact clause naming what stays "
        "unchanged; transferred people keep their identity, face and accessories unless asked otherwise. Transcribe "
        "on-image text verbatim in its original language inside double quotes, with line breaks as \\n, and match the "
        "original font, size and color when replacing it. Name recognizable public figures, brands and landmarks you "
        "can see. Directions are frame-relative; use no evaluative or quality words. Typical length is 50 to 150 "
        "words.\n"
        "\n"
        "Output only the instruction."
    ),
}
# The user turn of an edit: the reference images, then this text.
I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {instruction}\n"
    "Output the caption."
)

# ── sampling (ComfyUI FLUX.3 templates) ─────────────────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 3584
_BASE_SAMPLING = {
    "temperature": 1.0,
    "top_k": 1,
    "top_p": 0.95,
    "min_p": 0.0,
    "seed": 0,
}
# (mode, variant) -> (repeat_penalty, presence_penalty)
PENALTIES = {
    ("t2i", "distilled"): (1.05, 1.5),
    ("t2i", "base"): (1.0, 1.5),
    ("i2i", "distilled"): (1.05, 0.0),
    ("i2i", "base"): (1.0, 0.0),
}

# ── reference preparation (diffusers prepare_reference) ─────────────────────────────────────────────────────────────
MAX_REFERENCES = 10
REFERENCE_MULTIPLE = 32
MAX_ASPECT_RATIO = 200
MAX_PIXELS_BY_REFS = {
    1: 1024 * 1024,
    2: 724 * 724,
    **dict.fromkeys(range(3, MAX_REFERENCES + 1), 512 * 512),
}
JPEG_QUALITY = 92
# Header guard on decoded uploads, as diffusion.decode_b64_image's default.
_MAX_UPLOAD_SIDE = 8192

# ── server ──────────────────────────────────────────────────────────────────────────────────────────────────────────
# Holds the system prompt (~1k tokens), up to ten references (256-1024 image tokens each) and the 3584-token caption.
# The model's own 262k default would allocate a KV cache many times the weights.
_CONTEXT_TOKENS = 16384
_SERVER_START_TIMEOUT_SECONDS = 180.0
_REQUEST_TIMEOUT_SECONDS = 600.0
_NUM_LAYERS = 32
# KV cache (8 full-attention layers at 16k), recurrent state and compute buffers on top of the weights.
_GPU_OVERHEAD_MIB = 2560
# Fewer layers than this on the GPU is not worth the split.
_MIN_GPU_LAYERS = 4
# After a call, keep the server resident only when at least this much accelerator memory stays free; else unload it
# before the DiT denoises.
_TIGHT_FREE_MIB = 6144
_TIGHT_FREE_FRACTION = 0.20
_CACHE_ENTRIES = 128

STATUS_APPLIED = "applied"
STATUS_OFF = "off"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"


class UpsamplerUnavailableError(RuntimeError):
    """The upsampler cannot run here: no llama-server, or the GGUF is not downloaded."""


class UpsamplerNotDownloadedError(UpsamplerUnavailableError):
    pass


class UpsamplerCancelledError(RuntimeError):
    pass


# ── family defaults ─────────────────────────────────────────────────────────────────────────────────────────────────
def supports_family(family_name: Optional[str]) -> bool:
    return family_name in FLUX3_UPSAMPLER_FAMILIES


def variant_for_family(family_name: Optional[str]) -> str:
    """"base" for a base family, else "distilled" (the frontend applies the same rule)."""
    return "base" if "base" in (family_name or "") else "distilled"


def resolve_enhance(family_name: Optional[str], enhance_prompt: Optional[bool]) -> bool:
    """Whether a generate call upsamples: None is the family default (on for every FLUX.3 Klein family), and every
    other family ignores the flag."""
    if not supports_family(family_name):
        return False
    return True if enhance_prompt is None else bool(enhance_prompt)


def family_capabilities(family_name: Optional[str]) -> dict:
    """The two fields the loaded-model status carries."""
    supported = supports_family(family_name)
    return {"supports_prompt_upsampler": supported, "prompt_upsampler_default": supported}


# ── request building ────────────────────────────────────────────────────────────────────────────────────────────────
def get_reference_size(width: int, height: int, num_references: int) -> tuple[int, int]:
    """Sides rounded to a multiple of 32, under the pixel budget for this many references."""
    if max(width, height) / max(1, min(width, height)) > MAX_ASPECT_RATIO:
        raise ValueError(
            f"The aspect ratio of a {width}x{height} reference image exceeds {MAX_ASPECT_RATIO}:1."
        )

    def floor(value):
        return max(
            REFERENCE_MULTIPLE, int(math.floor(value / REFERENCE_MULTIPLE)) * REFERENCE_MULTIPLE
        )

    budget = MAX_PIXELS_BY_REFS[min(max(1, num_references), MAX_REFERENCES)]
    w = max(REFERENCE_MULTIPLE, round(width / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    h = max(REFERENCE_MULTIPLE, round(height / REFERENCE_MULTIPLE) * REFERENCE_MULTIPLE)
    if w * h > budget:
        scale = math.sqrt(width * height / budget)
        w, h = floor(width / scale), floor(height / scale)
    if w * h > budget:
        if w == REFERENCE_MULTIPLE:
            h = floor(budget / w)
        else:
            w = floor(budget / h)
    return w, h


def prepare_reference(image: Any, num_references: int) -> bytes:
    """One reference as the upsampler saw them in training, as JPEG bytes: EXIF-upright, RGB over white, LANCZOS to
    the size budget, JPEG quality 92."""
    from PIL import Image, ImageOps

    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    image = image.convert("RGB")
    size = get_reference_size(image.width, image.height, num_references)
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format = "JPEG", quality = JPEG_QUALITY)
    return buffer.getvalue()


def decode_reference(data: str) -> Any:
    """A base64 / data-URL upload as a PIL image, untouched (prepare_reference does the rest)."""
    import base64
    import binascii

    from PIL import Image

    raw = (data or "").strip()
    if raw.startswith("data:"):
        _, _, raw = raw.partition(",")
    try:
        blob = base64.b64decode(raw, validate = False)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"Invalid base64 image data: {exc}") from exc
    try:
        image = Image.open(io.BytesIO(blob))
        w, h = image.size
        if w > _MAX_UPLOAD_SIDE or h > _MAX_UPLOAD_SIDE:
            raise ValueError(f"Image is too large ({w}x{h}).")
        image.load()
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 -- surfaced as a 400 / an unenhanced generate
        raise ValueError(f"Could not decode image: {exc}") from exc
    return image


def prepare_references(images: Optional[list]) -> list[bytes]:
    """Data URLs or PIL images as the upsampler's JPEG references (at most ten, in order)."""
    items = list(images or [])[:MAX_REFERENCES]
    decoded = [decode_reference(item) if isinstance(item, str) else item for item in items]
    return [prepare_reference(image, len(decoded)) for image in decoded]


def references_hash(references: list[bytes]) -> str:
    digest = hashlib.sha256()
    for ref in references:
        digest.update(hashlib.sha256(ref).digest())
    return digest.hexdigest()


def build_chat_payload(
    prompt: str,
    variant: str = "distilled",
    references: Optional[list[bytes]] = None,
    *,
    edit: Optional[bool] = None,
) -> dict:
    """The /v1/chat/completions body. ``references`` are prepared JPEGs; ``edit`` forces the edit prompt without
    images (a missing mmproj runs an edit text-only)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(VARIANTS)}")
    references = list(references or [])
    edit = bool(references) if edit is None else bool(edit)
    mode = "i2i" if edit else "t2i"
    if edit:
        import base64

        content: list[dict] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/jpeg;base64," + base64.b64encode(ref).decode("ascii")
                },
            }
            for ref in references
        ]
        content.append({"type": "text", "text": I2I_SCAFFOLD.format(instruction = prompt)})
    else:
        content = [{"type": "text", "text": prompt}]
    repeat_penalty, presence_penalty = PENALTIES[(mode, variant)]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[mode]},
            {"role": "user", "content": content},
        ],
        **_BASE_SAMPLING,
        "repeat_penalty": repeat_penalty,
        "presence_penalty": presence_penalty,
        "max_tokens": MAX_NEW_TOKENS,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": False,
    }


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


def clean_caption(text: Optional[str]) -> str:
    """The caption without a stray (empty) think block or surrounding quotes and whitespace."""
    text = _THINK_BLOCK.sub("", text or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


# ── weights on disk ─────────────────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen = True)
class _UpsamplerSpec:
    repo: str
    model_file: str
    mmproj_file: str
    label: str


UPSAMPLER_SPEC = _UpsamplerSpec(
    repo = UPSAMPLER_REPO,
    model_file = UPSAMPLER_MODEL_FILE,
    mmproj_file = UPSAMPLER_MMPROJ_FILE,
    label = "FLUX.3 prompt upsampler",
)


def _env_path(name: str) -> Optional[str]:
    value = (os.environ.get(name) or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    return str(path) if path.is_file() else None


def _cached_file(
    filename: str, *, hub_cache: Optional[Path] = None, revision: Optional[str] = None
) -> Optional[str]:
    try:
        from huggingface_hub import hf_hub_download

        from core.inference.stt_sidecar import _active_hf_hub_cache

        root = hub_cache if hub_cache is not None else _active_hf_hub_cache()
        return hf_hub_download(
            repo_id = UPSAMPLER_REPO,
            filename = filename,
            revision = revision,
            local_files_only = True,
            cache_dir = str(root),
        )
    except Exception:
        return None


def _cached_at_any_revision(filename: str, hub_cache: Optional[Path] = None) -> Optional[str]:
    from core.inference.stt_sidecar import _fallback_revisions, _read_revision_record

    recorded = _read_revision_record(UPSAMPLER_REPO)
    candidates = ([recorded] if recorded else []) + [
        rev for rev in _fallback_revisions(UPSAMPLER_REPO, hub_cache = hub_cache) if rev != recorded
    ]
    for revision in candidates:
        path = _cached_file(filename, hub_cache = hub_cache, revision = revision)
        if path is not None:
            return path
    return None


def _cached_download_paths(
    model_id: str, *, hub_cache: Optional[Path] = None, revision: Optional[str] = None
) -> Optional[tuple[str, str]]:
    """Both repo files at one revision, or None (the download state's completion check)."""
    if revision is None:
        model = _cached_at_any_revision(UPSAMPLER_MODEL_FILE, hub_cache)
        mmproj = _cached_at_any_revision(UPSAMPLER_MMPROJ_FILE, hub_cache)
    else:
        model = _cached_file(UPSAMPLER_MODEL_FILE, hub_cache = hub_cache, revision = revision)
        mmproj = _cached_file(UPSAMPLER_MMPROJ_FILE, hub_cache = hub_cache, revision = revision)
    if model is None or mmproj is None:
        return None
    return model, mmproj


def model_path() -> Optional[str]:
    """The upsampler GGUF: the env override, else the downloaded repo file."""
    if (os.environ.get(ENV_MODEL_OVERRIDE) or "").strip():
        # An override that does not exist is a typo, not a reason to read the cache instead.
        return _env_path(ENV_MODEL_OVERRIDE)
    return _cached_at_any_revision(UPSAMPLER_MODEL_FILE)


def mmproj_path() -> Optional[str]:
    if (os.environ.get(ENV_MMPROJ_OVERRIDE) or "").strip():
        return _env_path(ENV_MMPROJ_OVERRIDE)
    return _cached_at_any_revision(UPSAMPLER_MMPROJ_FILE)


def find_llama_server_binary() -> Optional[str]:
    try:
        from core.inference.stt_mtmd_sidecar import find_llama_server_binary as find
        return find()
    except Exception:  # noqa: BLE001 -- no llama.cpp is "not supported", never a crash
        return None


def _resolve_upsampler_id(model_id: Optional[str]) -> str:
    if (model_id or UPSAMPLER_ID) != UPSAMPLER_ID:
        raise ValueError(f"Unknown prompt upsampler '{model_id}'.")
    return UPSAMPLER_ID


_download_state_lock = threading.Lock()
_download_state = None


def _get_download_state():
    global _download_state
    with _download_state_lock:
        if _download_state is None:
            from core.inference.stt_mtmd_sidecar import _MtmdDownloadState

            _download_state = _MtmdDownloadState(
                specs = {UPSAMPLER_ID: UPSAMPLER_SPEC},
                cached_paths = _cached_download_paths,
                resolve = _resolve_upsampler_id,
                label = "FLUX.3 prompt upsampler",
            )
        return _download_state


def start_download(hf_token: Optional[str] = None) -> str:
    return _get_download_state().start(UPSAMPLER_ID, hf_token)


def download_status() -> dict:
    return _get_download_state().status()


def cancel_download(download_id: Optional[str] = None) -> bool:
    return _get_download_state().cancel(UPSAMPLER_ID, download_id)


def status() -> dict:
    """The prompt-upsampler status route body."""
    model = model_path()
    mmproj = mmproj_path()
    model_name = Path(model).name if model and os.environ.get(ENV_MODEL_OVERRIDE) else UPSAMPLER_MODEL_FILE
    size = None
    if model:
        with suppress(OSError):
            size = os.path.getsize(model)
    return {
        "supported": find_llama_server_binary() is not None,
        "downloaded": model is not None,
        "loaded": get_sidecar().loaded,
        "repo": UPSAMPLER_REPO,
        "filename": model_name,
        "size_bytes": size or UPSAMPLER_FILE_SIZES.get(UPSAMPLER_MODEL_FILE),
        "mmproj_filename": UPSAMPLER_MMPROJ_FILE,
        "mmproj_size_bytes": UPSAMPLER_FILE_SIZES.get(UPSAMPLER_MMPROJ_FILE),
        "mmproj_downloaded": mmproj is not None,
        "download": download_status(),
    }


# ── accelerator memory ──────────────────────────────────────────────────────────────────────────────────────────────
def _accelerator_memory_mib() -> tuple[Optional[int], Optional[int], str]:
    """(free, total, device) of the accelerator llama.cpp would offload to, through the diffusion backend's helper.
    ``device`` is cuda / mps / cpu; free is None when unknown."""
    try:
        import torch
        from types import SimpleNamespace

        from core.inference.diffusion_memory import snapshot_device_memory

        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(getattr(torch, "backends", None), "mps", None) is not None and torch.backends.mps.is_available():
            device = "mps"
        else:
            return None, None, "cpu"
        snap = snapshot_device_memory(SimpleNamespace(device = device, backend = device))
        return snap.free_mib, snap.total_mib, device
    except Exception:  # noqa: BLE001 -- unknown memory leaves the choice to the defaults
        return None, None, "unknown"


def plan_gpu_layers(
    free_mib: Optional[int],
    model_bytes: int,
    mmproj_bytes: int = 0,
    device: str = "cuda",
    training: bool = False,
) -> int:
    """How many layers to offload: all when free memory holds weights + overhead, a share of them when it holds some,
    else none. Unknown memory offloads all (llama.cpp's own default); a CPU-only host and training get none."""
    if training or device == "cpu":
        return 0
    if device == "mps" or free_mib is None:
        return 99
    model_mib = model_bytes / (1024 * 1024)
    budget = free_mib - _GPU_OVERHEAD_MIB - mmproj_bytes / (1024 * 1024)
    if budget >= model_mib:
        return 99
    layers = int(_NUM_LAYERS * max(0.0, budget) / max(1.0, model_mib))
    return layers if layers >= _MIN_GPU_LAYERS else 0


def vram_is_tight(free_mib: Optional[int], total_mib: Optional[int]) -> bool:
    if free_mib is None:
        return False
    floor = _TIGHT_FREE_MIB
    if total_mib:
        floor = max(floor, int(total_mib * _TIGHT_FREE_FRACTION))
    return free_mib < floor


# ── the sidecar ─────────────────────────────────────────────────────────────────────────────────────────────────────
class PromptUpsamplerSidecar:
    """One llama-server serving the upsampler GGUF, one request at a time."""

    def __init__(self, keep_alive_seconds: Optional[float] = None) -> None:
        if keep_alive_seconds is None:
            from core.inference.stt_sidecar import STT_KEEP_ALIVE_SECONDS
            keep_alive_seconds = STT_KEEP_ALIVE_SECONDS
        self._keep_alive_seconds = float(keep_alive_seconds)
        self._lock = threading.RLock()
        # One request at a time (--parallel 1); also serialises load against use.
        self._request_lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._api_key: Optional[str] = None
        self._key: Optional[tuple] = None
        self._gpu_layers: Optional[int] = None
        self._loading = False
        self._starting_process: Optional[subprocess.Popen] = None
        self._load_cancel = threading.Event()
        self._update_in_progress = False
        self._idle_timer: Optional[threading.Timer] = None
        self._generation = 0
        self._active_requests = 0

    @property
    def loaded(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    @property
    def loaded_model(self) -> Optional[str]:
        return UPSAMPLER_ID if self.loaded else None

    @property
    def gpu_layers(self) -> Optional[int]:
        return self._gpu_layers if self.loaded else None

    def is_loading(self) -> bool:
        return self._loading

    def holds_vram(self) -> bool:
        return self.loaded and bool(self._gpu_layers)

    # keep-alive
    def _cancel_idle_locked(self) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None

    def _schedule_idle_locked(self) -> None:
        self._cancel_idle_locked()
        if self._keep_alive_seconds <= 0 or self._active_requests:
            return
        generation = self._generation
        timer = threading.Timer(self._keep_alive_seconds, self._idle_unload, args = (generation,))
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _idle_unload(self, generation: int) -> None:
        with self._lock:
            if generation != self._generation or self._active_requests:
                return
            self._release_locked()

    def _release_locked(self) -> None:
        from core.inference.stt_mtmd_sidecar import _reap

        self._cancel_idle_locked()
        self._generation += 1
        process = self._process
        try:
            _reap(process)
        finally:
            self._process = None
            self._port = None
            self._api_key = None
            self._key = None
            self._gpu_layers = None

    def unload(self, wait: bool = True) -> None:
        """Stop the server. ``wait=False`` leaves a busy one alone."""
        if not wait and (self._loading or self._active_requests):
            return
        self.cancel_pending_load()
        if wait:
            deadline = time.monotonic() + 30.0
            while self._active_requests and time.monotonic() < deadline:
                time.sleep(0.1)
        with self._lock:
            if not wait and self._active_requests:
                return
            self._release_locked()

    def cancel_pending_load(self) -> bool:
        if not self._loading:
            return False
        self._load_cancel.set()
        process = self._starting_process
        if process is not None and process.poll() is None:
            with suppress(Exception):
                process.terminate()
        return True

    @contextmanager
    def update_maintenance(self) -> Iterator[bool]:
        """Refuse loads and stop the server while the llama.cpp tree it runs from is replaced."""
        self._update_in_progress = True
        try:
            self.cancel_pending_load()
            with self._request_lock, self._lock:
                was_active = self.loaded
                self._release_locked()
                yield was_active
        finally:
            self._update_in_progress = False

    def _launch(self, binary: str, model: str, mmproj: Optional[str]) -> None:
        """Start llama-server for (model, mmproj). Called under ``_request_lock``."""
        from core.inference.stt_mtmd_sidecar import (
            MtmdSttSidecar,
            _llama_server_child_env,
            _reap,
        )
        from core.inference.stt_sidecar import _training_active
        from utils.process_lifetime import (
            adopt_pid,
            child_popen_kwargs,
            is_process_shutting_down,
            spawn_on_lifetime_thread,
        )

        model_bytes = os.path.getsize(model)
        mmproj_bytes = os.path.getsize(mmproj) if mmproj else 0
        free_mib, _total, device = _accelerator_memory_mib()
        training = _training_active()
        layers = plan_gpu_layers(free_mib, model_bytes, mmproj_bytes, device, training)
        sock, port = MtmdSttSidecar._reserve_free_port()
        cmd = [
            binary,
            "-m",
            model,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--parallel",
            "1",
            "-c",
            str(_CONTEXT_TOKENS),
            "-ngl",
            str(layers),
            "--jinja",
        ]
        if mmproj:
            cmd += ["--mmproj", mmproj]
            if layers == 0:
                cmd.append("--no-mmproj-offload")
        sock.close()
        if is_process_shutting_down():
            raise UpsamplerUnavailableError("Unsloth is shutting down.")
        from core.inference.llama_cpp import (
            LlamaCppBackend,
            _llama_server_api_key_enabled,
            _llama_server_key_launch,
        )

        api_key = key_file = None
        key_env: dict = {}
        if _llama_server_api_key_enabled() and LlamaCppBackend.probe_server_capabilities(
            binary
        ).get("supports_api_key_file", True):
            api_key = secrets.token_urlsafe(32)
            key_argv, key_env, key_file = _llama_server_key_launch(api_key)
            cmd[1:1] = key_argv
        logger.info(
            "flux3.prompt_upsampler: starting llama-server (%s, -ngl %s, free %s MiB on %s%s)",
            Path(model).name,
            layers,
            free_mib,
            device,
            ", vision" if mmproj else "",
        )
        self._load_cancel.clear()
        with self._lock:
            self._loading = True
        # Undrained pipes would block llama-server once its log fills them, so its output goes nowhere unless a log
        # file is asked for.
        log_path = (os.environ.get(ENV_SERVER_LOG) or "").strip()
        log_file = open(log_path, "ab") if log_path else None
        try:
            env = {**_llama_server_child_env(binary), **key_env}
            # Forked from the process-lifetime thread: PDEATHSIG fires when the forking THREAD exits, and requests
            # run on executor workers, so a server spawned there died with its worker and the keep-alive never held.
            process = spawn_on_lifetime_thread(
                lambda: subprocess.Popen(
                    cmd,
                    stdout = log_file or subprocess.DEVNULL,
                    stderr = subprocess.STDOUT if log_file else subprocess.DEVNULL,
                    stdin = subprocess.DEVNULL,
                    env = env,
                    **child_popen_kwargs(),
                )
            )
            with self._lock:
                self._starting_process = process
            adopt_pid(process.pid)
            if is_process_shutting_down():
                _reap(process)
                raise UpsamplerUnavailableError("Unsloth is shutting down.")
            try:
                ready = _wait_for_server(process, port, self._load_cancel)
            finally:
                if key_file is not None:
                    with suppress(OSError):
                        key_file.unlink()
            if not ready:
                _reap(process)
                if self._load_cancel.is_set():
                    raise UpsamplerCancelledError("The prompt upsampler load was cancelled.")
                raise RuntimeError("llama-server did not become ready for the prompt upsampler.")
            with self._lock:
                self._process = process
                self._port = port
                self._api_key = api_key
                self._key = (binary, model, mmproj)
                self._gpu_layers = layers
                self._generation += 1
        finally:
            if log_file is not None:
                with suppress(OSError):
                    log_file.close()
            with self._lock:
                self._loading = False
                self._starting_process = None

    def generate(
        self,
        payload: dict,
        *,
        binary: str,
        model: str,
        mmproj: Optional[str],
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Run one chat completion and return the raw text, (re)starting the server for these files."""
        if self._update_in_progress:
            raise UpsamplerUnavailableError("llama.cpp is being updated.")
        with self._request_lock:
            if cancel_event is not None and cancel_event.is_set():
                raise UpsamplerCancelledError("Cancelled.")
            with self._lock:
                self._cancel_idle_locked()
                current = self._key if self.loaded else None
                # A server with the vision projector also serves text-only requests, so a t2i after an edit does
                # not relaunch.
                if current is not None and not (
                    current[:2] == (binary, model) and (mmproj is None or current[2] == mmproj)
                ):
                    self._release_locked()
            if not self.loaded:
                self._launch(binary, model, mmproj)
            with self._lock:
                port, api_key = self._port, self._api_key
                self._active_requests += 1
            try:
                return _post_chat(port, payload, api_key = api_key, cancel_event = cancel_event)
            finally:
                with self._lock:
                    self._active_requests -= 1
                    self._schedule_idle_locked()

    def release_if_tight(self) -> bool:
        """Unload now when the server holds VRAM and the accelerator is short, so the DiT and text encoder do not
        denoise next to it. True when it unloaded."""
        if not self.holds_vram():
            return False
        free_mib, total_mib, _device = _accelerator_memory_mib()
        if not vram_is_tight(free_mib, total_mib):
            return False
        logger.info(
            "flux3.prompt_upsampler: unloading after the call (free %s of %s MiB)", free_mib, total_mib
        )
        self.unload(wait = False)
        return True


def _wait_for_server(process, port: int, cancel_event: threading.Event) -> bool:
    from core.inference.stt_mtmd_sidecar import MtmdSttSidecar

    return MtmdSttSidecar._wait_for_server(process, port, cancel_event)


def _post_chat(
    port: int,
    payload: dict,
    *,
    api_key: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    from core.inference.stt_sidecar import _close_connection_on_cancel

    connection = http.client.HTTPConnection("127.0.0.1", port, timeout = _REQUEST_TIMEOUT_SECONDS)
    done = threading.Event()
    if cancel_event is not None:
        threading.Thread(
            target = _close_connection_on_cancel,
            args = (connection, cancel_event, done),
            daemon = True,
        ).start()
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            body = json.dumps(payload).encode("utf-8"),
            headers = {
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            },
        )
        with connection.getresponse() as response:
            raw = response.read()
            if not 200 <= response.status < 300:
                raise RuntimeError(f"The prompt upsampler returned HTTP {response.status}.")
        body = json.loads(raw.decode("utf-8"))
    except Exception:
        if cancel_event is not None and cancel_event.is_set():
            raise UpsamplerCancelledError("Cancelled.")
        raise
    finally:
        done.set()
        connection.close()
    choices = body.get("choices") or []
    return (choices[0].get("message", {}).get("content") or "") if choices else ""


_sidecar: Optional[PromptUpsamplerSidecar] = None
_sidecar_lock = threading.Lock()


def get_sidecar() -> PromptUpsamplerSidecar:
    global _sidecar
    with _sidecar_lock:
        if _sidecar is None:
            _sidecar = PromptUpsamplerSidecar()
        return _sidecar


# ── upsampling with cache, progress and fallback ────────────────────────────────────────────────────────────────────
_cache_lock = threading.Lock()
_cache: "collections.OrderedDict[tuple, str]" = collections.OrderedDict()


def _cache_get(key: tuple) -> Optional[str]:
    with _cache_lock:
        value = _cache.get(key)
        if value is not None:
            _cache.move_to_end(key)
        return value


def _cache_put(key: tuple, value: str) -> None:
    with _cache_lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_ENTRIES:
            _cache.popitem(last = False)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# The run in flight, for generate-progress and the Stop button.
_progress_lock = threading.Lock()
_progress: Optional[dict] = None
_progress_cancel: Optional[threading.Event] = None


def progress() -> Optional[dict]:
    """``{"done", "total", "message"}`` while prompts are being enhanced, else None."""
    with _progress_lock:
        return dict(_progress) if _progress is not None else None


def cancel_active() -> bool:
    with _progress_lock:
        event = _progress_cancel
    if event is None or event.is_set():
        return False
    event.set()
    return True


@contextmanager
def _tracking(total: int, cancel_event: threading.Event) -> Iterator[None]:
    global _progress, _progress_cancel
    with _progress_lock:
        _progress = {"done": 0, "total": total, "message": "Enhancing prompt..."}
        _progress_cancel = cancel_event
    try:
        yield
    finally:
        with _progress_lock:
            _progress = None
            _progress_cancel = None


def _advance() -> None:
    with _progress_lock:
        if _progress is not None:
            _progress["done"] += 1


def upsample(
    prompt: str,
    *,
    variant: str = "distilled",
    reference_images: Optional[list] = None,
    cancel_event: Optional[threading.Event] = None,
    sidecar: Optional[PromptUpsamplerSidecar] = None,
) -> str:
    """The caption for ``prompt`` (cached). Raises UpsamplerUnavailableError when nothing can run, a cancel as
    UpsamplerCancelledError, and anything else the server does. An empty caption returns ``prompt``."""
    binary = find_llama_server_binary()
    if binary is None:
        raise UpsamplerUnavailableError("llama.cpp (llama-server) is not installed.")
    model = model_path()
    if model is None:
        raise UpsamplerNotDownloadedError("The FLUX.3 prompt upsampler is not downloaded.")
    edit = bool(reference_images)
    mmproj = mmproj_path() if edit else None
    references: list[bytes] = []
    if edit:
        if mmproj is None:
            logger.warning(
                "flux3.prompt_upsampler: %s is not downloaded; enhancing the edit text-only",
                UPSAMPLER_MMPROJ_FILE,
            )
        else:
            references = prepare_references(reference_images)
    payload = build_chat_payload(prompt, variant, references, edit = edit)
    key = (prompt, variant, edit, references_hash(references), model)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    sidecar = sidecar or get_sidecar()
    text = clean_caption(
        sidecar.generate(
            payload, binary = binary, model = model, mmproj = mmproj, cancel_event = cancel_event
        )
    )
    if not text:
        logger.warning("flux3.prompt_upsampler: empty caption; using the prompt as typed")
        return prompt
    _cache_put(key, text)
    return text


@dataclass
class EnhanceOutcome:
    """Per-prompt captions for one generate call. ``status`` is None for families without an upsampler."""

    status: Optional[str]
    captions: dict = field(default_factory = dict)
    statuses: dict = field(default_factory = dict)

    def caption_for(self, prompt: str) -> Optional[str]:
        return self.captions.get(prompt)

    def status_for(self, prompt: str) -> Optional[str]:
        return self.statuses.get(prompt, self.status)

    def encoded(self, prompt: str) -> str:
        return self.captions.get(prompt) or prompt


def enhance_for_generation(
    prompts: list[str],
    *,
    family_name: Optional[str],
    enhance_prompt: Optional[bool],
    reference_images: Optional[list] = None,
    cancel_event: Optional[threading.Event] = None,
) -> EnhanceOutcome:
    """Captions for each distinct prompt of a generate call. Never raises except on cancel: anything that stops the
    upsampler leaves the prompt as typed and says why in the status."""
    if not supports_family(family_name):
        return EnhanceOutcome(status = None)
    if not resolve_enhance(family_name, enhance_prompt):
        return EnhanceOutcome(status = STATUS_OFF)
    if find_llama_server_binary() is None or model_path() is None:
        logger.warning(
            "flux3.prompt_upsampler: unavailable (%s); generating with the prompt as typed",
            "llama-server not found" if find_llama_server_binary() is None else "GGUF not downloaded",
        )
        return EnhanceOutcome(status = STATUS_UNAVAILABLE)
    variant = variant_for_family(family_name)
    distinct = list(dict.fromkeys(p for p in prompts if isinstance(p, str)))
    cancel_event = cancel_event or threading.Event()
    outcome = EnhanceOutcome(status = STATUS_APPLIED)
    sidecar = get_sidecar()
    with _tracking(len(distinct), cancel_event):
        for prompt in distinct:
            if cancel_event.is_set():
                raise UpsamplerCancelledError("Cancelled.")
            try:
                caption = upsample(
                    prompt,
                    variant = variant,
                    reference_images = reference_images,
                    cancel_event = cancel_event,
                    sidecar = sidecar,
                )
            except UpsamplerCancelledError:
                raise
            except UpsamplerUnavailableError as exc:
                logger.warning("flux3.prompt_upsampler: unavailable (%s); prompt as typed", exc)
                outcome.statuses[prompt] = STATUS_UNAVAILABLE
            except Exception as exc:  # noqa: BLE001 -- an upsampler failure never fails the image
                if cancel_event.is_set():
                    raise UpsamplerCancelledError("Cancelled.") from exc
                logger.warning("flux3.prompt_upsampler: failed (%s); prompt as typed", exc)
                outcome.statuses[prompt] = STATUS_FAILED
            else:
                if caption == prompt:
                    outcome.statuses[prompt] = STATUS_FAILED
                else:
                    outcome.captions[prompt] = caption
                    outcome.statuses[prompt] = STATUS_APPLIED
            _advance()
    with suppress(Exception):
        sidecar.release_if_tight()
    if outcome.statuses and not outcome.captions:
        # Nothing applied: the batch-level status is the first reason.
        outcome.status = next(iter(outcome.statuses.values()))
    return outcome
