# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 prompt upsampler via llama-server; weights never fetched by generate, failures keep prompt"""

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
    "UD-Q4_K_XL": "flux-3-image-klein-pup-UD-Q4_K_XL.gguf",
    "Q4_K_M": "flux-3-image-klein-pup-Q4_K_M.gguf",
    "Q8_0": "flux-3-image-klein-pup-Q8_0.gguf",
}
# Unsloth Dynamic 4-bit with an imatrix: against BF16 on the upsampler's own captions, about 39% lower mean KL
# divergence than the imatrix Q4_K_M for 6% more bytes, at the same speed.
UPSAMPLER_DEFAULT_QUANT = "UD-Q4_K_XL"
UPSAMPLER_MODEL_FILE = UPSAMPLER_MODEL_FILES[UPSAMPLER_DEFAULT_QUANT]
# What a run reads, best first: an install holding Q4_K_M (the default until UD-Q4_K_XL) keeps enhancing offline.
UPSAMPLER_CACHED_FALLBACK_QUANTS = ("Q8_0", "Q4_K_M")
UPSAMPLER_MODEL_FILE_PREFERENCE = (UPSAMPLER_MODEL_FILE,) + tuple(
    UPSAMPLER_MODEL_FILES[quant] for quant in UPSAMPLER_CACHED_FALLBACK_QUANTS
)
# Q8_0 is closer to BF16 than the default: a complete Q8_0 pair is not offered a smaller download "for better captions".
UPSAMPLER_SUPERSEDED_FILES = frozenset((UPSAMPLER_MODEL_FILES["Q4_K_M"],))
# Vision projector, needed only for edits (reference images).
UPSAMPLER_MMPROJ_FILE = "mmproj-F16.gguf"
# Expected sizes of the files as built, for the UI before the download starts (the download reads the real ones).
UPSAMPLER_FILE_SIZES = {
    "flux-3-image-klein-pup-UD-Q4_K_XL.gguf": 5_966_095_584,
    "flux-3-image-klein-pup-Q4_K_M.gguf": 5_629_109_472,
    "flux-3-image-klein-pup-Q8_0.gguf": 9_527_501_824,
    "mmproj-F16.gguf": 918_165_984,
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
        'request may begin with a "TARGET CANVAS: W:H" line: compose for that canvas (wide spreads content sideways, '
        "tall stacks it) and never mention the ratio or the canvas.\n"
        "\n"
        "Fidelity and planning: keep every explicit subject, attribute, count, spatial constraint, style cue and piece "
        "of wording; never contradict, substitute, drop or narrow the request, and keep an abstract concept abstract. "
        "Decide everything left open as ONE concrete, consistent plan; never offer alternatives or hedge. Never "
        "re-express a stated constraint through another mechanism; add camera optics (depth of field, bokeh, blur) only "
        "when asked for or implied. Without a stated style, plan an actual photograph and open with the medium "
        'convention ("Full-color outdoor photograph, ...", "Full-color indoor photograph, ...", "Studio '
        'photograph, ...", "Close-up photograph, ..."); keep "Photorealistic" for renders, and turn '
        '"photorealistic" or "realistic" requests for photos into the photograph convention. Give a photograph a '
        "few ordinary, non-ideal specifics drawn from the scene's own objects instead of the stock-photo ideal. Do not "
        "invent names, brands, logos or celebrities; keep user-named ones. A long request keeps its content but not "
        "its wording: drop appraisal and sales vocabulary, turn moods into observable physical facts, and add only new "
        "concrete content with positions. Time of day is lighting, not rendered text.\n"
        "\n"
        'Caption format: one continuous prose block in the generative voice, never "this image shows", "we can '
        'see" or "is visible". Lead with medium and style, then scene, lighting and camera, then the subjects with '
        "concrete colors, materials and state. Every person in focus gets an observable profile: ethnicity, gender, "
        'age range, skin tone ("fair skin", "brown skin"), hair and clothing. Make the layout reconstructable from '
        "words: frame-relative positions (frame-left, frame-right, upper, lower, foreground, midground, background), "
        'sentence-initial anchors ("On the left, ...", "In the background, ..."), relations (holds, faces, '
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
        'instruction on it ("Remove the ...", "Change the ... to ...", "Zoom in on ..."); a whole-image '
        'transformation opens with its scope ("Convert the entire image into ...", "Apply a ... filter over the '
        'entire scene", "Horizontally flip the entire image so that ...", "Shift the camera viewpoint to ...", '
        "then where the major elements should now be). When the references are drawn into a newly composed frame "
        '(combining subjects, a new scene, a garment, object or style donor), write a generation request ("Create a '
        'photograph of ...", "Create an anime-style illustration of ...", "Generate a new image featuring ...") '
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
    "t2i_bbox": (
        "You are the prompt upsampler of a text-to-image model. Plan the full image the user's short request implies and "
        "output ONE structured caption on a single line: the global caption paragraph, one space, then ONE JSON array, "
        "the element table that places every cited element with a bounding box in the image to be generated. Output "
        "nothing else (no Markdown fences, commentary or refusals); visualize the concept of a non-image input. The "
        'request may begin with a "TARGET CANVAS: W:H" line: compose for that canvas and never mention it. Boxes live on '
        "a 0-1000 grid stretched onto the canvas: an element of physical width-to-height ratio a gets an x-span of about "
        "y-span x a x H/W.\n"
        "\n"
        "Planning: keep every explicit subject, attribute, count, spatial constraint, style cue and "
        "piece of wording, and decide everything left open as ONE concrete composition that the caption and the table "
        'both describe. Without a stated style plan an actual photograph and open with the medium convention ("Full-color '
        'outdoor photograph, ...", "Studio photograph, ..."); add camera optics only when asked for; do not invent names, '
        "brands, logos or celebrities; drop appraisal and sales vocabulary.\n"
        "\n"
        "The table: one row per distinct object, "
        "person, animal, building or salient region the caption mentions and one per coherent text block; no rows for "
        "style, lighting or atmosphere. Structured layouts (tables, grids, calendars, keyboards, charts, maps, posters, "
        "pages) decompose into their regions, each row carrying that region's complete verbatim content; more than about "
        "ten interchangeable items become one collection row giving the exact count and arrangement, while items whose "
        'positions matter keep a row each. Row keys, in order: "id", "bbox", "desc". id: the bare class noun plus an '
        "index numbered left to right, then top to bottom, from 1 (person_1, dog_2); a text block is its two-letter "
        "language code plus _Text_ plus an index (En_Text_1); one entity is one row and ids are unique. bbox: [y_min, "
        "x_min, y_max, x_max] as integers from 0 to 1000, top-left origin, clamped at the borders, with plausible "
        "relative sizes and a small margin around text. desc: one self-contained paragraph in the generative voice that "
        'opens with the element\'s identity (never "the X"): shape, material, colors, state and markings, but no position. '
        "People in focus get the richest desc: ethnicity, gender, age range, skin tone, build, hair, face, each garment, "
        "accessories, pose and expression. A text row carries its complete wording verbatim in double quotes, with line "
        "breaks as \\n, plus its typography.\n"
        "\n"
        "The global caption: one flowing paragraph in the generative voice (medium "
        "first, then scene, lighting, camera and frame-relative layout; no quality judgments) that names each table "
        'element at identity level at its first mention and cites it with its id in its own angle brackets: "a wicker '
        'table <table_1> holding a tiered stand <stand_1>". Cite every table id and no other; appearance detail and text '
        "content live in the descs; no coordinates. Text-dense documents are laid out region by region with every visible "
        "text transcribed verbatim at a readable size (cut content rather than shrink type), and a grid quotes every "
        "visible cell. All prose is English; only quoted text keeps its language.\n"
        "\n"
        "Output only the caption and the table."
    ),
    "i2i_bbox": (
        "You are the annotation system an image-editing model was trained on. You see the source image(s), attached in "
        "order as ref_image_0, ref_image_1, ..., and the user's editing instruction; the edited target image does not "
        "exist yet. Determine the edit precisely, imagine the target consistently with the source, and write the record "
        "its annotator would have written, on one line: the instruction paragraph, one space, then ONE JSON array. Output "
        "nothing else.\n"
        "\n"
        'The paragraph: English, instruction voice ("Change the ...", "Remove the ...", or "Create a '
        'photograph of ..." when the references are drawn into a new frame), never observer phrasing and never a '
        "comparison with the target. Cite every involved object, person or text block at its first mention with its table "
        'id in its own angle brackets ("remove the delivery van <van_1>"), every table id exactly once and no other id; '
        'cite a reference image by its token when content is taken from it ("the cup <cup_1> from <ref_image_1>"). End '
        "states (new color, new text in quotes, pose, degree) live in the paragraph, static appearance in the descs; no "
        "coordinates. An editing instruction closes with one compact clause that everything else stays identical to the "
        "source.\n"
        "\n"
        'The table, rows {"id", "from", "src_bbox", "tgt_bbox", "desc"}: one row per edited, removed, added, '
        "moved or anchoring item, plus the scene or background when it is itself changed or preserved; no rows for "
        "untouched scenery. id: a lowercase class noun plus an index (person_1, cup_2) numbered left to right, then top "
        'to bottom; text blocks are En_Text_1 and so on; ids are unique. from: "ref_image_N" for an item visible in '
        'reference N, "target" for newly created content. src_bbox is the item\'s box in its from-image and null for '
        "target rows; tgt_bbox is its box in the imagined result and null for removed items; moved or resized items carry "
        "both; new items are sized like similar objects in the frame. Boxes are [y_min, x_min, y_max, x_max] integers "
        "from 0 to 1000, top-left origin, tight around objects, with 5 to 15 units of margin around text. A row may also "
        "carry src_bbox_1stage and tgt_bbox_1stage, a first-pass estimate of the same boxes. desc: one English paragraph "
        "of the item's static appearance in its from-image (for target rows, its decided appearance), opening with its "
        "identity; people get the richest descs (ethnicity, gender, age range, skin tone, hair, face, garments, "
        "accessories, pose, expression); text rows quote their text verbatim.\n"
        "\n"
        "State only decided facts: visible in the "
        "source, demanded by the instruction, or a concrete default needed to realize it. Settle whether the edit targets "
        "one instance, all of a class or several; land every magnitude on a concrete value; include visible knock-on "
        "effects; added objects are new objects at real-world size; a zoom-in names the neighbors it excludes; an "
        "outpainting keeps the source whole and continues it in the same medium. Everything the instruction does not "
        "involve stays identical to the source.\n"
        "\n"
        "Output only the paragraph and the table."
    ),
}
# The user turn of an edit: the reference images, then this text.
I2I_SCAFFOLD = (
    "Reference images follow in order (the first image, the second image, ...).\n"
    "User request: {instruction}\n"
    "Output the caption."
)
# The same for a layout edit (``i2i_bbox``), from BFL's flux3/pup.py.
I2I_BBOX_SCAFFOLD = (
    "Reference images follow in attachment order: ref_image_0, ref_image_1, ...\n"
    "User editing instruction: {instruction}\n"
    "Decide the edit, imagine the target image, and output the annotation record."
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
# ComfyUI and BFL's reference penalize every generated token and never the prompt. llama-server feeds the prompt into
# the penalty history unless it has `penalize_prompt` (unslothai/llama.cpp#283); with it the window spans the whole
# caption. Older servers keep their default 64-token window: a wider one there would mostly penalize the system prompt.
GENERATED_ONLY_PENALTIES = {"penalize_prompt": False, "repeat_last_n": 4096}

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
# Header guard on decoded uploads, diffusion.decode_b64_image's default: a larger reference fails the generate anyway.
_MAX_UPLOAD_SIDE = 4096

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
    """ "base" for a base family, else "distilled" (the frontend applies the same rule)."""
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
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
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


# Decoded uploads are large (an 8192x8192 RGBA is 256 MB): one at a time per request, two requests at once.
_DECODE_SLOTS = threading.BoundedSemaphore(2)


def prepare_references(images: Optional[list]) -> list[bytes]:
    """Data URLs or PIL images as the upsampler's JPEG references (at most ten, in order)."""
    items = list(images or [])[:MAX_REFERENCES]
    prepared = []
    with _DECODE_SLOTS:
        for item in items:
            image = decode_reference(item) if isinstance(item, str) else item
            prepared.append(prepare_reference(image, len(items)))
            del image
    return prepared


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
    locked: Optional[list[dict]] = None,
) -> dict:
    """The /v1/chat/completions body. ``references`` are prepared JPEGs; ``edit`` forces the edit prompt without
    images (a missing mmproj runs an edit text-only). ``locked`` box rows select the layout prompts
    (``t2i_bbox`` / ``i2i_bbox``) and wrap ``prompt`` in BFL's locked-annotation request."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(VARIANTS)}")
    references = list(references or [])
    edit = bool(references) if edit is None else bool(edit)
    mode = "i2i" if edit else "t2i"
    system_key = f"{mode}_bbox" if locked else mode
    if locked:
        prompt = layout_request(prompt, locked)
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
        scaffold = I2I_BBOX_SCAFFOLD if locked else I2I_SCAFFOLD
        content.append({"type": "text", "text": scaffold.format(instruction = prompt)})
    else:
        content = [{"type": "text", "text": prompt}]
    repeat_penalty, presence_penalty = PENALTIES[(mode, variant)]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPTS[system_key]},
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


# ── layout captions (bounding boxes) ────────────────────────────────────────────────────────────────────────────────
# Adapted from BFL's src/flux3/bbox.py (black-forest-labs/flux-3-image-klein at bfc8889a, Apache-2.0). Boxes arrive
# inside the prompt: "<caption> [rows]", rows {"id", "bbox": [y_min, x_min, y_max, x_max] on 0-1000, "desc"} (edits:
# "from", "src_bbox", "tgt_bbox"). The plain t2i / i2i prompts forbid ids and coordinates and would drop them.
BOX_CANVAS = 1000
_BOX_ID = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def split_box_table(text: str) -> tuple[str, Optional[list]]:
    """(caption, rows) for the JSON list closing the text, if any. Not a regex: descs may hold brackets."""
    for i in range(len(text) - 1, -1, -1):
        if text[i] == "[":
            try:
                rows = json.loads(text[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(rows, list):
                return text[:i].rstrip(), rows
    return text.strip(), None


def valid_canvas_box(box: Any) -> Optional[list]:
    """The box if it is a well-formed [y_min, x_min, y_max, x_max] on the 0-1000 canvas, else None."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or abs(v) == float("inf")
        for v in box
    ):
        return None
    y1, x1, y2, x2 = box
    # max <= 1 means 0-1 coordinates, not the 0-1000 canvas
    if (
        y1 < 0
        or x1 < 0
        or y2 <= y1
        or x2 <= x1
        or y2 > BOX_CANVAS
        or x2 > BOX_CANVAS
        or max(box) <= 1
    ):
        return None
    return list(box)


def detect_box_table(prompt: str) -> Optional[tuple[str, list[dict]]]:
    """(caption, rows) when ``prompt`` ends in a box table of only box rows (not a caption ending in "[SALE]")."""
    text, rows = split_box_table(prompt)
    if not rows or not all(
        isinstance(row, dict)
        and isinstance(row.get("id"), str)
        and _BOX_ID.fullmatch(row["id"])
        and any(valid_canvas_box(row.get(key)) for key in ("bbox", "src_bbox", "tgt_bbox"))
        for row in rows
    ):
        return None
    locked = re.fullmatch(
        r"ORIGINAL PROMPT:\n(.*?)\n\nLOCKED BOUNDING-BOX ANNOTATIONS .*", text, re.S
    )
    return (locked.group(1) if locked else text), rows


def is_finished_layout_caption(prompt: str) -> bool:
    """Every box id cited as <id> in the caption: an upsampler answer, which BFL's generate.py sends as it is."""
    found = detect_box_table(prompt)
    return found is not None and all(f"<{row['id']}>" in found[0] for row in found[1])


def layout_request(prompt: str, rows: list[dict]) -> str:
    return (
        f"ORIGINAL PROMPT:\n{prompt.strip()}\n\n"
        "LOCKED BOUNDING-BOX ANNOTATIONS ([y_min, x_min, y_max, x_max] on a 0-1000 canvas, with (0, 0) at "
        "the top-left). This is an authoritative JSON list of dictionaries; preserve every supplied identity, "
        f"provenance value, null, and coordinate exactly:\n{json.dumps(rows, ensure_ascii = False)}"
    )


def _valid_added_row(row: Any, caption: str) -> Optional[dict]:
    if not isinstance(row, dict):
        return None
    row_id, desc = row.get("id"), row.get("desc")
    if (
        not isinstance(row_id, str)
        or not _BOX_ID.fullmatch(row_id)
        or f"<{row_id}>" not in caption
        or not isinstance(desc, str)
        or not desc.strip()
    ):
        return None
    if (box := valid_canvas_box(row.get("bbox"))) is not None:
        return {"id": row_id, "bbox": box, "desc": desc.strip()}
    origin, src, tgt = row.get("from"), row.get("src_bbox"), row.get("tgt_bbox")
    src_box = valid_canvas_box(src) if src is not None else None
    tgt_box = valid_canvas_box(tgt) if tgt is not None else None
    if (
        not isinstance(origin, str)
        or not re.fullmatch(r"ref_image_\d+|target", origin)
        or (src is not None and src_box is None)
        or (tgt is not None and tgt_box is None)
        or (src_box is None and tgt_box is None)
        or (origin == "target") == (src_box is not None)
    ):
        return None
    return {
        "id": row_id,
        "from": origin,
        "src_bbox": src_box,
        "tgt_bbox": tgt_box,
        "desc": desc.strip(),
    }


def _first_valid_box(row: dict) -> Optional[list]:
    for key in ("bbox", "tgt_bbox", "src_bbox"):
        if (box := valid_canvas_box(row.get(key))) is not None:
            return box
    return None


def _box_iou(a: list, b: list) -> float:
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def reconcile_layout_caption(text: str, locked: list[dict]) -> str:
    """The upsampler's answer with the user's boxes re-imposed. A row re-planning a locked box under its own id
    (IoU >= 0.9) takes the locked id, in the rows and the caption's citations; the locked rows are kept verbatim with
    the upsampler's description; rows it added follow if they are valid."""
    caption, rows = split_box_table(text)
    rows = rows or []
    locked_ids = {row["id"] for row in locked}
    row_ids = {row.get("id") for row in rows if isinstance(row, dict)}
    renamed: dict = {}
    for lock in locked:
        lock_box = _first_valid_box(lock)
        if lock["id"] in row_ids or lock_box is None:
            continue
        for row in rows:
            if not isinstance(row, dict) or row.get("id") in renamed or row.get("id") in locked_ids:
                continue
            box = _first_valid_box(row)
            if box is not None and _box_iou(box, lock_box) >= 0.9:
                renamed[row["id"]] = lock["id"]
                break
    for row in rows:
        if isinstance(row, dict) and row.get("id") in renamed:
            row["id"] = renamed[row["id"]]
    for old, new in renamed.items():
        caption = caption.replace(f"<{old}>", f"<{new}>")
    by_id = {
        row.get("id"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    out = []
    for lock in locked:
        desc = by_id.get(lock["id"], {}).get("desc")
        out.append(
            {
                **lock,
                "desc": desc.strip()
                if isinstance(desc, str) and desc.strip()
                else lock.get("desc", ""),
            }
        )
    for row in rows:
        added = _valid_added_row(row, caption)
        if added is not None and added["id"] not in {r["id"] for r in out}:
            out.append(added)
    return f"{caption} {json.dumps(out, ensure_ascii = False)}"


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
    filename: str,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
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


def _candidate_revisions(hub_cache: Optional[Path] = None) -> list:
    from core.inference.stt_sidecar import _fallback_revisions, _read_revision_record
    recorded = _read_revision_record(UPSAMPLER_REPO)
    return ([recorded] if recorded else []) + [
        rev for rev in _fallback_revisions(UPSAMPLER_REPO, hub_cache = hub_cache) if rev != recorded
    ]


def _cached_at_any_revision(filename: str, hub_cache: Optional[Path] = None) -> Optional[str]:
    for revision in _candidate_revisions(hub_cache):
        path = _cached_file(filename, hub_cache = hub_cache, revision = revision)
        if path is not None:
            return path
    return None


def _cached_pair(
    hub_cache: Optional[Path] = None, model_files: tuple = UPSAMPLER_MODEL_FILE_PREFERENCE
) -> tuple[Optional[str], Optional[str]]:
    """The model and projector from ONE revision (the recorded one first), as the downloader pins them; a model
    with no projector at its revision when no revision holds both. A complete pair of any ``model_files`` beats a
    model alone, so an interrupted default download does not cost edits their reference images."""
    revisions = _candidate_revisions(hub_cache)
    model_only = None
    for model_file in model_files:
        for revision in revisions:
            model = _cached_file(model_file, hub_cache = hub_cache, revision = revision)
            if model is None:
                continue
            mmproj = _cached_file(UPSAMPLER_MMPROJ_FILE, hub_cache = hub_cache, revision = revision)
            if mmproj is not None:
                return model, mmproj
            model_only = model_only or model
    return model_only, None


def _cached_download_paths(
    model_id: str,
    *,
    hub_cache: Optional[Path] = None,
    revision: Optional[str] = None,
) -> Optional[tuple[str, str]]:
    """Both repo files at one revision, or None (the download state's completion check). A pinned revision is the
    one just downloaded, so it checks the default quant the download fetched; without one, any cached quant counts."""
    if revision is None:
        model, mmproj = _cached_pair(hub_cache)
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
    return _cached_pair()[0]


def mmproj_path() -> Optional[str]:
    if (os.environ.get(ENV_MMPROJ_OVERRIDE) or "").strip():
        return _env_path(ENV_MMPROJ_OVERRIDE)
    if _overridden():
        # Beside a local model file there is no revision to match.
        return _cached_at_any_revision(UPSAMPLER_MMPROJ_FILE)
    return _cached_pair()[1]


def _overridden() -> bool:
    return any(
        (os.environ.get(name) or "").strip() for name in (ENV_MODEL_OVERRIDE, ENV_MMPROJ_OVERRIDE)
    )


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
    model_name = Path(model).name if model else UPSAMPLER_MODEL_FILE
    size = None
    if model:
        with suppress(OSError):
            size = os.path.getsize(model)
    overridden = _overridden()
    # Not the unpinned _cached_download_paths: any cached quant completes that one.
    current_downloaded = (
        overridden
        or None not in _cached_pair(model_files = (UPSAMPLER_MODEL_FILE,))
        or (mmproj is not None and model_name not in UPSAMPLER_SUPERSEDED_FILES)
    )
    # Bytes still to fetch: a file already cached at any revision is a blob the hub cache reuses.
    missing = (
        []
        if current_downloaded
        else [
            name
            for name in (UPSAMPLER_MODEL_FILE, UPSAMPLER_MMPROJ_FILE)
            if _cached_at_any_revision(name) is None
        ]
    )
    return {
        "supported": find_llama_server_binary() is not None,
        "downloaded": model is not None,
        "loaded": get_sidecar().loaded,
        "repo": UPSAMPLER_REPO,
        "filename": model_name,
        "size_bytes": size
        or UPSAMPLER_FILE_SIZES.get(model_name)
        or UPSAMPLER_FILE_SIZES.get(UPSAMPLER_MODEL_FILE),
        "mmproj_filename": UPSAMPLER_MMPROJ_FILE,
        "mmproj_size_bytes": UPSAMPLER_FILE_SIZES.get(UPSAMPLER_MMPROJ_FILE),
        "mmproj_downloaded": mmproj is not None,
        "current_filename": UPSAMPLER_MODEL_FILE,
        "current_downloaded": current_downloaded,
        "current_missing_bytes": sum(UPSAMPLER_FILE_SIZES.get(name, 0) for name in missing),
        # Local files from the env overrides: the repo download is not offered beside them.
        "override": overridden,
        "download": download_status(),
    }


# ── accelerator memory ──────────────────────────────────────────────────────────────────────────────────────────────
def _resident_cuda_ordinal() -> Optional[int]:
    """The CUDA card the resident image model sits on, else None."""
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        ordinal = getattr(
            getattr(get_active_diffusion_engine(), "_state", None), "placed_ordinal", None
        )
        return None if ordinal is None else int(ordinal)
    except Exception:  # noqa: BLE001
        return None


def _child_device_mask(ordinal: Optional[int]) -> dict:
    """Pin llama-server to the image model's card: left alone it splits layers over every visible GPU, onto a card the
    memory probe never read."""
    # LLAMA_ARG_DEVICE is the user's own pick, and names devices a narrowed mask would renumber.
    if ordinal is None or ordinal < 0 or str(os.environ.get("LLAMA_ARG_DEVICE", "")).strip():
        return {}
    try:
        import torch

        if torch.cuda.device_count() <= 1:
            return {}
        from core.inference.llama_cpp import LlamaCppBackend

        var = "CUDA_VISIBLE_DEVICES"
        # ROCm reads a non-empty HIP_VISIBLE_DEVICES over CUDA_VISIBLE_DEVICES; an empty one counts as unset.
        if LlamaCppBackend._torch_is_rocm(torch) and (
            (os.environ.get("HIP_VISIBLE_DEVICES") or "").strip() or os.environ.get(var) is None
        ):
            var = "HIP_VISIBLE_DEVICES"
        parent = (os.environ.get(var) or "").strip() or None
        if parent is None:
            return {var: str(ordinal)}
        # Torch indexes inside the parent's mask, so the child keeps that entry (an index or a UUID).
        entries = [entry.strip() for entry in parent.split(",") if entry.strip()]
        if var == "HIP_VISIBLE_DEVICES":
            # clr keeps the first of a repeated HIP entry (utils/hardware/amd.py), so 0,0,1 exposes two devices.
            entries = list(dict.fromkeys(entries))
        return {var: entries[ordinal]} if 0 <= ordinal < len(entries) else {}
    except Exception:  # noqa: BLE001 -- no mask: llama.cpp's own device choice
        return {}


_VULKAN_ROWS_TTL_S = 120.0
_vulkan_rows_memo: dict = {}
# One probe at a time: concurrent prefetches read the answer the first one cached.
_vulkan_rows_lock = threading.Lock()


def _vulkan_rows(
    binary: str,
    retry_empty: bool = False,
    throttle_retry: bool = False,
) -> list:
    """Vulkan probe rows, cached briefly; ``retry_empty`` re-probes a cached (maybe transient) empty answer. With
    ``throttle_retry`` (a speculative prefetch) an empty answer that already was a retry stands until the window ends."""
    from core.inference.llama_cpp import LlamaCppBackend

    try:
        key = (binary, os.path.getmtime(binary))
    except OSError:
        key = (binary, None)
    with _vulkan_rows_lock:
        hit = _vulkan_rows_memo.get(key)
        live = hit is not None and time.monotonic() - hit[1] < _VULKAN_ROWS_TTL_S
        # Only a speculative prefetch is throttled: a real launch always asks again.
        if live and (hit[0] or not retry_empty or (throttle_retry and len(hit) > 2 and hit[2])):
            return hit[0]
        rows = list(LlamaCppBackend._run_vulkan_probe(binary) or [])
        _vulkan_rows_memo.clear()
        # (rows, probed at, this probe retried a live empty answer)
        _vulkan_rows_memo[key] = (rows, time.monotonic(), live)
        return rows


def _user_device_pick(binary: Optional[str] = None) -> bool:
    """LLAMA_ARG_DEVICE / LLAMA_ARG_MAIN_GPU / Vulkan mask set by the user; with ``binary``, a Vulkan mask counts
    only where Vulkan may load (CUDA and HIP ignore it)."""
    if any(
        str(os.environ.get(var, "")).strip() for var in ("LLAMA_ARG_DEVICE", "LLAMA_ARG_MAIN_GPU")
    ):
        return True
    if not str(os.environ.get("GGML_VK_VISIBLE_DEVICES", "")).strip():
        return False
    if binary is None or str(os.environ.get("GGML_BACKEND_PATH", "")).strip():
        return True
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return bool(LlamaCppBackend._is_vulkan_backend(binary))
    except Exception:  # noqa: BLE001
        return True


def _vulkan_pin_possible(binary: str, torch) -> bool:
    """Whether a Vulkan pin is worth the probe: a torch card, a Vulkan build, nothing the user picked."""
    from core.inference.llama_cpp import LlamaCppBackend
    return (
        torch.cuda.device_count() >= 1
        and not _user_device_pick()
        and not str(os.environ.get("GGML_BACKEND_PATH", "")).strip()
        and bool(LlamaCppBackend._is_vulkan_backend(binary))
        and not (
            (os.environ.get("HIP_VISIBLE_DEVICES") or "").strip()
            and not LlamaCppBackend._torch_is_rocm(torch)
        )
    )


def _prefetch_vulkan_rows(binary: str, cancel: Optional[threading.Event] = None) -> None:
    """Probe (and read the card lookup's inventory) before the render slot, waited for cancellably: the probe can take
    its 15 s timeout, which a blocking unload behind this warm start must not sit out."""
    try:
        import torch
        if not _vulkan_pin_possible(binary, torch):
            return
    except Exception:  # noqa: BLE001
        return

    def _work() -> None:
        try:
            # An empty answer (cached or fresh) is retried once, as the launch it stands in for did.
            rows = _vulkan_rows(binary, retry_empty = True, throttle_retry = True) or _vulkan_rows(
                binary, retry_empty = True, throttle_retry = True
            )
            # One row on one torch card launches unpinned, so it needs no card lookup.
            if rows and (len(rows) > 1 or torch.cuda.device_count() > 1):
                ordinal = _resident_cuda_ordinal()
                if ordinal is not None and ordinal >= 0:
                    _vulkan_card_name(ordinal, torch)
        except Exception:  # noqa: BLE001
            pass

    # A maintenance pass that cancelled this start before the probe began must not find one running.
    if cancel is not None and cancel.is_set():
        raise UpsamplerCancelledError("Cancelled.")
    worker = threading.Thread(target = _work, name = "flux3-pup-vulkan-probe", daemon = True)
    # Registered once started: joining a thread that has not started raises.
    worker.start()
    _vulkan_probe_workers[:] = [w for w in _vulkan_probe_workers if w.is_alive()] + [worker]
    while worker.is_alive():
        worker.join(0.1)
        if cancel is not None and cancel.is_set():
            raise UpsamplerCancelledError("Cancelled.")


_vulkan_probe_workers: list = []


def _wait_for_vulkan_probe(timeout: float = 20.0) -> None:
    """Join prefetches cancelled warm starts abandoned: their probes can still hold the llama.cpp tree's libraries."""
    deadline = time.monotonic() + timeout
    for worker in list(_vulkan_probe_workers):
        worker.join(max(0.0, deadline - time.monotonic()))


def _vulkan_rows_came_back_empty(binary: str) -> bool:
    """A pinnable multi-GPU host whose cached Vulkan probe listed nothing: a warm start would stay unpinned for good.
    One torch card warm-starts unpinned as before."""
    try:
        import torch
        return (
            torch.cuda.device_count() > 1
            and _vulkan_pin_possible(binary, torch)
            and not _vulkan_rows(binary)
        )
    except Exception:  # noqa: BLE001
        return False


def _vulkan_card_name(ordinal: int, torch) -> tuple:
    """``physical_card_name`` on ROCm. A CUDA torch reads no HIP/ROCR mask and its cards are not amd-smi's, so that
    lookup would place it by an AMD card's position: name it in torch order, unplaced under a CUDA mask (Vulkan still
    lists the cards it hides)."""
    from core.inference.llama_cpp import LlamaCppBackend
    from core.inference.sd_cpp_backend import physical_card_name

    if LlamaCppBackend._torch_is_rocm(torch):
        return physical_card_name(ordinal)
    if not torch.cuda.is_available() or not 0 <= ordinal < torch.cuda.device_count():
        return None, None
    names = [
        (torch.cuda.get_device_name(i) or "").strip() for i in range(torch.cuda.device_count())
    ]
    name = names[ordinal]
    if not name:
        return None, None
    if (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip():
        return name, None
    # Between identical cards only PCI order matches Vulkan's; FASTEST_FIRST (CUDA's default) may not.
    if names.count(name) > 1 and os.environ.get("CUDA_DEVICE_ORDER") != "PCI_BUS_ID":
        return name, None
    return name, names[:ordinal].count(name)


# Vendor by positive evidence: older NVIDIA drivers name Vulkan devices "Tesla T4" or "GeForce GTX 1080 Ti".
_VULKAN_VENDOR_RES = (
    ("nvidia", re.compile(r"\b(?:nvidia|geforce|quadro|tesla|titan|rtx)\b")),
    ("amd", re.compile(r"\b(?:amd|radeon|radv|instinct|firepro)\b")),
    ("intel", re.compile(r"\b(?:intel|arc|iris|uhd)\b")),
    ("software", re.compile(r"\b(?:llvmpipe|lavapipe|swiftshader)\b")),
)


def _vulkan_row_vendor(row_name: str) -> Optional[str]:
    """``nvidia`` / ``amd`` / ``intel`` / ``software`` when the Vulkan row names one, else None (unknown)."""
    name = str(row_name or "").lower()
    return next((vendor for vendor, pattern in _VULKAN_VENDOR_RES if pattern.search(name)), None)


def _vulkan_row_is_another_vendor(row_name: str, torch) -> bool:
    """A CUDA torch's card is NVIDIA's and a ROCm torch's AMD's: a Vulkan row is another card only when it names
    another vendor. A row naming no vendor at all is not evidence either way."""
    from core.inference.llama_cpp import LlamaCppBackend

    vendor = _vulkan_row_vendor(row_name)
    expected = "amd" if LlamaCppBackend._torch_is_rocm(torch) else "nvidia"
    return vendor is not None and vendor != expected


def _vulkan_device_args(
    binary: str,
    ordinal: Optional[int],
    retry_empty: bool = False,
) -> Optional[list]:
    """``--device Vulkan<i>`` for the image model's card (Vulkan ignores visibility masks); [] unpinned, None unknown.
    One torch card and one Vulkan device launch as before Vulkan pinning; non-Vulkan builds never probe."""
    if ordinal is None or ordinal < 0:
        return []
    try:
        import torch

        count = torch.cuda.device_count()
        if count < 1 or _user_device_pick():
            return []
        if not _vulkan_pin_possible(binary, torch):
            # Backends from GGML_BACKEND_PATH are not the ones the probe lists; a HIP mask on a CUDA torch steers nothing.
            from core.inference.llama_cpp import LlamaCppBackend
            if count == 1 or not (
                str(os.environ.get("GGML_BACKEND_PATH", "")).strip()
                or LlamaCppBackend._is_vulkan_backend(binary)
            ):
                return []
            return None
        # Vulkan also lists an iGPU and cards a mask hides (ggml's default takes discrete AND integrated GPUs).
        rows = _vulkan_rows(binary, retry_empty = retry_empty)
        if not rows:
            return None if count > 1 else []
        if len(rows) == 1 and count == 1:
            # Nothing to choose, so no name match (a renamed card never declines a single-GPU host),
            # unless that one device is plainly another vendor's: then it is not the image model's card.
            return None if _vulkan_row_is_another_vendor(rows[0]["name"], torch) else []
    except Exception:  # noqa: BLE001
        return []
    try:
        from core.inference.sd_cpp_backend import (
            _normalized_card_name,
            _without_driver_tag,
            sd_cpp_device_named,
        )

        # By name, as the image engine pins its Vulkan build: no torch ordinal maps onto Vulkan's index.
        listing = "".join(f"Vulkan{row['index']}\t{row['name']}\n" for row in rows)
        name, position = _vulkan_card_name(ordinal, torch)
        device = sd_cpp_device_named(binary, name, position = position, listing = listing)
        if not device:
            return None
        # Exact model match: containment takes an `RX 7600 XT` for an `RX 7600`.
        matched = next((row["name"] for row in rows if f"Vulkan{row['index']}" == device), "")
        exact = bool(name) and _normalized_card_name(
            _without_driver_tag(matched)
        ) == _normalized_card_name(name)
        if not exact:
            return None
        return [] if len(rows) == 1 else ["--device", device]
    except Exception:  # noqa: BLE001
        return None


def _accelerator_memory_mib(
    ordinal: Optional[int] = None,
) -> tuple[Optional[int], Optional[int], str]:
    """(free, total, device) of the accelerator llama.cpp would offload to (CUDA card ``ordinal`` when given), through
    the diffusion backend's helper. ``device`` is cuda / mps / cpu; free is None when unknown."""
    try:
        import torch
        from types import SimpleNamespace

        from core.inference.diffusion_memory import snapshot_device_memory

        if torch.cuda.is_available():
            device = "cuda"
        elif (
            getattr(getattr(torch, "backends", None), "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            device = "mps"
        else:
            return None, None, "cpu"
        from core.inference.diffusion_device import diffusion_device_scope

        with diffusion_device_scope(ordinal if device == "cuda" else None):
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
        self._penalize_prompt_supported = False
        self._loading = False
        self._starting_process: Optional[subprocess.Popen] = None
        self._load_cancel = threading.Event()
        self._update_in_progress = False
        self._idle_timer: Optional[threading.Timer] = None
        self._generation = 0
        self._active_requests = 0
        # A blocking unload bumps the epoch and holds ``_stopping``: requests queued before it never start a server.
        self._stop_epoch = 0
        self._stopping = 0
        self._gpu_ordinal: Optional[int] = None
        # A warm start owns the load in progress: a teardown, an unenhanced render or a Stop cancels it.
        self._warming = False

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
            self._penalize_prompt_supported = False

    def unload(self, wait: bool = True) -> None:
        """Stop the server. ``wait=False`` leaves a busy one alone."""
        if not wait and (self._loading or self._active_requests):
            return
        if wait:
            with self._lock:
                self._stop_epoch += 1
                self._stopping += 1
        try:
            self.cancel_pending_load()
            if wait:
                deadline = time.monotonic() + 30.0
                while self._active_requests and time.monotonic() < deadline:
                    time.sleep(0.1)
            with self._lock:
                if not wait and self._active_requests:
                    return
                self._release_locked()
        finally:
            if wait:
                with self._lock:
                    self._stopping -= 1

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
            _wait_for_vulkan_probe()
            with self._request_lock:
                # Again once no warm start can still be about to start one.
                _wait_for_vulkan_probe()
                with self._lock:
                    was_active = self.loaded
                    self._release_locked()
                    yield was_active
        finally:
            self._update_in_progress = False

    def _launch(self, binary: str, model: str, mmproj: Optional[str]) -> None:
        """Start llama-server for (model, mmproj). Called under ``_request_lock``."""
        # Published before the training probe: training that starts after it finds a load to cancel.
        self._load_cancel.clear()
        with self._lock:
            self._loading = True
        slot = None
        try:
            # Offloaded layers size themselves to the free memory read at spawn: a render denoising meanwhile (another
            # request's) has not reached its peak yet, so the start waits for the generation slot, as a warm start
            # holds it. Called ahead of the request's own render, never inside one.
            slot = self._wait_for_render_slot()
            self._spawn_server(binary, model, mmproj)
        finally:
            if slot is not None:
                slot.release()
            with self._lock:
                self._loading = False
                self._starting_process = None

    def _wait_for_render_slot(self) -> Any:
        """The active image engine's generation lock, acquired, or None where the start would not offload (training,
        nobody on the GPU, a CPU-only host) or the engine cannot be read. Waited for cancellably: a Stop, training
        start or handover sets ``_load_cancel``."""
        from core.inference.stt_sidecar import _training_active

        if _training_active() or _gpu_unowned():
            return None
        if _accelerator_memory_mib(_resident_cuda_ordinal())[2] == "cpu":
            return None
        try:
            from core.inference.diffusion_engine_router import get_active_diffusion_engine
            engine = get_active_diffusion_engine()
            lock = getattr(engine, "_generate_lock", None)
        except Exception:  # noqa: BLE001 -- unreadable: start as before rather than fail the enhancement
            return None
        if lock is None:
            return None
        while not lock.acquire(timeout = 0.2):
            if self._load_cancel.is_set():
                raise UpsamplerCancelledError("The prompt upsampler load was cancelled.")
        if self._load_cancel.is_set():
            lock.release()
            raise UpsamplerCancelledError("The prompt upsampler load was cancelled.")
        if _engine_loading(engine) or not _image_model_owns_gpu():
            # The model this caption is for is leaving: the run goes as typed (or retries with the new model).
            lock.release()
            raise UpsamplerUnavailableError("The image model is being replaced.")
        return lock

    def _spawn_server(self, binary: str, model: str, mmproj: Optional[str]) -> None:
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
        ordinal = _resident_cuda_ordinal()
        free_mib, _total, device = _accelerator_memory_mib(ordinal)
        training = _training_active()
        # Nobody holds the GPU (a CPU image load, no model): offloaded layers would sit outside every eviction.
        cpu_only = training or _gpu_unowned()
        layers = plan_gpu_layers(free_mib, model_bytes, mmproj_bytes, device, cpu_only)
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
        if layers:
            # A warm start probed just before its slot: re-probing an empty answer here would run it twice.
            cmd += _vulkan_device_args(binary, ordinal, retry_empty = not self._warming) or []
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
        if self._load_cancel.is_set():
            raise UpsamplerCancelledError("The prompt upsampler load was cancelled.")
        # Undrained pipes would block llama-server once its log fills them, so its output goes nowhere unless a log
        # file is asked for.
        log_path = (os.environ.get(ENV_SERVER_LOG) or "").strip()
        log_file = open(log_path, "ab") if log_path else None
        try:
            env = {**_llama_server_child_env(binary), **key_env, **_child_device_mask(ordinal)}
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
            penalize_prompt_supported = _server_has_penalize_prompt(port, api_key)
            with self._lock:
                self._process = process
                self._port = port
                self._api_key = api_key
                self._penalize_prompt_supported = penalize_prompt_supported
                self._key = (binary, model, mmproj)
                self._gpu_layers = layers
                self._gpu_ordinal = ordinal
                self._generation += 1
        finally:
            if log_file is not None:
                with suppress(OSError):
                    log_file.close()

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
        epoch = self._stop_epoch
        with self._request_slot(cancel_event):
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
                if self._stopping or epoch != self._stop_epoch:
                    # Queued behind a request a training start or GPU handover drained: starting now would sit beside
                    # the new owner.
                    raise UpsamplerUnavailableError(
                        "The prompt upsampler was stopped for another GPU user."
                    )
                # Claimed under the same lock: an idle callback already waiting on it then sees the request.
                self._active_requests += 1
            try:
                if not self.loaded:
                    if not _image_model_owns_gpu():
                        # Chat or video took the GPU after this request read the FLUX.3 status: starting now would
                        # sit beside the new owner for the keep-alive.
                        raise UpsamplerUnavailableError("The image model no longer holds the GPU.")
                    with _forward_cancel(cancel_event, self.cancel_pending_load):
                        self._launch(binary, model, mmproj)
                with self._lock:
                    port, api_key = self._port, self._api_key
                    generated_only = self._penalize_prompt_supported
                if generated_only:
                    payload = {**payload, **GENERATED_ONLY_PENALTIES}
                return _post_chat(port, payload, api_key = api_key, cancel_event = cancel_event)
            finally:
                with self._lock:
                    self._active_requests -= 1
                    self._schedule_idle_locked()

    @contextmanager
    def _request_slot(self, cancel_event: Optional[threading.Event]) -> Iterator[None]:
        """The request lock, waited for cancellably: a Stop on a request queued behind a warm start cancels that start
        instead of waiting it out."""
        while not self._request_lock.acquire(timeout = 0.2):
            if cancel_event is not None and cancel_event.is_set():
                if self._warming:
                    self.cancel_pending_load()
                raise UpsamplerCancelledError("Cancelled.")
        try:
            yield
        finally:
            self._request_lock.release()

    def cancel_warm_start(self) -> bool:
        """Cancel a warm start still launching, without waiting. True when one was."""
        return self._warming and self.cancel_pending_load()

    def release_if_tight(self) -> bool:
        """Unload now when the server holds VRAM and the accelerator is short, so the DiT and text encoder do not
        denoise next to it. True when it unloaded."""
        if not self.holds_vram():
            return False
        free_mib, total_mib, _device = _accelerator_memory_mib(self._gpu_ordinal)
        if not vram_is_tight(free_mib, total_mib):
            return False
        logger.info(
            "flux3.prompt_upsampler: unloading after the call (free %s of %s MiB)",
            free_mib,
            total_mib,
        )
        self.unload(wait = False)
        return True

    @property
    def stop_epoch(self) -> int:
        """Bumped by every blocking unload (a GPU handover, a training start). A warm start scheduled under an older
        epoch never starts."""
        return self._stop_epoch

    def serves(self, binary: str, model: str, mmproj: Optional[str]) -> bool:
        """Up for these files (a vision server also serves a text-only ask)."""
        with self._lock:
            current = self._key if self.loaded else None
        return (
            current is not None
            and current[:2] == (binary, model)
            and (mmproj is None or current[2] == mmproj)
        )

    def warm(
        self,
        *,
        binary: str,
        model: str,
        mmproj: Optional[str],
        expected_epoch: Optional[int] = None,
    ) -> str:
        """Start the server ahead of the first enhanced run, so that run skips the ~20 s start. Never waits on a
        request: one in flight already starts or holds the server. Declines where a call would not keep it resident
        (training, another GPU owner, unknown or tight memory once it is up) and after any blocking unload since
        ``expected_epoch``. Returns why it did or did not start."""
        if self._update_in_progress:
            return "updating"
        epoch = self._stop_epoch if expected_epoch is None else expected_epoch
        if not self._request_lock.acquire(blocking = False):
            return "busy"
        try:
            with self._lock:
                current = self._key if self.loaded else None
                if (
                    current is not None
                    and current[:2] == (binary, model)
                    and (mmproj is None or current[2] == mmproj)
                ):
                    # Already up for these files: the toggle counts as use, so the keep-alive restarts.
                    self._schedule_idle_locked()
                    return "loaded"
                if self._stopping or epoch != self._stop_epoch:
                    return "stopped"
                self._cancel_idle_locked()
                # Counted as a request while it starts, as generate does: a non-blocking unload leaves it alone.
                self._active_requests += 1
            try:
                return self._warm_start(binary, model, mmproj, epoch, replace = current is not None)
            finally:
                with self._lock:
                    self._active_requests -= 1
                    self._schedule_idle_locked()
        except Exception as exc:  # noqa: BLE001 -- a warm start is an optimisation; the run starts it itself
            logger.warning("flux3.prompt_upsampler: warm start failed (%s)", exc)
            return "failed"
        finally:
            self._request_lock.release()

    def _warm_start(
        self, binary: str, model: str, mmproj: Optional[str], epoch: int, *, replace: bool
    ) -> str:
        from core.inference.stt_sidecar import _training_active

        # Published before the training and owner probes, as _launch does: a training start that passes them finds a
        # load to cancel (free_prompt_upsampler_for_training reads is_loading).
        self._load_cancel.clear()
        with self._lock:
            self._loading = True
            self._warming = True
        slot_held = None
        try:
            # The image model must hold the GPU: with nobody on it the server would start CPU-only and stay outside
            # every eviction.
            if _gpu_unowned() or not _image_model_owns_gpu() or _training_active():
                return "gpu"
            # A user-picked device is not the card the memory probe reads; its first enhanced run starts it.
            if _user_device_pick(binary) and _cuda_device_count() > 1:
                return "placement"
            # The Vulkan probe can take its full timeout, so it runs before the slot that renders queue behind.
            _prefetch_vulkan_rows(binary, self._load_cancel)
            # The engine's generation slot, held until the server is up: a render or model transition that arrives
            # meanwhile queues behind the start instead of overlapping it (a load cancels it first through
            # stop_for_handover). Not taken while one runs or a load is in flight.
            slot, why = _take_render_slot()
            if slot is None:
                return why
            slot_held = slot
            # An empty (maybe transient) probe would leave the server unpinned for good; a real launch re-probes.
            if _vulkan_rows_came_back_empty(binary):
                return "placement"
            ordinal = _resident_cuda_ordinal()
            # Unpinnable on a multi-GPU host (the native engine records its card elsewhere): the server would split
            # over every visible card and the probe below would read the wrong one. Its first enhanced run starts it.
            if (ordinal is None and _cuda_device_count() > 1) or _vulkan_device_args(
                binary, ordinal
            ) is None:
                return "placement"
            free_mib, total_mib, device = _accelerator_memory_mib(ordinal)
            if free_mib is None or device not in ("cuda", "mps"):
                return "memory"
            need_mib = _footprint_mib(model, mmproj)
            # A replaced server's memory comes back before the new one loads: free memory already excludes it.
            reclaim_mib = 0.0
            if replace and self._gpu_layers and self._key is not None:
                reclaim_mib = _footprint_mib(self._key[1], self._key[2], self._gpu_layers)
            # The same floor release_if_tight applies after a call: a server that floor would unload is not started.
            if vram_is_tight(int(free_mib + reclaim_mib - need_mib), total_mib):
                return "tight"
            with self._lock:
                if self._load_cancel.is_set() or self._stopping or epoch != self._stop_epoch:
                    return "stopped"
                if replace:
                    # Up for other files (text-only, and an edit needs the projector): replaced only once starting.
                    self._release_locked()
            self._spawn_server(binary, model, mmproj)
            # Revalidate once up: a load published or a handover made while the server came up means it now serves a
            # model that is leaving.
            with self._lock:
                stale = self._load_cancel.is_set() or self._stopping or epoch != self._stop_epoch
            if stale or _active_engine_loading():
                with self._lock:
                    self._release_locked()
                return "stale"
            return "started"
        except UpsamplerCancelledError:
            return "cancelled"
        finally:
            if slot_held is not None:
                slot_held.release()
            with self._lock:
                self._loading = False
                self._warming = False
                self._starting_process = None


def _footprint_mib(
    model: str,
    mmproj: Optional[str],
    gpu_layers: Optional[int] = None,
) -> float:
    """What a server holds on the accelerator: its share of the GGUF (all of it by default, ``gpu_layers`` of
    _NUM_LAYERS for a partial offload), the projector (offloaded with any GPU layer) and the KV / compute overhead."""

    def _size(path: Optional[str]) -> int:
        if not path:
            return 0
        with suppress(OSError):
            return os.path.getsize(path)
        return 0

    share = 1.0 if gpu_layers is None else min(max(gpu_layers, 0), _NUM_LAYERS) / _NUM_LAYERS
    return (_size(model) * share + _size(mmproj)) / (1024 * 1024) + _GPU_OVERHEAD_MIB


def _cuda_device_count() -> int:
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # noqa: BLE001
        return 0


def cancel_warm_start() -> bool:
    """Cancel a warm start still launching (it holds the image engine's generation slot until its server is up), for
    a teardown, a model switch or an unenhanced render that would otherwise wait it out. Never creates the sidecar,
    never blocks."""
    sidecar = _sidecar
    return sidecar is not None and sidecar.cancel_warm_start()


def _take_render_slot() -> tuple[Any, str]:
    """(the active image engine's generation lock, acquired) or (None, why not): ``generating`` while a render or model
    transition holds it, ``loading`` while a load is in flight (the old model still reads as loaded), ``engine``
    when the engine cannot be read. Never blocks."""
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        engine = get_active_diffusion_engine()
        lock = getattr(engine, "_generate_lock", None)
    except Exception:  # noqa: BLE001 -- fail closed
        return None, "engine"
    if lock is None:
        return None, "engine"
    if _engine_loading(engine):
        return None, "loading"
    if not lock.acquire(blocking = False):
        return None, "generating"
    # A load that began while this waited for nothing: re-read under the slot.
    if _engine_loading(engine):
        lock.release()
        return None, "loading"
    return lock, ""


def _active_engine_loading() -> bool:
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        return _engine_loading(get_active_diffusion_engine())
    except Exception:  # noqa: BLE001 -- unreadable: treat as a load in flight
        return True


def _engine_loading(engine: Any) -> bool:
    loading = getattr(engine, "_loading", None)
    return loading is not None and not getattr(loading, "error", None)


def _gpu_unowned() -> bool:
    try:
        from core.inference import gpu_arbiter
        return gpu_arbiter.current_owner() is None
    except Exception:  # noqa: BLE001
        return False


def _image_model_owns_gpu() -> bool:
    """The GPU arbiter's owner is the image model (or nobody yet, as on a fresh start). Fail-open."""
    try:
        from core.inference import gpu_arbiter
        return gpu_arbiter.current_owner() in (None, gpu_arbiter.DIFFUSION)
    except Exception:  # noqa: BLE001
        return True


@contextmanager
def _forward_cancel(cancel_event: Optional[threading.Event], cancel: Any) -> Iterator[None]:
    """Stop pressed while llama-server starts cancels the start, not just the request after it."""
    if cancel_event is None:
        yield
        return
    done = threading.Event()

    def _watch() -> None:
        # Retried until it lands: a Stop just before the start flips `_loading` would otherwise be dropped.
        while not done.wait(0.2):
            if cancel_event.is_set() and cancel():
                return

    watcher = threading.Thread(target = _watch, name = "flux3-upsampler-cancel", daemon = True)
    watcher.start()
    try:
        yield
    finally:
        done.set()
        watcher.join(timeout = 1)


def _wait_for_server(process, port: int, cancel_event: threading.Event) -> bool:
    from core.inference.stt_mtmd_sidecar import MtmdSttSidecar
    return MtmdSttSidecar._wait_for_server(process, port, cancel_event)


def _server_has_penalize_prompt(port: int, api_key: Optional[str] = None) -> bool:
    """Whether this llama-server takes the `penalize_prompt` request field: builds that have it echo it among the
    default generation settings in /props. Any failure reads as an older server."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout = 10)
    try:
        connection.request(
            "GET", "/props", headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        )
        response = connection.getresponse()
        if response.status != 200:
            return False
        settings = (json.loads(response.read()) or {}).get("default_generation_settings") or {}
        return "penalize_prompt" in settings or "penalize_prompt" in (settings.get("params") or {})
    except (OSError, ValueError, AttributeError, http.client.HTTPException):
        return False
    finally:
        connection.close()


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


def stop_for_handover() -> None:
    """The image model the server serves is leaving the GPU: stop every run still enhancing and the server, blocking,
    so nothing of it outlives the DIFFUSION claim."""
    cancel_active()
    unload_sidecar(wait = True)


def _resident_identity() -> Optional[tuple]:
    """(repo_id, family) of the resident image model, else None."""
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        status = get_active_diffusion_engine().status()
    except Exception:  # noqa: BLE001 -- no engine reads as nothing loaded
        return None
    return (status.get("repo_id"), status.get("family")) if status.get("loaded") else None


def warm_for_family(
    family_name: Optional[str],
    *,
    expected_epoch: Optional[int] = None,
    resident: Optional[tuple] = None,
) -> str:
    """Start the server for the resident ``family_name`` when Enhance prompt is on, in the background. The vision
    projector comes up too when downloaded, so a later edit does not relaunch it. ``expected_epoch`` (the sidecar's
    stop epoch read before the resident was) and ``resident`` (its (repo_id, family)) bind the start to that model: a
    handover or model swap before the worker runs cancels it. Returns ``scheduled`` or why not."""
    if not supports_family(family_name):
        return "unsupported"
    binary = find_llama_server_binary()
    model = model_path()
    if binary is None:
        return "unavailable"
    if model is None:
        return "not_downloaded"
    sidecar = get_sidecar()
    mmproj = mmproj_path()
    if sidecar.is_loading() or sidecar.serves(binary, model, mmproj):
        # Starting, or up for these files (the call refreshes its keep-alive): nothing to wait for on a thread. A
        # server for other files is replaced on the thread below, so the route never waits for a start.
        return sidecar.warm(
            binary = binary, model = model, mmproj = mmproj, expected_epoch = expected_epoch
        )

    def _run() -> None:
        if resident is not None and _resident_identity() != resident:
            outcome = "stale"
        else:
            outcome = sidecar.warm(
                binary = binary, model = model, mmproj = mmproj, expected_epoch = expected_epoch
            )
        logger.info("flux3.prompt_upsampler: warm start %s", outcome)

    threading.Thread(target = _run, name = "flux3-upsampler-warm", daemon = True).start()
    return "scheduled"


def unload_sidecar(wait: bool = False) -> None:
    """Stop the server if one was ever started (the FLUX.3 model it serves was unloaded). Never creates one.
    ``wait`` also stops a busy one, for a GPU handover."""
    sidecar = _sidecar
    if sidecar is not None:
        sidecar.unload(wait = wait)


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


# Runs in flight, oldest (the one holding llama-server) first, for generate-progress and the Stop button.
_progress_lock = threading.Lock()
_runs: list = []


def progress() -> Optional[dict]:
    """``{"done", "total", "message"}`` of the run being enhanced, else None."""
    with _progress_lock:
        return dict(_runs[0][0]) if _runs else None


def cancel_active() -> bool:
    """Stop every run still enhancing (a queued one waits on the same server)."""
    with _progress_lock:
        events = [event for _state, event in _runs if not event.is_set()]
    for event in events:
        event.set()
    return bool(events)


@contextmanager
def _tracking(total: int, cancel_event: threading.Event) -> Iterator[dict]:
    run = ({"done": 0, "total": total, "message": "Enhancing prompt..."}, cancel_event)
    with _progress_lock:
        _runs.append(run)
    try:
        yield run[0]
    finally:
        with _progress_lock:
            _runs.remove(run)


def _advance(state: dict) -> None:
    with _progress_lock:
        state["done"] += 1


def upsample(
    prompt: str,
    *,
    variant: str = "distilled",
    reference_images: Optional[list] = None,
    cancel_event: Optional[threading.Event] = None,
    sidecar: Optional[PromptUpsamplerSidecar] = None,
) -> str:
    """The caption for ``prompt`` (cached). Raises UpsamplerUnavailableError when nothing can run, a cancel as
    UpsamplerCancelledError, and anything else the server does. An empty caption returns ``prompt``, as does a
    finished layout caption, which the image model takes as it is."""
    if is_finished_layout_caption(prompt):
        return prompt
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
    # A box table in the prompt switches to the layout prompts with its rows locked (BFL's bbox.plan).
    found = detect_box_table(prompt)
    request, locked = found if found is not None else (prompt, None)
    payload = build_chat_payload(request, variant, references, edit = edit, locked = locked)
    key = (prompt, variant, edit, references_hash(references), model)
    cached = _cache_get(key)
    if cached is not None:
        # A Stop while the references decoded never reaches the sidecar, which is where it is normally enforced.
        if cancel_event is not None and cancel_event.is_set():
            raise UpsamplerCancelledError("Cancelled.")
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
    if locked:
        text = reconcile_layout_caption(text, locked)
    _cache_put(key, text)
    return text


@dataclass
class EnhanceOutcome:
    """Per-prompt captions for one generate call. ``status`` is None for families without an upsampler."""

    status: Optional[str]
    captions: dict = field(default_factory = dict)
    statuses: dict = field(default_factory = dict)
    # The resident model the captions were written for, set by the generate route.
    load: Any = None

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
            "llama-server not found"
            if find_llama_server_binary() is None
            else "GGUF not downloaded",
        )
        return EnhanceOutcome(status = STATUS_UNAVAILABLE)
    variant = variant_for_family(family_name)
    distinct = list(dict.fromkeys(p for p in prompts if isinstance(p, str)))
    cancel_event = cancel_event or threading.Event()
    outcome = EnhanceOutcome(status = STATUS_APPLIED)
    sidecar = get_sidecar()
    # On every exit, a Stop included: a retry with enhancement off must not denoise beside a tight server.
    try:
        with _tracking(len(distinct), cancel_event) as tracked:
            for index, prompt in enumerate(distinct):
                if cancel_event.is_set():
                    raise UpsamplerCancelledError("Cancelled.")
                if is_finished_layout_caption(prompt):
                    # Already an upsampler caption with its box table: rewriting it would only drop the boxes.
                    outcome.statuses[prompt] = STATUS_OFF
                    _advance(tracked)
                    continue
                systemic = None
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
                    systemic = STATUS_UNAVAILABLE
                except Exception as exc:  # noqa: BLE001 -- an upsampler failure never fails the image
                    if cancel_event.is_set():
                        raise UpsamplerCancelledError("Cancelled.") from exc
                    logger.warning("flux3.prompt_upsampler: failed (%s); prompt as typed", exc)
                    systemic = STATUS_FAILED
                else:
                    if caption == prompt:
                        outcome.statuses[prompt] = STATUS_FAILED
                    else:
                        outcome.captions[prompt] = caption
                        outcome.statuses[prompt] = STATUS_APPLIED
                if systemic is not None:
                    # The runtime itself failed (a start that timed out, a dead server): the rest of the batch would wait
                    # out the same timeout each, so they run as typed now.
                    for rest in distinct[index:]:
                        outcome.statuses[rest] = systemic
                    break
                _advance(tracked)
    finally:
        with suppress(Exception):
            sidecar.release_if_tight()
    if outcome.statuses and not outcome.captions:
        # Nothing applied: the batch-level status is the first reason.
        outcome.status = next(iter(outcome.statuses.values()))
    return outcome
