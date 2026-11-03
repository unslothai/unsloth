# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detailer (face fix) post-pass, after ADetailer and ComfyUI-Impact-Pack's FaceDetailer: SAM 3 finds each target,
its padded crop is enlarged, inpainted at a moderate denoise and pasted back through a feathered mask (pixels outside
the mask stay bit-identical). Defaults follow FaceDetailer; prompts, [SEP]/[PROMPT] and infotext keys follow ADetailer.
Departures: the mask is SAM 3's dilated instance mask, not the box, and SAM 3 is parked in host RAM before the first
inpaint so it never sits on the GPU beside the image model.
"""

from __future__ import annotations

import math

import base64
import contextlib
import io
import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

# Defaults: FaceDetailer, modules/impact/impact_pack.py:743-769; utils.py:454.
DEFAULT_TARGETS = "face"
DEFAULT_CONFIDENCE = 0.5  # bbox_threshold
DEFAULT_DENOISE = 0.5  # denoise
DEFAULT_FEATHER = 5  # feather (kernel 2 * feather + 1)
FEATHER_SIGMA = 10.0  # tensor_gaussian_blur_mask(mask, kernel_size, sigma = 10.0)
DEFAULT_DILATION = 10  # bbox_dilation
DEFAULT_CROP_FACTOR = 3.0  # bbox_crop_factor
DEFAULT_GUIDE_SIZE = 512  # guide_size
DEFAULT_GUIDE_SIZE_FOR = "bbox"  # guide_size_for: True = "bbox", False = "crop_region"
DEFAULT_MAX_SIZE = 1024  # max_size
DEFAULT_DROP_SIZE = 10  # drop_size
DEFAULT_PER_TARGET = 8
MAX_PER_TARGET = 20
MAX_DETECTIONS = 32
# Matches diffusion.py _snap_to_multiple so inpaint never resizes.
SIZE_MULTIPLE = 16
_SEED_SPACE = 2**53

DETAILER_UNAVAILABLE_INPAINT = (
    "The Detailer needs a model with an inpaint pipeline; this model has none."
)

_MAX_SUFFIX = re.compile(r"^(.+?)\s*:\s*([\d.]+)\s*$")
_SEP = re.compile(r"\s*\[SEP\]\s*")


class DetailerCancelled(RuntimeError):
    """Stop pressed between two inpaints."""


# Running passes' stop events -> owning account (None: no account scope).
_active: dict[threading.Event, Optional[str]] = {}
_active_lock = threading.Lock()


def cancel_active(account: Optional[str] = None) -> bool:
    """Stop running passes (all when ``account`` is None); True when one was stopped."""
    with _active_lock:
        hit = [event for event, owner in _active.items() if account is None or owner == account]
        for event in hit:
            event.set()
        return bool(hit)


@dataclass(frozen = True)
class DetailerSettings:
    targets: str = DEFAULT_TARGETS
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    confidence: float = DEFAULT_CONFIDENCE
    denoise: float = DEFAULT_DENOISE
    crop_factor: float = DEFAULT_CROP_FACTOR
    guide_size: int = DEFAULT_GUIDE_SIZE
    guide_size_for: str = DEFAULT_GUIDE_SIZE_FOR
    max_size: int = DEFAULT_MAX_SIZE
    dilation: int = DEFAULT_DILATION
    feather: int = DEFAULT_FEATHER
    drop_size: int = DEFAULT_DROP_SIZE
    steps: Optional[int] = None
    guidance: Optional[float] = None

    @classmethod
    def from_request(cls, spec: Any) -> "DetailerSettings":
        values = {}
        for name in cls.__dataclass_fields__:
            value = getattr(spec, name, None)
            if value is not None:
                values[name] = value
        settings = cls(**values)
        if not target_concepts(settings.targets):
            raise ValueError("Name what the Detailer should fix, for example: face")
        from core.inference import diffusion_segment

        # Validate SAM 3 limits before the render, not after.
        diffusion_segment.parse_prompt(segment_prompt(settings.targets))
        return settings


@dataclass
class Detection:
    concept: str
    score: float
    box: tuple[float, float, float, float]
    # Full-size bool mask, or None with ``plane`` (low-res logits) expanded lazily to bound batch memory.
    mask: Any = None
    plane: Any = None

    def full_mask(self, height: int, width: int) -> Any:
        if self.mask is not None:
            return self.mask
        return _plane_to_mask(self.plane, height, width)


@dataclass
class DetailResult:
    image: Any
    record: dict = field(default_factory = dict)


def target_concepts(targets: Optional[str]) -> list[tuple[str, int]]:
    """``face, hand:2`` -> [("face", DEFAULT_PER_TARGET), ("hand", 2)]."""
    out: list[tuple[str, int]] = []
    text = (targets or "").replace("(", "").replace(")", "")
    for part in (p.strip() for p in text.split(",")):
        if not part:
            continue
        match = _MAX_SUFFIX.match(part)
        label, count = part, DEFAULT_PER_TARGET
        if match:
            try:
                label, count = match.group(1).strip(), max(1, round(float(match.group(2))))
            except (ValueError, OverflowError):
                label, count = part, DEFAULT_PER_TARGET
        if label:
            out.append((label, min(count, MAX_PER_TARGET)))
    return out


def segment_prompt(targets: Optional[str]) -> str:
    return ", ".join(f"{label}:{count}" for label, count in target_concepts(targets))


def _normalize_region(limit: int, start: int, size: float) -> tuple[int, int]:
    """Impact-Pack utils.normalize_region."""
    if start < 0:
        return 0, int(min(limit, size))
    if start + size > limit:
        return int(max(0, limit - size)), int(limit)
    return int(start), int(min(limit, start + size))


def crop_region(
    width: int, height: int, box: tuple[float, float, float, float], crop_factor: float
) -> tuple[int, int, int, int]:
    """Impact-Pack utils.make_crop_region."""
    x1, y1, x2, y2 = box
    bbox_w, bbox_h = x2 - x1, y2 - y1
    crop_w, crop_h = bbox_w * crop_factor, bbox_h * crop_factor
    new_x1 = int(x1 + bbox_w / 2 - crop_w / 2)
    new_y1 = int(y1 + bbox_h / 2 - crop_h / 2)
    cx1, cx2 = _normalize_region(width, new_x1, crop_w)
    cy1, cy2 = _normalize_region(height, new_y1, crop_h)
    return cx1, cy1, cx2, cy2


def upscaled_size(
    bbox_w: float,
    bbox_h: float,
    crop_w: int,
    crop_h: int,
    guide_size: int,
    guide_size_for: str,
    max_size: int,
) -> tuple[float, int, int]:
    """Impact-Pack core.enhance_detail sizing with force_inpaint: never shrinks."""
    short = min(bbox_w, bbox_h) if guide_size_for == "bbox" else min(crop_w, crop_h)
    upscale = guide_size / short if short > 0 else 1.0
    new_w, new_h = int(crop_w * upscale), int(crop_h * upscale)
    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w, new_h = int(crop_w * upscale), int(crop_h * upscale)
    if upscale <= 1.0 or new_w == 0 or new_h == 0:
        return 1.0, crop_w, crop_h
    return upscale, new_w, new_h


def snap_size(
    width: int,
    height: int,
    multiple: int = SIZE_MULTIPLE,
) -> tuple[int, int]:
    return max(multiple, width // multiple * multiple), max(multiple, height // multiple * multiple)


def dilate(mask: Any, pixels: int) -> Any:
    """cv2.dilate / cv2.erode (Impact-Pack utils.dilate_masks), OpenCV border semantics."""
    import numpy as np

    mask = np.asarray(mask, dtype = bool)
    size = abs(int(pixels))
    if size <= 1:
        return mask.copy()
    grow = pixels > 0
    anchor = size // 2
    out = mask.copy()
    for axis in (0, 1):
        src, acc = out, out.copy()
        length = src.shape[axis]
        for k in range(size):
            shift = k - anchor
            if shift == 0:
                continue
            moved = np.full_like(src, not grow)
            lo, hi = max(0, -shift), min(length, length - shift)
            if lo < hi:
                dst = [slice(None)] * 2
                srcs = [slice(None)] * 2
                dst[axis] = slice(lo, hi)
                srcs[axis] = slice(lo + shift, hi + shift)
                moved[tuple(dst)] = src[tuple(srcs)]
            acc = acc | moved if grow else acc & moved
        out = acc
    return out


def _gaussian_kernel(size: int, sigma: float) -> Any:
    """torchvision's GaussianBlur kernel (transforms.functional._get_gaussian_kernel1d)."""
    import numpy as np

    half = (size - 1) * 0.5
    x = np.linspace(-half, half, size, dtype = np.float32)
    pdf = np.exp(-0.5 * (x / np.float32(sigma)) ** 2)
    return (pdf / pdf.sum()).astype(np.float32)


def feather_mask(
    mask: Any,
    feather: int,
    sigma: float = FEATHER_SIGMA,
) -> Any:
    """Impact-Pack utils.tensor_gaussian_blur_mask."""
    import numpy as np

    alpha = np.asarray(mask, dtype = np.float32)
    if feather <= 0:
        return alpha
    size = feather * 2 + 1
    shortest = min(alpha.shape)
    if shortest <= size:
        size = int(shortest / 2)
        if size % 2 == 0:
            size += 1
        if size < 3:
            return alpha
    kernel = _gaussian_kernel(size, sigma)
    pad = size // 2
    for axis in (0, 1):
        padding = [(0, 0), (0, 0)]
        padding[axis] = (pad, pad)
        padded = np.pad(alpha, padding, mode = "reflect")
        acc = np.zeros_like(alpha)
        for k in range(size):
            index = [slice(None)] * 2
            index[axis] = slice(k, k + alpha.shape[axis])
            acc += kernel[k] * padded[tuple(index)]
        alpha = acc
    return np.clip(alpha, 0.0, 1.0)


def composite(base: Any, patch: Any, alpha: Any) -> Any:
    """Impact-Pack utils.tensor_paste; alpha 0 keeps ``base`` byte for byte."""
    import numpy as np

    base = np.asarray(base, dtype = np.uint8)
    patch = np.asarray(patch, dtype = np.uint8)
    a = np.asarray(alpha, dtype = np.float32)[..., None]
    mixed = base.astype(np.float32) * (1.0 - a) + patch.astype(np.float32) * a
    mixed = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return np.where(a > 0, mixed, base)


def detection_seed(base_seed: int, index: int) -> int:
    return (int(base_seed) + int(index)) % _SEED_SPACE


def detection_prompt(template: Optional[str], main: str, index: int) -> Optional[str]:
    """ADetailer [SEP]/[PROMPT] rule (last part repeats); None means [SKIP]."""
    parts = _SEP.split(template or "")
    part = parts[min(index, len(parts) - 1)].strip()
    if part == "[SKIP]":
        return None
    if not part:
        return main
    return part.replace("[PROMPT]", main)


def inherits_main(template: Optional[str]) -> bool:
    if not template:
        return True
    return any(not part.strip() or "[PROMPT]" in part for part in _SEP.split(template))


def resolve_inherited(template: str, main: str) -> str:
    return " [SEP] ".join(
        main if not part.strip() else part.strip().replace("[PROMPT]", main)
        for part in _SEP.split(template)
    )


def schedulable_steps(steps: int, denoise: float) -> int:
    """Smallest steps >= ``steps`` with int(steps * denoise) >= 1; inpaint pipelines refuse zero timesteps."""
    n = max(1, int(steps))
    denoise = float(denoise)
    if not (0.0 < denoise < 1.0):  # also NaN
        return n
    # Closed form plus a bounded nudge for float rounding; counting up would loop forever at tiny denoise.
    m = max(n, math.ceil(1.0 / denoise))
    for _ in range(4):
        if m > n and int((m - 1) * denoise) >= 1:
            m -= 1
        elif int(m * denoise) < 1:
            m += 1
        else:
            break
    return m


InpaintFn = Callable[[Any, Any, int, str, Optional[str]], Any]


def _big_enough(box: tuple[float, float, float, float], drop_size: int) -> bool:
    x1, y1, x2, y2 = box
    return x2 - x1 > drop_size and y2 - y1 > drop_size


def detail_image(
    image: Any,
    detections: list[Detection],
    settings: DetailerSettings,
    *,
    base_seed: int,
    prompt: str,
    negative_prompt: Optional[str],
    inpaint: InpaintFn,
    stop: Optional[Callable[[], bool]] = None,
) -> DetailResult:
    import numpy as np
    from PIL import Image

    started = time.perf_counter()
    current = np.array(image.convert("RGB"), dtype = np.uint8)
    height, width = current.shape[:2]
    done: list[dict] = []
    # Undersized boxes take no [SEP] prompt, seed or cap slot (Impact-Pack drops them at detection).
    sized = [d for d in detections if _big_enough(d.box, settings.drop_size)]
    skipped = len(detections) - len(sized)
    for index, det in enumerate(sized[:MAX_DETECTIONS]):
        if stop is not None and stop():
            raise DetailerCancelled("cancelled")
        x1, y1, x2, y2 = det.box
        det_prompt = detection_prompt(settings.prompt, prompt, index)
        if det_prompt is None:
            skipped += 1
            continue
        det_negative = (
            detection_prompt(settings.negative_prompt, negative_prompt or "", index)
            if settings.negative_prompt
            else negative_prompt
        )
        cx1, cy1, cx2, cy2 = crop_region(width, height, det.box, settings.crop_factor)
        crop_w, crop_h = cx2 - cx1, cy2 - cy1
        if crop_w <= 0 or crop_h <= 0:
            skipped += 1
            continue
        mask = dilate(
            np.asarray(det.full_mask(height, width), dtype = bool)[cy1:cy2, cx1:cx2],
            settings.dilation,
        )
        if not mask.any():
            skipped += 1
            continue
        _, new_w, new_h = upscaled_size(
            x2 - x1,
            y2 - y1,
            crop_w,
            crop_h,
            settings.guide_size,
            settings.guide_size_for,
            settings.max_size,
        )
        new_w, new_h = snap_size(new_w, new_h)
        crop = Image.fromarray(current[cy1:cy2, cx1:cx2])
        crop_up = crop.resize((new_w, new_h), Image.LANCZOS)
        mask_up = Image.fromarray(mask.astype(np.uint8) * 255, mode = "L").resize(
            (new_w, new_h), Image.NEAREST
        )
        seed = detection_seed(base_seed, index)
        rendered = inpaint(crop_up, mask_up, seed, det_prompt, det_negative)
        patch = np.array(rendered.convert("RGB").resize((crop_w, crop_h), Image.LANCZOS))
        alpha = feather_mask(mask, settings.feather)
        current[cy1:cy2, cx1:cx2] = composite(current[cy1:cy2, cx1:cx2], patch, alpha)
        entry = {
            "target": det.concept,
            "score": round(float(det.score), 4),
            "box": [int(round(v)) for v in det.box],
            "crop": [cx1, cy1, cx2, cy2],
            "size": [new_w, new_h],
            "seed": seed,
        }
        if det_prompt != prompt:
            entry["prompt"] = det_prompt
        done.append(entry)
    record = {
        **settings_record(settings),
        "detections": done,
        "found": len(detections),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
    }
    if skipped:
        record["skipped"] = skipped
    if len(sized) > MAX_DETECTIONS:
        record["dropped"] = len(sized) - MAX_DETECTIONS
    return DetailResult(Image.fromarray(current), record)


def settings_record(settings: DetailerSettings) -> dict:
    out = {
        "targets": settings.targets,
        "confidence": settings.confidence,
        "denoise": settings.denoise,
        "crop_factor": settings.crop_factor,
        "guide_size": settings.guide_size,
        "guide_size_for": settings.guide_size_for,
        "max_size": settings.max_size,
        "dilation": settings.dilation,
        "feather": settings.feather,
        "drop_size": settings.drop_size,
    }
    for name in ("prompt", "negative_prompt", "steps", "guidance"):
        value = getattr(settings, name)
        if value not in (None, ""):
            out[name] = value
    return out


# Infotext keys: adetailer/args.py _all_args.
def _quote(value: Any) -> str:
    """AUTOMATIC1111 infotext_utils.quote."""
    text = str(value)
    if "," not in text and "\n" not in text and ":" not in text:
        return text
    return json.dumps(text, ensure_ascii = False)


def infotext_fields(record: Optional[dict], model: Optional[str] = None) -> str:
    """``, ADetailer model: ...`` for the parameters line, or "" when no detailer ran."""
    if not record or not isinstance(record, dict) or record.get("error"):
        return ""
    pairs: list[tuple[str, Any]] = [
        ("ADetailer model", model or record.get("model") or "sam3"),
        ("ADetailer model classes", record.get("targets")),
    ]
    if record.get("prompt"):
        pairs.append(("ADetailer prompt", record["prompt"]))
    if record.get("negative_prompt"):
        pairs.append(("ADetailer negative prompt", record["negative_prompt"]))
    pairs += [
        ("ADetailer confidence", record.get("confidence")),
        ("ADetailer dilate erode", record.get("dilation")),
        ("ADetailer mask blur", record.get("feather")),
        ("ADetailer denoising strength", record.get("denoise")),
        ("ADetailer inpaint only masked", True),
    ]
    if record.get("steps") is not None:
        pairs += [("ADetailer use separate steps", True), ("ADetailer steps", record["steps"])]
    if record.get("guidance") is not None:
        pairs += [
            ("ADetailer use separate CFG scale", True),
            ("ADetailer CFG scale", record["guidance"]),
        ]
    return "".join(f", {key}: {_quote(value)}" for key, value in pairs if value is not None)


def check_available(status: dict) -> None:
    """Raise ValueError (400) without an inpaint pipeline (covers sd.cpp) or SAM 3."""
    from core.inference import diffusion_segment

    if not status.get("loaded"):
        return  # the render itself answers "no model loaded"
    if "inpaint" not in (status.get("workflows") or []):
        raise ValueError(DETAILER_UNAVAILABLE_INPAINT)
    if not diffusion_segment.transformers_supports_sam3():
        raise ValueError(
            "The Detailer finds faces with SAM 3, which needs transformers 5 (Python 3.10 or newer)."
        )
    if diffusion_segment.model_dir() is None:
        variant = diffusion_segment.active_variant()
        raise ValueError(
            f"The Detailer finds faces with {variant.label}, which is not downloaded. Download it from "
            "Mask from text in the Inpaint tab first."
        )


def _plane_to_mask(plane: Any, height: int, width: int) -> Any:
    import torch
    import torch.nn.functional as F
    with torch.inference_mode():
        up = F.interpolate(
            plane[None, None].float(), size = (height, width), mode = "bilinear", align_corners = False
        )[0, 0]
        return (up > 0).numpy()


def _take_generation_slot(stop: Optional[Callable[[], bool]] = None) -> Any:
    """Acquire the active engine's generation lock (None if it has none); cancellable while waiting."""
    try:
        from core.inference.diffusion_engine_router import get_active_diffusion_engine
        lock = getattr(get_active_diffusion_engine(), "_generate_lock", None)
    except Exception:  # noqa: BLE001 -- no engine: no render to collide with
        return None
    if lock is None or not hasattr(lock, "acquire"):
        return None
    while not lock.acquire(timeout = 0.1):
        if stop is not None and stop():
            raise DetailerCancelled("cancelled")
    return lock


def detect(
    images: list,
    settings: DetailerSettings,
    stop: Optional[Callable[[], bool]] = None,
) -> tuple[list[list[Detection]], dict]:
    """SAM 3 on every image, then parked in host RAM; at most MAX_DETECTIONS sized planes per image."""
    from core.inference import diffusion_segment

    prompt = segment_prompt(settings.targets)
    out: list[list[Detection]] = []
    found: list[int] = []
    undersized: list[int] = []
    info: dict = {}
    started = time.perf_counter()
    # Hold the generation slot so a queued render waits for SAM 3 to leave the GPU.
    slot = _take_generation_slot(stop)
    try:
        segmenter = diffusion_segment.get_segmenter()
        hold = getattr(segmenter, "hold", None)
        with contextlib.ExitStack() as held:
            if slot is not None:
                # Register as slot holder so this account's cancel is not refused as foreign.
                from hub.services.models.account_access import media_generation_slot
                held.enter_context(media_generation_slot("diffusion"))
            if hold is not None:
                held.enter_context(hold())
            for image in images:
                if stop is not None and stop():
                    raise DetailerCancelled("cancelled")
                result = segmenter.segment(
                    image.convert("RGB"),
                    prompt,
                    threshold = settings.confidence,
                    instance_masks = True,
                    plane_filter = lambda box: _big_enough(box, settings.drop_size),
                    max_planes = MAX_DETECTIONS,
                )
                pairs = list(zip(result.instances, result.instance_logits))
                found.append(len(pairs))
                sized = [p for p in pairs if _big_enough(p[0].box, settings.drop_size)]
                undersized.append(len(pairs) - len(sized))
                out.append(
                    [
                        Detection(inst.concept, inst.score, inst.box, plane = plane)
                        for inst, plane in sized[:MAX_DETECTIONS]
                    ]
                )
                info = {"device": result.device, "precision": result.precision}
                del result, pairs, sized
    finally:
        try:
            diffusion_segment.release_segmenter("detailer", wait = True, park = True)
        finally:
            if slot is not None:
                slot.release()
    info["model"] = diffusion_segment.active_variant().id
    info["found"] = found
    info["undersized"] = undersized
    info["detect_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return out, info


# A same-repo reload differing in any of these is another model.
BUILD_KEYS = (
    "repo_id",
    "model_kind",
    "gguf_filename",
    "transformer_quant",
    "text_encoder_quant",
    "baked_loras",
)


def same_build(render: dict, inpaint: dict) -> bool:
    """Keys missing from either result are not compared."""
    for key in BUILD_KEYS:
        if key in render and key in inpaint:
            a, b = render[key], inpaint[key]
            if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
                a, b = list(a), list(b)
            if a != b:
                return False
    return True


def _png_b64(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format = "PNG", compress_level = 1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def detail_result(
    generate: Callable[..., dict],
    result: dict,
    settings: DetailerSettings,
    *,
    prompt_for: Callable[[int], str],
    negative_prompt: Optional[str],
    steps: int,
    guidance: float,
    generate_kwargs: dict,
    account: Optional[str] = None,
) -> None:
    """Detail a finished render in place; failures are recorded per image and the render is kept."""
    images = list(result.get("images") or [])
    if not images:
        return
    renders = list(images)
    stop_event = threading.Event()
    with _active_lock:
        _active[stop_event] = account
    finished = False
    try:
        _detail_images(
            generate,
            result,
            images,
            settings,
            stop_event,
            prompt_for = prompt_for,
            negative_prompt = negative_prompt,
            steps = steps,
            guidance = guidance,
            generate_kwargs = generate_kwargs,
        )
        finished = True
    finally:
        with _active_lock:
            _active.pop(stop_event, None)
            # Under the lock: a Stop cancel_active acknowledged is honoured even after the last inpaint.
            stopped = stop_event.is_set()
    if finished and stopped:
        result["images"] = renders
        result["detailer"] = [
            record
            if record and record.get("error")
            else {**settings_record(settings), "error": "cancelled"}
            for record in (result.get("detailer") or [None] * len(renders))
        ]


def _detail_images(
    generate: Callable[..., dict],
    result: dict,
    images: list,
    settings: DetailerSettings,
    stop_event: threading.Event,
    *,
    prompt_for: Callable[[int], str],
    negative_prompt: Optional[str],
    steps: int,
    guidance: float,
    generate_kwargs: dict,
) -> None:
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    seeds = result.get("seeds") or []
    records: list[Optional[dict]] = [None] * len(images)
    try:
        found, info = detect(images, settings, stop = stop_event.is_set)
    except Exception as exc:  # noqa: BLE001 -- the render stands; the recipe says why it was not detailed
        cancelled = isinstance(exc, DetailerCancelled)
        if not cancelled:
            logger.warning("images.detailer: detection failed: %s", exc)
        error = "cancelled" if cancelled else str(exc)
        result["detailer"] = [{**settings_record(settings), "error": error} for _ in images]
        return
    totals = info.get("found") or []
    tiny = info.get("undersized") or []
    step_count = schedulable_steps(
        settings.steps if settings.steps is not None else steps, settings.denoise
    )
    cfg = settings.guidance if settings.guidance is not None else guidance
    cancelled = stop_event.is_set()
    for index, image in enumerate(images):
        base_seed = seeds[index] if index < len(seeds) else result.get("seed") or 0
        if cancelled:
            records[index] = {**settings_record(settings), "error": "cancelled"}
            continue

        inherited: dict = {}

        def _inpaint(crop, mask, seed, det_prompt, det_negative):
            from core.inference import diffusion_bg_compile

            # Every box is its own size: run eager and uncounted so no crop compiles, captures or feeds the compiler.
            with diffusion_bg_compile.auxiliary_pass():
                out = generate(
                    prompt = det_prompt,
                    negative_prompt = det_negative,
                    width = crop.width,
                    height = crop.height,
                    steps = step_count,
                    guidance = cfg,
                    seed = seed,
                    batch_size = 1,
                    init_image = _png_b64(crop),
                    mask_image = _png_b64(mask),
                    strength = settings.denoise,
                    **generate_kwargs,
                )
            if not same_build(result, out):
                raise ValueError("the model changed")
            if det_negative and out.get("negative_prompt") and not result.get("negative_prompt"):
                # Negative prompt engaged only at the Detailer's CFG: record it.
                if not settings.negative_prompt:
                    inherited["negative_prompt"] = det_negative
                elif negative_prompt and inherits_main(settings.negative_prompt):
                    inherited["negative_prompt"] = resolve_inherited(
                        settings.negative_prompt, negative_prompt
                    )
            return out["images"][0]

        try:
            detailed = detail_image(
                image,
                found[index],
                settings,
                base_seed = int(base_seed),
                prompt = prompt_for(index),
                negative_prompt = negative_prompt,
                inpaint = _inpaint,
                stop = stop_event.is_set,
            )
        except Exception as exc:  # noqa: BLE001
            if isinstance(exc, DetailerCancelled) or str(exc) == DIFFUSION_CANCELLED_MSG:
                cancelled = True
            else:
                logger.warning("images.detailer: image %d failed: %s", index, exc)
            records[index] = {
                **settings_record(settings),
                "error": "cancelled" if cancelled else str(exc),
            }
            continue
        images[index] = detailed.image
        records[index] = {
            **detailed.record,
            **inherited,
            "model": info.get("model"),
            "detect_device": info.get("device"),
        }
        small = tiny[index] if index < len(tiny) else 0
        if small:
            records[index]["skipped"] = records[index].get("skipped", 0) + small
        total = totals[index] if index < len(totals) else None
        if total is not None and total > len(found[index]):
            records[index]["found"] = total
            if total - small > len(found[index]):
                records[index]["dropped"] = total - small - len(found[index])
        if index == 0:
            records[index]["detect_ms"] = info.get("detect_ms")
    result["images"] = images
    result["detailer"] = records


__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_CROP_FACTOR",
    "DEFAULT_DENOISE",
    "DEFAULT_DILATION",
    "DEFAULT_DROP_SIZE",
    "DEFAULT_FEATHER",
    "DEFAULT_GUIDE_SIZE",
    "DEFAULT_MAX_SIZE",
    "DEFAULT_TARGETS",
    "DetailerSettings",
    "Detection",
    "check_available",
    "composite",
    "crop_region",
    "detail_image",
    "detail_result",
    "detect",
    "dilate",
    "feather_mask",
    "infotext_fields",
    "segment_prompt",
    "target_concepts",
    "upscaled_size",
]
