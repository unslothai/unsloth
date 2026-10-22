# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Detailer (face fix) post-pass for the Images page, after AUTOMATIC1111's ADetailer and ComfyUI-Impact-Pack's
FaceDetailer: find the targets with SAM 3's text prompts ("face", "hand", any text), then for each detection crop a
padded region around it, enlarge it, inpaint the masked target at a moderate denoise with the same model, and paste the
result back through a feathered mask. Pixels the feathered mask does not reach are the original ones, bit for bit.

Where the two tools differ, the setting follows FaceDetailer (ComfyUI-Impact-Pack, modules/impact/impact_pack.py
FaceDetailer.INPUT_TYPES and modules/impact/core.py / utils.py):

- detection threshold 0.5 (``bbox_threshold``), mask dilation 10 px (``bbox_dilation``), and a detection whose box
  side is not above 10 px is dropped (``drop_size``, ONNXDetector.detect);
- crop region = the box scaled by 3.0 about its centre (``bbox_crop_factor``, utils.make_crop_region), shifted back
  inside the picture (utils.normalize_region);
- the crop is enlarged so the box's short side reaches 512 (``guide_size``, ``guide_size_for`` = bbox), capped so
  neither side passes 1024 (``max_size``), and never shrunk (``force_inpaint``: core.enhance_detail renders a big
  target at 1x);
- denoise 0.5 (``denoise``), only the mask is denoised (``noise_mask``), the paste-back mask is the crop's mask blurred
  with a (2 * 5 + 1) Gaussian kernel at sigma 10 (``feather``, utils.tensor_gaussian_blur_mask);
- detections run one after another on the image the previous one left (impact_pack.py DetailerForEach.do_detail), each
  with seed + its index (``seg_seed = seed + i``), which is also ADetailer's per-detection seed.

ADetailer (Bing-su/adetailer) is followed where FaceDetailer has no counterpart: the optional separate prompt and
negative prompt, where empty means the image's own prompt, ``[PROMPT]`` stands for it and ``[SEP]`` gives later
detections their own text (!adetailer.py:289-307, 867), and the infotext keys it writes (adetailer/args.py
``_all_args``), so a pasted or dropped PNG restores the pass. The steps, CFG and sampler are the image's own unless
overridden, as in ADetailer (its "use separate steps / CFG" are off by default).

Two departures, both because Studio's detector is SAM 3 rather than a YOLO box model: the mask is SAM 3's instance mask
(dilated), not the box rectangle, and a target with no ``:N`` keeps up to DEFAULT_PER_TARGET detections instead of
SAM 3's usual single best one. SAM 3 is loaded only for the detection and released before the first inpaint, so it is
never resident beside the image model while that denoises.
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

# ── defaults (FaceDetailer, modules/impact/impact_pack.py:743-769; utils.py:454) ────────────────────────────────────
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
# A target with no ":N" keeps this many detections; an explicit N is capped at MAX_PER_TARGET.
DEFAULT_PER_TARGET = 8
MAX_PER_TARGET = 20
# Every detection costs one inpaint render: the pass stops after this many per image.
MAX_DETECTIONS = 32
# Studio's inpaint snaps the source to a multiple of 16 (diffusion.py _snap_to_multiple); sized here so it never has to.
SIZE_MULTIPLE = 16
# The seed space the request accepts (DiffusionGenerateRequest: JSON-safe seeds).
_SEED_SPACE = 2**53

DETAILER_UNAVAILABLE_INPAINT = (
    "The Detailer needs a model with an inpaint pipeline; this model has none."
)

_MAX_SUFFIX = re.compile(r"^(.+?)\s*:\s*([\d.]+)\s*$")
_SEP = re.compile(r"\s*\[SEP\]\s*")


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
        return settings


@dataclass
class Detection:
    concept: str
    score: float
    box: tuple[float, float, float, float]
    # (H, W) bool at the image's size.
    mask: Any


@dataclass
class DetailResult:
    image: Any
    record: dict = field(default_factory = dict)


# ── pure geometry ───────────────────────────────────────────────────────────────────────────────────────────────────
def target_concepts(targets: Optional[str]) -> list[tuple[str, int]]:
    """``face, hand:2`` -> [("face", DEFAULT_PER_TARGET), ("hand", 2)]: SAM 3's prompt syntax (diffusion_segment
    .parse_prompt), with the detailer's own count when a target names none."""
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
    """Impact-Pack utils.normalize_region: shift the span back inside [0, limit)."""
    if start < 0:
        return 0, int(min(limit, size))
    if start + size > limit:
        return int(max(0, limit - size)), int(limit)
    return int(start), int(min(limit, start + size))


def crop_region(
    width: int, height: int, box: tuple[float, float, float, float], crop_factor: float
) -> tuple[int, int, int, int]:
    """Impact-Pack utils.make_crop_region: the box scaled by ``crop_factor`` about its centre, kept in the image."""
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
    """Impact-Pack core.enhance_detail's size with force_inpaint on: the box's (or crop's) short side to
    ``guide_size``, then capped at ``max_size``; a factor at or below 1 renders the crop at its own size."""
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
    """Down to the inpaint grid, at least one cell."""
    return max(multiple, width // multiple * multiple), max(multiple, height // multiple * multiple)


def dilate(mask: Any, pixels: int) -> Any:
    """cv2.dilate / cv2.erode with an ``abs(pixels)`` square kernel (Impact-Pack utils.dilate_masks), anchored at its
    centre; erosion treats outside the array as set, as OpenCV's default border does."""
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
            # moved[i] = src[i + shift]
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
    """Impact-Pack utils.tensor_gaussian_blur_mask: a (2 * feather + 1) Gaussian at ``sigma`` with reflect padding,
    the kernel shrunk to half the short side (odd) on a small crop, and no blur below a 3 px kernel. Float32 in [0, 1]."""
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
    """``(1 - alpha) * base + alpha * patch`` (Impact-Pack utils.tensor_paste) on uint8 RGB, rounded; a pixel with
    alpha 0 is ``base``'s byte for byte."""
    import numpy as np

    base = np.asarray(base, dtype = np.uint8)
    patch = np.asarray(patch, dtype = np.uint8)
    a = np.asarray(alpha, dtype = np.float32)[..., None]
    mixed = base.astype(np.float32) * (1.0 - a) + patch.astype(np.float32) * a
    mixed = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return np.where(a > 0, mixed, base)


def detection_seed(base_seed: int, index: int) -> int:
    """Impact-Pack ``seg_seed = seed + i`` (ADetailer's ``seed + j``), kept in the request's seed range."""
    return (int(base_seed) + int(index)) % _SEED_SPACE


def detection_prompt(template: Optional[str], main: str, index: int) -> Optional[str]:
    """ADetailer's prompt rule: split on [SEP], detection ``index`` takes the matching part (the last one repeats),
    an empty part is the image's own prompt and [PROMPT] stands for it. None means [SKIP]."""
    parts = _SEP.split(template or "")
    part = parts[min(index, len(parts) - 1)].strip()
    if part == "[SKIP]":
        return None
    if not part:
        return main
    return part.replace("[PROMPT]", main)


# ── the pass ────────────────────────────────────────────────────────────────────────────────────────────────────────
InpaintFn = Callable[[Any, Any, int, str, Optional[str]], Any]


def detail_image(
    image: Any,
    detections: list[Detection],
    settings: DetailerSettings,
    *,
    base_seed: int,
    prompt: str,
    negative_prompt: Optional[str],
    inpaint: InpaintFn,
) -> DetailResult:
    """Run every detection in order on ``image`` (PIL RGB). ``inpaint(crop, mask, seed, prompt, negative)`` renders
    one enlarged crop with its binary mask and returns the image at that size."""
    import numpy as np
    from PIL import Image

    started = time.perf_counter()
    current = np.array(image.convert("RGB"), dtype = np.uint8)
    height, width = current.shape[:2]
    done: list[dict] = []
    skipped = 0
    for index, det in enumerate(detections[:MAX_DETECTIONS]):
        x1, y1, x2, y2 = det.box
        if not (x2 - x1 > settings.drop_size and y2 - y1 > settings.drop_size):
            skipped += 1
            continue
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
        mask = dilate(np.asarray(det.mask, dtype = bool)[cy1:cy2, cx1:cx2], settings.dilation)
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
    if len(detections) > MAX_DETECTIONS:
        record["dropped"] = len(detections) - MAX_DETECTIONS
    return DetailResult(Image.fromarray(current), record)


def settings_record(settings: DetailerSettings) -> dict:
    """The settings as the recipe stores them: every value that ran, overrides only when set."""
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


# ── infotext (ADetailer's keys, adetailer/args.py _all_args) ────────────────────────────────────────────────────────
def _quote(value: Any) -> str:
    """AUTOMATIC1111 infotext_utils.quote: values holding a comma, colon or newline are JSON strings."""
    text = str(value)
    if "," not in text and "\n" not in text and ":" not in text:
        return text
    return json.dumps(text, ensure_ascii = False)


def infotext_fields(record: Optional[dict], model: Optional[str] = None) -> str:
    """``, ADetailer model: ..., ...`` for the parameters line, or "" when no detailer ran. ADetailer always writes
    model, confidence, dilate erode, mask blur and denoising strength, and the rest only off their defaults."""
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


# ── engine glue ─────────────────────────────────────────────────────────────────────────────────────────────────────
def check_available(status: dict) -> None:
    """Raise ValueError (400) when the loaded model cannot run the pass: no inpaint pipeline (which also covers the
    native sd.cpp engine, whose status offers no inpaint), or no SAM 3."""
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
    """One instance's mask logits -> (height, width) bool, as diffusion_segment.union_mask upsamples them."""
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        up = F.interpolate(
            plane[None, None].float(), size = (height, width), mode = "bilinear", align_corners = False
        )[0, 0]
        return (up > 0).numpy()


def detect(images: list, settings: DetailerSettings) -> tuple[list[list[Detection]], dict]:
    """SAM 3 on every image, then released: it never stays resident beside the image model while that denoises."""
    from core.inference import diffusion_segment

    prompt = segment_prompt(settings.targets)
    segmenter = diffusion_segment.get_segmenter()
    out: list[list[Detection]] = []
    info: dict = {}
    started = time.perf_counter()
    try:
        for image in images:
            result = segmenter.segment(
                image.convert("RGB"), prompt, threshold = settings.confidence, instance_masks = True
            )
            width, height = image.size
            out.append(
                [
                    Detection(
                        inst.concept, inst.score, inst.box, _plane_to_mask(plane, height, width)
                    )
                    for inst, plane in zip(result.instances, result.instance_logits)
                ]
            )
            info = {"device": result.device, "precision": result.precision}
    finally:
        diffusion_segment.release_segmenter("detailer", wait = True)
    info["model"] = diffusion_segment.active_variant().id
    info["detect_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
    return out, info


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
) -> None:
    """Detail every image of a finished render in place: ``result["images"]`` are replaced and
    ``result["detailer"]`` holds one recipe record per image. ``generate`` is the engine's generate() with the
    run's model identity, LoRAs and sampler bound; a failure is recorded on the image (the render is kept), a cancel
    stops the pass."""
    from core.inference.diffusion_families import DIFFUSION_CANCELLED_MSG

    images = list(result.get("images") or [])
    if not images:
        return
    seeds = result.get("seeds") or []
    records: list[Optional[dict]] = [None] * len(images)
    try:
        found, info = detect(images, settings)
    except Exception as exc:  # noqa: BLE001 -- the render stands; the recipe says why it was not detailed
        logger.warning("images.detailer: detection failed: %s", exc)
        result["detailer"] = [{**settings_record(settings), "error": str(exc)} for _ in images]
        return
    step_count = settings.steps if settings.steps is not None else steps
    cfg = settings.guidance if settings.guidance is not None else guidance
    cancelled = False
    for index, image in enumerate(images):
        base_seed = seeds[index] if index < len(seeds) else result.get("seed") or 0
        if cancelled:
            records[index] = {**settings_record(settings), "error": "cancelled"}
            continue

        def _inpaint(crop, mask, seed, det_prompt, det_negative):
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
            )
        except Exception as exc:  # noqa: BLE001
            if str(exc) == DIFFUSION_CANCELLED_MSG:
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
            "model": info.get("model"),
            "detect_device": info.get("device"),
        }
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
