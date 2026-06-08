# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Optional

from PIL import Image


_RESAMPLE = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen = True)
class ImageEnhancePlan:
    mode: str = "off"
    upscale_enabled: bool = False
    upscale_mode: str = "pixel"
    scale: float = 1.0
    method: str = "lanczos"
    tile_size: Optional[int] = None
    tile_overlap: int = 0
    strength: Optional[float] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    tile_policy: str = "auto"
    vae_decode: str = "auto"

    @property
    def enabled(self) -> bool:
        return self.mode != "off" or self.upscale_enabled or self.vae_decode == "on"

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "upscale_enabled": self.upscale_enabled,
            "upscale_mode": self.upscale_mode,
            "scale": self.scale,
            "method": self.method,
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "tile_policy": self.tile_policy,
            "vae_decode": self.vae_decode,
        }


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _accepts_kwarg(callable_obj: Any, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
    return name in signature.parameters


def build_image_enhance_plan(enhance: Any) -> ImageEnhancePlan:
    if enhance is None:
        return ImageEnhancePlan()
    mode = str(_field(enhance, "mode", "off") or "off")
    if mode == "off":
        return ImageEnhancePlan()

    upscale = _field(enhance, "upscale")
    tiling = _field(enhance, "tiling")
    upscale_enabled = bool(_field(upscale, "enabled", mode in ("upscale", "creative_upscale", "large_tiled")))
    upscale_mode = str(
        _field(
            upscale,
            "mode",
            "diffusion" if mode == "creative_upscale" else "pixel",
        )
        or "pixel"
    )
    scale = float(_field(upscale, "scale", 2.0 if upscale_enabled else 1.0) or 1.0)
    method = str(_field(upscale, "method", "lanczos") or "lanczos")
    if method not in _RESAMPLE:
        raise ValueError("upscale method must be nearest, bilinear, bicubic, or lanczos")
    tile_policy = str(_field(tiling, "enabled", "auto") or "auto")
    if tile_policy not in ("auto", "on", "off"):
        raise ValueError("tiling.enabled must be auto, on, or off")
    vae_decode = str(
        _field(
            tiling,
            "vae_decode",
            "on" if mode == "large_tiled" else "auto",
        )
        or "auto"
    )
    if vae_decode not in ("auto", "on", "off"):
        raise ValueError("tiling.vae_decode must be auto, on, or off")
    tile_size = _field(upscale, "tile_size", None)
    if tile_size is None:
        tile_size = _field(tiling, "tile_size", None)
    tile_overlap = _field(upscale, "tile_overlap", None)
    if tile_overlap is None:
        tile_overlap = _field(tiling, "overlap", 0)
    return ImageEnhancePlan(
        mode = mode,
        upscale_enabled = upscale_enabled,
        upscale_mode = upscale_mode,
        scale = scale,
        method = method,
        tile_size = int(tile_size) if tile_size is not None else None,
        tile_overlap = int(tile_overlap or 0),
        strength = _field(upscale, "strength", None),
        num_inference_steps = _field(upscale, "num_inference_steps", None),
        guidance_scale = _field(upscale, "guidance_scale", None),
        prompt = _field(upscale, "prompt", None),
        negative_prompt = _field(upscale, "negative_prompt", None),
        tile_policy = tile_policy,
        vae_decode = vae_decode,
    )


def apply_vae_tiling_policy(
    pipe: Any,
    plan: ImageEnhancePlan,
    *,
    width: int,
    height: int,
) -> Optional[dict[str, Any]]:
    if plan.vae_decode == "off":
        _call_if_present(pipe, "disable_vae_tiling")
        return {"vae_decode": "off", "applied": False}
    should_enable = plan.vae_decode == "on" or (
        plan.vae_decode == "auto" and (width * height) > (1024 * 1024)
    )
    if not should_enable:
        return {"vae_decode": plan.vae_decode, "applied": False}
    applied = False
    for method in ("enable_vae_tiling", "enable_vae_slicing"):
        applied = _call_if_present(pipe, method) or applied
    return {"vae_decode": plan.vae_decode, "applied": applied}


def _call_if_present(obj: Any, method_name: str) -> bool:
    method = getattr(obj, method_name, None)
    if not callable(method):
        return False
    method()
    return True


def apply_image_enhancement(
    images: list[Any],
    plan: ImageEnhancePlan,
    *,
    diffusion_upscaler_pipe: Any = None,
    prompt: str,
    negative_prompt: Optional[str],
    generator_factory: Optional[Callable[[], Any]] = None,
) -> tuple[list[Any], dict[str, Any]]:
    if not plan.enabled:
        return images, {"mode": "off", "stages": []}

    stages: list[dict[str, Any]] = []
    enhanced = images
    if plan.upscale_enabled:
        if plan.upscale_mode == "pixel":
            enhanced = [_pixel_upscale(image, plan) for image in enhanced]
            stages.append(
                {
                    "type": "pixel_upscale",
                    "scale": plan.scale,
                    "method": plan.method,
                    "tile_size": plan.tile_size,
                    "tile_overlap": plan.tile_overlap,
                }
            )
        elif plan.upscale_mode == "super_resolution":
            if diffusion_upscaler_pipe is None:
                raise RuntimeError(
                    "Super-resolution upscale requested, but no image upscaler model is loaded."
                )
            next_images = []
            tiled_count = 0
            for image in enhanced:
                if _should_tile_image(image, plan):
                    image, tile_count = _tiled_super_resolution_upscale(
                        diffusion_upscaler_pipe,
                        image,
                        plan,
                    )
                    tiled_count += tile_count
                else:
                    image = _super_resolution_upscale(
                        diffusion_upscaler_pipe,
                        image,
                        plan,
                    )
                next_images.append(image)
            enhanced = next_images
            stages.append(
                {
                    "type": "super_resolution_upscale",
                    "scale": plan.scale,
                    "tiled": tiled_count > 0,
                    "tile_count": tiled_count,
                    "tile_size": plan.tile_size,
                    "tile_overlap": plan.tile_overlap,
                }
            )
        elif plan.upscale_mode == "diffusion":
            if diffusion_upscaler_pipe is None:
                raise RuntimeError(
                    "Creative upscale requested, but no diffusion upscaler pipeline is loaded."
                )
            next_images = []
            tiled_count = 0
            for image in enhanced:
                generator = generator_factory() if generator_factory else None
                call_prompt = plan.prompt or prompt
                call_negative_prompt = (
                    plan.negative_prompt
                    if plan.negative_prompt is not None
                    else negative_prompt
                )
                if _should_tile_image(image, plan):
                    image, tile_count = _tiled_diffusion_upscale(
                        diffusion_upscaler_pipe,
                        image,
                        plan,
                        prompt = call_prompt,
                        negative_prompt = call_negative_prompt,
                        generator = generator,
                    )
                    tiled_count += tile_count
                else:
                    image = _diffusion_upscale(
                        diffusion_upscaler_pipe,
                        image,
                        plan,
                        prompt = call_prompt,
                        negative_prompt = call_negative_prompt,
                        generator = generator,
                    )
                next_images.append(image)
            enhanced = next_images
            stages.append(
                {
                    "type": "diffusion_upscale",
                    "scale": plan.scale,
                    "num_inference_steps": plan.num_inference_steps,
                    "guidance_scale": plan.guidance_scale,
                    "strength": plan.strength,
                    "tiled": tiled_count > 0,
                    "tile_count": tiled_count,
                    "tile_size": plan.tile_size,
                    "tile_overlap": plan.tile_overlap,
                }
            )
        else:
            raise ValueError("upscale.mode must be pixel, super_resolution, or diffusion")

    if plan.mode == "large_tiled":
        stages.append(
            {
                "type": "large_tiled",
                "tile_policy": plan.tile_policy,
                "tile_size": plan.tile_size,
                "tile_overlap": plan.tile_overlap,
            }
        )
    return enhanced, {"mode": plan.mode, "stages": stages}


def _pixel_upscale(image: Any, plan: ImageEnhancePlan) -> Any:
    if not isinstance(image, Image.Image):
        return image
    out_size = (
        max(1, int(round(image.width * plan.scale))),
        max(1, int(round(image.height * plan.scale))),
    )
    if out_size == image.size:
        return image.copy()
    tile_size = plan.tile_size
    if not _should_tile_image(image, plan):
        return image.resize(out_size, _RESAMPLE[plan.method])
    return _tiled_resize(image, out_size, plan)


def _should_tile_image(image: Any, plan: ImageEnhancePlan) -> bool:
    if not isinstance(image, Image.Image):
        return False
    tile_size = plan.tile_size
    should_tile = (
        plan.tile_policy == "on"
        or (plan.tile_policy == "auto" and tile_size is not None)
    )
    return bool(
        should_tile
        and tile_size
        and (image.width > tile_size or image.height > tile_size)
    )


def _tiled_resize(image: Image.Image, out_size: tuple[int, int], plan: ImageEnhancePlan) -> Image.Image:
    tile = max(128, int(plan.tile_size or 512))
    overlap = max(0, min(int(plan.tile_overlap or 0), tile // 2))
    step = max(1, tile - overlap)
    scale_x = out_size[0] / image.width
    scale_y = out_size[1] / image.height
    result = Image.new(image.mode, out_size)
    for top in range(0, image.height, step):
        for left in range(0, image.width, step):
            right = min(image.width, left + tile)
            bottom = min(image.height, top + tile)
            crop = image.crop((left, top, right, bottom))
            dst_box = (
                int(round(left * scale_x)),
                int(round(top * scale_y)),
                int(round(right * scale_x)),
                int(round(bottom * scale_y)),
            )
            resized = crop.resize(
                (max(1, dst_box[2] - dst_box[0]), max(1, dst_box[3] - dst_box[1])),
                _RESAMPLE[plan.method],
            )
            result.paste(resized, dst_box[:2])
    return result


def _tiled_diffusion_upscale(
    pipe: Any,
    image: Image.Image,
    plan: ImageEnhancePlan,
    *,
    prompt: str,
    negative_prompt: Optional[str],
    generator: Any,
) -> tuple[Image.Image, int]:
    out_size = (
        max(1, int(round(image.width * plan.scale))),
        max(1, int(round(image.height * plan.scale))),
    )
    tile = max(128, int(plan.tile_size or 768))
    overlap = max(0, min(int(plan.tile_overlap or 0), tile // 2))
    step = max(1, tile - overlap)
    scale_x = out_size[0] / image.width
    scale_y = out_size[1] / image.height
    result = Image.new("RGBA", out_size, (0, 0, 0, 0))
    tile_count = 0

    for top in range(0, image.height, step):
        for left in range(0, image.width, step):
            right = min(image.width, left + tile)
            bottom = min(image.height, top + tile)
            crop = image.crop((left, top, right, bottom))
            dst_box = (
                int(round(left * scale_x)),
                int(round(top * scale_y)),
                int(round(right * scale_x)),
                int(round(bottom * scale_y)),
            )
            tile_plan = ImageEnhancePlan(
                mode = plan.mode,
                upscale_enabled = plan.upscale_enabled,
                upscale_mode = plan.upscale_mode,
                scale = plan.scale,
                method = plan.method,
                tile_size = None,
                tile_overlap = 0,
                strength = plan.strength,
                num_inference_steps = plan.num_inference_steps,
                guidance_scale = plan.guidance_scale,
                prompt = plan.prompt,
                negative_prompt = plan.negative_prompt,
                tile_policy = "off",
                vae_decode = plan.vae_decode,
            )
            upscaled = _diffusion_upscale(
                pipe,
                crop,
                tile_plan,
                prompt = prompt,
                negative_prompt = negative_prompt,
                generator = generator,
            )
            if not isinstance(upscaled, Image.Image):
                return upscaled, tile_count + 1
            target_size = (
                max(1, dst_box[2] - dst_box[0]),
                max(1, dst_box[3] - dst_box[1]),
            )
            if upscaled.size != target_size:
                upscaled = upscaled.resize(target_size, _RESAMPLE[plan.method])
            mask = _tile_feather_mask(
                target_size,
                feather_x = int(round(overlap * scale_x)),
                feather_y = int(round(overlap * scale_y)),
                fade_left = left > 0,
                fade_top = top > 0,
                fade_right = right < image.width,
                fade_bottom = bottom < image.height,
            )
            result.paste(upscaled.convert("RGBA"), dst_box[:2], mask)
            tile_count += 1

    if image.mode == "RGBA":
        return result, tile_count
    return result.convert(image.mode), tile_count


def _tiled_super_resolution_upscale(
    pipe: Any,
    image: Image.Image,
    plan: ImageEnhancePlan,
) -> tuple[Image.Image, int]:
    out_size = (
        max(1, int(round(image.width * plan.scale))),
        max(1, int(round(image.height * plan.scale))),
    )
    tile = max(128, int(plan.tile_size or 768))
    overlap = max(0, min(int(plan.tile_overlap or 0), tile // 2))
    step = max(1, tile - overlap)
    scale_x = out_size[0] / image.width
    scale_y = out_size[1] / image.height
    result = Image.new("RGBA", out_size, (0, 0, 0, 0))
    tile_count = 0

    for top in range(0, image.height, step):
        for left in range(0, image.width, step):
            right = min(image.width, left + tile)
            bottom = min(image.height, top + tile)
            crop = image.crop((left, top, right, bottom))
            dst_box = (
                int(round(left * scale_x)),
                int(round(top * scale_y)),
                int(round(right * scale_x)),
                int(round(bottom * scale_y)),
            )
            upscaled = _super_resolution_upscale(pipe, crop, plan)
            if not isinstance(upscaled, Image.Image):
                return upscaled, tile_count + 1
            target_size = (
                max(1, dst_box[2] - dst_box[0]),
                max(1, dst_box[3] - dst_box[1]),
            )
            if upscaled.size != target_size:
                upscaled = upscaled.resize(target_size, _RESAMPLE[plan.method])
            mask = _tile_feather_mask(
                target_size,
                feather_x = int(round(overlap * scale_x)),
                feather_y = int(round(overlap * scale_y)),
                fade_left = left > 0,
                fade_top = top > 0,
                fade_right = right < image.width,
                fade_bottom = bottom < image.height,
            )
            result.paste(upscaled.convert("RGBA"), dst_box[:2], mask)
            tile_count += 1

    if image.mode == "RGBA":
        return result, tile_count
    return result.convert(image.mode), tile_count


def _tile_feather_mask(
    size: tuple[int, int],
    *,
    feather_x: int,
    feather_y: int,
    fade_left: bool,
    fade_top: bool,
    fade_right: bool,
    fade_bottom: bool,
) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 255)
    px = mask.load()
    feather_x = max(0, min(feather_x, width // 2))
    feather_y = max(0, min(feather_y, height // 2))
    for y in range(height):
        y_alpha = 255
        if feather_y:
            if fade_top and y < feather_y:
                y_alpha = min(y_alpha, int(255 * (y + 1) / feather_y))
            if fade_bottom and y >= height - feather_y:
                y_alpha = min(y_alpha, int(255 * (height - y) / feather_y))
        for x in range(width):
            alpha = y_alpha
            if feather_x:
                if fade_left and x < feather_x:
                    alpha = min(alpha, int(255 * (x + 1) / feather_x))
                if fade_right and x >= width - feather_x:
                    alpha = min(alpha, int(255 * (width - x) / feather_x))
            px[x, y] = max(0, min(255, alpha))
    return mask


def _super_resolution_upscale(pipe: Any, image: Any, plan: ImageEnhancePlan) -> Any:
    if not isinstance(image, Image.Image):
        return image
    call = getattr(pipe, "__call__")
    if _accepts_kwarg(call, "images"):
        out = pipe(images = image)
    elif _accepts_kwarg(call, "image"):
        out = pipe(image = image)
    else:
        out = pipe(image)
    upscaled = _extract_first_image(out)
    target_size = (
        max(1, int(round(image.width * plan.scale))),
        max(1, int(round(image.height * plan.scale))),
    )
    if isinstance(upscaled, Image.Image) and upscaled.size != target_size:
        upscaled = upscaled.resize(target_size, _RESAMPLE[plan.method])
    return upscaled


def _extract_first_image(out: Any) -> Any:
    images = getattr(out, "images", None)
    if images is not None:
        out = images
    if isinstance(out, Image.Image):
        return out
    if isinstance(out, dict):
        for key in ("image", "generated_image", "output", "images"):
            value = out.get(key)
            if isinstance(value, list) and value:
                return _extract_first_image(value[0])
            if value is not None:
                return value
    if isinstance(out, list) and out:
        return _extract_first_image(out[0])
    raise RuntimeError("Image upscaler returned no image.")


def _diffusion_upscale(
    pipe: Any,
    image: Any,
    plan: ImageEnhancePlan,
    *,
    prompt: str,
    negative_prompt: Optional[str],
    generator: Any,
) -> Any:
    call = getattr(pipe, "__call__")
    target_width = max(1, int(round(getattr(image, "width", 0) * plan.scale)))
    target_height = max(1, int(round(getattr(image, "height", 0) * plan.scale)))
    kwargs: dict[str, Any] = {"prompt": prompt, "image": image}
    if plan.num_inference_steps is not None and _accepts_kwarg(call, "num_inference_steps"):
        kwargs["num_inference_steps"] = int(plan.num_inference_steps)
    if plan.guidance_scale is not None and _accepts_kwarg(call, "guidance_scale"):
        kwargs["guidance_scale"] = float(plan.guidance_scale)
    if plan.strength is not None and _accepts_kwarg(call, "strength"):
        kwargs["strength"] = float(plan.strength)
    if negative_prompt and _accepts_kwarg(call, "negative_prompt"):
        kwargs["negative_prompt"] = negative_prompt
    if generator is not None and _accepts_kwarg(call, "generator"):
        kwargs["generator"] = generator
    if _accepts_kwarg(call, "width"):
        kwargs["width"] = target_width
    if _accepts_kwarg(call, "height"):
        kwargs["height"] = target_height
    out = pipe(**kwargs)
    images = getattr(out, "images", out)
    if isinstance(images, list) and images:
        return images[0]
    raise RuntimeError("Diffusion upscaler returned no images.")
