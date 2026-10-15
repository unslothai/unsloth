# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 Image Klein: a classic pipeline face over Diffusers' modular FLUX.3 blocks.

Diffusers ships FLUX.3 only as a Modular Diffusers workflow (``Flux3DistilledAutoBlocks`` for the
16-step flow-map checkpoint, ``Flux3AutoBlocks`` for the undistilled base), and the repos carry a
``modular_model_index.json`` instead of a ``model_index.json``. The image backend drives every
other family through a ``DiffusionPipeline``: placement (``to`` / model and group offload), the
step callback (progress, cancel), the decode-phase hook and the scheduler swap all address
``pipe.<component>``. ``Flux3KleinPipeline`` is that face: it registers the same component
objects the blocks run, so offload hooks, compiled or quantised replacements and a swapped
scheduler reach the blocks unchanged, and its ``__call__`` takes the keywords the generate path
passes.

The component objects are loaded per subfolder (``text_encoder`` also carries the Qwen3-VL
processor), so only ``transformer/``, ``text_encoder/`` and ``vae/`` are ever read; the repo-root
BFL single file and ``ae.safetensors`` and the ``variants/`` quantised packages are not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

FLUX3_KLEIN_FAMILY_NAME = "flux.3-klein"
FLUX3_KLEIN_BASE_FAMILY_NAME = "flux.3-klein-base"
FLUX3_FAMILY_NAMES = (FLUX3_KLEIN_FAMILY_NAME, FLUX3_KLEIN_BASE_FAMILY_NAME)

# The class every FLUX.3 load needs from diffusers; its absence is what an unreleased diffusers looks like.
FLUX3_TRANSFORMER_CLASS = "Flux3Transformer2DModel"
FLUX3_DISTILLED_BLOCKS = "Flux3DistilledAutoBlocks"
FLUX3_BASE_BLOCKS = "Flux3AutoBlocks"
# Folders a load reads. The tokenizer / processor files live in text_encoder/.
FLUX3_COMPONENTS = ("text_encoder", "transformer", "vae")
FLUX3_INDEX = "modular_model_index.json"


def is_flux3_family(fam: Any) -> bool:
    return getattr(fam, "name", None) in FLUX3_FAMILY_NAMES


def flux3_available() -> bool:
    """Whether the installed diffusers can build FLUX.3 at all (imports diffusers)."""
    try:
        import diffusers
        return all(
            hasattr(diffusers, name)
            for name in (FLUX3_TRANSFORMER_CLASS, FLUX3_DISTILLED_BLOCKS, FLUX3_BASE_BLOCKS)
        )
    except Exception:  # noqa: BLE001 -- unimportable diffusers cannot build it either
        return False


def _live_cache_dir() -> str:
    """The live hub cache root (see diffusion_krea2._live_cache_dir for why it is pinned)."""
    from utils.hf_cache_settings import active_hf_hub_cache
    return active_hf_hub_cache()


def _read_index(repo_id: str, hf_token: Optional[str], local_files_only: bool) -> dict:
    try:
        root = Path(repo_id).expanduser()
        if root.is_dir():
            path = root / FLUX3_INDEX
            if not path.is_file():
                raise FileNotFoundError(f"{FLUX3_INDEX} not found in local model dir {repo_id}")
            payload = json.loads(path.read_text(encoding = "utf-8-sig"))
            return payload if isinstance(payload, dict) else {}
    except OSError:
        pass
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id,
        FLUX3_INDEX,
        token = hf_token or None,
        local_files_only = local_files_only,
        cache_dir = _live_cache_dir(),
    )
    payload = json.loads(Path(path).read_text(encoding = "utf-8-sig"))
    return payload if isinstance(payload, dict) else {}


def flux3_blocks_class(index: dict, transformer: Any = None) -> str:
    """The block graph a checkpoint runs: the index names it; else the denoiser's ``flow_map``."""
    named = index.get("_blocks_class_name") if isinstance(index, dict) else None
    if named in (FLUX3_DISTILLED_BLOCKS, FLUX3_BASE_BLOCKS):
        return named
    flow_map = getattr(getattr(transformer, "config", None), "flow_map", None)
    return FLUX3_DISTILLED_BLOCKS if flow_map in (None, True) else FLUX3_BASE_BLOCKS


@dataclass
class Flux3PipelineOutput:
    """``.images``, the one field the generate path reads (``ImagePipelineOutput``'s shape)."""

    images: list


class _Flux3Interrupted(Exception):
    """Raised out of the denoise loop when the step callback set ``_interrupt``."""


# Each reference is scaled to this area before the VAE, aspect kept, Lanczos: ComfyUI's
# ``ReferenceMultiLatent`` with the templates' ``["resize", 1]`` (1.0 megapixels), and the BFL
# reference's multi-reference rule (each <= 1 MiP). Diffusers alone only downsizes above 16 MiP, so
# a phone photo would otherwise be encoded at full size. The 16-px grid crop stays with the blocks.
REFERENCE_AREA = 1024 * 1024


def resize_reference(image: Any) -> Any:
    """``image`` at ~1 MiP, aspect kept (ComfyUI: ``round(side * sqrt(area / (w * h)))``, Lanczos)."""
    import math

    from PIL import Image

    width, height = image.size
    scale = math.sqrt(REFERENCE_AREA / float(max(1, width * height)))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size == (width, height):
        return image
    return image.resize(size, Image.LANCZOS)


def reference_noise(seed: int, height: int, width: int):
    """The BFL reference's initial noise as the blocks' (1, 128, h, w) latent.

    The reference draws ``randn((h * w, 128))`` row-major on a CPU generator and rounds it to bf16;
    diffusers draws ``(B, 128, h, w)`` on the request's generator. Same seed, different image. This
    makes a Studio seed render what the BFL CLI renders for that seed (verified against its outputs),
    on any device."""
    import torch

    h, w = height // 16, width // 16
    tokens = torch.randn((h * w, 128), generator = torch.Generator().manual_seed(int(seed)))
    return tokens.to(torch.bfloat16).reshape(h, w, 128).permute(2, 0, 1)[None].contiguous()


def _generator_seed(generator: Any) -> Optional[int]:
    initial_seed = getattr(generator, "initial_seed", None)
    if not callable(initial_seed):
        return None
    try:
        return int(initial_seed())
    except Exception:  # noqa: BLE001 -- a generator without a readable seed keeps diffusers' draw
        return None


_PIPELINE_CLASS: Optional[type] = None


def flux3_pipeline_class() -> type:
    """``Flux3KleinPipeline``, built on first use so importing this module never imports diffusers."""
    global _PIPELINE_CLASS
    if _PIPELINE_CLASS is None:
        from diffusers import DiffusionPipeline

        _PIPELINE_CLASS = type(
            "Flux3KleinPipeline", (_Flux3KleinPipelineMixin, DiffusionPipeline), {}
        )
    return _PIPELINE_CLASS


class _Flux3KleinPipelineMixin:
    """FLUX.3 Klein (distilled or base) as a ``DiffusionPipeline``; the denoise runs in the modular blocks."""

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components: list = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer,
        vae,
        text_encoder,
        tokenizer,
        scheduler,
        blocks_class_name: str = FLUX3_DISTILLED_BLOCKS,
    ):
        super().__init__()
        self.register_modules(
            transformer = transformer,
            vae = vae,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            scheduler = scheduler,
        )
        self.register_to_config(blocks_class_name = blocks_class_name)
        self._modular = None
        self._interrupt = False

    @property
    def is_distilled(self) -> bool:
        return self.config.blocks_class_name == FLUX3_DISTILLED_BLOCKS

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    def modular_pipeline(self):
        """The modular pipeline over THIS pipeline's current component objects."""
        import diffusers

        if self._modular is None:
            blocks = getattr(diffusers, self.config.blocks_class_name)()
            self._modular = blocks.init_pipeline()
        mp = self._modular
        updates = {}
        for name in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
            current = getattr(self, name, None)
            if current is not None and getattr(mp, name, None) is not current:
                updates[name] = current
        if updates:
            mp.update_components(**updates)
        return mp

    def _set_guidance(self, mp, guidance_scale: Optional[float]) -> None:
        guider = getattr(mp, "guider", None)
        if guider is None or guidance_scale is None:
            return
        if float(getattr(guider.config, "guidance_scale", -1.0)) == float(guidance_scale):
            return
        from diffusers import ClassifierFreeGuidance

        mp.update_components(guider = ClassifierFreeGuidance(guidance_scale = float(guidance_scale)))

    def __call__(
        self,
        prompt = None,
        image = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 16,
        guidance_scale: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        num_images_per_prompt: int = 1,
        generator = None,
        latents = None,
        max_sequence_length: int = 8192,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs = ("latents",),
    ):
        self._interrupt = False
        mp = self.modular_pipeline()
        self._set_guidance(mp, guidance_scale)

        prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        if any(not isinstance(p, str) for p in prompts):
            raise ValueError("FLUX.3 takes a text prompt (str) per image.")
        generators = generator if isinstance(generator, (list, tuple)) else None
        # The text encoder sizes its window per prompt, so the blocks take ONE prompt per call. A batch of
        # distinct prompts, or one prompt with a generator per image, runs as one call per image.
        if len(prompts) > 1 or generators is not None:
            count = max(len(prompts), len(generators or ()))
            if len(prompts) == 1:
                prompts = prompts * count
            if generators is not None and len(generators) != count:
                raise ValueError(
                    f"Got {len(generators)} generators for {count} images; pass one per image."
                )
            calls = [
                (prompts[i], generators[i] if generators is not None else generator, 1)
                for i in range(count)
            ]
        else:
            calls = [(prompts[0], generator, int(num_images_per_prompt or 1))]

        images: list = []
        try:
            for call_index, (one_prompt, one_generator, per_prompt) in enumerate(calls):
                kwargs: dict[str, Any] = {
                    "prompt": one_prompt,
                    "num_inference_steps": int(num_inference_steps),
                    "num_images_per_prompt": per_prompt,
                    "max_sequence_length": int(max_sequence_length),
                    "output_type": output_type,
                }
                if one_generator is not None:
                    kwargs["generator"] = one_generator
                if image is not None:
                    refs = list(image) if isinstance(image, (list, tuple)) else [image]
                    refs = [resize_reference(ref) for ref in refs]
                    kwargs["image"] = refs if len(refs) > 1 else refs[0]
                if height is not None:
                    kwargs["height"] = int(height)
                if width is not None:
                    kwargs["width"] = int(width)
                if latents is not None:
                    kwargs["latents"] = latents
                elif per_prompt == 1 and height is not None and width is not None:
                    seed = _generator_seed(one_generator)
                    if seed is not None:
                        kwargs["latents"] = reference_noise(seed, int(height), int(width))
                if negative_prompt is not None and not self.is_distilled:
                    kwargs["negative_prompt"] = negative_prompt
                with self._step_hooks(
                    mp,
                    callback_on_step_end,
                    num_inference_steps = int(num_inference_steps),
                    first_call = call_index == 0,
                ):
                    out = mp(**kwargs, output = "images")
                images.extend(out if isinstance(out, list) else [out])
        except _Flux3Interrupted:
            images = []
        if not return_dict:
            return (images,)
        return Flux3PipelineOutput(images = images)

    def _step_hooks(
        self,
        mp,
        callback: Optional[Callable],
        *,
        num_inference_steps: int,
        first_call: bool,
    ):
        """Fire ``callback_on_step_end`` after every scheduler step, and open ``self.progress_bar`` at
        the first denoiser forward, which is where every classic pipeline opens its loop (the
        generate path's denoise-phase hook watches it)."""
        import contextlib

        pipe = self

        @contextlib.contextmanager
        def _hooks():
            scheduler = mp.scheduler
            had_own_step = "step" in scheduler.__dict__
            original_step = scheduler.step
            counter = {"i": 0}

            def _step(*args, **kwargs):
                result = original_step(*args, **kwargs)
                i = counter["i"]
                counter["i"] += 1
                if callback is not None:
                    latents = result[0] if isinstance(result, tuple) else result.prev_sample
                    timestep = args[1] if len(args) > 1 else kwargs.get("timestep")
                    callback(pipe, i, timestep, {"latents": latents})
                if pipe._interrupt:
                    raise _Flux3Interrupted()
                return result

            handle = None
            transformer = getattr(mp, "transformer", None)
            if first_call and transformer is not None and hasattr(transformer, "register_forward_pre_hook"):
                fired = {"done": False}

                def _pre_hook(_module, _args):
                    if not fired["done"]:
                        fired["done"] = True
                        try:
                            bar = pipe.progress_bar(total = num_inference_steps)
                            close = getattr(bar, "close", None)
                            if callable(close):
                                close()
                        except Exception:  # noqa: BLE001 -- a phase label must never fail a render
                            pass

                handle = transformer.register_forward_pre_hook(_pre_hook)
            scheduler.step = _step
            try:
                yield
            finally:
                if had_own_step:
                    scheduler.step = original_step
                else:
                    try:
                        del scheduler.step
                    except AttributeError:
                        pass
                if handle is not None:
                    handle.remove()

        return _hooks()


def load_flux3_pipeline(
    repo_id: str,
    dtype,
    hf_token: Optional[str] = None,
    transformer = None,
    text_encoder = None,
    local_files_only: bool = False,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> Any:
    """A ready ``Flux3KleinPipeline`` for ``repo_id`` (on CPU; the caller places it).

    ``transformer`` / ``text_encoder`` let a quantised or pre-cast build hand in a component;
    the rest load from the repo's own subfolders. ``local_files_only`` keeps a load that was
    cleared as fully local from reaching the Hub (see diffusion_krea2.load_krea2_pipeline)."""
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    if not flux3_available():
        from .diffusion_families import assert_pipeline_class_available
        assert_pipeline_class_available(FLUX3_TRANSFORMER_CLASS, FLUX3_KLEIN_FAMILY_NAME)
        raise RuntimeError("FLUX.3 needs a diffusers build with the FLUX.3 modular blocks.")
    import diffusers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    token = hf_token or None
    common: dict[str, Any] = {"local_files_only": local_files_only, "cache_dir": _live_cache_dir()}
    if token:
        common["token"] = token
    index = _read_index(repo_id, token, local_files_only)
    check_cancelled()
    tokenizer = AutoProcessor.from_pretrained(repo_id, subfolder = "text_encoder", **common)
    check_cancelled()
    if text_encoder is None:
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            repo_id, subfolder = "text_encoder", dtype = dtype, **common
        )
        text_encoder.eval()
        check_cancelled()
    if transformer is None:
        transformer = diffusers.Flux3Transformer2DModel.from_pretrained(
            repo_id, subfolder = "transformer", torch_dtype = dtype, **common
        )
        check_cancelled()
    vae = diffusers.AutoencoderKLFlux2.from_pretrained(
        repo_id, subfolder = "vae", torch_dtype = dtype, **common
    )
    check_cancelled()
    blocks = flux3_blocks_class(index, transformer)
    # The scheduler is the block graph's own (shift 6.93, the reference timestep shift); the repos ship none.
    scheduler = getattr(diffusers, blocks)().init_pipeline().scheduler
    return flux3_pipeline_class()(
        transformer = transformer,
        vae = vae,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        scheduler = scheduler,
        blocks_class_name = blocks,
    )
