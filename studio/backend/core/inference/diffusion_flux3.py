# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 pipeline facade over Modular Diffusers blocks; reads only transformer/, text_encoder/, vae/."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from loggers import get_logger

from . import diffusion_cond_blocks as cond_blocks

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
# The manifest components load_flux3_pipeline reads, and the folder of this repo each comes from (the processor
# lives with the encoder), whatever source a modular spec names. Nothing else in the manifest is loaded: since
# 2026-10-16 the official manifests also list the optional prompt upsampler (``prompt_upsampler`` /
# ``prompt_upsampler_processor``, from the separate ~19 GB flux-3-image-klein-pup repo), which an image load never
# needs. Validation and the locality checks read this map, so they require exactly what will load: every one of
# these folders (the loader opens all of them, listed in the manifest or not), and nothing else.
FLUX3_COMPONENT_FOLDERS = {
    "text_encoder": ("text_encoder", "transformers", "Qwen3VLForConditionalGeneration"),
    "tokenizer": ("text_encoder", "transformers", "Qwen3VLProcessor"),
    "transformer": ("transformer", "diffusers", "Flux3Transformer2DModel"),
    "vae": ("vae", "diffusers", "AutoencoderKLFlux2"),
}


# The prompt upsampler step Diffusers main puts first in both auto block graphs since huggingface/diffusers#15058
# (2026-10-16). It needs the separate ~19 GB Qwen3.5-9B upsampler, which Studio never loads, and would otherwise run
# on every call (its text-to-image branch has no trigger input), so it is dropped, as its docstring prescribes.
FLUX3_PROMPT_UPSAMPLER_STEP = "prompt_upsampler"


# diffusers ``Flux3TextEncoderStep`` helper args; a build whose helper takes others keeps its own text step.
FLUX3_TEXT_STEP = "text_encoder"
_TEXT_HELPER = "_get_qwen3_vl_prompt_embeds"
_TEXT_HELPER_ARGS = frozenset(
    {
        "text_encoder",
        "tokenizer",
        "prompt",
        "device",
        "max_sequence_length",
        "pad_multiple",
        "hidden_states_layers",
    }
)
_TEXT_LAYERS_INPUT = "text_encoder_out_layers"


def _text_step(mp: Any) -> tuple[Any, Any]:
    """``(step, helper)``: the modular pipeline's text step and its class's prompt helper, or ``(None, None)``."""
    import inspect

    sub_blocks = getattr(getattr(mp, "_blocks", None), "sub_blocks", None)
    step = sub_blocks.get(FLUX3_TEXT_STEP) if sub_blocks is not None else None
    helper = getattr(type(step), _TEXT_HELPER, None) if step is not None else None
    if not callable(helper):
        return None, None
    try:
        if not _TEXT_HELPER_ARGS <= set(inspect.signature(helper).parameters):
            return None, None
    except (TypeError, ValueError):
        return None, None
    return step, helper


def _bridge_text_step(pipe: Any, mp: Any) -> bool:
    """Route the text step's per-prompt encode through ``pipe.encode_prompt`` (prompt cache, dead-tail probe).
    Only for this pipe's own encoder and tokenizer; a foreign encoder runs the step's helper unchanged."""
    import inspect
    import weakref

    step, helper = _text_step(mp)
    if step is None:
        return False
    signature = inspect.signature(helper)
    owner_ref = weakref.ref(pipe)

    def bridged(*args: Any, **kwargs: Any) -> Any:
        owner = owner_ref()
        try:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
        except TypeError:
            return helper(*args, **kwargs)
        a = bound.arguments
        if (
            owner is None
            or a["text_encoder"] is not getattr(owner, "text_encoder", None)
            or a["tokenizer"] is not getattr(owner, "tokenizer", None)
        ):
            return helper(*args, **kwargs)
        return owner.encode_prompt(
            prompt = a["prompt"],
            max_sequence_length = a["max_sequence_length"],
            device = a["device"],
            pad_multiple = a["pad_multiple"],
            hidden_states_layers = a["hidden_states_layers"],
        )

    setattr(step, _TEXT_HELPER, bridged)
    return True


def flux3_blocks(blocks_class_name: str) -> Any:
    """The named FLUX.3 auto blocks without the prompt upsampler step (absent before diffusers#15058)."""
    import diffusers

    blocks = getattr(diffusers, blocks_class_name)()
    sub_blocks = getattr(blocks, "sub_blocks", None)
    if sub_blocks is not None and FLUX3_PROMPT_UPSAMPLER_STEP in sub_blocks:
        sub_blocks.pop(FLUX3_PROMPT_UPSAMPLER_STEP)
    return blocks


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


FLUX3_FAMILY_BLOCKS = {
    FLUX3_KLEIN_FAMILY_NAME: FLUX3_DISTILLED_BLOCKS,
    FLUX3_KLEIN_BASE_FAMILY_NAME: FLUX3_BASE_BLOCKS,
}
_BLOCKS_FAMILY = {blocks: family for family, blocks in FLUX3_FAMILY_BLOCKS.items()}
# A manifest's ``_class_name`` names the pipeline; either it or ``_blocks_class_name`` identifies the variant.
_PIPELINE_FAMILY = {
    "Flux3DistilledModularPipeline": FLUX3_KLEIN_FAMILY_NAME,
    "Flux3ModularPipeline": FLUX3_KLEIN_BASE_FAMILY_NAME,
}


def flux3_index_family(index: Optional[dict]) -> Optional[str]:
    """The FLUX.3 family a ``modular_model_index.json`` declares, or None when it says nothing recognisable."""
    if not isinstance(index, dict):
        return None
    blocks = index.get("_blocks_class_name")
    if isinstance(blocks, str) and blocks in _BLOCKS_FAMILY:
        return _BLOCKS_FAMILY[blocks]
    name = index.get("_class_name")
    return _PIPELINE_FAMILY.get(name) if isinstance(name, str) else None


def flux3_family_blocks_mismatch(
    family_name: Optional[str], declared: Optional[str], source: str
) -> Optional[str]:
    """Why ``source`` (declaring the FLUX.3 family or block class ``declared``) cannot load as ``family_name``."""
    declared = _BLOCKS_FAMILY.get(declared, declared) if declared else None
    if (
        family_name not in FLUX3_FAMILY_BLOCKS
        or declared not in FLUX3_FAMILY_BLOCKS
        or declared == family_name
    ):
        return None
    what = "distilled FLUX.3 Klein" if declared == FLUX3_KLEIN_FAMILY_NAME else "FLUX.3 Klein base"
    return (
        f"'{source}' is the {what} pipeline, but it was loaded as {family_name}. Load it with family {declared} "
        "(or leave the family on auto)."
    )


def flux3_blocks_class(index: dict, transformer: Any = None) -> str:
    """The block graph a checkpoint runs. The loaded denoiser decides first: a flow-map DiT (it has the target-time
    embedder) runs the distilled graph, a CFG DiT the base graph, whatever repo supplied the companions (a single
    file or GGUF of one variant may sit on the other variant's base). Else the index names it."""
    flow_map = None
    if transformer is not None:
        if hasattr(transformer, "target_timestep_embedder"):
            flow_map = getattr(transformer, "target_timestep_embedder") is not None
        else:
            flow_map = getattr(getattr(transformer, "config", None), "flow_map", None)
    if flow_map is not None:
        return FLUX3_DISTILLED_BLOCKS if flow_map else FLUX3_BASE_BLOCKS
    named = index.get("_blocks_class_name") if isinstance(index, dict) else None
    if named in (FLUX3_DISTILLED_BLOCKS, FLUX3_BASE_BLOCKS):
        return named
    return FLUX3_DISTILLED_BLOCKS


@dataclass
class Flux3PipelineOutput:
    """``.images``, the one field the generate path reads (``ImagePipelineOutput``'s shape)."""

    images: list


class _Flux3Interrupted(Exception):
    """Raised out of the denoise loop when the step callback set ``_interrupt``."""


# Each reference is scaled to this area before the VAE, aspect kept, Lanczos: ComfyUI's
# ``ReferenceMultiLatent`` ``["resize", 1]`` and BFL's multi-reference rule. Diffusers alone keeps a
# single reference up to 4 MiP (16 MiP before 0.42.0); the 16-px grid fit stays with the blocks.
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

        # The pinned build's FLUX.3 LoRA loader (transformer-only, like the FLUX.2 one): load_lora_weights,
        # set_adapters and unload_lora_weights act on self.transformer, the same module the blocks run.
        try:
            from diffusers.loaders import Flux3LoraLoaderMixin as lora
        except ImportError:
            lora = None
        bases = (
            (_Flux3KleinPipelineMixin,)
            + ((lora,) if lora is not None else ())
            + (DiffusionPipeline,)
        )
        _PIPELINE_CLASS = type("Flux3KleinPipeline", bases, {})
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
            self._modular = flux3_blocks(self.config.blocks_class_name).init_pipeline()
            _bridge_text_step(self, self._modular)
        mp = self._modular
        updates = {}
        for name in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler"):
            current = getattr(self, name, None)
            if current is not None and getattr(mp, name, None) is not current:
                updates[name] = current
        if updates:
            mp.update_components(**updates)
        return mp

    def encode_prompt(
        self,
        prompt: str,
        max_sequence_length: int = 8192,
        device: Any = None,
        pad_multiple: Optional[int] = None,
        hidden_states_layers: Any = None,
    ):
        """One prompt's FLUX.3 text embeddings via diffusers' text step helper; ``None`` takes the step's defaults."""
        import torch

        step, helper = _text_step(self.modular_pipeline())
        if step is None:
            raise RuntimeError("this diffusers build's FLUX.3 text step has no known prompt helper")
        if pad_multiple is None:
            pad_multiple = step.pad_multiple
        if hidden_states_layers is None:
            hidden_states_layers = next(
                (p.default for p in step.inputs if getattr(p, "name", None) == _TEXT_LAYERS_INPUT),
                None,
            )
            if hidden_states_layers is None:
                raise RuntimeError(
                    "this diffusers build's FLUX.3 text step names no default layers"
                )
        with torch.no_grad():
            return helper(
                text_encoder = self.text_encoder,
                tokenizer = self.tokenizer,
                prompt = prompt,
                device = device,
                max_sequence_length = int(max_sequence_length),
                pad_multiple = int(pad_multiple),
                hidden_states_layers = tuple(hidden_states_layers),
            )

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
        negatives = list(negative_prompt) if isinstance(negative_prompt, (list, tuple)) else None
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
            if negatives is not None and len(negatives) not in (1, count):
                raise ValueError(
                    f"Got {len(negatives)} negative prompts for {count} images; pass one, or one per image."
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
                    # A per-image list pairs entry i with call i (the blocks take one prompt per call).
                    kwargs["negative_prompt"] = (
                        negative_prompt
                        if negatives is None
                        else negatives[call_index if len(negatives) > 1 else 0]
                    )
                # The base model's text and reference streams run once per call, not once per step and branch.
                transformer = getattr(self, "transformer", None)
                if not self.is_distilled:
                    cond_blocks.begin_render(transformer)
                try:
                    with self._step_hooks(
                        mp,
                        callback_on_step_end,
                        num_inference_steps = int(num_inference_steps),
                        first_call = call_index == 0,
                    ):
                        out = mp(**kwargs, output = "images")
                finally:
                    cond_blocks.end_render(transformer)
                images.extend(out if isinstance(out, list) else [out])
        except _Flux3Interrupted:
            images = []
        if not return_dict:
            return (images,)
        return Flux3PipelineOutput(images = images)

    def _step_hooks(
        self, mp, callback: Optional[Callable], *, num_inference_steps: int, first_call: bool
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
            if (
                first_call
                and transformer is not None
                and hasattr(transformer, "register_forward_pre_hook")
            ):
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


def flux3_storage_dtype(target: Any, dtype: Any) -> Any:
    """bf16 when a CUDA load computes FLUX.3 in float32: keep bf16 weights and upcast per op instead of doubling
    the load (about 40 GB for klein). None otherwise."""
    from .diffusion_manual_cast import manual_cast_disabled

    if manual_cast_disabled() or getattr(target, "device", None) != "cuda":
        return None
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch, no load
        return None
    return torch.bfloat16 if dtype is torch.float32 else None


def _component_files(
    repo_id: str, subfolder: str, hf_token: Optional[str], local_files_only: bool
) -> list[Path]:
    """Safetensors files from_pretrained reads for ``subfolder``; empty when unresolvable."""
    from .diffusion_small_host import _weight_files

    root = Path(str(repo_id)).expanduser()
    if root.is_dir():
        sub = root / subfolder
        return _weight_files(sub) if sub.is_dir() else []
    from huggingface_hub import hf_hub_download

    kwargs: dict[str, Any] = {
        "token": hf_token or None,
        "local_files_only": local_files_only,
        "cache_dir": _live_cache_dir(),
    }
    stem = "model" if subfolder.startswith("text_encoder") else "diffusion_pytorch_model"
    try:
        index = Path(
            hf_hub_download(repo_id, f"{subfolder}/{stem}.safetensors.index.json", **kwargs)
        )
        shards = sorted(
            set(json.loads(index.read_text(encoding = "utf-8")).get("weight_map", {}).values())
        )
        return [Path(hf_hub_download(repo_id, f"{subfolder}/{s}", **kwargs)) for s in shards]
    except Exception:  # noqa: BLE001 - unsharded, or unreachable: try the single file
        pass
    try:
        return [Path(hf_hub_download(repo_id, f"{subfolder}/{stem}.safetensors", **kwargs))]
    except Exception:  # noqa: BLE001 - no claim: the caller keeps the widened load
        return []


def _stored_as(
    repo_id: str,
    subfolder: str,
    storage_dtype: Any,
    hf_token: Optional[str],
    local_files_only: bool,
) -> bool:
    """True when every float tensor of ``subfolder`` is stored as ``storage_dtype`` (lossless load)."""
    from .diffusion_small_host import _safetensors_bytes_by_dtype

    want = str(storage_dtype).replace("torch.", "")
    files = _component_files(repo_id, subfolder, hf_token, local_files_only)
    if not files:
        return False
    seen = False
    for f in files:
        sizes = _safetensors_bytes_by_dtype(f)
        if not sizes:
            return False
        for dtype in sizes:
            if dtype in ("i64", "i32", "i16", "i8", "u8", "bool"):
                continue
            if dtype != want:
                return False
            seen = True
    return seen


def _manual_cast_load(
    loader: Callable[[Any], Any],
    *,
    name: str,
    repo_id: str,
    dtype: Any,
    storage_dtype: Any,
    hf_token: Optional[str],
    local_files_only: bool,
) -> Any:
    """Load at ``storage_dtype`` with per-op upcast to ``dtype`` if stored exactly so; else the widened load."""
    if storage_dtype is None or not _stored_as(
        repo_id, name, storage_dtype, hf_token, local_files_only
    ):
        if storage_dtype is not None:
            logger.info(
                "diffusion.flux3: %s loads widened to %s (its stored weights are not all %s)",
                name,
                dtype,
                storage_dtype,
            )
        return loader(dtype)
    from .diffusion_manual_cast import apply_manual_cast_

    module = loader(storage_dtype)
    kept = apply_manual_cast_(module, dtype)
    logger.info(
        "diffusion.flux3: %s keeps %.2f GiB of %s weights as stored and computes in %s (manual cast)",
        name,
        kept / (1 << 30),
        str(storage_dtype).replace("torch.", ""),
        str(dtype).replace("torch.", ""),
    )
    return module


def _pin_compute_dtype(transformer: Any, dtype: Any) -> None:
    """Make ``transformer.dtype`` answer the compute dtype for a GGUF denoiser.

    The FLUX.3 denoise block casts the latents to ``components.transformer.dtype``. diffusers reads that as the first
    floating parameter outside ``_keep_in_fp32_modules``; in a GGUF FLUX.3 every 2-D weight is packed uint8 and the
    only float tensors are the fp32-kept QK norms, so it falls through to uint8 and the first Linear refuses a uint8
    input. A float buffer is what it reads next, so one empty non-persistent buffer at ``dtype`` fixes the answer
    without touching any weight."""
    try:
        import torch

        if getattr(transformer, "dtype", None) in (None, dtype) or not isinstance(
            transformer, torch.nn.Module
        ):
            return
        if getattr(transformer, "dtype").is_floating_point:
            return
        transformer.register_buffer(
            "_unsloth_compute_dtype", torch.empty(0, dtype = dtype), persistent = False
        )
    except Exception as exc:  # noqa: BLE001 -- the denoise raises its own clear error if this did not take
        logger.warning("diffusion.flux3: could not pin the GGUF compute dtype (%s)", exc)


def load_flux3_pipeline(
    repo_id: str,
    dtype,
    hf_token: Optional[str] = None,
    transformer = None,
    text_encoder = None,
    local_files_only: bool = False,
    check_cancelled: Optional[Callable[[], None]] = None,
    family_name: Optional[str] = None,
    storage_dtype: Any = None,
    transformer_storage: bool = True,
) -> Any:
    """A ready ``Flux3KleinPipeline`` for ``repo_id`` (on CPU; the caller places it).

    ``storage_dtype`` (``flux3_storage_dtype``): the text encoder, and the transformer unless ``transformer_storage``
    is False (a transformer quant will rebuild its Linears), load at that dtype and upcast per op to ``dtype``.

    ``family_name`` (the family the load resolved) must agree with the block graph the checkpoint runs: a base
    pipeline loaded as the distilled family would run CFG while Studio suppresses its negative prompt, and the
    reverse. The pre-eviction preflight refuses that from the manifest; this is the backstop.

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
        text_encoder = _manual_cast_load(
            lambda load_dtype: Qwen3VLForConditionalGeneration.from_pretrained(
                repo_id, subfolder = "text_encoder", dtype = load_dtype, **common
            ),
            name = "text_encoder",
            repo_id = repo_id,
            dtype = dtype,
            storage_dtype = storage_dtype,
            hf_token = token,
            local_files_only = local_files_only,
        )
        text_encoder.eval()
        check_cancelled()
    if transformer is None:
        transformer = _manual_cast_load(
            lambda load_dtype: diffusers.Flux3Transformer2DModel.from_pretrained(
                repo_id, subfolder = "transformer", torch_dtype = load_dtype, **common
            ),
            name = "transformer",
            repo_id = repo_id,
            dtype = dtype,
            storage_dtype = storage_dtype if transformer_storage else None,
            hf_token = token,
            local_files_only = local_files_only,
        )
        check_cancelled()
    vae = diffusers.AutoencoderKLFlux2.from_pretrained(
        repo_id, subfolder = "vae", torch_dtype = dtype, **common
    )
    check_cancelled()
    _pin_compute_dtype(transformer, dtype)
    blocks = flux3_blocks_class(index, transformer)
    reason = flux3_family_blocks_mismatch(family_name, blocks, repo_id)
    if reason is not None:
        raise ValueError(reason)
    # Before the speed layer: a whole-step CUDA graph binds the class forward when it is built.
    cond_blocks.prepare(transformer)
    # The scheduler is the block graph's own (shift 6.93, the reference timestep shift); the repos ship none.
    scheduler = flux3_blocks(blocks).init_pipeline().scheduler
    return flux3_pipeline_class()(
        transformer = transformer,
        vae = vae,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        scheduler = scheduler,
        blocks_class_name = blocks,
    )
