# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Anima (Cosmos-Predict2 DiT + ``AnimaTextConditioner`` over Qwen3-0.6B): diffusers ships modular blocks only."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

ANIMA_FAMILY_NAME = "anima"
ANIMA_PIPELINE_CLASS = "AnimaPipeline"
ANIMA_BLOCKS = "AnimaAutoBlocks"
ANIMA_INDEX = "modular_model_index.json"
ANIMA_COMPONENTS = (
    "scheduler",
    "t5_tokenizer",
    "text_conditioner",
    "text_encoder",
    "tokenizer",
    "transformer",
    "vae",
)
ANIMA_COMPONENT_FOLDERS = {
    "text_encoder": ("text_encoder", "transformers", "Qwen3Model"),
    "tokenizer": ("tokenizer", "transformers", "Qwen2Tokenizer"),
    "t5_tokenizer": ("t5_tokenizer", "transformers", "T5Tokenizer"),
    "text_conditioner": ("text_conditioner", "diffusers", "AnimaTextConditioner"),
    "transformer": ("transformer", "diffusers", "CosmosTransformer3DModel"),
    "vae": ("vae", "diffusers", "AutoencoderKLQwenImage"),
    "scheduler": ("scheduler", "diffusers", "FlowMatchEulerDiscreteScheduler"),
}

_CONTAINER_PREFIXES = ("model.diffusion_model.", "net.")
_ADAPTER_PREFIX = "llm_adapter."
# Bookkeeping tensors diffusers' Cosmos converter drops too.
_DROPPED = ("pos_embedder.", "accum_", "_extra_state")


def anima_available() -> bool:
    try:
        import diffusers
        return all(
            hasattr(diffusers, name)
            for name in ("AnimaAutoBlocks", "AnimaTextConditioner", "CosmosTransformer3DModel")
        )
    except Exception:  # noqa: BLE001 -- an unimportable diffusers just means no Anima here
        return False


def is_anima_family(fam: Any) -> bool:
    return getattr(fam, "name", None) == ANIMA_FAMILY_NAME


def split_anima_checkpoint(checkpoint: dict) -> tuple[dict, dict]:
    """``(transformer, text_conditioner)`` state dicts; a quantized file raises (codes would load as weights)."""
    from diffusers.loaders.single_file_utils import (
        convert_cosmos_transformer_checkpoint_to_diffusers,
    )

    transformer: dict = {}
    adapter: dict = {}
    for key, value in checkpoint.items():
        for prefix in _CONTAINER_PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                break
        if key.endswith((".comfy_quant", ".weight_scale", ".input_scale")):
            raise ValueError(
                "this Anima file is ComfyUI-quantised (int8 / fp8 codes); load its bf16 / fp16 version"
            )
        if key.startswith(_ADAPTER_PREFIX):
            adapter[key[len(_ADAPTER_PREFIX) :]] = value
        elif key.startswith(_DROPPED):
            continue
        else:
            transformer["net." + key] = value
    if not adapter:
        raise ValueError(
            "not an Anima checkpoint: it has no llm_adapter (text conditioner) weights"
        )
    return convert_cosmos_transformer_checkpoint_to_diffusers(transformer), adapter


def _block_count(state: dict) -> int:
    return len(
        {
            k.split(".", 2)[1]
            for k in state
            if k.startswith("transformer_blocks.") and k.split(".")[1].isdigit()
        }
    )


def _assign_on_meta(cls: Any, config: dict, state: dict, dtype) -> Any:
    """``cls`` on meta, ``state`` cast as ``from_pretrained`` casts: init buffers (rotary ``inv_freq``) stay fp32."""
    import torch
    from accelerate import init_empty_weights

    from .diffusion_single_file_converters import charge_cast

    keep = getattr(cls, "_keep_in_fp32_modules", None) or ()
    with init_empty_weights():
        model = cls.from_config(config)
    for key, value in list(state.items()):
        want = torch.float32 if any(m in key.split(".") for m in keep) else dtype
        if value.is_floating_point() and value.dtype != want:
            charge_cast(value, want)
            state[key] = value.to(want)
    model.load_state_dict(state, strict = True, assign = True)
    return model.eval()


def load_anima_single_file(
    path: str,
    config_source: str,
    dtype,
    *,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
) -> tuple[Any, Any]:
    """``(transformer, text_conditioner)`` on CPU; ``num_layers`` follows the file (40-block 2.9B, 28-block base)."""
    from diffusers import AnimaTextConditioner, CosmosTransformer3DModel
    from safetensors.torch import load_file

    from .diffusion_single_file_converters import HostRamBudget, converted_host_bytes

    common: dict[str, Any] = {"local_files_only": local_files_only}
    if cache_dir:
        common["cache_dir"] = cache_dir
    if hf_token:
        common["token"] = hf_token
    config = CosmosTransformer3DModel.load_config(config_source, subfolder = "transformer", **common)
    conditioner_config = AnimaTextConditioner.load_config(
        config_source, subfolder = "text_conditioner", **common
    )
    budget = HostRamBudget(
        path, f"casting its weights to {str(dtype).replace('torch.', '')}", logger
    )
    budget.check(converted_host_bytes(path, dtype))
    with budget:
        t_state, c_state = split_anima_checkpoint(load_file(path))
        blocks = _block_count(t_state)
        if blocks and blocks != config.get("num_layers"):
            logger.info(
                "diffusion.anima: %s has %d blocks (base config %s)",
                path,
                blocks,
                config.get("num_layers"),
            )
            config = {**config, "num_layers": blocks}
        transformer = _assign_on_meta(CosmosTransformer3DModel, config, t_state, dtype)
        conditioner = _assign_on_meta(AnimaTextConditioner, conditioner_config, c_state, dtype)
    return transformer, conditioner


@dataclass
class AnimaPipelineOutput:
    images: list


class _AnimaInterrupted(Exception):
    pass


_PIPELINE_CLASS: Optional[type] = None


def anima_pipeline_class() -> type:
    """``AnimaPipeline``, built on first use so importing this module never imports diffusers."""
    global _PIPELINE_CLASS
    if _PIPELINE_CLASS is None:
        from diffusers import DiffusionPipeline

        try:
            from diffusers.loaders import AnimaLoraLoaderMixin as lora
        except ImportError:
            lora = None
        bases = (
            (_AnimaPipelineMixin,) + ((lora,) if lora is not None else ()) + (DiffusionPipeline,)
        )
        _PIPELINE_CLASS = type(ANIMA_PIPELINE_CLASS, bases, {})
    return _PIPELINE_CLASS


_COMPONENTS = (
    "transformer",
    "text_conditioner",
    "text_encoder",
    "tokenizer",
    "t5_tokenizer",
    "vae",
    "scheduler",
)


class _AnimaPipelineMixin:
    model_cpu_offload_seq = "text_encoder->text_conditioner->transformer->vae"
    _optional_components: list = []
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self, transformer, text_conditioner, text_encoder, tokenizer, t5_tokenizer, vae, scheduler
    ):
        super().__init__()
        self.register_modules(
            transformer = transformer,
            text_conditioner = text_conditioner,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            t5_tokenizer = t5_tokenizer,
            vae = vae,
            scheduler = scheduler,
        )
        self._modular = None
        self._interrupt = False

    @property
    def interrupt(self) -> bool:
        return self._interrupt

    def modular_pipeline(self):
        import diffusers

        if self._modular is None:
            self._modular = getattr(diffusers, ANIMA_BLOCKS)().init_pipeline()
        mp = self._modular
        updates = {
            name: getattr(self, name)
            for name in _COMPONENTS
            if getattr(self, name, None) is not None
            and getattr(mp, name, None) is not getattr(self, name)
        }
        if updates:
            mp.update_components(**updates)
        return mp

    def _set_guidance(self, mp, guidance_scale: Optional[float]) -> None:
        if guidance_scale is None:
            return
        guider = getattr(mp, "guider", None)
        if guider is not None and float(getattr(guider.config, "guidance_scale", -1.0)) == float(
            guidance_scale
        ):
            return
        from diffusers import ClassifierFreeGuidance

        mp.update_components(guider = ClassifierFreeGuidance(guidance_scale = float(guidance_scale)))

    def __call__(
        self,
        prompt = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: Optional[float] = 4.0,
        negative_prompt = None,
        num_images_per_prompt: int = 1,
        generator = None,
        latents = None,
        output_type: str = "pil",
        return_dict: bool = True,
        callback_on_step_end: Optional[Callable] = None,
        callback_on_step_end_tensor_inputs = ("latents",),
    ):
        self._interrupt = False
        mp = self.modular_pipeline()
        self._set_guidance(mp, guidance_scale)
        prompts = list(prompt) if isinstance(prompt, (list, tuple)) else [prompt]
        if any(not isinstance(p, str) for p in prompts):
            raise ValueError("Anima takes a text prompt (str) per image.")
        generators = list(generator) if isinstance(generator, (list, tuple)) else None
        negatives = list(negative_prompt) if isinstance(negative_prompt, (list, tuple)) else None
        # One image per modular call, so each image keeps its own generator and negative prompt.
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
            for index, (one_prompt, one_generator, per_prompt) in enumerate(calls):
                kwargs: dict[str, Any] = {
                    "prompt": one_prompt,
                    "num_inference_steps": int(num_inference_steps),
                    "num_images_per_prompt": per_prompt,
                    "output_type": output_type,
                }
                if negative_prompt is not None:
                    kwargs["negative_prompt"] = (
                        negative_prompt
                        if negatives is None
                        else negatives[index if len(negatives) > 1 else 0]
                    )
                if one_generator is not None:
                    kwargs["generator"] = one_generator
                if height is not None:
                    kwargs["height"] = int(height)
                if width is not None:
                    kwargs["width"] = int(width)
                if latents is not None:
                    kwargs["latents"] = latents
                with self._step_hooks(
                    mp, callback_on_step_end, int(num_inference_steps), index == 0
                ):
                    out = mp(**kwargs, output = "images")
                images.extend(out if isinstance(out, list) else [out])
        except _AnimaInterrupted:
            images = []
        # The modular pipeline does not own the hooks enable_model_cpu_offload put on these components.
        self.maybe_free_model_hooks()
        if not return_dict:
            return (images,)
        return AnimaPipelineOutput(images = images)

    def _step_hooks(
        self, mp, callback: Optional[Callable], num_inference_steps: int, first_call: bool
    ):
        """``callback_on_step_end`` per scheduler step; ``progress_bar`` opened at the first denoiser forward."""
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
                    raise _AnimaInterrupted()
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
                    with contextlib.suppress(AttributeError):
                        del scheduler.step
                if handle is not None:
                    handle.remove()

        return _hooks()


def _load_tokenizer(auto_tokenizer: Any, repo_id: str, subfolder: str, common: dict) -> Any:
    from .diffusion_offline_source import offline_snapshot_source

    # Offline by repo id, transformers 5.x fails on cached tokenizers (see diffusion_offline_source).
    source = offline_snapshot_source(
        repo_id,
        subfolder,
        local_files_only = common.get("local_files_only", False),
        cache_dir = common.get("cache_dir"),
    )
    return auto_tokenizer.from_pretrained(source, subfolder = subfolder, **common)


def load_anima_pipeline(
    repo_id: str,
    dtype,
    *,
    hf_token: Optional[str] = None,
    transformer: Any = None,
    text_conditioner: Any = None,
    text_encoder: Any = None,
    vae: Any = None,
    local_files_only: bool = False,
    cache_dir: Optional[str] = None,
    check_cancelled: Optional[Callable[[], None]] = None,
) -> Any:
    """``AnimaPipeline`` from the diffusers repo / folder ``repo_id``, on CPU; handed-in components replace its own."""
    check_cancelled = check_cancelled or (lambda: None)
    check_cancelled()
    if not anima_available():
        raise RuntimeError(
            "Anima needs a diffusers build with the Anima modular blocks (0.39 or newer)."
        )
    import diffusers
    from transformers import AutoTokenizer, Qwen3Model

    common: dict[str, Any] = {"local_files_only": local_files_only}
    if cache_dir:
        common["cache_dir"] = cache_dir
    if hf_token:
        common["token"] = hf_token
    tokenizer, t5_tokenizer = (
        _load_tokenizer(AutoTokenizer, repo_id, subfolder, common)
        for subfolder in ("tokenizer", "t5_tokenizer")
    )
    check_cancelled()
    if text_encoder is None:
        text_encoder = Qwen3Model.from_pretrained(
            repo_id, subfolder = "text_encoder", dtype = dtype, **common
        )
        text_encoder.eval()
        check_cancelled()
    if transformer is None:
        transformer = diffusers.CosmosTransformer3DModel.from_pretrained(
            repo_id, subfolder = "transformer", torch_dtype = dtype, **common
        )
        check_cancelled()
    if text_conditioner is None:
        text_conditioner = diffusers.AnimaTextConditioner.from_pretrained(
            repo_id, subfolder = "text_conditioner", torch_dtype = dtype, **common
        )
        check_cancelled()
    if vae is None:
        vae = diffusers.AutoencoderKLQwenImage.from_pretrained(
            repo_id, subfolder = "vae", torch_dtype = dtype, **common
        )
        check_cancelled()
    scheduler = diffusers.FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo_id, subfolder = "scheduler", **common
    )
    logger.info("diffusion.anima: assembled pipeline from %s per component", repo_id)
    return anima_pipeline_class()(
        transformer = transformer,
        text_conditioner = text_conditioner,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        t5_tokenizer = t5_tokenizer,
        vae = vae,
        scheduler = scheduler,
    )
