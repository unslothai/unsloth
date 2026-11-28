# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.3 pipeline facade over Modular Diffusers blocks; reads only transformer/, text_encoder/, vae/."""

from __future__ import annotations

import contextvars
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


# Diffusers main puts this step first in both auto block graphs (huggingface/diffusers#15058); needs
# ~19 GB Qwen3.5-9B upsampler Studio never loads, and would run every call, so it's dropped.
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


class _Flux3RowsMismatch(Exception):
    """Batched text step did not match the rows: run them one by one."""


_ROW_TEXT = contextvars.ContextVar("unsloth_flux3_row_text", default = None)


class _RowText:
    """Rows of one batched call: text call 1 takes ``positive``, call 2 (unconditional branch) ``negative``."""

    __slots__ = ("positive", "negative", "encoded", "args", "calls", "references", "processed")

    def __init__(
        self,
        positive: list,
        negative: Optional[list],
        encoded: dict,
        args: tuple,
        references: Optional[list] = None,
    ) -> None:
        self.positive = positive
        self.negative = negative
        self.encoded = encoded
        self.args = args
        self.calls = 0
        # None: references shared by every row
        self.references = references
        self.processed = 0


def _row_text_embeds(owner: Any, rows: _RowText, a: dict) -> Any:
    import torch

    which = rows.calls
    rows.calls += 1
    if which == 0:
        prompts = rows.positive
        expected = prompts[0]
    elif which == 1 and rows.negative is not None:
        prompts = rows.negative
        expected = prompts[0] or ""
    else:
        raise _Flux3RowsMismatch(f"text step call {which + 1} has no rows")
    if a["prompt"] != expected:
        raise _Flux3RowsMismatch("the text step encoded a prompt that is not the batch's first row")
    args = (int(a["max_sequence_length"]), int(a["pad_multiple"]), tuple(a["hidden_states_layers"]))
    cache = rows.encoded if args == rows.args else {}
    distinct: dict = {}
    for prompt in prompts:
        key = prompt or ""
        if key not in distinct:
            embeds = cache.get(key)
            if embeds is None:
                embeds = owner.encode_prompt(
                    prompt = key,
                    max_sequence_length = args[0],
                    device = a["device"],
                    pad_multiple = args[1],
                    hidden_states_layers = args[2],
                )
            distinct[key] = embeds
    tensors = [distinct[prompt or ""] for prompt in prompts]
    if any(t.ndim != 3 or t.shape[0] != 1 or t.shape[1:] != tensors[0].shape[1:] for t in tensors):
        raise _Flux3RowsMismatch("rows encode to different text windows")
    return torch.cat(tensors, dim = 0)


def _text_defaults(step: Any, pad_multiple: Optional[int], hidden_states_layers: Any) -> tuple:
    if pad_multiple is None:
        pad_multiple = step.pad_multiple
    if hidden_states_layers is None:
        hidden_states_layers = next(
            (p.default for p in step.inputs if getattr(p, "name", None) == _TEXT_LAYERS_INPUT),
            None,
        )
        if hidden_states_layers is None:
            raise RuntimeError("this diffusers build's FLUX.3 text step names no default layers")
    return int(pad_multiple), tuple(hidden_states_layers)


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
        rows = _ROW_TEXT.get()
        if (
            owner is None
            or a["text_encoder"] is not getattr(owner, "text_encoder", None)
            or a["tokenizer"] is not getattr(owner, "tokenizer", None)
        ):
            if rows is not None:
                raise _Flux3RowsMismatch("a foreign text encoder cannot encode the batch's rows")
            return helper(*args, **kwargs)
        if rows is not None:
            return _row_text_embeds(owner, rows, a)
        return owner.encode_prompt(
            prompt = a["prompt"],
            max_sequence_length = a["max_sequence_length"],
            device = a["device"],
            pad_multiple = a["pad_multiple"],
            hidden_states_layers = a["hidden_states_layers"],
        )

    bridged._unsloth_row_bridge = True
    setattr(step, _TEXT_HELPER, bridged)
    return True


def _bridge_reference_preprocess(mp: Any) -> bool:
    """In a batched call with per-row references, ``preprocess`` returns every row's reference, each alone, stacked."""
    import weakref

    processor = getattr(mp, "image_processor", None)
    original = getattr(processor, "preprocess", None)
    if processor is None or not callable(original):
        return False
    if getattr(original, "_unsloth_row_bridge", False):
        return True
    processor_ref = weakref.ref(processor)
    own = type(processor).preprocess

    def bridged(image: Any, *args: Any, **kwargs: Any) -> Any:
        import torch

        rows = _ROW_TEXT.get()
        target = processor_ref()
        if rows is None or rows.references is None or target is None:
            return own(target, image, *args, **kwargs)
        index = rows.processed
        rows.processed += 1
        if index >= len(rows.references[0]) or image is not rows.references[0][index]:
            raise _Flux3RowsMismatch(
                "the image step preprocessed a reference that is not the first row's"
            )
        outs = [own(target, refs[index], *args, **kwargs) for refs in rows.references]
        tensors = [out[0] for out in outs]
        if any(t.shape != tensors[0].shape for t in tensors) or any(
            o[1] != outs[0][1] for o in outs
        ):
            raise _Flux3RowsMismatch("rows' references preprocess to different canvases")
        return torch.cat(tensors, dim = 0), outs[0][1]

    bridged._unsloth_row_bridge = True
    processor.preprocess = bridged
    return True


def flux3_blocks(blocks_class_name: str) -> Any:
    """The named FLUX.3 auto blocks without the prompt upsampler step (absent before diffusers#15058)."""
    import diffusers

    blocks = getattr(diffusers, blocks_class_name)()
    sub_blocks = getattr(blocks, "sub_blocks", None)
    if sub_blocks is not None and FLUX3_PROMPT_UPSAMPLER_STEP in sub_blocks:
        sub_blocks.pop(FLUX3_PROMPT_UPSAMPLER_STEP)
    return blocks


# BFL's reference shift (``alpha`` in src/flux3/pipeline.py), raised 6.93 -> 16 in flux-3-image-klein 62e90b9278;
# diffusers 0.42.1's block graph still uses 6.93. LPIPS vs BFL fell 0.095 -> 0.007 (distilled), 0.079 -> 0.026 (base).
FLUX3_REFERENCE_SHIFT = 16.0


def flux3_reference_scheduler(scheduler: Any) -> Any:
    """``scheduler`` at BFL's reference shift; unchanged if already there or it has no static ``shift``."""
    config = getattr(scheduler, "config", None)
    try:
        current = config.get("shift") if config is not None else None
        if current is None or float(current) == FLUX3_REFERENCE_SHIFT:
            return scheduler
        if config.get("use_dynamic_shifting", False):
            return scheduler
    except (AttributeError, TypeError, ValueError):
        return scheduler
    return type(scheduler).from_config(config, shift = FLUX3_REFERENCE_SHIFT)


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


def _read_index(
    repo_id: str,
    hf_token: Optional[str],
    local_files_only: bool,
    revision: Optional[str] = None,
) -> dict:
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
        **({"revision": revision} if revision else {}),
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
    """The loaded denoiser decides first: flow-map DiT (has target_timestep_embedder) runs distilled
    graph, CFG DiT the base, whatever repo gave companions; else the index names it."""
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


def _guider_runs_both(mp: Any) -> bool:
    """Both CFG branches on every step. A guidance interval answers False throughout: mixed prefix / no-prefix steps
    would change the CUDA graph key and the sealed scope would send the later key eager."""
    guider = getattr(mp, "guider", None)
    if guider is None or int(getattr(guider, "num_conditions", 1)) <= 1:
        return False
    config = getattr(guider, "config", None)
    try:
        start = float(getattr(config, "start", 0.0) or 0.0)
        stop = float(
            getattr(config, "stop", 1.0) if getattr(config, "stop", None) is not None else 1.0
        )
    except (TypeError, ValueError):
        return False
    return start <= 0.0 and stop >= 1.0


def _reset_guider_step(mp: Any) -> None:
    """Rewind the guider's step state. diffusers keeps the last call's step: after a call whose last steps ran past a
    Fast CFG stop, the next text step would read one condition and skip the negative embeddings its CFG steps need."""
    guider = getattr(mp, "guider", None)
    if guider is None:
        return
    guider._step = 0
    guider._num_inference_steps = None
    guider._timestep = None


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
            _bridge_reference_preprocess(self._modular)
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
        pad_multiple, layers = _text_defaults(step, pad_multiple, hidden_states_layers)
        from .diffusion_te_select import run_selected

        def call() -> Any:
            return helper(
                text_encoder = self.text_encoder,
                tokenizer = self.tokenizer,
                prompt = prompt,
                device = device,
                max_sequence_length = int(max_sequence_length),
                pad_multiple = int(pad_multiple),
                hidden_states_layers = layers,
            )

        with torch.no_grad():
            return run_selected(self.text_encoder, layers, call)

    def _set_guidance(
        self,
        mp,
        guidance_scale: Optional[float],
        guidance_stop: Optional[float] = None,
    ) -> None:
        """Install a guider for this call's scale and interval. ``guidance_stop`` < 1 drops the unconditional
        branch for the steps from that fraction on (diffusers ``ClassifierFreeGuidance(stop = ...)``)."""
        from .diffusion_cfg_batch import set_guided_step_probe

        guider = getattr(mp, "guider", None)
        stop = 1.0 if guidance_stop is None else float(guidance_stop)
        # CFG batching: the steps past the stop run one branch by design, not a missed second branch
        set_guided_step_probe(
            self,
            None if guider is None or stop >= 1.0 else (lambda: mp.guider.num_conditions > 1),
        )
        if guider is None:
            return
        current = getattr(guider.config, "guidance_scale", None)
        # No scale keeps the guider's own, but never an interval a previous Fast CFG render left on it.
        scale = guidance_scale if guidance_scale is not None else current
        if scale is not None and (
            current is None
            or float(current) != float(scale)
            or float(getattr(guider.config, "stop", 1.0)) != stop
        ):
            from diffusers import ClassifierFreeGuidance
            mp.update_components(
                guider = ClassifierFreeGuidance(guidance_scale = float(scale), stop = stop)
            )
        _reset_guider_step(mp)

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
        guidance_stop: Optional[float] = None,
        row_images: Optional[list] = None,
    ):
        """``row_images``: per-image references, instead of shared ``image``."""
        self._interrupt = False
        mp = self.modular_pipeline()
        if row_images is not None and image is not None:
            raise ValueError(
                "Pass either image (shared references) or row_images (per image), not both."
            )
        if guidance_stop is not None and not 0.0 < float(guidance_stop) <= 1.0:
            raise ValueError("guidance_stop must be in (0, 1].")
        self._set_guidance(mp, guidance_scale, None if self.is_distilled else guidance_stop)

        prompts = prompt if isinstance(prompt, (list, tuple)) else [prompt]
        if any(not isinstance(p, str) for p in prompts):
            raise ValueError("FLUX.3 takes a text prompt (str) per image.")
        generators = generator if isinstance(generator, (list, tuple)) else None
        negatives = list(negative_prompt) if isinstance(negative_prompt, (list, tuple)) else None
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
        if row_images is not None and (
            len(row_images) != len(calls) or any(per != 1 for _, _, per in calls)
        ):
            raise ValueError(
                f"Got {len(row_images)} per-image references for {len(calls)} images; pass one entry per image."
            )
        row_refs = (
            [
                [
                    resize_reference(ref)
                    for ref in (refs if isinstance(refs, (list, tuple)) else [refs])
                ]
                for refs in row_images
            ]
            if row_images is not None
            else None
        )

        def call_kwargs(call_index: int) -> dict:
            one_prompt, one_generator, per_prompt = calls[call_index]
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
            elif row_refs is not None and row_refs[call_index]:
                refs = row_refs[call_index]
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
                kwargs["negative_prompt"] = row_negative(call_index)
            return kwargs

        def row_negative(call_index: int) -> Any:
            if negatives is None:
                return negative_prompt
            return negatives[call_index if len(negatives) > 1 else 0]

        def run_modular(
            kwargs: dict,
            first: bool,
            rows: Optional[_RowText] = None,
        ) -> Any:
            transformer = getattr(self, "transformer", None)
            if not self.is_distilled:
                cond_blocks.begin_render(transformer, guided = lambda: _guider_runs_both(mp))
            token = _ROW_TEXT.set(rows) if rows is not None else None
            try:
                _reset_guider_step(mp)
                with self._step_hooks(
                    mp,
                    callback_on_step_end,
                    num_inference_steps = int(num_inference_steps),
                    first_call = first,
                ):
                    return mp(**kwargs, output = "images")
            finally:
                if token is not None:
                    _ROW_TEXT.reset(token)
                cond_blocks.end_render(transformer)

        def run_rows(group: list, first: bool) -> list:
            kwargs = call_kwargs(group[0])
            kwargs["num_images_per_prompt"] = 1
            row_generators = [calls[i][1] for i in group]
            if any(g is not None for g in row_generators):
                kwargs["generator"] = row_generators
            if height is not None and width is not None:
                kwargs["latents"] = torch.cat(
                    [
                        reference_noise(_generator_seed(calls[i][1]), int(height), int(width))
                        for i in group
                    ],
                    dim = 0,
                )
            negative_rows = None
            if not self.is_distilled:
                negative_rows = [row_negative(i) for i in group]
            rows = _RowText(
                [calls[i][0] for i in group],
                negative_rows,
                encoded,
                text_args,
                [row_refs[i] for i in group] if row_refs is not None else None,
            )
            out = run_modular(kwargs, first, rows)
            if isinstance(out, list):
                if len(out) != len(group):
                    raise RuntimeError(
                        f"a batched FLUX.3 call returned {len(out)} images for {len(group)} rows"
                    )
                return out
            if getattr(out, "shape", None) is None or int(out.shape[0]) != len(group):
                raise RuntimeError("a batched FLUX.3 call returned an unexpected output")
            return [out[i : i + 1] for i in range(len(group))]

        import torch

        groups, encoded, text_args = self._row_groups(
            mp,
            calls,
            height = height,
            width = width,
            latents = latents,
            max_sequence_length = int(max_sequence_length),
            negative = (lambda i: row_negative(i) or "") if not self.is_distilled else None,
            seeded_noise = generators is not None,
            references = (
                [tuple(ref.size for ref in refs) for refs in row_refs]
                if row_refs is not None
                else None
            ),
        )
        from .diffusion_step_skip import rewind_static_step_skip

        images: list = []
        trajectories = [0]

        def next_trajectory() -> None:
            # The engine resets the static skip schedule once per chunk; each modular call here is its own trajectory.
            if trajectories[0]:
                rewind_static_step_skip(self)
            trajectories[0] += 1

        try:
            by_row: dict = {}
            for group_index, group in enumerate(groups):
                first = group_index == 0
                if len(group) > 1:
                    try:
                        next_trajectory()
                        for i, img in zip(group, run_rows(group, first)):
                            by_row[i] = [img]
                        continue
                    except _Flux3RowsMismatch as exc:
                        logger.info("FLUX.3 rows render one by one: %s", exc)
                for i in group:
                    next_trajectory()
                    out = run_modular(call_kwargs(i), first and i == group[0])
                    by_row[i] = out if isinstance(out, list) else [out]
            for i in range(len(calls)):
                images.extend(by_row[i])
        except _Flux3Interrupted:
            images = []
        if not return_dict:
            return (images,)
        return Flux3PipelineOutput(images = images)

    def _row_groups(
        self,
        mp: Any,
        calls: list,
        *,
        height: Optional[int],
        width: Optional[int],
        latents: Any,
        max_sequence_length: int,
        negative: Optional[Callable[[int], str]],
        seeded_noise: bool,
        references: Optional[list] = None,
    ) -> tuple:
        """``(groups, encoded, text_args)``. Groups share text window (the transformer attends to padding),
        negative and reference sizes; anything that could couple rows renders every call alone."""
        alone = ([[i] for i in range(len(calls))], {}, None)
        if len(calls) < 2 or latents is not None or batch_rows_disabled():
            return alone
        if references is not None and not row_refs_batch_enabled():
            return alone
        if any(per_prompt != 1 for _, _, per_prompt in calls):
            return alone
        if not seeded_noise or any(_generator_seed(g) is None for _, g, _ in calls):
            return alone
        step, _helper = _text_step(mp)
        if step is None or not getattr(
            step.__dict__.get(_TEXT_HELPER), "_unsloth_row_bridge", False
        ):
            return alone
        if _per_call_scaled(getattr(self, "transformer", None)) is not None:
            return alone
        try:
            pad, layers = _text_defaults(step, None, None)
            device = mp._execution_device
            text_args = (int(max_sequence_length), pad, layers)
            encoded: dict = {}
            texts = [prompt for prompt, _, _ in calls]
            if negative is not None and getattr(mp, "requires_unconditional_embeds", False):
                # negatives too, so a later group never reloads the encoder
                texts += [negative(i) for i in range(len(calls))]
            for prompt in texts:
                if prompt not in encoded:
                    encoded[prompt] = self.encode_prompt(
                        prompt = prompt,
                        max_sequence_length = int(max_sequence_length),
                        device = device,
                        pad_multiple = pad,
                        hidden_states_layers = layers,
                    )
        except Exception as exc:  # noqa: BLE001 -- the per-row calls encode (and report) on their own
            logger.info("FLUX.3 rows render one by one (encode: %s)", exc)
            return alone
        keyed: dict = {}
        for i, (prompt, _, _) in enumerate(calls):
            embeds = encoded[prompt]
            key = (
                tuple(embeds.shape[1:]),
                None if negative is None else negative(i),
                None if references is None else references[i],
            )
            keyed.setdefault(key, []).append(i)
        return list(keyed.values()), encoded, text_args

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


BATCH_ROWS_ENV = "UNSLOTH_DIFFUSION_FLUX3_BATCH"


def flux3_row_key(
    pipe: Any,
    prompt: str,
    max_sequence_length: int = 8192,
) -> Optional[tuple]:
    """A row's batch key: its padded chat-templated text window, or None if unmeasurable."""
    import math
    try:
        step, _helper = _text_step(pipe.modular_pipeline())
        processor = getattr(pipe, "tokenizer", None)
        if step is None or processor is None:
            return None
        pad = int(step.pad_multiple)
        text = processor.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize = False, add_generation_prompt = True
        )
        real = len(
            processor.tokenizer(text, truncation = True, max_length = int(max_sequence_length))[
                "input_ids"
            ]
        )
        return ("flux3 window", min(math.ceil(real / pad) * pad, int(max_sequence_length)))
    except Exception:  # noqa: BLE001 - unmeasurable: no key
        return None


def batch_rows_disabled() -> bool:
    import os
    return (os.environ.get(BATCH_ROWS_ENV) or "").strip().lower() in ("0", "off", "false", "no")


# Per-row references render one by one until GPU-proven neighbour-independent; ``=1`` batches them (to measure).
ROW_REFS_BATCH_ENV = "UNSLOTH_DIFFUSION_FLUX3_ROW_REFS_BATCH"


def row_refs_batch_enabled() -> bool:
    import os
    return (os.environ.get(ROW_REFS_BATCH_ENV) or "").strip().lower() in ("1", "on", "true", "yes")


def _per_call_scaled(module: Any) -> Optional[str]:
    """The denoiser's first layer whose activation scale comes from the whole call (cached), or None."""
    if module is None or not hasattr(module, "named_modules"):
        return None
    cached = module.__dict__.get("_unsloth_per_call_scaled")
    if cached is None:
        from .diffusion_cfg_batch import per_call_scaled
        try:
            cached = (per_call_scaled(module),)
        except Exception:  # noqa: BLE001 - unreadable: rows render apart rather than batch blind
            cached = ("unreadable layers",)
        module.__dict__["_unsloth_per_call_scaled"] = cached
    return cached[0]


HALF_STORAGE_ENV = "UNSLOTH_DIFFUSION_FLUX3_HALF_STORAGE"


def half_storage_disabled() -> bool:
    """``UNSLOTH_DIFFUSION_FLUX3_HALF_STORAGE=0``: a float16 load widens every component on the host again."""
    import os
    return (os.environ.get(HALF_STORAGE_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def flux3_storage_dtype(target: Any, dtype: Any) -> Any:
    """bf16 for a CUDA fp32/fp16 load: mmap stored bf16 and cast per op (exact); else None."""
    from .diffusion_manual_cast import manual_cast_disabled

    if manual_cast_disabled() or getattr(target, "device", None) != "cuda":
        return None
    try:
        import torch
    except Exception:  # noqa: BLE001 - no torch, no load
        return None
    if dtype is torch.float32:
        return torch.bfloat16
    if dtype is torch.float16 and not half_storage_disabled():
        return torch.bfloat16
    return None


def storage_saves_memory(storage_dtype: Any, dtype: Any) -> bool:
    try:
        import torch
        return (
            torch.empty(0, dtype = dtype).element_size()
            > torch.empty(0, dtype = storage_dtype).element_size()
        )
    except Exception:  # noqa: BLE001 - not dtypes: no claim
        return False


def transformer_storage_dtype(
    storage_dtype: Any, dtype: Any, transformer_storage: Optional[bool]
) -> Any:
    """Transformer load dtype: storage when True, or (None) only when it saves memory."""
    if storage_dtype is None or transformer_storage is False:
        return None
    if transformer_storage or storage_saves_memory(storage_dtype, dtype):
        return storage_dtype
    return None


# "0": always widen a dense fp16 transformer on host. Conversion peaks ~1.6x stored (klein ~18 GB, killed a 12.7 GB T4).
DENSE_STORAGE_ENV = "UNSLOTH_DIFFUSION_FLUX3_DENSE_STORAGE"
_WIDEN_PEAK = 1.6
_FLOAT_KEYS = frozenset({"bfloat16", "float16", "float32", "f8_e4m3", "f8_e5m2"})


def dense_storage_disabled() -> bool:
    import os
    return (os.environ.get(DENSE_STORAGE_ENV) or "").strip().lower() in ("0", "off", "false", "no")


def _stored_float_mib(files: list) -> int:
    from .diffusion_small_host import _safetensors_bytes_by_dtype

    total = 0
    for f in files:
        for dtype, nbytes in (_safetensors_bytes_by_dtype(f) or {}).items():
            if dtype in _FLOAT_KEYS:
                total += int(nbytes)
    return total >> 20


def widened_transformer_exceeds_host(
    repo_id: str,
    hf_token: Optional[str],
    local_files_only: bool,
    revision: Optional[str] = None,
) -> Optional[str]:
    """Reason a widened transformer load would not fit host RAM, else None (unreadable sizes keep widening)."""
    if dense_storage_disabled():
        return None
    from .diffusion_small_host import host_ram_check_disabled, host_ram_mib, host_reserve_mib

    if host_ram_check_disabled():
        return None
    total, available = host_ram_mib()
    if total is None or available is None:
        return None
    files = _component_files(repo_id, "transformer", hf_token, local_files_only, revision = revision)
    stored = _stored_float_mib(files) if files else 0
    if stored <= 0:
        return None
    need = int(stored * _WIDEN_PEAK) + host_reserve_mib(total)
    if need <= available:
        return None
    return f"a widened load needs ~{need} MiB of host RAM ({stored} MiB stored), {available} MiB available"


def transformer_load_storage(
    storage_dtype: Any,
    dtype: Any,
    transformer_storage: Optional[bool],
    repo_id: str,
    hf_token: Optional[str],
    local_files_only: bool,
    revision: Optional[str] = None,
) -> Any:
    """``transformer_storage_dtype``, but by default a same-width transformer reads as stored when widening won't fit."""
    picked = transformer_storage_dtype(storage_dtype, dtype, transformer_storage)
    if picked is not None or transformer_storage is not None or storage_dtype is None:
        return picked
    why = widened_transformer_exceeds_host(repo_id, hf_token, local_files_only, revision = revision)
    if why is None:
        return None
    logger.info("diffusion.flux3: transformer reads as stored: %s", why)
    return storage_dtype


def _component_files(
    repo_id: str,
    subfolder: str,
    hf_token: Optional[str],
    local_files_only: bool,
    revision: Optional[str] = None,
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
    if revision:
        kwargs["revision"] = revision
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
    revision: Optional[str] = None,
) -> bool:
    """True when every float tensor of ``subfolder`` is stored as ``storage_dtype`` (lossless load)."""
    from .diffusion_small_host import _safetensors_bytes_by_dtype

    want = str(storage_dtype).replace("torch.", "")
    files = _component_files(repo_id, subfolder, hf_token, local_files_only, revision = revision)
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
    revision: Optional[str] = None,
) -> Any:
    """Load at ``storage_dtype`` with per-op upcast to ``dtype`` if stored exactly so; else the widened load."""
    if storage_dtype is None or not _stored_as(
        repo_id, name, storage_dtype, hf_token, local_files_only, revision = revision
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
    transformer_storage: Optional[bool] = None,
    with_transformer: bool = True,
) -> Any:
    """A ready ``Flux3KleinPipeline`` for ``repo_id`` (on CPU; the caller places it).

    ``with_transformer=False`` (the LoRA trainer's conditioning phase): only the text encoder, processor and VAE load,
    and the block graph comes from the manifest; ``pipe.transformer`` is None.

    ``storage_dtype`` (``flux3_storage_dtype``): the text encoder loads at that dtype and casts per op to ``dtype``.
    The transformer does too when ``transformer_storage`` is True (a native quant rebuilds its Linears from the
    stored weights, so a same-width float16 conversion would only be a host copy thrown away). None (default): only
    when ``dtype`` is wider than the storage (float32), since a dense same-width transformer would cast every step.
    False: never.

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
    from . import diffusion_hub_snapshot as snap

    # One Hub call: an unchanged, already validated snapshot reads every component from the cache.
    components = {
        "loader": "flux3",
        "text_encoder": text_encoder is None,
        "transformer": transformer is None and with_transformer,
        "dtype": str(dtype),
        "storage_dtype": str(storage_dtype),
        "transformer_storage": transformer_storage,
    }
    # The validated commit: every read below pins it, so a concurrent download that moves the cache's refs/main
    # mid-load, or a Hub repair of one missing component after main moved, cannot mix in another commit's files.
    commit = (
        None
        if local_files_only
        else snap.current_snapshot(
            repo_id, components, token = token, cache_dir = common["cache_dir"], logger = logger
        )
    )
    cached = commit is not None

    def cache_first(load: Callable[[bool], Any], *, name: str) -> Any:
        return snap.cache_first(
            load,
            cached = cached,
            name = name,
            logger = logger,
            on_fallback = lambda: snap.forget(repo_id, components, cache_dir = common["cache_dir"]),
        )

    def _kw(local: bool) -> dict[str, Any]:
        kw = dict(common, local_files_only = local_files_only or local)
        if commit:
            kw["revision"] = commit
        return kw

    index = cache_first(
        lambda local: _read_index(repo_id, token, local_files_only or local, revision = commit),
        name = FLUX3_INDEX,
    )
    check_cancelled()
    tokenizer = cache_first(
        lambda local: AutoProcessor.from_pretrained(
            repo_id, subfolder = "text_encoder", **_kw(local)
        ),
        name = "the processor",
    )
    check_cancelled()
    from .diffusion_te_vision import drop_enabled, drop_vision_tower, vision_tower_skipped

    # No FLUX.3 mode hands the encoder pixels (references go through the VAE): its vision tower is never read.
    drop_vision = drop_enabled()

    def _te_load(load_dtype: Any, local: bool) -> Any:
        with vision_tower_skipped(drop_vision):
            return Qwen3VLForConditionalGeneration.from_pretrained(
                repo_id, subfolder = "text_encoder", dtype = load_dtype, **_kw(local)
            )

    if text_encoder is None:
        text_encoder = cache_first(
            lambda local: _manual_cast_load(
                lambda load_dtype: _te_load(load_dtype, local),
                name = "text_encoder",
                repo_id = repo_id,
                dtype = dtype,
                storage_dtype = storage_dtype,
                hf_token = token,
                local_files_only = local_files_only or local,
                revision = commit,
            ),
            name = "the text encoder",
        )
        text_encoder.eval()
        check_cancelled()
    if drop_vision and drop_vision_tower(text_encoder):
        logger.info("diffusion.flux3: the text encoder's unused vision tower is not loaded")
    if transformer is None and with_transformer:

        def _transformer(local: bool) -> Any:
            transformer_storage_at = transformer_load_storage(
                storage_dtype,
                dtype,
                transformer_storage,
                repo_id,
                token,
                local_files_only or local,
                revision = commit,
            )
            return _manual_cast_load(
                lambda load_dtype: diffusers.Flux3Transformer2DModel.from_pretrained(
                    repo_id, subfolder = "transformer", torch_dtype = load_dtype, **_kw(local)
                ),
                name = "transformer",
                repo_id = repo_id,
                dtype = dtype,
                storage_dtype = transformer_storage_at,
                hf_token = token,
                local_files_only = local_files_only or local,
                revision = commit,
            )

        transformer = cache_first(_transformer, name = "the transformer")
        check_cancelled()
    vae = cache_first(
        lambda local: diffusers.AutoencoderKLFlux2.from_pretrained(
            repo_id, subfolder = "vae", torch_dtype = dtype, **_kw(local)
        ),
        name = "the VAE",
    )
    check_cancelled()
    if not cached and not local_files_only:
        # This load revalidated every file it read: the next one may read them from the cache.
        snap.note_validated(repo_id, components, token = token, cache_dir = common["cache_dir"])
    _pin_compute_dtype(transformer, dtype)
    blocks = flux3_blocks_class(index, transformer)
    reason = flux3_family_blocks_mismatch(family_name, blocks, repo_id)
    if reason is not None:
        raise ValueError(reason)
    # Before the speed layer: a whole-step CUDA graph binds the class forward when it is built.
    if transformer is not None:
        cond_blocks.prepare(transformer)
    # The repos ship no scheduler; use the block graph's at BFL's reference shift.
    scheduler = flux3_reference_scheduler(flux3_blocks(blocks).init_pipeline().scheduler)
    return flux3_pipeline_class()(
        transformer = transformer,
        vae = vae,
        text_encoder = text_encoder,
        tokenizer = tokenizer,
        scheduler = scheduler,
        blocks_class_name = blocks,
    )
