# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared, family-agnostic building blocks for diffusion LoRA training.

Everything here is independent of a specific model architecture, so the SDXL U-Net
trainer (``diffusion_lora_trainer``) and the flow-matching DiT trainers share it:
dataset discovery, the request config + validation, image loading, event emission,
the stop protocol, adapter publishing, and the family/trainer registry.

The pure helpers have no torch/diffusers import at call time and are unit-tested
without a GPU. ``run_diffusion_lora_training`` and its per-family siblings own the
actual training loop; this module only routes a request to the right one.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Optional

from core.inference.diffusion_families import (
    detect_family,
    detect_family_for_pick,
    supported_family_names,
    trainable_family_names,
)

# Default LoRA target modules: the attention projections common to the SDXL U-Net and the
# DiT transformers (the diffusers/kohya convention). A family whose trainer wants a wider
# set overrides this in its own defaults; kept here so DiffusionLoraConfig has a sane fallback.
DEFAULT_LORA_TARGETS: tuple[str, ...] = ("to_k", "to_q", "to_v", "to_out.0")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_CAPTION_EXTS = (".txt", ".caption")
# diffusers' canonical single-file LoRA name, so load_lora_weights(dir) finds it.
DEFAULT_LORA_FILENAME = "pytorch_lora_weights.safetensors"

# Architectures Studio can neither train nor even load, so they are not in the family
# registry, but are recognisable by name. Rejecting them by name turns a confusing
# mid-run crash into an immediate, clear error. Families that ARE in the registry
# (flux / qwen-image / z-image / kontext) are handled by the positive registry check in
# ``resolve_trainable_family`` instead, so they are intentionally absent here.
_NON_TRAINABLE_RESIDUAL_TOKENS = frozenset({"sd3", "pixart", "sana", "lumina", "cogview"})
_NON_TRAINABLE_RESIDUAL_PHRASES = ("stable-diffusion-3", "hunyuan-dit")

EventCb = Callable[[dict[str, Any]], None]
# Returns a falsy value to keep training, or a truthy stop signal: bare True, or a dict
# that may carry ``save=False`` to cancel without saving a partial adapter.
StopCb = Callable[[], Any]


def _trainable_hint() -> str:
    """A user-facing hint listing the families Studio can train today. Always names SDXL
    explicitly so the message is actionable even as more families become trainable."""
    names = ", ".join(trainable_family_names()) or "sdxl"
    return (
        f"Trainable families right now: {names} "
        f"(for example the SDXL base stabilityai/stable-diffusion-xl-base-1.0). "
        f"Other families can load LoRAs but not train them yet."
    )


def resolve_trainable_family(base_model: str, model_family: Optional[str] = None) -> str:
    """Resolve the trainer family for a base model, or raise ValueError with a clear reason.

    Positive resolution, before anything is downloaded:
    - a ``.gguf`` name can never be a training base -> reject;
    - an explicit ``model_family`` must name a registry family, and that family must be
      trainable;
    - otherwise the family is detected from the base-model name; a KNOWN but non-trainable
      family (e.g. a DiT family before its trainer ships) is rejected;
    - a name that resolves to no registry family but matches a known non-trainable
      architecture (SD3 / PixArt / ...) is rejected;
    - an unclassifiable custom name/path falls through to the SDXL trainer (backwards
      compatible: a genuinely wrong pick still fails cleanly later in from_pretrained).
    """
    name = str(base_model or "").strip().lower()
    # GGUF weights (a ``.gguf`` file or a ``*-GGUF`` repo) are inference-only: training needs
    # the full diffusers pipeline (transformer + VAE + text encoders), which a GGUF repo does
    # not provide. Reject by name even when the family itself is trainable.
    if name.endswith(".gguf") or "gguf" in name:
        raise ValueError(
            f"'{base_model}' is a GGUF checkpoint/repo, which can't be a training base "
            f"(training needs the full diffusers model). {_trainable_hint()}"
        )
    if model_family and str(model_family).strip():
        key = str(model_family).strip().lower()
        fam = detect_family("", override = key)
        if fam is None:
            known = ", ".join(supported_family_names())
            raise ValueError(f"Unknown model_family {model_family!r}. Known families: {known}.")
        if not fam.trainable:
            raise ValueError(f"'{fam.name}' models can't be trained yet. {_trainable_hint()}")
        return fam.name

    fam = detect_family_for_pick(base_model)
    if fam is not None:
        if not fam.trainable:
            raise ValueError(
                f"'{base_model}' looks like a {fam.name} model, which isn't trainable yet. "
                f"{_trainable_hint()}"
            )
        return fam.name

    condensed = re.sub(r"[^a-z0-9]+", "-", name)
    hit = next(
        (p for p in _NON_TRAINABLE_RESIDUAL_PHRASES if p in condensed),
        None,
    ) or next(
        (t for t in condensed.split("-") if t in _NON_TRAINABLE_RESIDUAL_TOKENS),
        None,
    )
    if hit:
        raise ValueError(
            f"'{base_model}' looks like a {hit} model, which isn't trainable. {_trainable_hint()}"
        )
    # Unknown custom name / local path: default to the SDXL trainer (unchanged behaviour).
    return "sdxl"


def get_trainer(family: str) -> Callable[..., str]:
    """Return the training entrypoint for ``family``. Imports the trainer module lazily so
    this shared module stays free of the heavy trainer imports (and any import cycle)."""
    key = (family or "sdxl").strip().lower()
    if key == "sdxl":
        from core.training.diffusion_lora_trainer import run_diffusion_lora_training
        return run_diffusion_lora_training
    if key in ("flux.1", "qwen-image", "z-image"):
        from core.training.diffusion_dit_trainer import run_dit_lora_training
        return run_dit_lora_training
    raise ValueError(f"No trainer is registered for family {family!r}.")


# Per-family training defaults surfaced by the Train UI. Distilled/turbo bases and the big
# DiTs want different rank / learning rate / resolution; these are starting points, not
# hard limits. Families absent here fall back to the DiffusionLoraConfig defaults.
FAMILY_TRAIN_DEFAULTS: dict[str, dict[str, Any]] = {
    "sdxl": {"lora_rank": 16, "learning_rate": 1e-4, "resolution": 1024},
    "flux.1": {"lora_rank": 16, "learning_rate": 1e-4, "resolution": 512},
    "qwen-image": {"lora_rank": 16, "learning_rate": 5e-5, "resolution": 512},
    "z-image": {"lora_rank": 16, "learning_rate": 1e-4, "resolution": 768},
}


def train_defaults(family: str) -> dict[str, Any]:
    """Recommended starting hyperparameters for ``family`` (empty if unknown)."""
    return dict(FAMILY_TRAIN_DEFAULTS.get((family or "").strip().lower(), {}))


# Display labels + a short VRAM/access note per trainable family, surfaced by the Train UI
# so users pick a base with realistic expectations. Kept next to the defaults they pair with.
_FAMILY_LABELS = {
    "sdxl": "SDXL",
    "flux.1": "FLUX.1-dev",
    "qwen-image": "Qwen-Image",
    "z-image": "Z-Image",
}
_FAMILY_VRAM_NOTES = {
    "sdxl": "Trains on ~12 GB+ (bf16 LoRA). The lightest, fastest option.",
    "flux.1": (
        "12B model, QLoRA (nf4) by default (~16 GB+). Gated on Hugging Face: accept the "
        "FLUX.1-dev license and add your HF token before training."
    ),
    "qwen-image": "20B model, QLoRA (nf4) by default (~24 GB+). The heaviest option.",
    "z-image": "6B model, QLoRA (nf4) by default (~12 GB+). bf16 only.",
}


def family_train_infos() -> list[dict[str, Any]]:
    """Describe every trainable family for the Train UI: name, label, the default + allowed
    base repos, the recommended starting hyperparameters, and a VRAM/access note. Built from
    the family registry so it stays in sync with what the trainers actually support."""
    from core.inference.diffusion_families import detect_family

    infos: list[dict[str, Any]] = []
    for name in trainable_family_names():
        fam = detect_family("", override = name)
        if fam is None:
            continue
        repos = list(fam.train_base_repos) or [fam.base_repo]
        infos.append(
            {
                "name": name,
                "label": _FAMILY_LABELS.get(name, name),
                "default_base": repos[0],
                "base_repos": repos,
                "defaults": train_defaults(name),
                "vram_note": _FAMILY_VRAM_NOTES.get(name, ""),
            }
        )
    return infos


@dataclass
class DiffusionLoraConfig:
    """Everything a diffusion LoRA training run needs. Only ``base_model`` /
    ``data_dir`` / ``output_dir`` are required; the rest have sensible defaults."""

    base_model: str
    data_dir: str
    output_dir: str
    # Dreambooth-style caption applied to any image without its own caption. Required if
    # the dataset has no captions.jsonl / sidecar files.
    instance_prompt: Optional[str] = None
    resolution: int = 1024
    train_steps: int = 500
    learning_rate: float = 1e-4
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    lora_rank: int = 16
    lora_alpha: Optional[int] = None  # defaults to lora_rank
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = DEFAULT_LORA_TARGETS
    seed: int = 42
    mixed_precision: str = "bf16"  # "bf16" | "fp16" | "no"
    snr_gamma: Optional[float] = 5.0  # min-SNR loss weighting; None disables
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    center_crop: bool = False
    random_flip: bool = True
    caption_column: str = "text"  # column in metadata.jsonl
    adapter_name: str = "default"
    hf_token: Optional[str] = None
    # How often to emit a progress event (in optimizer steps).
    log_every: int = 1
    # Optional explicit family override ("sdxl" / "flux.1" / ...); None = detect from
    # base_model. ``resolved_family`` is filled by normalized() with the trainer family
    # that will actually run, so the process adapter can dispatch through the registry.
    model_family: Optional[str] = None
    resolved_family: str = "sdxl"

    def normalized(self) -> "DiffusionLoraConfig":
        """Return a copy with derived/validated fields filled in. Raises ValueError on a
        request that cannot train (bad numbers, or an untrainable base model).

        Also coerces values that arrive as strings/blanks through the Studio config path
        (``learning_rate`` is preserved as a string there; ``hf_token`` defaults to "")."""
        resolved_family = resolve_trainable_family(self.base_model, self.model_family)
        if self.train_steps < 1:
            raise ValueError("train_steps must be >= 1")
        if self.train_batch_size < 1:
            raise ValueError("train_batch_size must be >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.lora_alpha is not None and self.lora_alpha < 1:
            raise ValueError(
                "lora_alpha must be >= 1 (a zero/negative alpha scales the adapter to nothing)"
            )
        if self.resolution < 64 or self.resolution % 8 != 0:
            raise ValueError("resolution must be a multiple of 8 and >= 64")
        if self.mixed_precision not in ("bf16", "fp16", "no"):
            raise ValueError("mixed_precision must be one of bf16 / fp16 / no")
        # learning_rate can arrive as a string ("1e-4") from the Studio config path, which
        # preserves it as a string after validation; coerce so AdamW receives a float.
        try:
            learning_rate = float(self.learning_rate)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"learning_rate must be a number, got {self.learning_rate!r}") from exc
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        alpha = self.lora_alpha if self.lora_alpha is not None else self.lora_rank
        targets = tuple(self.lora_target_modules) or DEFAULT_LORA_TARGETS
        # A blank Hub token (the Studio default when none is configured) must load
        # anonymously, not as an explicit empty credential.
        token = self.hf_token.strip() if isinstance(self.hf_token, str) else self.hf_token
        return replace(
            self,
            learning_rate = learning_rate,
            lora_alpha = alpha,
            lora_target_modules = targets,
            max_grad_norm = float(self.max_grad_norm),
            hf_token = token or None,
            resolved_family = resolved_family,
        )


def discover_image_caption_pairs(
    data_dir: str | os.PathLike[str],
    *,
    instance_prompt: Optional[str] = None,
    caption_column: str = "text",
) -> list[tuple[str, str]]:
    """Resolve ``(image_path, caption)`` pairs from a dataset directory.

    Caption sources, in priority order per image:
      1. a ``metadata.jsonl`` / ``captions.jsonl`` row keyed by ``file_name`` (or ``image``)
         carrying the caption in ``caption_column`` (default ``text``),
      2. a per-image sidecar ``<stem>.txt`` / ``<stem>.caption``,
      3. ``instance_prompt`` (dreambooth) for any remaining image.

    Images with no caption from any source are skipped. Pure filesystem + JSON, so it is
    unit-testable without torch. Raises FileNotFoundError for a missing dir and ValueError
    when nothing is captionable.
    """
    root = Path(data_dir).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"data_dir is not a directory: {data_dir}")

    images = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)

    # 1. metadata.jsonl / captions.jsonl (either name accepted).
    meta_caption: dict[str, str] = {}
    for meta_name in ("metadata.jsonl", "captions.jsonl"):
        meta_path = root / meta_name
        if not meta_path.is_file():
            continue
        for line in meta_path.read_text(encoding = "utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = row.get("file_name") or row.get("image") or row.get("file")
            if key and caption_column in row:
                meta_caption[str(key)] = str(row[caption_column])

    pairs: list[tuple[str, str]] = []
    for img in images:
        caption: Optional[str] = None
        # 1. metadata row keyed by file name (basename or the name as written).
        caption = meta_caption.get(img.name) or meta_caption.get(str(img.relative_to(root)))
        # 2. per-image sidecar caption file.
        if caption is None:
            for ext in _CAPTION_EXTS:
                sidecar = img.with_suffix(ext)
                if sidecar.is_file():
                    caption = sidecar.read_text(encoding = "utf-8").strip()
                    break
        # 3. dreambooth instance prompt.
        if caption is None and instance_prompt:
            caption = instance_prompt
        if caption:
            pairs.append((str(img), caption))

    if not pairs:
        raise ValueError(
            "No captioned images found. Provide a metadata.jsonl / captions.jsonl, per-image "
            ".txt captions, or an instance prompt."
        )
    return pairs


def _emit(on_event: Optional[EventCb], type_: str, **kw: Any) -> None:
    if on_event is not None:
        on_event({"type": type_, "ts": time.time(), **kw})


def _assert_trusted_base_model(base_model: str) -> None:
    """Gate the training base model the same way the inference backend gates non-GGUF loads:
    a local path or a trusted repo (``unsloth/*`` or an allowlisted official base). This runs
    BEFORE ``from_pretrained`` so an untrusted remote repo (which could ship pickle weights)
    is never fetched or deserialised."""
    from core.inference.diffusion import _is_trusted_diffusion_repo
    if not _is_trusted_diffusion_repo(base_model):
        raise ValueError(
            f"Refusing to train from untrusted base model '{base_model}'. Use a local path or "
            f"a trusted repo (an unsloth/* repo or an official base)."
        )


def _publish_to_lora_catalog(lora_path: str, cfg: DiffusionLoraConfig) -> Optional[str]:
    """Best-effort copy of the trained adapter into the Studio diffusion LoRA directory so
    the Images LoRA picker (which scans only files directly under ``loras/diffusion``) finds
    it without the user moving files. Also writes a ``<alias>.json`` metadata sidecar so the
    picker can family-gate the adapter (family, base model, trigger prompt, ...). Returns the
    published path, or None on any failure."""
    try:
        import shutil

        from core.inference.diffusion_lora import loras_dir, sanitize_alias

        base = (
            cfg.adapter_name
            if cfg.adapter_name and cfg.adapter_name != "default"
            else Path(cfg.output_dir).name
        )
        alias = sanitize_alias(base)
        dest = loras_dir() / f"{alias}.safetensors"
        if Path(lora_path).resolve() != dest.resolve():
            shutil.copy2(lora_path, dest)
        _write_lora_sidecar(dest.with_suffix(".json"), cfg)
        return str(dest)
    except Exception:  # noqa: BLE001 -- the catalog mirror is best-effort, never fatal
        return None


def _write_lora_sidecar(sidecar_path: Path, cfg: DiffusionLoraConfig) -> None:
    """Write the adapter metadata sidecar read back by diffusion_lora._scan_local. Best
    effort: a failure here must not fail publishing, so callers wrap it."""
    meta = {
        "family": cfg.resolved_family,
        "families": [cfg.resolved_family],
        "base_model": cfg.base_model,
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "steps": cfg.train_steps,
        "resolution": cfg.resolution,
        "trigger_prompt": cfg.instance_prompt,
        "created_at": time.time(),
        "source": "studio-trained",
    }
    sidecar_path.write_text(json.dumps(meta, indent = 2), encoding = "utf-8")


# Aliases from the generic Studio training payload onto DiffusionLoraConfig fields, so the
# diffusion trainer can also be driven by the shared training request shape (not only its
# own request model whose keys already match).
_CONFIG_ALIASES = {
    "model_name": "base_model",
    "max_steps": "train_steps",
    "batch_size": "train_batch_size",
    "lora_r": "lora_rank",
    "lr_scheduler_type": "lr_scheduler",
    "random_seed": "seed",
    "lr": "learning_rate",
}


def _coerce_gradient_checkpointing(value: Any) -> bool:
    """Studio sends gradient_checkpointing as a string ("none" / "true" / "unsloth"); the
    disable words are False, anything else truthy is True. A real bool passes through."""
    if isinstance(value, str):
        return value.strip().lower() not in ("", "none", "false", "0", "no", "off")
    return bool(value)


def _config_from_dict(config: dict) -> DiffusionLoraConfig:
    """Build a DiffusionLoraConfig from a plain dict. Unknown keys are ignored so a richer
    request payload (UI form) does not break construction; a small set of generic Studio
    training keys are aliased onto the diffusion field names, and string flags are coerced."""
    valid = DiffusionLoraConfig.__dataclass_fields__.keys()
    kwargs: dict[str, Any] = {}
    # Aliases first (lowest priority); a canonical key present in the payload overrides.
    for src, dst in _CONFIG_ALIASES.items():
        if src in config and config[src] is not None and dst in valid:
            kwargs[dst] = config[src]
    for k, v in config.items():
        if k in valid:
            kwargs[k] = v
    if kwargs.get("lora_target_modules"):
        kwargs["lora_target_modules"] = tuple(kwargs["lora_target_modules"])
    if "gradient_checkpointing" in kwargs:
        kwargs["gradient_checkpointing"] = _coerce_gradient_checkpointing(
            kwargs["gradient_checkpointing"]
        )
    return DiffusionLoraConfig(**kwargs)
