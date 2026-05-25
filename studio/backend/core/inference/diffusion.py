# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion image generation backend.

Loads Hugging Face diffusion checkpoints in either the standard
``diffusers`` layout or the single-file GGUF layout published under
``unsloth/*-GGUF`` (Flux 2, Flux 2 Klein, Qwen-Image, SD3, SDXL, ...).
GGUF files are dynamically dequantised on-device via
``diffusers.GGUFQuantizationConfig``, then the rest of the pipeline
(VAE, text encoders, scheduler) is pulled from the matching ``diffusers``
repo so end users only ever need one local file plus the metadata repo.

The module is intentionally torch-only: it never spawns a subprocess and
shares the active CUDA / MPS device with the rest of Studio. The cost of
not having a separate process is that loading a diffusion model and a
GGUF chat model at the same time can OOM on consumer GPUs; the routes
layer must therefore swap between the two as needed (the orchestrator
unloads llama-server before any diffusion load on hosts with < 24 GB).

The class deliberately exposes a small, llama-cpp-style surface:

    load_model(repo_id, ...)
    generate_image(prompt, ...) -> PIL.Image
    unload_model()
    status() -> dict

so the route layer at ``studio/backend/routes/inference.py`` can mirror
the existing llama-server lifecycle (probe + load + generate + unload)
without learning a second API.
"""

from __future__ import annotations

import asyncio
import gc
import io
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)


# ─── Pipeline registry ────────────────────────────────────────────────
#
# Keep this list narrow on purpose: only ship the small text-to-image
# families with first-class GGUF coverage on the Hub. Anything else is
# either video (LTX*, Wan) or research-grade (Sana, SD3.5) and can be
# added once it has a working GGUF release plus a smoke test.
#
# Each entry maps a substring of the loaded repo id (case-insensitive)
# to the (pipeline_class_name, transformer_class_name, default base
# repo for missing pieces). ``base_repo`` is what we pass to
# ``Pipeline.from_pretrained`` to pick up the VAE + text encoders when
# the user gave us a GGUF-only repo. The base_repo is documented to the
# user via ``status()`` so they understand why a second download fires.


@dataclass(frozen = True)
class DiffusionFamily:
    name: str
    pipeline_class: str
    transformer_class: str
    base_repo: str
    # Optional: list of HF "trigger" substrings besides ``name`` that map
    # to this family (e.g. "flux1-dev" plus "flux.1-dev"). Lowercased.
    aliases: tuple[str, ...] = field(default_factory = tuple)


_FAMILIES: tuple[DiffusionFamily, ...] = (
    # The "9b" alias is checked first so a "flux-2-klein-9b" GGUF picks
    # the 9B base instead of the 4B one when the user does not pass an
    # explicit base_repo. Apache 2.0 is preferred as the auto-default for
    # the 4B path because BFL's 9B base is gated.
    DiffusionFamily(
        name = "flux.2-klein",
        pipeline_class = "Flux2KleinPipeline",
        transformer_class = "Flux2Transformer2DModel",
        # Default for klein when no explicit base_repo: Apache-2.0 4B Base.
        # The frontend curated picker always passes base_repo explicitly,
        # so this default only fires for "custom HF repo" mode.
        base_repo = "black-forest-labs/FLUX.2-klein-base-4B",
        aliases = ("flux2-klein", "flux-2-klein", "flux.2.klein"),
    ),
    DiffusionFamily(
        name = "flux.2",
        pipeline_class = "Flux2Pipeline",
        transformer_class = "Flux2Transformer2DModel",
        base_repo = "black-forest-labs/FLUX.2-dev",
        aliases = ("flux2-dev", "flux-2-dev", "flux.2.dev"),
    ),
    DiffusionFamily(
        name = "flux.1",
        pipeline_class = "FluxPipeline",
        transformer_class = "FluxTransformer2DModel",
        base_repo = "black-forest-labs/FLUX.1-dev",
        aliases = ("flux1-dev", "flux-1-dev", "flux.1.dev", "flux-dev"),
    ),
    DiffusionFamily(
        name = "qwen-image",
        pipeline_class = "QwenImagePipeline",
        transformer_class = "QwenImageTransformer2DModel",
        base_repo = "Qwen/Qwen-Image",
        aliases = ("qwenimage", "qwen_image"),
    ),
    DiffusionFamily(
        name = "stable-diffusion-3",
        pipeline_class = "StableDiffusion3Pipeline",
        transformer_class = "SD3Transformer2DModel",
        base_repo = "stabilityai/stable-diffusion-3-medium-diffusers",
        # Intentionally NOT including "sd3.5" / "stable-diffusion-3.5"
        # here: the SD3.5 family uses a different transformer config and
        # base repo than SD3 Medium, and silently pairing SD3.5 GGUFs
        # with the Medium base produces a misleading load. Add a
        # dedicated SD3.5 family with its own base_repo when we ship
        # smoke coverage for it.
        aliases = ("sd3-medium", "stable-diffusion-3-medium"),
    ),
    # SDXL: full diffusers path only (no GGUF). SDXL uses a UNet (not a
    # transformer) and wiring UNet2DConditionModel.from_single_file +
    # GGUF is a separate code path the rest of this module does not
    # exercise. The family is intentionally NOT in _FAMILIES so the
    # frontend status panel does not advertise GGUF support we do not
    # implement; callers wanting SDXL full repos can still do so by
    # passing the diffusers repo with no gguf_filename and
    # family_override = "stable-diffusion-xl" via the route, which uses
    # the lookup in _FULL_REPO_FAMILIES.
)


# Families available via family_override on the routes layer when the
# user is loading a full diffusers checkpoint (no GGUF). Kept separate
# from _FAMILIES so the GGUF-only status panel does not over-advertise.
_FULL_REPO_FAMILIES: tuple[DiffusionFamily, ...] = (
    DiffusionFamily(
        name = "stable-diffusion-xl",
        pipeline_class = "StableDiffusionXLPipeline",
        transformer_class = "",
        base_repo = "stabilityai/stable-diffusion-xl-base-1.0",
        aliases = ("sdxl",),
    ),
)


def _smart_base_repo(fam: DiffusionFamily, repo_id: str) -> str:
    """Pick the best matching base diffusers repo for a given GGUF repo
    when the caller did not pass an explicit base_repo.

    Currently only specialises the flux.2-klein family: a repo name
    containing "9b" gets the 9B base, "base-4b" / "base-9b" map to the
    Base variants, everything else falls back to the family default
    (Apache 2.0 4B Base).
    """
    if fam.name != "flux.2-klein":
        return fam.base_repo
    lower = (repo_id or "").lower()
    is_9b = "9b" in lower
    is_base = "base" in lower
    if is_9b and is_base:
        return "black-forest-labs/FLUX.2-klein-base-9B"
    if is_9b:
        return "black-forest-labs/FLUX.2-klein-9B"
    if is_base:
        return "black-forest-labs/FLUX.2-klein-base-4B"
    # Distilled 4B is the default for any flux-2-klein GGUF that does
    # not advertise 9B or "base".
    return "black-forest-labs/FLUX.2-klein-4B"


# Negative substrings that disqualify a candidate family even when its
# name appears as a substring of the repo id. Prevents
# "stable-diffusion-3" matching SD3.5 and "qwen-image" matching
# Qwen-Image-Edit. Each entry maps a family name to substrings that
# must NOT appear anywhere in the repo id.
_FAMILY_EXCLUDE: dict[str, tuple[str, ...]] = {
    "stable-diffusion-3": (
        "3.5",
        "3-5",
        "3_5",
        "stable-diffusion-3.5",
        "stable_diffusion_3_5",
    ),
    # All underscore / hyphen spellings that appear in Hub repo ids for
    # the *-Edit family must exclude Qwen-Image, otherwise
    # ``unsloth/qwen_image_edit-GGUF`` matches the Qwen-Image base.
    "qwen-image": (
        "qwen-image-edit",
        "qwenimage-edit",
        "qwen_image_edit",
        "qwenimageedit",
    ),
}


def detect_family(
    repo_id: str, *, override_family: Optional[str] = None
) -> Optional[DiffusionFamily]:
    """Return the diffusion family matching ``repo_id``.

    Matching is substring-based and case-insensitive, with a small
    deny list (``_FAMILY_EXCLUDE``) for known false positives such as
    SD3.5 (would otherwise match SD3 Medium) and Qwen-Image-Edit
    (would otherwise match Qwen-Image). ``override_family`` bypasses
    substring matching and looks up by ``DiffusionFamily.name`` or
    (when explicitly asked) by ``_FULL_REPO_FAMILIES.name``. Returns
    ``None`` when no family applies so callers can surface a clear
    "unsupported model" error rather than guessing wrong.
    """
    if override_family:
        wanted = override_family.strip().lower()
        for fam in _FAMILIES + _FULL_REPO_FAMILIES:
            if fam.name == wanted:
                return fam
        return None
    needle = (repo_id or "").lower()
    if not needle:
        return None
    # Scan _FAMILIES first (GGUF-supported), then _FULL_REPO_FAMILIES
    # so a repo like ``stabilityai/stable-diffusion-xl-base-1.0`` is
    # auto-detected as SDXL instead of returning None.
    for fam in _FAMILIES + _FULL_REPO_FAMILIES:
        excludes = _FAMILY_EXCLUDE.get(fam.name, ())
        if any(e in needle for e in excludes):
            continue
        if fam.name in needle:
            return fam
        for alias in fam.aliases:
            if alias and alias in needle:
                return fam
    return None


def supported_families() -> list[dict[str, str]]:
    """Public-facing list of families for ``/api/inference/images/status``."""
    return [
        {
            "name": fam.name,
            "pipeline_class": fam.pipeline_class,
            "base_repo": fam.base_repo,
        }
        for fam in _FAMILIES
    ]


# ─── Backend ──────────────────────────────────────────────────────────


class DiffusionBackend:
    """Singleton-style diffusion backend.

    One pipeline at a time; ``load_model`` swaps the previous one out.
    Generation is mutex'd so concurrent requests serialise rather than
    racing GPU memory.
    """

    def __init__(self) -> None:
        self._pipe: Any = None
        # `_lock` protects mutations to the small state fields and is
        # the only lock taken by status(). It is intentionally NOT held
        # for the long pipeline forward pass: holding it for the whole
        # generate would block status() polls (frontend at 1 Hz) and
        # any concurrent unload requests for minutes at a time.
        #
        # `_load_lock` serialises the entire load_model call so two
        # concurrent /images/load requests cannot both reach
        # pipeline_cls.from_pretrained at the same time (which would
        # double-spend VRAM and corrupt _pipe).
        #
        # `_generate_lock` serialises pipeline __call__ since diffusers
        # pipelines are not thread-safe; overlapping forwards on the
        # shared pipe corrupt internal scheduler state.
        #
        # Lock order is load -> state and generate -> state (never
        # state -> load/generate) so a forward in flight cannot
        # deadlock the next load or a status poll.
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._family: Optional[DiffusionFamily] = None
        self._repo_id: Optional[str] = None
        self._gguf_path: Optional[str] = None
        self._base_repo: Optional[str] = None
        self._device: Optional[str] = None
        self._dtype: Optional[str] = None
        self._loaded_at: Optional[float] = None
        self._loading: bool = False
        self._last_error: Optional[str] = None
        # `_pending_*` fields advertise the target of an in-flight load
        # so cache- and finetuned-delete guards can refuse to rmtree a
        # repo while it is being downloaded / read. They are set under
        # _lock at the start of load_model and cleared on success or
        # in the finally block. The route layer reads them via
        # status() under _lock.
        self._pending_repo_id: Optional[str] = None
        self._pending_base_repo: Optional[str] = None
        self._pending_gguf_filename: Optional[str] = None

    # ── lifecycle ─────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def repo_id(self) -> Optional[str]:
        return self._repo_id

    def status(self) -> dict[str, Any]:
        # Take _lock so the snapshot cannot observe a torn state where
        # _pipe was already swapped but _family/_repo_id haven't been
        # updated yet (or vice versa). Frontend polling at 1 Hz would
        # otherwise render impossible "loaded but no repo_id" states.
        # Only echo the GGUF basename; full absolute path leaks the
        # local HF cache layout (and the system username on default
        # POSIX layouts) to any authenticated Studio session.
        with self._lock:
            gguf_basename = Path(self._gguf_path).name if self._gguf_path else None
            # Expose BOTH the resident pipeline's id AND the pending
            # load target. Delete guards must check both: when model A
            # is already loaded and a swap to model B is in flight,
            # only checking one would let the user rmtree whichever
            # repo the guard ignored. UI-facing ``repo_id`` /
            # ``base_repo`` / ``gguf_filename`` still prefer pending
            # during a swap so the panel shows the load target the
            # user just clicked.
            active_repo = self._repo_id
            active_base = self._base_repo
            pending_repo = self._pending_repo_id if self._loading else None
            pending_base = self._pending_base_repo if self._loading else None
            pending_gguf = self._pending_gguf_filename if self._loading else None
            return {
                "is_loaded": self._pipe is not None,
                "is_loading": self._loading,
                "repo_id": pending_repo or active_repo,
                "family": self._family.name if self._family else None,
                "pipeline_class": (
                    self._family.pipeline_class if self._family else None
                ),
                "base_repo": pending_base or active_base,
                "gguf_filename": pending_gguf or gguf_basename,
                # Guard-facing fields: every repo / path the backend
                # owns RIGHT NOW. Delete routes iterate both.
                "active_repo_id": active_repo,
                "active_base_repo": active_base,
                "pending_repo_id": pending_repo,
                "pending_base_repo": pending_base,
                "pending_gguf_filename": pending_gguf,
                "device": self._device,
                "dtype": self._dtype,
                "loaded_at": self._loaded_at,
                "last_error": self._last_error,
                "supported_families": supported_families(),
            }

    def _pick_device_and_dtype(self) -> tuple[str, "Any"]:
        """Pick (device, dtype) for the current host.

        CUDA-first because that is the only path our diffusion GGUFs are
        validated on. On macOS we use MPS in float16 to keep the pipeline
        on the Metal GPU. CPU is allowed only as a last resort because
        running FLUX on CPU is unusably slow (> 10 minutes per image).

        BF16 is gated on ``torch.cuda.is_bf16_supported`` because the
        Pascal / Turing class (sm_60 / sm_70 / sm_75) reports
        ``is_available() == True`` but lacks BF16 ALUs; FLUX kernels
        then fail inside ``from_pretrained`` or at the first denoise
        step. Those cards still work on FP16, so fall back rather than
        refuse to load.
        """
        import torch

        if torch.cuda.is_available():
            bf16_ok = False
            try:
                bf16_ok = bool(torch.cuda.is_bf16_supported())
            except Exception:
                bf16_ok = False
            return "cuda", torch.bfloat16 if bf16_ok else torch.float16
        if (
            hasattr(torch, "backends")
            and getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            return "mps", torch.float16
        return "cpu", torch.float32

    def load_model(
        self,
        repo_id: str,
        *,
        gguf_filename: Optional[str] = None,
        base_repo: Optional[str] = None,
        hf_token: Optional[str] = None,
        family_override: Optional[str] = None,
        enable_model_cpu_offload: bool = True,
    ) -> dict[str, Any]:
        """Load a diffusion model.

        ``repo_id`` is the Hugging Face repo id of either a GGUF-only
        repo (e.g. ``unsloth/FLUX.2-klein-4B-GGUF``) or a full diffusers
        repo (e.g. ``black-forest-labs/FLUX.2-klein``). When the repo
        contains a GGUF, ``gguf_filename`` picks which quant to load;
        otherwise diffusers' standard config-driven load runs.

        ``base_repo`` overrides the auto-detected diffusers base used
        for VAE / text encoders. ``family_override`` short-circuits the
        substring matcher when an exotic repo name confuses it.

        Raises ``RuntimeError`` on failure with a user-facing message.
        On a failed swap the previous pipeline is also released to
        keep peak VRAM bounded; status() reports is_loaded=false with
        last_error set so the caller can react.
        """
        from huggingface_hub import hf_hub_download
        import diffusers
        import torch

        fam = detect_family(repo_id, override_family = family_override)
        if fam is None:
            raise RuntimeError(
                f"Could not infer a diffusion family for '{repo_id}'. "
                "Pass family_override = 'flux.2-klein' / 'flux.2' / "
                "'flux.1' / 'qwen-image' / 'stable-diffusion-3' / "
                "'stable-diffusion-xl' to disambiguate."
            )

        device, dtype = self._pick_device_and_dtype()

        # _load_lock serialises the entire load so two concurrent calls
        # cannot both kick off a multi-GB download + GPU upload at once.
        # The second caller waits behind the first and then loads on top
        # of the now-populated state via the normal swap path.
        # _generate_lock is also taken so we do not start swapping the
        # pipeline (release old + allocate new) while a previous
        # generation is still iterating denoising steps; releasing the
        # pipe out from under an in-flight forward corrupts scheduler
        # state. Order: _load_lock -> _generate_lock -> _lock so a
        # forward (which only takes _generate_lock + briefly _lock)
        # cannot block a queued load forever.
        with self._load_lock, self._generate_lock:
            with self._lock:
                self._loading = True
                self._last_error = None
                # Publish the pending target so cache / finetuned
                # delete guards can see what is mid-download even
                # before _repo_id / _base_repo are populated on
                # success.
                self._pending_repo_id = repo_id
                self._pending_base_repo = base_repo
                self._pending_gguf_filename = (
                    Path(gguf_filename).name if gguf_filename else None
                )
            try:
                pipeline_cls = getattr(diffusers, fam.pipeline_class, None)
                if pipeline_cls is None:
                    raise RuntimeError(
                        f"diffusers {diffusers.__version__} has no "
                        f"{fam.pipeline_class}; upgrade diffusers and retry."
                    )
                transformer_cls = (
                    getattr(diffusers, fam.transformer_class, None)
                    if fam.transformer_class
                    else None
                )

                # Resolution rules for the "what repo to call
                # from_pretrained on" question:
                #   1. caller-supplied base_repo wins
                #   2. if no GGUF file was requested the user is loading a
                #      full diffusers repo; use repo_id directly so we do
                #      not silently substitute the family default
                #   3. otherwise use the family + repo_id heuristic so a
                #      9B GGUF picks the 9B base, not the 4B fallback
                if base_repo:
                    effective_base = base_repo
                    # Refresh pending so delete guards see the actual
                    # base, not just caller-supplied None.
                    with self._lock:
                        self._pending_base_repo = effective_base
                elif not gguf_filename:
                    # Guard: a repo that ends in "-GGUF" (the unsloth
                    # convention) is GGUF-only and will 500 on
                    # from_pretrained; surface a clear error instead of
                    # letting diffusers raise a confusing model-index
                    # failure deep in the loader.
                    if repo_id.lower().endswith("-gguf"):
                        raise RuntimeError(
                            f"'{repo_id}' looks like a GGUF-only repo. "
                            "Either provide gguf_filename to pick a quant, "
                            "or pass base_repo to override the full-repo "
                            "load target."
                        )
                    effective_base = repo_id
                    with self._lock:
                        self._pending_base_repo = effective_base
                else:
                    effective_base = _smart_base_repo(fam, repo_id)
                    with self._lock:
                        self._pending_base_repo = effective_base
                logger.info(
                    "Loading diffusion model %s (family=%s, device=%s, dtype=%s, base=%s)",
                    repo_id,
                    fam.name,
                    device,
                    dtype,
                    effective_base,
                )

                transformer = None
                local_gguf_path: Optional[str] = None
                if gguf_filename:
                    if transformer_cls is None:
                        raise RuntimeError(
                            f"Family {fam.name} does not have a GGUF transformer "
                            "path wired in this build; load the full repo instead."
                        )
                    local_gguf_path = hf_hub_download(
                        repo_id = repo_id,
                        filename = gguf_filename,
                        token = hf_token,
                    )

                # All cheap failure points (bad gguf_filename, missing
                # pipeline / transformer class, gated download token,
                # transient Hub error on the GGUF download) have now
                # been validated. Anything past this line allocates
                # GPU memory, so:
                #   1. Release competing GPU owners (chat + export).
                #   2. Release any *previous* diffusion pipeline so the
                #      new transformer / new from_pretrained does not
                #      race the old pipe for VRAM. Switching between
                #      FLUX.2 klein 4B and 9B on a 16-24 GB GPU OOMs
                #      otherwise: from_single_file allocates the new
                #      transformer while the old pipeline still owns
                #      its weights.
                #   3. THEN call from_single_file / from_pretrained.
                # Training is *not* unloaded here: the route layer
                # refuses /images/load with HTTP 409 when training is
                # active so the user keeps their long run.
                _release_chat_backend_for_diffusion()
                _release_other_gpu_owners_for_diffusion()

                old = self._pipe
                if old is not None:
                    with self._lock:
                        # Clear ALL metadata together so a failed swap
                        # cannot leave status() reporting the previous
                        # repo / family / base_repo on top of an empty
                        # pipe. The except block below will restore
                        # last_error so the caller knows what happened.
                        self._pipe = None
                        self._family = None
                        self._repo_id = None
                        self._gguf_path = None
                        self._base_repo = None
                        self._device = None
                        self._dtype = None
                        self._loaded_at = None
                    _release(old)
                    old = None

                if gguf_filename:
                    quant_config = diffusers.GGUFQuantizationConfig(compute_dtype = dtype)
                    # Diffusers-format GGUFs (FLUX.2 klein / Qwen-Image /
                    # SD3) need the matching base repo's component config
                    # at config=<base_repo>, subfolder="transformer".
                    # Older city96-style GGUFs ignore those kwargs. The
                    # token is also passed because gated GGUF repos
                    # require it both at download and at config read time.
                    single_file_kwargs: dict[str, Any] = {
                        "quantization_config": quant_config,
                        "torch_dtype": dtype,
                        "config": effective_base,
                        "subfolder": "transformer",
                    }
                    if hf_token:
                        single_file_kwargs["token"] = hf_token
                    transformer = transformer_cls.from_single_file(
                        local_gguf_path,
                        **single_file_kwargs,
                    )

                pipe_kwargs: dict[str, Any] = {
                    "torch_dtype": dtype,
                    # use_safetensors=True refuses pickle-backed .bin
                    # weights at load time. Diffusers will fall back to
                    # safetensors variants on repos that publish both,
                    # and hard-error on repos that only ship .bin (which
                    # is the threat model we want to block since pickle
                    # files can execute arbitrary code in this process).
                    "use_safetensors": True,
                }
                if transformer is not None:
                    pipe_kwargs["transformer"] = transformer
                if hf_token:
                    pipe_kwargs["token"] = hf_token

                try:
                    pipe = pipeline_cls.from_pretrained(effective_base, **pipe_kwargs)
                except Exception:
                    # If from_pretrained fails after the transformer was
                    # already loaded, the transformer object holds GPU
                    # weights that would only be freed at GC. Drop the
                    # local reference and force a collect so the next
                    # load attempt does not stack VRAM with a phantom
                    # transformer.
                    if transformer is not None:
                        _release(transformer)
                        transformer = None
                    raise
                if enable_model_cpu_offload and device == "cuda":
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to(device)

                with self._lock:
                    self._pipe = pipe
                    self._family = fam
                    self._repo_id = repo_id
                    self._gguf_path = local_gguf_path
                    self._base_repo = effective_base
                    self._device = device
                    self._dtype = str(dtype).replace("torch.", "")
                    self._loaded_at = time.time()

                return self.status()
            except Exception as exc:
                # Scrub hf_token and pipe_kwargs from frame locals BEFORE
                # logger.exception() captures them. Rich tracebacks and
                # some structlog formatters render frame locals, which
                # would otherwise echo the raw hf_... token into logs
                # and any error reporting sink the user has wired up.
                hf_token = None  # noqa: F841
                pipe_kwargs = None  # noqa: F841
                single_file_kwargs = None  # noqa: F841
                with self._lock:
                    self._last_error = str(exc)
                logger.exception("Diffusion load failed for %s", repo_id)
                raise RuntimeError(f"Failed to load diffusion model: {exc}") from exc
            finally:
                with self._lock:
                    self._loading = False
                    # Clear pending so status() falls back to publishing
                    # the resident pipeline (or nothing, on a failed
                    # swap). Keeping pending alive after the load
                    # finishes would falsely block deletes forever.
                    self._pending_repo_id = None
                    self._pending_base_repo = None
                    self._pending_gguf_filename = None

    def unload_model(self) -> dict[str, Any]:
        # Take the load lock and the generate lock so unload cannot:
        #   * race with an in-flight load_model and have the load
        #     thread overwrite the cleared state after we already
        #     returned {"is_loaded": false}.
        #   * return is_loaded=false while a forward pass is still
        #     iterating denoising steps on the soon-to-be-freed pipe.
        # The generate forward only holds _generate_lock (briefly
        # _lock), so acquiring _generate_lock here blocks until any
        # in-flight generation completes.
        with self._load_lock, self._generate_lock:
            with self._lock:
                old = self._pipe
                self._pipe = None
                self._family = None
                self._repo_id = None
                self._gguf_path = None
                self._base_repo = None
                self._device = None
                self._dtype = None
                self._loaded_at = None
            _release(old)
        return {"is_loaded": False}

    # ── generation ────────────────────────────────────────────────

    def generate_image(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 24,
        guidance_scale: float = 3.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
    ) -> "Any":
        """Generate a single PIL image and return it.

        Concurrent generations are serialised by ``_generate_lock`` so
        diffusion pipelines (not thread-safe; overlapping ``__call__``s
        corrupt internal scheduler state) only ever run one at a time.
        The state ``_lock`` is taken only to snapshot ``_pipe`` /
        ``_device`` and immediately released: holding it for the whole
        forward pass blocked ``status()`` polls and concurrent unload
        requests for the entire (minutes-long) generation, which made
        the UI feel frozen.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt is empty")
        if num_inference_steps < 1 or num_inference_steps > 200:
            raise ValueError("num_inference_steps must be in [1, 200]")
        if width <= 0 or height <= 0 or width > 2048 or height > 2048:
            raise ValueError("width and height must be in (0, 2048]")
        # Snap to a multiple of 8: Flux / SD pipelines require it and a
        # silent crash deep in the VAE is much worse than a clear error
        # message up front.
        if width % 8 or height % 8:
            raise ValueError("width and height must be multiples of 8")

        import torch

        # Take _generate_lock FIRST so a concurrent unload/load that
        # observes us holding it will queue behind this generation
        # (and `unload_model` then waits its turn before clearing
        # state). Snapshotting `self._pipe` outside the lock and then
        # taking the lock let a load/unload race in between, so the
        # forward could run against a freed or swapped pipeline.
        with self._generate_lock:
            with self._lock:
                if self._pipe is None:
                    raise RuntimeError("No diffusion model is loaded.")
                pipe = self._pipe
                device = self._device or "cpu"
            generator = None
            if seed is not None:
                # Match the device of the pipeline so determinism holds
                # across reload cycles. For CPU offload, the noise still
                # has to live on the device the diffusion forward runs on.
                gen_device = (
                    "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
                )
                generator = torch.Generator(device = gen_device).manual_seed(int(seed))

            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "num_inference_steps": int(num_inference_steps),
                "guidance_scale": float(guidance_scale),
                "width": int(width),
                "height": int(height),
            }
            # FLUX.2 / FLUX.2 klein pipelines do NOT accept
            # negative_prompt and 500 if you pass it in. Inspect the
            # signature and only forward when supported; warn otherwise
            # so the UI can disable the field for incompatible families.
            if negative_prompt is not None and negative_prompt.strip():
                if _pipe_accepts_kwarg(pipe, "negative_prompt"):
                    call_kwargs["negative_prompt"] = negative_prompt
                    # QwenImagePipeline and FluxPipeline treat
                    # guidance_scale as distilled CFG and use
                    # true_cfg_scale as the real classifier-free
                    # guidance knob; the negative prompt is only
                    # effective when true_cfg_scale > 1. Forward the
                    # user-supplied guidance_scale through both so the
                    # negative prompt actually steers generation.
                    if _pipe_accepts_kwarg(pipe, "true_cfg_scale"):
                        call_kwargs["true_cfg_scale"] = float(guidance_scale)
                else:
                    logger.info(
                        "Dropping negative_prompt: %s does not accept it",
                        type(pipe).__name__,
                    )
            if generator is not None:
                call_kwargs["generator"] = generator

            out = pipe(**call_kwargs)
            images = getattr(out, "images", None) or []
            if not images:
                raise RuntimeError("Diffusion pipeline returned no images.")
            return images[0]


def _pipe_accepts_kwarg(pipe: Any, name: str) -> bool:
    """True if ``pipe.__call__`` advertises a kwarg called ``name``.

    Cheap inspect-based probe so we do not have to maintain a manual
    list of which pipeline classes accept negative_prompt. Returns
    False on any introspection error so callers stay on the safe path.
    """
    import inspect

    try:
        sig = inspect.signature(pipe.__call__)
    except (TypeError, ValueError):
        return False
    if name in sig.parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def encode_png_base64(pil_image: "Any") -> str:
    """Encode a PIL image to base64-encoded PNG."""
    import base64

    buf = io.BytesIO()
    pil_image.save(buf, format = "PNG", optimize = True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ─── Helpers ──────────────────────────────────────────────────────────


def _release_chat_backend_for_diffusion() -> None:
    """Unload any running chat backend before a diffusion load.

    Diffusion pipelines on FLUX-class models can eat 12-24 GB of VRAM,
    and the chat backends (llama-server for GGUF, the safetensors
    Inference orchestrator for HF / Unsloth) typically hold onto their
    loaded weights until told to drop them. Asking both to release
    their weights first means a typical 24 GB consumer GPU can host
    one chat model OR one diffusion model without manual unload steps.

    Best effort: if a chat backend module is not importable (CI,
    isolated tests, custom builds) or fails on the unload, we log and
    continue; the diffusion load can still try and surface its own OOM.
    """
    # 1. GGUF chat backend (llama-server subprocess).
    try:
        from routes.inference import get_llama_cpp_backend  # type: ignore

        backend = get_llama_cpp_backend()
        if getattr(backend, "is_loaded", False):
            logger.info("Unloading llama-server before diffusion load")
            backend.unload_model()
    except Exception as exc:
        logger.debug("llama-server unload skipped: %s", exc)

    # 2. Safetensors / HF chat backend (the InferenceOrchestrator that
    #    serves FastVisionModel / FastLanguageModel weights). When this
    #    backend has a model resident on the same GPU, a diffusion load
    #    will OOM the same way. The orchestrator's unload_model takes a
    #    model_name; passing it without args raised TypeError and was
    #    swallowed, leaving the chat model resident.
    try:
        from core.inference import get_inference_backend  # type: ignore

        backend = get_inference_backend()
        active_model_name = getattr(backend, "active_model_name", None)
        if active_model_name:
            logger.info(
                "Unloading safetensors chat backend '%s' before diffusion load",
                active_model_name,
            )
            backend.unload_model(active_model_name)
    except Exception as exc:
        logger.debug("safetensors unload skipped: %s", exc)


def _release_other_gpu_owners_for_diffusion() -> None:
    """Best-effort: shut down export subprocess + active training before
    a diffusion load. Both can hold multi-GB of VRAM and would OOM the
    diffusion allocation on consumer GPUs."""
    # Export subprocess
    try:
        from core.export import get_export_backend  # type: ignore

        exp = get_export_backend()
        if getattr(exp, "current_checkpoint", None):
            logger.info("Shutting down export subprocess before diffusion load")
            exp._shutdown_subprocess()
            exp.current_checkpoint = None
            exp.is_vision = False
            exp.is_peft = False
    except Exception as exc:
        logger.debug("export unload skipped: %s", exc)

    # Note: active training is *not* stopped here. The route layer
    # (`_raise_if_training_active` in routes/inference.py) refuses
    # /images/load with HTTP 409 before this helper runs, so reaching
    # this point with training still active would only happen in
    # programmatic backend calls (tests, scripts). Silently terminating
    # someone's training run when the diffusion load might still fail
    # is worse than letting the load OOM and surfacing it explicitly.


def _release(obj: Any) -> None:
    """Best-effort GPU-memory release for a pipeline being swapped out."""
    if obj is None:
        return
    try:
        del obj
    except Exception:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ─── Module-level singleton ───────────────────────────────────────────


_singleton: Optional[DiffusionBackend] = None
_singleton_lock = threading.Lock()


def get_diffusion_backend() -> DiffusionBackend:
    """Return the process-wide diffusion backend (lazy-instantiated)."""
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = DiffusionBackend()
    return _singleton


async def async_generate(
    backend: DiffusionBackend,
    **kwargs: Any,
) -> "Any":
    """Run ``generate_image`` in the default executor so route handlers
    do not block the event loop for the 5-30 s a diffusion step takes."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: backend.generate_image(**kwargs))
