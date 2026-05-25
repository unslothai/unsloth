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
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
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

    Only the LAST segment of the repo id / path is inspected so a
    namespace or parent directory like ``baseorg/...`` or
    ``/home/me/.cache/base/...`` does not falsely select the Base
    variant (round 12 review #9). Splits on BOTH ``/`` and ``\\`` so
    Windows local paths like ``C:\\Users\\me\\base\\FLUX.2-klein-4B``
    do not get scored as "base" via the parent directory either
    (round 13 P2 #13).
    """
    if fam.name != "flux.2-klein":
        return fam.base_repo
    cleaned = (repo_id or "").rstrip("/\\")
    last_segment = re.split(r"[\\/]+", cleaned)[-1].lower() if cleaned else ""
    is_9b = "9b" in last_segment
    is_base = "base" in last_segment
    if is_9b and is_base:
        return "black-forest-labs/FLUX.2-klein-base-9B"
    if is_9b:
        return "black-forest-labs/FLUX.2-klein-9B"
    if is_base:
        return "black-forest-labs/FLUX.2-klein-base-4B"
    # Distilled 4B is the default for any flux-2-klein GGUF that does
    # not advertise 9B or "base".
    return "black-forest-labs/FLUX.2-klein-4B"


def _expand_existing_local_path(value: str) -> str:
    """Expand ``~`` in ``value`` when the expanded path exists locally.

    Round 14 P2 #11: the GGUF local path branch already calls
    ``Path(repo_id).expanduser()``, but the full-diffusers-repo and
    base-companion-repo paths passed the literal ``~/...`` straight
    into ``from_pretrained``, which treated it as a Hub id and tried
    to download. Keep behaviour identical for Hub ids (no leading
    ``~`` -> return as-is) and for non-existent expansions (the
    diffusers loader will surface its own ``not found`` error).
    """
    if not value or not isinstance(value, str) or not value.startswith("~"):
        return value
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate)
    return value


def _display_repo_id(value: Any) -> Any:
    """Return a public-facing label for a repo_id / base_repo.

    For Hub-style identifiers (``owner/repo``) the value passes
    through unchanged so the Images panel and result figcaption
    stay informative. Absolute local paths (``/home/me/exports/...``
    or ``C:\\Users\\...``) collapse to the leaf name so
    ``/images/status`` does not leak the user's filesystem layout
    to other authenticated browser sessions (round 15 P2 #6). HF
    tokens are scrubbed defensively in case they slipped past the
    request-side validator.
    """
    if not isinstance(value, str) or not value:
        return value
    try:
        candidate = Path(value).expanduser()
        if candidate.is_absolute() or candidate.exists():
            return candidate.name or value
    except (OSError, ValueError):
        pass
    return _redact_hf_tokens(value)


_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{20,}")


def _redact_hf_tokens(value: Any) -> Any:
    """Scrub embedded ``hf_xxxxxxxx`` tokens out of a string before
    logging. Round 14 P2 #9: callers can wrap an authenticated URL
    (``https://hf_token@huggingface.co/...``) into ``repo_id`` /
    ``base_repo`` / paths; the token would otherwise reach
    structured-log sinks via the load-info / load-failure log lines.
    Non-strings are returned unchanged so the helper is safe to
    sprinkle through ``logger.info`` / ``logger.error`` argument
    lists.
    """
    if not isinstance(value, str):
        return value
    return _HF_TOKEN_RE.sub("<redacted>", value)


def _resolve_local_gguf_child(repo_root: Path, gguf_filename: str) -> Path:
    """Resolve a GGUF filename inside a local repo directory safely.

    Returns the resolved absolute path or raises ``RuntimeError`` if:
    - ``gguf_filename`` is absolute (``/etc/passwd``) or contains a
      Windows separator (``..\\..\\secret.gguf``);
    - the parts contain ``""`` / ``.`` / ``..`` (``../other.gguf``);
    - the resolved candidate escapes ``repo_root`` after symlinks /
      ``..`` collapse;
    - the resolved candidate is not a regular file.

    This is the only path that bridges a user-supplied ``gguf_filename``
    string into ``Path``s the loader opens, so confining it to the
    chosen repo here protects the delete-ownership guards downstream
    (round 13 P1 #2). ``hf_hub_download`` already enforces the same
    invariant for Hub repos.
    """
    # ``Path("/etc/passwd").is_absolute()`` is False on Windows (POSIX
    # absolute paths read as drive-relative), so check both pathlib
    # flavours plus a leading separator so the rejection is portable.
    if (
        Path(gguf_filename).is_absolute()
        or PurePosixPath(gguf_filename).is_absolute()
        or gguf_filename.startswith(("/", "\\"))
        or "\\" in gguf_filename
    ):
        raise RuntimeError("gguf_filename must be a relative file path inside repo_id.")
    rel = PurePosixPath(gguf_filename)
    if any(part in ("", ".", "..") for part in rel.parts):
        raise RuntimeError(
            "gguf_filename must not contain empty, '.', or '..' segments."
        )
    root = repo_root.expanduser().resolve(strict = True)
    try:
        candidate = (root / Path(*rel.parts)).resolve(strict = True)
    except (OSError, FileNotFoundError) as exc:
        # strict=True raises FileNotFoundError on a missing leaf or
        # parent component, and OSError on a malformed Windows path
        # (e.g. drive letters injected through the user-supplied
        # string). Either way the candidate does not exist inside the
        # chosen repo, which is exactly the "file not in repo" failure
        # mode the caller cares about.
        raise RuntimeError(
            f"Local repo path '{repo_root}' does not contain '{gguf_filename}'."
        ) from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            "gguf_filename must stay inside the local repo_id directory."
        ) from exc
    if not candidate.is_file():
        raise RuntimeError(
            f"Local repo path '{repo_root}' does not contain '{gguf_filename}'."
        )
    return candidate


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
    # Round 17 P2 #10: if repo_id is an absolute local path, the
    # whole path goes into ``needle`` and the _FAMILY_EXCLUDE deny
    # lists match against parent-directory names too. That means
    # ``/home/me/qwen-image-edit-cache/flux-2-klein-4b`` would be
    # excluded from the Flux family because the parent contains
    # ``qwen-image-edit``. Reduce to the leaf when the candidate
    # looks like a filesystem path so excludes only consider the
    # model directory itself.
    if "/" in needle or "\\" in needle:
        try:
            candidate = Path(repo_id).expanduser()
            if candidate.is_absolute() or candidate.exists():
                leaf = candidate.name
                if leaf:
                    needle = leaf.lower()
        except (OSError, ValueError):
            pass
    # Normalise mixed separator spellings (``Qwen_Image-Edit-GGUF``,
    # ``Qwen-Image_Edit-GGUF``, ``Qwen.Image.Edit-GGUF``) and the
    # compact concatenation (``QwenImageEdit-GGUF``) so the
    # _FAMILY_EXCLUDE deny lists do not need every permutation of
    # ``-``, ``_``, ``.`` and run-together spellings to keep
    # Qwen-Image-Edit out of the base Qwen-Image family (round 14
    # P2 #8).
    needle_norm = re.sub(r"[^a-z0-9]+", "-", needle).strip("-")
    needle_compact = re.sub(r"[^a-z0-9]+", "", needle)
    # Per-token compact strings let ``unsloth/Flux2Klein-GGUF`` match
    # the ``flux2klein`` alias: the whole-needle compact is
    # ``unslothflux2kleingguf`` and the regex boundary check rejects
    # the embedded match, but the token ``Flux2Klein`` (between the
    # ``/`` and the ``-``) compacts to exactly ``flux2klein`` (round
    # 16 P2 #9).
    needle_compact_tokens = {
        re.sub(r"[^a-z0-9]+", "", token)
        for token in re.split(r"[^a-z0-9]+", needle)
        if token
    }

    def _matches_family_token(term: str) -> bool:
        """Token-boundary match on the normalised needle. Prevents
        ``owner/flux.20-model`` from matching ``flux.2`` because
        ``flux.20`` does not have a separator after ``flux-2``
        (round 15 P2 #8). Compact spellings (``flux2klein``) match
        only when they appear as a complete repo-name token, not
        as a substring of a longer token (round 16 P2 #9)."""
        term_norm = re.sub(r"[^a-z0-9]+", "-", term.lower()).strip("-")
        if not term_norm:
            return False
        if re.search(rf"(^|-){re.escape(term_norm)}($|-)", needle_norm):
            return True
        term_compact = re.sub(r"[^a-z0-9]+", "", term.lower())
        if not term_compact:
            return False
        return term_compact in needle_compact_tokens or term_compact == needle_compact

    # Scan _FAMILIES first (GGUF-supported), then _FULL_REPO_FAMILIES
    # so a repo like ``stabilityai/stable-diffusion-xl-base-1.0`` is
    # auto-detected as SDXL instead of returning None.
    for fam in _FAMILIES + _FULL_REPO_FAMILIES:
        excludes = _FAMILY_EXCLUDE.get(fam.name, ())
        if any(
            e in needle
            or re.sub(r"[^a-z0-9]+", "-", e).strip("-") in needle_norm
            or re.sub(r"[^a-z0-9]+", "", e) in needle_compact
            for e in excludes
        ):
            continue
        if _matches_family_token(fam.name):
            return fam
        for alias in fam.aliases:
            if alias and _matches_family_token(alias):
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
        # Original ``gguf_filename`` the caller passed in, preserved
        # so delete guards can compare against subdirectory variants
        # like ``BF16/model.gguf`` or ``Q4_K_M/model.gguf`` instead
        # of the collapsed basename (round 14 P1 #4). The basename
        # alone (``model.gguf``) loses the quant directory and lets
        # /delete-cached unlink the wrong file.
        self._gguf_filename: Optional[str] = None
        self._base_repo: Optional[str] = None
        self._device: Optional[str] = None
        self._dtype: Optional[str] = None
        # True when ``enable_model_cpu_offload()`` was applied on the
        # loaded pipeline. Diffusers' offload moves the active
        # submodule between CPU and GPU on each step, so a CUDA
        # ``torch.Generator`` mismatches the CPU-resident embeddings
        # and generation crashes mid-forward (round 14 P1 #6). When
        # this is True, seeded generation has to use a CPU generator
        # regardless of self._device.
        self._cpu_offload_enabled: bool = False
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

    def status(self, *, include_internal: bool = False) -> dict[str, Any]:
        # Take _lock so the snapshot cannot observe a torn state where
        # _pipe was already swapped but _family/_repo_id haven't been
        # updated yet (or vice versa). Frontend polling at 1 Hz would
        # otherwise render impossible "loaded but no repo_id" states.
        # Only echo the GGUF basename; full absolute path leaks the
        # local HF cache layout (and the system username on default
        # POSIX layouts) to any authenticated Studio session.
        #
        # Round 16 P1 #5: the guard-facing ``active_*`` / ``pending_*``
        # fields hold the EXACT raw path (so /delete-cached can match
        # an HF snapshot mmap) but are NOT safe to surface to the
        # browser. Callers that need the raw path (route-internal
        # delete guards) pass ``include_internal=True``; the public
        # ``/api/inference/images/status`` route always uses the
        # public payload.
        with self._lock:
            # UI-facing collapsed basename. Full local path leaks the
            # HF cache layout + system username; the original caller-
            # supplied filename (e.g. ``BF16/model.gguf``) is kept
            # separately as ``active_gguf_filename`` for delete
            # guards.
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
            active_gguf = self._gguf_filename
            pending_repo = self._pending_repo_id if self._loading else None
            pending_base = self._pending_base_repo if self._loading else None
            pending_gguf = self._pending_gguf_filename if self._loading else None
            # When a swap is in flight, the UI-facing repo_id /
            # base_repo / gguf_filename advertise the PENDING model
            # but ``self._family`` still points at the previously
            # loaded pipeline. Reporting them together produces a
            # repo/family pair that never existed (round 11 #6).
            # Null the family / pipeline_class while a swap is in
            # flight; the frontend can fall back to "unknown".
            ui_family = self._family.name if self._family else None
            ui_pipeline_class = self._family.pipeline_class if self._family else None
            if pending_repo and pending_repo != active_repo:
                ui_family = None
                ui_pipeline_class = None
            # UI-facing ``gguf_filename`` collapses to the basename
            # so the Images panel does not surface internal cache /
            # variant directory names. Guard-facing ``active_*`` /
            # ``pending_*`` retain the full caller-supplied filename
            # so /delete-cached can compare against subdirectory
            # variants like ``BF16/model.gguf`` (round 14 P1 #4-5).
            ui_gguf = pending_gguf or active_gguf
            ui_gguf_basename = Path(ui_gguf).name if ui_gguf else None
            # UI-facing ``repo_id`` / ``base_repo`` collapse absolute
            # local paths to their leaf name so ``/images/status``
            # does not leak the user's filesystem layout to other
            # authenticated browser sessions (round 15 P2 #6). The
            # guard-facing ``active_*`` / ``pending_*`` fields below
            # preserve the exact value so delete guards still match
            # against the snapshot path.
            payload: dict[str, Any] = {
                "is_loaded": self._pipe is not None,
                "is_loading": self._loading,
                "repo_id": _display_repo_id(pending_repo or active_repo),
                "family": ui_family,
                "pipeline_class": ui_pipeline_class,
                "base_repo": _display_repo_id(pending_base or active_base),
                "gguf_filename": ui_gguf_basename,
                "device": self._device,
                "dtype": self._dtype,
                "loaded_at": self._loaded_at,
                "last_error": self._last_error,
                "supported_families": supported_families(),
            }
            if include_internal:
                # Guard-facing fields: every repo / path / GGUF
                # filename the backend owns RIGHT NOW. Delete routes
                # iterate both, paired so the variant-filename check
                # is compared against the SAME repo that owns it
                # (round 13 P1 #3-5). Round 16 P1 #5: never returned
                # by the public /images/status route.
                payload.update(
                    {
                        "active_repo_id": active_repo,
                        "active_base_repo": active_base,
                        "active_gguf_filename": active_gguf,
                        "pending_repo_id": pending_repo,
                        "pending_base_repo": pending_base,
                        "pending_gguf_filename": pending_gguf,
                    }
                )
            return payload

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
        # Surface a friendly load error when the no-torch / partial
        # install path is active: the user clicked Load on the Images
        # page but the runtime never installed torch + diffusers (round
        # 13 P2 #12). Without this wrapper the import surfaces as a
        # raw ``ModuleNotFoundError`` -> 500 instead of a 400 the UI
        # can display.
        try:
            from huggingface_hub import hf_hub_download
            import diffusers
            import torch
        except ModuleNotFoundError as exc:
            missing = exc.name or str(exc)
            raise RuntimeError(
                "Diffusion image generation requires the torch / diffusers "
                f"runtime. Missing dependency: {missing}. Install the Studio "
                "torch runtime (re-run setup.sh / install.ps1) before "
                "loading an image model."
            ) from exc

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
                # Store the caller's full ``gguf_filename`` (e.g.
                # ``BF16/model.gguf``) so the variant-aware delete
                # guards have the subdirectory info. The UI side of
                # status() still collapses to the basename for display.
                self._pending_gguf_filename = gguf_filename if gguf_filename else None
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
                #   1. no GGUF file -> caller is loading a full
                #      diffusers repo; use repo_id directly so we do
                #      not silently substitute the family default
                #      AND ignore any base_repo input (it is only
                #      meaningful as a GGUF companion override). The
                #      old order let ``base_repo`` swap a fine-tuned
                #      ``owner/my-flux.1-finetune`` for
                #      ``black-forest-labs/FLUX.1-dev`` while status
                #      still advertised the user's repo (round 13
                #      P2 #10).
                #   2. otherwise prefer caller-supplied base_repo for
                #      the missing VAE / text encoder components.
                #   3. otherwise use the family + repo_id heuristic so
                #      a 9B GGUF picks the 9B base, not the 4B fallback.
                if not gguf_filename:
                    # Guard: a repo that ends in "-GGUF" (the unsloth
                    # convention) is GGUF-only and will 500 on
                    # from_pretrained; surface a clear error instead of
                    # letting diffusers raise a confusing model-index
                    # failure deep in the loader.
                    if repo_id.lower().endswith("-gguf"):
                        raise RuntimeError(
                            f"'{repo_id}' looks like a GGUF-only repo. "
                            "Either provide gguf_filename to pick a quant, "
                            "or load a full diffusers repo (base_repo only "
                            "applies when picking a GGUF quant)."
                        )
                    # ``~/models/my-flux`` must be expanded so
                    # diffusers' from_pretrained does not pass the
                    # literal tilde through to ``os.path.isdir`` and
                    # fall back to the Hub (round 14 P2 #11).
                    effective_base = _expand_existing_local_path(repo_id)
                    with self._lock:
                        self._pending_base_repo = effective_base
                elif base_repo:
                    effective_base = _expand_existing_local_path(base_repo)
                    # Refresh pending so delete guards see the actual
                    # base, not just caller-supplied None.
                    with self._lock:
                        self._pending_base_repo = effective_base
                else:
                    effective_base = _smart_base_repo(fam, repo_id)
                    with self._lock:
                        self._pending_base_repo = effective_base
                # ``repo_id`` / ``effective_base`` are user-supplied
                # strings that can embed an ``hf_xxxxx`` token via a
                # URL-style path (``https://hf_token@huggingface.co/...``).
                # Scrub them BEFORE the logger formats the line so the
                # token never reaches structured-log sinks (round 14
                # P2 #9).
                logger.info(
                    "Loading diffusion model %s (family=%s, device=%s, dtype=%s, base=%s)",
                    _redact_hf_tokens(repo_id),
                    fam.name,
                    device,
                    dtype,
                    _redact_hf_tokens(effective_base),
                )

                transformer = None
                local_gguf_path: Optional[str] = None
                if gguf_filename:
                    if transformer_cls is None:
                        raise RuntimeError(
                            f"Family {fam.name} does not have a GGUF transformer "
                            "path wired in this build; load the full repo instead."
                        )
                    # DiffusionLoadRequest.repo_id is documented to
                    # accept either a Hub repo id OR a local path
                    # (Studio export, downloaded HF snapshot, etc.).
                    # We accept BOTH absolute and relative local
                    # directories: Studio exports surface as relative
                    # paths like ``exports/my-flux`` and earlier
                    # versions only accepted absolute paths, falling
                    # through to ``hf_hub_download`` which then
                    # raised HFValidationError on the relative path
                    # (round 13 P1 #2). For local paths we route the
                    # gguf_filename through ``_resolve_local_gguf_child``
                    # so traversal (``../secret.gguf``) and absolute
                    # filename escapes (``/etc/passwd``) are rejected
                    # BEFORE the file is opened, which also keeps the
                    # delete-ownership guards aligned with what was
                    # actually loaded.
                    repo_id_path = Path(repo_id).expanduser()
                    if repo_id_path.is_dir():
                        local_gguf_path = str(
                            _resolve_local_gguf_child(repo_id_path, gguf_filename)
                        )
                    else:
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
                #   1. Verify training is idle and the export job (if
                #      any) is also idle. ``_release_other_gpu_owners
                #      _for_diffusion`` RAISES on conflict, so it must
                #      run BEFORE we unload chat (round 16 P1 #2): a
                #      route precheck -> worker race could otherwise
                #      drop the user's chat model only to bail out
                #      because training started in between, and a
                #      direct ``DiffusionBackend.load_model`` caller
                #      that did not run the route prechecks would also
                #      leave chat unloaded for nothing.
                #   2. Release the chat backend (llama-server + the
                #      safetensors orchestrator) now that we know the
                #      load can actually proceed.
                #   3. Release any *previous* diffusion pipeline so the
                #      new transformer / new from_pretrained does not
                #      race the old pipe for VRAM. Switching between
                #      FLUX.2 klein 4B and 9B on a 16-24 GB GPU OOMs
                #      otherwise: from_single_file allocates the new
                #      transformer while the old pipeline still owns
                #      its weights.
                #   4. THEN call from_single_file / from_pretrained.
                _release_other_gpu_owners_for_diffusion()
                _release_chat_backend_for_diffusion()

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
                        self._gguf_filename = None
                        self._base_repo = None
                        self._device = None
                        self._dtype = None
                        self._cpu_offload_enabled = False
                        self._loaded_at = None
                    _release(old)
                    old = None
                    # Now that both the attribute and the local
                    # have been nulled, the pipeline is unreachable;
                    # ask the CUDA allocator to release its slabs so
                    # the next from_pretrained does not OOM behind
                    # an already-freed-but-cached arena.
                    _drain_cuda_cache()

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

                pipe = None
                cpu_offload_enabled = bool(
                    enable_model_cpu_offload and device == "cuda"
                )
                try:
                    pipe = pipeline_cls.from_pretrained(effective_base, **pipe_kwargs)
                    # Device placement / offload can ALSO raise after
                    # from_pretrained succeeded (OOM at the .to(device)
                    # copy, accelerate offload hook misconfigured, etc.).
                    # If we let the exception escape now, the local
                    # ``pipe`` lives on the traceback frame until the
                    # caller drops it, holding multi-GB of VRAM behind
                    # the next load attempt. Explicitly release both
                    # pipe and transformer in the same try (round 13
                    # P2 #11).
                    if cpu_offload_enabled:
                        pipe.enable_model_cpu_offload()
                    else:
                        pipe.to(device)
                except Exception:
                    if pipe is not None:
                        _release(pipe)
                        pipe = None
                    if transformer is not None:
                        _release(transformer)
                        transformer = None
                    _drain_cuda_cache()
                    raise

                with self._lock:
                    self._pipe = pipe
                    self._family = fam
                    self._repo_id = repo_id
                    self._gguf_path = local_gguf_path
                    # Preserve the full caller-supplied filename, not
                    # just the basename, so per-variant delete guards
                    # see ``BF16/model.gguf`` (round 14 P1 #4).
                    self._gguf_filename = gguf_filename if gguf_filename else None
                    self._base_repo = effective_base
                    self._device = device
                    self._dtype = str(dtype).replace("torch.", "")
                    self._cpu_offload_enabled = cpu_offload_enabled
                    self._loaded_at = time.time()
                    # Clear loading + pending here, BEFORE returning,
                    # so the response payload reports the resident
                    # pipeline cleanly (is_loading=false, no pending_*).
                    # The ``finally`` block below is idempotent and
                    # still clears on error / early raise paths.
                    self._loading = False
                    self._pending_repo_id = None
                    self._pending_base_repo = None
                    self._pending_gguf_filename = None

                return self.status()
            except Exception as exc:
                # Scrub hf_token and pipe_kwargs from frame locals BEFORE
                # logger.exception() captures them. Rich tracebacks and
                # some structlog formatters render frame locals, which
                # would otherwise echo the raw hf_... token into logs
                # and any error reporting sink the user has wired up.
                # ALSO scrub the exception message itself: huggingface_hub
                # / diffusers can include the bearer token verbatim in
                # 401 / 403 messages, which would propagate through
                # ``_last_error`` (rendered in status()) and the
                # user-facing RuntimeError (rendered in route responses).
                scrub_token = hf_token
                hf_token = None  # noqa: F841
                pipe_kwargs = None  # noqa: F841
                single_file_kwargs = None  # noqa: F841
                exc_msg = str(exc)
                if scrub_token:
                    exc_msg = exc_msg.replace(scrub_token, "<redacted>")
                # Hugging Face tokens are prefixed ``hf_``; replace any
                # leftover ``hf_...`` substrings to catch tokens we did
                # not store as ``scrub_token`` (e.g. cached tokens that
                # huggingface_hub picked up on its own).
                import re

                exc_msg = re.sub(r"hf_[A-Za-z0-9]{20,}", "<redacted>", exc_msg)

                # Round 17 P2 #9: diffusers / safetensors raise errors
                # like ``FileNotFoundError: /home/alice/models/foo.gguf``
                # or ``OSError: Error while loading state dict from
                # C:\\Users\\bob\\repos\\flux``. These messages flow
                # into ``_last_error`` (rendered by status() to every
                # authenticated browser tab) and the user-facing
                # RuntimeError, which would leak the operator's
                # filesystem layout to other sessions. Collapse the
                # known repo / base / gguf paths to their leaf name
                # using the same convention as _display_repo_id().
                def _collapse_local(msg: str, candidate: Optional[str]) -> str:
                    if not candidate or not isinstance(candidate, str):
                        return msg
                    try:
                        p = Path(candidate).expanduser()
                    except (OSError, ValueError):
                        return msg
                    leaf = p.name or candidate
                    abs_str = None
                    if p.is_absolute() or p.exists():
                        try:
                            abs_str = str(p)
                        except (OSError, ValueError):
                            abs_str = None
                    if abs_str and abs_str in msg:
                        msg = msg.replace(abs_str, leaf)
                    if (
                        candidate != leaf
                        and candidate in msg
                        and ("/" in candidate or "\\" in candidate)
                    ):
                        msg = msg.replace(candidate, leaf)
                    return msg

                # ``effective_base`` and ``gguf_filename`` are local
                # to the try block above and may be unbound if the
                # exception fired before assignment (e.g. the GGUF
                # repo / filename validation raises before
                # ``effective_base`` is computed). ``locals().get``
                # keeps the scrub a no-op in that case.
                # Round 18 P2 #9: also scrub ``local_gguf_path``. The
                # GGUF quant is loaded via
                # ``transformer_cls.from_single_file(local_gguf_path)``,
                # and diffusers / safetensors errors include the
                # resolved absolute HF cache path
                # (``/home/alice/.cache/huggingface/hub/.../flux.gguf``).
                # Without this the cache path would leak into
                # ``_last_error`` (and therefore status() / log lines).
                _locals = locals()
                exc_msg = _collapse_local(exc_msg, repo_id)
                exc_msg = _collapse_local(exc_msg, _locals.get("effective_base"))
                exc_msg = _collapse_local(exc_msg, _locals.get("gguf_filename"))
                exc_msg = _collapse_local(exc_msg, _locals.get("local_gguf_path"))
                with self._lock:
                    self._last_error = exc_msg
                # ``logger.exception`` would emit the raw exception
                # (including any unredacted ``hf_...`` token inside
                # the message OR traceback locals on rich loggers).
                # Use ``logger.error`` with the already-scrubbed
                # message and exc_info=False so the bearer token
                # cannot leak through structured logging sinks.
                logger.error(
                    "Diffusion load failed for %s: %s",
                    _redact_hf_tokens(repo_id),
                    exc_msg,
                )
                raise RuntimeError(
                    f"Failed to load diffusion model: {exc_msg}"
                ) from exc
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
                self._gguf_filename = None
                self._base_repo = None
                self._device = None
                self._dtype = None
                self._cpu_offload_enabled = False
                self._loaded_at = None
            _release(old)
            old = None  # noqa: F841
            _drain_cuda_cache()
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
        # Take _generate_lock FIRST so a concurrent unload/load that
        # observes us holding it will queue behind this generation
        # (and `unload_model` then waits its turn before clearing
        # state). Snapshotting `self._pipe` outside the lock and then
        # taking the lock let a load/unload race in between, so the
        # forward could run against a freed or swapped pipeline.
        with self._generate_lock:
            return self._generate_image_unlocked(
                prompt = prompt,
                negative_prompt = negative_prompt,
                num_inference_steps = num_inference_steps,
                guidance_scale = guidance_scale,
                width = width,
                height = height,
                seed = seed,
            )

    def _generate_image_unlocked(
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
        """Inner body of ``generate_image`` that ASSUMES the caller
        already holds ``_generate_lock``. Lets
        ``generate_image_with_metadata`` snapshot metadata under the
        same lock without deadlocking on a non-reentrant
        ``threading.Lock`` (round 13 P2 #9)."""
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

        with self._lock:
            if self._pipe is None:
                raise RuntimeError("No diffusion model is loaded.")
            pipe = self._pipe
            device = self._device or "cpu"
            cpu_offload_enabled = self._cpu_offload_enabled
        generator = None
        if seed is not None:
            # Match the device of the pipeline so determinism holds
            # across reload cycles. When CPU offload is enabled
            # (the default on CUDA hosts), diffusers shuttles each
            # submodule between CPU and GPU on every step. A CUDA
            # torch.Generator then mismatches the CPU-resident
            # embeddings at the start of the forward and the run
            # crashes (round 14 P1 #6). Use a CPU generator in that
            # case; numerical determinism for the same seed is
            # preserved because the seed feeds an int rather than a
            # device-local RNG state.
            if cpu_offload_enabled:
                gen_device = "cpu"
            else:
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

    def generate_image_with_metadata(
        self,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Generate a single image AND snapshot its identifying metadata.

        Returns ``(pil_image, {"model": <repo_id>, "family": <name>})``
        where the metadata reflects the pipeline that produced the
        image. Snapshotted under ``_generate_lock + _lock`` so a
        queued unload / load that promotes a different pipeline
        cannot replace ``self._repo_id`` / ``self._family`` between
        the forward returning and the route reading status (round
        13 P2 #9). The route uses these values directly in the
        response instead of re-calling ``status()``.
        """
        with self._generate_lock:
            image = self._generate_image_unlocked(**kwargs)
            with self._lock:
                # Round 16 P1 #6: route ``model`` through
                # _display_repo_id so a generation response for a
                # locally-loaded model cannot echo back an absolute
                # filesystem path to the browser.
                meta = {
                    "model": _display_repo_id(self._repo_id),
                    "family": self._family.name if self._family else None,
                }
        return image, meta


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

    A missing chat backend module is a silent no-op (fresh install /
    no GGUF use). An unload that ACTUALLY fails (raises or leaves
    the backend resident) raises ``RuntimeError`` so the surrounding
    diffusion ``load_model`` bails out instead of double-owning VRAM
    (round 17 P1 #2).
    """
    # 1. GGUF chat backend (llama-server subprocess). We unload when
    #    EITHER is_loaded is True (resident model) OR is_active is
    #    True (mid-download / startup) OR loading_model_identifier is
    #    populated (HF GGUF download in progress, before is_active /
    #    is_loaded flip). The last case is what round 13 P1 #8 flagged.
    try:
        from routes.inference import get_llama_cpp_backend  # type: ignore
    except Exception as exc:
        logger.debug("llama-server unavailable before diffusion load: %s", exc)
    else:
        backend = get_llama_cpp_backend()
        is_loaded = bool(getattr(backend, "is_loaded", False))
        is_active = bool(getattr(backend, "is_active", False))
        is_loading = bool(getattr(backend, "loading_model_identifier", None))
        if is_loaded or is_active or is_loading:
            logger.info(
                "Unloading llama-server (loaded=%s active=%s loading=%s) before diffusion load",
                is_loaded,
                is_active,
                is_loading,
            )
            try:
                ok = backend.unload_model()
            except Exception as exc:
                raise RuntimeError(
                    "Could not unload the existing GGUF chat model before "
                    "loading a diffusion image model."
                ) from exc
            # Round 18 P1 #4: also reject when ``loading_model_identifier``
            # is still set after the unload call. Without this, a GGUF
            # download / startup that was already in flight before the
            # diffusion handoff (and which never flipped is_active to
            # True before the unload landed) keeps allocating into VRAM
            # while diffusion proceeds, double-owning the GPU.
            if (
                ok is False
                or getattr(backend, "is_loaded", False)
                or getattr(backend, "is_active", False)
                or getattr(backend, "loading_model_identifier", None)
            ):
                raise RuntimeError(
                    "The existing GGUF chat model is still active or loading "
                    "after unload; retry before loading a diffusion image model."
                )

    # 2. Safetensors / HF chat backend (the InferenceOrchestrator that
    #    serves FastVisionModel / FastLanguageModel weights). When this
    #    backend has a model resident on the same GPU, a diffusion load
    #    will OOM the same way. We also flush any loading_models set so
    #    a chat load that is mid-download cannot race the diffusion
    #    allocation.
    try:
        from core.inference import get_inference_backend  # type: ignore
    except Exception as exc:
        logger.debug("safetensors unavailable before diffusion load: %s", exc)
        return

    backend = get_inference_backend()
    active_model_name = getattr(backend, "active_model_name", None)
    loading_models = set(getattr(backend, "loading_models", set()) or set())

    def _require_unload(model_name: str) -> None:
        try:
            ok = backend.unload_model(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Could not unload safetensors chat model '{model_name}' "
                "before loading a diffusion image model."
            ) from exc
        if ok is False:
            raise RuntimeError(
                f"Safetensors backend refused to unload '{model_name}' "
                "before loading a diffusion image model."
            )

    if active_model_name:
        logger.info(
            "Unloading safetensors chat backend '%s' before diffusion load",
            active_model_name,
        )
        _require_unload(active_model_name)
    for loading in loading_models:
        if loading == active_model_name:
            continue
        logger.info(
            "Unloading in-flight safetensors chat load '%s' before diffusion",
            loading,
        )
        _require_unload(loading)


def _release_other_gpu_owners_for_diffusion() -> None:
    """Best-effort: shut down export subprocess + active training before
    a diffusion load. Both can hold multi-GB of VRAM and would OOM the
    diffusion allocation on consumer GPUs."""
    # Export resident checkpoint. We tear down a SETTLED export
    # (current_checkpoint populated AND is_export_active() False)
    # because that means the export ran to completion and the user
    # can re-load the result. An in-flight export job
    # (is_export_active() True) is NEVER touched here: terminating
    # it would corrupt the user's partial output artifact.
    #
    # The route layer also rejects /images/load with HTTP 409 via
    # _raise_if_export_active when is_export_active() is True. This
    # helper repeats the local check anyway so that direct backend
    # callers (tests, scripts, future routes that forget the
    # higher-level guard) cannot still kill an active export.
    # Training-active check runs FIRST so direct backend callers
    # (tests, scripts, future routes) cannot bypass the route layer's
    # 409 by calling ``load_model`` directly while a training run is
    # active (round 15 P1 #3). The route layer's
    # ``_raise_if_training_active`` still runs ahead of the load to
    # surface the conflict as 409; this helper re-raises so direct
    # callers see the same RuntimeError the export-active path raises.
    try:
        from core.training import get_training_backend  # type: ignore
    except Exception as exc:
        logger.debug("training module not importable: %s", exc)
    else:
        try:
            training_active = bool(get_training_backend().is_training_active())
        except Exception as exc:
            # Unverifiable status -> fail closed (might be active).
            raise RuntimeError(
                "Could not verify training status before loading a "
                "diffusion image model."
            ) from exc
        if training_active:
            raise RuntimeError(
                "Training is currently active. Stop the training run "
                "before loading a diffusion image model."
            )

    try:
        from core.export import get_export_backend  # type: ignore
    except Exception as exc:
        logger.debug("export module not importable: %s", exc)
        return

    # Round 18 P1 #6: ``get_export_backend()`` raising used to be a
    # silent ``return`` so direct ``DiffusionBackend.load_model``
    # callers could proceed toward GPU allocation without being able
    # to verify export ownership. Fail closed instead, matching the
    # route-level helper which already maps "Could not verify" /
    # "Could not access" failures to HTTP 503.
    try:
        exp = get_export_backend()
    except Exception as exc:
        raise RuntimeError(
            "Could not verify export status before loading a "
            "diffusion image model."
        ) from exc

    is_export_active_fn = getattr(exp, "is_export_active", None)
    if is_export_active_fn is not None:
        try:
            export_is_active = bool(is_export_active_fn())
        except Exception as exc:
            # Round 16 P2 #8: distinguish unverifiable status from
            # active export. The previous "treat as active" mapping
            # surfaced as a misleading 409 conflict; raise a
            # "Could not verify" RuntimeError so the route layer
            # maps it to 503 (retryable) instead.
            raise RuntimeError(
                "Could not verify export status before loading a "
                "diffusion image model."
            ) from exc
        if export_is_active:
            # Round 14 P2 #10: the prior behaviour logged a warning
            # and continued, so direct ``DiffusionBackend.load_model``
            # callers (tests, scripts) silently bypassed the route
            # layer's 409. Hard-refuse instead so any code path that
            # reaches this helper while an export is active sees the
            # same failure mode the route returns.
            raise RuntimeError(
                "An export job is currently active. Stop the export "
                "job before loading a diffusion image model."
            )

    if getattr(exp, "current_checkpoint", None):
        # Round 18 P1 #2: a wedged ``_shutdown_subprocess`` used to log
        # at debug level and continue, so direct backend callers could
        # allocate diffusion VRAM on top of an export checkpoint that
        # still owned the GPU. Mirror the route-level helper and raise
        # so the surrounding ``load_model`` bails out with a clean
        # RuntimeError that the route layer maps to HTTP 503.
        try:
            logger.info("Shutting down idle export subprocess before diffusion load")
            exp._shutdown_subprocess()
        except Exception as exc:
            raise RuntimeError(
                "Could not unload the idle export checkpoint before "
                "loading a diffusion image model."
            ) from exc
        exp.current_checkpoint = None
        exp.is_vision = False
        exp.is_peft = False

    # Note: active training is *not* stopped here. The route layer
    # (`_raise_if_training_active` in routes/inference.py) refuses
    # /images/load with HTTP 409 before this helper runs, so reaching
    # this point with training still active would only happen in
    # programmatic backend calls (tests, scripts). Silently terminating
    # someone's training run when the diffusion load might still fail
    # is worse than letting the load OOM and surfacing it explicitly.


def _release(obj: Any) -> None:
    """Best-effort GPU-memory release for a pipeline being swapped out.

    Only drops the local reference (which the caller has already
    nulled in its own scope) and runs ``gc.collect()`` so __del__
    fires. Does NOT call ``torch.cuda.empty_cache()`` here because
    when the caller still holds the actual reference in a local /
    attribute, ``empty_cache()`` would run before __del__ released
    the weights and would not actually free GPU memory. Use
    ``_drain_cuda_cache()`` AFTER the last reference has been nulled.
    """
    if obj is None:
        return
    try:
        del obj
    except Exception:
        pass
    gc.collect()


def _drain_cuda_cache() -> None:
    """Hand freed weights back to the active accelerator's allocator.

    Call this AFTER every reference to the freed object has been
    dropped (caller's local + attribute) and a ``gc.collect()`` has
    fired __del__. Calling earlier would empty an already-pinned
    cache and not actually release the memory.

    Handles CUDA *and* MPS (Apple Silicon) so a diffusion swap on
    macOS actually returns VRAM to the Metal allocator.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import torch

        mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            mps_module = getattr(torch, "mps", None)
            empty_cache = (
                getattr(mps_module, "empty_cache", None) if mps_module else None
            )
            if empty_cache is not None:
                empty_cache()
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


async def async_generate_with_metadata(
    backend: DiffusionBackend,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    """Run ``generate_image_with_metadata`` in the default executor.

    Used by the /images/generate route so the response model / family
    fields reflect the pipeline that actually produced the image, even
    if an unload races the route between the forward returning and the
    response being assembled (round 13 P2 #9)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: backend.generate_image_with_metadata(**kwargs),
    )
