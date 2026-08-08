# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resumable checkpoints for diffusion (image / future video) LoRA training.

A diffusion run used to write exactly one thing -- the deployable adapter -- at the very
end, so a stop at step 11 left the AdamW moments, the LR-schedule position, the RNG
streams and the step counter to die with the subprocess: restarting the same
configuration began again at step 1. This module is the single writer and the single
reader of the state that makes a run continuable.

Layout, under the run's own ``output_dir`` (so the resumed run keeps writing into the
same folder the adapter is published from)::

    <output_dir>/
        pytorch_lora_weights.safetensors   <- the deployable adapter (unchanged)
        checkpoint-11/
            trainer_state.json             <- manifest + completion marker
            adapter_model.safetensors      <- peft-format LoRA tensors
            ema_adapter.safetensors        <- the EMA shadow, when ema_decay > 0
            optimizer.pt                   <- optimizer.state_dict()
            scheduler.pt                   <- lr_scheduler.state_dict()
            rng_state.pt                   <- torch CPU + per-device CUDA RNG states

``trainer_state.json`` is written LAST inside a hidden staging directory, and the whole
staging directory is then promoted with a single ``os.replace``. A process killed at any
point leaves either the previous checkpoint or a ``.tmp-checkpoint-*`` directory that no
scanner ever matches -- never a half-written ``checkpoint-<N>`` that looks valid.

Identity: every bundle records the training identity it belongs to (family, base repo +
revision, dataset fingerprint, LoRA targets/rank/alpha, precision). ``preflight_resume``
compares that against the incoming request and refuses a mismatch with a user-facing
reason, so the start route can reject BEFORE it evicts the resident GPU model.

``kind`` ("image" today) is carried in both the manifest and the identity so a future
video trainer can write and validate its own bundles through this same code without a
format change: a video checkpoint simply records ``kind = "video"`` and will not be
offered to an image run (or vice versa).
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import random
import re
import shutil
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Iterable

# Bumped only for a breaking layout change. A reader refuses a version it does not know.
CHECKPOINT_FORMAT = "unsloth-diffusion-checkpoint"
CHECKPOINT_VERSION = 1

CHECKPOINT_PREFIX = "checkpoint-"
# Hidden, and does NOT start with "checkpoint-", so neither glob("checkpoint-*") nor a user's
# file browser mistakes a half-written bundle for a real one.
_STAGING_PREFIX = ".tmp-checkpoint-"

TRAINER_STATE_FILENAME = "trainer_state.json"
ADAPTER_FILENAME = "adapter_model.safetensors"
EMA_FILENAME = "ema_adapter.safetensors"
OPTIMIZER_FILENAME = "optimizer.pt"
SCHEDULER_FILENAME = "scheduler.pt"
RNG_FILENAME = "rng_state.pt"

# How many checkpoint-<N> bundles to keep in a run's output dir. A LoRA bundle is a few MB, but
# a 10000-step run at save_steps=50 would otherwise leave 200 of them.
DEFAULT_SAVE_TOTAL_LIMIT = 2


class ResumeError(ValueError):
    """A resume request that cannot be honoured. The message is shown to the user."""


# ── identity ──────────────────────────────────────────────────────────────────
# Human labels for the identity fields that HARD-REJECT a resume, in report order.
_IDENTITY_LABELS: tuple[tuple[str, str], ...] = (
    ("kind", "training type"),
    ("family", "model family"),
    ("base_model", "base model"),
    ("base_revision", "base model revision"),
    ("dataset_fingerprint", "training images"),
    ("lora_target_modules", "LoRA target modules"),
    ("lora_rank", "LoRA rank"),
    ("lora_alpha", "LoRA alpha"),
    ("lora_dropout", "LoRA dropout"),
    ("cfg_dropout", "caption dropout"),
    ("precision", "mixed precision"),
    ("base_precision", "base precision"),
)
# Fields whose value can legitimately be unknown on one side (an uncached Hub repo has no
# resolvable revision; the start route knows the dataset only after its own discovery pass).
# Unknown on either side means "cannot tell", which must not be reported as a mismatch.
_OPTIONAL_IDENTITY_FIELDS = frozenset(
    {"base_revision", "dataset_fingerprint", "lora_dropout", "cfg_dropout"}
)
# What source_revision() returns when it cannot resolve a revision offline.
_UNRESOLVED_REVISION = "unresolved"


def _revision_is_comparable(value: Any) -> bool:
    """Only a real Hub commit (``rev-<sha>``) is a hard identity for the base model.

    ``source_revision`` also returns a ``dir-<hash>`` built from the file SIZES AND MTIMES of a
    LOCAL base directory. That is the right key for a conditioning cache, but far too brittle for
    a resume gate: re-downloading, re-quantizing, or merely touching the local base changes it
    while the weights are the same, and the user would get an unactionable
    "different base model revision (dir-abc vs dir-def)" refusal. Compare Hub revisions, treat a
    local directory's marker as advisory."""
    return isinstance(value, str) and value.startswith("rev-")


@dataclass(frozen = True)
class CheckpointIdentity:
    """What a checkpoint was trained as. Two bundles are interchangeable only when every
    field here agrees, so resuming can never continue a FLUX run into an SDXL adapter, or
    feed rank-16 moments to a rank-32 optimizer.

    ``base_revision`` and ``dataset_fingerprint`` are optional: the first is resolved from
    the local Hub cache and reads ``unresolved`` when the repo has not been fetched yet, the
    second is only known once the dataset has been walked. Either being unknown skips that
    comparison rather than failing it."""

    family: str
    base_model: str
    lora_target_modules: tuple[str, ...]
    lora_rank: int
    lora_alpha: int
    precision: str
    base_precision: str
    resolution: int
    kind: str = "image"
    base_revision: Optional[str] = None
    dataset_fingerprint: Optional[str] = None
    lora_dropout: Optional[float] = None
    # Caption (classifier-free-guidance) dropout. The DiT loop asks rng.random() once per
    # sample when it is above zero and swaps in the empty prompt on a hit, so changing it
    # across a resume diverges the restored RNG stream on the very next step AND changes the
    # objective, while the run reports a clean continue. A bundle from before this field
    # existed reads None, which the optional rule skips rather than reporting as a mismatch.
    cfg_dropout: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "family": self.family,
            "base_model": self.base_model,
            "base_revision": self.base_revision,
            "dataset_fingerprint": self.dataset_fingerprint,
            "lora_target_modules": list(self.lora_target_modules),
            "lora_rank": int(self.lora_rank),
            "lora_alpha": int(self.lora_alpha),
            # Both trainers pass this to the LoRA constructor, so it changes the stochastic
            # forward pass the restored optimizer moments were produced against. Written as a
            # rounded float: a manifest from before this field existed reads as None, which the
            # optional-field rule below skips rather than reporting as a mismatch.
            "lora_dropout": self.lora_dropout,
            "cfg_dropout": self.cfg_dropout,
            "precision": self.precision,
            "base_precision": self.base_precision,
            # Recorded for the UI / debugging only; a resolution change is a different training
            # regime but leaves the adapter tensors loadable, so it is not a hard reject.
            "resolution": int(self.resolution),
        }

    @classmethod
    def from_dict(cls, data: Any) -> Optional["CheckpointIdentity"]:
        """Rebuild an identity from a manifest. Returns None for anything unreadable, so a
        hand-edited or truncated record reads as "no identity" instead of raising."""
        if not isinstance(data, dict):
            return None
        try:
            targets = data.get("lora_target_modules") or []
            if not isinstance(targets, (list, tuple)):
                return None
            return cls(
                family = str(data.get("family") or ""),
                base_model = str(data.get("base_model") or ""),
                lora_target_modules = tuple(str(t) for t in targets),
                lora_rank = int(data.get("lora_rank") or 0),
                lora_alpha = int(data.get("lora_alpha") or 0),
                lora_dropout = _optional_float(data.get("lora_dropout")),
                precision = str(data.get("precision") or ""),
                base_precision = str(data.get("base_precision") or ""),
                resolution = int(data.get("resolution") or 0),
                kind = str(data.get("kind") or "image"),
                base_revision = _optional_str(data.get("base_revision")),
                dataset_fingerprint = _optional_str(data.get("dataset_fingerprint")),
            )
        except (TypeError, ValueError):
            return None

    def with_dataset(self, fingerprint: Optional[str]) -> "CheckpointIdentity":
        """A copy carrying the dataset fingerprint, filled in once the images are known."""
        return replace(self, dataset_fingerprint = fingerprint)

    def mismatch_reason(self, other: "CheckpointIdentity") -> Optional[str]:
        """Why ``other`` (the incoming request) cannot continue ``self`` (the checkpoint),
        or None when they are compatible. Reports the FIRST difference so the message names
        one concrete thing to change."""
        mine, theirs = self.as_dict(), other.as_dict()
        for field, label in _IDENTITY_LABELS:
            a, b = mine.get(field), theirs.get(field)
            # None / "" is "cannot tell". NOT falsiness: a lora_dropout of 0.0 is a real value
            # and the commonest one, so truthiness would skip exactly the comparison that
            # matters -- 0.0 against 0.15.
            if field in _OPTIONAL_IDENTITY_FIELDS and (a in (None, "") or b in (None, "")):
                continue
            if field == "base_revision" and not (
                _revision_is_comparable(a) and _revision_is_comparable(b)
            ):
                continue
            if a == b:
                continue
            if field == "dataset_fingerprint":
                return (
                    "The training images have changed since this checkpoint was written, so "
                    "the run cannot continue from it. Restore the original dataset, or start "
                    "a new run."
                )
            return (
                f"This checkpoint was trained with a different {label} "
                f"({_render(a)} vs {_render(b)}), so it cannot be resumed into this run."
            )
        return None


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: Any) -> Optional[float]:
    """A rounded float, or None for anything unreadable (including a manifest that predates
    the field). None is the "cannot tell" the optional-field rule skips."""
    if value is None:
        return None
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return None


def _render(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) or "none"
    return str(value) if value not in (None, "") else "unset"


def dataset_fingerprint(pairs: Any) -> str:
    """A stable content marker for the (image, caption) pairs a run trains on.

    Built from each image's FILE NAME, byte size and resolved caption, sorted, so it is
    invariant to the dataset folder being moved or re-imported under a different root
    (which the run config already records) while still catching an added, removed,
    re-captioned or replaced image. Never raises: an unstatable file contributes ``?``
    rather than failing a preflight."""
    parts: list[str] = []
    for entry in pairs or ():
        try:
            path, caption = entry[0], entry[1]
        except (IndexError, KeyError, TypeError):
            continue
        parts.append(f"{Path(str(path)).name}:{_content_probe(path)}:{caption}")
    digest = hashlib.sha256("|".join(sorted(parts)).encode("utf-8", "replace")).hexdigest()
    return f"ds-{len(parts)}-{digest[:24]}"


# Enough of each file to tell two images apart, without reading a multi-gigabyte dataset on a
# preflight that runs before the resident GPU model is evicted.
_PROBE_BYTES = 65536


def _content_probe(path: Any) -> str:
    """``size-digest`` for one dataset file, or ``?`` when it cannot be read.

    Size alone let an image be overwritten IN PLACE with different content of exactly the same
    length -- same filename, same caption -- and the preflight accepted the dataset, so the
    restored optimizer and scheduler carried on an old experiment against different images.

    The head and tail rather than the whole file: hashing every image would make this scale with
    the dataset, and two different images agreeing on their first and last 64 KiB as well as
    their exact byte length is not a case that arises from editing a dataset. It is not a
    tamper-proof digest and does not need to be."""
    try:
        size = os.path.getsize(path)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            digest.update(handle.read(_PROBE_BYTES))
            if size > _PROBE_BYTES:
                # Every byte past the head is covered, either as the tail sample or, for a file
                # under 2x the probe, in full. The old `> 2 * _PROBE_BYTES` gate left a band
                # (64 KiB to 128 KiB, which is most JPEGs) reading its first 64 KiB and nothing
                # else, so a same-length replacement sharing a head went unnoticed.
                handle.seek(max(_PROBE_BYTES, size - _PROBE_BYTES))
                digest.update(handle.read(_PROBE_BYTES))
    # ValueError: open() rejects an embedded NUL rather than raising OSError.
    except (OSError, ValueError):
        return "?"
    return f"{size}-{digest.hexdigest()[:16]}"


def _resolve_lora_targets(cfg: Any) -> tuple[str, ...]:
    """The LoRA target modules the trainer will actually attach for ``cfg``.

    A DiT family replaces the generic attention default with its own projections
    (``_select_lora_targets``), so the identity must resolve the same tuple the trainer
    will; otherwise the start route would compute a different fingerprint from the run and
    every resume would look like a mismatch. Imported lazily -- the DiT trainer imports this
    module at load time, so the reverse edge has to be deferred."""
    configured = tuple(cfg.lora_target_modules)
    if str(getattr(cfg, "resolved_family", "") or "").strip().lower() == "sdxl":
        return configured
    # Deliberately NOT wrapped in a try/except: quietly falling back to the generic tuple would
    # make the route compute a different fingerprint from the trainer, its preflight would pass,
    # and the mismatch would surface in the child AFTER the GPU residents were freed.
    from core.training.diffusion_dit_trainer import _SPECS, _select_lora_targets

    spec = _SPECS.get(cfg.resolved_family)
    if spec is None:
        return configured
    return tuple(_select_lora_targets(configured, spec.lora_targets))


def with_resolved_revision(identity: "CheckpointIdentity", base_model: Any) -> "CheckpointIdentity":
    """``identity`` with its base revision re-read now that the base model is on disk.

    The identity is built BEFORE the multi-GB load, deliberately -- a mismatched resume should
    fail in seconds, not after a download. But on the first run of an uncached Hub repo there is
    no local ref to read, so the revision records "unresolved" and the bundle it goes into can
    never enforce which commit it was trained on. Re-read once the loader has populated the
    cache: an unresolved value is not comparable, so this only ever ADDS the constraint, and a
    later resume against a repo that has advanced is refused instead of quietly restoring the
    adapter and the Adam moments onto different frozen base weights.
    """
    from core.training.diffusion_train_extras import source_revision

    if _revision_is_comparable(identity.base_revision):
        return identity
    resolved = source_revision(base_model)
    if not _revision_is_comparable(resolved):
        return identity
    return replace(identity, base_revision = resolved)


def identity_for_config(
    cfg: Any,
    *,
    dataset_pairs: Any = None,
    resolved_targets: Optional[tuple[str, ...]] = None,
    kind: str = "image",
) -> CheckpointIdentity:
    """The identity a run with ``cfg`` would produce.

    ``resolved_targets`` is the LoRA target tuple the trainer will actually attach; pass it
    where it is already known (the DiT trainer holds its family spec), otherwise it is
    resolved here the same way, so the start route and the trainer always agree.
    ``base_precision`` is the REQUESTED mode, not the one ``auto`` resolves to at load time
    -- the start route has to compute the same identity before anything is loaded."""
    from core.training.diffusion_train_common import effective_mixed_precision
    from core.training.diffusion_train_extras import source_revision

    targets = tuple(resolved_targets) if resolved_targets else _resolve_lora_targets(cfg)
    return CheckpointIdentity(
        family = str(getattr(cfg, "resolved_family", "") or ""),
        base_model = str(cfg.base_model or ""),
        lora_target_modules = targets,
        lora_rank = int(cfg.lora_rank),
        lora_alpha = int(cfg.lora_alpha if cfg.lora_alpha is not None else cfg.lora_rank),
        lora_dropout = round(float(getattr(cfg, "lora_dropout", 0.0) or 0.0), 6),
        cfg_dropout = round(float(getattr(cfg, "cfg_dropout", 0.0) or 0.0), 6),
        # The EFFECTIVE precision, not the request: a pre-Ampere card resolves bf16 to fp16, and
        # recording the request let an fp16 bundle resume in bf16 on a newer card -- restored
        # moments continuing under different frozen-base numerics, reported as a clean resume.
        precision = effective_mixed_precision(cfg),
        base_precision = str(getattr(cfg, "base_precision", "") or ""),
        resolution = int(cfg.resolution),
        kind = kind,
        base_revision = source_revision(cfg.base_model),
        dataset_fingerprint = dataset_fingerprint(dataset_pairs) if dataset_pairs else None,
    )


# ── RNG capture / restore ─────────────────────────────────────────────────────
def capture_rng_state(streams: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Snapshot every random stream a diffusion run draws from.

    ``streams`` maps a name to a ``random.Random`` instance the trainer owns (the loop's
    index/crop stream and the latent-cache variant stream are separate objects, so the
    module-level ``random`` state does not cover them). Returns
    ``{"json": <JSON-safe dict>, "tensors": <torch tensors>}``; the caller writes the first
    into the manifest and the second into ``rng_state.pt``. Never raises."""
    payload: dict[str, Any] = {
        "python": _random_state_to_json(random.getstate()),
        "streams": {},
        "numpy": None,
    }
    for name, stream in (streams or {}).items():
        try:
            payload["streams"][str(name)] = _random_state_to_json(stream.getstate())
        except Exception:  # noqa: BLE001 -- a stream we cannot read simply is not restored
            continue
    try:
        import numpy as np
        kind, keys, pos, has_gauss, cached = np.random.get_state()
        payload["numpy"] = {
            "bit_generator": str(kind),
            "keys": [int(k) for k in keys],
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached_gaussian": float(cached),
        }
    except Exception:  # noqa: BLE001 -- no numpy / a non-MT19937 global generator
        payload["numpy"] = None

    tensors: dict[str, Any] = {}
    try:
        import torch
        tensors["torch_cpu"] = torch.get_rng_state()
        if torch.cuda.is_available():
            for i, state in enumerate(torch.cuda.get_rng_state_all()):
                tensors[f"torch_cuda_{i}"] = state
    except Exception:  # noqa: BLE001 -- torch RNG capture is best-effort
        pass
    return {"json": payload, "tensors": tensors}


def restore_rng_state(
    payload: Optional[dict[str, Any]],
    tensors: Optional[dict[str, Any]] = None,
    streams: Optional[dict[str, Any]] = None,
) -> None:
    """Undo ``capture_rng_state``. Every part is independent and best-effort: a checkpoint
    written on a 2-GPU box restored on a 1-GPU box still restores everything else."""
    payload = payload or {}
    state = _random_state_from_json(payload.get("python"))
    if state is not None:
        try:
            random.setstate(state)
        except (TypeError, ValueError):
            pass
    saved_streams = payload.get("streams")
    if isinstance(saved_streams, dict):
        for name, stream in (streams or {}).items():
            got = _random_state_from_json(saved_streams.get(str(name)))
            if got is None:
                continue
            try:
                stream.setstate(got)
            except (TypeError, ValueError):
                continue
    np_state = payload.get("numpy")
    if isinstance(np_state, dict):
        try:
            import numpy as np
            np.random.set_state(
                (
                    str(np_state.get("bit_generator") or "MT19937"),
                    np.array(np_state.get("keys") or [], dtype = np.uint32),
                    int(np_state.get("pos") or 0),
                    int(np_state.get("has_gauss") or 0),
                    float(np_state.get("cached_gaussian") or 0.0),
                )
            )
        except Exception:  # noqa: BLE001 -- no numpy, or an incompatible generator
            pass
    if not tensors:
        return
    try:
        import torch

        cpu = tensors.get("torch_cpu")
        if cpu is not None:
            torch.set_rng_state(cpu.cpu().to(torch.uint8))
        if torch.cuda.is_available():
            # Per device, not set_rng_state_all. That call needs one state per visible
            # device, so covering it with an all-or-nothing guard meant a bundle written
            # with fewer devices visible than the resume sees restored NOTHING -- including
            # cuda:0, the only device training touches. The whole per-step noise stream
            # (randn_like for the latent and the noise, randint for the timestep) then
            # restarted at the wrong offset while every other part of the resume looked
            # correct, so the run finished healthy and silently wrong: measured on a real
            # B200 SDXL run, 192/192 LoRA tensors differed from the uninterrupted control
            # and the first step after the resume was off by 0.024. The exposure is
            # ordinary -- the trainer is a spawned child inheriting CUDA_VISIBLE_DEVICES,
            # which Studio masks elsewhere, so a mask change between the run and the Resume
            # click was enough. Restoring device by device covers whatever the bundle
            # holds and leaves the rest alone, and is identical to the old behaviour when
            # the counts match (verified bit-exact against the same control).
            for i in range(torch.cuda.device_count()):
                state = tensors.get(f"torch_cuda_{i}")
                if state is None:
                    continue
                torch.cuda.set_rng_state(state.cpu().to(torch.uint8), i)
    except Exception:  # noqa: BLE001 -- best-effort restore, never fatal
        pass


def _random_state_to_json(state: Any) -> Optional[list[Any]]:
    """``random.Random.getstate()`` is ``(version, tuple[int, ...], gauss)``; JSON has no
    tuples, so store it as nested lists."""
    try:
        version, keys, gauss = state
        return [int(version), [int(k) for k in keys], gauss]
    except (TypeError, ValueError):
        return None


def _random_state_from_json(value: Any) -> Optional[tuple]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (int(value[0]), tuple(int(k) for k in value[1]), value[2])
    except (TypeError, ValueError):
        return None


# ── writing ───────────────────────────────────────────────────────────────────
def save_checkpoint(
    *,
    output_dir: str | os.PathLike[str],
    step: int,
    adapter_state: dict[str, Any],
    identity: CheckpointIdentity,
    target_steps: int,
    optimizer: Any = None,
    lr_scheduler: Any = None,
    ema_state: Optional[dict[str, Any]] = None,
    ema_updates: int = 0,
    rng: Optional[dict[str, Any]] = None,
    sampler_state: Optional[dict[str, Any]] = None,
    progress: Optional[dict[str, Any]] = None,
    save_total_limit: int = DEFAULT_SAVE_TOTAL_LIMIT,
    discard_existing: bool = False,
    source_checkpoint: Optional[str | os.PathLike[str]] = None,
) -> str:
    """Write one resumable ``checkpoint-<step>`` bundle and return its path.

    The ONE writer: periodic saves (``cfg.save_steps``) and stop-and-save both come through
    here, so a stopped run and a crashed-then-restarted run resume from byte-identical
    state. Everything lands in a hidden staging directory whose manifest is written last;
    the directory is then promoted with a single ``os.replace``, so a kill at any instant
    leaves the previous checkpoint intact and no valid-looking partial behind.

    ``source_checkpoint`` is the bundle this run resumed FROM, when it resumed. It is what
    makes the "this step is already written" shortcut below safe: without it, a stop at a
    step some OTHER run happens to have written would silently keep that run's state.

    ``discard_existing`` drops any bundle already in the directory first. A run that did NOT
    resume owns its output dir outright (it overwrites the published adapter there too), so
    checkpoints left by an EARLIER run of the same adapter name must go: they would otherwise
    outrank the new run's lower-numbered ones and a later Resume would silently continue the
    wrong training.

    Raises OSError / ValueError / RuntimeError on a real write failure -- callers treat that
    as "this run cannot be resumed" and surface it, rather than failing the training."""
    import torch
    from safetensors.torch import save_file

    step = int(step)
    if step < 0:
        raise ValueError("checkpoint step must be >= 0")
    if not adapter_state:
        # An empty safetensors file has no keys, so the bundle would fail its own validation and
        # read as "no checkpoint" -- a silent un-resumable run. Fail loudly at write time instead.
        raise ValueError("refusing to write a checkpoint with no adapter tensors")
    root = Path(output_dir).expanduser()
    root.mkdir(parents = True, exist_ok = True)
    doomed: list[Path] = []
    if discard_existing:
        # Recorded now, deleted AFTER the new bundle is promoted. Clearing first means a kill or
        # an I/O failure mid-write leaves the directory with no checkpoint at all -- and any
        # earlier run sharing the folder has irreversibly lost its own, which is the opposite of
        # what this writer promises.
        doomed = list_checkpoints(root)
    else:
        # Re-reaching a step that already has a VALID bundle happens one way SAFELY: the run
        # resumed at N and stopped before N+1 completed, so its state is byte-identical to what
        # is already there, and keeping it avoids the one destructive branch in _promote
        # (swapping a good bundle out to make room), where a kill would leave the slot empty.
        #
        # It has to be THAT bundle, though. Resuming checkpoint-10 in a folder that also holds
        # checkpoint-15 and stopping at 15 is the same arithmetic and a completely different
        # situation: the freshly trained optimizer, scheduler, sampler and RNG would be dropped
        # on the floor and checkpoint_saved would name a bundle from an earlier run.
        existing = root / f"{CHECKPOINT_PREFIX}{step}"
        if (
            source_checkpoint is not None
            and Path(source_checkpoint).expanduser().resolve() == existing.resolve()
            and read_checkpoint(existing) is not None
        ):
            return str(existing)
    staging = root / f"{_STAGING_PREFIX}{step}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents = True, exist_ok = False)

    try:
        _save_tensors(save_file, adapter_state, staging / ADAPTER_FILENAME)
        files = {"adapter": ADAPTER_FILENAME}
        optimizer_class: Optional[str] = None
        if ema_state:
            _save_tensors(save_file, ema_state, staging / EMA_FILENAME)
            files["ema"] = EMA_FILENAME
        if optimizer is not None:
            # torch.save, not safetensors: an optimizer state dict is a nested structure of
            # tensors AND scalars (bitsandbytes AdamW8bit additionally carries uint8 moments
            # plus their quantization maps), which safetensors cannot express.
            _torch_save(torch, optimizer.state_dict(), staging / OPTIMIZER_FILENAME)
            files["optimizer"] = OPTIMIZER_FILENAME
            optimizer_class = optimizer_key(optimizer)
        if lr_scheduler is not None:
            _torch_save(torch, lr_scheduler.state_dict(), staging / SCHEDULER_FILENAME)
            files["scheduler"] = SCHEDULER_FILENAME
        rng_json: Optional[dict[str, Any]] = None
        if rng:
            rng_json = rng.get("json")
            rng_tensors = rng.get("tensors") or {}
            if rng_tensors:
                _torch_save(torch, rng_tensors, staging / RNG_FILENAME)
                files["rng"] = RNG_FILENAME

        manifest: dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "kind": identity.kind,
            "global_step": step,
            "target_steps": int(target_steps),
            # Checkpoints are only ever taken on an optimizer-step boundary, so the
            # gradient-accumulation position is always clean. Recorded explicitly so a future
            # mid-accumulation checkpoint is a value change, not a format change.
            "micro_step": 0,
            "created_at": time.time(),
            "identity": identity.as_dict(),
            "sampler": sampler_state or None,
            "rng": rng_json,
            "ema_updates": int(ema_updates),
            # Which optimizer wrote the moments. bitsandbytes AdamW8bit stores "state1"/"state2",
            # torch AdamW stores "exp_avg"/"exp_avg_sq", and the trainers pick between them from
            # the host (bnb present, fused kernel available, UNSLOTH_DIFFUSION_FP32_OPTIM). Both
            # accept the other's state_dict and then KeyError on the first step, so the resume
            # has to compare this instead.
            "optimizer_class": optimizer_class,
            # Loop bookkeeping that is not state to restore into an object (running loss, wall
            # clock). Nested rather than merged, so a caller can never shadow a reserved key.
            "progress": dict(progress or {}),
            "files": files,
            # Byte size per listed file, so read_checkpoint can catch a truncation the
            # header parse structurally cannot. _valid_state_file walks a torch zip's pickle
            # to STOP and requires one non-empty data/ member; truncating the tensor storages
            # themselves leaves both intact, and torch.load then returns UNINITIALIZED memory
            # -- measured at +/-1e22 and non-finite -- which a resume feeds straight into the
            # optimizer as Adam moments. The run diverges on its first step while reporting a
            # clean resume. This is free to write and free to check.
            #
            # Additive on purpose: an older bundle has no sizes and skips the check rather
            # than failing it, and an older build ignores the key, so no version bump.
            "file_sizes": _file_sizes(staging, files),
        }
        # LAST: the manifest is the completion marker, so nothing may be written after it.
        _write_text(staging / TRAINER_STATE_FILENAME, json.dumps(manifest, indent = 2))
        _fsync_dir(staging)
        final = _promote(staging, root, step)
    except BaseException:
        shutil.rmtree(staging, ignore_errors = True)
        raise
    for stale in doomed:
        # `final` can be one of these when the new step reuses an existing bundle's name;
        # _promote already replaced it, so removing it here would delete the new checkpoint.
        if stale != final:
            shutil.rmtree(stale, ignore_errors = True)
    _prune_staging(root)
    prune_checkpoints(root, keep = save_total_limit, protect = final)
    return str(final)


def optimizer_key(optimizer: Any) -> str:
    """A stable name for an optimizer implementation, e.g. ``bitsandbytes.optim.adamw.AdamW8bit``.
    Fused vs. non-fused torch AdamW share a class and a state layout, so they compare equal."""
    cls = type(optimizer)
    return f"{getattr(cls, '__module__', '?')}.{getattr(cls, '__qualname__', cls.__name__)}"


def _save_tensors(save_file: Any, state: dict[str, Any], path: Path) -> None:
    """safetensors refuses tensors that share storage, so detach/clone every entry onto CPU
    (a LoRA state dict is megabytes, and this runs at most once per save_steps)."""
    payload = {
        str(k): v.detach().to("cpu", copy = True).contiguous()
        for k, v in (state or {}).items()
        if v is not None
    }
    save_file(payload, str(path))
    _fsync_file(path)


def _torch_save(torch: Any, obj: Any, path: Path) -> None:
    torch.save(obj, str(path))
    _fsync_file(path)


def _file_sizes(staging: Path, files: dict[str, str]) -> dict[str, int]:
    """``{role: bytes}`` for everything already written into the staging dir."""
    sizes: dict[str, int] = {}
    for role, name in files.items():
        try:
            sizes[role] = int((staging / name).stat().st_size)
        except OSError:
            continue
    return sizes


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding = "utf-8")
    _fsync_file(path)


# errno values that mean "this platform / filesystem will not flush that handle" rather than
# "the data did not make it". Windows' _commit needs write access, network and container
# filesystems return EINVAL/ENOTSUP for fsync, and neither says anything about the write.
_FSYNC_UNSUPPORTED = frozenset(
    code
    for code in (
        getattr(errno, name, None)
        for name in ("EACCES", "EPERM", "EINVAL", "ENOTSUP", "EOPNOTSUPP", "EBADF", "ENOSYS")
    )
    if code is not None
)


def _fsync_file(path: Path) -> None:
    """Flush the file to the device. Without it a host crash (not just a process kill) can
    promote a directory whose manifest is durable but whose tensors are still in page cache.

    A REAL fsync failure is raised, not swallowed: delayed allocation means ENOSPC and writeback
    EIO surface here rather than at write() time, and ``_valid_state_file`` only parses a
    safetensors header, so a bundle whose tensor bytes never reached the device would be
    promoted and later read as valid. The caller turns that into "this run cannot be resumed".
    A platform that simply cannot flush the handle is not such a failure and is ignored.

    That discrimination is real on POSIX and NOT available on Windows: CPython maps os.fsync
    to the UCRT's _commit, which collapses every FlushFileBuffers failure into EBADF and puts
    the actual Win32 code in _doserrno, where Python does not look. EBADF has to stay in the
    set below, because a volume that cannot flush reports it too, and failing every save on
    such a volume is the worse error -- especially as the resulting write error is sticky for
    the life of the run record. So on Windows a genuine flush failure is swallowed here, and
    the durability guarantee rests on the per-file size check in read_checkpoint instead,
    which catches the truncation a lost writeback actually produces."""
    try:
        # O_RDWR, not O_RDONLY: Windows' _commit maps to FlushFileBuffers, which needs write
        # access on the handle. We just wrote the file, so we own it.
        fd = os.open(str(path), os.O_RDWR)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError as error:
        if error.errno not in _FSYNC_UNSUPPORTED:
            raise
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):  # Windows cannot open a directory as a file descriptor
        return
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _promote(staging: Path, root: Path, step: int) -> Path:
    """Rename the staging directory into place.

    ``os.replace`` onto an EXISTING directory is not supported (POSIX rename() needs an empty
    target; Windows MoveFileEx rejects it outright), so an occupied slot is first swapped out
    to a staging name. The old bundle is deleted only AFTER the new one is in place, so a kill
    during the swap leaves the old one recoverable on disk rather than already gone."""
    final = root / f"{CHECKPOINT_PREFIX}{step}"
    doomed: Optional[Path] = None
    if final.exists():
        doomed = root / f"{_STAGING_PREFIX}stale-{step}-{uuid.uuid4().hex[:8]}"
        try:
            os.replace(final, doomed)
        except OSError:
            # Cannot move it aside (a lock, a cross-device oddity): removing it is the only way
            # to free the slot, and it is about to be superseded anyway.
            doomed = None
            shutil.rmtree(final, ignore_errors = True)
    try:
        os.replace(staging, final)
    except OSError:
        # The slot is now empty and the only copy of the last resumable state is the one we
        # moved aside, so put it back before the failure propagates. Without this, a resave of
        # an occupied step (a resumed run rewriting checkpoint-15) that fails to promote leaves
        # the run with no checkpoint at all -- strictly worse than the stale one it replaced.
        if doomed is not None:
            with contextlib.suppress(OSError):
                os.replace(doomed, final)
        raise
    _fsync_dir(root)
    if doomed is not None:
        shutil.rmtree(doomed, ignore_errors = True)
    return final


# ``.tmp-checkpoint-stale-<step>-<uuid>``: the step is what _promote encodes, so an orphan can
# be handed back to the slot it came from.
_STALE_SLOT = re.compile(r"^stale-(\d+)-")


def _prune_staging(root: Path) -> None:
    """Drop abandoned staging directories from an earlier killed process. Safe because only
    one trainer ever writes into a run's output dir, and this runs after our own promotion.

    A ``stale-<step>`` orphan whose slot is EMPTY is not abandoned work: it is the previous
    bundle, moved aside by a promotion that was killed before the rename landed. Deleting it
    would throw away the run's last resumable state, so it is handed back to its slot instead.
    """
    try:
        entries = list(root.glob(f"{_STAGING_PREFIX}*"))
    except OSError:
        return
    for entry in entries:
        match = _STALE_SLOT.match(entry.name[len(_STAGING_PREFIX) :])
        if match is not None:
            slot = root / f"{CHECKPOINT_PREFIX}{int(match.group(1))}"
            if not slot.exists():
                try:
                    os.replace(entry, slot)
                except OSError:
                    # Could not hand it back. Leaving it on disk is still better than
                    # deleting the only copy of the run's last resumable state.
                    pass
                continue
        shutil.rmtree(entry, ignore_errors = True)


def prune_checkpoints(
    output_dir: str | os.PathLike[str],
    keep: int = DEFAULT_SAVE_TOTAL_LIMIT,
    *,
    protect: Optional[Path] = None,
) -> None:
    """Keep only the ``keep`` newest ``checkpoint-<N>`` bundles. ``keep <= 0`` keeps all.

    ``protect`` is never pruned. Newest is by STEP, and a resume can legitimately write a bundle
    that is not the highest-numbered one in the folder: resume checkpoint-10, stop at 15, with 20
    and 30 still present and keep=2, and the bundle just written is the one deleted -- while the
    service reports checkpoint_saved for a path that no longer exists, and the run's own start
    fence stops those older bundles from making it resumable. So the caller pins the one it just
    promoted, and the limit applies to the rest."""
    if keep <= 0:
        return
    survivors = list_checkpoints(output_dir)
    if protect is not None and protect in survivors:
        # It occupies one of the kept slots whether or not it sorted into the top `keep`.
        survivors = [c for c in survivors if c != protect]
        keep = max(0, keep - 1)
    for stale in survivors[keep:]:
        shutil.rmtree(stale, ignore_errors = True)


def clear_checkpoints(output_dir: str | os.PathLike[str]) -> None:
    """Remove every ``checkpoint-<N>`` bundle in ``output_dir``. Used when a fresh (non-resumed)
    run takes over an output directory that an earlier run of the same name left checkpoints in."""
    for stale in list_checkpoints(output_dir):
        shutil.rmtree(stale, ignore_errors = True)


def resumed_into_this_dir(cfg: Any, output_dir: "str | os.PathLike[str]") -> bool:
    """Whether the bundle this run resumed FROM lives in ``output_dir``.

    The usual case, and the one the first-save discard is skipped for: a resumed run shares its
    source's directory, so clearing "everything that was here" would take the source with it.
    An API caller can instead resume from directory A into a reused output_dir B, and then B's
    contents are as foreign as they would be to a fresh run -- keeping them lets a higher-step
    bundle outlive this run and be picked by a later resume by directory."""
    source = getattr(cfg, "resume_from_checkpoint", None)
    if not source:
        return False
    try:
        root = Path(output_dir).expanduser().resolve()
        candidate = Path(str(source)).expanduser().resolve()
    except OSError:
        return False
    # The request names either the bundle itself or the directory holding it.
    return candidate == root or candidate.parent == root


def snapshot_checkpoints(output_dir: str | os.PathLike[str]) -> list[tuple[Path, Optional[tuple]]]:
    """The bundles in ``output_dir`` right now, each with its identity.

    Captured BEFORE the run's first write, because ownership cannot be decided by pathname at
    cleanup time: a resumed run can overwrite a bundle that was already there, and by then the
    directory holds this run's state under the old name."""
    return [(path, _bundle_identity(path)) for path in list_checkpoints(output_dir)]


def clear_own_checkpoints(output_dir: str | os.PathLike[str], preexisting: "Iterable[Any]") -> None:
    """Remove the bundles THIS run wrote, leaving the ones it found.

    A discard must not take the checkpoint the run resumed FROM with it: a resumed run writes
    into the same directory the source bundle lives in, so an accidental resume followed by
    "stop without saving" would otherwise leave the original stopped run unresumable -- the one
    thing the user was trying not to disturb. Bundles are identified by the set captured before
    the run's first write, not by step number, because a resume writes lower numbers than the
    ones already there."""
    # Keyed by path AND by the bundle's own identity, not by path alone. A resumed run can
    # periodically save at a step whose directory was already there (resume checkpoint-10 in a
    # folder that also holds checkpoint-15, save at 15), which REPLACES that bundle: the
    # original is gone either way, and matching on the name alone then preserved the discarded
    # run's replacement as though it were the bundle it overwrote.
    keep: dict[Path, Optional[tuple]] = {}
    for entry in preexisting:
        # A bare path (an older caller) keeps the pathname-only behaviour for that entry.
        if isinstance(entry, tuple):
            path, identity = entry
            keep[Path(path)] = identity
        else:
            keep[Path(entry)] = _bundle_identity(Path(entry))
    for stale in list_checkpoints(output_dir):
        if stale in keep and keep[stale] == _bundle_identity(stale):
            continue
        shutil.rmtree(stale, ignore_errors = True)


def _bundle_identity(path: Path) -> Optional[tuple]:
    """What distinguishes one bundle at this path from another written over it.

    The manifest is written last and carries the writing run's own start time, so it separates
    "the bundle that was here" from "the bundle this run put here" without hashing tensors.
    None for an unreadable or absent manifest, which compares equal to itself and so leaves an
    unreadable pre-existing directory alone."""
    try:
        manifest = json.loads((path / TRAINER_STATE_FILENAME).read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    # created_at is the moment THIS bundle's manifest was written -- the completion marker, so
    # it exists on every valid bundle and differs between two writes at the same step.
    return (manifest.get("created_at"), manifest.get("global_step"))


# ── reading + validation ──────────────────────────────────────────────────────
def checkpoint_step(path: Path) -> int:
    """The step encoded in a ``checkpoint-<N>`` directory name, or -1."""
    name = path.name
    if not name.startswith(CHECKPOINT_PREFIX):
        return -1
    try:
        step = int(name[len(CHECKPOINT_PREFIX) :])
    except ValueError:
        return -1
    return step if step >= 0 else -1


def list_checkpoints(output_dir: str | os.PathLike[str]) -> list[Path]:
    """Every ``checkpoint-<N>`` directory under ``output_dir``, newest step first. Does not
    validate; use ``read_checkpoint`` / ``latest_valid_checkpoint`` for that."""
    root = Path(output_dir).expanduser()
    try:
        found = [p for p in root.glob(f"{CHECKPOINT_PREFIX}*") if p.is_dir()]
    except OSError:
        return []
    return sorted((p for p in found if checkpoint_step(p) >= 0), key = checkpoint_step, reverse = True)


def read_checkpoint(path: str | os.PathLike[str]) -> Optional[dict[str, Any]]:
    """The manifest of a COMPLETE, self-consistent bundle, or None.

    Every gate a resume depends on is checked here: the manifest parses and declares a
    format/version this build understands, its ``global_step`` agrees with the directory
    name, and every file the manifest lists is present and parses as the state file it
    claims to be. A staging directory promoted mid-write cannot pass, because the manifest
    is the last thing written."""
    directory = Path(path).expanduser()
    if not directory.is_dir():
        return None
    try:
        manifest = json.loads((directory / TRAINER_STATE_FILENAME).read_text(encoding = "utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("format") != CHECKPOINT_FORMAT:
        return None
    version = manifest.get("version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or not 1 <= version <= CHECKPOINT_VERSION
    ):
        return None
    step = manifest.get("global_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        return None
    named_step = checkpoint_step(directory)
    if named_step >= 0 and named_step != step:
        return None
    files = manifest.get("files")
    if not isinstance(files, dict) or not files.get("adapter"):
        return None
    sizes = manifest.get("file_sizes")
    sizes = sizes if isinstance(sizes, dict) else {}
    for role, name in files.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            return None
        # Optimizer / scheduler state can be validly tensor-free (SGD without momentum, a
        # constant LR schedule), so only the weight bundles must carry tensors.
        if not _valid_state_file(directory / name, require_tensor = role in ("adapter", "ema")):
            return None
        expected_size = sizes.get(role)
        if isinstance(expected_size, int) and not isinstance(expected_size, bool):
            # Absent on a bundle written before sizes were recorded, which skips the check
            # rather than failing it -- an old checkpoint stays resumable.
            try:
                if (directory / name).stat().st_size != expected_size:
                    return None
            except OSError:
                return None
    return manifest


def _valid_state_file(path: Path, require_tensor: bool = True) -> bool:
    """Reuse the LLM resume validator: it already parses a safetensors header and walks a
    torch zip's pickle to the STOP opcode without importing torch.

    Wrapped in a blanket catch because a resume path is CLIENT-supplied: that validator only
    guards ``(OSError, ValueError, BadZipFile)``, so a deflate-corrupt member raises
    ``zlib.error`` straight out of a route preflight and 500s it. Anything unreadable is, by
    definition, not a usable checkpoint."""
    from core.training.resume import _valid_state_file as _validate
    try:
        return _validate(path, require_tensor = require_tensor)
    except Exception:  # noqa: BLE001 -- unreadable in any way == not resumable
        return False


def latest_valid_checkpoint(
    output_dir: str | os.PathLike[str],
    not_before: Optional[float] = None,
    not_after: Optional[float] = None,
) -> Optional[tuple[Path, dict]]:
    """The newest complete bundle under ``output_dir`` as ``(path, manifest)``, or None.
    Scans newest-first and skips any bundle that fails validation, so one corrupt
    checkpoint does not hide the good one before it.

    ``not_before`` (a run's start time) skips bundles written BEFORE it. Two runs can share an
    output directory -- training the same adapter name twice writes into the same folder -- and
    the earlier run's bundles can carry higher step numbers than the later run's, so without
    this a fresh run would advertise, and then resume, a different run's training state.

    ``not_after`` (a run's end time) is the same fence in the other direction, and it matters
    just as much: with only a lower bound, an EARLIER run that has already finished still sees
    every bundle a LATER run wrote into the shared folder, and offers the newest of them as
    its own. The identity gate cannot catch that -- same family, same base, same dataset, same
    LoRA shape -- so the earlier run resumes the later run's optimizer moments, LR position
    and RNG under its own config, and records the wrong lineage while doing it."""
    for candidate in list_checkpoints(output_dir):
        manifest = read_checkpoint(candidate)
        if manifest is None:
            continue
        if not_before is not None:
            try:
                created = float(manifest.get("created_at") or 0.0)
            except (TypeError, ValueError):
                created = 0.0
            if created < float(not_before):
                continue
        if not_after is not None:
            try:
                created = float(manifest.get("created_at") or 0.0)
            except (TypeError, ValueError):
                created = 0.0
            # A bundle with no usable created_at (0.0) is not fenced out by the upper bound:
            # it predates the field, and refusing it would make an old run unresumable.
            if created and created > float(not_after):
                continue
        return candidate, manifest
    return None


# Run statuses that can never be continued, with the reason shown in the UI.
_UNRESUMABLE_STATUS = {
    "completed": "This run finished its full step count, so there is nothing left to train.",
    "running": "This run is still training.",
}


def _source_checkpoint_bundle(source_checkpoint) -> Optional[tuple[Path, dict[str, Any]]]:
    """The bundle a run RESUMED FROM, when it is still readable on disk.

    A resume that ends before writing a bundle of its own -- an OOM on the first restored step
    is the usual way -- has nothing under its own output dir, and if that dir is a new one it
    may not even exist. The source it was validated against is still there and still correct to
    continue from, so it is read directly rather than by widening the started_at fence, which
    exists to keep an unrelated earlier run's bundles out.
    """
    if not source_checkpoint:
        return None
    try:
        candidate = Path(str(source_checkpoint)).expanduser()
        manifest = read_checkpoint(candidate) if candidate.is_dir() else None
    except OSError:
        return None
    return (candidate, manifest) if manifest is not None else None


def describe_resume_state(
    output_dir: Optional[str],
    *,
    status: Optional[str] = None,
    started_at: Optional[float] = None,
    ended_at: Optional[float] = None,
    source_checkpoint: Optional[str] = None,
    total_steps: Optional[int] = None,
) -> dict[str, Any]:
    """What the UI needs to offer (or explain the absence of) a Resume action for a run.

    Returns ``can_resume`` / ``checkpoint_step`` / ``checkpoint_path`` /
    ``resume_blocked_reason``. Mirrors the LLM ``can_resume_run`` rules: a completed run has
    nothing left, a stopped run needs remaining steps, an errored run is resumable purely on
    the state it left behind. Never raises: a deleted or unreadable output directory simply
    reports that there is nothing to resume from."""
    blank: dict[str, Any] = {
        "can_resume": False,
        "checkpoint_step": None,
        "checkpoint_path": None,
        "resume_blocked_reason": None,
    }
    if not output_dir:
        return blank
    blocked = _UNRESUMABLE_STATUS.get(str(status or "").strip().lower())
    if blocked:
        return {**blank, "resume_blocked_reason": blocked}
    try:
        root = Path(str(output_dir)).expanduser()
        if not root.is_dir():
            # A resume into a NEW output dir that died before its first save never created it,
            # and the bundle it was validated against is still sitting in the source dir. The
            # fallback below is exactly for that, so it has to be reachable from here.
            recovered = _source_checkpoint_bundle(source_checkpoint)
            if recovered is None:
                return {
                    **blank,
                    "resume_blocked_reason": "This run's output folder no longer exists.",
                }
            found = recovered
        else:
            found = latest_valid_checkpoint(root, not_before = started_at, not_after = ended_at)
        if found is None and source_checkpoint:
            # This run RESUMED and then ended before writing a bundle of its own -- an OOM on
            # the first restored step is the usual way. The source it was validated against is
            # still on disk and still correct to continue from, but it predates started_at, so
            # the fence above excludes it and the run reads as unresumable when retrying is the
            # obvious thing to do. Read it directly rather than widening the fence, which exists
            # to stop an unrelated earlier run's bundles being offered.
            found = _source_checkpoint_bundle(source_checkpoint)
    except OSError:
        return blank
    if found is None:
        if not list_checkpoints(root):
            reason = "This run saved no resume checkpoint, so it cannot be continued."
        elif started_at is not None and latest_valid_checkpoint(root) is not None:
            # Bundles are there, but every one predates this run: they belong to an earlier run
            # that trained into the same folder. Saying "corrupt" here would be a lie.
            reason = (
                "This run saved no resume checkpoint of its own; the checkpoints in its folder "
                "were left by an earlier run of the same adapter."
            )
        else:
            reason = "This run's checkpoints are incomplete or corrupt, so it cannot be resumed."
        return {**blank, "resume_blocked_reason": reason}
    path, manifest = found
    step = int(manifest.get("global_step") or 0)
    # The target the resume would run to. The record's own total_steps wins (it is what the UI
    # replays); the manifest's copy covers a record written before that field existed.
    target = int(total_steps or manifest.get("target_steps") or 0)
    if target and step >= target:
        return {
            **blank,
            "checkpoint_step": step,
            "checkpoint_path": str(path),
            "resume_blocked_reason": (
                f"This run's checkpoint is already at step {step} of {target}, so there is "
                "nothing left to train."
            ),
        }
    return {
        "can_resume": True,
        "checkpoint_step": step,
        "checkpoint_path": str(path),
        "resume_blocked_reason": None,
    }


# ── resume preflight ──────────────────────────────────────────────────────────
def resolve_resume_dir(path_value: str) -> Path:
    """Contain a client-supplied resume path under the Studio outputs root.

    Accepts either the run's ``output_dir`` (what the UI replays, matching the LLM resume
    flow) or an explicit ``checkpoint-<N>`` directory. Raises ResumeError with a
    user-facing message for a path that escapes outputs, comes from another operating
    system, or no longer exists."""
    from core.training.resume import normalize_resume_output_dir
    from utils.paths import outputs_root

    try:
        resolved = Path(normalize_resume_output_dir(str(path_value)))
    except ValueError as error:
        message = str(error)
        if not message.startswith("Resume checkpoint"):
            # The containment resolver raises an internal "path escapes root: <abs> -> <abs>"
            # that quotes server paths at the user; replace it with the same wording the LLM
            # resume flow shows.
            message = "Resume checkpoint must be inside Unsloth outputs."
        raise ResumeError(message) from error
    # A name that cleans away to nothing (".", "outputs", "./.") lands on the outputs ROOT, where
    # the scan would sweep checkpoint dirs across unrelated runs. Same guard the start route
    # applies to output_dir.
    if resolved.resolve(strict = False) == outputs_root().resolve(strict = False):
        raise ResumeError(
            f"'{path_value}' is the outputs folder itself, not a training run inside it."
        )
    return resolved


# What restore_resume_state() refuses to continue without. Kept here rather than inferred from
# the manifest, because the manifest is exactly what a truncated or hand-edited bundle gets
# wrong: _assert_loadable only opens the roles the bundle happens to LIST, so a checkpoint that
# lists an adapter and nothing else passed the route preflight, the resident GPU model was
# evicted, and the child then died on "This checkpoint carries no optimizer state".
_REQUIRED_STATE: tuple[tuple[str, str], ...] = (
    ("adapter", "the trained LoRA weights"),
    ("optimizer", "the optimizer moments"),
    ("scheduler", "the learning-rate schedule position"),
    # Both image trainers draw the latent, the noise and the timestep from torch immediately
    # after resume, so a bundle with no RNG file continues a different random stream while
    # restore_rng_state leaves the generator at its fresh seed and says nothing.
    ("rng", "the random-number generator state"),
)


def _assert_required_state(path: Path, manifest: dict[str, Any]) -> None:
    """Refuse a bundle missing state the trainer treats as mandatory, BEFORE teardown."""
    files = manifest.get("files")
    listed = files if isinstance(files, dict) else {}
    missing = [
        label
        for role, label in _REQUIRED_STATE
        if not isinstance(listed.get(role), str) or not listed.get(role)
    ]
    # The sampler's permutation lives in the manifest rather than a file. Required for an
    # image bundle, where both trainers always supply a sampler and the child refuses without
    # it -- after the route has evicted the resident models. Left optional for any other kind,
    # whose trainer may legitimately have none.
    if str(manifest.get("kind") or "image") == "image" and not isinstance(
        manifest.get("sampler"), dict
    ):
        missing.append("the dataset sampler position")
    if not missing:
        return
    raise ResumeError(
        f"'{path.name}' is missing {_join_clauses(missing)}, so the run cannot be continued "
        "from it. Resume from an earlier checkpoint, or start a new run."
    )


def _join_clauses(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _assert_loadable(path: Path, manifest: dict[str, Any]) -> None:
    """Actually open every state file the manifest lists, and turn any failure into a
    ResumeError.

    read_checkpoint deliberately only parses headers -- it is called once per bundle by the
    run-history listing, which must stay cheap -- so it cannot tell a torch zip that walks to
    STOP from one torch.load will refuse. A pickle carrying a global outside the weights_only
    allowlist is exactly that, and it passed the route preflight: 200 OK, resident GPU model
    evicted, then the child died on a raw UnpicklingError. That is precisely the outcome the
    module docstring promises this preflight prevents, so pay the load here, on the one
    bundle a user actually asked to resume. These are MB-scale files and this runs in a
    worker thread."""
    loaded = LoadedCheckpoint(path = path, manifest = manifest)
    files = manifest.get("files")
    for role in (files or {}) if isinstance(files, dict) else ():
        try:
            if role in ("adapter", "ema"):
                loaded.tensors(role)
            else:
                state = loaded.torch_state(role)
                if role == "rng" and not (
                    isinstance(state, dict) and state.get("torch_cpu") is not None
                ):
                    # An rng file with no torch state restores nothing torch draws from, which
                    # is every latent, noise and timestep the loop asks for.
                    raise ResumeError(
                        f"'{path.name}' carries no torch random-number state, so the run "
                        "would continue on a different random stream. Resume from an earlier "
                        "checkpoint, or start a new run."
                    )
        except ResumeError:
            raise
        except Exception as error:  # noqa: BLE001 -- any unreadable state file is a refusal
            raise ResumeError(
                f"The '{role}' file in '{path.name}' could not be read back "
                f"({type(error).__name__}), so this checkpoint cannot be resumed."
            ) from error


def preflight_resume(
    path_value: str, *, identity: CheckpointIdentity, target_steps: int
) -> tuple[str, int]:
    """Validate a resume request and return ``(checkpoint_dir, resumed_step)``.

    Called by the start route BEFORE the resident GPU model is evicted, and again inside
    the trainer, so a rejected resume never costs the user their loaded pipeline. Raises
    ResumeError with a message written for the user.

    ``identity`` may leave ``dataset_fingerprint`` unset on the first (pre-discovery) pass;
    that comparison is then skipped and re-run once the images are known."""
    root = resolve_resume_dir(path_value)
    # An explicit checkpoint-<N> directory resumes exactly that step; a run directory takes
    # its newest complete bundle.
    #
    # The name alone does not settle which it is. An adapter can legitimately be called
    # "checkpoint-2026", and its output directory then matches the bundle pattern while
    # containing checkpoint-11 rather than a trainer_state.json of its own -- read as an
    # explicit bundle it is simply rejected, with the real checkpoint sitting inside it. So the
    # explicit branch is taken only when the path IS a valid bundle.
    explicit = read_checkpoint(root) if checkpoint_step(root) >= 0 else None
    found: Optional[tuple[Path, dict]] = None
    if explicit is not None:
        found = (root, explicit)
    else:
        found = latest_valid_checkpoint(root)
        if found is None and checkpoint_step(root) >= 0:
            # Named like a bundle, is not one, and holds no bundles either: the original
            # message is the accurate one.
            raise ResumeError(
                f"'{root.name}' is not a complete training checkpoint (it is missing files, "
                "or was left behind by an interrupted save)."
            )
    if found is None:
        raise ResumeError(
            "No complete training checkpoint was found for this run, so there is nothing to "
            "resume from. Start a new run instead."
        )
    path, manifest = found
    saved = CheckpointIdentity.from_dict(manifest.get("identity"))
    if saved is None:
        raise ResumeError(
            "This checkpoint does not record what it was trained from, so it cannot be "
            "safely resumed."
        )
    reason = saved.mismatch_reason(identity)
    if reason:
        raise ResumeError(reason)
    # After the identity gate, so a mismatched bundle still fails fast on the cheap check.
    _assert_required_state(path, manifest)
    _assert_loadable(path, manifest)
    step = int(manifest.get("global_step") or 0)
    if target_steps and step >= int(target_steps):
        raise ResumeError(
            f"This checkpoint is already at step {step} of {int(target_steps)}, so there is "
            "nothing left to train. Raise the step count to continue it."
        )
    return str(path), step


# ── loading ───────────────────────────────────────────────────────────────────
@dataclass
class LoadedCheckpoint:
    """A validated bundle, with its tensors read lazily so a preflight never pays for them."""

    path: Path
    manifest: dict[str, Any]

    @property
    def step(self) -> int:
        return int(self.manifest.get("global_step") or 0)

    @property
    def target_steps(self) -> int:
        return int(self.manifest.get("target_steps") or 0)

    @property
    def ema_updates(self) -> int:
        return int(self.manifest.get("ema_updates") or 0)

    @property
    def optimizer_class(self) -> Optional[str]:
        value = self.manifest.get("optimizer_class")
        return str(value) if value else None

    @property
    def progress(self) -> dict[str, Any]:
        state = self.manifest.get("progress")
        return state if isinstance(state, dict) else {}

    @property
    def running_loss(self) -> float:
        """The loss total accumulated up to ``step``, so a resumed run's reported average
        stays an average over the WHOLE run rather than restarting at the resume point."""
        try:
            return float(self.progress.get("running_loss") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @property
    def sampler_state(self) -> Optional[dict[str, Any]]:
        state = self.manifest.get("sampler")
        return state if isinstance(state, dict) else None

    @property
    def rng_json(self) -> Optional[dict[str, Any]]:
        state = self.manifest.get("rng")
        return state if isinstance(state, dict) else None

    def _file(self, role: str) -> Optional[Path]:
        files = self.manifest.get("files")
        name = files.get(role) if isinstance(files, dict) else None
        if not isinstance(name, str) or not name or Path(name).name != name:
            return None
        candidate = self.path / name
        return candidate if candidate.is_file() else None

    def tensors(
        self,
        role: str,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """The safetensors bundle for ``role`` (``adapter`` / ``ema``), or {}."""
        from safetensors.torch import load_file

        path = self._file(role)
        return load_file(str(path), device = device) if path is not None else {}

    def torch_state(self, role: str) -> Optional[Any]:
        """A ``torch.save``d state dict (``optimizer`` / ``scheduler`` / ``rng``), or None.

        Loaded with ``weights_only = True``: these files are written by Studio into its own
        outputs directory, but a resume path is client-supplied, so the loader must never be
        able to execute pickled code. Verified to round-trip bitsandbytes AdamW8bit state,
        whose quantized moments and maps are plain uint8/fp32 tensors."""
        import torch

        path = self._file(role)
        if path is None:
            return None
        return torch.load(str(path), map_location = "cpu", weights_only = True)


def load_checkpoint(path: str | os.PathLike[str]) -> LoadedCheckpoint:
    """Open a bundle that ``preflight_resume`` already accepted. Raises ResumeError if it
    became unreadable in between (a concurrent delete, a half-mounted volume)."""
    directory = Path(path).expanduser()
    manifest = read_checkpoint(directory)
    if manifest is None:
        raise ResumeError(
            f"The training checkpoint at '{directory}' could not be read; it may have been "
            "deleted or damaged since the run started."
        )
    return LoadedCheckpoint(path = directory, manifest = manifest)
