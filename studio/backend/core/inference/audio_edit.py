# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Checks for an Edit speech request, and the source clip an edit keeps in history.

The client renders the word diff (DotTTS markup or FireRedAudio instructions); each check holds it
against both transcripts, so a malformed or injected value never reaches the runtime.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional, Sequence

from loggers import get_logger

logger = get_logger(__name__)

# Refused, never cut: a cut recording no longer matches its transcript.
EDIT_SOURCE_MAX_SECONDS = 30.0

TOO_LONG = "Edit works on recordings up to 30 s. Record or upload a shorter take."
NO_CHANGE = "Change at least one word."
EMPTY_TARGET = "Keep at least one word."
MISMATCH = "The changes do not match the transcript. Check ① and ② again."
UNKNOWN_INSTRUCTION = "That change is not one FireRedAudio understands."
NEEDS_DELIVERY_MODEL = "Delivery changes need FireRedAudio."
NO_DELIVERY = "Pick a speed or a pitch change."

_MARKUP_TAG_RE = re.compile(
    r'<sub targ="([^"<>]*)">([^<>]*)</sub>|<del>([^<>]*)</del>|<ins>([^<>]*)</ins>'
)
INSTRUCTION_RE = re.compile(
    r"^(?:Replace '(.+)' with '(.+)'|Delete '(.+)'|Insert '(.+)' before '(.+)')\.$"
)


def _collapse(text: Optional[str]) -> str:
    return " ".join(str(text or "").split())


def markup_tags(markup: str) -> int:
    return len(_MARKUP_TAG_RE.findall(markup or ""))


def markup_sides(markup: str) -> Optional[tuple[str, str]]:
    """``(source, target)`` text a DotTTS markup describes, whitespace-collapsed; None when it holds
    anything but the three tags."""
    source: list[str] = []
    target: list[str] = []
    position = 0
    for match in _MARKUP_TAG_RE.finditer(markup or ""):
        between = markup[position : match.start()]
        if "<" in between or ">" in between:
            return None
        source.append(between)
        target.append(between)
        targ, sub, deleted, inserted = match.groups()
        if sub is not None:
            source.append(sub)
            target.append(targ)
        elif deleted is not None:
            source.append(deleted)
        else:
            target.append(inserted)
        position = match.end()
    rest = (markup or "")[position:]
    if "<" in rest or ">" in rest:
        return None
    source.append(rest)
    target.append(rest)
    return _collapse("".join(source)), _collapse("".join(target))


def check_markup(markup: str, original: str, edited: str) -> Optional[str]:
    """Why ``markup`` does not turn ``original`` into ``edited``; None when it does."""
    sides = markup_sides(markup)
    if sides is None:
        return MISMATCH
    if not markup_tags(markup):
        return NO_CHANGE
    if sides != (_collapse(original), _collapse(edited)):
        return MISMATCH
    return None


def check_instructions(
    instructions: Sequence[str],
    original: str,
    edited: str,
    max_changes: Optional[int],
    label: str = "FireRedAudio",
) -> Optional[str]:
    """Why FireRedAudio ``instructions`` are refused; None when each is a known form whose old words
    (or anchor) are in ``original`` and new words in ``edited``, and the words they remove and add
    turn the one transcript into the other."""
    items = list(instructions or ())
    if not items:
        return NO_CHANGE
    if max_changes is not None and len(items) > max_changes:
        return (
            f"{label} applies at most {max_changes} changes. "
            "Make fewer changes, or use DotTTS Edit."
        )
    before, after = _collapse(original), _collapse(edited)
    # Runtime gets the instructions, history gets ``edited``: they must agree as bags of words
    # (which occurrence of a repeated word is meant is the runtime's call).
    words = Counter(before.split())
    for instruction in items:
        match = INSTRUCTION_RE.match(str(instruction))
        if match is None:
            return UNKNOWN_INSTRUCTION
        old, new, deleted, inserted, anchor = match.groups()
        if old is not None:
            olds, news, removed = [old], [new], [old]
        elif deleted is not None:
            olds, news, removed = [deleted], [], [deleted]
        else:
            olds, news, removed = [anchor], [inserted], []
        if not all(_collapse(o) and _collapse(o) in before for o in olds):
            return MISMATCH
        if not all(_collapse(n) and _collapse(n) in after for n in news):
            return MISMATCH
        # Running bag, not the transcript: a word an earlier instruction removed is gone.
        if not all(words[token] > 0 for o in olds for token in o.split()):
            return MISMATCH
        for phrase in removed:
            words.subtract(phrase.split())
        if any(count < 0 for count in words.values()):
            return MISMATCH
        for phrase in news:
            words.update(phrase.split())
    if +words != Counter(after.split()):
        return MISMATCH
    return None


def delivery_instructions(speed: Optional[float], pitch_steps: Optional[int]) -> list[str]:
    """FireRedAudio acoustic_edit instructions, one per call; pitch only rises."""
    out: list[str] = []
    if speed is not None and float(speed) != 1.0:
        out.append(f"adjust the speed to {float(speed):g}x")
    if pitch_steps:
        out.append(f"shift the pitch by {int(pitch_steps)} steps")
    return out


def change_count(edit: dict[str, Any]) -> int:
    if edit.get("mode") == "delivery":
        return len(delivery_instructions(edit.get("speed"), edit.get("pitch_steps")))
    if edit.get("markup"):
        return markup_tags(edit["markup"])
    if edit.get("instructions"):
        return len(edit["instructions"])
    return 1


def request_problem(
    rules: Optional[dict[str, Any]],
    edit: Optional[dict[str, Any]],
    text: str,
    reference_text: Optional[str],
    label: str,
) -> Optional[str]:
    """Why the loaded model's ``audio_edit`` rules refuse this edit; else None."""
    edit = edit or {}
    rules = rules or {}
    style = rules.get("style")
    original = reference_text or ""
    if edit.get("mode") == "delivery":
        if not rules.get("delivery"):
            return NEEDS_DELIVERY_MODEL
        if not delivery_instructions(edit.get("speed"), edit.get("pitch_steps")):
            return NO_DELIVERY
        return None
    # Before the per-style checks: a no-op such as <sub targ="x">x</sub> passes them.
    if style in ("markup", "instructions", "sentence"):
        if not _collapse(text):
            return EMPTY_TARGET
        if _collapse(text) == _collapse(original):
            return NO_CHANGE
    if style == "markup":
        markup = edit.get("markup")
        if not markup:
            return f"{label} needs the marked-up changes."
        return check_markup(markup, original, text)
    if style == "instructions":
        return check_instructions(
            edit.get("instructions") or (), original, text, rules.get("max_changes"), label
        )
    if style == "sentence":
        return None
    return "Load a model that can edit speech."


def wav_seconds(path: Path) -> Optional[float]:
    """Length of a WAV from its header; None when unreadable."""
    import wave
    try:
        with wave.open(str(path)) as w:
            return w.getnframes() / w.getframerate()
    except Exception:  # noqa: BLE001 - the caller measures the prepared copy instead
        return None


def save_source_clip(
    source, prepared_path: Path, reference_text: Optional[str]
) -> Optional[dict[str, Any]]:
    """Keep an uploaded recording in history beside its edit, so A/B outlives the upload. Reuses a
    source clip of the same audio; None for a history clip or an upload with no record."""
    import wave

    from core.inference import audio_gallery, audio_inputs

    if getattr(source, "kind", None) != "input":
        return None
    sidecar = audio_inputs._read_sidecar(audio_inputs._sidecar(source.path.parent, source.id))
    sha = (sidecar or {}).get("sha256")
    if not sha:
        return None
    directory = audio_gallery.gallery_dir()
    try:
        sidecars = list(directory.glob("*.json"))
    except OSError:
        sidecars = []
    for path in sidecars:
        meta = audio_gallery._read_meta(path)
        if (
            meta is not None
            and meta.get("role") == "source"
            and meta.get("workflow") == "edit"
            and meta.get("source_sha256") == sha
            and audio_gallery.owned_audio_path(path.stem) is not None
        ):
            return audio_gallery._record(path.stem, meta)
    with wave.open(str(prepared_path)) as w:
        sample_rate, duration_s = w.getframerate(), round(w.getnframes() / w.getframerate(), 3)
    meta: dict[str, Any] = {
        "prompt": " ".join(str(reference_text or "").split()) or source.name,
        "model": "Recording",
        "audio_type": "recording",
        "workflow": "edit",
        "role": "source",
        "source_sha256": sha,
        "reference_name": source.name,
        "sample_rate": sample_rate,
        "duration_s": duration_s,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return audio_gallery.save(Path(prepared_path).read_bytes(), meta)
