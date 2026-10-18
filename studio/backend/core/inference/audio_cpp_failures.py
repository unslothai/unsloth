# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Known audio.cpp failures, named so a client can offer the fix. Raw runtime text is only
classified, never returned: any runtime detail in a message goes through ``sanitize_runtime_tail``."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from core.inference.audio_errors import sanitize_runtime_tail

# Ordered: the first code with a matching pattern wins. Lowercased text is matched.
_PATTERNS = tuple(
    (code, re.compile(pattern))
    for code, pattern in (
        # Missing library or data only: a request the runtime refuses can mention eSpeak too.
        (
            "espeak_missing",
            r"could not load espeak|espeak-ng failed to initialize|espeak data package missing"
            r"|cannot (?:read|open) espeak data|needs an audio runtime built with espeak",
        ),
        ("runtime_missing", r"the audio runtime is not installed"),
        (
            "family_unknown_to_runtime",
            r"unsupported model family hint:|no registered model loader can load"
            r"|needs a newer audio runtime|update the audio runtime",
        ),
        (
            "gpu_unavailable",
            r"backend requested but|failed to initialize \w+ backend on device"
            r"|failed to initialize (?:cuda|rocm|musa)\b|ggml_vulkan: no devices found|has no device",
        ),
        (
            "out_of_memory",
            r"backend weight buffer|failed to allocate \S+ buffer of size|cudamalloc failed"
            r"|cuda error: out of memory|memory allocation of size \d+ failed"
            r"|does not fit in device memory|insufficient_memory|headroom exceeds available"
            r"|no single gpu has enough free memory",
        ),
        (
            "file_incomplete",
            r"failed to read (?:magic|header|tensor info|tensor data)|invalid magic characters"
            r"|failed to read gguf (?:metadata|tensor source)|tensor data range is out of bounds"
            r"|model path does not exist",
        ),
    )
)
_CODES = frozenset(code for code, _ in _PATTERNS) | {"runtime_stopped"}
_STATUS = {"file_incomplete": 422}  # else 503

# Messages that already come from Unsloth Studio, so no runtime line is appended.
_STUDIO_CODES = frozenset({"runtime_missing", "family_unknown_to_runtime", "espeak_missing"})

_UPDATE = "Update it from the update banner in Unsloth Studio, or run `unsloth studio update`."
_MESSAGES = {
    "runtime_missing": "The audio runtime isn't installed. Install it from the update banner in "
    "Unsloth Studio, or run `unsloth studio update`.",
    "family_unknown_to_runtime": "{name} needs a newer audio runtime than the one installed. "
    + _UPDATE,
    "espeak_missing": "{name} needs an audio runtime built with eSpeak-ng. " + _UPDATE,
    "gpu_unavailable": "The audio runtime could not use the GPU. Set the audio device to CPU in "
    "Unsloth Studio, or set UNSLOTH_AUDIO_CPP_BACKEND=cpu.",
    "out_of_memory": "{name} ran out of memory. Free GPU memory, choose a smaller variant, or set "
    "the audio device to CPU.",
    "out_of_memory_cpu": "{name} ran out of memory. Close other apps to free RAM, or choose a "
    "smaller variant.",
    "file_incomplete": "The model file is incomplete or damaged. Delete it and download it again, "
    "or choose another variant.",
    # Runtime 4xx: the input is too big for the device (upstream CapacityError).
    "capacity": "This input does not fit in device memory. Try a shorter input, or set the audio "
    "device to CPU.",
    # Unsloth Studio's own refusal before launch.
    "placement": "No single GPU has enough free memory for {name}. Free GPU memory, choose a "
    "smaller variant, or set the audio device to CPU.",
}


@dataclass(frozen = True)
class AudioCppFailure:
    code: str
    message: str
    status: int
    # smaller_variants / offload_options / device only: never a path or raw runtime text.
    extras: dict = field(default_factory = dict)

    def detail(self) -> dict:
        return {"code": self.code, "message": self.message, **self.extras}

    def payload(self) -> dict:
        return asdict(self)

    @classmethod
    def from_payload(cls, data: Any) -> Optional["AudioCppFailure"]:
        if not isinstance(data, dict) or data.get("code") not in _CODES:
            return None
        code = str(data["code"])
        extras = data.get("extras") or {}
        extras = {
            k: v
            for k, v in extras.items()
            if k in ("smaller_variants", "offload_options", "device")
        }
        return cls(
            code,
            str(data.get("message") or ""),
            int(data.get("status") or _STATUS.get(code, 503)),
            extras,
        )


class AudioCppFailureError(RuntimeError):
    """A worker's classified failure, rebuilt in the parent process."""

    def __init__(self, message: str, failure: AudioCppFailure) -> None:
        super().__init__(message)
        self.audio_cpp_failure = failure


def failure_of(exc: Any) -> Optional[AudioCppFailure]:
    failure = getattr(exc, "audio_cpp_failure", None)
    return failure if isinstance(failure, AudioCppFailure) else None


def attach_failure(exc: BaseException, failure: Optional[AudioCppFailure]) -> BaseException:
    if failure is not None and failure_of(exc) is None:
        try:
            exc.audio_cpp_failure = failure
        except AttributeError:
            pass
    return exc


def estimate_resident_gb(size_bytes: int) -> float:
    """Resident GPU memory for an audio.cpp package: weights plus graph and cache buffers.
    Measured TTS peaks run about 3x the q8 weights (Qwen3-TTS 0.6B: 1.9 GB file, 5.9 GB peak)."""
    return round(size_bytes / (1024**3) * 2.5 + 1.5, 2)


def smaller_variants(
    model,
    *,
    free_gb: Optional[float] = None,
    limit: int = 3,
) -> list[dict]:
    """Variants smaller than the loaded one, largest first; only those that fit ``free_gb``."""
    if model is None:
        return []
    from core.inference.audio_cpp_models import quant_label, resolve

    extra = 0
    for companion in getattr(model, "companions", ()) or ():
        try:
            found = resolve(companion.id, companion.variant, network = False)
        except Exception:  # noqa: BLE001 - an unknown companion counts as nothing
            found = None
        extra += found.size_bytes if found is not None else 0
    rows = []
    for variant in sorted(model.variants, key = lambda v: v.size_bytes, reverse = True):
        estimate = estimate_resident_gb(variant.size_bytes + extra)
        if not 0 < variant.size_bytes < model.size_bytes or (
            free_gb is not None and estimate > free_gb
        ):
            continue
        label = variant.label or quant_label(variant.main_file)
        rows.append(
            {
                # The quant /api/models/gguf-variants lists for this row.
                "variant": variant.key,
                "label": label,
                "size_bytes": variant.size_bytes,
                "estimate_gb": estimate,
            }
        )
    return rows[:limit]


def offload_options(model) -> list[str]:
    """Session options of the installed runtime's spec that move experts off the GPU."""
    if model is None:
        return []
    from core.inference.audio_cpp_models import runtime_spec

    spec = runtime_spec(model.family) or {}
    session = (spec.get("options") or {}).get("session") if isinstance(spec, dict) else None
    return [
        str(option["name"])
        for option in session or ()
        if isinstance(option, dict) and "expert" in str(option.get("name") or "")
    ]


def classify_audio_cpp_failure(
    text: str,
    *,
    backend: Optional[str] = None,
    model = None,
    runtime_status: Optional[int] = None,
    free_gb: Optional[float] = None,
) -> Optional[AudioCppFailure]:
    """The known failure in ``text`` (a runtime log tail or error), or None."""
    raw = str(text or "")
    lowered = raw.lower()
    for code, pattern in _PATTERNS:
        # A CUDA build prints its CUDA init failure on CPU launches too.
        if code == "gpu_unavailable" and backend in (None, "cpu"):
            continue
        match = pattern.search(lowered)
        if match is None:
            continue
        capacity = (
            code == "out_of_memory" and runtime_status is not None and 400 <= runtime_status < 500
        )
        placement = "no single gpu" in lowered
        on_cpu = code == "out_of_memory" and backend == "cpu" and not capacity
        key = "placement" if placement else "capacity" if capacity else code
        key = "out_of_memory_cpu" if on_cpu and not placement else key
        message = _MESSAGES[key].format(name = getattr(model, "display_name", None) or "This model")
        # "device": the run was already on the CPU, so "set the audio device to CPU" can't help.
        extras: dict = {"device": "cpu"} if on_cpu else {}
        if code == "out_of_memory" and not capacity:
            if variants := smaller_variants(model, free_gb = free_gb):
                extras["smaller_variants"] = variants
                listed = ", ".join(
                    f"{row['label']} (~{row['estimate_gb']:.1f} GB)" for row in variants
                )
                message += f" Smaller variants: {listed}."
            if offload := offload_options(model):
                extras["offload_options"] = offload
                message += f" This model can offload experts: set {' or '.join(offload[:2])}."
        if code not in _STUDIO_CODES and not placement:
            start = raw.rfind("\n", 0, match.start()) + 1
            end = raw.find("\n", match.end())
            line = sanitize_runtime_tail(raw[start : end if end >= 0 else len(raw)], 200)
            if line:
                message += f" Details: {line}"
        return AudioCppFailure(code, message, 400 if capacity else _STATUS.get(code, 503), extras)
    return None


def problem_http_exception(problem: str, model = None):
    """A preflight refusal as an HTTPException: the classified shape, else today's 501 string."""
    from fastapi import HTTPException

    failure = classify_audio_cpp_failure(problem, model = model)
    if failure is None:
        return HTTPException(status_code = 501, detail = problem)
    return HTTPException(status_code = failure.status, detail = failure.detail())
