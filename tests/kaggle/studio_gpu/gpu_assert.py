# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Did the GGUF actually run on the GPU, or did it quietly fall back to CPU?

This module is the whole reason the Unsloth Kaggle leg is worth its quota. A
CPU fallback in Unsloth is invisible from the outside: llama.cpp loads the same
file, answers the same prompt, and returns the same text, only slower. Every
Unsloth inference check in this repo today would pass against a CPU-only
build, because every one of them runs on a machine that has no GPU to fall
back from.

So "the model produced text" is explicitly NOT evidence here. The verdict is
built from three independent observations, and the rules are stated up front
because the interesting cases are the ambiguous ones:

**Negative evidence is conclusive.** llama.cpp saying it offloaded 0 layers,
Unsloth reporting ``cpu_fallback_reason``, or Unsloth reporting an effective
``gpu_layers`` of 0 each fail on their own. No amount of positive evidence
overrides them: they are the fallback announcing itself.

**Positive evidence has to come from somewhere.** At least one of

* the llama-server process appearing in ``nvidia-smi --query-compute-apps``
  with a non-trivial resident allocation,
* llama.cpp's own load line reporting N of M layers offloaded with N > 0,
* device-wide VRAM in use climbing by more than a model's worth across the
  load,

must hold. Any one of them is enough; requiring all three would turn a
container quirk into a red pull request.

**Silence is a failure, not a pass.** If all three come back unreadable, the
verdict is FAIL. That is the deliberate asymmetry this file exists for: the
question being asked is "prove the GPU was used", and an unanswerable
question has not proved it. Three independent probes are enough that all of
them going blank is itself worth looking at.

The one known way to get a blank first probe is a container whose PID
namespace does not match the one nvidia-smi reports, where
``--query-compute-apps`` lists nothing at all even under load. That is why it
is not the only probe.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# A llama-server holding less than this on the card is not offloading anything worth calling offload -- CUDA context
# plus scratch alone is tens of MiB, and a fully CPU-resident model can still show a context-sized allocation if
# anything touched the device.
MIN_PROCESS_VRAM_MIB = 96

# Device-wide growth across the load that no CPU-resident model explains.
MIN_DEVICE_VRAM_DELTA_MIB = 256

# GGUF's four-byte file magic.
# Checked before anything tries to load a file the export path claims to have written: a truncated or half-moved file is
# a far more likely export bug than a wrong-magic one, and both look like a present file to `os.path.exists`.
GGUF_MAGIC = b"GGUF"

_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers?\s+to\s+GPU")

_CUDA_BUFFER_RE = re.compile(
    r"(CUDA\d+|ROCm\d+)\s+model buffer size\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*MiB",
    re.IGNORECASE,
)

# Install kinds from studio/install_llama_prebuilt.py that mean the binaries on disk carry CUDA kernels.
CUDA_INSTALL_KINDS = frozenset({"linux-cuda", "linux-arm64-cuda"})

# "cuda12", "cuda13", and whatever major comes next, anywhere in a runtime
# line or an asset filename. Anchored on the digit so "cudart" or a repo name
# containing "cuda" cannot match on its own.
_CUDA_RUNTIME_RE = re.compile(r"cuda\d+")


def parse_compute_apps(csv_text: str) -> dict[int, int]:
    """``nvidia-smi --query-compute-apps=pid,used_gpu_memory`` as {pid: MiB}.

    Accepts the ``--format=csv,noheader,nounits`` shape and tolerates the unit
    suffix being present anyway, because which of the two you get depends on
    the driver version and getting it wrong silently yields an empty dict.
    """
    apps: dict[int, int] = {}
    for line in csv_text.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("pid"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        mem = parts[1].replace("MiB", "").replace("MB", "").strip()
        if mem.lower() in ("[n/a]", "n/a", ""):
            continue
        try:
            apps[pid] = int(float(mem))
        except ValueError:
            continue
    return apps


def offloaded_layers(log_text: str) -> tuple[int, int] | None:
    """The last ``offloaded N/M layers to GPU`` in a llama.cpp log.

    The last one, not the first: Unsloth loads a model more than once in a
    session (an export is verified by loading the file it just wrote), and the
    question is always about the most recent load.
    """
    matches = _OFFLOAD_RE.findall(log_text or "")
    if not matches:
        return None
    offloaded, total = matches[-1]
    return int(offloaded), int(total)


def cuda_buffer_mib(log_text: str) -> float | None:
    """Largest device model-buffer allocation llama.cpp reported, in MiB."""
    matches = _CUDA_BUFFER_RE.findall(log_text or "")
    if not matches:
        return None
    return max(float(size) for _, size in matches)


def install_kind(marker_path: Path | None) -> str | None:
    """What KIND of llama.cpp bundle is installed, from its marker.

    There is no ``install_kind`` key in UNSLOTH_PREBUILT_INFO.json and there
    never was. ``install_llama_prebuilt.py`` writes ``install_kind`` only into
    the JSON its resolver prints to stdout (line ~7678); the marker it writes
    to disk (~line 5931) records ``asset``, ``tag``, ``runtime_line``,
    ``coverage_class``, ``bundle_profile`` and no kind at all.

    So reading ``install_kind`` from the marker answered None on every box, for
    every bundle, including a perfectly good CUDA one -- six hardware runs
    reported ``install_kind=None`` and failed the export assertion for it while
    a working CUDA llama.cpp sat on disk. The installer said as much when asked
    to install again: "existing llama.cpp install already matches selected
    release b10360-mix-87da1a2; skipping download and install".

    ``runtime_line`` is the field that answers the question actually being
    asked. It is ``cuda12``/``cuda13`` for the CUDA bundles and names the
    non-CUDA backends otherwise, and it is written on the same line of the same
    dict as the asset, so the two cannot disagree. ``asset`` is the fallback
    for a marker old enough to predate it.

    ``None`` in means no marker was found, and answers ``None`` rather than
    raising: ``llama_cpp_marker`` returns ``None`` for that case.
    """
    if marker_path is None:
        return None
    try:
        payload = json.loads(Path(marker_path).read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(payload, dict):
        return None
    runtime_line = payload.get("runtime_line")
    if runtime_line:
        return str(runtime_line)
    asset = payload.get("asset")
    return str(asset) if asset else None


def is_cuda_install(kind: str | None) -> bool:
    """Is this a CUDA bundle?

    ``kind`` is now a ``runtime_line`` (``cuda12``, ``cuda13``, ...) or, for a
    marker too old to carry one, an asset filename like
    ``app-b10360-mix-87da1a2-linux-x64-cuda13-older.tar.gz``. Both are matched
    by looking for a cuda runtime line rather than by equality against a fixed
    set, because the set would need a new entry on every CUDA major and would
    fail closed -- reporting a working cuda14 install as not-CUDA -- which is
    the same failure mode this whole function just spent six runs in.
    """
    if not kind:
        return False
    lowered = str(kind).lower()
    return bool(_CUDA_RUNTIME_RE.search(lowered)) or lowered in CUDA_INSTALL_KINDS


# Where a llama.cpp install can be, most specific first.
# STUDIO_HOME is checked because a caller may point an install there explicitly;
# the canonical location is what `install_llama_prebuilt.py` uses by default (its `Path.home() / ".unsloth" /
# "llama.cpp"`), and it is where `install.sh --local` actually puts one.
def llama_cpp_marker(studio_home: Path) -> Path | None:
    """The UNSLOTH_PREBUILT_INFO.json of the llama.cpp this box will use.

    Five hardware runs reported install_kind=None and failed the export
    assertion for it. There was no missing llama.cpp: install.sh --local had
    installed one into ~/.unsloth/llama.cpp, and this payload was reading
    STUDIO_HOME/llama.cpp, a path nothing ever wrote to. The installer even
    said so when asked to install again --

        existing llama.cpp install already matches selected release
        b10360-mix-87da1a2; skipping download and install

    -- while the directory it had been pointed at stayed empty.

    Returns None when neither location has a marker, which is then a real
    absence rather than a guess about where to look.
    """
    candidates = (
        Path(studio_home) / "llama.cpp" / "UNSLOTH_PREBUILT_INFO.json",
        Path.home() / ".unsloth" / "llama.cpp" / "UNSLOTH_PREBUILT_INFO.json",
    )
    for marker in candidates:
        if marker.is_file():
            return marker
    return None


def gguf_magic_ok(path: Path) -> bool:
    """Does this file start with GGUF's magic? Cheap, and catches a truncation."""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == GGUF_MAGIC
    except OSError:
        return False


def offload_verdict(
    *,
    server_pid: int | None,
    compute_apps: dict[int, int] | None,
    log_text: str,
    device_vram_delta_mib: float | None,
    status: dict | None,
    server_pids: list[int] | None = None,
) -> dict:
    """Was the model on the GPU? Returns a verdict dict with its evidence.

    ``status`` is Unsloth's ``GET /api/inference/status`` body. Only two of its
    fields are load-bearing here and both are negative signals:
    ``cpu_fallback_reason`` (Unsloth replayed the launch on CPU) and an
    effective ``gpu_layers`` of exactly 0 (nothing was placed on the card).
    A ``gpu_layers`` of -1 is Unsloth's Auto mode, which says nothing either
    way and is left to the other probes.
    """
    evidence: list[str] = []
    failures: list[str] = []
    positives: list[str] = []

    status = status or {}
    fallback = status.get("cpu_fallback_reason")
    if fallback:
        failures.append(f"Unsloth reported a CPU fallback: cpu_fallback_reason={fallback!r}")

    effective_layers = status.get("gpu_layers")
    if isinstance(effective_layers, int):
        evidence.append(f"status.gpu_layers={effective_layers}")
        if effective_layers == 0:
            failures.append(
                "Unsloth reports gpu_layers=0, so nothing was placed on the GPU "
                "even though the load asked for it"
            )

    counts = offloaded_layers(log_text)
    if counts is None:
        evidence.append("llama.cpp log: no offload line found")
    else:
        offloaded, total = counts
        evidence.append(f"llama.cpp log: offloaded {offloaded}/{total} layers to GPU")
        if offloaded <= 0:
            failures.append(
                f"llama.cpp offloaded {offloaded}/{total} layers, which is the CPU path"
            )
        else:
            positives.append(f"llama.cpp offloaded {offloaded}/{total} layers")

    buffer_mib = cuda_buffer_mib(log_text)
    if buffer_mib is not None:
        evidence.append(f"llama.cpp device model buffer: {buffer_mib:.0f} MiB")

    # Unsloth's status body carries no pid, so the caller also discovers the llama-server processes itself; either
    # source is accepted here.
    candidates: list[int] = []
    for pid in [server_pid, *(server_pids or [])]:
        if isinstance(pid, int) and pid not in candidates:
            candidates.append(pid)

    if compute_apps is None:
        evidence.append("nvidia-smi compute-apps: unreadable")
    elif not compute_apps:
        evidence.append("nvidia-smi compute-apps: no process listed")
    elif not candidates:
        evidence.append(
            f"nvidia-smi compute-apps: {len(compute_apps)} process(es) listed, but "
            f"no llama-server pid was found to match them against"
        )
    else:
        matched = {pid: compute_apps[pid] for pid in candidates if pid in compute_apps}
        if not matched:
            evidence.append(
                f"nvidia-smi compute-apps: llama-server pid(s) "
                f"{', '.join(str(p) for p in candidates)} are not among the "
                f"{len(compute_apps)} process(es) holding GPU memory"
            )
        for pid, used in matched.items():
            evidence.append(f"nvidia-smi compute-apps: pid {pid} holds {used} MiB")
            if used >= MIN_PROCESS_VRAM_MIB:
                positives.append(f"llama-server pid {pid} holds {used} MiB of VRAM")
            else:
                evidence.append(
                    f"that is below the {MIN_PROCESS_VRAM_MIB} MiB floor, so it is "
                    f"consistent with a bare CUDA context and no weights"
                )

    if device_vram_delta_mib is None:
        evidence.append("device VRAM delta: unreadable")
    else:
        evidence.append(f"device VRAM delta across the load: {device_vram_delta_mib:.0f} MiB")
        if device_vram_delta_mib >= MIN_DEVICE_VRAM_DELTA_MIB:
            positives.append(f"device VRAM in use grew by {device_vram_delta_mib:.0f} MiB")

    if not failures and not positives:
        failures.append(
            "no probe could show the GPU was used: the process was not visible to "
            "nvidia-smi, llama.cpp logged no offload line, and device VRAM did not "
            "move. Text was returned, but nothing here distinguishes that from a "
            "CPU fallback, so this is a failure rather than a pass"
        )

    return {
        "passed": not failures,
        "failures": failures,
        "positives": positives,
        "evidence": evidence,
    }
