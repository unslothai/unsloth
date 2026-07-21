# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Coverage-aware prebuilt artifact selection, shared by the llama.cpp and
whisper.cpp installers.

Lifted from install_llama_prebuilt.py (`linux_cuda_choice_from_release` /
`_artifact_covers_sms` / `_sm_range`) and generalised to a normalised artifact
view (`SelArtifact`). The caller pre-filters to one os/arch/backend, so this is
component agnostic: given the host GPUs and candidate CUDA/ROCm artifacts it
returns which bundle(s) run here, best first. whisper lacked this -- it took the
first os/arch/backend match and ignored SM coverage, serving a B200 (sm_100) the
cuda12-legacy bundle (sms 50-61).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

# Lowest CUDA major we ship: a driver of major N runs down to this floor. Mirrors
# install_llama_prebuilt.py `_MIN_CUDA_MAJOR`.
_MIN_CUDA_MAJOR = 12

# Blackwell floor sm_100 (B100/B200 sm_100, B300 sm_103, RTX50 sm_120); needs
# toolkit >= 12.8, or 12.9 for sm_103/sm_121. Synced with llama's _BLACKWELL_*.
_BLACKWELL_MIN_SM = 100
_BLACKWELL_MIN_TOOLKIT = (12, 8)
_BLACKWELL_SM_MIN_TOOLKIT = {103: (12, 9), 121: (12, 9)}


# ── Compute-capability normalisation (lifted verbatim from llama) ──
def normalize_compute_cap(value: Any) -> str | None:
    """Normalise a single compute capability to its bare SM string: '8.9' or
    '89' -> '89'. Returns None for empty/malformed values."""
    raw = str(value).strip()
    if not raw:
        return None
    if "." in raw:
        parts = raw.split(".", 1)
        if len(parts) != 2:
            return None
        major, minor = parts
        if not major.isdigit() or not minor.isdigit():
            return None
        return f"{int(major)}{int(minor)}"
    if raw.isdigit():
        return str(int(raw))
    return None


def normalize_compute_caps(compute_caps: Iterable[Any]) -> list[str]:
    """De-dupe + sort the host's compute caps ascending, so the highest SM is
    last (`caps[-1]`)."""
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in compute_caps:
        value = normalize_compute_cap(raw)
        if value is None or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    normalized.sort(key = int)
    return normalized


def cuda_runtime_lines_for_major(major: int) -> list[str]:
    """Runtime lines a driver of this CUDA major can use, newest major first down
    to the minimum we ship. A driver runs its own major and any older one."""
    return [f"cuda{m}" for m in range(major, _MIN_CUDA_MAJOR - 1, -1)]


def compatible_runtime_lines_for_driver(driver_cuda_version: tuple[int, int] | None) -> list[str]:
    """CUDA runtime lines the driver can support (newest first): major N runs its
    own major down to the floor. Only an upper bound -- the caller still intersects
    with the on-disk scan (bundles omit libcudart/libcublas). Empty below the floor
    or when the driver version is unknown."""
    if not driver_cuda_version:
        return []
    major, _minor = driver_cuda_version
    if major < _MIN_CUDA_MAJOR:
        return []
    return cuda_runtime_lines_for_major(major)


def host_is_blackwell(host_sms: list[str]) -> bool:
    """True when the highest visible SM is Blackwell-class (sm_100+)."""
    return bool(host_sms) and int(host_sms[-1]) >= _BLACKWELL_MIN_SM


def blackwell_min_toolkit_for_caps(host_sms: list[str]) -> tuple[int, int]:
    """Minimum CUDA toolkit a Blackwell host needs: 12.8, or 12.9 for sm_103/sm_121.
    Unused by the whisper path (supported_sms already gates Blackwell); retained for
    Phase B llama Windows selection. Synced with llama's _blackwell_min_toolkit."""
    req = _BLACKWELL_MIN_TOOLKIT
    for sm in host_sms:
        req = max(req, _BLACKWELL_SM_MIN_TOOLKIT.get(int(sm), _BLACKWELL_MIN_TOOLKIT))
    return req


@dataclass
class SelArtifact:
    """Normalised, component-agnostic view of a manifest artifact used for
    selection. `raw` retains the original manifest dict so the caller can recover
    component-specific fields after a pick."""

    asset: str
    os: str
    arch: str
    backend: str
    runtime_line: str | None = None
    coverage_class: str | None = None
    supported_sms: tuple[str, ...] = ()
    min_sm: int | None = None
    max_sm: int | None = None
    gfx_target: str | None = None
    mapped_targets: tuple[str, ...] = ()
    min_os: str | None = None
    rank: int = 0
    raw: dict[str, Any] = field(default_factory = dict)

    @classmethod
    def from_manifest(cls, artifact: dict[str, Any]) -> "SelArtifact":
        """Build a SelArtifact from a whisper/llama manifest artifact dict,
        coercing SM fields to the types the selector expects (supported_sms ->
        tuple[str], min_sm/max_sm -> int|None, rank -> int)."""
        return cls(
            asset = str(artifact.get("asset") or artifact.get("asset_name") or ""),
            os = str(artifact.get("os") or ""),
            arch = str(artifact.get("arch") or ""),
            backend = str(artifact.get("backend") or ""),
            runtime_line = artifact.get("runtime_line"),
            coverage_class = artifact.get("coverage_class"),
            supported_sms = tuple(
                s
                for s in (normalize_compute_cap(v) for v in (artifact.get("supported_sms") or ()))
                if s is not None
            ),
            min_sm = _coerce_int(artifact.get("min_sm")),
            max_sm = _coerce_int(artifact.get("max_sm")),
            gfx_target = artifact.get("gfx_target"),
            mapped_targets = tuple(str(v) for v in (artifact.get("mapped_targets") or ())),
            min_os = artifact.get("min_os"),
            rank = _coerce_int(artifact.get("rank")) or 0,
            raw = artifact,
        )


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ── Coverage primitives (lifted from llama `_artifact_covers_sms` / `_sm_range`) ──
def artifact_covers_sms(artifact: SelArtifact, host_sms: Iterable[str]) -> bool:
    """True when every host SM is listed in supported_sms AND falls within
    [min_sm, max_sm]."""
    if not artifact.supported_sms or artifact.min_sm is None or artifact.max_sm is None:
        return False
    supported = {str(v) for v in artifact.supported_sms}
    return all(sm in supported and artifact.min_sm <= int(sm) <= artifact.max_sm for sm in host_sms)


def sm_range(artifact: SelArtifact) -> int:
    """SM-coverage span used as a sort key: a tighter (smaller) range wins. An
    artifact with no SM metadata sorts last so it can't outrank a targeted one."""
    if artifact.min_sm is not None and artifact.max_sm is not None:
        return artifact.max_sm - artifact.min_sm
    return 9999


def blackwell_capable_runtime_lines(
    host_sms: list[str], artifacts: Iterable[SelArtifact]
) -> list[str]:
    """CUDA runtime lines (highest major first) shipping a bundle that covers
    every visible host SM. Lets a Blackwell host prefer a native cuda13 line over
    torch's reported line. Malformed lines (e.g. 'cuda13.1') are skipped, not
    ranked, mirroring llama."""
    lines: set[str] = set()
    for artifact in artifacts:
        line = artifact.runtime_line
        if not (line and line.startswith("cuda") and line[len("cuda") :].isdigit()):
            continue
        if artifact_covers_sms(artifact, host_sms):
            lines.add(line)
    return sorted(lines, key = lambda line: int(line[len("cuda") :]), reverse = True)


def rank_cuda_attempts(
    candidates: list[SelArtifact],
    host_sms: list[str],
    ordered_runtime_lines: list[str],
    log: list[str],
) -> list[SelArtifact]:
    """Per-runtime-line coverage filter + ranking (llama's
    `linux_cuda_choice_from_release` inner loop): within each line the tightest
    covering bundle wins, portable kept as an explicit fallback. `candidates` must
    be pre-filtered to one os/arch and backend='cuda'."""
    attempts: list[SelArtifact] = []
    seen: set[str] = set()
    for runtime_line in ordered_runtime_lines:
        coverage_candidates: list[SelArtifact] = []
        portable_candidate: SelArtifact | None = None
        for artifact in candidates:
            if artifact.runtime_line != runtime_line:
                continue
            asset = artifact.asset
            if not host_sms and artifact.coverage_class != "portable":
                log.append(
                    f"cuda_selection: reject {asset} runtime_line={runtime_line} "
                    f"coverage_class={artifact.coverage_class} "
                    "reason=unknown_compute_caps_prefer_portable"
                )
                continue
            if not artifact.supported_sms:
                log.append(
                    f"cuda_selection: reject {asset} runtime_line={runtime_line} "
                    f"coverage_class={artifact.coverage_class} reason=artifact_missing_supported_sms"
                )
                continue
            if artifact.min_sm is None or artifact.max_sm is None:
                log.append(
                    f"cuda_selection: reject {asset} runtime_line={runtime_line} "
                    f"coverage_class={artifact.coverage_class} reason=artifact_missing_sm_bounds"
                )
                continue
            supported = {str(v) for v in artifact.supported_sms}
            missing = [sm for sm in host_sms if sm not in supported]
            out_of_range = [
                sm for sm in host_sms if not (artifact.min_sm <= int(sm) <= artifact.max_sm)
            ]
            reasons: list[str] = []
            if missing:
                reasons.append(f"missing_sms={','.join(missing)}")
            if out_of_range:
                reasons.append(f"out_of_range_sms={','.join(out_of_range)}")
            if reasons:
                log.append(
                    f"cuda_selection: reject {asset} runtime_line={runtime_line} "
                    f"coverage_class={artifact.coverage_class} "
                    f"coverage={artifact.min_sm}-{artifact.max_sm} "
                    f"supported={','.join(artifact.supported_sms)} reasons={' '.join(reasons)}"
                )
                continue
            log.append(
                f"cuda_selection: accept {asset} runtime_line={runtime_line} "
                f"coverage_class={artifact.coverage_class} "
                f"coverage={artifact.min_sm}-{artifact.max_sm} "
                f"supported={','.join(artifact.supported_sms)}"
            )
            if artifact.coverage_class == "portable":
                portable_candidate = artifact
            else:
                coverage_candidates.append(artifact)

        if coverage_candidates:
            best = sorted(
                coverage_candidates,
                key = lambda a: (sm_range(a), a.rank, a.max_sm or 0),
            )[0]
            if best.asset not in seen:
                seen.add(best.asset)
                attempts.append(best)
                log.append(
                    f"cuda_selection: selected {best.asset} runtime_line={best.runtime_line} "
                    f"coverage_class={best.coverage_class} reason=best_coverage_for_runtime_line"
                )
        if portable_candidate is not None and portable_candidate.asset not in seen:
            seen.add(portable_candidate.asset)
            attempts.append(portable_candidate)
            log.append(
                f"cuda_selection: selected {portable_candidate.asset} "
                f"runtime_line={portable_candidate.runtime_line} "
                "coverage_class=portable reason=portable_fallback_for_runtime_line"
            )
    return attempts


def select_cuda_attempts(
    candidates: list[SelArtifact],
    host_sms: list[str],
    driver_cuda_version: tuple[int, int] | None,
    torch_runtime_line: str | None,
    log: list[str],
    *,
    detected_runtime_lines: list[str] | None = None,
) -> list[SelArtifact]:
    """Ordered CUDA bundle attempts for this host, best first.

    A runtime line is usable only when the driver supports it AND its runtime libs
    are on disk (bundles omit libcudart/libcublas): `detected_runtime_lines`, when
    given, is intersected with the driver lines, else the driver lines alone.
    Ordered newest-first, then a Blackwell host prefers the highest major that
    natively covers its SMs, else torch's line moves front. Empty when nothing is
    usable (caller falls back, CPU for whisper)."""
    driver_lines = compatible_runtime_lines_for_driver(driver_cuda_version)
    log.append(f"cuda_selection: detected_sms={','.join(host_sms) if host_sms else 'unknown'}")
    log.append(
        "cuda_selection: driver_runtime_lines="
        + (",".join(driver_lines) if driver_lines else "none")
    )
    if detected_runtime_lines is not None:
        log.append(
            "cuda_selection: detected_runtime_lines="
            + (",".join(detected_runtime_lines) if detected_runtime_lines else "none")
        )
        # Preserve the detected (newest-first) order, keep only driver-compatible.
        runtime_lines = [line for line in detected_runtime_lines if line in driver_lines]
    else:
        runtime_lines = list(driver_lines)
    log.append(
        "cuda_selection: compatible_runtime_lines="
        + (",".join(runtime_lines) if runtime_lines else "none")
    )
    if not runtime_lines:
        log.append("cuda_selection: no usable CUDA runtime line (driver + on-disk runtime)")
        return []

    ordered = list(runtime_lines)
    # Key on `blackwell_lines` (not host_is_blackwell): a Blackwell host whose
    # covering lines were filtered out still falls through to torch, like llama.
    blackwell_lines = (
        [line for line in blackwell_capable_runtime_lines(host_sms, candidates) if line in ordered]
        if host_is_blackwell(host_sms)
        else []
    )
    if blackwell_lines:
        ordered = blackwell_lines + [line for line in ordered if line not in blackwell_lines]
        log.append("cuda_selection: blackwell_runtime_override prefer=" + ",".join(blackwell_lines))
    elif torch_runtime_line and torch_runtime_line in ordered:
        ordered = [torch_runtime_line] + [line for line in ordered if line != torch_runtime_line]
        log.append(f"cuda_selection: torch_preferred_runtime_line={torch_runtime_line}")

    return rank_cuda_attempts(candidates, host_sms, ordered, log)


def match_rocm_artifact(
    candidates: list[SelArtifact], host_gfx: str | None, log: list[str]
) -> SelArtifact | None:
    """Select the ROCm bundle whose gfx target covers the host GPU. A host's
    concrete gfx is matched against the bundle's `mapped_targets` (exact built
    arch list) or its umbrella family label (`gfx_target`) -- never a loose
    prefix -- so an in-family but unbuilt arch (e.g. gfx1033) is not served the
    family bundle. `candidates` must already be os/arch/backend='rocm' filtered."""
    if not host_gfx:
        return None
    gfx = host_gfx.lower().strip()
    for artifact in candidates:
        mapped = {t.lower() for t in artifact.mapped_targets}
        if gfx in mapped or gfx == (artifact.gfx_target or "").lower():
            log.append(f"rocm_selection: gpu={host_gfx} selected {artifact.asset}")
            return artifact
    return None
