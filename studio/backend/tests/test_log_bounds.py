# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Each family in debug_log_sources.FAMILIES must prune old files, or its logs grow forever."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils import debug_log_sources  # noqa: E402

_STUDIO = Path(__file__).resolve().parents[2]
_MAIN_RS = _STUDIO / "src-tauri" / "src" / "main.rs"
_DIAGNOSTICS_RS = _STUDIO / "src-tauri" / "src" / "diagnostics" / "mod.rs"
_PROCESS_RS = _STUDIO / "src-tauri" / "src" / "process.rs"

# Self-expiring: the test below fails once a listed family gains retention.
KNOWN_UNBOUNDED_FAMILIES: frozenset[str] = frozenset()

# Bounded in Rust (rotation), not by a Python pruner.
_DESKTOP_FAMILIES = frozenset(
    {
        "desktop-backend",
        "desktop-install",
        "desktop-update",
        "desktop-repair",
        "desktop-shell",
    }
)


def _python_retention_sources() -> str:
    """Every Python file that could plausibly prune a log directory."""
    backend = Path(_BACKEND_DIR)
    parts = []
    for relative in ("run.py", "utils/log_retention.py", "core/inference/llama_cpp.py"):
        path = backend / relative
        if path.is_file():
            parts.append(path.read_text(encoding = "utf-8", errors = "replace"))
    return "\n".join(parts)


class TestFamiliesAreBounded:
    def test_every_python_written_family_prunes(self):
        """A family whose glob never appears beside a retention call site is flagged as unbounded."""
        source = _python_retention_sources()
        unbounded = []
        for family, (_subdir, glob) in debug_log_sources.FAMILIES.items():
            if family in _DESKTOP_FAMILIES:
                continue
            if glob not in source:
                unbounded.append(f"{family} ({glob})")

        new = sorted(
            set(unbounded)
            - {f"{f} ({debug_log_sources.FAMILIES[f][1]})" for f in KNOWN_UNBOUNDED_FAMILIES}
        )
        assert not new, (
            "these log families are written but never pruned, so they grow for the life "
            "of the install:\n  "
            + "\n  ".join(new)
            + "\n\nPrune them where they are opened, keeping the newest N and protecting "
            "the handle you just opened."
        )

    def test_the_unbounded_list_does_not_outlive_the_problem(self):
        source = _python_retention_sources()
        stale = sorted(
            family
            for family in KNOWN_UNBOUNDED_FAMILIES
            if family in debug_log_sources.FAMILIES
            and debug_log_sources.FAMILIES[family][1] in source
        )
        assert not stale, (
            "these families now prune but are still listed in "
            "KNOWN_UNBOUNDED_FAMILIES:\n  "
            + "\n  ".join(stale)
            + "\n\nDelete the entries so the list keeps meaning something."
        )

    def test_a_new_family_cannot_be_added_unnoticed(self):
        """FAMILIES is the inventory; adding to it is a decision about disk growth."""
        reviewed = {
            "server",
            "llama-server",
            "diffusion-server",
            "desktop-backend",
            "desktop-install",
            "desktop-update",
            "desktop-repair",
            "desktop-shell",
        }
        actual = set(debug_log_sources.FAMILIES)
        added = sorted(actual - reviewed)
        assert not added, (
            "new log families:\n  "
            + "\n  ".join(added)
            + "\n\nGive each one retention, then add it to the reviewed list here. A "
            "family with no pruning is an install that grows until the disk is full."
        )


class TestLineCapsAgree:
    def test_the_desktop_line_cap_matches_the_phase_log(self):
        """One line, two sinks, one length. A cap on one only is a silent asymmetry."""
        if not _DIAGNOSTICS_RS.is_file():
            pytest.skip("desktop sources not present")
        phase = re.search(
            r"MAX_PHASE_LINE_BYTES: usize = ([0-9 *]+);",
            _DIAGNOSTICS_RS.read_text(encoding = "utf-8"),
        )
        assert phase is not None, (
            "MAX_PHASE_LINE_BYTES is no longer a plain literal in diagnostics/mod.rs; "
            "this test reads it to compare the two caps"
        )
        phase_bytes = eval(phase.group(1).strip())  # noqa: S307 - digits and '*' only

        process = _PROCESS_RS.read_text(encoding = "utf-8") if _PROCESS_RS.is_file() else ""
        backend_cap = re.search(r"MAX_BACKEND_LOG_LINE_BYTES: usize = ([0-9 *]+);", process)
        if backend_cap is None:
            pytest.skip(
                "the desktop shell does not cap mirrored backend lines on this revision; "
                "this check activates when that lands"
            )
        backend_bytes = eval(backend_cap.group(1).strip())  # noqa: S307

        assert backend_bytes == phase_bytes, (
            f"tauri.log caps a backend line at {backend_bytes} bytes but the phase log "
            f"caps the same line at {phase_bytes}. The same line would be truncated on one "
            "sink and not the other, which is what makes two logs of one event disagree."
        )

    def test_the_desktop_log_still_rotates_by_size(self):
        """Keeping N files is not a bound if any one of them can be any size."""
        if not _MAIN_RS.is_file():
            pytest.skip("desktop sources not present")
        source = _MAIN_RS.read_text(encoding = "utf-8")
        assert "RotatingLogFile" in source, (
            "tauri.log no longer uses RotatingLogFile, so nothing bounds its size while "
            "the app runs"
        )
        assert re.search(r"max_log_bytes\s*=\s*[0-9 *]+;", source), (
            "the tauri.log rotation threshold is gone; a session left open for days grows "
            "the file until the next restart"
        )
