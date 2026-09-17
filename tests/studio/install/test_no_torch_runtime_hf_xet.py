# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF-only installs apply no-torch-runtime.txt with --no-deps, so Hub extras must be declared."""

from __future__ import annotations

import pathlib

from packaging.requirements import Requirement
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
NO_TORCH_RUNTIME = REPO_ROOT / "studio" / "backend" / "requirements" / "no-torch-runtime.txt"
STUDIO_TXT = REPO_ROOT / "studio" / "backend" / "requirements" / "studio.txt"


def _requirement_names(path: pathlib.Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding = "utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text or text.startswith("-"):
            continue
        names.add(Requirement(text).name.lower().replace("_", "-"))
    return names


def _hf_xet_requirements(path: pathlib.Path) -> list[Requirement]:
    out = []
    for line in path.read_text(encoding = "utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text or text.startswith("-"):
            continue
        req = Requirement(text)
        if req.name.lower().replace("_", "-") == "hf-xet":
            out.append(req)
    return out


def test_no_torch_runtime_declares_hf_xet_for_hub_large_downloads():
    names = _requirement_names(NO_TORCH_RUNTIME)
    assert "hf-xet" in names, (
        "no-torch-runtime.txt is installed --no-deps; without an explicit hf-xet pin, "
        "Desktop GGUF-only envs fall back to HTTP and huggingface_hub raises on large blobs."
    )


def test_hf_xet_pins_are_python_310_plus_only():
    """hf-xet has no wheels for 3.9; Python < 3.10 keeps huggingface-hub 0.36.x without Xet."""
    for path in (NO_TORCH_RUNTIME, STUDIO_TXT):
        pins = _hf_xet_requirements(path)
        assert len(pins) == 1, f"expected one hf-xet line in {path.name}, got {pins!r}"
        marker = pins[0].marker
        assert marker is not None, f"{path.name} hf-xet must be environment-marked"
        assert marker.evaluate({"python_version": "3.9"}) is False
        assert marker.evaluate({"python_version": "3.10"}) is True
        assert marker.evaluate({"python_version": "3.11"}) is True
