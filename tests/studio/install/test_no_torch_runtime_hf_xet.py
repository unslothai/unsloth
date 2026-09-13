# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF-only installs apply no-torch-runtime.txt with --no-deps, so Hub extras must be declared."""

from __future__ import annotations

import pathlib

from packaging.requirements import Requirement

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
NO_TORCH_RUNTIME = REPO_ROOT / "studio" / "backend" / "requirements" / "no-torch-runtime.txt"


def _requirement_names(path: pathlib.Path) -> set[str]:
    names = set()
    for line in path.read_text(encoding = "utf-8").splitlines():
        text = line.split("#", 1)[0].strip()
        if not text or text.startswith("-"):
            continue
        names.add(Requirement(text).name.lower().replace("_", "-"))
    return names


def test_no_torch_runtime_declares_hf_xet_for_hub_large_downloads():
    names = _requirement_names(NO_TORCH_RUNTIME)
    assert "hf-xet" in names, (
        "no-torch-runtime.txt is installed --no-deps; without an explicit hf-xet pin, "
        "Desktop GGUF-only envs fall back to HTTP and huggingface_hub raises on large blobs."
    )
