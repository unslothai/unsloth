# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Listing a models folder asks whether it is a bare single-file checkpoint. That check is path-only,
so it must not import core.inference.diffusion (15k lines plus its pipeline siblings)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent


def test_diffusion_re_exports_the_same_function():
    from core.inference import diffusion, diffusion_local_files
    assert diffusion.resolve_local_single_file is diffusion_local_files.resolve_local_single_file


def test_scanning_a_models_folder_does_not_import_the_diffusion_stack(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    probe = (
        "import json, sys\n"
        "from pathlib import Path\n"
        "from core.inference.diffusion_local_files import resolve_local_single_file\n"
        f"sole = resolve_local_single_file({str(tmp_path)!r})\n"
        "print(json.dumps([sole, 'core.inference.diffusion' in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output = True,
        text = True,
        cwd = str(_BACKEND),
    )
    assert result.returncode == 0, result.stderr
    sole, loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert sole == "model.safetensors"
    assert loaded is False


def test_the_listing_and_catalog_paths_use_the_light_module():
    for relative in ("routes/models.py", "hub/services/models/catalog_classification.py"):
        source = (_BACKEND / relative).read_text(encoding = "utf-8")
        assert (
            "from core.inference.diffusion_local_files import resolve_local_single_file" in source
        )
        assert "from core.inference.diffusion import resolve_local_single_file" not in source
