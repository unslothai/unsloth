# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Model listing and catalog classification must not import core.inference.diffusion."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent


def _probe(tmp_path, call: str) -> list:
    (tmp_path / "model.safetensors").write_bytes(b"")
    probe = (
        "import json, sys\n"
        # cwd alone does not select this checkout under PYTHONSAFEPATH=1.
        f"sys.path.insert(0, {str(_BACKEND)!r})\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        f"d = Path({str(tmp_path)!r})\n"
        f"found = {call}\n"
        "print(json.dumps([found, 'core.inference.diffusion' in sys.modules]))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output = True,
        text = True,
        cwd = str(_BACKEND),
        timeout = 300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_scanning_a_models_folder_does_not_import_the_diffusion_stack(tmp_path):
    found, loaded = _probe(
        tmp_path,
        "[m.path for m in __import__('routes.models', fromlist = ['_'])._scan_models_dir(d)]",
    )
    assert found == [str(tmp_path)]
    assert loaded is False


def test_catalog_classification_does_not_import_the_diffusion_stack(tmp_path):
    found, loaded = _probe(
        tmp_path,
        "'model.safetensors' in __import__('hub.services.models.catalog_classification',"
        " fromlist = ['_'])._local_family_needles(SimpleNamespace(model_id = None,"
        " display_name = d.name, id = str(d), path = str(d)))",
    )
    assert found is True
    assert loaded is False


_LOCAL_MODEL = (
    "SimpleNamespace(model_id = None, display_name = 'flux1-dev', id = str(d), path = str(d),"
    " model_format = 'safetensors', base_model = None)"
)


def test_classifying_a_local_image_checkpoint_does_not_import_the_diffusion_stack(tmp_path):
    task, loaded = _probe(
        tmp_path,
        "__import__('hub.services.models.catalog_classification', fromlist = ['_'])"
        f"._local_model_task({_LOCAL_MODEL})",
    )
    assert task == "text-to-image"
    assert loaded is False


def test_classifying_a_cached_non_video_repo_does_not_import_the_diffusion_stack(tmp_path):
    task, loaded = _probe(
        tmp_path,
        "__import__('hub.services.models.catalog_classification', fromlist = ['_'])"
        "._cached_repo_task(SimpleNamespace(repo_id = 'unsloth/Llama-3.2-1B-Instruct',"
        " revisions = ()), None)",
    )
    assert task is None
    assert loaded is False
