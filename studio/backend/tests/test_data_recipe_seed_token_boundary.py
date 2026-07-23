# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_SEED_PATH = _BACKEND_ROOT / "routes" / "data_recipe" / "seed.py"
_SPEC = importlib.util.spec_from_file_location("seed_token_boundary_under_test", _SEED_PATH)
seed = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(seed)


@pytest.mark.parametrize(
    ("token", "expected"),
    ((False, False), ("request-token", "request-token")),
)
def test_seed_hf_preview_forwards_only_explicit_token(monkeypatch, token, expected):
    captured = []

    class _Api:
        def list_repo_files(self, *_args, **kwargs):
            captured.append(kwargs["token"])
            return ["train/data.parquet"]

    monkeypatch.setenv("HF_TOKEN", "operator-secret-token")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi = _Api))
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub.utils",
        SimpleNamespace(HfHubHTTPError = RuntimeError),
    )

    files = seed._list_hf_data_files(dataset_name = "Org/Data", token = token)
    kwargs = seed._build_stream_load_kwargs(
        dataset_name = "Org/Data",
        split = "train",
        subset = None,
        token = token,
    )

    assert files == ["train/data.parquet"]
    assert captured == [expected]
    assert kwargs["token"] == expected
