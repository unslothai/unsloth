# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.inference import audio_cpp_models as acm
from hub.utils import gguf

OFFLINE_ENVS = [
    ("HF_HUB_OFFLINE", "on"),
    ("HF_HUB_OFFLINE", " TRUE "),
    ("TRANSFORMERS_OFFLINE", "1"),
]


@pytest.fixture
def offline_env(monkeypatch, request):
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising = False)
    var, value = request.param
    monkeypatch.setenv(var, value)


@pytest.mark.parametrize("offline_env", OFFLINE_ENVS, indirect = True)
def test_gguf_listing_reads_the_cache_without_asking_the_hub(monkeypatch, offline_env):
    asked = []

    def hub_down(repo_id, hf_token):
        asked.append(repo_id)
        raise ConnectionError("offline")

    cached = ([gguf.GgufVariantInfo("m-Q4_K_M.gguf", "Q4_K_M", 5)], False, {"Q4_K_M"})
    monkeypatch.setattr(gguf, "list_gguf_variants_from_hf_cache", lambda repo_id: cached)
    # A failed Hub listing also falls back to the cache, so record the call rather than raise.
    monkeypatch.setattr(gguf, "_hub_model_info", hub_down)

    variants, _, _ = gguf.list_gguf_variants("unsloth/Demo-GGUF")

    assert [v.quant for v in variants] == ["Q4_K_M"]
    assert asked == []


@pytest.mark.parametrize(
    "env, hub_asked",
    [
        ({"HF_HUB_OFFLINE": "on"}, False),
        ({"TRANSFORMERS_OFFLINE": "1"}, False),
        ({"HF_HUB_OFFLINE": " TRUE "}, True),
        ({"HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "1"}, True),
    ],
)
def test_audio_stale_gate_follows_the_download_jobs_hub_client(
    monkeypatch, tmp_path, env, hub_asked
):
    """The job fetches whenever its Hub client is online, so only then may the gate skip the etag check."""
    import huggingface_hub

    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(var, raising = False)
    for var, value in env.items():
        monkeypatch.setenv(var, value)
    asked = []
    (tmp_path / "m.gguf").write_bytes(b"GGUF")
    monkeypatch.setattr(acm, "snapshot_dirs", lambda repo_id, hub_cache = None: [tmp_path])
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", lambda *a, **k: asked.append(a))

    assert acm._fetches_gguf("audio-cpp/demo", "m.gguf", None) is False
    assert bool(asked) is hub_asked


@pytest.mark.parametrize(
    "var, value, hub_asked",
    [
        ("HF_HUB_OFFLINE", "on", False),
        ("TRANSFORMERS_OFFLINE", "1", False),
        ("HF_HUB_OFFLINE", " TRUE ", True),
    ],
)
def test_trusted_org_skips_verification_only_when_the_hub_client_is_offline(var, value, hub_asked):
    """A fresh process, since huggingface_hub reads the env at import: " TRUE " leaves its client online, so ownership is verified."""
    code = (
        "from unittest.mock import MagicMock, patch\n"
        "from utils.security.trusted_org import is_trusted_org_repo\n"
        "api = MagicMock()\n"
        "api.return_value.model_info.side_effect = ConnectionError('down')\n"
        "with patch('huggingface_hub.HfApi', api):\n"
        "    print(is_trusted_org_repo('unsloth/Demo'), api.return_value.model_info.called)\n"
    )
    env = {
        k: v for k, v in os.environ.items() if k not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    env[var] = value
    out = subprocess.run(
        [sys.executable, "-c", code],
        cwd = Path(__file__).resolve().parent.parent,
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split()[-2:] == [str(not hub_asked), str(hub_asked)]
