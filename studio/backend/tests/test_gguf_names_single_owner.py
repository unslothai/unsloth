# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

from utils import gguf_archs, gguf_names  # noqa: E402


def test_every_layer_reads_the_same_rule_objects():
    import hub.utils.gguf as hub_gguf
    import utils.models.model_config as model_config
    from core.inference import llama_cpp

    assert hub_gguf._GGUF_QUANT_RE is gguf_names.GGUF_QUANT_RE
    assert model_config._GGUF_KNOWN_QUANT_RE is gguf_names.GGUF_QUANT_RE
    assert llama_cpp._GGUF_KNOWN_QUANT_RE is gguf_names.GGUF_QUANT_RE
    for module in (hub_gguf, model_config, llama_cpp):
        assert module._DRAFTER_KINDS is gguf_names.DRAFTER_KINDS
        assert module._DRAFTER_DIR_KINDS is gguf_names.DRAFTER_DIR_KINDS
        assert module._DSPARK_BEFORE_QUANT_RE is gguf_names.DSPARK_BEFORE_QUANT_RE
        assert module._IMATRIX_TOKEN_RE is gguf_names.IMATRIX_TOKEN_RE


def test_the_chat_refusal_and_the_listing_share_every_arch_set():
    from core.inference.llama_cpp import LlamaCppBackend
    import hub.services.models.catalog_classification as catalog

    assert LlamaCppBackend._IMAGE_ARCHES is catalog._DIFFUSION_GGUF_ARCHS
    assert LlamaCppBackend._AMBIGUOUS_IMAGE_ARCHES is catalog._AMBIGUOUS_DIFFUSION_GGUF_ARCHS
    assert LlamaCppBackend._VIDEO_ARCHES is catalog._VIDEO_GGUF_ARCHS
    assert LlamaCppBackend._UNRUNNABLE_MEDIA_ARCHES is catalog._UNSUPPORTED_DIFFUSION_GGUF_ARCHS
    assert LlamaCppBackend._PLACEHOLDER_ARCHES is catalog._PLACEHOLDER_DIFFUSION_GGUF_ARCHS


@pytest.mark.parametrize("arch", ["flux2", "z_image", "zimage", "qwenimage", "qwenimage21"])
def test_image_archs_the_old_chat_gate_missed_are_refused_before_launch(arch):
    from core.inference.llama_cpp import LlamaCppBackend
    from hub.services.models.catalog_classification import _arch_to_task

    assert arch in LlamaCppBackend._DIFFUSION_ARCHES
    assert _arch_to_task(arch) in ("text-to-image", "image-diffusion-unsupported")


@pytest.mark.parametrize(
    "path,companion",
    [
        ("mtp-Q8_0.gguf", True),
        ("MTP/gemma-4-E2B-it-Q8_0-MTP.gguf", True),
        ("dspark/dspark-DeepSeek-V4-Flash-Q8_0.gguf", True),
        ("Bonsai-8B-dspark-Q8_0.gguf", True),
        ("eagle3-gpt-oss-20b.gguf", True),
        ("lilicorr-Nemotron-3.5-Q8_0.gguf", True),
        ("dflash-Qwen3.6-27B-Q8_0.gguf", True),
        ("Qwen3.6-35B-A3B-DFlash-Q4_K_M.gguf", False),
        ("dflash/Qwen3.6-27B-Q4_K_M.gguf", False),
        ("imatrix_unsloth.gguf", True),
        ("Qwen3-Imatrix-Tuned-Q4_K_M.gguf", False),
        ("gemma-4-E2B-it-Q4_K_M.gguf", False),
    ],
)
def test_hub_loader_and_backend_agree_on_companions(path, companion):
    import hub.utils.gguf as hub_gguf
    import utils.models.model_config as model_config
    from core.inference import llama_cpp

    hub_says = hub_gguf.is_mtp_drafter_path(path) or hub_gguf.is_imatrix_filename(path)
    loader_says = model_config._is_mtp_drafter(path) or model_config._is_imatrix_path(path)
    assert hub_says == loader_says == llama_cpp._is_companion_gguf_path(path) == companion


def test_cli_drafter_kinds_match_the_server():
    """Runs in a subprocess: importing unsloth_cli here would shadow backend modules (``state``)."""
    import json
    import subprocess

    repo = _BACKEND.parents[1]
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from unsloth_cli.commands import start; "
            "print(json.dumps([start._DRAFTER_KINDS, start._DRAFTER_DIR_KINDS]))",
        ],
        cwd = str(repo),
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    kinds, dir_kinds = json.loads(out.stdout.strip().splitlines()[-1])
    assert tuple(kinds) == gguf_names.DRAFTER_KINDS
    assert tuple(dir_kinds) == gguf_names.DRAFTER_DIR_KINDS


def test_the_leaf_modules_import_nothing_from_the_layers_they_serve():
    import ast
    for module in (gguf_names, gguf_archs):
        tree = ast.parse(Path(module.__file__).read_text(encoding = "utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] in ("__future__", "re", "typing")
            elif isinstance(node, ast.Import):
                assert all(alias.name in ("re",) for alias in node.names)


def _hf_cache_repo(tmp_path, files):
    repo = tmp_path / "models--org--m-GGUF"
    snap = repo / "snapshots" / "abc123"
    snap.mkdir(parents = True)
    for name, size in files.items():
        (snap / name).write_bytes(b"x" * size)
    return repo


def test_auto_switch_does_not_offer_a_torn_split_quant_from_the_hf_cache(tmp_path):
    from types import SimpleNamespace

    from core.inference import local_model_resolver as resolver

    repo = _hf_cache_repo(tmp_path, {"m-Q4_K_M-00001-of-00003.gguf": 400})
    entry = resolver._local_gguf_entry(
        "org/m-GGUF", SimpleNamespace(path = str(repo), source = "hf_cache")
    )
    assert entry is None


def test_a_complete_quant_beside_a_torn_one_is_still_offered(tmp_path):
    from types import SimpleNamespace

    from core.inference import local_model_resolver as resolver

    repo = _hf_cache_repo(tmp_path, {"m-Q8_0.gguf": 300, "m-Q4_K_M-00001-of-00003.gguf": 400})
    entry = resolver._local_gguf_entry(
        "org/m-GGUF", SimpleNamespace(path = str(repo), source = "hf_cache")
    )
    assert entry is not None
    assert [q.upper() for q in entry.variants] == ["Q8_0"]


def test_the_inventory_row_calls_a_torn_split_partial_without_any_marker(tmp_path):
    from hub.utils.inventory_scan import is_gguf_repo_partial

    repo = _hf_cache_repo(tmp_path, {"m-Q4_K_M-00001-of-00003.gguf": 400})
    snapshot = repo / "snapshots" / "abc123"
    assert is_gguf_repo_partial("org/m-GGUF", repo, snapshot_dir = snapshot) is True


def test_a_whole_quant_keeps_the_row_complete(tmp_path):
    from hub.utils.inventory_scan import is_gguf_repo_partial

    repo = _hf_cache_repo(tmp_path, {"m-Q8_0.gguf": 300, "m-Q4_K_M-00001-of-00003.gguf": 400})
    snapshot = repo / "snapshots" / "abc123"
    assert is_gguf_repo_partial("org/m-GGUF", repo, snapshot_dir = snapshot) is False


def test_an_older_whole_revision_keeps_the_row_complete_beside_a_newer_torn_one(tmp_path):
    import os

    from hub.utils.inventory_scan import is_gguf_repo_partial

    repo = tmp_path / "models--org--m-GGUF"
    old = repo / "snapshots" / "old"
    new = repo / "snapshots" / "new"
    old.mkdir(parents = True)
    new.mkdir(parents = True)
    (old / "m-Q4_K_M.gguf").write_bytes(b"GGUF" + b"\0" * 60)
    (new / "m-Q8_0-00001-of-00002.gguf").write_bytes(b"GGUF" + b"\0" * 60)
    os.utime(old, (1, 1))
    assert is_gguf_repo_partial("org/m-GGUF", repo) is False
    (old / "m-Q4_K_M.gguf").unlink()
    os.utime(old, (1, 1))
    assert is_gguf_repo_partial("org/m-GGUF", repo) is True


def test_a_request_for_a_torn_only_repo_is_refused_not_answered_by_the_resident(
    tmp_path, monkeypatch
):
    import asyncio

    from fastapi import HTTPException

    import routes.inference as inference_route
    from core.inference import local_model_resolver as resolver

    repo = tmp_path / "models--publisher--My-Model"
    snapshot = repo / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "My-Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"GGUF" + b"\0" * 60)
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda **_kw: [tmp_path])
    assert resolver.local_gguf_torn_only("publisher/My-Model") is True
    assert resolver.local_gguf_torn_only("publisher/Other") is False

    loaded = type(
        "L",
        (),
        {
            "is_loaded": True,
            "model_identifier": "unsloth/A-GGUF",
            "hf_variant": "Q4_K_M",
            "_openai_advertised_id": None,
        },
    )()
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: loaded)
    monkeypatch.setattr(
        inference_route,
        "get_inference_backend",
        lambda: type("B", (), {"active_model_name": None})(),
    )
    monkeypatch.setattr(resolver, "resolve_local_gguf", lambda name, **_kw: None)
    monkeypatch.setattr(resolver, "index_is_built", lambda: True)
    monkeypatch.setattr(resolver, "warm_index_soon", lambda: None)
    monkeypatch.setattr(
        "utils.openai_auto_switch_settings.get_openai_auto_switch_enabled", lambda: False
    )
    request = type("R", (), {"url": type("U", (), {"path": "/v1/chat/completions"})()})()
    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(inference_route._reject_unservable_model("publisher/My-Model", request))
    assert excinfo.value.status_code == 404
    assert "partly downloaded" in str(excinfo.value.detail)
    assert asyncio.run(inference_route._reject_unservable_model("publisher/Other", request)) is None
