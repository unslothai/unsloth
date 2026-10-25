# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vulkan drafts sample on the CPU when the vocab misaligns the logits rows (#14987)."""

from __future__ import annotations

import struct
import subprocess
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference.llama_cpp import (  # noqa: E402
    GgufLoadIntent,
    LlamaCppBackend,
    _draft_backend_sampling_off,
)

_FLAG = "--no-spec-draft-backend-sampling"
_CAPS = {"flags": {_FLAG: "", "--spec-draft-backend-sampling": ""}}
_DSPARK = [
    "--model-draft",
    "/m/dspark.gguf",
    "--spec-type",
    "draft-dspark",
    "--spec-draft-n-max",
    "3",
]
_REAL_POPEN = subprocess.Popen

_MINICPM_V47_VOCAB = 248094


@pytest.mark.parametrize(
    "spec_flags",
    [
        _DSPARK,
        [
            "--model-draft",
            "/m/dflash.gguf",
            "--spec-type",
            "draft-dflash",
            "--spec-draft-n-max",
            "3",
        ],
        ["--spec-type", "mtp", "--spec-draft-n-max", "2"],
        ["--spec-type", "draft-mtp,ngram-mod", "--spec-draft-n-max", "2"],
    ],
)
def test_misaligned_vocab_on_vulkan_samples_drafts_on_the_cpu(spec_flags):
    assert _draft_backend_sampling_off(spec_flags, _CAPS, _MINICPM_V47_VOCAB, True) == [_FLAG]


def test_unknown_vocab_on_vulkan_takes_the_safe_side():
    assert _draft_backend_sampling_off(_DSPARK, _CAPS, None, True) == [_FLAG]


@pytest.mark.parametrize("vocab", [151936, 248320, 262144, 128256, 201088])
def test_aligned_vocabs_keep_backend_sampling(vocab):
    assert _draft_backend_sampling_off(_DSPARK, _CAPS, vocab, True) == []


def test_other_backends_are_untouched():
    assert _draft_backend_sampling_off(_DSPARK, _CAPS, _MINICPM_V47_VOCAB, False) == []


@pytest.mark.parametrize(
    "spec_flags",
    [[], ["--spec-default"], ["--spec-type", "ngram-mod", "--spec-ngram-mod-n-max", "8"]],
)
def test_no_draft_model_means_no_flag(spec_flags):
    assert _draft_backend_sampling_off(spec_flags, _CAPS, _MINICPM_V47_VOCAB, True) == []


def test_a_build_without_the_flag_is_not_handed_it():
    # llama-server exits on an argument it does not know, taking the load with it.
    assert _draft_backend_sampling_off(_DSPARK, {"flags": {}}, _MINICPM_V47_VOCAB, True) == []
    assert _draft_backend_sampling_off(_DSPARK, {}, _MINICPM_V47_VOCAB, True) == []


def _write_gguf(path: Path) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("qwen35")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _launch(
    tmp_path,
    monkeypatch,
    *,
    vulkan,
    vocab,
    extra_args = None,
):
    caps = dict(LlamaCppBackend.probe_server_capabilities("/nonexistent/llama-server"))
    caps.update(
        found = True,
        supports_dspark = True,
        mtp_probe_inconclusive = False,
        spec_draft_n_max_flag = "--spec-draft-n-max",
        help_probe_ok = True,
        flags = dict(_CAPS["flags"]),
    )
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, b = None: caps)
    )
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []

    def read_metadata(_path):
        backend._vocab_size = vocab

    backend._read_gguf_metadata = read_metadata
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    backend._wait_for_health = lambda timeout, **_kw: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    draft = tmp_path / "dspark-model-F16.gguf"
    draft.write_bytes(b"GGUF")

    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(_write_gguf(tmp_path / "model.gguf")),
                dspark_draft_path = str(draft),
                model_identifier = "openbmb/MiniCPM-V-4.7-1B-gguf",
                speculative_type = "dspark",
                n_ctx = 4096,
                extra_args = extra_args,
            )
        )
    return captured["cmd"]


def test_load_model_launches_a_misaligned_vulkan_dspark_without_backend_draft_sampling(
    tmp_path, monkeypatch
):
    cmd = _launch(tmp_path, monkeypatch, vulkan = True, vocab = _MINICPM_V47_VOCAB)
    assert "draft-dspark" in cmd
    assert cmd.count(_FLAG) == 1


@pytest.mark.parametrize("vulkan, vocab", [(False, _MINICPM_V47_VOCAB), (True, 248320)])
def test_load_model_leaves_every_other_dspark_launch_as_it_was(
    tmp_path, monkeypatch, vulkan, vocab
):
    cmd = _launch(tmp_path, monkeypatch, vulkan = vulkan, vocab = vocab)
    assert "draft-dspark" in cmd
    assert _FLAG not in cmd


def test_an_explicit_backend_sampling_extra_still_wins(tmp_path, monkeypatch):
    # llama.cpp is last-wins and the extras follow the spec block.
    cmd = _launch(
        tmp_path,
        monkeypatch,
        vulkan = True,
        vocab = _MINICPM_V47_VOCAB,
        extra_args = ["--spec-draft-backend-sampling"],
    )
    assert cmd.index("--spec-draft-backend-sampling") > cmd.index(_FLAG)
