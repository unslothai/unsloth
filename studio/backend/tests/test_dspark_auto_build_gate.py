# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto promotes DSpark on CUDA/HIP/Vulkan only from the first build with the DSpark speed fixes; Metal
and an explicit dspark choice keep today's behaviour."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_llama_cpp_mtp_detection import LlamaCppBackend  # noqa: E402 (stubs are installed on import)

import core.inference.llama_cpp as llama_cpp  # noqa: E402

_SHIPPED = 12776  # b12776-mix-ef2df21, measured below 1x on T4/L4/A100/B200/N1X


def _gate(
    monkeypatch,
    *,
    backends,
    metal = False,
    build = _SHIPPED,
    spec = "auto",
):
    calls = []

    def _build(cls, binary = None):
        calls.append("/fake/llama-server")
        if isinstance(build, Exception):
            raise build
        return build

    monkeypatch.setattr(LlamaCppBackend, "probe_build_number", classmethod(_build))
    monkeypatch.setattr(
        LlamaCppBackend, "_build_has_metal", staticmethod(lambda binary = None: metal)
    )
    # A real install layout: the ggml libraries beside llama-server name its backends.
    with tempfile.TemporaryDirectory() as root:
        binary = Path(root) / "llama-server"
        binary.write_bytes(b"\x7fELF")
        for name in backends:
            (Path(root) / f"libggml-{name}.so").write_bytes(b"")
        won = LlamaCppBackend()._dspark_wins_auto(
            binary = str(binary),
            dspark_draft_path = "/fake/dspark-model-Q8_0.gguf",
            spec_canon = spec,
            extra_args = None,
            caps = lambda binary: {"supports_dspark": True},
        )
    return won, calls


@pytest.mark.parametrize("gpu", ["cuda", "hip", "vulkan"])
def test_auto_skips_dspark_on_a_pre_fix_gpu_build(monkeypatch, gpu):
    won, calls = _gate(monkeypatch, backends = {"base", "cpu", gpu})
    assert _SHIPPED < llama_cpp._DSPARK_AUTO_MIN_BUILD
    assert won is False
    assert calls == ["/fake/llama-server"]


def test_auto_picks_dspark_on_cuda_at_the_threshold(monkeypatch):
    won, _ = _gate(monkeypatch, backends = {"cpu", "cuda"}, build = llama_cpp._DSPARK_AUTO_MIN_BUILD)
    assert won is True
    won, _ = _gate(
        monkeypatch, backends = {"cpu", "cuda"}, build = llama_cpp._DSPARK_AUTO_MIN_BUILD - 1
    )
    assert won is False


def test_metal_keeps_auto_dspark_on_the_shipped_build(monkeypatch):
    won, calls = _gate(monkeypatch, backends = set(), metal = True)
    assert won is True
    assert calls == []  # no build probe on Metal


def test_cpu_only_build_is_not_gated(monkeypatch):
    won, calls = _gate(monkeypatch, backends = {"base", "cpu"})
    assert won is True
    assert calls == []


def test_explicit_dspark_is_not_an_auto_decision(monkeypatch):
    # An explicit choice never goes through the Auto promotion, so the build gate cannot touch it.
    won, calls = _gate(monkeypatch, backends = {"cuda"}, spec = "dspark")
    assert won is False
    assert calls == []


@pytest.mark.parametrize("build", [None, RuntimeError("--version failed")])
def test_auto_skips_dspark_when_the_build_probe_fails(monkeypatch, build):
    won, _ = _gate(monkeypatch, backends = {"cuda"}, build = build)
    assert won is False


def test_explicit_dspark_still_emits_draft_dspark_on_a_pre_fix_cuda_build(monkeypatch, tmp_path):
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(
            lambda cls, binary = None: {
                "found": True,
                "mtp_token": "draft-mtp",
                "supports_mtp": True,
                "supports_dspark": True,
                "mtp_probe_inconclusive": False,
                "ngram_mod_flavor": "new",
                "supports_ngram_mod": True,
                "spec_draft_n_max_flag": "--spec-draft-n-max",
            }
        ),
    )
    monkeypatch.setattr(
        LlamaCppBackend, "probe_build_number", classmethod(lambda cls, binary = None: _SHIPPED)
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_installed_ggml_backends",
        staticmethod(lambda binary = None: frozenset({"cuda"})),
    )
    sidecar = tmp_path / "dspark-DeepSeek-V4-Flash-0731-Q8_0.gguf"
    sidecar.write_bytes(b"draft")
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = None
    flags = backend._build_speculative_flags(
        speculative_type = "dspark",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = "unsloth/DeepSeek-V4-Flash-0731-GGUF",
        model_path = None,
        gpus = True,
        binary = "/fake/llama-server",
        dspark_draft_path = str(sidecar),
    )
    assert "draft-dspark" in flags


@pytest.mark.parametrize(
    ("backends", "plugin"),
    [
        (set(), None),  # static build: no ggml libraries beside the binary
        ({"base", "cpu", "sycl"}, None),
        ({"base", "cpu", "opencl"}, None),
        ({"base", "cpu", "cann"}, None),
        ({"base", "cpu", "musa"}, None),
        ({"base", "cpu", "virtgpu"}, None),
        ({"base", "cpu"}, "/opt/ggml/libggml-vulkan.so"),  # GPU backend loaded as a plugin
    ],
    ids = ["static", "sycl", "opencl", "cann", "musa", "virtgpu", "backend_path_plugin"],
)
def test_builds_not_proven_cpu_only_are_gated(monkeypatch, backends, plugin):
    if plugin:
        monkeypatch.setenv("GGML_BACKEND_PATH", plugin)
    else:
        monkeypatch.delenv("GGML_BACKEND_PATH", raising = False)
    won, calls = _gate(monkeypatch, backends = backends)
    assert won is False
    assert calls == ["/fake/llama-server"]
    won, _ = _gate(monkeypatch, backends = backends, build = llama_cpp._DSPARK_AUTO_MIN_BUILD)
    assert won is True


def _hub_load_fetches(monkeypatch, allowed):
    """A first Hub load in Auto, stopped at launch; records whether the DSpark sidecar is fetched."""
    from test_llama_cpp_remote_non_chat_gguf_preflight import _chat_repo_load

    fetched = []
    backend = LlamaCppBackend()
    with pytest.raises(RuntimeError, match = "stop here"):
        _chat_repo_load(
            monkeypatch,
            backend,
            [],
            _dspark_auto_build_allowed = lambda _binary = None: allowed,
            _download_dspark = lambda **_k: fetched.append("dspark"),
        )
    return fetched


@pytest.mark.parametrize("allowed", [False, True])
def test_auto_fetches_the_dspark_sidecar_only_past_the_build_gate(monkeypatch, allowed):
    fetched = _hub_load_fetches(monkeypatch, allowed)
    assert fetched == (["dspark"] if allowed else [])


def test_the_training_guard_charges_a_dspark_sidecar_only_past_the_build_gate(tmp_path):
    from unittest.mock import patch

    from test_chat_load_during_training import _gguf_cfg, _load_inference_route

    route = _load_inference_route()
    target = tmp_path / "model.gguf"
    sidecar = tmp_path / "dspark-model-Q8_0.gguf"
    target.write_bytes(b"t" * 2000)
    sidecar.write_bytes(b"s" * 3000)
    cfg = _gguf_cfg(
        gguf_file = str(target),
        gguf_dspark_file = str(sidecar),
        gguf_hf_repo = None,
        gguf_variant = None,
    )

    def _required(mode, allowed):
        with (
            patch.object(route, "_estimate_gguf_kv_gb", return_value = 0.0),
            patch.object(route, "_remote_gguf_compute_reserve_gb", return_value = 0.0),
            patch.object(
                LlamaCppBackend,
                "probe_server_capabilities",
                classmethod(lambda cls, binary = None: {"supports_dspark": True}),
            ),
            patch.object(
                LlamaCppBackend,
                "_dspark_auto_build_allowed",
                classmethod(lambda cls, binary = None: allowed),
            ),
        ):
            return route._estimate_gguf_required_gb(cfg, speculative_type = mode)

    assert _required("auto", True) == pytest.approx(5000 / 1024**3)
    assert _required("auto", False) == pytest.approx(2000 / 1024**3)
    # An explicit choice launches the sidecar whatever the build, so it stays charged.
    assert _required("dspark", False) == pytest.approx(5000 / 1024**3)
