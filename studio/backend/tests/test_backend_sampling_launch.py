# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server's GPU sampler (--backend-sampling) on the launches where it pays off.

Without it every decode step copies each slot's full-vocabulary logits to the host and
runs the sampler chain on the CPU. Measured on a Colab L4 with 4 slots (-c 8192
--kv-unified, Studio's own sampler chains): Qwen3.5-0.8B 20.3 -> 7.0 ms per token at
4 streams, Qwen3-0.6B 7.3 -> 6.1 ms, flood throughput up to 2.2x, for ~20-25 ms more TTFT
per request. With identical logits and seed it draws the same tokens as the CPU chain. It is per request and steps aside on
its own for grammar / tools, a reasoning budget and pre-sampling logprobs.

The gate: the binary must advertise the flag (an unknown argument stops an older
llama-server from starting), a GPU must be there and hold the output layer, the backend
must be one it was measured on (not Vulkan, not macOS), and no speculative decoding may
run (penalties cannot run on the GPU when a step verifies several tokens per slot). The
user's own flag and UNSLOTH_LLAMA_BACKEND_SAMPLING=0 win.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_llama_cpp_placement import _backend, _launch, _write_gguf  # noqa: E402

import core.inference.llama_cpp as llama_cpp  # noqa: E402
from core.inference.llama_cpp import (  # noqa: E402
    LlamaCppBackend,
    _backend_sampling_flags,
    _launch_speculates,
)

# As llama.cpp #269 (18b4ec5f1) prints it: common/arg.cpp:2347.
NEW_HELP = """usage: llama-server [options]

-m,    --model FNAME                    model path
-bs,   --backend-sampling               enable backend sampling (experimental) (default: disabled)
                                        (env: LLAMA_ARG_BACKEND_SAMPLING)
--spec-draft-backend-sampling, --no-spec-draft-backend-sampling
                                        offload draft sampling to the backend (default: enabled)
--kv-unified, -kvu                      use single unified KV buffer shared across all sequences
"""
# A build from before backend sampling; the draft-only namesake must not count.
OLD_HELP = """usage: llama-server [options]

-m,    --model FNAME                    model path
--spec-draft-backend-sampling, --no-spec-draft-backend-sampling
                                        offload draft sampling to the backend (default: enabled)
--kv-unified, -kvu                      use single unified KV buffer shared across all sequences
"""
REMOVED_HELP = """usage: llama-server [options]

-bs,   --backend-sampling               (argument has been removed)
"""

ON = {"supports_backend_sampling": True}
GPU = {"gpus_detected": True, "vulkan": False, "platform": "linux", "env": {}, "spec_env": {}}


def _probe(
    tmp_path,
    monkeypatch,
    help_text,
    returncode = 0,
):
    binary = tmp_path / "llama-server"
    binary.write_text("")
    completed = subprocess.CompletedProcess(
        args = [str(binary), "--help"], returncode = returncode, stdout = help_text, stderr = ""
    )
    monkeypatch.setattr(llama_cpp.subprocess, "run", lambda *a, **k: completed)
    LlamaCppBackend._capability_cache.clear()
    return LlamaCppBackend.probe_server_capabilities(str(binary))


class TestTheProbe:
    def test_a_build_with_the_flag_advertises_it(self, tmp_path, monkeypatch):
        assert _probe(tmp_path, monkeypatch, NEW_HELP)["supports_backend_sampling"] is True

    @pytest.mark.parametrize(
        "help_text,returncode",
        [(OLD_HELP, 0), (REMOVED_HELP, 0), ("", 0), ("", 1)],
        ids = ["old-build", "removal-stub", "empty-help", "failed-probe"],
    )
    def test_the_gate_is_closed_on_anything_but_a_positive_answer(
        self, tmp_path, monkeypatch, help_text, returncode
    ):
        caps = _probe(tmp_path, monkeypatch, help_text, returncode)
        assert caps["supports_backend_sampling"] is False

    def test_a_missing_binary_reports_it_unsupported(self, tmp_path):
        LlamaCppBackend._capability_cache.clear()
        caps = LlamaCppBackend.probe_server_capabilities(str(tmp_path / "nope"))
        assert caps["supports_backend_sampling"] is False


class TestTheGate:
    BASE = ["/x/llama-server", "-m", "m.gguf", "--parallel", "4", "-ngl", "-1"]

    def test_a_plain_gpu_launch_gets_it(self):
        assert _backend_sampling_flags(ON, self.BASE, **GPU) == ["--backend-sampling"]

    @pytest.mark.parametrize(
        "caps", [{}, {"supports_backend_sampling": False}], ids = ["unknown", "old"]
    )
    def test_an_older_build_gets_nothing(self, caps):
        assert _backend_sampling_flags(caps, self.BASE, **GPU) == []

    @pytest.mark.parametrize(
        "kw",
        [{"gpus_detected": False}, {"vulkan": True}, {"cuda_build": False}, {"platform": "darwin"}],
        ids = ["no-gpu", "vulkan", "rocm-or-other", "macos"],
    )
    def test_unmeasured_or_cpu_backends_get_nothing(self, kw):
        assert _backend_sampling_flags(ON, self.BASE, **{**GPU, **kw}) == []

    @pytest.mark.parametrize(
        "tail",
        [["-ngl", "0"], ["--gpu-layers", "0"], ["--n-gpu-layers=0"], ["--device", "none"]],
        ids = ["ngl0", "gpu-layers0", "inline", "device-none"],
    )
    def test_a_cpu_placed_model_gets_nothing(self, tail):
        assert _backend_sampling_flags(ON, self.BASE + tail, **GPU) == []

    def test_a_later_gpu_layer_count_wins(self):
        argv = self.BASE + ["-ngl", "0", "-ngl", "99"]
        assert _backend_sampling_flags(ON, argv, **GPU) == ["--backend-sampling"]

    @pytest.mark.parametrize(
        "tail",
        [
            ["--spec-default"],
            ["--spec-type", "ngram-mod"],
            ["--spec-type=draft-mtp"],
            ["-md", "d.gguf"],
        ],
        ids = ["spec-default", "ngram-mod", "mtp-inline", "draft-model"],
    )
    def test_speculative_launches_get_nothing(self, tail):
        assert _backend_sampling_flags(ON, self.BASE + tail, **GPU) == []

    def test_spec_type_none_is_not_speculation(self):
        argv = self.BASE + ["--spec-default", "--spec-type", "none"]
        assert _backend_sampling_flags(ON, argv, **GPU) == ["--backend-sampling"]

    def test_a_kept_spec_env_counts(self):
        kw = {**GPU, "spec_env": {"LLAMA_ARG_SPEC_TYPE": "ngram-mod"}}
        assert _backend_sampling_flags(ON, self.BASE, **kw) == []

    @pytest.mark.parametrize("flag", ["--backend-sampling", "-bs"])
    def test_the_users_own_flag_is_not_doubled(self, flag):
        assert _backend_sampling_flags(ON, self.BASE + [flag], **GPU) == []

    def test_the_env_twin_is_the_users_too(self):
        kw = {**GPU, "env": {"LLAMA_ARG_BACKEND_SAMPLING": "1"}}
        assert _backend_sampling_flags(ON, self.BASE, **kw) == []

    @pytest.mark.parametrize("value", ["0", "false", "off", "no"])
    def test_the_opt_out(self, monkeypatch, value):
        monkeypatch.setenv("UNSLOTH_LLAMA_BACKEND_SAMPLING", value)
        assert _backend_sampling_flags(ON, self.BASE, **GPU) == []


@pytest.mark.parametrize(
    "argv,env,expected",
    [
        ([], {}, False),
        (["--spec-default"], {}, True),
        (["--spec-type", "none"], {}, False),
        (["--spec-default", "--spec-type", "none"], {}, False),
        (["--spec-type", "none", "--spec-default"], {}, True),
        ([], {"LLAMA_ARG_SPEC_TYPE": "none"}, False),
        ([], {"LLAMA_ARG_MODEL_DRAFT": "d.gguf"}, True),
        (["--spec-type", "none"], {"LLAMA_ARG_SPEC_TYPE": "ngram-mod"}, False),
    ],
)
def test_launch_speculates(argv, env, expected):
    assert _launch_speculates(argv, env) is expected


# ---- the real load path ----

_MEMORY = [(0, 40 * 1024**3, 48 * 1024**3)]
_CAPS = {"found": True, "supports_kv_unified": True, "supports_backend_sampling": True}


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_BACKEND_SAMPLING", raising = False)
    monkeypatch.delenv("LLAMA_ARG_BACKEND_SAMPLING", raising = False)
    backend, _ = _backend(tmp_path, vulkan = False, memory = _MEMORY)
    gguf = _write_gguf(tmp_path / "Qwen3-0.6B-Q8_0.gguf")

    def go(
        caps = _CAPS,
        ggml_backends = frozenset({"cpu", "cuda"}),
        **kw,
    ):
        monkeypatch.setattr(
            type(backend), "probe_server_capabilities", classmethod(lambda cls, binary = None: caps)
        )
        monkeypatch.setattr(
            type(backend),
            "_installed_ggml_backends",
            staticmethod(lambda binary = None: ggml_backends),
        )
        kw.setdefault("n_parallel", 4)
        kw.setdefault("speculative_type", "off")
        return _launch(backend, gguf, **kw)["cmd"]

    return go


def test_a_multi_slot_gpu_load_launches_with_it(launch):
    cmd = launch()
    assert cmd.count("--backend-sampling") == 1


def test_a_single_slot_load_launches_with_it_too(launch):
    assert "--backend-sampling" in launch(n_parallel = 1)


def test_a_rocm_build_keeps_the_cpu_sampler(launch):
    assert "--backend-sampling" not in launch(ggml_backends = frozenset({"cpu", "hip"}))


def test_an_older_binary_launches_as_before(launch):
    assert "--backend-sampling" not in launch(caps = {**_CAPS, "supports_backend_sampling": False})


def test_auto_speculation_keeps_the_cpu_sampler(launch):
    cmd = launch(speculative_type = "auto")
    assert "--spec-default" in cmd or "--spec-type" in cmd
    assert "--backend-sampling" not in cmd


def test_a_cpu_placed_load_keeps_the_cpu_sampler(launch):
    assert "--backend-sampling" not in launch(gpu_memory_mode = "manual", gpu_layers = 0)


def test_the_users_flag_is_passed_once(launch):
    cmd = launch(extra_args = ["--backend-sampling"])
    assert cmd.count("--backend-sampling") == 1


def test_the_opt_out_launches_as_before(launch, monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_BACKEND_SAMPLING", "0")
    assert "--backend-sampling" not in launch()
