# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""llama-server's GPU sampler (--backend-sampling) on the launches where it pays off.

Without it every decode step copies each slot's full-vocabulary logits to the host and
runs the sampler chain on the CPU. Measured on a Colab L4 with 4 slots (-c 8192
--kv-unified, Studio's own sampler chains): Qwen3.5-0.8B 20.3 -> 7.0 ms per token at
4 streams, Qwen3-0.6B 7.3 -> 6.1 ms, flood throughput up to 2.2x, for ~20-25 ms more TTFT
per request. It is per request and steps aside on its own for grammar / tools, a reasoning
budget and pre-sampling logprobs; Studio sends every request outside the envelope where it
samples as the CPU chain does (min_p near 1, top_k past the vocabulary, min_keep, dynatemp,
a custom sampler order, logit_bias, a fixed seed without top_p, an active sampler with no
backend implementation: XTC, DRY, typical-p, top-n-sigma, mirostat) back to the CPU with
backend_sampling: false. Launch flags are read as llama.cpp reads them (underscore spellings,
short aliases, every --spec-type appended). A field the request omits takes the default the
running server reports in /props, which also holds a llama.cpp config.ini; a server whose /props
does not carry them sends every request to the CPU.

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
        argv = self.BASE + ["--spec-type", "none"]
        assert _backend_sampling_flags(ON, argv, **GPU) == ["--backend-sampling"]

    @pytest.mark.parametrize(
        "tail",
        [
            ["--spec_default"],
            ["--spec_type", "ngram-mod"],
            ["--model_draft", "d.gguf"],
            ["--spec-draft-model", "d.gguf"],
            ["--spec_draft_model", "d.gguf"],
            # llama.cpp appends every --spec-type: a later none does not undo the default.
            ["--spec-default", "--spec-type", "none"],
            ["--spec-type", "ngram-mod", "--spec-type", "none"],
        ],
    )
    def test_speculation_spelled_any_way_llama_cpp_reads_gets_nothing(self, tail):
        assert _backend_sampling_flags(ON, self.BASE + tail, **GPU) == []

    @pytest.mark.parametrize(
        "tail", [["--gpu_layers", "0"], ["--n_gpu_layers", "0"], ["--n-gpu-layers", "0"]]
    )
    def test_a_cpu_placed_model_spelled_another_way_gets_nothing(self, tail):
        assert _backend_sampling_flags(ON, self.BASE + tail, **GPU) == []

    def test_a_kept_spec_env_counts(self):
        kw = {**GPU, "spec_env": {"LLAMA_ARG_SPEC_TYPE": "ngram-mod"}}
        assert _backend_sampling_flags(ON, self.BASE, **kw) == []

    @pytest.mark.parametrize("flag", ["--backend-sampling", "-bs"])
    def test_the_users_own_flag_is_not_doubled(self, flag):
        assert _backend_sampling_flags(ON, self.BASE + [flag], **GPU) == []

    def test_the_env_twin_is_the_users_too(self):
        kw = {**GPU, "env": {"LLAMA_ARG_BACKEND_SAMPLING": "1"}}
        assert _backend_sampling_flags(ON, self.BASE, **kw) == []

    @pytest.mark.parametrize(
        "tail, env",
        [
            (["--min-p", "1"], {}),
            (["--min-p=1.0"], {}),
            ([], {"LLAMA_ARG_MIN_P": "1"}),
            (["--top-k", "1000000"], {}),
            ([], {"LLAMA_ARG_TOP_K": "1000000"}),
            (["--samplers", "temperature;top_k"], {}),
            (["--sampler-seq", "tk"], {}),
            (["-l", "15+5"], {}),
            (["--seed", "42", "--top-p", "1"], {}),
            (["--seed", "42"], {"LLAMA_ARG_TOP_P": "1.0"}),
            (["--dynatemp-range", "0.5"], {}),
        ],
        ids = [
            "min-p",
            "min-p-inline",
            "min-p-env",
            "top-k",
            "top-k-env",
            "samplers",
            "sampler-seq",
            "logit-bias",
            "seed-top-p-off",
            "seed-top-p-env",
            "dynatemp",
        ],
    )
    def test_a_launch_default_outside_the_envelope_gets_nothing(self, tail, env):
        # Every request that leaves the field to the launch would sample on that default.
        assert _backend_sampling_flags(ON, self.BASE + tail, **{**GPU, "env": env}) == []

    @pytest.mark.parametrize(
        "tail",
        [
            ["--sampler_seq", "tk"],
            ["--sampling_seq", "tk"],
            ["--sampling-seq", "tk"],
            ["--samplers", "temperature;top_k"],
            ["--top_k", "1000000"],
            ["--min_p", "1"],
            ["--min_p=1"],
            ["--logit_bias", "15+5"],
            ["--dynatemp_range", "0.5"],
            ["-s", "42", "--top_p", "1"],
            ["--seed", "42", "--top-p=1"],
        ],
    )
    def test_a_launch_default_spelled_another_way_gets_nothing(self, tail):
        # llama.cpp folds a long option's underscores (common/arg.cpp), and so do Studio's extras.
        assert _backend_sampling_flags(ON, self.BASE + tail, **GPU) == []

    @pytest.mark.parametrize(
        "tail",
        [
            ["--xtc-probability", "0.5"],  # llama.cpp's threshold is 0.1
            ["--xtc_probability", "1", "--xtc-threshold", "0"],
            ["--dry-multiplier", "0.8"],
            ["--dry_multiplier", "0.8"],
            ["--typical", "0.9"],
            ["--typical-p", "0.9"],
            ["--typical_p", "0.9"],
            ["--top-nsigma", "1"],
            ["--top-n-sigma", "1"],
            ["--top_n_sigma", "1"],
            ["--mirostat", "1"],
            ["--mirostat", "2"],
        ],
    )
    def test_a_launch_default_with_a_cpu_only_sampler_gets_nothing(self, tail):
        # It would run after a GPU prefix on the candidates the GPU masked.
        assert _backend_sampling_flags(ON, self.BASE + tail, **GPU) == []

    @pytest.mark.parametrize(
        "tail",
        [
            ["--xtc-probability", "0.5", "--xtc-threshold", "0.6"],
            ["--xtc-probability", "0"],
            ["--dry-multiplier", "0.8", "--dry-penalty-last-n", "0"],
            ["--dry-multiplier", "0"],
            ["--typical", "1"],
            ["--top-nsigma", "-1"],
            ["--mirostat", "2", "--mirostat", "0"],
        ],
    )
    def test_a_cpu_only_sampler_left_off_at_launch_still_gets_it(self, tail):
        assert _backend_sampling_flags(ON, self.BASE + tail, **GPU) == ["--backend-sampling"]

    @pytest.mark.parametrize(
        "tail",
        [
            ["--min-p", "0.99"],
            ["--top-k", "40"],
            ["--seed", "-1"],
            ["--seed", "42"],
            ["--sampler-seq", "edskypmxt"],
            ["--samplers", "penalties;dry;top_n_sigma;top_k;typ_p;top_p;min_p;xtc;temperature"],
            ["--repeat-penalty", "1.1"],
            ["--mirostat", "0"],
        ],
    )
    def test_a_launch_default_inside_it_still_gets_it(self, tail):
        assert _backend_sampling_flags(ON, self.BASE + tail, **GPU) == ["--backend-sampling"]

    def test_a_launch_top_k_is_held_to_the_vocabulary_when_known(self):
        argv = self.BASE + ["--top-k", "1000"]
        assert _backend_sampling_flags(ON, argv, **GPU) == []  # unknown vocabulary: 256
        assert _backend_sampling_flags(ON, argv, **GPU, n_vocab = 151936) == ["--backend-sampling"]
        assert _backend_sampling_flags(ON, argv, **GPU, n_vocab = 512) == []

    def test_an_env_layer_count_of_zero_gets_nothing(self):
        argv = ["/x/llama-server", "-m", "m.gguf", "--parallel", "4"]
        env = {"LLAMA_ARG_N_GPU_LAYERS": "0"}
        assert _backend_sampling_flags(ON, argv, **{**GPU, "env": env}) == []
        # argv wins over the env, as in llama.cpp
        assert _backend_sampling_flags(ON, argv + ["-ngl", "99"], **{**GPU, "env": env}) == [
            "--backend-sampling"
        ]

    @pytest.mark.parametrize(
        "line,expected",
        [
            ("spec-type = ngram-mod", []),
            ("spec-default = true", []),
            ("model-draft = /m/draft.gguf", []),
            ("spec-type = none", ["--backend-sampling"]),
            ("temp = 0.7", ["--backend-sampling"]),
        ],
        ids = ["spec-type", "spec-default", "draft-model", "spec-none", "unrelated"],
    )
    def test_a_config_ini_that_speculates_gets_nothing(self, tmp_path, line, expected):
        (tmp_path / "llama.cpp").mkdir()
        (tmp_path / "llama.cpp" / "config.ini").write_text(f"[default]\n{line}\n")
        env = {"XDG_CONFIG_HOME": str(tmp_path)}
        assert _backend_sampling_flags(ON, self.BASE, **{**GPU, "env": env}) == expected

    def test_a_config_ini_layer_count_of_zero_gets_nothing_unless_argv_sets_one(self, tmp_path):
        (tmp_path / "llama.cpp").mkdir()
        (tmp_path / "llama.cpp" / "config.ini").write_text("[default]\nn-gpu-layers = 0\n")
        env = {"XDG_CONFIG_HOME": str(tmp_path)}
        argv = ["/x/llama-server", "-m", "m.gguf", "--parallel", "4"]
        assert _backend_sampling_flags(ON, argv, **{**GPU, "env": env}) == []
        assert _backend_sampling_flags(ON, self.BASE, **{**GPU, "env": env}) == [
            "--backend-sampling"
        ]

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
        (["--spec-default", "--spec-type", "none"], {}, True),  # appended, not replaced
        (["--spec_default"], {}, True),
        (["--spec_type", "ngram-mod"], {}, True),
        (["--spec-type", "ngram-mod,none"], {}, False),  # none drops its own value
        (["--model_draft", "d.gguf"], {}, True),
        (["--spec-type", "none", "--spec-default"], {}, True),
        ([], {"LLAMA_ARG_SPEC_TYPE": "none"}, False),
        ([], {"LLAMA_ARG_MODEL_DRAFT": "d.gguf"}, True),
        (["--spec-type", "none"], {"LLAMA_ARG_SPEC_TYPE": "ngram-mod"}, True),  # env first
        (["--spec-type", "none"], {"LLAMA_ARG_SPEC_TYPE": "none"}, False),
    ],
)
def test_launch_speculates(argv, env, expected):
    assert _launch_speculates(argv, env) is expected


# ---- the real load path ----

_MEMORY = [(0, 40 * 1024**3, 48 * 1024**3)]
_CAPS = {"found": True, "supports_kv_unified": True, "supports_backend_sampling": True}

# llama-server's own defaults as /props reports them (default_generation_settings.params,
# tools/server/server-task.cpp task_params::to_json(true) at 18b4ec5f1, trimmed).
_PROPS_PARAMS = {
    "seed": 4294967295,
    "temperature": 0.8,
    "dynatemp_range": 0.0,
    "top_k": 40,
    "top_p": 0.95,
    "min_p": 0.05,
    "top_n_sigma": -1.0,
    "xtc_probability": 0.0,
    "xtc_threshold": 0.1,
    "typical_p": 1.0,
    "dry_multiplier": 0.0,
    "dry_penalty_last_n": -1,
    "mirostat": 0,
    "min_keep": 0,
    "samplers": [
        "penalties",
        "dry",
        "top_n_sigma",
        "top_k",
        "typ_p",
        "top_p",
        "min_p",
        "xtc",
        "temperature",
    ],
    "backend_sampling": False,
}
_SAMPLER_CHARS = dict(zip("edskypmxt", _PROPS_PARAMS["samplers"]))


def _props(backend_sampling = True, **params):
    return {
        "default_generation_settings": {
            "n_ctx": 4096,
            "params": {**_PROPS_PARAMS, "backend_sampling": backend_sampling, **params},
        }
    }


def _props_as_launched(backend):
    """What the fake server reports: llama.cpp's defaults under the argv it was spawned with."""
    launch = backend._backend_sampling_launch
    params = {}
    for field, raw in (launch or {}).items():
        if field == "logit_bias":
            continue  # not in the /props view
        if field == "samplers" and isinstance(raw, str):
            params[field] = raw.split(";") if ";" in raw else [_SAMPLER_CHARS[c] for c in raw]
        elif isinstance(raw, list):
            params[field] = raw
        else:
            params[field] = float(raw)
    return _props(launch is not None, **params)


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSLOTH_LLAMA_BACKEND_SAMPLING", raising = False)
    monkeypatch.delenv("LLAMA_ARG_BACKEND_SAMPLING", raising = False)
    backend, _ = _backend(tmp_path, vulkan = False, memory = _MEMORY)
    gguf = _write_gguf(tmp_path / "Qwen3-0.6B-Q8_0.gguf")
    monkeypatch.setattr(
        llama_cpp, "_llama_config_files", lambda env, platform = None: [], raising = False
    )
    # The running server's /props; a test swaps go.props for another server's answer.
    backend._query_server_props = lambda: go.props(backend)

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

    go.backend = backend
    go.props = _props_as_launched
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


@pytest.mark.parametrize(
    "extra",
    [
        ["--sampler_seq", "tk"],
        ["--top_k", "1000000"],
        ["--xtc-probability", "0.5"],
        ["--mirostat", "2"],
        ["--spec_type", "ngram-mod"],
    ],
)
def test_a_launch_default_outside_it_keeps_the_cpu_sampler(launch, extra):
    assert "--backend-sampling" not in launch(extra_args = extra)


def test_a_users_own_flag_records_the_underscore_order(launch):
    # Review round 3: --sampler_seq tk with the GPU sampler on, then temperature 0 and top_k 20,
    # aborted llama-server (GPU temperature leaves one candidate ahead of top-k's 20).
    cmd = launch(extra_args = ["-bs", "--sampler_seq", "tk"])
    assert cmd.count("-bs") == 1 and "--backend-sampling" not in cmd
    guarded = launch.backend.guard_gpu_sampler({"temperature": 0, "top_k": 20})
    assert guarded.get("backend_sampling") is False


# ---- the running server's own defaults, read back from /props ----
# llama.cpp also takes defaults from a config.ini (/etc/llama.cpp, ${XDG_CONFIG_HOME:-~/.config}/
# llama.cpp, %PROGRAMDATA% / %APPDATA% on Windows; common/arg.cpp) that no argv or env shows, so
# the guard fills omitted fields from what /props reports, and fails closed without it.

_TK = ["temperature", "top_k"]  # [default] sampler-seq = tk, as /props reports it


def test_a_config_ini_sampler_order_seen_through_props_goes_to_the_cpu(launch):
    # Review round 4: sampler-seq = tk in config.ini, then temperature 0 + top_k 20, aborted
    # llama-server with the GPU sampler on; the argv shows nothing.
    launch.props = lambda backend: _props(samplers = _TK)
    assert "--backend-sampling" in launch()
    assert launch.backend.guard_gpu_sampler({"temperature": 0, "top_k": 20}) == {
        "temperature": 0,
        "top_k": 20,
        "backend_sampling": False,
    }
    own_order = {"temperature": 0, "top_k": 20, "samplers": "edskypmxt"}
    assert launch.backend.guard_gpu_sampler(own_order) is own_order


@pytest.mark.parametrize(
    "server, inside, outside",
    [
        # general.sampling.* style defaults, or a config.ini key: whatever /props reports.
        ({"min_p": 1.0}, {"min_p": 0.05}, {"top_k": 20}),
        ({"top_k": 10**6}, {"top_k": 40}, {"min_p": 0.05}),
        ({"xtc_probability": 0.5, "xtc_threshold": 0.1}, {"xtc_probability": 0}, {}),
        ({"mirostat": 2}, {"mirostat": 0}, {"temperature": 0}),
        ({"seed": 42, "top_p": 1.0}, {"top_p": 0.9}, {"top_k": 20}),
        ({"samplers": ["top_k", "typ_p", "top_p", "min_p", "temperature"]}, None, {}),
    ],
)
def test_the_servers_reported_defaults_are_what_an_omitted_field_takes(
    launch, server, inside, outside
):
    launch.props = lambda backend: _props(**server)
    assert "--backend-sampling" in launch()
    launch.backend._vocab_size = _VOCAB
    if inside is None:  # the client's own order, llama.cpp's default
        inside = {"samplers": _PROPS_PARAMS["samplers"]}
    assert launch.backend.guard_gpu_sampler(inside) is inside
    assert launch.backend.guard_gpu_sampler(outside) == {**outside, "backend_sampling": False}


@pytest.mark.parametrize(
    "props",
    [
        None,  # unreadable: an error, a 401, a timeout
        {"default_generation_settings": {"n_ctx": 4096}},  # no params at all
        {"default_generation_settings": {"n_ctx": 4096, "top_k": 40}},  # an older flat shape
        _props(samplers = "tk"),  # not the list /props sends
        {
            "default_generation_settings": {
                "params": {k: v for k, v in _PROPS_PARAMS.items() if k != "min_p"}
            }
        },
    ],
    ids = ["unreadable", "no-params", "flat", "samplers-not-a-list", "missing-field"],
)
def test_unknown_defaults_send_every_request_to_the_cpu(launch, props):
    launch.props = lambda backend: props
    assert "--backend-sampling" in launch()
    launch.backend._vocab_size = _VOCAB
    for payload in ({}, {"top_k": 20, "min_p": 0.05}, {"temperature": 0}):
        assert launch.backend.guard_gpu_sampler(payload) == {**payload, "backend_sampling": False}


def test_a_config_ini_that_turns_the_gpu_sampler_on_is_guarded_too(launch, monkeypatch):
    # backend-sampling = true in config.ini: no flag in the argv, the server samples on the GPU.
    monkeypatch.setattr(llama_cpp, "_llama_config_files", lambda env, platform = None: ["c.ini"])
    launch.props = lambda backend: _props(samplers = _TK)
    cmd = launch(caps = {**_CAPS, "supports_backend_sampling": False})
    assert "--backend-sampling" not in cmd
    guarded = launch.backend.guard_gpu_sampler({"temperature": 0, "top_k": 20})
    assert guarded.get("backend_sampling") is False
    # Unreadable there too: a config file might have turned it on, so nothing is assumed.
    launch.props = lambda backend: None
    launch.backend._healthy = False
    launch(caps = {**_CAPS, "supports_backend_sampling": False})
    assert launch.backend.guard_gpu_sampler({}) == {"backend_sampling": False}


def test_a_config_ini_logit_bias_goes_to_the_cpu(launch, monkeypatch, tmp_path):
    # Review round 5: /props leaves logit_bias out, and llama.cpp appends the [*] and [default]
    # biases, which the GPU sampler scatters instead of summing.
    ini = tmp_path / "config.ini"
    ini.write_text("[*]\nlogit-bias = 15+2\n[default]\nlogit-bias = 15+3\n")
    monkeypatch.setattr(llama_cpp, "_llama_config_files", lambda env, platform = None: [str(ini)])
    assert "--backend-sampling" in launch()
    payload = {"top_k": 0, "top_p": 1, "min_p": 0, "temperature": 1}
    assert launch.backend.guard_gpu_sampler(payload) == {**payload, "backend_sampling": False}
    # A config without a bias leaves the same request on the GPU.
    ini.write_text("[default]\nctx-size = 4096\n")
    launch.backend._healthy = False
    launch()
    assert launch.backend.guard_gpu_sampler(payload) is payload


def test_an_unreadable_config_ini_counts_as_a_bias(tmp_path):
    from core.inference.llama_cpp import _config_files_set_logit_bias

    assert _config_files_set_logit_bias([str(tmp_path / "missing.ini")])
    ini = tmp_path / "c.ini"
    ini.write_text("[default]\n  logit_bias = 1+1\n")
    assert _config_files_set_logit_bias([str(ini)])
    ini.write_text("[default]\ntop-k = 20\n")
    assert not _config_files_set_logit_bias([str(ini)])


def test_a_server_reporting_the_cpu_sampler_is_left_alone(launch):
    launch.props = lambda backend: _props(backend_sampling = False, samplers = _TK)
    assert "--backend-sampling" in launch()
    payload = {"temperature": 0, "top_k": 20}
    assert launch.backend.guard_gpu_sampler(payload) is payload
    # A client's own opt-in is checked against the same reported defaults.
    assert launch.backend.guard_gpu_sampler({**payload, "backend_sampling": True}) == {
        **payload,
        "backend_sampling": False,
    }


def test_a_respawn_refreshes_the_reported_defaults(launch):
    # _respawn_if_dead marks the dead server unhealthy and reloads through load_model, the
    # path launch() drives; the next server's /props replaces the last one's.
    payload = {"temperature": 0, "top_k": 20}

    def respawn(props):
        launch.props = props
        launch.backend._healthy = False
        assert "--backend-sampling" in launch()

    respawn(lambda backend: _props(samplers = _TK))
    assert launch.backend.guard_gpu_sampler(payload).get("backend_sampling") is False
    respawn(_props_as_launched)
    assert launch.backend.guard_gpu_sampler(payload) is payload
    respawn(lambda backend: None)
    assert launch.backend.guard_gpu_sampler(payload).get("backend_sampling") is False
    respawn(_props_as_launched)
    assert launch.backend.guard_gpu_sampler(payload) is payload


def test_an_argv_logit_bias_still_counts_beside_props(launch):
    # /props leaves logit_bias out of its view, so the argv reading still supplies it.
    cmd = launch(extra_args = ["-bs", "-l", "1+1"])
    assert "-bs" in cmd
    assert launch.backend.guard_gpu_sampler({}) == {"backend_sampling": False}


def test_the_config_files_llama_cpp_reads(tmp_path):
    from core.inference.llama_cpp import _llama_config_files

    for root in ("xdg", "home/.config", "appdata", "programdata"):
        (tmp_path / root / "llama.cpp").mkdir(parents = True)
        (tmp_path / root / "llama.cpp" / "config.ini").write_text("[default]\nsampler-seq = tk\n")
    xdg = str(tmp_path / "xdg" / "llama.cpp" / "config.ini")
    home = str(tmp_path / "home" / ".config" / "llama.cpp" / "config.ini")
    found = _llama_config_files({"XDG_CONFIG_HOME": str(tmp_path / "xdg")}, "linux")
    assert xdg in found
    assert home in _llama_config_files({"HOME": str(tmp_path / "home")}, "linux")
    assert home not in _llama_config_files(
        {"HOME": str(tmp_path / "home"), "XDG_CONFIG_HOME": str(tmp_path / "none")}, "linux"
    )
    assert _llama_config_files(
        {"PROGRAMDATA": str(tmp_path / "programdata"), "APPDATA": str(tmp_path / "appdata")},
        "win32",
    ) == [
        str(tmp_path / "programdata" / "llama.cpp" / "config.ini"),
        str(tmp_path / "appdata" / "llama.cpp" / "config.ini"),
    ]


def test_the_opt_out_launches_as_before(launch, monkeypatch):
    monkeypatch.setenv("UNSLOTH_LLAMA_BACKEND_SAMPLING", "0")
    assert "--backend-sampling" not in launch()


# ---- requests outside the GPU sampler's envelope ----
# llama.cpp's GPU sampler crashes or samples another distribution than the CPU chain on these
# (src/llama-sampler.cpp at 18b4ec5f1); those requests go back to the CPU, and only on a server
# with the GPU sampler on (or a client opting in itself).

_ON = ["/fake/llama-server", "--backend-sampling"]
_VOCAB = 151936  # Qwen3

# One per rule: (payload, why) outside, each with its nearest value that stays inside.
_OUTSIDE = [
    ({"min_p": 1.0}, "min_p"),
    ({"min_p": "1"}, "min_p"),
    ({"min_p": 0.9999}, "min_p"),
    ({"top_k": 1000000}, "top_k"),
    ({"top_k": 140000}, "top_k"),  # inside Qwen3's vocabulary, past the 128K probe
    ({"min_keep": 3}, "min_keep"),
    ({"min_keep": 2}, "min_keep"),
    ({"dynatemp_range": 0.5}, "dynatemp_range"),
    ({"dynatemp_range": 0.5, "temperature": 0}, "dynatemp_range"),
    ({"samplers": ["temperature", "top_k"]}, "samplers"),
    ({"samplers": "kt"}, "samplers"),
    ({"logit_bias": {"15": 5}}, "logit_bias"),
    ({"logit_bias": [[15, 2], [15, 3]]}, "logit_bias"),
    ({"seed": 42, "top_p": 1.0}, "seed"),
    ({"seed": 0, "top_p": 1}, "seed"),
    # A sampler with no backend implementation, active: it would see GPU-masked candidates.
    (
        {"top_k": 3, "top_p": 0.8, "temperature": 1, "xtc_probability": 1, "xtc_threshold": 0},
        "xtc_probability",
    ),
    ({"xtc_probability": 0.5}, "xtc_probability"),  # llama.cpp's threshold is 0.1
    ({"xtc_probability": 0.1, "xtc_threshold": 0.5}, "xtc_probability"),
    ({"dry_multiplier": 0.8}, "dry_multiplier"),
    ({"dry_multiplier": -0.5, "dry_penalty_last_n": 16}, "dry_multiplier"),
    ({"typical_p": 0.9}, "typical_p"),
    ({"typical_p": 0}, "typical_p"),
    ({"top_n_sigma": 1.0}, "top_n_sigma"),
    ({"mirostat": 1}, "mirostat"),
    ({"mirostat": 2, "temperature": 0}, "mirostat"),
    # Unparseable: fail closed, the raw /v1/completions body is the client's own.
    ({"top_k": "lots"}, "top_k"),
    ({"min_keep": [3]}, "min_keep"),
    ({"seed": True, "top_p": 1.0}, "seed"),
    ({"seed": 42, "top_p": "off"}, "seed"),
    ({"min_p": float("nan")}, "min_p"),
    ({"xtc_probability": "nan"}, "xtc_probability"),
]
_INSIDE = [
    {},
    {"min_p": 0.999},
    {"min_p": 0},
    {"min_p": None},
    {"top_k": 128 * 1024},  # the probe's own width
    {"top_k": 0},
    {"top_k": -1},
    {"top_k": 40, "top_p": 0.95, "min_p": 0.05, "temperature": 0.6},
    {"temperature": 0},
    {"top_p": 0.0},
    {"min_keep": 1},
    {"min_keep": 0},
    {"dynatemp_range": 0},
    {
        "samplers": [
            "penalties",
            "dry",
            "top_n_sigma",
            "top_k",
            "typ_p",
            "top_p",
            "min_p",
            "xtc",
            "temperature",
        ]
    },
    {"samplers": "edskypmxt"},
    {"logit_bias": {}},
    {"logit_bias": []},
    {"repeat_penalty": 1.1, "presence_penalty": 1.5, "frequency_penalty": 0.5},
    {"repeat_penalty": 0},  # llama-server refuses it on either sampler
    {"mirostat": 0},
    {"xtc_probability": 0, "xtc_threshold": 0},
    {"xtc_probability": 1, "xtc_threshold": 0.6},  # > 0.5 turns XTC off
    {"dry_multiplier": 0},
    {"dry_multiplier": 0.8, "dry_penalty_last_n": 0},
    {"typical_p": 1.0},
    {"top_n_sigma": -1},
    {"top_n_sigma": 0},
    {"seed": -1, "top_p": 1.0},
    {"seed": 0xFFFFFFFF, "top_p": 1.0},
    {"seed": 42},  # llama.cpp's top_p 0.95 sorts the candidates before the draw
    {"seed": 42, "top_p": 0.95, "top_k": 40},
    {"n_probs": 10**9, "post_sampling_probs": True},
    {"ignore_eos": True},
]


@pytest.mark.parametrize("payload, why", _OUTSIDE)
def test_the_envelope_names_what_is_outside(payload, why):
    from core.inference.llama_cpp import _gpu_sampler_envelope_breach
    assert _gpu_sampler_envelope_breach(payload, {}, _VOCAB) == why


@pytest.mark.parametrize("payload", _INSIDE)
def test_the_envelope_leaves_the_rest_on_the_gpu(payload):
    from core.inference.llama_cpp import _cpu_sampling_outside_gpu_envelope
    assert _cpu_sampling_outside_gpu_envelope(payload, {}, _VOCAB) is payload


@pytest.mark.parametrize("payload, why", _OUTSIDE)
def test_outside_the_envelope_goes_back_to_the_cpu(payload, why):
    from core.inference.llama_cpp import _cpu_sampling_outside_gpu_envelope
    assert _cpu_sampling_outside_gpu_envelope(payload, {}, _VOCAB) == {
        **payload,
        "backend_sampling": False,
    }
    assert "backend_sampling" not in payload


def test_top_k_is_held_to_the_vocabulary_and_the_probe():
    from core.inference.llama_cpp import _gpu_sampler_envelope_breach

    assert _gpu_sampler_envelope_breach({"top_k": 32000}, {}, 32000) is None
    assert _gpu_sampler_envelope_breach({"top_k": 32001}, {}, 32000) == "top_k"
    assert _gpu_sampler_envelope_breach({"top_k": 128 * 1024 + 1}, {}, 262144) == "top_k"
    # A vocabulary Studio could not read is taken as 256.

    assert _gpu_sampler_envelope_breach({"top_k": 256}, {}, None) is None
    assert _gpu_sampler_envelope_breach({"top_k": 257}, {}, None) == "top_k"
    assert _gpu_sampler_envelope_breach({"top_k": 257}, {}, 0) == "top_k"


def test_an_omitted_field_takes_the_launch_default():
    from core.inference.llama_cpp import _cpu_sampling_outside_gpu_envelope
    for launch, inside, outside in [
        ({"min_p": "1"}, {"min_p": 0.1}, {}),
        ({"top_k": "1000000"}, {"top_k": 40}, {"top_k": None}),
        ({"seed": "42", "top_p": "1"}, {"seed": -1}, {}),
        ({"seed": "42"}, {"top_p": 0.9}, {"top_p": 1.0}),
        ({"samplers": "tk"}, {"samplers": "edskypmxt"}, {}),
        ({"xtc_probability": "1", "xtc_threshold": "0"}, {"xtc_probability": 0}, {}),
        ({"xtc_probability": "1"}, {"xtc_threshold": 0.9}, {"xtc_threshold": 0}),
        ({"dry_multiplier": "0.8"}, {"dry_multiplier": 0}, {"top_k": 20}),
        ({"typical_p": "0.9"}, {"typical_p": 1}, {}),
        ({"top_n_sigma": "1"}, {"top_n_sigma": -1}, {}),
        ({"mirostat": "2"}, {"mirostat": 0}, {"temperature": 0}),
    ]:
        assert _cpu_sampling_outside_gpu_envelope(inside, launch, _VOCAB) is inside
        assert _cpu_sampling_outside_gpu_envelope(outside, launch, _VOCAB) == {
            **outside,
            "backend_sampling": False,
        }


@pytest.mark.parametrize(
    "argv, env, expected",
    [
        (["/fake/llama-server"], {}, None),
        (["/fake/llama-server", "--spec-draft-backend-sampling"], {}, None),
        (_ON, {}, {}),
        (["/fake/llama-server", "-bs"], {}, {}),
        (["/fake/llama-server"], {"LLAMA_ARG_BACKEND_SAMPLING": "1"}, {}),
        (["/fake/llama-server"], {"LLAMA_ARG_BACKEND_SAMPLING": "0"}, None),
        ([*_ON, "--min-p", "1"], {}, {"min_p": "1"}),
        ([*_ON, "--min-p=0.5"], {"LLAMA_ARG_MIN_P": "1"}, {"min_p": "0.5"}),
        (_ON, {"LLAMA_ARG_MIN_P": "1", "LLAMA_ARG_TOP_K": "9"}, {"min_p": "1", "top_k": "9"}),
        (
            [*_ON, "--top-k", "9", "-s", "3", "-l", "1+1"],
            {},
            {"top_k": "9", "seed": "3", "logit_bias": "1+1"},
        ),
        ([*_ON, "--sampler-seq", "edskypmxt"], {}, {}),
        ([*_ON, "--samplers", "top_k;temperature"], {}, {"samplers": "top_k;temperature"}),
        # Underscore spellings llama.cpp folds, and the short aliases.
        ([*_ON, "--sampler_seq", "tk"], {}, {"samplers": "tk"}),
        (
            ["/fake/llama-server", "--backend_sampling", "--sampling_seq", "kt"],
            {},
            {"samplers": "kt"},
        ),
        ([*_ON, "--top_k", "9", "--min_p", "1"], {}, {"top_k": "9", "min_p": "1"}),
        ([*_ON, "--top_k", "9"], {"LLAMA_ARG_TOP_K": "40"}, {"top_k": "9"}),  # argv over env
        (
            [*_ON, "--logit_bias", "1+1", "--dynatemp_range", "1"],
            {},
            {"logit_bias": "1+1", "dynatemp_range": "1"},
        ),
        (
            [*_ON, "--xtc_probability", "1", "--xtc-threshold=0", "--typical", "0.9"],
            {},
            {"xtc_probability": "1", "xtc_threshold": "0", "typical_p": "0.9"},
        ),
        (
            [*_ON, "--dry_multiplier", "0.8", "--top-n-sigma", "1", "--mirostat", "2"],
            {},
            {"dry_multiplier": "0.8", "top_n_sigma": "1", "mirostat": "2"},
        ),
    ],
)
def test_launch_gpu_sampler(argv, env, expected):
    from core.inference.llama_cpp import _launch_gpu_sampler
    assert _launch_gpu_sampler(argv, env) == expected


def test_a_server_without_the_flag_never_sees_the_field():
    from core.inference.llama_cpp import _cpu_sampling_outside_gpu_envelope
    payload = {"min_p": 1.0, "top_k": 1000000}
    assert _cpu_sampling_outside_gpu_envelope(payload, None, _VOCAB) is payload


def test_a_clients_own_opt_in_is_pinned_off_too():
    from core.inference.llama_cpp import _cpu_sampling_outside_gpu_envelope

    for payload in (
        {"min_p": 1, "backend_sampling": True},
        {"top_k": 10**6, "backend_sampling": True},
    ):
        assert _cpu_sampling_outside_gpu_envelope(payload, None, _VOCAB) == {
            **payload,
            "backend_sampling": False,
        }
    inside = {"top_k": 40, "backend_sampling": True}
    assert _cpu_sampling_outside_gpu_envelope(inside, None, _VOCAB) is inside


# ---- every path, through the real load and the entry points the requests take ----
# Version-neutral: a launched backend and the public builders, so the same cells run on the
# commit before the envelope (where the new rules fail) and after it.

# (payload fields, backend_sampling expected on a launch with the GPU sampler on)
_PATH_CASES = [
    ({"min_p": 1.0}, False),
    ({"top_k": 1000000}, False),
    ({"min_keep": 3}, False),
    ({"top_k": 20, "min_p": 0.05}, None),
]
_PATH_IDS = ["min_p-1", "top_k-past-vocab", "min_keep-3", "inside"]


@pytest.fixture
def gpu_backend(launch):
    assert "--backend-sampling" in launch()
    launch.backend._vocab_size = _VOCAB
    return launch.backend


@pytest.fixture
def cpu_backend(launch):
    assert "--backend-sampling" not in launch(caps = {**_CAPS, "supports_backend_sampling": False})
    launch.backend._vocab_size = _VOCAB
    return launch.backend


def test_a_load_without_the_flag_leaves_its_requests_alone(cpu_backend):
    from routes.inference import _build_passthrough_payload

    args = ([{"role": "user", "content": "hi"}], None, 0.7, 0.9, 1000000, 16, False)
    body = _build_passthrough_payload(*args, min_p = 1.0, llama_backend = cpu_backend)
    assert "backend_sampling" not in body


@pytest.mark.parametrize("fields, expected", _PATH_CASES, ids = _PATH_IDS)
def test_every_stream_is_guarded(monkeypatch, gpu_backend, fields, expected):
    # _open_stream is the one transport of generate_chat_completion and both tool loops.
    import contextlib

    sent = []

    @contextlib.contextmanager
    def fake_stream(
        client,
        url,
        payload,
        cancel_event = None,
        **kwargs,
    ):
        sent.append(payload)
        yield type("Response", (), {"status_code": 200})()

    monkeypatch.setattr(LlamaCppBackend, "_stream_with_retry", staticmethod(fake_stream))
    monkeypatch.setattr(LlamaCppBackend, "_auth_headers", {}, raising = False)
    with gpu_backend._open_stream("http://llama.test/v1/chat/completions", dict(fields), None):
        pass
    assert sent.pop().get("backend_sampling") is expected


@pytest.mark.parametrize(
    "kwargs, expected",
    [({"min_p": 1.0}, False), ({"top_k": 1000000}, False), ({"top_k": 40}, None)],
    ids = ["min_p-1", "top_k-past-vocab", "inside"],
)
def test_gguf_tts_is_guarded(monkeypatch, gpu_backend, kwargs, expected):
    import httpx

    sent = []

    class Client:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, *, json):
            sent.append(json)
            raise httpx.ConnectError("stop here")

    monkeypatch.setattr(llama_cpp.httpx, "Client", Client)
    monkeypatch.setattr(LlamaCppBackend, "_auth_headers", {}, raising = False)
    monkeypatch.setattr(LlamaCppBackend, "base_url", "http://llama.test", raising = False)
    with pytest.raises(httpx.ConnectError):
        gpu_backend.generate_audio_response("hi", "snac", **kwargs)
    assert sent.pop().get("backend_sampling", None) is expected


@pytest.mark.parametrize(
    "top_k, min_p, expected",
    [(20, 1.0, False), (1000000, 0.05, False), (20, 0.05, None)],
    ids = ["min_p-1", "top_k-past-vocab", "inside"],
)
def test_every_chat_passthrough_body_is_guarded(gpu_backend, top_k, min_p, expected):
    # /v1/messages (both) build through _build_passthrough_payload; OpenAI /v1/chat/completions
    # and the streaming /v1/responses passthrough through _build_openai_passthrough_body on it.
    from models.inference import ChatCompletionRequest, ChatMessage
    from routes.inference import _build_openai_passthrough_body, _build_passthrough_payload

    gpu_backend._request_reasoning_kwargs = lambda *a: None
    args = ([{"role": "user", "content": "hi"}], None, 0.7, 0.9, top_k, 16, False)
    body = _build_passthrough_payload(*args, min_p = min_p, llama_backend = gpu_backend)
    assert body.get("backend_sampling") is expected
    if top_k <= 100:  # the OpenAI schema caps top_k at 100
        request = ChatCompletionRequest(
            model = "default",
            messages = [ChatMessage(role = "user", content = "hi")],
            min_p = min_p,
            top_k = top_k,
        )
        body = _build_openai_passthrough_body(request, llama_backend = gpu_backend)
        assert body.get("backend_sampling") is expected


def test_every_passthrough_build_hands_over_the_backend():
    # Both /v1/messages passthroughs and the OpenAI body: a call without the backend is unguarded.
    import ast
    import inspect

    import routes.inference as inf_mod

    calls = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(inf_mod)))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "_build_passthrough_payload"
    ]
    assert len(calls) == 3
    for call in calls:
        assert "llama_backend" in {kw.arg for kw in call.keywords}


@pytest.mark.parametrize(
    "gpu, raw, expected",
    [
        (True, {"min_p": 1}, False),
        (True, {"top_k": 1000000}, False),
        (True, {"min_keep": 3, "top_k": 3, "top_p": 0.1, "min_p": 0}, False),
        (True, {"samplers": ["temperature", "top_k"], "temperature": 0}, False),
        (True, {"logit_bias": [[15, 2], [15, 3]]}, False),
        (True, {"dynatemp_range": 0.5}, False),
        (True, {"seed": 42, "top_p": 1.0}, False),
        (
            True,
            {"top_k": 3, "top_p": 0.8, "temperature": 1, "xtc_probability": 1, "xtc_threshold": 0},
            False,
        ),
        (True, {"dry_multiplier": 0.8}, False),
        (True, {"mirostat": 2}, False),
        (True, {"seed": 42, "top_p": 0.9}, None),
        (True, {"min_p": 0.99, "top_k": 40}, None),
        (True, {}, None),
        (False, {"min_p": 1, "top_k": 1000000}, None),
        (False, {"top_k": 1000000, "backend_sampling": True}, False),
        (False, {"top_k": 40, "backend_sampling": True}, True),
    ],
)
def test_the_raw_completions_proxy_is_guarded(request, monkeypatch, gpu, raw, expected):
    import asyncio
    from types import SimpleNamespace

    import httpx

    import routes.inference as inf_mod
    from core.inference.api_monitor import ApiMonitor

    sent = []

    class Request:
        state = SimpleNamespace()
        url = SimpleNamespace(path = "/v1/completions")
        method = "POST"
        headers = {}

        async def json(self):
            return {"prompt": "hi", "stream": False, **raw}

        async def is_disconnected(self):
            return False

    class Client:
        async def aclose(self):
            return None

        async def post(self, _url, *, json, **_kwargs):
            sent.append(dict(json))
            return httpx.Response(200, json = {"id": "cmpl-test", "choices": [{"text": "ok"}]})

    backend = request.getfixturevalue("gpu_backend" if gpu else "cpu_backend")
    monkeypatch.setattr(type(backend), "is_loaded", property(lambda self: True), raising = False)
    monkeypatch.setattr(type(backend), "base_url", "http://llama.test", raising = False)
    monkeypatch.setattr(inf_mod, "api_monitor", ApiMonitor(max_entries = 3))
    monkeypatch.setattr(inf_mod, "_cancelable_nonstreaming_client", lambda: Client())
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
    asyncio.run(inf_mod.openai_completions(Request(), current_subject = "test"))
    assert sent[0].get("backend_sampling", None) is expected
