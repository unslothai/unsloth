# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MTP drafting at T > 0: Studio passes --spec-draft-sampling probabilistic (the draft is sampled and the target
verifies it by rejection sampling, so the output distribution is unchanged) when the build lists the flag, and
never overrides a choice the user made in the extras or the LLAMA_ARG_SPEC_DRAFT_SAMPLING env."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_llama_cpp_mtp_detection import (  # noqa: F401 (stubs are installed on import)
    LlamaCppBackend,
    _NEEDS_BASH,
    _clear_caps_cache,
    _flags_dict,
    _make_fake_llama_server,
)

import core.inference.llama_cpp as _llama_cpp_mod
from core.inference.llama_cpp import (
    _extra_args_select_stateful_sampler,
    _llama_config_paths,
    _spec_draft_sampling_flags,
)

_MTP_MODEL = "unsloth/Qwen3.6-27B-MTP-GGUF"
_NON_MTP_MODEL = "unsloth/Qwen3-7B-Instruct-GGUF"
_PROB = ["--spec-draft-sampling", "probabilistic"]

_HELP_WITH = """\
--spec-draft-n-max N                    number of tokens to draft for speculative decoding (default: 3)
--spec-draft-sampling {greedy,probabilistic}
                                        how the draft is sampled: greedy takes its argmax, probabilistic samples it and has the target
                                        verify by rejection sampling (default: greedy)
                                        (env: LLAMA_ARG_SPEC_DRAFT_SAMPLING)
--spec-type none,draft-simple,draft-mtp,ngram-mod                                        comma-separated list of types of speculative decoding to use
"""

_HELP_WITHOUT = """\
--spec-draft-n-max N                    number of tokens to draft for speculative decoding (default: 3)
--spec-type none,draft-simple,draft-mtp,ngram-mod                                        comma-separated list of types of speculative decoding to use
"""


@pytest.fixture(autouse = True)
def _no_inherited_choice(monkeypatch, tmp_path):
    monkeypatch.delenv("LLAMA_ARG_SPEC_DRAFT_SAMPLING", raising = False)
    monkeypatch.delenv("UNSLOTH_SPEC_DRAFT_SAMPLING", raising = False)
    # No real /etc or ~/.config llama.cpp config.ini leaks into the tests.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        _llama_cpp_mod, "_LLAMA_SYSTEM_CONFIG_PATH", str(tmp_path / "etc" / "config.ini")
    )


def _backend(monkeypatch, *, sampling = True):
    caps = {
        "found": True,
        "mtp_token": "draft-mtp",
        "supports_mtp": True,
        "supports_dspark": True,
        "mtp_probe_inconclusive": False,
        "ngram_mod_flavor": "new",
        "supports_ngram_mod": True,
        "spec_draft_n_max_flag": "--spec-draft-n-max",
        "supports_spec_draft_sampling": sampling,
    }
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, binary = None: caps)
    )
    backend = LlamaCppBackend()
    backend._nextn_predict_layers = None
    # A complete header read of a model with no general.sampling.* defaults.
    backend._gguf_header_parsed = True
    backend._gguf_sampling = {}
    return backend


def _flags(
    backend,
    mode,
    model = _MTP_MODEL,
    extra_args = None,
    gpus = True,
    mtp_draft_path = None,
):
    return backend._build_speculative_flags(
        speculative_type = mode,
        spec_draft_n_max = None,
        extra_args = extra_args,
        model_identifier = model,
        model_path = None,
        gpus = gpus,
        binary = "/fake/llama-server",
        mtp_draft_path = mtp_draft_path,
    )


@_NEEDS_BASH
def test_probe_detects_spec_draft_sampling(tmp_path):
    fake = _make_fake_llama_server(tmp_path / "llama-server", _HELP_WITH)
    _clear_caps_cache()
    assert (
        LlamaCppBackend.probe_server_capabilities(str(fake))["supports_spec_draft_sampling"] is True
    )


@_NEEDS_BASH
def test_probe_without_the_flag_fails_closed(tmp_path):
    fake = _make_fake_llama_server(tmp_path / "llama-server", _HELP_WITHOUT)
    _clear_caps_cache()
    assert (
        LlamaCppBackend.probe_server_capabilities(str(fake))["supports_spec_draft_sampling"]
        is False
    )


def test_missing_binary_reports_no_support(tmp_path):
    _clear_caps_cache()
    caps = LlamaCppBackend.probe_server_capabilities(str(tmp_path / "no-such-llama-server"))
    assert caps["supports_spec_draft_sampling"] is False


@pytest.mark.parametrize(
    "mode, gpus", [("auto", True), ("auto", False), ("mtp", True), ("mtp+ngram", True)]
)
def test_mtp_emits_probabilistic_after_spec_type(monkeypatch, mode, gpus):
    flags = _flags(_backend(monkeypatch), mode, gpus = gpus)
    assert flags[-2:] == _PROB
    assert flags.index("--spec-type") < flags.index("--spec-draft-sampling")
    assert flags.count("--spec-draft-sampling") == 1


def test_gemma_assistant_keeps_greedy(monkeypatch, tmp_path):
    drafter = tmp_path / "mtp-gemma-4-26B-A4B-it-Q8_0.gguf"
    drafter.write_bytes(b"GGUF")
    flags = _flags(
        _backend(monkeypatch),
        "mtp",
        model = "unsloth/gemma-4-26B-A4B-it-GGUF",
        mtp_draft_path = str(drafter),
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--model-draft") == str(drafter)
    assert "--spec-draft-sampling" not in flags


def test_separate_qwen_drafter_emits_it(monkeypatch, tmp_path):
    drafter = tmp_path / "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
    drafter.write_bytes(b"GGUF")
    flags = _flags(
        _backend(monkeypatch),
        "mtp",
        model = "unsloth/Qwen3.8-Flash-Next-GGUF",
        mtp_draft_path = str(drafter),
    )
    parsed = _flags_dict(flags)
    assert parsed.get("--model-draft") == str(drafter)
    assert parsed.get("--spec-draft-sampling") == "probabilistic"


def test_build_without_the_flag_emits_nothing(monkeypatch):
    flags = _flags(_backend(monkeypatch, sampling = False), "mtp")
    assert "--spec-type" in flags
    assert "--spec-draft-sampling" not in flags


@pytest.mark.parametrize(
    "mode, model", [("ngram", _MTP_MODEL), ("off", _MTP_MODEL), ("auto", _NON_MTP_MODEL)]
)
def test_no_mtp_drafter_no_flag(monkeypatch, mode, model):
    assert "--spec-draft-sampling" not in _flags(_backend(monkeypatch), mode, model = model)


def test_user_extras_choice_wins(monkeypatch):
    assert "--spec-draft-sampling" not in _flags(
        _backend(monkeypatch), "mtp", extra_args = ["--spec-draft-sampling", "greedy"]
    )
    assert "--spec-draft-sampling" not in _flags(
        _backend(monkeypatch), "mtp", extra_args = ["--spec-draft-sampling=greedy"]
    )


def test_user_spec_type_extras_emit_nothing(monkeypatch):
    assert _flags(_backend(monkeypatch), "mtp", extra_args = ["--spec-type", "draft-mtp"]) == []


def test_inherited_env_choice_wins(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_SAMPLING", "greedy")
    assert "--spec-draft-sampling" not in _flags(_backend(monkeypatch), "mtp")


def test_operator_opt_out(monkeypatch):
    monkeypatch.setenv("UNSLOTH_SPEC_DRAFT_SAMPLING", "greedy")
    assert "--spec-draft-sampling" not in _flags(_backend(monkeypatch), "mtp")
    monkeypatch.setenv("UNSLOTH_SPEC_DRAFT_SAMPLING", "nonsense")
    assert (
        _flags_dict(_flags(_backend(monkeypatch), "mtp")).get("--spec-draft-sampling")
        == "probabilistic"
    )


def test_helper_contract():
    caps = {"supports_spec_draft_sampling": True}
    assert _spec_draft_sampling_flags(caps, None, {}) == _PROB
    assert _spec_draft_sampling_flags(None, None, {}) == []
    assert _spec_draft_sampling_flags({}, None, {}) == []
    assert _spec_draft_sampling_flags(caps, ["--spec-draft-sampling", "probabilistic"], {}) == []
    # GGUF header not read: its sampler defaults are unknown, so nothing is emitted.
    assert _spec_draft_sampling_flags(caps, None) == []


def test_renamed_gemma_target_keeps_greedy(monkeypatch, tmp_path):
    # general.architecture, not the file name: a renamed Gemma 4 target still drafts greedily.
    drafter = tmp_path / "mtp-custom-26B-Q8_0.gguf"
    drafter.write_bytes(b"GGUF")
    backend = _backend(monkeypatch)
    backend._architecture = "gemma4"
    flags = _flags(backend, "mtp", model = "local-custom-model", mtp_draft_path = str(drafter))
    assert _flags_dict(flags).get("--model-draft") == str(drafter)
    assert "--spec-draft-sampling" not in flags


def test_renamed_gemma_assistant_keeps_greedy(monkeypatch, tmp_path):
    drafter = tmp_path / "mtp-custom-26B-Q8_0.gguf"
    drafter.write_bytes(b"GGUF")
    backend = _backend(monkeypatch)

    class _Draft:
        _architecture = "gemma4-assistant"

    monkeypatch.setattr(backend, "_draft_backend_for", lambda path: _Draft())
    flags = _flags(backend, "mtp", model = "local-custom-model", mtp_draft_path = str(drafter))
    assert _flags_dict(flags).get("--model-draft") == str(drafter)
    assert "--spec-draft-sampling" not in flags


@pytest.mark.parametrize(
    "extra",
    [
        ["--mirostat", "2"],
        ["--mirostat=1"],
        ["--samplers", "penalties;top_k;adaptive_p"],
        ["--sampler-seq", "kpa"],
        ["--sampling-seq=ka"],
    ],
)
def test_stateful_sampler_keeps_greedy(monkeypatch, extra):
    # Mirostat updates mu inside apply on the target's own draw, and adaptive-p only updates its
    # EMA when the accepted token is that draw, so rejection sampling (which may emit another
    # token) would corrupt their state.
    flags = _flags(_backend(monkeypatch), "mtp", extra_args = extra)
    assert "--spec-type" in flags
    assert "--spec-draft-sampling" not in flags


@pytest.mark.parametrize(
    "extra",
    [["--mirostat", "0"], ["--samplers", "top_k;top_p;temperature"], ["--sampler-seq", "kpt"]],
)
def test_stateless_sampler_extras_keep_probabilistic(monkeypatch, extra):
    flags = _flags(_backend(monkeypatch), "mtp", extra_args = extra)
    assert _flags_dict(flags).get("--spec-draft-sampling") == "probabilistic"


# llama.cpp (common/sampling.cpp common_sampler_types_from_names) lowercases each ';' name and
# accepts kebab-case and underscore-free aliases; --sampler-seq chars are case-sensitive.
@pytest.mark.parametrize(
    "extra",
    [
        ["--samplers", "penalties;top-k;adaptive-p"],
        ["--samplers", "top_k;adaptivep"],
        ["--samplers", "top_k;Adaptive_P"],
        ["--sampler_seq", "ka"],
        ["--mirostat", " 2"],
        ["--mirostat", "2x"],
        ["--mirostat"],
        ["--samplers"],
        ["--mirostat", "0", "--mirostat", "1"],
        ["--samplers", "top_k", "--sampler-seq", "ka"],
        ["--sampler-seq", "kpt", "--samplers", "adaptive_p"],
    ],
)
def test_stateful_sampler_detection_matches_llama_cpp(extra):
    assert _extra_args_select_stateful_sampler(extra) is True


# Each occurrence overwrites the same field in common/arg.cpp, so the last one wins; names that
# llama.cpp cannot match (surrounding whitespace, upper-case sequence chars) are dropped.
@pytest.mark.parametrize(
    "extra",
    [
        ["--mirostat", "2", "--mirostat", "0"],
        ["--mirostat=1", "--mirostat", "0"],
        ["--samplers", "adaptive_p", "--samplers", "top_k;top_p;temperature"],
        ["--sampler-seq", "a", "--sampler-seq", "kpt"],
        ["--samplers", "adaptive_p", "--sampling-seq", "kypxt"],
        ["--sampler-seq", "ka", "--samplers", "top_k;min_p"],
        ["--samplers", "top_k; adaptive_p"],
        ["--sampler-seq", "kA"],
        ["--mirostat", "-0"],
        ["--mirostat", "0x"],
    ],
)
def test_stateless_after_llama_cpp_parsing(extra):
    assert _extra_args_select_stateful_sampler(extra) is False


def test_overridden_stateful_extras_keep_probabilistic(monkeypatch):
    flags = _flags(
        _backend(monkeypatch),
        "mtp",
        extra_args = [
            "--mirostat",
            "2",
            "--samplers",
            "adaptive_p",
            "--mirostat",
            "0",
            "--samplers",
            "top_k",
        ],
    )
    assert _flags_dict(flags).get("--spec-draft-sampling") == "probabilistic"


# llama.cpp (common/arg.cpp common_params_apply_system_config) applies /etc/llama.cpp/config.ini,
# then ${XDG_CONFIG_HOME:-~/.config}/llama.cpp/config.ini, before the env and the CLI.
def _write_config(
    tmp_path,
    text,
    where = "user",
):
    if where == "user":
        path = tmp_path / "xdg" / "llama.cpp" / "config.ini"
    elif where == "home":
        path = tmp_path / "home" / ".config" / "llama.cpp" / "config.ini"
    else:
        path = tmp_path / "etc" / "config.ini"
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text)
    return path


def _emits(monkeypatch, extra_args = None):
    flags = _flags(_backend(monkeypatch), "mtp", extra_args = extra_args)
    assert "--spec-type" in flags
    return _flags_dict(flags).get("--spec-draft-sampling") == "probabilistic"


def test_config_mirostat_and_greedy_choice_win(monkeypatch, tmp_path):
    _write_config(tmp_path, "[*]\nmirostat = 2\nspec-draft-sampling = greedy\n")
    assert not _emits(monkeypatch)


@pytest.mark.parametrize(
    "text",
    [
        "[*]\nspec-draft-sampling = probabilistic\n",
        "[*]\nspec-draft-sampling = greedy\n",
        "spec-draft-sampling = greedy\n",
        "[default]\nLLAMA_ARG_SPEC_DRAFT_SAMPLING = greedy\n",
    ],
)
def test_config_draft_sampling_choice_wins(monkeypatch, tmp_path, text):
    _write_config(tmp_path, text)
    assert not _emits(monkeypatch)
    # Even extras that would otherwise allow it: the config already chose.
    assert not _emits(monkeypatch, ["--mirostat", "0"])


@pytest.mark.parametrize(
    "text",
    [
        "[*]\nmirostat = 2\n",
        "mirostat=1 ; comment\n",
        "[default]\nsamplers = adaptive_p\n",
        "[*]\nsampler-seq = ka\n",
        "[*]\nsampling-seq = kpa\n",
        "[*]\nmirostat = 2x\n",
        "[*]\nmirostat = 0\n[default]\nmirostat = 2\n",
    ],
)
def test_config_stateful_sampler_keeps_greedy(monkeypatch, tmp_path, text):
    _write_config(tmp_path, text)
    assert not _emits(monkeypatch)


def test_cli_extras_override_config_samplers(monkeypatch, tmp_path):
    _write_config(tmp_path, "[*]\nmirostat = 2\nsamplers = adaptive_p\n")
    assert not _emits(monkeypatch, ["--mirostat", "0"])
    assert _emits(monkeypatch, ["--mirostat", "0", "--samplers", "top_k;top_p"])
    _write_config(tmp_path, "[*]\nsampler-seq = ka\n")
    assert _emits(monkeypatch, ["--sampler-seq", "kpt"])


@pytest.mark.parametrize(
    "text",
    [
        "[*]\nmirostat = 0\nctx-size = 8192\n",
        "[*]\nmirostat = 2\n[default]\nmirostat = 0\n",
        "[*]\nmirostat = 0 ; was 2\n",
        "[*]\nmirostat = 2\n[*]\nctx-size = 4096\n",
        "[unsloth/Qwen3.6-27B-MTP-GGUF:Q4_K_M]\nmirostat = 2\nspec-draft-sampling = greedy\n",
        "[*]\nsamplers = top_k;top_p;temperature\n",
        # ';' starts an INI comment, so llama.cpp reads only "top_k" here.
        "[*]\nsamplers = top_k;adaptive_p\n",
        "[*]\nSPEC-DRAFT-SAMPLING = greedy\nMirostat = 2\n",
    ],
)
def test_stateless_config_keeps_probabilistic(monkeypatch, tmp_path, text):
    _write_config(tmp_path, text)
    assert _emits(monkeypatch)


def test_user_config_overrides_system_config(monkeypatch, tmp_path):
    _write_config(tmp_path, "[*]\nmirostat = 2\n", where = "etc")
    assert not _emits(monkeypatch)
    _write_config(tmp_path, "[*]\nmirostat = 0\n")
    assert _emits(monkeypatch)
    _write_config(tmp_path, "[*]\nspec-draft-sampling = greedy\n", where = "etc")
    assert not _emits(monkeypatch)


def test_home_config_used_without_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising = False)
    _write_config(tmp_path, "[*]\nmirostat = 2\n", where = "home")
    assert not _emits(monkeypatch)
    # A set XDG_CONFIG_HOME replaces ~/.config.
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    assert _emits(monkeypatch)


@pytest.mark.parametrize(
    "text",
    [
        "[*]\n  mirostat = 2\n",
        "[*]\nmirostat = 2\n???\n",
        "[ ]\nmirostat = 2\n",
    ],
)
def test_unparsable_config_is_ignored(monkeypatch, tmp_path, text):
    # llama-server refuses to start on these, so they decide nothing here.
    _write_config(tmp_path, text)
    assert _emits(monkeypatch)


def test_unreadable_config_never_breaks_the_load(monkeypatch, tmp_path):
    (tmp_path / "xdg" / "llama.cpp" / "config.ini").mkdir(parents = True)
    assert _emits(monkeypatch)
    _write_config(tmp_path, "[*]\nmirostat = 2\n", where = "etc")

    def _boom(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr("builtins.open", _boom)
    assert _spec_draft_sampling_flags({"supports_spec_draft_sampling": True}, None, {}) == _PROB


def test_config_paths_per_platform(tmp_path):
    assert _llama_config_paths({"PROGRAMDATA": "P", "APPDATA": "A"}, "win32") == [
        str(Path("P") / "llama.cpp" / "config.ini"),
        str(Path("A") / "llama.cpp" / "config.ini"),
    ]
    assert _llama_config_paths({}, "win32") == []
    etc = _llama_cpp_mod._LLAMA_SYSTEM_CONFIG_PATH
    for plat in ("linux", "darwin"):
        assert _llama_config_paths({"XDG_CONFIG_HOME": "/x", "HOME": "/h"}, plat) == [
            etc,
            "/x/llama.cpp/config.ini",
        ]
        assert _llama_config_paths({"XDG_CONFIG_HOME": "", "HOME": "/h"}, plat) == [
            etc,
            "/h/.config/llama.cpp/config.ini",
        ]


def test_config_over_size_cap_counts_as_unknown(monkeypatch, tmp_path):
    # llama.cpp reads the whole file, so a setting past any prefix still applies.
    _write_config(tmp_path, "#" + "x" * (1 << 20) + "\n[*]\nspec-draft-sampling = greedy\n")
    assert not _emits(monkeypatch)
    _write_config(tmp_path, "#" + "x" * (1 << 20) + "\n[*]\nmirostat = 2\n")
    assert not _emits(monkeypatch)
    _write_config(tmp_path, "#" + "x" * (1 << 20) + "\n[*]\nmirostat = 0\n")
    assert _emits(monkeypatch)
    monkeypatch.setattr(_llama_cpp_mod, "_LLAMA_CONFIG_MAX_BYTES", 64)
    assert not _emits(monkeypatch)


# llama.cpp common_init_sampler_from_model (common/common.cpp) applies general.sampling.mirostat
# and general.sampling.sequence after the CLI and config.ini, each only when --mirostat /
# --samplers (or their config keys) did not set the field; --sampler-seq does not count.
def _emits_with_gguf(
    monkeypatch,
    gguf,
    extra_args = None,
):
    backend = _backend(monkeypatch)
    backend._gguf_sampling = dict(gguf)
    flags = _flags(backend, "mtp", extra_args = extra_args)
    assert "--spec-type" in flags
    return _flags_dict(flags).get("--spec-draft-sampling") == "probabilistic"


@pytest.mark.parametrize(
    "gguf",
    [
        {"general.sampling.mirostat": "2"},
        {"general.sampling.mirostat": "1"},
        {"general.sampling.mirostat": "2.000000"},
        {"general.sampling.mirostat": " -1x"},
        {"general.sampling.sequence": "top_k;adaptive_p"},
        {"general.sampling.sequence": "penalties;Adaptive-P;temperature"},
        {"general.sampling.mirostat": None},
        {"general.sampling.sequence": None},
    ],
)
def test_gguf_stateful_sampler_defaults_keep_greedy(monkeypatch, gguf):
    assert not _emits_with_gguf(monkeypatch, gguf)


@pytest.mark.parametrize(
    "gguf",
    [
        {"general.sampling.mirostat": "0"},
        {"general.sampling.mirostat": "true"},
        {"general.sampling.mirostat": "0.900000"},
        # strtol is 64-bit outside Windows, so this narrows to int32 0; Windows' 32-bit long
        # saturates it to a nonzero 2147483647 (the stateful case below).
        pytest.param(
            {"general.sampling.mirostat": "4294967296"},
            marks = pytest.mark.skipif(sys.platform == "win32", reason = "32-bit long"),
        ),
        {"general.sampling.sequence": "top_k;top_p;temperature"},
        {"general.sampling.sequence": ""},
        {"general.sampling.sequence": "top_k;" + "x" * 600 + ";adaptive_p"},
        {"general.sampling.top_k": "20"},
    ],
)
def test_gguf_stateless_sampler_defaults_keep_probabilistic(monkeypatch, gguf):
    assert _emits_with_gguf(monkeypatch, gguf)


def test_gguf_defaults_yield_to_user_set_fields(monkeypatch, tmp_path):
    mirostat = {"general.sampling.mirostat": "2"}
    adaptive = {"general.sampling.sequence": "top_k;adaptive_p"}
    assert _emits_with_gguf(monkeypatch, mirostat, ["--mirostat", "0"])
    assert _emits_with_gguf(monkeypatch, adaptive, ["--samplers", "top_k;top_p"])
    # --sampler-seq leaves the field unset, so the GGUF sequence still replaces it.
    assert not _emits_with_gguf(monkeypatch, adaptive, ["--sampler-seq", "kpt"])
    assert _emits_with_gguf(
        monkeypatch, {"general.sampling.sequence": "top_k;temperature"}, ["--sampler-seq", "ka"]
    )
    _write_config(tmp_path, "[*]\nmirostat = 0\nsamplers = top_k\n")
    assert _emits_with_gguf(monkeypatch, {**mirostat, **adaptive})
    _write_config(tmp_path, "[*]\nsampler-seq = kpt\n")
    assert not _emits_with_gguf(monkeypatch, adaptive)


def test_unread_gguf_header_emits_nothing(monkeypatch):
    backend = _backend(monkeypatch)
    backend._gguf_header_parsed = False
    assert "--spec-draft-sampling" not in _flags(backend, "mtp")
    backend = _backend(monkeypatch)
    backend._gguf_path = "/models/a.gguf"
    flags = backend._build_speculative_flags(
        speculative_type = "mtp",
        spec_draft_n_max = None,
        extra_args = None,
        model_identifier = _MTP_MODEL,
        model_path = "/models/b.gguf",
        gpus = True,
        binary = "/fake/llama-server",
    )
    assert "--spec-type" in flags
    assert "--spec-draft-sampling" not in flags


def _gguf_bytes(kvs):
    import struct

    def _s(text):
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    body = b""
    for key, vtype, payload in kvs:
        body += _s(key) + struct.pack("<I", vtype) + payload
    return b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs)) + body


def test_header_reader_keeps_llama_cpp_text(tmp_path):
    import struct

    path = tmp_path / "m.gguf"
    path.write_bytes(
        _gguf_bytes(
            [
                ("general.architecture", 8, struct.pack("<Q", 5) + b"qwen3"),
                ("general.sampling.mirostat", 5, struct.pack("<i", 2)),
                ("general.sampling.sequence", 8, struct.pack("<Q", 15) + b"top_k;adaptive_p"[:15]),
                ("general.sampling.top_k", 4, struct.pack("<I", 20)),
            ]
        )
    )
    backend = LlamaCppBackend()
    backend._read_gguf_metadata(str(path))
    assert backend._gguf_header_parsed is True
    assert backend._gguf_sampling == {
        "general.sampling.mirostat": "2",
        "general.sampling.sequence": "top_k;adaptive_",
    }
    path.write_bytes(
        _gguf_bytes(
            [
                ("general.sampling.mirostat", 6, struct.pack("<f", 2.0)),
                ("general.sampling.sequence", 9, struct.pack("<IQ", 4, 1) + struct.pack("<I", 1)),
                ("general.architecture", 8, struct.pack("<Q", 5) + b"qwen3"),
            ]
        )
    )
    backend._read_gguf_metadata(str(path))
    assert backend._gguf_header_parsed is True
    assert backend._gguf_sampling == {
        "general.sampling.mirostat": "2.000000",
        "general.sampling.sequence": None,
    }
    # Truncated inside the sampler value: the walk is incomplete, so the defaults are unknown.
    path.write_bytes(
        _gguf_bytes(
            [
                ("general.architecture", 8, struct.pack("<Q", 5) + b"qwen3"),
                ("general.sampling.sequence", 8, struct.pack("<Q", 16) + b"top_k;adaptive_p"),
            ]
        )[:-3]
    )
    backend._read_gguf_metadata(str(path))
    assert backend._gguf_header_parsed is False


def test_header_reader_stops_strings_at_the_first_nul(tmp_path):
    # gguf_kv_to_str returns gguf_get_val_str as a C string, so llama.cpp stops at the first NUL.
    import struct

    path = tmp_path / "m.gguf"
    seq = b"top_k;adaptive_p\0;temperature"
    path.write_bytes(
        _gguf_bytes(
            [
                ("general.architecture", 8, struct.pack("<Q", 5) + b"qwen3"),
                ("general.sampling.sequence", 8, struct.pack("<Q", len(seq)) + seq),
            ]
        )
    )
    backend = LlamaCppBackend()
    backend._read_gguf_metadata(str(path))
    assert backend._gguf_header_parsed is True
    assert backend._gguf_sampling == {"general.sampling.sequence": "top_k;adaptive_p"}


# A request can switch a stateful sampler on per slot, after the launch-side check above.
@pytest.mark.parametrize(
    "body, stateful",
    [
        ({"mirostat": 2}, True),
        ({"mirostat": 1.0, "temperature": 0.7}, True),
        ({"mirostat": 0}, False),
        ({"mirostat": True}, True),
        ({"mirostat": False}, False),
        ({"mirostat": 10**1000}, True),
        ({"mirostat": float("inf")}, True),
        ({"mirostat": -1}, False),
        ({"mirostat": 0.9}, False),
        ({"mirostat": 3}, True),
        ({"mirostat": 2**32 + 1}, True),
        ({"mirostat": 2**32}, False),
        ({"mirostat": float("nan")}, True),
        ({"samplers": ["top_k", "Adaptive-P"]}, True),
        ({"samplers": ["top_k", "adaptivep"]}, True),
        ({"samplers": "kpa"}, True),
        ({"samplers": "kpA"}, False),
        ({"samplers": "kpa", "adaptive_target": -1}, False),
        ({"samplers": ["adaptive_p"], "adaptive_target": -0.5}, False),
        ({"samplers": ["adaptive_p"], "adaptive_target": -(10**400)}, False),
        ({"samplers": ["adaptive_p"], "adaptive_target": -1e-50}, True),
        ({"samplers": ["adaptive_p"], "adaptive_target": 0.4}, True),
        ({"samplers": ["adaptive_p"], "adaptive_target": float("nan")}, True),
        ({"mirostat": 2, "samplers": "a", "adaptive_target": -1}, True),
        ({"samplers": ["top_k", "temperature"]}, False),
        ({"mirostat": 2, "temperature": 0}, False),
        ({"mirostat": 2, "temperature": False}, False),
        ({"mirostat": 2, "temperature": True}, True),
        ({"samplers": "a", "temperature": 0.0}, False),
        ({"temperature": 0.8}, False),
    ],
)
def test_request_stateful_sampler_detection(body, stateful):
    from core.inference.llama_cpp import _request_selects_stateful_sampler
    assert _request_selects_stateful_sampler(body) is stateful


@pytest.mark.parametrize("probabilistic", [True, False])
def test_load_records_whether_drafts_are_sampled(monkeypatch, probabilistic):
    backend = _backend(monkeypatch, sampling = probabilistic)
    _flags(backend, "mtp")
    assert backend._spec_draft_sampling_emitted is probabilistic
    _flags(backend, "off")
    assert backend._spec_draft_sampling_emitted is False


class _Reached(Exception):
    pass


def _run_completions(monkeypatch, body, *, probabilistic):
    import asyncio
    from types import SimpleNamespace

    import routes.inference as inference_route
    import utils.hardware as hardware

    backend = SimpleNamespace(
        is_loaded = True,
        context_length = 4096,
        base_url = "http://127.0.0.1:1",
        model_identifier = _MTP_MODEL,
        _spec_draft_probabilistic = probabilistic,
    )

    async def _same_body(request, *_a, **_kw):
        return await request.json()

    def _reached(*_a, **_kw):
        raise _Reached

    monkeypatch.setattr(hardware, "get_device", lambda: hardware.DeviceType.CUDA)
    monkeypatch.setattr(inference_route, "_should_validate_before_switch", lambda: False)
    monkeypatch.setattr(inference_route, "_auto_switch_from_request_body", _same_body)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    monkeypatch.setattr(inference_route, "_fill_recommended_sampling_completions", lambda *_a: None)
    monkeypatch.setattr(inference_route, "_request_used_api_key", _reached)

    class _Req:
        url = SimpleNamespace(path = "/v1/completions")

        async def json(self):
            return dict(body, prompt = "Hi")

    return asyncio.run(inference_route.openai_completions(_Req(), "tester"))


def test_completions_refuses_a_stateful_sampler_against_sampled_drafts(monkeypatch):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        _run_completions(monkeypatch, {"mirostat": 2, "temperature": 0.7}, probabilistic = True)
    assert exc.value.status_code == 400
    assert "UNSLOTH_SPEC_DRAFT_SAMPLING=greedy" in exc.value.detail


@pytest.mark.parametrize(
    "body, probabilistic",
    [
        ({"mirostat": 2, "temperature": 0.7}, False),
        ({"mirostat": 2, "temperature": 0}, True),
        ({"temperature": 0.7, "top_k": 20}, True),
    ],
)
def test_completions_forwards_what_the_server_verifies_exactly(monkeypatch, body, probabilistic):
    with pytest.raises(_Reached):
        _run_completions(monkeypatch, body, probabilistic = probabilistic)


@pytest.mark.skipif(sys.platform != "win32", reason = "64-bit long")
def test_gguf_mirostat_overflow_saturates_on_windows(monkeypatch):
    assert not _emits_with_gguf(monkeypatch, {"general.sampling.mirostat": "4294967296"})


def _spec_prob_mtp_backend(tmp_path):
    import test_llama_cpp_placement as placement

    backend, gguf = placement._per_slot_mtp_backend(tmp_path, gb_per_slot = 0.0)
    caps = dict(backend.probe_server_capabilities())
    caps["supports_spec_draft_sampling"] = True
    backend.probe_server_capabilities = lambda _binary = None: caps
    read = backend._read_gguf_metadata

    def read_metadata(path):
        read(path)
        backend._gguf_header_parsed = True
        backend._gguf_path = path
        backend._gguf_sampling = {}
        backend._architecture = backend._architecture or "qwen35"

    backend._read_gguf_metadata = read_metadata
    return placement, backend, gguf


def test_a_recovery_without_speculation_clears_the_sampled_draft_marker(tmp_path):
    placement, backend, gguf = _spec_prob_mtp_backend(tmp_path)
    cmds = placement._launch_with_failing_drafter(backend, gguf)
    assert "probabilistic" in cmds[0]
    assert "draft-mtp" not in cmds[-1] and "--spec-draft-sampling" not in cmds[-1]
    assert backend._spec_draft_probabilistic is False


def test_a_launch_with_sampled_drafts_sets_the_marker_and_unload_clears_it(tmp_path):
    placement, backend, gguf = _spec_prob_mtp_backend(tmp_path)
    cmd = placement._launch_auto_spec(backend, gguf)["cmd"]
    assert cmd[cmd.index("--spec-draft-sampling") + 1] == "probabilistic"
    assert backend._spec_draft_probabilistic is True
    backend.unload_model()
    assert backend._spec_draft_probabilistic is False


@pytest.mark.parametrize(
    "body, launch, stateful",
    [
        ({"samplers": "kpa"}, -1.0, False),
        ({"samplers": "kpa", "adaptive_target": None}, -1.0, False),
        ({"samplers": ["adaptive_p"], "adaptive_target": None}, 0.45, True),
        ({"samplers": ["adaptive_p"]}, 0.45, True),
        ({"samplers": ["adaptive_p"]}, None, True),
        ({"samplers": ["adaptive_p"], "adaptive_target": 0.3}, -1.0, True),
        ({"samplers": ["adaptive_p"], "adaptive_target": -1}, 0.45, False),
    ],
)
def test_an_omitted_adaptive_target_is_the_launchs(body, launch, stateful):
    from core.inference.llama_cpp import _request_selects_stateful_sampler
    assert _request_selects_stateful_sampler(body, launch) is stateful


def test_launch_adaptive_target_reads_config_then_the_command(tmp_path):
    from core.inference.llama_cpp import _launch_adaptive_target

    env = {"XDG_CONFIG_HOME": str(tmp_path), "HOME": str(tmp_path)}
    assert _launch_adaptive_target(["llama-server"], env, "linux") == -1.0
    (tmp_path / "llama.cpp").mkdir()
    (tmp_path / "llama.cpp" / "config.ini").write_text("adaptive-target = 0.5\n")
    assert _launch_adaptive_target(["llama-server"], env, "linux") == 0.5
    cmd = ["llama-server", "--adaptive-target", "-0.25"]
    assert _launch_adaptive_target(cmd, env, "linux") == -0.25
    assert _launch_adaptive_target(["llama-server", "--adaptive-target", "x"], env, "linux") is None


@pytest.mark.parametrize(
    "value, expected",
    [
        ("-0x1p-1", -0.5),
        ("0X1.8P1", 3.0),
        (" -0x.8", -0.5),
        ("0x1p", 1.0),
        ("0x", 0.0),
        ("-0.25e1x", -2.5),
        ("1e", 1.0),
        ("INFINITY", float("inf")),
        ("0x1p99999", None),
        ("١", None),
    ],
)
def test_launch_adaptive_target_reads_floats_like_stof(tmp_path, value, expected):
    # std::stof is strtof: hex floats parse ("-0x1p-1" is -0.5, so adaptive-p stays off), only
    # ASCII digits count, and an out-of-range value throws.
    from core.inference.llama_cpp import _launch_adaptive_target, _request_selects_stateful_sampler

    env = {"XDG_CONFIG_HOME": str(tmp_path), "HOME": str(tmp_path)}
    cmd = ["llama-server", "--adaptive-target", value]
    assert _launch_adaptive_target(cmd, env, "linux") == expected
    (tmp_path / "llama.cpp").mkdir()
    (tmp_path / "llama.cpp" / "config.ini").write_text(f"adaptive-target = {value.strip()}\n")
    assert _launch_adaptive_target(["llama-server"], env, "linux") == expected
    if expected is not None:
        body = {"prompt": "Hi", "temperature": 0.7, "samplers": ["adaptive_p"]}
        assert _request_selects_stateful_sampler(body, expected) is not (expected < 0)


def _sampled_draft_state(backend):
    return {
        "emitted": backend._spec_draft_sampling_emitted,
        "probabilistic": backend._spec_draft_probabilistic,
        "target": backend._launch_adaptive_target,
    }


_CLEARED = {"emitted": False, "probabilistic": False, "target": None}


def test_unload_clears_the_launchs_adaptive_target(tmp_path):
    placement, backend, gguf = _spec_prob_mtp_backend(tmp_path)
    placement._launch_auto_spec(backend, gguf, extra_args = ["--adaptive-target", "0.5"])
    assert _sampled_draft_state(backend) == {"emitted": True, "probabilistic": True, "target": 0.5}
    backend.unload_model()
    assert _sampled_draft_state(backend) == _CLEARED


def test_a_diffusion_load_after_sampled_drafts_clears_the_marker_before_the_runner(tmp_path):
    """load_model kills the previous server without unload_model and returns before the
    command builder, so the diffusion branch itself must drop the MTP launch's marker."""
    placement, backend, gguf = _spec_prob_mtp_backend(tmp_path)
    placement._launch_auto_spec(backend, gguf, extra_args = ["--adaptive-target", "0.5"])
    assert backend._spec_draft_probabilistic is True
    diffusion = placement._write_gguf(tmp_path / "diffusion.gguf", "diffusion-gemma")
    backend._read_gguf_metadata = lambda _path: setattr(backend, "_is_diffusion", True)
    seen = {}

    def start_diffusion_server(**_kwargs):
        seen.update(_sampled_draft_state(backend))
        return True

    backend._start_diffusion_server = start_diffusion_server
    assert backend.load_model(
        placement.GgufLoadIntent(
            gguf_path = str(diffusion),
            model_identifier = "unsloth/diffusiongemma-26B-A4B-it-GGUF",
            speculative_type = "off",
        )
    )
    assert seen == _CLEARED


def test_the_diffusion_runner_clears_a_previous_sampled_draft_marker(monkeypatch, tmp_path):
    import test_diffusion_auto_fit_ngl as dfit

    fit_host = dfit._fit_host

    def stale_fit_host(*args, **kwargs):
        backend = fit_host(*args, **kwargs)
        backend._spec_draft_sampling_emitted = True
        backend._spec_draft_probabilistic = True
        backend._launch_adaptive_target = 0.5
        return backend

    monkeypatch.setattr(dfit, "_fit_host", stale_fit_host)
    _cmd, backend = dfit._launch(monkeypatch, tmp_path, [(0, 7_800, 8_188)])
    assert _sampled_draft_state(backend) == _CLEARED


def test_the_adaptive_target_is_read_with_the_childs_env(tmp_path, monkeypatch):
    """config.ini is located from the env the child is spawned with, as its other argv reads are."""
    seen = {}
    real = _llama_cpp_mod._launch_adaptive_target

    def spy(
        cmd,
        env = None,
        platform = None,
    ):
        seen["env"] = env
        return real(cmd, env, platform)

    monkeypatch.setattr(_llama_cpp_mod, "_launch_adaptive_target", spy)
    placement, backend, gguf = _spec_prob_mtp_backend(tmp_path)
    captured = placement._launch_auto_spec(backend, gguf)
    assert backend._spec_draft_probabilistic is True
    assert seen["env"] is not None and seen["env"] == captured["env"]
