# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import pytest

from core.inference import llama_cpp as llama
from core.inference.llama_custom_config import (
    CustomConfigError,
    compile_custom_config,
    parse_config_source,
)

_OPTIONS = (
    "--port; -c --ctx-size; -ngl --gpu-layers; -fa --flash-attn; -np --parallel; -m --model; "
    "-mm --mmproj; --temp --temperature; --top-k; --top-p; --warmup --no-warmup; "
    "--mmproj-offload --no-mmproj-offload; --jinja --no-jinja; -dev --device; "
    "-ctk --cache-type-k; --host; --cache-prompt --no-cache-prompt; -mmdev --mmproj-device; "
    "-md --model-draft; -devd --device-draft"
)
# Like the probe: every spelling of one option maps to that option's help block.
FLAGS = {flag: group for group in _OPTIONS.split("; ") for flag in group.split()}
SWITCHES = {
    "--warmup",
    "--no-warmup",
    "--mmproj-offload",
    "--no-mmproj-offload",
    "--jinja",
    "--no-jinja",
    "--cache-prompt",
    "--no-cache-prompt",
}


def source(ini, section = None):
    return {"version": 1, "mode": "custom", "ini": ini, "section": section}


def compile_ini(ini, section = None):
    return compile_custom_config(source(ini, section), FLAGS, SWITCHES)


def test_preset_grammar_compiles_to_argv_and_defaults():
    compiled = compile_ini(
        "; comment\n[*]\nc = 8192\nngl=-1\n[qwen]\nc=128000 # wins\nfa=on\nwarmup=false\n"
        "mmproj-offload=off\nnp=4\nm=/ignored.gguf\ntemp=1\ntop-k=20",
        "qwen",
    )
    assert compiled.argv == (
        "-c",
        "128000",
        "-ngl",
        "-1",
        "-fa",
        "on",
        "--no-warmup",
        "--no-mmproj-offload",
        "--temp",
        "1",
        "--top-k",
        "20",
    )
    assert compiled.n_parallel == 4
    assert compiled.request_defaults == {"temperature": 1.0, "top_k": 20}
    assert compiled.diagnostics and not compiled.cpu_only


@pytest.mark.parametrize(
    "ini,section,message",
    [
        ("[a]\nc=1\n[b]\nc=2", None, "Select which"),
        ("c=1", "missing", "does not exist"),
        ("bogus=1", None, "not an option"),
        ("host=0.0.0.0", None, "managed by Studio"),
        ("np=65", None, "between"),
        ("jinja=false", None, "tool calling"),
        ("warmup=maybe", None, "true or false"),
        ("temp=hot", None, "number"),
        ("not an entry", None, "line 1"),
    ],
)
def test_refusals_name_the_key_not_the_value(ini, section, message):
    with pytest.raises(CustomConfigError, match = message):
        compile_ini(ini, section)


@pytest.mark.parametrize("ini", ["c=1024\nctx-size=2048", "warmup=true\nno-warmup=true"])
def test_two_spellings_of_one_option_are_refused(ini):
    with pytest.raises(CustomConfigError, match = "set the same option"):
        compile_ini(ini)


def test_sampling_aliases_and_router_keys():
    compiled = compile_ini(
        "temperature=0.2\nload-on-startup=true\nstop-timeout=5\ndedup-cache-models=1"
    )
    assert compiled.request_defaults == {"temperature": 0.2}
    assert compiled.argv == ("--temperature", "0.2")


@pytest.mark.parametrize("ini", ["top-k=180", "temp=nan", "top-p=inf", "temp=-1"])
def test_defaults_outside_the_chat_schema_are_refused(ini):
    with pytest.raises(CustomConfigError, match = "between"):
        compile_ini(ini)


def test_unprobed_binary_is_refused():
    with pytest.raises(CustomConfigError, match = "did not report"):
        compile_custom_config(source("c=1"), {}, set())


@pytest.mark.parametrize(
    "value",
    [
        {"version": 2, "mode": "custom", "ini": "c=1"},
        {"version": 1, "mode": "custom", "ini": " "},
        {"version": 1, "mode": "custom", "ini": "c=1\x00"},
        {"version": 1, "mode": "custom", "ini": "x" * (64 * 1024 + 1)},
        {"version": 1, "mode": "custom", "ini": "c=1", "section": "a]b"},
        {"version": 1, "mode": "managed", "ini": "c=1"},
    ],
)
def test_wire_validation(value):
    with pytest.raises(CustomConfigError):
        parse_config_source(value)


@pytest.mark.parametrize(
    "ini,cpu_only",
    [
        ("dev=none", True),
        ("dev=none\nmmdev=none", True),
        ("dev=none\nmd=d.gguf\ndevd=none", True),
        ("ngl=0", False),  # op offload and the projector still use the GPU
        ("dev=none\nmmdev=CUDA0", False),
        ("dev=none\nmd=d.gguf", False),
    ],
)
def test_cpu_only_counts_every_companion(ini, cpu_only):
    assert compile_ini(ini).cpu_only is cpu_only


def test_unsectioned_keys_are_the_default_section_not_global():
    ini = "c=1000\n[*]\nngl=-1\n[large]\nc=9000"
    assert compile_ini(ini, "large").option("-c") == "9000"
    assert compile_ini("fa=on\n[large]\nc=9000", "large").option("-fa") is None
    assert compile_ini("fa=on\n[large]\nc=9000", "default").option("-fa") == "on"
    assert compile_ini("c=1000\n[*]\nngl=-1").argv == ("-ngl", "-1", "-c", "1000")
    with pytest.raises(CustomConfigError, match = "Select which"):
        compile_ini("fa=on\n[large]\nc=9000")


@pytest.fixture
def launch(monkeypatch, tmp_path):
    backend = llama.LlamaCppBackend(manages_processes = False)
    model, binary = tmp_path / "selected.gguf", tmp_path / "llama-server"
    model.touch()
    binary.touch()
    caps = {"help_probe_ok": True, "flags": FLAGS, "switch_flags": SWITCHES}
    for name, value in {
        "_find_llama_server_binary": lambda: str(binary),
        "_exec_path_for_launch": lambda p: p,
        "probe_server_capabilities": lambda *a, **k: caps,
        "_binary_revision": lambda p: (str(p), 1),
        "_cuda_sm_gate_error": lambda p: None,
        "_arch_gate_survivors": lambda p: [],
        "_non_chat_gguf_refusal_for_path": lambda *a: None,
        "_gguf_path_is_diffusion": lambda *a: False,
        "_spawn_is_stale": lambda: False,
        "_find_free_port": lambda: 8123,
        "_wait_for_health": lambda **k: True,
        "_detect_audio_type_strict": lambda: None,
        "_llama_server_env_for_binary": lambda p: {
            "PATH": "libs",
            "LLAMA_ARG_CTX_SIZE": "1",
            "KEEP": "1",
        },
        "_query_server_props": lambda: {
            "total_slots": 2,
            "default_generation_settings": {"n_ctx": 28000},
            "modalities": {"vision": False},
        },
    }.items():
        monkeypatch.setattr(backend, name, value)
    monkeypatch.setattr(llama, "_metal_device_is_paravirtual", lambda: False, raising = False)
    monkeypatch.setattr(
        llama.LlamaCppBackend,
        "_read_gguf_metadata",
        lambda self, path: setattr(self, "_context_length", 128000),
    )
    captured = []

    def spawn(cmd, env, **kwargs):
        captured.append((list(cmd), dict(env)))
        backend._process = SimpleNamespace(poll = lambda: None, pid = 1)
        return True

    monkeypatch.setattr(backend, "_start_llama_process", spawn)
    monkeypatch.setattr(
        backend, "_kill_process", lambda *a, **k: setattr(backend, "_process", None)
    )
    intent = llama.GgufLoadIntent(
        model_identifier = "selected",
        gguf_path = str(model),
        n_ctx = 4096,
        n_parallel = 8,
        extra_args = ("--ctx-size", "99999"),
        llama_cpp_config = source("[*]\nnp=2\nc=56000\nngl=-1\nno-warmup=true\ntemp=0.3"),
    )
    return SimpleNamespace(backend = backend, intent = intent, captured = captured)


def test_custom_launch_runs_only_the_ini(launch):
    assert launch.backend.load_model(launch.intent)
    cmd, env = launch.captured[0]
    assert cmd[cmd.index("--parallel") + 1] == "2"
    assert "99999" not in cmd and "--jinja" in cmd
    assert cmd[cmd.index("-c") + 1] == "56000" and "--no-warmup" in cmd
    assert not any(k.startswith("LLAMA_ARG_") for k in env)
    assert env["KEEP"] == "1"
    summary = launch.backend.llama_cpp_config_summary
    assert summary["request_defaults"] == {"temperature": 0.3}
    assert launch.backend.extra_args == []
    # The duplicate-load fast path only reuses a probed server.
    assert launch.backend._audio_probed is True


def test_cpu_only_custom_load_still_masks_unsupported_rocm_cards(launch, monkeypatch):
    from dataclasses import replace

    masked = []
    monkeypatch.setattr(launch.backend, "_arch_gate_survivors", lambda p: [1])
    monkeypatch.setattr(
        launch.backend, "_emit_child_gpu_visibility", lambda env, ids, **k: masked.append(ids)
    )
    cpu = replace(launch.intent, llama_cpp_config = parse_config_source(source("dev=none")))
    assert launch.backend.load_model(cpu)
    assert masked == ["1"]
    # The GPU arbiter leaves image/video work alone for a zero-VRAM resident server.
    assert launch.backend.holds_no_vram


def test_gpu_custom_load_holds_vram(launch):
    assert launch.backend.load_model(launch.intent)
    assert not launch.backend.holds_no_vram


def test_bad_config_never_spawns(launch):
    from dataclasses import replace

    bad = replace(launch.intent, llama_cpp_config = parse_config_source(source("bogus=1")))
    with pytest.raises(CustomConfigError):
        launch.backend.load_model(bad)
    assert launch.captured == []
    assert launch.backend.llama_cpp_config_summary is None


def test_start_failure_carries_llama_servers_reason(launch, monkeypatch):
    launch.backend._stdout_lines = ["load_model: error: invalid device: ROCm1"]
    monkeypatch.setattr(launch.backend, "_wait_for_health", lambda **k: False)
    monkeypatch.setattr(launch.backend, "_arch_gate_survivors", lambda p: [1])
    monkeypatch.setattr(launch.backend, "_emit_child_gpu_visibility", lambda *a, **k: None)
    with pytest.raises(CustomConfigError, match = "invalid device: ROCm1.*0 to 0"):
        launch.backend.load_model(launch.intent)
