# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from dataclasses import replace
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from core.inference import llama_cpp as llama
from core.inference.llama_custom_config import CustomConfigError, parse_option_catalog

_spec = importlib.util.spec_from_file_location(
    "_custom_config_fixtures", Path(__file__).with_name("test_llama_custom_config.py")
)
_fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixtures)
HELP, source = _fixtures.HELP, _fixtures.source


@pytest.fixture
def launch(monkeypatch, tmp_path):
    backend = llama.LlamaCppBackend(manages_processes = False)
    model = tmp_path / "selected.gguf"
    binary = tmp_path / "llama-server.exe"
    model.touch()
    binary.touch()
    catalog = parse_option_catalog(
        HELP
        + "--alias NAME                            public model id\n--jinja                                use Jinja\n"
    )
    caps = {"help_probe_ok": True, "option_catalog": catalog}
    monkeypatch.setattr(backend, "_find_llama_server_binary", lambda: str(binary))
    monkeypatch.setattr(backend, "_exec_path_for_launch", lambda p: p)
    monkeypatch.setattr(backend, "probe_server_capabilities", lambda p: caps)
    monkeypatch.setattr(backend, "_binary_revision", lambda p: (str(p), 1))
    monkeypatch.setattr(backend, "_cuda_sm_gate_error", lambda p: None)
    monkeypatch.setattr(backend, "_arch_gate_survivors", lambda p: [])
    monkeypatch.setattr(backend, "_non_chat_gguf_refusal_for_path", lambda *args: None)
    monkeypatch.setattr(backend, "_gguf_path_is_diffusion", lambda *args: False)
    monkeypatch.setattr(llama, "_metal_device_is_paravirtual", lambda: False)
    monkeypatch.setattr(
        "utils.model_memory_settings.get_model_memory_settings", lambda: (False, False)
    )
    monkeypatch.setattr(backend, "_spawn_is_stale", lambda: False)
    monkeypatch.setattr(
        backend,
        "_llama_server_env_for_binary",
        lambda p: {
            "PATH": "runtime-libraries",
            "CUDA_VISIBLE_DEVICES": "1",
            "LLAMA_ARG_CTX_SIZE": "128000",
            "LLAMA_ARG_MODELS_PRESET": "never-read.ini",
            "LLAMA_API_KEY": "inherited-secret",
            "LLAMA_ARG_MMPROJ": "different-projector.gguf",
            "RUNTIME_UNRELATED": "retained",
        },
    )
    monkeypatch.setattr(
        llama.LlamaCppBackend,
        "_read_gguf_metadata",
        lambda self, path: setattr(self, "_context_length", 128000),
    )
    monkeypatch.setattr(backend, "_find_free_port", lambda: 8123)
    monkeypatch.setattr(backend, "_detect_audio_type_strict", lambda: None)
    monkeypatch.setattr(backend, "_apply_detected_audio", lambda *a: True)
    monkeypatch.delenv("UNSLOTH_DIRECT_STREAM", raising = False)
    captured, kills = [], []

    def kill():
        kills.append(backend._process)
        backend._process = None
        backend._healthy = False

    def spawn(cmd, env, *, child_gpu_physical_ids):
        captured.append((list(cmd), dict(env)))
        backend._process = SimpleNamespace(poll = lambda: None)
        return True

    monkeypatch.setattr(backend, "_kill_process", kill)
    monkeypatch.setattr(backend, "_start_llama_process", spawn)
    monkeypatch.setattr(backend, "_wait_for_health", lambda **kwargs: True)
    monkeypatch.setattr(
        backend,
        "_query_server_props",
        lambda: {
            "total_slots": 2,
            "default_generation_settings": {"n_ctx": 28000},
            "modalities": {"video": False},
        },
    )
    intent = llama.GgufLoadIntent(
        model_identifier = "selected",
        gguf_path = str(model),
        n_ctx = 128000,
        n_parallel = 8,
        gpu_layers = 55,
        n_batch = 100,
        extra_args = ("--ctx-size", "99999"),
        llama_cpp_config = source(
            "[*]\nnp=2\nc=56000\nfit=off\nngl=0\nb=100000\nub=3000\nno-warmup=true\nmmproj-offload=false"
        ),
    )
    return SimpleNamespace(
        backend = backend, intent = intent, captured = captured, kills = kills, caps = caps
    )


def test_custom_dispatch_precedes_managed_mutation_and_preserves_tuning(launch, monkeypatch):
    backend = launch.backend
    monkeypatch.setattr(
        backend,
        "_preserve_cpu_fallback_intent",
        lambda *_: pytest.fail("managed fallback must not run"),
    )
    monkeypatch.setattr(
        backend,
        "_build_speculative_flags",
        lambda **_: pytest.fail("managed speculation must not run"),
    )
    assert backend.load_model(launch.intent)
    cmd, env = launch.captured[0]
    assert cmd.count("--parallel") == 1 and cmd[cmd.index("--parallel") + 1] == "2"
    assert cmd[cmd.index("--ctx-size") + 1] == "56000"
    assert cmd[cmd.index("--fit") + 1] == "off"
    assert cmd[cmd.index("--batch-size") + 1] == "100000"
    assert "128000" not in cmd and "99999" not in cmd and "--flash-attn" not in cmd
    assert "--no-warmup" in cmd and "--no-mmproj-offload" in cmd
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1"
    assert env == {
        "PATH": "runtime-libraries",
        "CUDA_VISIBLE_DEVICES": "1",
        "RUNTIME_UNRELATED": "retained",
    }
    assert backend.effective_parallel_slots == 2
    assert backend.context_length == 28000
    assert backend.last_load_intent.llama_cpp_config == launch.intent.llama_cpp_config
    assert backend.llama_cpp_config_summary["tuning"]["n_ctx"] == 56000
    assert backend._mtp_runtime_fallback_active is False


@pytest.mark.parametrize(
    "text",
    [
        "host=evil",
        "c=bad",
        "np=99",
        "unknown=1",
        "config=second.ini",
        "temp=1e100",
        "temp=1e-100",
        "temp=1e-1000",
    ],
)
def test_invalid_config_does_not_replace_healthy_resident(launch, text):
    backend = launch.backend
    resident = backend._process = object()
    backend._healthy = True
    with pytest.raises(CustomConfigError):
        backend.load_model(replace(launch.intent, llama_cpp_config = source("[*]\n" + text)))
    assert backend._process is resident and backend._healthy
    assert not launch.kills and not launch.captured


@pytest.mark.parametrize(
    "keep,no_reserve", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("mlock", [False, True])
def test_custom_memory_policy_respects_no_reserve_precedence(
    launch, monkeypatch, keep, no_reserve, mlock
):
    monkeypatch.setattr(
        "utils.model_memory_settings.get_model_memory_settings", lambda: (keep, no_reserve)
    )
    intent = replace(
        launch.intent, llama_cpp_config = source(f"[*]\nnp=2\nmlock={str(mlock).lower()}")
    )
    allowed = (not no_reserve or not mlock) and (not keep or no_reserve or mlock)
    if allowed:
        launch.backend.prepare_custom_config(intent)
    else:
        with pytest.raises(CustomConfigError, match = "memory settings"):
            launch.backend.prepare_custom_config(intent)
    assert not launch.kills and not launch.captured


def test_incomplete_help_does_not_unload(launch):
    launch.caps["help_probe_ok"] = False
    with pytest.raises(CustomConfigError, match = "complete successful help"):
        launch.backend.load_model(launch.intent)
    assert not launch.kills


def test_explicit_custom_cpu_placement_reports_no_vram(launch):
    launch.caps["option_catalog"] = parse_option_catalog(
        HELP + "\n--device DEVICES                       device selection\n"
        "--alias NAME                            public model id\n"
        "--jinja                                 Jinja templates\n"
    )
    intent = replace(
        launch.intent,
        llama_cpp_config = source("[*]\nnp=2\nc=56000\nngl=0\ndevice=none\nfit=off"),
    )
    assert launch.backend.load_model(intent)
    assert launch.backend.holds_no_vram
    assert "--device" in launch.captured[0][0]


def test_start_failure_is_terminal_and_does_not_commit_intent(launch, monkeypatch):
    old_intent = llama.GgufLoadIntent("previous")
    launch.backend._last_load_intent = old_intent
    monkeypatch.setattr(launch.backend, "_wait_for_health", lambda **_: False)
    launch.backend._health_wait_cancelled = False
    with pytest.raises(CustomConfigError, match = "no tuning fallback"):
        launch.backend.load_model(launch.intent)
    assert len(launch.captured) == 1
    assert launch.backend._process is None
    assert launch.backend.last_load_intent is old_intent


def test_cancel_before_validation_does_not_replace(launch):
    cancel = threading.Event()
    cancel.set()
    assert launch.backend.load_model(launch.intent, load_cancel_event = cancel) is False
    assert not launch.kills and not launch.captured


def test_cancel_during_health_cleans_child_and_does_not_retry(launch, monkeypatch):
    cancel = threading.Event()

    def wait(**_):
        cancel.set()
        return False

    monkeypatch.setattr(launch.backend, "_wait_for_health", wait)
    assert launch.backend.load_model(launch.intent, load_cancel_event = cancel) is False
    assert len(launch.captured) == 1 and launch.backend._process is None
    assert launch.backend.last_load_intent is None


def test_props_accounting_mismatch_is_terminal(launch, monkeypatch):
    monkeypatch.setattr(
        launch.backend,
        "_query_server_props",
        lambda: {"total_slots": 8, "default_generation_settings": {"n_ctx": 1}},
    )
    with pytest.raises(CustomConfigError, match = "confirm.*accounting"):
        launch.backend.load_model(launch.intent)
    assert len(launch.captured) == 1 and launch.backend.last_load_intent is None


def test_comment_only_change_dedupes_but_tuning_change_relaunches(launch):
    assert launch.backend.load_model(launch.intent)
    old_digest = launch.backend.llama_cpp_config_summary["digest"]
    commented = replace(
        launch.intent, llama_cpp_config = source(launch.intent.llama_cpp_config.ini + "\n# note")
    )
    assert launch.backend.load_model(commented)
    assert len(launch.captured) == 1
    assert launch.backend.requested_llama_cpp_config["ini"].endswith("# note")
    changed = replace(
        launch.intent,
        llama_cpp_config = source(launch.intent.llama_cpp_config.ini.replace("56000", "57000")),
    )
    assert launch.backend.load_model(changed)
    assert len(launch.captured) == 2
    assert launch.backend.llama_cpp_config_summary["digest"] != old_digest


def test_replay_preserves_compiled_tuning_and_bypasses_mtp_recovery(launch):
    assert launch.backend.load_model(launch.intent)
    replay = launch.backend.last_load_intent
    launch.backend._healthy = False
    assert launch.backend.load_model(replay)
    assert launch.captured[0] == launch.captured[1]
    launch.backend._mtp_runtime_fallback_active = True
    assert launch.backend._maybe_recover_from_mtp_crash() is False


def test_managed_intent_cannot_dedupe_against_custom_and_unload_clears(launch):
    assert launch.backend.load_model(launch.intent)
    assert (
        launch.backend._runtime_matches_intent(
            replace(launch.intent, llama_cpp_config = {"version": 1, "mode": "managed"}), []
        )
        is False
    )
    assert launch.backend.unload_model()
    assert launch.backend.llama_cpp_config_summary is None
    assert launch.backend.requested_llama_cpp_config is None


def test_jinja_conflict_and_virtual_metal_refused_before_kill(launch, monkeypatch):
    bad = replace(launch.intent, llama_cpp_config = source("[*]\nnp=2\njinja=false"))
    with pytest.raises(CustomConfigError, match = "Jinja"):
        launch.backend.load_model(bad)
    monkeypatch.setattr(llama, "_metal_device_is_paravirtual", lambda: True)
    with pytest.raises(CustomConfigError, match = "Virtualized Metal"):
        launch.backend.load_model(launch.intent)
    assert not launch.kills


def test_implicit_native_config_is_refused(monkeypatch, tmp_path):
    monkeypatch.setattr(llama.sys, "platform", "win32")
    native = tmp_path / "llama.cpp"
    native.mkdir()
    (native / "config.ini").touch()
    with pytest.raises(CustomConfigError, match = "single configuration source"):
        llama.LlamaCppBackend._reject_implicit_custom_config({"APPDATA": str(tmp_path)})


def test_binary_changed_between_preflight_and_kill_is_refused(launch, monkeypatch):
    calls = 0

    def revision(path):
        nonlocal calls
        calls += 1
        return (str(path), calls)

    monkeypatch.setattr(launch.backend, "_binary_revision", revision)
    with pytest.raises(CustomConfigError, match = "changed during validation"):
        launch.backend.load_model(launch.intent)
    assert not launch.kills


def test_remote_preflight_cannot_defer_selected_resource_identity(launch):
    remote = replace(
        launch.intent,
        gguf_path = None,
        hf_repo = "example/model-GGUF",
        llama_cpp_config = source("[*]\nnp=2\nm=unresolved.gguf"),
    )
    with pytest.raises(CustomConfigError, match = "must match"):
        launch.backend.prepare_custom_config(remote, validate_resources = False)
    assert not launch.kills and not launch.captured


def test_remote_resolution_cannot_change_the_requested_quantization(launch, monkeypatch):
    seen = []

    def download(**kwargs):
        seen.append(kwargs)
        return launch.intent.gguf_path

    monkeypatch.setattr(launch.backend, "_download_gguf", download)
    monkeypatch.setattr(llama, "_hf_offline_if_unreachable", nullcontext)
    monkeypatch.setattr(llama, "_resolve_repo_id_casing", lambda repo: repo)
    monkeypatch.setattr(llama, "_hub_download_blocks_gguf_load", lambda *a, **k: False)
    remote = replace(launch.intent, gguf_path = None, hf_repo = "example/model-GGUF")
    assert launch.backend.load_model(remote)
    assert len(seen) == 1 and seen[0]["allow_smaller_fallback"] is False


def test_vision_disabled_resource_conflict_is_refused_in_preflight(launch, tmp_path, monkeypatch):
    projector = tmp_path / "mmproj.gguf"
    projector.touch()
    monkeypatch.setattr(llama, "_mmproj_env_is_audio_only", lambda p: False)
    intent = replace(
        launch.intent,
        mmproj_path = str(projector),
        disable_vision = True,
        llama_cpp_config = source(f"[*]\nnp=2\nmm={projector}"),
    )
    with pytest.raises(CustomConfigError, match = "must match"):
        launch.backend.prepare_custom_config(intent)
    assert not launch.kills


def test_strict_adoption_never_applies_managed_placement(launch, monkeypatch):
    assert launch.backend.load_model(launch.intent)
    monkeypatch.setattr(
        launch.backend,
        "_preserve_cpu_fallback_intent",
        lambda *a, **k: pytest.fail("managed placement"),
    )
    monkeypatch.setattr(launch.backend, "_binary_changed_since_launch", lambda: False)
    changed_source = replace(
        launch.intent, llama_cpp_config = source(launch.intent.llama_cpp_config.ini + "\n# edit")
    )
    assert launch.backend.adopt_load_intent_if_matched(changed_source)
    assert launch.backend.last_load_intent.llama_cpp_config == changed_source.llama_cpp_config
    assert len(launch.captured) == 1


def test_reasoning_kwargs_merge_preserves_custom_arbitrary_values(launch):
    config = replace(
        launch.intent,
        llama_cpp_config = source(
            '[*]\nnp=2\nchat-template-kwargs={"enable_thinking":false,"custom":{"nested":1},"reasoning_effort":"high"}'
        ),
    )
    compiled = launch.backend.prepare_custom_config(config)
    launch.backend._compiled_custom_config = compiled
    launch.backend._supports_reasoning = True
    launch.backend._reasoning_style = "enable_thinking"
    assert launch.backend._request_reasoning_kwargs(None)["enable_thinking"] is False
    value = launch.backend._request_reasoning_kwargs(
        True, request_template_kwargs = {"custom": {"nested": 2}, "extra": 0}
    )
    assert value == {
        "enable_thinking": True,
        "custom": {"nested": 2},
        "reasoning_effort": "high",
        "extra": 0,
    }
    assert launch.backend.llama_cpp_config_summary["request_defaults"]["chat_template_kwargs"][
        "custom"
    ] == {"nested": 1}


def test_actual_native_template_controls_reasoning_capabilities(launch, monkeypatch):
    launch.backend._supports_reasoning = True
    launch.backend._supports_tools = True
    template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    monkeypatch.setattr(
        launch.backend,
        "_query_server_props",
        lambda: {
            "total_slots": 2,
            "default_generation_settings": {"n_ctx": 28000},
            "chat_template": template,
        },
    )
    assert launch.backend.load_model(launch.intent)
    assert launch.backend._chat_template == template
    assert launch.backend._supports_reasoning is False
    assert launch.backend._supports_tools is False


@pytest.mark.parametrize(
    ("line", "enabled"),
    [
        ("llama_context: Flash Attention enabled", True),
        ("llama_context: Flash Attention not supported, set to disabled", False),
        ("llama_context: flash_attn = disabled", False),
    ],
)
def test_flash_accounting_uses_observed_native_outcome(launch, line, enabled):
    launch.backend._stdout_lines = ["llama_context: flash_attn = auto", line]
    assert launch.backend.load_model(launch.intent)
    assert launch.backend._flash_attn_enabled is enabled


def test_unreported_native_flash_is_conservative_and_diagnosed(launch):
    assert launch.backend.load_model(launch.intent)
    assert launch.backend._flash_attn_enabled is False
    assert any(
        "not reported" in note for note in launch.backend.llama_cpp_config_summary["diagnostics"]
    )
