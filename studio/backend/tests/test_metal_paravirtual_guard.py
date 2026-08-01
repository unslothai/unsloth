# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A virtualised Apple GPU must fall back to CPU; a real one must not.

The second half is the half that matters: forcing gpu_layers=0 on every Mac would
"fix" the corrupt output by throwing away Metal for every real user, so these tests
pin the discrimination and not just the fallback.
"""

from __future__ import annotations

import ast
import inspect
import sys
import textwrap
import types

import pytest

import core.inference.llama_cpp as llama_cpp
import core.inference.llama_server_args as llama_server_args
from core.inference.llama_cpp import _metal_device_is_paravirtual


@pytest.fixture(autouse = True)
def _clear_cache():
    """The detector is lru_cached so a real machine only pays for the probe once."""
    _metal_device_is_paravirtual.cache_clear()
    yield
    _metal_device_is_paravirtual.cache_clear()


def _fake_mlx(device_name: str):
    module = types.ModuleType("mlx.core")
    module.device_info = lambda: {"device_name": device_name}
    parent = types.ModuleType("mlx")
    parent.core = module
    return parent, module


@pytest.mark.parametrize(
    "device_name, expected",
    [
        ("Apple Paravirtual device", True),
        ("apple paravirtual device", True),  # matching must not be case-sensitive
        ("Apple M1", False),
        ("Apple M3 Max", False),
        ("Apple M4 Pro", False),
    ],
)
def test_only_virtualised_apple_gpus_fall_back(monkeypatch, device_name, expected):
    monkeypatch.setattr(sys, "platform", "darwin")
    parent, core = _fake_mlx(device_name)
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    # Inert probe output so this measures the name matching, not system_profiler.
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(stdout = "Chipset Model: Apple M3 Max"),
    )
    assert _metal_device_is_paravirtual() is expected


def test_non_darwin_never_pays_for_the_probe(monkeypatch):
    """Linux and Windows short-circuit: there is no Metal to be virtualised."""
    monkeypatch.setattr(sys, "platform", "linux")

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("probed for a Metal device off macOS")

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", explode)
    assert _metal_device_is_paravirtual() is False


def test_system_profiler_catches_it_when_mlx_is_absent(monkeypatch):
    """MLX is not on every Mac; without this fallback a virtualised machine without
    MLX would be treated as bare metal and would emit gibberish."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "mlx", None)  # import raises
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(
            stdout = "Graphics/Displays:\n  Apple Paravirtual device:\n    Vendor: Apple"
        ),
    )
    assert _metal_device_is_paravirtual() is True


def test_a_broken_probe_leaves_gpu_offload_alone(monkeypatch):
    """If neither source can answer, assume a real Mac. Guessing "virtualised" would
    silently drop everyone the probe fails on down to CPU."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "mlx", None)

    def explode(*args, **kwargs):
        raise OSError("system_profiler not found")

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", explode)
    assert _metal_device_is_paravirtual() is False


# ── the fallback must actually hold, and must not churn the server ────


class _FakeProcess:
    def poll(self):
        return None

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0


def _paravirtual(monkeypatch):
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: True)


def _load_model_source() -> str:
    return inspect.getsource(llama_cpp.LlamaCppBackend.load_model)


def test_fallback_is_settled_before_the_duplicate_load_check():
    """The launch records the normalized placement ("manual"/0), so a repeat Auto
    request has to be normalized before it is compared, or every duplicate /load
    tears down a healthy CPU server and reloads it."""
    src = _load_model_source()
    assert src.index("_metal_device_is_paravirtual()") < src.index("self._already_in_target_state(")


def test_repeat_auto_load_does_not_reload_a_healthy_cpu_server(monkeypatch, tmp_path):
    """End-to-end on the dedupe path: the second identical Auto /load must take the
    fast path rather than respawn."""
    _paravirtual(monkeypatch)
    backend = llama_cpp.LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._healthy = True
    backend._audio_probed = True
    backend._model_identifier = "owner/repo"
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    backend._gguf_path = str(gguf)
    backend._requested_n_ctx = 8192
    backend._requested_n_parallel = 1
    backend._requested_spec_mode = "auto"
    # What the first Auto load on a virtualised Mac left behind.
    backend._gpu_memory_mode = "manual"
    backend._gpu_layers = 0

    def _never(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("tore down a healthy server on a duplicate /load")

    monkeypatch.setattr(backend, "_find_llama_server_binary", _never)
    monkeypatch.setattr(backend, "_kill_process", _never)

    assert (
        backend.load_model(
            model_identifier = "owner/repo",
            gguf_path = str(gguf),
            n_ctx = 8192,
            gpu_memory_mode = "auto",
            gpu_layers = -1,
            n_parallel = 1,
        )
        is True
    )


def test_a_pass_through_layer_flag_cannot_re_enable_the_corrupt_offload(monkeypatch):
    """Auto never strips -ngl at the route, and user extras are appended after the
    managed --gpu-layers 0, so llama.cpp's last-wins parser would undo the pin."""
    _paravirtual(monkeypatch)
    backend = llama_cpp.LlamaCppBackend()
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(backend, "_already_in_target_state", _capture)
    monkeypatch.setattr(backend, "_apply_detected_audio", lambda _d: True)
    backend._audio_probed = True
    backend._healthy = True
    backend.load_model(
        model_identifier = "owner/repo",
        gguf_path = "/nonexistent/model.gguf",
        gpu_memory_mode = "auto",
        gpu_layers = -1,
        extra_args = ["-ngl", "99", "--top-k", "40"],
    )
    assert seen["gpu_memory_mode"] == "manual"
    assert seen["gpu_layers"] == 0
    assert seen["extra_args"] == ["--top-k", "40"]


def test_a_real_mac_keeps_its_offload_flags(monkeypatch):
    """The stripping is scoped to the fallback: a physical Mac must keep -ngl."""
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: False)
    backend = llama_cpp.LlamaCppBackend()
    seen = {}

    def _capture(**kwargs):
        seen.update(kwargs)
        return True

    monkeypatch.setattr(backend, "_already_in_target_state", _capture)
    monkeypatch.setattr(backend, "_apply_detected_audio", lambda _d: True)
    backend._audio_probed = True
    backend._healthy = True
    backend.load_model(
        model_identifier = "owner/repo",
        gguf_path = "/nonexistent/model.gguf",
        gpu_memory_mode = "auto",
        gpu_layers = -1,
        extra_args = ["-ngl", "99"],
    )
    assert seen["gpu_memory_mode"] == "auto"
    assert seen["extra_args"] == ["-ngl", "99"]


def test_gpu_companions_are_pinned_to_cpu_too():
    """--gpu-layers 0 does not reach them: clip.cpp never reads n_gpu_layers (it
    offloads on the mmproj_use_gpu boolean, default true) and a separate drafter
    gets params.speculative.draft.n_gpu_layers, default -1 = auto."""
    src = _load_model_source()
    assert '"--no-mmproj-offload"' in src
    # The drafter flag name comes from the probe, never a literal: --spec-draft-ngl
    # only exists from llama.cpp b8955, and an older build exposing only -ngld would
    # refuse to start on the newer name, which is exactly what the gate prevents.
    assert (
        '"--spec-draft-ngl", "0"' not in src
    ), "hardcoding the flag defeats the capability probe on builds that only have -ngld"
    assert 'server_caps["spec_draft_ngl_flag"]' in src


def _load_model_tree() -> ast.AST:
    return ast.parse(textwrap.dedent(_load_model_source()))


def test_the_companion_pins_are_keyed_on_the_hardware_not_the_request():
    """A caller that already asks for Manual + 0 layers is on the same corrupt
    device, but the main model's placement needs no rewrite -- so skipping the
    whole block for it left _paravirtual_cpu_forced False and silently dropped the
    mmproj / drafter pins, which do not read --gpu-layers at all."""
    tree = _load_model_tree()
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_paravirtual_cpu_forced" for t in node.targets)
    ]
    assert len(assigns) == 1, "the flag must have exactly one source of truth"
    (assign,) = assigns
    assert isinstance(assign.value, ast.Call)
    assert assign.value.func.id == "_metal_device_is_paravirtual"

    # The negative: it must NOT be nested under a test of the requested placement,
    # which is what made a manual CPU load lose the companion pins.
    def _mentions(node, names):
        found = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        return names <= found

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not _mentions(node.test, {"gpu_memory_mode", "gpu_layers"}):
            continue
        nested = [n for stmt in node.body for n in ast.walk(stmt)]
        assert (
            assign not in nested
        ), "_paravirtual_cpu_forced is gated on the requested placement again"


def test_a_user_owned_drafter_is_pinned_to_cpu_too():
    """A user --spec-type makes _build_speculative_flags emit nothing, so their
    --model-draft never appeared in spec_flags and the drafter kept running on the
    corrupt Metal path."""
    user_extras = ["--spec-type", "draft-simple", "--model-draft", "/models/d.gguf"]
    # Studio emits no spec block at all here, which is why spec_flags alone is blind.
    backend = llama_cpp.LlamaCppBackend()
    assert (
        backend._build_speculative_flags(
            speculative_type = "auto",
            spec_draft_n_max = None,
            extra_args = user_extras,
            model_identifier = "owner/repo",
            model_path = None,
            gpus = False,
            binary = None,
        )
        == []
    )
    assert llama_cpp._extra_args_mtp_draft_path([*[], *user_extras]) == "/models/d.gguf"
    # The negative: no drafter anywhere means no pin, so an embedded MTP head (which
    # follows the main --gpu-layers) is not handed a flag it does not need.
    assert llama_cpp._extra_args_mtp_draft_path(["--spec-type", "draft-mtp"], {}) is None

    tree = _load_model_tree()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_extra_args_mtp_draft_path"
    ]
    assert calls, "the drafter CPU pin no longer resolves a drafter path"
    assert any(
        {n.id for n in ast.walk(call) if isinstance(n, ast.Name)} >= {"spec_flags", "extra_args"}
        for call in calls
    ), "the pin still looks at spec_flags only, so a user-owned drafter is missed"


def test_the_drafter_cpu_pin_outlives_the_pass_through_extras():
    """llama.cpp is last-wins and the paravirtual strip only covers the main offload
    flags, so a user -ngld 99 appended after a managed --spec-draft-ngl 0 would put
    the drafter straight back on the corrupt device."""
    assert "-ngld" not in llama_server_args._OFFLOAD_SHADOWING_FLAGS
    assert "--spec-draft-ngl" not in llama_server_args._OFFLOAD_SHADOWING_FLAGS
    # ...and it is not stripped as a spec flag either (the budget parses it).
    assert "-ngld" not in llama_server_args._SPEC_FLAGS
    # A trailing value is what wins, which is the parser behaviour being defended.
    assert (
        llama_cpp._extra_args_draft_offloaded_to_cpu(["--spec-draft-ngl", "0", "-ngld", "99"], {})
        is False
    )

    tree = _load_model_tree()
    extras_at = pin_at = None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "extend":
            continue
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        if "_emit_extra_args" in names:
            extras_at = node.lineno
        if "_pv_draft_cpu_pin" in names:
            pin_at = node.lineno
    assert extras_at is not None and pin_at is not None
    assert pin_at > extras_at, "the drafter CPU pin is emitted before the user extras"


# ── the MTP slot clamp must follow the flags that actually launch ─────


def test_a_stale_mtp_env_does_not_clamp_a_non_mtp_launch():
    """LLAMA_ARG_SPEC_TYPE=draft-mtp with a user --spec-type ngram-mod launches
    non-MTP (extras are appended last and win), so the slots must survive."""
    env = {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"}
    assert llama_cpp._extra_args_requests_mtp(["--spec-type", "ngram-mod"], env) is False
    # The pre-flight clamp reads extras only, so the env cannot reach it.
    assert llama_cpp._extra_args_requests_mtp(None, {}) is False
    src = _load_model_source()
    assert "_extra_args_requests_mtp(extra_args, env = {})" in src


def test_the_backstop_defers_to_a_user_owned_spec_type():
    """_build_speculative_flags emits nothing when the user owns --spec-type, and
    judging that empty list would fall through to the env and clamp anyway."""
    env = {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"}
    assert llama_cpp._extra_args_requests_mtp([], env) is True  # why the guard is needed
    assert llama_cpp._extra_args_set_spec_type(["--spec-type", "ngram-mod"]) is True
    src = _load_model_source()
    assert "not _extra_args_set_spec_type(extra_args)" in src
