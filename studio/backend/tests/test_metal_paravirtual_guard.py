# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A virtualised Apple GPU must fall back to CPU; a real one must not.

The second half matters most: forcing gpu_layers=0 on every Mac would "fix" the corrupt
output by throwing away Metal for every real user, so these tests pin the discrimination
and not just the fallback.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import asdict
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
    """MLX is not on every Mac; without this fallback a virtualised machine without it
    would read as bare metal and emit gibberish."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "mlx", None)  # import raises
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(
            stdout = "Graphics/Displays:\n  Apple Paravirtual device:\n    Vendor: Apple"
        ),
    )
    assert _metal_device_is_paravirtual() is True


def _probe_dispatch(responses):
    """subprocess.run stub keyed on the probe, so SPDisplaysDataType and hw.model can
    answer differently."""

    def run(cmd, *a, **k):
        key = "hw.model" if "sysctl" in cmd[0] else "spdisplays"
        return types.SimpleNamespace(stdout = responses.get(key, ""))

    return run


def test_a_headless_vm_is_caught_when_spdisplays_says_nothing(monkeypatch):
    """The case the OS fallback exists for and used to miss. Measured on macos-14/15:
    SPDisplaysDataType returns zero bytes on a VM with no display, so every cloud and CI
    Mac without MLX read as bare metal and kept the offload that corrupts its output.
    hw.model still names the machine."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        _probe_dispatch({"hw.model": "VirtualMac2,1\n", "spdisplays": ""}),
    )
    assert _metal_device_is_paravirtual() is True


def test_a_physical_mac_is_not_dragged_down_by_the_model_probe(monkeypatch):
    """Negative: real hardware reports Mac<n>,<n>, which must not read as virtual."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        _probe_dispatch(
            {
                "hw.model": "Mac15,3\n",
                "spdisplays": "Graphics/Displays:\n  Apple M3 Max:\n    Vendor: Apple",
            }
        ),
    )
    assert _metal_device_is_paravirtual() is False


def test_a_desktop_vm_still_answers_through_spdisplays(monkeypatch):
    """Negative on the ordering: a VM whose hw.model does not say virtual is still caught
    by the display probe, so adding hw.model did not narrow the net."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "mlx", None)
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        _probe_dispatch(
            {
                "hw.model": "Mac14,2\n",
                "spdisplays": "Graphics/Displays:\n  Apple Paravirtual device:",
            }
        ),
    )
    assert _metal_device_is_paravirtual() is True


def test_mlx_short_circuits_before_any_subprocess(monkeypatch):
    """Negative on cost: MLX answered, so neither probe should spawn. Roughly 40 ms for
    MLX against 300 ms for SPDisplaysDataType."""
    monkeypatch.setattr(sys, "platform", "darwin")
    parent, core = _fake_mlx("Apple Paravirtual device")
    monkeypatch.setitem(sys.modules, "mlx", parent)
    monkeypatch.setitem(sys.modules, "mlx.core", core)

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("spawned a probe after MLX already named the device")

    monkeypatch.setattr("core.inference.llama_cpp.subprocess.run", explode)
    assert _metal_device_is_paravirtual() is True


def test_a_broken_probe_leaves_gpu_offload_alone(monkeypatch):
    """If neither source can answer, assume a real Mac: guessing "virtualised" would
    silently drop everyone the probe fails on to CPU."""
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
    """The launch records the normalized placement ("manual"/0), so a repeat Auto request
    must be normalized before comparison, or every duplicate /load tears down a healthy
    CPU server."""
    src = _load_model_source()
    assert src.index("_metal_device_is_paravirtual()") < src.index(
        "self.adopt_load_intent_if_matched("
    )


def test_repeat_auto_load_does_not_reload_a_healthy_cpu_server(monkeypatch, tmp_path):
    """End-to-end: the second identical Auto /load must take the fast path."""
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
            llama_cpp.GgufLoadIntent(
                model_identifier = "owner/repo",
                gguf_path = str(gguf),
                n_ctx = 8192,
                gpu_memory_mode = "auto",
                gpu_layers = -1,
                n_parallel = 1,
            )
        )
        is True
    )


def test_a_pass_through_layer_flag_cannot_re_enable_the_corrupt_offload(monkeypatch):
    """Auto never strips -ngl at the route, and user extras are appended after the managed
    --gpu-layers 0, so llama.cpp's last-wins parser would undo the pin."""
    _paravirtual(monkeypatch)
    backend = llama_cpp.LlamaCppBackend()
    seen = {}

    def _capture(intent):
        captured = asdict(intent)
        # The intent freezes its sequences into tuples; compare on content.
        for _key in ("extra_args", "tensor_split", "gpu_ids"):
            if captured.get(_key) is not None:
                captured[_key] = list(captured[_key])
        seen.update(captured)
        return True

    monkeypatch.setattr(backend, "adopt_load_intent_if_matched", _capture)
    monkeypatch.setattr(backend, "_apply_detected_audio", lambda _d: True)
    backend._audio_probed = True
    backend._healthy = True
    backend.load_model(
        llama_cpp.GgufLoadIntent(
            model_identifier = "owner/repo",
            gguf_path = "/nonexistent/model.gguf",
            gpu_memory_mode = "auto",
            gpu_layers = -1,
            extra_args = ["-ngl", "99", "--top-k", "40"],
        )
    )
    assert seen["gpu_memory_mode"] == "manual"
    assert seen["gpu_layers"] == 0
    assert seen["extra_args"] == ["--top-k", "40"]


def test_a_real_mac_keeps_its_offload_flags(monkeypatch):
    """The stripping is scoped to the fallback: a physical Mac must keep -ngl."""
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: False)
    backend = llama_cpp.LlamaCppBackend()
    seen = {}

    def _capture(intent):
        captured = asdict(intent)
        # The intent freezes its sequences into tuples; compare on content.
        for _key in ("extra_args", "tensor_split", "gpu_ids"):
            if captured.get(_key) is not None:
                captured[_key] = list(captured[_key])
        seen.update(captured)
        return True

    monkeypatch.setattr(backend, "adopt_load_intent_if_matched", _capture)
    monkeypatch.setattr(backend, "_apply_detected_audio", lambda _d: True)
    backend._audio_probed = True
    backend._healthy = True
    backend.load_model(
        llama_cpp.GgufLoadIntent(
            model_identifier = "owner/repo",
            gguf_path = "/nonexistent/model.gguf",
            gpu_memory_mode = "auto",
            gpu_layers = -1,
            extra_args = ["-ngl", "99"],
        )
    )
    assert seen["gpu_memory_mode"] == "auto"
    assert seen["extra_args"] == ["-ngl", "99"]


def test_gpu_companions_are_pinned_to_cpu_too():
    """--gpu-layers 0 does not reach them: clip.cpp offloads on the mmproj_use_gpu
    boolean (default true) and a separate drafter takes
    params.speculative.draft.n_gpu_layers, default -1 = auto."""
    src = _load_model_source()
    assert '"--no-mmproj-offload"' in src
    # The drafter flag name comes from the probe, never a literal: --spec-draft-ngl only
    # exists from b8955, and an older build exposing only -ngld would refuse to start.
    assert (
        '"--spec-draft-ngl", "0"' not in src
    ), "hardcoding the flag defeats the capability probe on builds that only have -ngld"
    assert "_paravirtual_draft_ngl_flag(server_caps)" in src


@pytest.mark.parametrize(
    "caps, drops, pins",
    [
        # Conclusive probe, flag genuinely absent: drop, as before.
        ({"supports_no_mmproj_offload": False, "mtp_probe_inconclusive": False}, True, False),
        # Conclusive probe, flag present: pin, as before.
        ({"supports_no_mmproj_offload": True, "mtp_probe_inconclusive": False}, False, True),
        # Probe never answered: must NOT drop. --no-mmproj-offload is b5178 and the base
        # argv already needs b6325, so a build that can start here always has it; a false
        # capability means the probe failed, and dropping vision would be self-inflicted.
        ({"supports_no_mmproj_offload": False, "mtp_probe_inconclusive": True}, False, True),
    ],
)
def test_a_failed_probe_does_not_cost_the_user_their_projector(caps, drops, pins):
    mmproj, vision, _warnings = _mmproj_gate(
        paravirtual = True, caps = caps, mmproj = "/m/mmproj-F32.gguf", is_vision = True
    )
    assert (mmproj is None) is drops
    assert (vision is False) is drops
    assert llama_cpp._paravirtual_mmproj_pinnable(caps) is pins


@pytest.mark.parametrize(
    "caps, expected",
    [
        (
            {"spec_draft_ngl_flag": "--spec-draft-ngl", "mtp_probe_inconclusive": False},
            "--spec-draft-ngl",
        ),
        (
            {"spec_draft_ngl_flag": "--gpu-layers-draft", "mtp_probe_inconclusive": False},
            "--gpu-layers-draft",
        ),
        # Unanswered probe falls back to the 2023 spelling rather than reading as
        # unpinnable, so a failed probe costs speculation nothing.
        ({"spec_draft_ngl_flag": None, "mtp_probe_inconclusive": True}, "--gpu-layers-draft"),
        # Conclusive and genuinely absent stays unpinnable.
        ({"spec_draft_ngl_flag": None, "mtp_probe_inconclusive": False}, None),
    ],
)
def test_the_drafter_pin_falls_back_before_it_gives_up(caps, expected):
    assert llama_cpp._paravirtual_draft_ngl_flag(caps) == expected


def test_a_failed_probe_does_not_cost_the_user_their_drafter():
    """The drop half of the same decision: an unanswered probe must not drop."""
    drafter, _extras, _warnings = _drafter_gate(
        paravirtual = True,
        caps = {"spec_draft_ngl_flag": None, "mtp_probe_inconclusive": True},
        drafter = "/m/mtp-model.gguf",
        extra_args = None,
    )
    assert drafter == "/m/mtp-model.gguf"


@pytest.mark.parametrize(
    "extras, kept",
    [
        # A GPU-bound override survives --gpu-layers 0: llama.cpp applies it while picking
        # each weight's buffer type, before any layer is assigned.
        (["-ot", ".*=Metal"], False),
        (["-ot=.*=Metal"], False),
        (["-otd", ".*=Metal"], False),
        (["--override-tensor", "blk.*=Metal"], False),
        (["-ot", "exps=CPU,attn=Metal"], False),
        # Negatives: a CPU target moves weights the same way this fallback does, so
        # stripping it would slow the load it rescues.
        (["-ot", "exps=CPU"], True),
        (["--override-tensor", "blk.*=CPU"], True),
        (["--top-k", "40"], True),
    ],
)
def test_only_gpu_bound_tensor_overrides_are_dropped(extras, kept):
    out = llama_cpp._paravirtual_strip_gpu_overrides(extras)
    assert (out == list(extras)) is kept


def test_the_override_strip_leaves_other_extras_alone():
    """Negative: stripping must not disturb neighbouring arguments."""
    out = llama_cpp._paravirtual_strip_gpu_overrides(
        ["--top-k", "40", "-ot", ".*=Metal", "--temp", "0.7"]
    )
    assert out == ["--top-k", "40", "--temp", "0.7"]


def _load_model_tree() -> ast.AST:
    return ast.parse(textwrap.dedent(_load_model_source()))


def test_the_companion_pins_are_keyed_on_the_hardware_not_the_request():
    """A caller already asking for Manual + 0 layers needs no placement rewrite, so
    skipping the whole block for it left _paravirtual_cpu_forced False and silently
    dropped the mmproj / drafter pins, which do not read --gpu-layers at all."""
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

    # The negative: it must NOT be nested under a test of the requested placement, which
    # is what made a manual CPU load lose the companion pins.
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
    --model-draft never appeared in spec_flags and the drafter kept running corrupt."""
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
    # No drafter anywhere means no pin, so an embedded MTP head (which follows the main
    # --gpu-layers) is not handed a flag it does not need.
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
    """llama.cpp is last-wins and the paravirtual strip covers only the main offload
    flags, so a user -ngld 99 after a managed --spec-draft-ngl 0 would put the drafter
    straight back on the corrupt device."""
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


# ── a projector that cannot be pinned must not be served at all ───────


class _FakeLogger:
    def __init__(self):
        self.warnings: list = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        pass


def _mmproj_gate(
    *,
    paravirtual: bool,
    caps: dict,
    is_vision: bool = True,
    mmproj = "/p.gguf",
):
    """Run load_model's real projector-resolution statements and report what the launch
    would see: (mmproj path, vision flag, warnings)."""
    body = None
    for node in ast.walk(_load_model_tree()):
        stmts = getattr(node, "body", None)
        if not isinstance(stmts, list):
            continue
        starts = [
            i
            for i, s in enumerate(stmts)
            if isinstance(s, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "launch_mmproj_path" for t in s.targets)
        ]
        ends = [
            i
            for i, s in enumerate(stmts)
            if isinstance(s, ast.If) and {"is_vision", "effective_is_vision"} <= _names(s.test)
        ]
        if starts and ends:
            body = stmts[starts[0] : ends[0] + 1]
            break
    assert body is not None, "the projector-resolution block moved out of load_model"
    log = _FakeLogger()
    # Seed from the real module so module-level helpers resolve, rather than re-listing
    # them by hand and breaking when one is added.
    scope = {
        **vars(llama_cpp),
        "extra_args": None,
        "extra_args_disable_mmproj": lambda _a: False,
        "self": types.SimpleNamespace(
            _resolve_launch_mmproj_path = lambda **_kw: mmproj,
        ),
        "model_path": "/m.gguf",
        "mmproj_path": None,
        "_paravirtual_cpu_forced": paravirtual,
        "server_caps": caps,
        "is_vision": is_vision,
        "logger": log,
    }
    exec(ast.unparse(ast.Module(body = body, type_ignores = [])), scope)
    return scope["launch_mmproj_path"], scope["effective_is_vision"], log.warnings


def test_a_projector_that_cannot_be_pinned_is_dropped():
    """clip.cpp never reads --gpu-layers, so without --no-mmproj-offload the vision
    encoder stays on the virtualised device. Serving image embeddings built from corrupt
    output is the failure being prevented, so drop the projector."""
    path, vision, warnings = _mmproj_gate(paravirtual = True, caps = {})
    assert path is None
    assert vision is False
    assert any("--no-mmproj-offload" in w for w in warnings), "the reason must be named"
    assert any("unsloth studio update" in w for w in warnings), "the fix must be named"


def test_dropping_the_projector_does_not_double_warn():
    """The generic "no usable mmproj" line would blame a missing projector for a
    capability gap, so exactly one explanation reaches the user."""
    _path, _vision, warnings = _mmproj_gate(paravirtual = True, caps = {})
    assert len(warnings) == 1, warnings


def test_a_pinnable_projector_survives_on_the_same_hardware():
    """The negative that matters most: a build WITH the flag keeps vision, since
    --no-mmproj-offload puts the encoder on the CPU where it is correct."""
    path, vision, warnings = _mmproj_gate(
        paravirtual = True, caps = {"supports_no_mmproj_offload": True}
    )
    assert path == "/p.gguf"
    assert vision is True
    assert warnings == []


def test_a_real_mac_keeps_vision_on_a_build_without_the_flag():
    """The drop is scoped to the virtualised device: an old llama.cpp on physical Apple
    Silicon (or any non-Mac) must keep its projector on the GPU."""
    path, vision, warnings = _mmproj_gate(paravirtual = False, caps = {})
    assert path == "/p.gguf"
    assert vision is True
    assert warnings == []


def test_a_text_only_gguf_is_unaffected_either_way():
    """No projector to drop, so the guard must not manufacture a warning."""
    for caps in ({}, {"supports_no_mmproj_offload": True}):
        path, vision, warnings = _mmproj_gate(
            paravirtual = True, caps = caps, is_vision = False, mmproj = None
        )
        assert path is None
        assert vision is False
        assert warnings == []


def test_the_drop_precedes_everything_the_projector_feeds():
    """launch_mmproj_path drives the --mmproj flag, the mmproj VRAM budget and the
    audio-encoder probe, so clearing it later would launch a projector the guard believes
    it dropped (or offer audio input that is gone)."""
    src = _load_model_source()
    gate_at = src.index("_pv_mmproj_unpinnable = bool(")
    assert gate_at < src.index('cmd.extend(["--mmproj", launch_mmproj_path])')
    assert gate_at < src.index("self._mmproj_vram_bytes(launch_mmproj_path)")
    # The probe reads launch_mmproj_path, or the env projector only when the gate
    # dropped nothing, so both the gate and that choice precede it.
    assert gate_at < src.index("_audio_probe = launch_mmproj_path or (")
    assert gate_at < src.index("read_mmproj_audio_capability(_audio_probe)")
    # ...and the session flag the frontend reads follows the same variable.
    assert "self._is_vision = effective_is_vision" in src


# ── a drafter that cannot be pinned must not be launched either ──────


def _drafter_gate(
    *,
    paravirtual: bool,
    caps: dict,
    drafter = "/models/mtp-gemma.gguf",
    extra_args = None,
):
    """Run load_model's real unpinnable-drafter statements and report what the launch
    would see: (resolved drafter, extra args, warnings)."""
    body = None
    for node in ast.walk(_load_model_tree()):
        stmts = getattr(node, "body", None)
        if not isinstance(stmts, list):
            continue
        starts = [
            i
            for i, s in enumerate(stmts)
            if isinstance(s, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "_pv_draft_unpinnable" for t in s.targets)
        ]
        ends = [
            i
            for i, s in enumerate(stmts)
            if isinstance(s, ast.If)
            and isinstance(s.test, ast.Name)
            and s.test.id == "_pv_draft_unpinnable"
        ]
        if starts and ends:
            body = stmts[starts[0] : ends[0] + 1]
            break
    assert body is not None, "the unpinnable-drafter block moved out of load_model"
    log = _FakeLogger()
    # Seed from the real module so module-level helpers resolve, rather than re-listing
    # them by hand and breaking when one is added.
    scope = {
        **vars(llama_cpp),
        "_paravirtual_cpu_forced": paravirtual,
        "server_caps": caps,
        "launch_mtp_draft_path": drafter,
        "extra_args": list(extra_args) if extra_args else extra_args,
        "_extra_args_mtp_draft_path": llama_cpp._extra_args_mtp_draft_path,
        "_extra_args_draft_offloaded_to_cpu": llama_cpp._extra_args_draft_offloaded_to_cpu,
        "_extra_args_requests_mtp": llama_cpp._extra_args_requests_mtp,
        "_child_spec_env": llama_cpp._child_spec_env,
        "strip_shadowing_flags": llama_cpp.strip_shadowing_flags,
        # The slot restore rides in this block; 0 = nothing clamped, so it stays out of
        # the way of the drafter cases these helpers cover.
        "_pv_extras_clamped_slots": 0,
        "n_parallel": 1,
        "cmd": ["--parallel", "1"],
        "logger": log,
    }
    exec(ast.unparse(ast.Module(body = body, type_ignores = [])), scope)
    return scope["launch_mtp_draft_path"], scope["extra_args"], log.warnings


@pytest.fixture(autouse = True)
def _no_inherited_draft_env(monkeypatch):
    """The drafter parsers fall back to os.environ, so a stray var on the host must not
    decide these cases."""
    for var in (
        "LLAMA_ARG_SPEC_DRAFT_MODEL",
        "LLAMA_ARG_SPEC_DRAFT_HF_REPO",
        "LLAMA_ARG_N_GPU_LAYERS_DRAFT",
    ):
        monkeypatch.delenv(var, raising = False)


def test_a_drafter_that_cannot_be_pinned_is_dropped(monkeypatch):
    """A separate drafter takes params.speculative.draft.n_gpu_layers (default -1 = auto),
    not the main --gpu-layers, so without a draft-layer flag it runs full-offload on the
    virtualised device."""
    drafter, extras, warnings = _drafter_gate(paravirtual = True, caps = {})
    assert drafter is None
    assert any("draft-layer flag" in w for w in warnings), "the reason must be named"
    assert any("unsloth studio update" in w for w in warnings), "the fix must be named"
    # Dropping speculation cannot change a single emitted token:
    # common_sampler_sample_and_accept_n pushes the TARGET model's own draw and stops at
    # the first draft mismatch, so the warning must not claim the output was wrong.
    assert not any("corrupt output" in w for w in warnings), warnings
    assert any("only costs speed" in w for w in warnings), warnings
    # An env-only drafter needs no drop: with no extras owning --spec-type the launch
    # scrubs LLAMA_ARG_SPEC_DRAFT_*, so it never loads, and dropping anyway would strip
    # the caller's own speculative tuning for nothing.
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_MODEL", "/models/env.gguf")
    tuning = ["--spec-draft-n-max", "6"]
    drafter, extras, warnings = _drafter_gate(
        paravirtual = True, caps = {}, drafter = None, extra_args = tuning
    )
    assert warnings == []
    assert extras == tuning
    # But an env drafter the extras DO keep alive still drops: their --spec-type is what
    # stops the scrub.
    owned = ["--spec-type", "draft-simple"]
    drafter, _extras, warnings = _drafter_gate(
        paravirtual = True, caps = {}, drafter = None, extra_args = owned
    )
    assert any("draft-layer flag" in w for w in warnings), warnings


def test_the_drop_takes_a_user_owned_drafter_with_it():
    """A user --spec-type makes _build_speculative_flags emit nothing, so clearing only
    Unsloth's resolved path would leave their --model-draft on the device."""
    extras = ["--spec-type", "draft-simple", "--model-draft", "/models/d.gguf", "--top-k", "40"]
    drafter, out, warnings = _drafter_gate(
        paravirtual = True, caps = {}, drafter = None, extra_args = extras
    )
    assert drafter is None
    assert warnings
    assert llama_cpp._extra_args_mtp_draft_path(out, {}) is None
    # The spec type leaves with the model it needs, or a kept draft-simple would reach a
    # llama-server with no draft model to serve it.
    assert "--spec-type" not in out and "draft-simple" not in out
    # ...and nothing else is touched.
    assert out == ["--top-k", "40"]


def test_a_drafter_free_spec_mode_keeps_the_speculation_it_asked_for():
    """A sibling mtp-*.gguf on disk is not a drafter the launch loads: the user owns
    --spec-type, so _build_speculative_flags returns before emitting --model-draft for
    it, and n-gram speculation loads no model of its own. Dropping here would cost the
    user the very mode they asked for (--spec-type and the ngram knobs are one strip
    group) to protect a drafter that was never going to launch."""
    extras = [
        "--spec-type",
        "ngram-mod",
        "--spec-ngram-mod-n-max",
        "5",
        "--spec-ngram-mod-n-min",
        "2",
        "--top-k",
        "40",
    ]
    drafter, out, warnings = _drafter_gate(paravirtual = True, caps = {}, extra_args = extras)
    assert drafter == "/models/mtp-gemma.gguf"
    assert out == extras
    assert warnings == []


def test_a_sibling_drafter_still_goes_when_unsloth_owns_the_spec_block():
    """The negative for the case above: with no user --spec-type the resolver may emit
    --model-draft for the sibling, so it is a real placement and still drops."""
    drafter, out, warnings = _drafter_gate(paravirtual = True, caps = {}, extra_args = ["--top-k", "40"])
    assert drafter is None
    assert out == ["--top-k", "40"]
    assert warnings


def test_a_drafter_requiring_mode_still_loses_its_drafter():
    """draft-simple/draft-eagle3 name a separate model the child really loads, so a
    user-owned spec type is no exemption: the drafter and the type that needs it both
    go."""
    for mode in ("draft-simple", "draft-eagle3"):
        extras = ["--spec-type", mode, "--model-draft", "/models/d.gguf", "--top-k", "40"]
        drafter, out, warnings = _drafter_gate(paravirtual = True, caps = {}, extra_args = extras)
        assert drafter is None
        assert out == ["--top-k", "40"]
        assert warnings


def test_a_drafter_free_mode_that_names_its_own_drafter_still_drops_it():
    """llama.cpp loads the draft model whenever its path is set, so a --model-draft
    passed alongside ngram-mod still lands on the corrupt device. The exemption is for a
    sibling the launch never emits, not for an explicit drafter."""
    extras = ["--spec-type", "ngram-mod", "--model-draft", "/models/d.gguf"]
    drafter, out, warnings = _drafter_gate(paravirtual = True, caps = {}, extra_args = extras)
    assert drafter is None
    assert llama_cpp._extra_args_mtp_draft_path(out, {}) is None
    assert warnings


def test_an_inherited_drafter_env_is_not_exempted_by_a_drafter_free_mode(monkeypatch):
    """Same through the env the child reads directly: the drafter loads whatever
    --spec-type says, so it cannot ride out the drop on the mode alone."""
    monkeypatch.setenv("LLAMA_ARG_SPEC_DRAFT_MODEL", "/models/env.gguf")
    drafter, _out, warnings = _drafter_gate(
        paravirtual = True,
        caps = {},
        drafter = None,
        extra_args = ["--spec-type", "ngram-mod"],
    )
    assert drafter is None
    assert warnings


def test_a_real_mac_keeps_the_sibling_and_the_mode_alike():
    """The exemption changes nothing off the virtualised device: physical Apple Silicon
    keeps both halves."""
    extras = ["--spec-type", "ngram-mod", "--model-draft", "/models/d.gguf"]
    drafter, out, warnings = _drafter_gate(paravirtual = False, caps = {}, extra_args = extras)
    assert drafter == "/models/mtp-gemma.gguf"
    assert out == extras
    assert warnings == []


def test_the_env_the_child_inherits_is_dropped_too():
    """argv cannot un-set LLAMA_ARG_SPEC_DRAFT_MODEL: llama.cpp reads it directly, and
    appends spec types rather than replacing them, so an inherited draft-simple would
    outlive the model just removed. The same scrub covers every Unsloth-owned spec
    block."""
    src = _load_model_source()
    for var in (
        "LLAMA_ARG_SPEC_DRAFT_MODEL",
        "LLAMA_ARG_SPEC_DRAFT_HF_REPO",
        "LLAMA_ARG_SPEC_TYPE",
        # Pre-b8955 spellings, live between the launchable floor and the rename; without
        # them the drop leaves the drafter in the child env.
        "LLAMA_ARG_MODEL_DRAFT",
        "LLAMA_ARG_HFD_REPO",
    ):
        assert (
            var in llama_cpp._SPEC_ENV_VARS
        ), f"{var} survives the drafter drop into the child env"
    gate_at = src.index("for _pv_spec_var in _SPEC_ENV_VARS")
    assert "env.pop(_pv_spec_var, None)" in src[gate_at : gate_at + 200]


def test_a_managed_spec_block_clears_the_inherited_spec_env():
    """Nothing Unsloth emits can undo an inherited LLAMA_ARG_SPEC_TYPE: llama.cpp applies
    the env first and appends. So a managed non-MTP launch would still run MTP, a
    crash-recovery replay could not drop it, and the fit never budgeted the drafter the
    env adds; the launch clears it instead. Extras that own --spec-type keep theirs,
    since there the two genuinely accumulate."""
    src = _load_model_source()
    at = src.index("for _pv_spec_var in _SPEC_ENV_VARS")
    gate = src[src.rindex("if ", 0, at) : at]
    assert "not _extra_args_set_spec_type(extra_args)" in gate
    assert "_pv_draft_unpinnable" in gate


def test_the_drafter_drop_hands_back_the_slots_it_no_longer_needs():
    """The extras-MTP clamp cuts a multi-slot request to one. If the drop then strips
    that very spec group the server is not speculating at all, and would serve one chat
    at a time forever: the dedupe records the original ask, so no Apply restores it."""
    src = _load_model_source()
    clamp_at = src.index("_pv_extras_clamped_slots = n_parallel")
    restore_at = src.index("n_parallel = _pv_extras_clamped_slots")
    assert clamp_at < restore_at
    # Restored before the spec flags are rebuilt, so the backstop can re-clamp if
    # Unsloth's own resolution turns out to be MTP.
    assert restore_at < src.index("spec_flags = self._build_speculative_flags(")
    # And it only fires once the stripped extras really are non-MTP.
    assert "not _extra_args_requests_mtp(" in src[restore_at - 400 : restore_at]


def test_the_training_guard_sizes_the_cpu_pin_not_the_raw_request():
    """load_model rewrites a GGUF placement to CPU here, so a guard sizing the raw Auto
    request could refuse a chat load for VRAM it never takes."""
    guard_src = inspect.getsource(_routes()._guard_chat_load_against_training)
    assert "_metal_device_is_paravirtual()" in guard_src
    assert "paravirtual_normalized_request(" in guard_src
    # Before the manual/GGUF early return, or the rewrite could not reach it.
    assert guard_src.index("paravirtual_normalized_request(") < guard_src.index(
        'gpu_memory_mode == "manual"'
    )


def test_a_diffusion_split_that_cannot_be_pinned_is_refused():
    """An older shim without --ngl drops the zero-layer split, and nothing else keeps the
    diffusion runner off Metal: cpu_only is torch.cuda only, so it reads 0 on a Mac and
    the empty --gpu token still leaves Metal available. Refuse rather than serve output
    that may be corrupt."""
    src = inspect.getsource(llama_cpp.LlamaCppBackend._start_diffusion_server)
    drop_at = src.index("not _shim_supports_ngl(shim_cmd)")
    guard = src[drop_at : drop_at + 700]
    assert "_metal_device_is_paravirtual()" in guard
    assert "_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR" in guard
    # Only the zero-layer pin: a non-zero manual split is the user's own placement.
    assert "manual_ngl == 0" in guard
    # And it is raised before the warning that would otherwise carry on.
    assert guard.index("raise ValueError") < guard.index("logger.warning")
    # The message has to say how to get out of it.
    msg = llama_cpp._PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR
    assert "unsloth studio update" in msg
    assert "UNSLOTH_ALLOW_PARAVIRTUAL_METAL=1" in msg
    # Settled above the teardown, so a refusal leaves the running server alone.
    load_src = _load_model_source()
    assert load_src.index("_PARAVIRTUAL_DIFFUSION_NO_NGL_ERROR") < load_src.index(
        "self._kill_process()"
    )


def test_the_hf_diffusion_refusal_also_lands_above_the_teardown():
    """The local path settled this before Phase 1, but an HF load has no gguf_path there,
    so the refusal used to fire only after the healthy server was killed and the model
    downloaded. The shim probe is cheap, so it gates the preflight too."""
    src = _load_model_source()
    probe_at = src.index("_pv_diffusion_unpinnable = bool(")
    kill_at = src.index("self._kill_process()")
    assert probe_at < kill_at
    # The HF branch raises from the preflight classification, above the teardown.
    hf_raise = src.index("if _pv_diffusion_unpinnable:\n                        raise ValueError(")
    assert hf_raise < kill_at
    # And the preflight download is reached for this case at all.
    assert "or _pv_diffusion_unpinnable" in src[: src.index("_preflight_is_diffusion =")]


def test_the_crash_replay_keeps_the_extras_the_caller_still_sends():
    """The replay launches a stripped list but the caller keeps sending the original, so
    a comparator seeing only the stripped one would reload the MTP setup that crashed."""
    src = inspect.getsource(llama_cpp.LlamaCppBackend._maybe_recover_from_mtp_crash)
    load_at = src.index("self.load_model(fallback)")
    restore_at = src.index("self._requested_extra_args = (")
    assert load_at < restore_at
    # Only when the strip actually rewrote the list.
    assert "if fallback_extra_args is not snapshot.extra_args:" in src
    # Device-stripped the same way the launch records it, so both sides match.
    assert "self._strip_device_extra_args(_ea)" in src


def test_a_pinnable_drafter_survives_on_the_same_hardware():
    """The negative that matters most: a build WITH a draft-layer flag keeps its drafter,
    because --spec-draft-ngl 0 puts it on the CPU where it is correct."""
    extras = ["--model-draft", "/models/d.gguf"]
    drafter, out, warnings = _drafter_gate(
        paravirtual = True,
        caps = {"spec_draft_ngl_flag": "--spec-draft-ngl"},
        extra_args = extras,
    )
    assert drafter == "/models/mtp-gemma.gguf"
    assert out == extras
    assert warnings == []
    # The legacy alias is a supported build too, so it must not lose its drafter.
    drafter, out, warnings = _drafter_gate(
        paravirtual = True,
        caps = {"spec_draft_ngl_flag": "--gpu-layers-draft"},
        extra_args = extras,
    )
    assert drafter == "/models/mtp-gemma.gguf"
    assert out == extras
    assert warnings == []


@pytest.mark.parametrize(
    "pin",
    [
        ["--spec-draft-ngl", "0"],
        ["-ngld", "0"],
        ["--gpu-layers-draft=0"],
        ["--spec-draft-device", "cpu"],
        ["--device-draft", "none"],
    ],
)
def test_a_drafter_the_user_already_pinned_is_left_alone(pin):
    """The probe only decides whether Unsloth can emit the flag. A user who passed one
    themselves already has the drafter on the CPU, so dropping it would cost speed for
    nothing."""
    extras = [*pin, "--model-draft", "/models/d.gguf"]
    drafter, out, warnings = _drafter_gate(paravirtual = True, caps = {}, extra_args = extras)
    assert drafter == "/models/mtp-gemma.gguf"
    assert out == extras
    assert warnings == []
    # ...and last-wins still decides: a pin the user overrode is no pin at all.
    overridden = ["--spec-draft-ngl", "0", "-ngld", "99", "--model-draft", "/models/d.gguf"]
    drafter, _out, warnings = _drafter_gate(paravirtual = True, caps = {}, extra_args = overridden)
    assert drafter is None
    assert warnings


def test_a_real_mac_keeps_its_drafter():
    """The drop is scoped to the virtualised device: physical Apple Silicon (or any
    non-Mac) keeps the drafter on the GPU."""
    extras = ["--spec-type", "draft-simple", "--model-draft", "/models/d.gguf"]
    drafter, out, warnings = _drafter_gate(paravirtual = False, caps = {}, extra_args = extras)
    assert drafter == "/models/mtp-gemma.gguf"
    assert out == extras
    assert warnings == []


def test_a_load_with_no_separate_drafter_is_unaffected():
    """Nothing to drop, so no manufactured warning and no stripping the spec group from a
    load that never had a draft model. An embedded MTP head is exactly that case: with no
    draft model llama.cpp skips the n_gpu_layers override, so the head already follows
    --gpu-layers 0."""
    for caps in ({}, {"spec_draft_ngl_flag": "--spec-draft-ngl"}):
        extras = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
        drafter, out, warnings = _drafter_gate(
            paravirtual = True, caps = caps, drafter = None, extra_args = extras
        )
        assert drafter is None
        assert out == extras
        assert warnings == []
    # ...and neither is a plain text load with no extras at all.
    drafter, out, warnings = _drafter_gate(paravirtual = True, caps = {}, drafter = None)
    assert (drafter, out, warnings) == (None, None, [])


def test_mtp_detection_reads_every_accumulated_type():
    """llama.cpp inserts each --spec-type rather than replacing, and applies the env
    first, so MTP is on if ANY source names it. Reading only the last left the slot clamp
    off for a launch that really does run MTP."""
    f = llama_cpp._extra_args_requests_mtp
    env = {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"}
    assert f(["--spec-type", "ngram-mod"], env) is True
    assert f(["--spec-type", "draft-mtp", "--spec-type", "ngram-mod"], {}) is True
    assert f(["--spec_type=draft-mtp"], {}) is True
    assert f([], env) is True
    # Negatives: nothing names MTP, so nothing clamps.
    assert f(None, {}) is False
    assert f(["--spec-type", "ngram-mod"], {}) is False
    assert f(["--spec-default"], {}) is False


def test_a_diffusion_load_drops_the_drafter_state_it_inherits():
    """The diffusion path returns before the assignment that records the drafter, and
    only unload clears it, so a drafter from the previous load would linger and the dedupe
    would reload a healthy diffusion server."""
    src = _load_model_source()
    at = src.index("self._layer_preserves_tensor_intent = False")
    tail = src[at : at + 600]
    assert "self._mtp_draft_path = None" in tail
    assert "self._mtp_draft_suppressed_path = None" in tail
    assert at < src.index("return self._start_diffusion_server(")


def test_the_drafter_pin_covers_the_device_not_just_the_layers():
    """common_base_params_to_speculative replaces the draft context's device list with the
    draft one, so the main --device none never reaches it and an empty draft list leaves
    every device visible. The layer count alone would leave the drafter on the corrupt
    device, exactly as --gpu-layers 0 did."""
    src = _load_model_source()
    pin_at = src.index("_pv_draft_cpu_pin = [")
    pin = src[pin_at : src.index("]", pin_at)]
    assert "--device-draft" in pin
    assert '"none"' in pin
    # And the accounting must read that back as a CPU drafter, not a GPU one.
    cmd = ["llama-server", "-md", "d.gguf", "--gpu-layers-draft", "0", "--device-draft", "none"]
    assert llama_cpp._extra_args_draft_offloaded_to_cpu(cmd) is True
    assert llama_cpp.LlamaCppBackend._cmd_has_gpu_companion(cmd, {}) is False


def test_the_drafter_drop_precedes_the_flags_it_feeds():
    """launch_mtp_draft_path becomes --model-draft and extra_args the pass-through tail,
    so clearing either after _build_speculative_flags ran would launch a drafter the
    guard believes it dropped."""
    src = _load_model_source()
    gate_at = src.index("_pv_draft_unpinnable = bool(")
    assert gate_at < src.index("spec_flags = self._build_speculative_flags(")
    assert gate_at < src.index("cmd.extend(str(a) for a in _emit_extra_args)")
    # The CPU pin is the other half of the same decision, reachable only on a build that
    # advertises a flag to pin with.
    assert gate_at < src.index("_pv_draft_cpu_pin = [")


# ── a pass-through GPU split mode must not fail the CPU-only load ────


@pytest.mark.parametrize(
    "extra_args, expected",
    [
        # `-sm row` throws "device <X> does not support split buffers" in
        # make_gpu_buft_list for every backend except SYCL, and that runs over
        # model->devices before any layer is assigned, so --gpu-layers 0 does not save it.
        # Metal is always in that list: the zero-offload mask writes only CUDA/HIP vars.
        (["-sm", "row"], ["--split-mode", "layer"]),
        (["--split-mode", "row"], ["--split-mode", "layer"]),
        # `-sm tensor` throws "not implemented for architecture" for every arch
        # llm_arch_supports_sm_tensor excludes, also independently of ngl.
        (["--top-k", "40", "-sm", "tensor"], ["--split-mode", "layer"]),
        (["-sm", "none"], ["--split-mode", "layer"]),
        (["-sm=row"], ["--split-mode", "layer"]),
        # Negatives: nothing to neutralise means no flag, so no redundant argument.
        (None, []),
        ([], []),
        (["--top-k", "40"], []),
        (["-sm", "layer"], []),
        (["--split-mode", "LAYER"], []),
        # Last-wins, like the parser: a user who already ends on layer is fine.
        (["-sm", "row", "--split-mode", "layer"], []),
        # ...and one who ends on row is not, however it started.
        (["--split-mode", "layer", "-sm", "row"], ["--split-mode", "layer"]),
        # --tensor-split is genuinely inert at --gpu-layers 0: llama-model.cpp computes
        # the split points but never indexes them once act_gpu_layers is 0.
        (["-ts", "3,1"], []),
        (["--tensor-split", "3,1"], []),
    ],
)
def test_a_pass_through_split_mode_cannot_fail_the_cpu_only_load(extra_args, expected):
    assert llama_cpp._paravirtual_split_mode_pin(extra_args) == expected


def test_the_split_mode_override_is_scoped_to_the_virtualised_mac():
    """The strongest negative: a real Mac (or any non-Mac) must keep the user's row/none
    split, so the call site is gated on the detector, not on the zero-layer placement a
    plain Manual load also has."""
    tree = _load_model_tree()
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_paravirtual_split_mode_pin"
    ]
    assert len(calls) == 1, "expected exactly one split-mode override call site"
    guarded = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        # Direct body only: an unrelated enclosing `if` must not read as the guard.
        if not any(calls[0] in ast.walk(stmt) for stmt in node.body):
            continue
        assert {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)} == {
            "_paravirtual_cpu_forced"
        }, "the split-mode override is no longer gated on the hardware alone"
        guarded = True
    assert guarded, "the split-mode override lost its _paravirtual_cpu_forced guard"


def test_the_split_mode_override_outlives_the_pass_through_extras():
    """llama.cpp is last-wins, so an override emitted before the user's extras would be
    undone by the very flag it exists to neutralise."""
    # Overridden, not stripped: the route's comparator compares stored extras verbatim
    # and the UI does not round-trip the extras box, so a strip would make every later
    # Apply a real model swap.
    assert "-sm" not in llama_server_args._OFFLOAD_SHADOWING_FLAGS
    assert "--split-mode" not in llama_server_args._OFFLOAD_SHADOWING_FLAGS
    # The parser behaviour being relied on.
    assert llama_server_args.resolve_tensor_parallel(["-sm", "tensor"], False) is True
    assert (
        llama_server_args.resolve_tensor_parallel(["-sm", "tensor", "--split-mode", "layer"], False)
        is False
    ), "the override must also clear the tensor state the zero-VRAM mask reads"

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
        if "_pv_split_mode_pin" in names:
            pin_at = node.lineno
    assert extras_at is not None and pin_at is not None
    assert pin_at > extras_at, "the split-mode override is emitted before the user extras"


# ── the MTP slot clamp must follow the flags that actually launch ─────


def test_the_clamp_judges_the_env_the_child_will_actually_get():
    """An inherited draft-mtp does launch MTP, but the launch scrubs it whenever Unsloth
    owns the spec block, so the slots must survive rather than clamp for a server that
    will not run MTP. The env counts only when the extras own --spec-type, the one case
    it reaches the child."""
    env = {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"}
    # Managed block: scrubbed, so nothing to clamp for.
    assert llama_cpp._child_spec_env([]) == {}
    assert llama_cpp._extra_args_requests_mtp([], llama_cpp._child_spec_env([])) is False
    # Extras own the spec type: their flags and the env accumulate and both launch.
    extras = ["--spec-type", "ngram-mod"]
    assert (
        llama_cpp._extra_args_requests_mtp(extras, llama_cpp._child_spec_env(extras, env)) is True
    )
    src = _load_model_source()
    assert "_child_spec_env(extra_args)" in src
    assert "_extra_args_requests_mtp(extra_args, env = {})" not in src


def test_the_training_guard_does_not_shrink_itself_for_mtp():
    """The launch clamps MTP to one slot, but the guard's estimate counts only the drafter
    file and the main KV: not the draft KV, the duplicated target context under MLA, or
    the draft compute reserve. Sizing for one slot would drop the slot KV without adding
    those back, and a guard that under-sizes evicts the training run it protects; the
    unclamped slots stand in for the difference."""
    route_src = inspect.getsource(_routes())
    assert "_extra_args_requests_mtp(llama_extra_args" not in route_src
    # The diffusion and kv-unified clamps stay: neither drops a modelled term.
    assert "if diffusion_kind is True:" in route_src
    assert 'caps.get("supports_kv_unified")' in route_src


def test_the_backstop_defers_to_a_user_owned_spec_type():
    """_build_speculative_flags emits nothing when the user owns --spec-type, and judging
    that empty list would fall through to the env and clamp anyway."""
    env = {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"}
    assert llama_cpp._extra_args_requests_mtp([], env) is True  # why the guard is needed
    assert llama_cpp._extra_args_set_spec_type(["--spec-type", "ngram-mod"]) is True
    src = _load_model_source()
    assert "not _extra_args_set_spec_type(extra_args)" in src


# ── ...and give the slots back when MTP never launches ────────────────


def _if_block(predicate, tree = None):
    """The one `if` statement in load_model whose test satisfies `predicate`."""
    found = [
        node
        for node in ast.walk(tree if tree is not None else _load_model_tree())
        if isinstance(node, ast.If) and predicate(node.test)
    ]
    assert len(found) == 1, f"expected exactly one matching block, found {len(found)}"
    return found[0]


def _names(node) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _is_mtp_clamp(test) -> bool:
    return "spec_flags" in _names(test) and "n_parallel" in _names(test)


def _is_slot_restore(test) -> bool:
    return "_mtp_clamped_slots" in _names(test)


def _is_retry_spec_strip(test) -> bool:
    return "_launch_spec_env" in _names(test)


def _run_clamp_then_fallback(
    *,
    n_parallel,
    extra_args,
    spec_flags,
    cmd,
    asked_for = None,
    n_batch = None,
):
    """Run the real clamp block, rebuild fallback_cmd the way the MTP retry does, then run
    the real restore block. Returns the two argvs and the slot count
    _commit_effective_parallel_slots would receive."""
    scope = {
        "n_parallel": n_parallel,
        "extra_args": extra_args,
        "spec_flags": spec_flags,
        "cmd": list(cmd),
        "_mtp_clamped_slots": 0,
        "model_identifier": "owner/repo",
        "logger": llama_cpp.logger,
        "n_batch": n_batch,
        "_extra_args_set_spec_type": llama_cpp._extra_args_set_spec_type,
        "_extra_args_requests_mtp": llama_cpp._extra_args_requests_mtp,
        "_child_spec_env": llama_cpp._child_spec_env,
        "_repatch_parallel_slots": llama_cpp._repatch_parallel_slots,
        # The pre-fit ask, which the restore must NOT reach for.
        "_pending_load_kwargs": {"n_parallel": n_parallel if asked_for is None else asked_for},
    }
    exec(ast.unparse(_if_block(_is_mtp_clamp)), scope)
    clamped = list(scope["cmd"])
    # The retry swaps the spec slice for --spec-default; it sits at this argv's tail.
    spec_at = len(clamped) - len(spec_flags)
    scope["fallback_cmd"] = clamped[:spec_at] + ["--spec-default"]
    exec(ast.unparse(_if_block(_is_slot_restore)), scope)
    return clamped, scope["fallback_cmd"], scope["n_parallel"]


def _cmd(slots: int, spec_flags: list) -> list:
    return ["llama-server", "-m", "/m.gguf", "--parallel", str(slots), "--kv-unified", *spec_flags]


def test_the_mtp_fallback_gets_the_requested_slots_back():
    """MTP aborts at startup and the retry drops speculative decoding entirely, so that
    server must not inherit the single slot MTP needed. The KV fit was sized for the full
    count."""
    spec = ["--spec-type", "draft-mtp"]
    clamped, fallback, slots = _run_clamp_then_fallback(
        n_parallel = 4,
        extra_args = ["--top-k", "40"],
        spec_flags = spec,
        cmd = _cmd(4, spec),
    )
    assert clamped[clamped.index("--parallel") + 1] == "1"  # MTP itself still gets one
    assert fallback[fallback.index("--parallel") + 1] == "4"
    # _commit_effective_parallel_slots reads this, and /status echoes it.
    assert slots == 4


def test_the_fallback_raises_the_batch_flag_with_the_slots():
    """--batch-size is emitted from the slot count as it stands then, and llama-server
    aborts below it, so handing 4 slots back to an argv carrying an explicit -b 2 would
    abort the very retry the restore exists to make work."""
    spec = ["--spec-type", "draft-mtp"]
    cmd = ["llama-server", "-m", "/m.gguf", "--parallel", "4", "--batch-size", "2", *spec]
    clamped, fallback, slots = _run_clamp_then_fallback(
        n_parallel = 4,
        extra_args = ["--top-k", "40"],
        spec_flags = spec,
        cmd = cmd,
        n_batch = 2,
    )
    assert fallback[fallback.index("--parallel") + 1] == "4"
    assert fallback[fallback.index("--batch-size") + 1] == "4"
    assert slots == 4
    # the clamped argv keeps what it launched with: one slot needs no raise
    assert clamped[clamped.index("--batch-size") + 1] == "2"


def test_the_mtp_backstop_lowers_the_batch_with_the_slots():
    """The floor is the SMALLEST legal batch, not merely a sufficient one. This clamp runs
    after --batch-size is emitted, so dropping to one slot has to undo the raise the old
    count forced: -b 1 with 64 slots emits 64, and leaving it there launched MTP at a
    64-token micro-batch for a request of 1 (a much larger compute buffer), while the
    recorded micro-batch, derived from the clamped count, said 2."""
    spec = ["--spec-type", "draft-mtp"]
    cmd = ["llama-server", "-m", "/m.gguf", "--parallel", "64", "--batch-size", "64", *spec]
    clamped, _fallback, _slots = _run_clamp_then_fallback(
        n_parallel = 64,
        extra_args = ["--top-k", "40"],
        spec_flags = spec,
        cmd = cmd,
        n_batch = 1,
    )
    assert clamped[clamped.index("--parallel") + 1] == "1"
    # max(1, max(2, 1)) = 2, which is what _ubatch_for_slots(1) records
    assert clamped[clamped.index("--batch-size") + 1] == "2"


def test_the_startup_retry_drops_the_mtp_the_extras_and_the_env_carry():
    """A trailing --spec-default cannot override MTP or DSpark that extras or the env
    supplied: llama.cpp applies the env first and appends types rather than replacing
    them, so the retry would relaunch the mode that just failed and lose a main model
    that loads fine without it. It strips the spec group, and takes the child env with
    it."""
    src = _load_model_source()
    retry = src[
        src.index("_fb_tail = cmd[_spec_start") : src.index("fallback_cmd = cmd[:_spec_start]")
    ]
    # Whitespace-insensitive: the guard wraps across lines once both drafters are named.
    compact = "".join(retry.split())
    assert "_extra_args_requests_mtp(extra_args,env=_launch_spec_env)" in compact
    assert "_extra_args_requests_dspark(extra_args,env=_launch_spec_env)" in compact
    assert "strip_spec = True" in retry
    assert "env.pop(_fb_spec_var, None)" in retry
    # ...and that strip really removes the group rather than shadowing it.
    assert llama_cpp.strip_shadowing_flags(
        ["--spec-type", "draft-mtp", "--top-k", "40"],
        strip_context = False,
        strip_cache = False,
        strip_spec = True,
        strip_template = False,
        strip_split_mode = False,
    ) == ["--top-k", "40"]


def test_the_retry_only_hands_back_slots_no_gpu_had_to_admit():
    """The extras clamp cuts n_parallel to 1 before the GPU slot fit, whose gate needs >1,
    so on a GPU box that count is never admitted and restoring it would size buffers
    nothing approved. Off GPU there is nothing to budget, so the slots come back."""

    def _restored(detected_gpus):
        scope = {
            "cmd": ["llama-server", "--parallel", "1", "--spec-type", "draft-mtp"],
            "_spec_start": 3,
            "spec_flags": [],
            "_fb_tail": ["--spec-type", "draft-mtp"],
            "extra_args": ["--spec-type", "draft-mtp"],
            "_launch_spec_env": {},
            "env": {},
            "n_parallel": 1,
            "_pv_extras_clamped_slots": 4,
            "_mtp_clamped_slots": 0,
            "_detected_gpus": detected_gpus,
            "_extra_args_requests_mtp": llama_cpp._extra_args_requests_mtp,
            "strip_shadowing_flags": llama_cpp.strip_shadowing_flags,
            "_SPEC_ENV_VARS": llama_cpp._SPEC_ENV_VARS,
            "logger": llama_cpp.logger,
        }
        exec(ast.unparse(_if_block(_is_retry_spec_strip)), scope)
        return scope["_mtp_clamped_slots"]

    assert _restored([]) == 4, "a CPU launch must get its slots back"
    assert _restored([(0, 8 << 30)]) == 0, "a GPU launch must not restore unbudgeted slots"


def test_the_startup_fallback_records_the_extras_it_actually_launched():
    """The success path stores extra_args as the launched list, but the fallback launched
    without the spec group. Leaving the MTP list there means the next Apply that omits
    extras inherits it and repeats the crash, so only the requested state keeps it."""
    src = _load_model_source()
    strip_at = src.index("_fb_stripped_extras = strip_shadowing_flags(")
    swap_at = src.index("extra_args = _fb_stripped_extras")
    record_at = src.index("self._extra_args = (")
    assert strip_at < swap_at < record_at, "the swap must land before the launch is recorded"
    # The original survives as the requested state, which the comparators read.
    kept = src[src.index("_pv_suppressed_spec_extra_args = (", swap_at - 400) : swap_at]
    assert "list(extra_args or [])" in kept
    # Only on a healthy retry: a failed one must not rewrite what was asked for.
    assert src.index("_mtp_active_for_launched_server = False", 0, swap_at) < swap_at


def test_the_extras_own_clamp_comes_back_from_the_startup_retry_too():
    """Extras owning --spec-type park the displaced slots in _pv_extras_clamped_slots and
    leave _mtp_clamped_slots at 0. The retry strips that MTP for real now, so a restore
    reading only _mtp_clamped_slots would leave --parallel 1 forever: _requested_n_parallel
    still holds the original ask, so every later identical load dedupes onto it."""
    scope = {
        "cmd": ["llama-server", "--parallel", "1", "--spec-type", "draft-mtp"],
        "_spec_start": 3,
        "spec_flags": [],
        "_fb_tail": ["--spec-type", "draft-mtp"],
        "extra_args": ["--spec-type", "draft-mtp"],
        "_launch_spec_env": {},
        "env": {"LLAMA_ARG_SPEC_TYPE": "draft-mtp"},
        "n_parallel": 1,
        # What the extras clamp displaced, and what the Unsloth-resolved clamp holds.
        "_pv_extras_clamped_slots": 4,
        "_mtp_clamped_slots": 0,
        "_detected_gpus": [],  # CPU launch: nothing had to budget the slots
        "n_batch": None,
        "_extra_args_requests_mtp": llama_cpp._extra_args_requests_mtp,
        "_repatch_parallel_slots": llama_cpp._repatch_parallel_slots,
        "strip_shadowing_flags": llama_cpp.strip_shadowing_flags,
        "_SPEC_ENV_VARS": llama_cpp._SPEC_ENV_VARS,
        "logger": llama_cpp.logger,
    }
    exec(ast.unparse(_if_block(_is_retry_spec_strip)), scope)
    assert scope["env"] == {}, "the child kept the spec env the retry dropped"
    scope["fallback_cmd"] = ["llama-server", "--parallel", "1", "--spec-default"]
    exec(ast.unparse(_if_block(_is_slot_restore)), scope)
    assert scope["fallback_cmd"][scope["fallback_cmd"].index("--parallel") + 1] == "4"
    assert scope["n_parallel"] == 4


def test_the_fallback_gets_back_what_the_fit_sized_not_what_the_user_asked():
    """_slots_that_fit_on_gpu can already have cut the count before the clamp, so restoring
    the raw ask would launch a server the VRAM budget never covered."""
    spec = ["--spec-type", "draft-mtp"]
    _, fallback, slots = _run_clamp_then_fallback(
        n_parallel = 4,
        extra_args = None,
        spec_flags = spec,
        cmd = _cmd(4, spec),
        asked_for = 8,
    )
    assert fallback[fallback.index("--parallel") + 1] == "4"
    assert slots == 4


def test_a_single_slot_mtp_load_stays_single_slot_on_the_fallback():
    """The negative that matters most: nothing was clamped, so nothing is owed."""
    spec = ["--spec-type", "draft-mtp"]
    clamped, fallback, slots = _run_clamp_then_fallback(
        n_parallel = 1,
        extra_args = None,
        spec_flags = spec,
        cmd = _cmd(1, spec),
    )
    assert clamped[clamped.index("--parallel") + 1] == "1"
    assert fallback[fallback.index("--parallel") + 1] == "1"
    assert slots == 1


def test_a_user_owned_spec_type_is_not_handed_slots_it_never_asked_to_lose():
    """The pre-flight clamp already reduced this load before the KV fit, so the fit is
    sized for one slot and the backstop never records a debt."""
    clamped, fallback, slots = _run_clamp_then_fallback(
        n_parallel = 1,
        extra_args = ["--spec-type", "draft-mtp"],
        spec_flags = [],
        cmd = _cmd(1, []),
    )
    assert clamped[clamped.index("--parallel") + 1] == "1"
    assert fallback[fallback.index("--parallel") + 1] == "1"
    assert slots == 1


def test_a_non_mtp_resolution_keeps_its_slots_end_to_end():
    """No clamp, so the restore must not touch a count that was already correct."""
    spec = ["--spec-default"]
    clamped, fallback, slots = _run_clamp_then_fallback(
        n_parallel = 4,
        extra_args = None,
        spec_flags = spec,
        cmd = _cmd(4, spec),
    )
    assert clamped[clamped.index("--parallel") + 1] == "4"
    assert fallback[fallback.index("--parallel") + 1] == "4"
    assert slots == 4


def test_the_restore_is_scoped_to_the_retry_that_actually_drops_mtp():
    """A successful MTP launch, and the FA-off retry that keeps MTP, both stay at one
    slot: the restore belongs to the --spec-default retry alone."""
    tree = _load_model_tree()
    restore = _if_block(_is_slot_restore, tree)
    owners = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(restore in ast.walk(stmt) for stmt in node.body)
        and "_spec_requested_mtp" in _names(node.test)
    ]
    assert len(owners) == 1, "the slot restore left the MTP-fallback branch"
    assert "healthy" in _names(owners[0].test), "the restore must only run on a failed MTP start"
    # ...and after the retry argv exists, or it would rewrite nothing.
    src = _load_model_source()
    assert src.index("fallback_cmd = cmd[:_spec_start]") < src.index("_mtp_clamped_slots > 1")
    assert src.index("_mtp_clamped_slots > 1") < src.index("_spawn_and_wait(fallback_cmd")


# ── a suppressed drafter must not churn the server it left healthy ───


def _loaded_cpu_backend(monkeypatch, tmp_path):
    """A healthy paravirtual CPU server, plus the drafter still sitting on disk."""
    _paravirtual(monkeypatch)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_kill_orphaned_servers", staticmethod(lambda: 0)
    )
    backend = llama_cpp.LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._healthy = True
    backend._audio_probed = True
    backend._model_identifier = "owner/repo"
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    drafter = tmp_path / "mtp-model.gguf"
    drafter.write_bytes(b"GGUF")
    backend._gguf_path = str(gguf)
    backend._requested_n_ctx = 8192
    backend._requested_n_parallel = 1
    backend._requested_spec_mode = "auto"
    # What the first load on a virtualised Mac left behind.
    backend._gpu_memory_mode = "manual"
    backend._gpu_layers = 0
    return backend, gguf, drafter


def _target_state(backend, gguf, **overrides):
    kwargs = dict(
        model_identifier = "owner/repo",
        gguf_path = str(gguf),
        hf_variant = None,
        n_ctx = 8192,
        cache_type_kv = None,
        speculative_type = None,
        chat_template_override = None,
        extra_args = None,
        is_vision = False,
        gpu_memory_mode = "manual",
        gpu_layers = 0,
        n_parallel = 1,
    )
    kwargs.update(overrides)
    return backend.adopt_load_intent_if_matched(llama_cpp.GgufLoadIntent(**kwargs))


def _gpu_pin_recorders():
    """Every `if` in load_model that writes self._gpu_ids, in source order. load_model
    records the pin more than once, so judging one site would miss a later overwrite."""
    tree = _load_model_tree()
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(n, ast.Attribute) and n.attr == "_gpu_ids" and isinstance(n.ctx, ast.Store)
            for stmt in [*node.body, *node.orelse]
            for n in ast.walk(stmt)
        )
    ]
    assert found, "no _gpu_ids recording block found in load_model"
    # ast.walk descends into elif chains, so drop the ones that are another's else-branch:
    # running those on their own would skip the guard that precedes them.
    nested = {id(alt) for node in found for alt in node.orelse if isinstance(alt, ast.If)}
    return sorted((n for n in found if id(n) not in nested), key = lambda n: n.lineno)


def test_a_forced_cpu_launch_records_no_effective_gpu_pin():
    """--device none means the runtime uses no GPU, so echoing the requested pick as
    effective both misreports /status and makes clearing that pick reload a CPU server
    already in the target state. Runs every recorder in order: the last one wins, and an
    earlier clear is worth nothing if a later block re-assigns the request."""
    scope = {
        "self": types.SimpleNamespace(_gpu_ids = None, _requested_gpu_ids = None),
        "_paravirtual_cpu_forced": True,
        "is_vulkan_backend": False,
        "gpu_ids": [0],
        "gpu_indices": [0],
        "_vulkan_pin_ids": None,
        # The unmatched-ordinal guard clears the pin and raises; inert here.
        "_vulkan_explicit_unmatched": False,
    }
    for node in _gpu_pin_recorders():
        exec(ast.unparse(node), {}, scope)
        assert scope["self"]._gpu_ids is None, f"recorder at line {node.lineno} re-pinned a GPU"
    # The raw pick survives, so repeating it still dedupes; only clearing it changes.
    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._is_diffusion = False
    backend._requested_gpu_ids, backend._gpu_ids = [0], None
    assert backend.matches_gpu_ids([0]) is True
    assert backend.matches_gpu_ids(None) is True


def test_a_forced_cpu_diffusion_runner_records_no_effective_gpu_pin():
    """A zero-layer split on virtualised Metal masks the child's devices, so the runner
    uses none of them. Recording the pick anyway made /status name an unused device and
    made clearing it reload the same CPU runner; the pick still has to round-trip."""
    src = inspect.getsource(llama_cpp.LlamaCppBackend._start_diffusion_server)
    assert "_pv_diffusion_cpu = manual_ngl == 0 and _metal_device_is_paravirtual()" in src
    assert "if gpu_ids and not _pv_diffusion_cpu else None" in src

    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._is_diffusion = True
    backend._gpu_ids, backend._requested_gpu_ids = None, [1]
    assert backend.matches_gpu_ids([1]) is True  # re-sending the pick must not reload
    assert backend.matches_gpu_ids(None) is True  # nor must clearing it
    # A runner that really pinned a device still reloads when the pick changes.
    backend._gpu_ids = backend._requested_gpu_ids = [1]
    assert backend.matches_gpu_ids([0]) is False


def test_a_real_mac_still_records_the_gpu_it_pinned(monkeypatch):
    """The negative: on physical Apple Silicon the pick is effective and must be echoed."""
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: False)
    backend = llama_cpp.LlamaCppBackend.__new__(llama_cpp.LlamaCppBackend)
    backend._is_diffusion = False
    backend._requested_gpu_ids = backend._gpu_ids = [0]
    assert backend.matches_gpu_ids(None) is False  # dropping a real pin must reload


def test_a_suppressed_drafter_does_not_reload_the_server_it_left_healthy(monkeypatch, tmp_path):
    """The drop clears the launched drafter, but the file stays on disk and the caller
    keeps supplying it, so comparing against the stored None would respawn the same
    drafter-free server on every repeat Apply."""
    backend, gguf, drafter = _loaded_cpu_backend(monkeypatch, tmp_path)
    backend._mtp_draft_path = None
    backend._mtp_draft_suppressed_path = str(drafter)

    def _never(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("tore down a healthy server on a duplicate /load")

    monkeypatch.setattr(backend, "_find_llama_server_binary", _never)
    monkeypatch.setattr(backend, "_kill_process", _never)

    assert (
        backend.load_model(
            llama_cpp.GgufLoadIntent(
                model_identifier = "owner/repo",
                gguf_path = str(gguf),
                mtp_draft_path = str(drafter),
                n_ctx = 8192,
                gpu_memory_mode = "auto",
                gpu_layers = -1,
                n_parallel = 1,
            )
        )
        is True
    )


def test_the_route_dedupe_reads_the_suppressed_drafter_too(monkeypatch, tmp_path):
    """The route re-detects the sibling before load_model runs, so handling it on the
    backend path alone would still reload on every Apply."""
    from models.inference import LoadRequest
    from routes.inference import _active_gguf_intent

    # _paravirtual alone leaves the route's own binding on real hardware.
    _paravirtual_everywhere(monkeypatch)
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_kill_orphaned_servers", staticmethod(lambda: 0)
    )
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    drafter = tmp_path / "mtp-model.gguf"
    drafter.write_bytes(b"GGUF")
    backend = llama_cpp.LlamaCppBackend()
    backend._process = _FakeProcess()
    backend._healthy = True
    backend._model_identifier = str(gguf)
    backend._gguf_path = str(gguf)
    backend._requested_n_ctx = 4096
    backend._requested_n_parallel = 1
    backend._mtp_draft_path = None
    backend._mtp_draft_suppressed_path = str(drafter)
    # What the first load on a virtualised Mac left behind, so this compares against a
    # server that really launched under the pin.
    backend._gpu_memory_mode = "manual"
    backend._gpu_layers = 0
    request = LoadRequest(model_path = str(gguf), max_seq_length = 4096)

    def _intent():
        # The route rebuilds the intent from the request, re-detecting the sibling
        # drafter, so the dedupe has to hold on what it hands the backend.
        return _active_gguf_intent(
            request,
            backend,
            model_identifier = str(gguf),
            chat_template_override = None,
            n_parallel = 1,
            native_grant_backed = False,
        )

    assert backend.adopt_load_intent_if_matched(_intent())
    # Nothing suppressed: a drafter that genuinely appeared must still reload.
    backend._mtp_draft_suppressed_path = None
    assert not backend.adopt_load_intent_if_matched(_intent())


def test_a_drafter_that_genuinely_appeared_still_reloads(monkeypatch, tmp_path):
    """The negative that matters most: without a suppression on record, a drafter dropped
    next to the weights must still force the reload that engages it."""
    backend, gguf, drafter = _loaded_cpu_backend(monkeypatch, tmp_path)
    backend._mtp_draft_path = None
    backend._mtp_draft_suppressed_path = None
    assert _target_state(backend, gguf, mtp_draft_path = str(drafter)) is False


def test_only_the_drafter_that_was_suppressed_dedupes(monkeypatch, tmp_path):
    """A different drafter, or the suppressed one disappearing, is a real change: the fast
    path is scoped to the exact file the load decided not to launch."""
    backend, gguf, drafter = _loaded_cpu_backend(monkeypatch, tmp_path)
    backend._mtp_draft_path = None
    backend._mtp_draft_suppressed_path = str(drafter)
    other = tmp_path / "mtp-other.gguf"
    other.write_bytes(b"GGUF")
    assert _target_state(backend, gguf, mtp_draft_path = str(other)) is False
    assert _target_state(backend, gguf, mtp_draft_path = None) is False
    # ...and the one that was suppressed still matches.
    assert _target_state(backend, gguf, mtp_draft_path = str(drafter)) is True


def test_the_suppression_does_not_outlive_the_server(monkeypatch, tmp_path):
    """Unload clears it beside the launched path: a stale record would dedupe the next
    load of a different model, and an update (the only way to lift the suppression)
    unloads first."""
    backend, _gguf, drafter = _loaded_cpu_backend(monkeypatch, tmp_path)
    backend._mtp_draft_suppressed_path = str(drafter)
    backend.unload_model()
    assert backend.mtp_draft_suppressed_path is None


def test_a_launched_drafter_records_no_suppression(monkeypatch, tmp_path):
    """The negative on the recording side: a build that can pin the drafter launches it,
    so nothing is suppressed and the ordinary comparison applies."""
    backend, _gguf, _drafter = _loaded_cpu_backend(monkeypatch, tmp_path)
    assert backend.mtp_draft_suppressed_path is None
    src = _load_model_source()
    # Only the unpinnable branch records it, and it records what it is about to clear.
    assert src.index("_pv_suppressed_draft_path = launch_mtp_draft_path") < src.index(
        "                    launch_mtp_draft_path = None"
    )
    assert "self._mtp_draft_suppressed_path = _pv_suppressed_draft_path" in src


# ── an inherited projector must not slip past the projector guard ────


def _mmproj_env_scrub(*, paravirtual: bool) -> dict:
    """Run load_model's real inherited-projector env scrub and report the child's env."""
    block = _if_block(
        lambda test: isinstance(test, ast.Name) and test.id == "_paravirtual_cpu_forced",
        _mmproj_env_tree(),
    )
    scope = {
        "_paravirtual_cpu_forced": paravirtual,
        "env": {
            "LLAMA_ARG_MMPROJ": "/inherited/proj.gguf",
            "LLAMA_ARG_MMPROJ_URL": "https://example.invalid/proj.gguf",
            "LLAMA_ARG_THREADS": "8",
        },
    }
    exec(ast.unparse(ast.Module(body = [block], type_ignores = [])), scope)
    return scope["env"]


def _mmproj_env_tree() -> ast.AST:
    """load_model's statements, narrowed to the ones mentioning _pv_mmproj_var."""
    keep = [
        node
        for node in ast.walk(_load_model_tree())
        if isinstance(node, ast.If) and "_pv_mmproj_var" in _names(node)
    ]
    assert keep, "the inherited-projector env scrub left load_model"
    return ast.Module(body = keep, type_ignores = [])


def test_an_inherited_projector_is_dropped_from_the_child_env():
    """argv cannot un-set LLAMA_ARG_MMPROJ: llama.cpp reads it directly, so an inherited
    projector loads on the virtualised device independently of --gpu-layers 0.
    LLAMA_ARG_MMPROJ_URL goes with it because its download overwrites mmproj.path, so it
    outranks even the --mmproj Unsloth emits."""
    env = _mmproj_env_scrub(paravirtual = True)
    assert "LLAMA_ARG_MMPROJ" not in env
    assert "LLAMA_ARG_MMPROJ_URL" not in env
    # ...and nothing else in the inherited env is touched here.
    assert env == {"LLAMA_ARG_THREADS": "8"}


def test_a_real_mac_keeps_its_inherited_projector():
    """The scrub is scoped to the virtualised device: on physical Apple Silicon (or any
    non-Mac) the projector runs correctly on the GPU, so a deliberate LLAMA_ARG_MMPROJ
    must reach llama-server untouched."""
    env = _mmproj_env_scrub(paravirtual = False)
    assert env["LLAMA_ARG_MMPROJ"] == "/inherited/proj.gguf"
    assert env["LLAMA_ARG_MMPROJ_URL"] == "https://example.invalid/proj.gguf"


def test_the_env_scrub_lands_before_the_spawn():
    """The env is built once and handed to Popen, so a scrub after the spawn would leave
    the projector loaded on the device it just dropped."""
    src = _load_model_source()
    scrub = 'for _pv_mmproj_var in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL")'
    assert src.index(scrub) < src.index("_spawn_and_wait(")


# ── one normalization, shared by the launch and both duplicate-load comparators ──
#
# The rewrite used to live inside load_model while the comparators judged the RAW
# request, so a repeat identical Apply mismatched the backend's own recorded state and
# tore down a healthy CPU server, 409-ing or cancelling an active generation. Every case
# below carries NON-EMPTY extra_args, the blind spot that let the drafter-drop through.


def _routes():
    """routes/inference.py, imported lazily so the cheap AST tests do not pay for
    FastAPI."""
    from routes import inference as routes_inference
    return routes_inference


def _paravirtual_everywhere(monkeypatch):
    """The route imports the detector by value, so patching only the llama_cpp attribute
    would leave the route comparator on real hardware."""
    _paravirtual(monkeypatch)
    monkeypatch.setattr(_routes(), "_metal_device_is_paravirtual", lambda: True)


def _real_mac_everywhere(monkeypatch):
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: False)
    monkeypatch.setattr(_routes(), "_metal_device_is_paravirtual", lambda: False)


def _normalized(**kwargs):
    return llama_cpp.paravirtual_normalized_request(**kwargs)


def test_the_normalizer_pins_every_placement_knob_to_cpu():
    """The one definition of the rewrite: whatever came in, this is what launches."""
    out = _normalized(
        gpu_memory_mode = "auto",
        gpu_layers = -1,
        tensor_parallel = True,
        tensor_split = [0.5, 0.5],
        n_cpu_moe = 8,
        extra_args = ["-ngl", "99", "--top-k", "40"],
    )
    assert out.gpu_memory_mode == "manual"
    assert out.gpu_layers == 0
    assert out.tensor_parallel is False
    assert out.tensor_split is None
    assert out.n_cpu_moe == 0
    assert out.extra_args == ["--top-k", "40"]


def test_the_normalizer_is_idempotent():
    """load_model normalizes, then calls a comparator that normalizes again (and a respawn
    replays the raw kwargs): a second pass must change nothing."""
    once = _normalized(
        gpu_memory_mode = "auto", gpu_layers = -1, extra_args = ["-ot", ".*=Metal", "--top-k", "40"]
    )
    twice = _normalized(
        gpu_memory_mode = once.gpu_memory_mode,
        gpu_layers = once.gpu_layers,
        tensor_parallel = once.tensor_parallel,
        tensor_split = once.tensor_split,
        n_cpu_moe = once.n_cpu_moe,
        extra_args = once.extra_args,
    )
    assert twice == once


def test_the_normalizer_still_runs_for_a_request_that_already_asks_for_cpu():
    """The severe one: the whole block used to be skipped for Manual + 0 layers, so an
    --override-tensor in extras (never stripped by the route, and applied while choosing
    each weight's buffer type, before any layer is assigned) put the weights straight back
    on the corrupt device."""
    out = _normalized(
        gpu_memory_mode = "manual",
        gpu_layers = 0,
        extra_args = ["-ot", ".*=Metal", "--top-k", "40"],
    )
    assert out.extra_args == ["--top-k", "40"]


def test_the_normalizer_keeps_extras_it_has_no_business_touching():
    """The negatives: a CPU-targeted override does what this fallback does, a --split-mode
    is neutralised at the launch instead (the comparators compare the stored list
    verbatim), and "no opinion" must stay None, not become []."""
    out = _normalized(extra_args = ["-ot", "exps=CPU", "-sm", "row", "--temp", "0.7"])
    assert out.extra_args == ["-ot", "exps=CPU", "-sm", "row", "--temp", "0.7"]
    assert _normalized(extra_args = None).extra_args is None
    assert _normalized(extra_args = []).extra_args == []


def test_the_launch_normalizes_a_manual_cpu_request_too(monkeypatch):
    """End-to-end: the flags reaching the duplicate-load check are the ones that launch,
    for a request that already said Manual + 0."""
    _paravirtual(monkeypatch)
    backend = llama_cpp.LlamaCppBackend()
    seen = {}

    def _capture(intent):
        captured = asdict(intent)
        # The intent freezes its sequences into tuples; compare on content.
        for _key in ("extra_args", "tensor_split", "gpu_ids"):
            if captured.get(_key) is not None:
                captured[_key] = list(captured[_key])
        seen.update(captured)
        return True

    monkeypatch.setattr(backend, "adopt_load_intent_if_matched", _capture)
    monkeypatch.setattr(backend, "_apply_detected_audio", lambda _d: True)
    backend._audio_probed = True
    backend._healthy = True
    backend.load_model(
        llama_cpp.GgufLoadIntent(
            model_identifier = "owner/repo",
            gguf_path = "/nonexistent/model.gguf",
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            n_cpu_moe = 8,
            tensor_parallel = True,
            extra_args = ["-ot", ".*=Metal", "--top-k", "40"],
        )
    )
    assert seen["extra_args"] == ["--top-k", "40"]
    assert seen["n_cpu_moe"] == 0
    assert seen["tensor_parallel"] is False
    assert seen["tensor_split"] is None


def test_a_real_mac_keeps_its_tensor_override_on_a_manual_cpu_load(monkeypatch):
    """None of this applies to physical Apple Silicon, where a deliberate -ot must reach
    llama-server exactly as written."""
    monkeypatch.setattr(llama_cpp, "_metal_device_is_paravirtual", lambda: False)
    backend = llama_cpp.LlamaCppBackend()
    seen = {}

    def _capture(intent):
        captured = asdict(intent)
        # The intent freezes its sequences into tuples; compare on content.
        for _key in ("extra_args", "tensor_split", "gpu_ids"):
            if captured.get(_key) is not None:
                captured[_key] = list(captured[_key])
        seen.update(captured)
        return True

    monkeypatch.setattr(backend, "adopt_load_intent_if_matched", _capture)
    monkeypatch.setattr(backend, "_apply_detected_audio", lambda _d: True)
    backend._audio_probed = True
    backend._healthy = True
    backend.load_model(
        llama_cpp.GgufLoadIntent(
            model_identifier = "owner/repo",
            gguf_path = "/nonexistent/model.gguf",
            gpu_memory_mode = "manual",
            gpu_layers = 0,
            n_cpu_moe = 8,
            extra_args = ["-ot", ".*=Metal", "--top-k", "40"],
        )
    )
    assert seen["extra_args"] == ["-ot", ".*=Metal", "--top-k", "40"]
    assert seen["n_cpu_moe"] == 8


def _cpu_server(
    monkeypatch,
    tmp_path,
    *,
    launched_extras,
    requested_extras = None,
):
    """A healthy paravirtual CPU server that recorded a normalized placement."""
    monkeypatch.setattr(
        llama_cpp.LlamaCppBackend, "_kill_orphaned_servers", staticmethod(lambda: 0)
    )
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
    backend._gpu_memory_mode = "manual"
    backend._gpu_layers = 0
    backend._n_cpu_moe = 0
    backend._tensor_split = None
    backend._tensor_parallel = False
    backend._extra_args = list(launched_extras)
    backend._requested_extra_args = list(
        requested_extras if requested_extras is not None else launched_extras
    )
    backend._extra_args_source = ("owner/repo", None)
    return backend, gguf


def _route_matches(request, backend):
    """What the route now does: rebuild the intent from the request, then let the single
    backend comparator judge it. The dedupe moved off the route in #7663, so this is the
    route-side path end to end."""
    intent = _routes()._active_gguf_intent(
        request,
        backend,
        model_identifier = "owner/repo",
        chat_template_override = None,
        n_parallel = 1,
        native_grant_backed = False,
    )
    return backend.adopt_load_intent_if_matched(intent)


def _load_request(gguf, **overrides):
    from models.inference import LoadRequest

    fields = dict(
        model_path = str(gguf),
        max_seq_length = 8192,
        speculative_type = "auto",
        gpu_memory_mode = "auto",
        gpu_layers = -1,
    )
    fields.update(overrides)
    return LoadRequest(**fields)


def test_a_repeat_auto_request_with_extras_matches_the_cpu_server_it_left(monkeypatch, tmp_path):
    """Both comparators. An API client, a second tab or a saved preset keeps sending Auto:
    the browser adopting the normalized echo does not cover them, and never covers the
    extras box, which the UI does not round-trip."""
    _paravirtual_everywhere(monkeypatch)
    backend, gguf = _cpu_server(monkeypatch, tmp_path, launched_extras = ["--top-k", "40"])
    request = _load_request(
        gguf,
        llama_extra_args = ["-ngl", "99", "--top-k", "40"],
        tensor_parallel = True,
        n_cpu_moe = 8,
    )
    assert _route_matches(request, backend) is True
    assert (
        _target_state(
            backend,
            gguf,
            speculative_type = "auto",
            gpu_memory_mode = "auto",
            gpu_layers = -1,
            tensor_parallel = True,
            n_cpu_moe = 8,
            extra_args = ["-ngl", "99", "--top-k", "40"],
        )
        is True
    )


def test_the_same_pair_still_mismatches_on_a_real_mac(monkeypatch, tmp_path):
    """The negative that keeps the normalization honest: off this hardware an Auto request
    against a Manual/0 server is a genuine settings change and must reload."""
    _real_mac_everywhere(monkeypatch)
    backend, gguf = _cpu_server(monkeypatch, tmp_path, launched_extras = ["--top-k", "40"])
    request = _load_request(gguf, llama_extra_args = ["--top-k", "40"])
    assert _route_matches(request, backend) is False
    assert (
        _target_state(
            backend,
            gguf,
            speculative_type = "auto",
            gpu_memory_mode = "auto",
            gpu_layers = -1,
            extra_args = ["--top-k", "40"],
        )
        is False
    )


def test_a_genuinely_different_extras_box_still_reloads(monkeypatch, tmp_path):
    """Normalizing must not swallow a real edit: only the offload family and GPU-bound
    tensor overrides are rewritten, so a changed sampler flag is still a change."""
    _paravirtual_everywhere(monkeypatch)
    backend, gguf = _cpu_server(monkeypatch, tmp_path, launched_extras = ["--top-k", "40"])
    request = _load_request(gguf, llama_extra_args = ["--top-k", "20"])
    assert _route_matches(request, backend) is False
    assert (
        _target_state(
            backend,
            gguf,
            speculative_type = "auto",
            gpu_memory_mode = "auto",
            gpu_layers = -1,
            extra_args = ["--top-k", "20"],
        )
        is False
    )


def test_a_tensor_split_mode_in_extras_does_not_reload_a_cpu_server(monkeypatch, tmp_path):
    """The pin overrides --split-mode at the launch rather than stripping it, so the stored
    extras still say tensor while nothing tensor ever launched; judging the raw extras
    would reload on every Apply."""
    _paravirtual_everywhere(monkeypatch)
    backend, gguf = _cpu_server(monkeypatch, tmp_path, launched_extras = ["-sm", "tensor"])
    request = _load_request(gguf, llama_extra_args = ["-sm", "tensor"], tensor_parallel = True)
    assert _route_matches(request, backend) is True
    assert (
        _target_state(
            backend,
            gguf,
            speculative_type = "auto",
            gpu_memory_mode = "auto",
            gpu_layers = -1,
            tensor_parallel = True,
            extra_args = ["-sm", "tensor"],
        )
        is True
    )


def test_a_dropped_drafter_does_not_reload_over_the_extras_it_rewrote(monkeypatch, tmp_path):
    """The drop strips the whole spec group from extra_args and stores the rewrite, but the
    caller keeps sending its own list, so comparing launched-vs-requested mismatched on
    every Apply and reloaded without bound."""
    _paravirtual_everywhere(monkeypatch)
    asked = ["--draft-max", "8", "--top-k", "40"]
    backend, gguf = _cpu_server(
        monkeypatch, tmp_path, launched_extras = ["--top-k", "40"], requested_extras = asked
    )
    request = _load_request(gguf, llama_extra_args = list(asked))
    assert _route_matches(request, backend) is True
    assert (
        _target_state(
            backend,
            gguf,
            speculative_type = "auto",
            gpu_memory_mode = "auto",
            gpu_layers = -1,
            extra_args = list(asked),
        )
        is True
    )


def test_an_apply_that_inherits_the_extras_does_not_reload_the_rewritten_server(
    monkeypatch, tmp_path
):
    """The browser never sends llama_extra_args, so the route inherits the LAUNCHED list and
    the comparison would judge it against the INVOKED one. After the drafter drop those
    differ by the whole spec group, so every Apply from the UI tore the server down."""
    _paravirtual_everywhere(monkeypatch)
    backend, gguf = _cpu_server(
        monkeypatch,
        tmp_path,
        launched_extras = ["--top-k", "40"],
        requested_extras = ["--draft-max", "8", "--top-k", "40"],
    )
    # No llama_extra_args: the inherit path the UI actually takes.
    assert _route_matches(_load_request(gguf), backend) is True
    # Control: an Apply that does name its extras is still judged on the invoked list.
    assert _route_matches(_load_request(gguf, llama_extra_args = ["--top-k", "20"]), backend) is False
    # And naming exactly the stripped list is a deliberate clear of the failed drafter,
    # not an inherit: judging it by value would dedupe and strand the user on MTP.
    assert _route_matches(_load_request(gguf, llama_extra_args = ["--top-k", "40"]), backend) is False


def test_an_edited_spec_flag_still_reloads_after_a_dropped_drafter(monkeypatch, tmp_path):
    """Requested-vs-requested must still notice a real change to the flags the drop
    removed."""
    _paravirtual_everywhere(monkeypatch)
    backend, gguf = _cpu_server(
        monkeypatch,
        tmp_path,
        launched_extras = ["--top-k", "40"],
        requested_extras = ["--draft-max", "8", "--top-k", "40"],
    )
    request = _load_request(gguf, llama_extra_args = ["--draft-max", "4", "--top-k", "40"])
    assert _route_matches(request, backend) is False
    assert (
        _target_state(
            backend,
            gguf,
            speculative_type = "auto",
            gpu_memory_mode = "auto",
            gpu_layers = -1,
            extra_args = ["--draft-max", "4", "--top-k", "40"],
        )
        is False
    )


def test_the_requested_extras_default_to_the_launched_ones():
    """Nothing rewritten means the two records are the same list, so a load that never hits
    the drop keeps comparing exactly what it compared before."""
    backend = llama_cpp.LlamaCppBackend()
    assert backend.requested_extra_args is None
    backend._extra_args = ["--top-k", "40"]
    assert backend.requested_extra_args == ["--top-k", "40"]
    backend._requested_extra_args = ["--draft-max", "8", "--top-k", "40"]
    assert backend.requested_extra_args == ["--draft-max", "8", "--top-k", "40"]
    # A copy, like extra_args: a caller mutating the result must not edit state.
    backend.requested_extra_args.append("--temp")
    assert backend.requested_extra_args == ["--draft-max", "8", "--top-k", "40"]


def test_the_drop_records_the_requested_extras_before_rewriting_them():
    """The recording contract, mirroring _pv_suppressed_draft_path: capture, then strip."""
    src = _load_model_source()
    assert src.index("_pv_suppressed_spec_extra_args = list(extra_args)") < src.index(
        "                            strip_spec = True,"
    )
    assert "self._requested_extra_args = (" in src


def _restore_requested_spec_mode(*, requested_extras):
    """Run load_model's real spec-mode restore against a backend whose
    _build_speculative_flags already judged the stripped list."""
    block = _if_block(
        lambda test: "_pv_suppressed_spec_extra_args" in _names(test),
    )
    backend = llama_cpp.LlamaCppBackend()
    backend._requested_spec_mode = "auto"
    scope = {
        "self": backend,
        "_pv_suppressed_spec_extra_args": requested_extras,
        "_extra_args_set_spec_type": llama_cpp._extra_args_set_spec_type,
    }
    exec(ast.unparse(ast.Module(body = [block], type_ignores = [])), scope)
    return backend._requested_spec_mode


def test_a_user_owned_spec_type_survives_the_drafter_drop():
    """The drop removed --spec-type before _build_speculative_flags saw the list, so the
    backend recorded "auto" while the caller's extras still mean "the user owns it"
    (None), and the spec-mode compare mismatched forever."""
    assert (
        _restore_requested_spec_mode(
            requested_extras = ["--spec-type", "draft-simple", "--model-draft", "/m/d.gguf"]
        )
        is None
    )


def test_the_restore_is_scoped_to_a_drop_that_owned_spec_type():
    """No drop, or a drop whose extras never set --spec-type, must leave the mode
    _build_speculative_flags resolved exactly as it is."""
    assert _restore_requested_spec_mode(requested_extras = None) == "auto"
    assert _restore_requested_spec_mode(requested_extras = ["--draft-max", "8"]) == "auto"


def test_the_route_really_can_deliver_a_manual_cpu_request_carrying_an_override():
    """The reachability leg for the block above: manual mode owns the offload flags at the
    route, but that strip is the --gpu-layers family only. An -ot survives it untouched
    and parse_gpu_layers_override reads nothing from it, so the request arrives as Manual
    + 0 with a GPU-bound override still in the extras. (An -ngl cannot: the route
    translates it into gpu_layers first.)"""
    extras = ["-ot", ".*=Metal", "--top-k", "40"]
    assert llama_server_args.parse_gpu_layers_override(extras) is None
    assert (
        llama_server_args.strip_shadowing_flags(
            extras,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_tensor_split = True,
            strip_offload = True,
        )
        == extras
    )
