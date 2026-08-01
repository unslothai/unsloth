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
    """Execute load_model's real projector-resolution statements and report what
    the launch would see: (mmproj path, vision flag, warnings)."""
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
    scope = {
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
    """clip.cpp never reads --gpu-layers, so on a build without
    --no-mmproj-offload the vision encoder stays on the virtualised device the
    rest of this fallback exists to avoid. Serving image embeddings built from
    corrupt output is the failure being prevented, so drop the projector."""
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
    """The drop is scoped to the virtualised device: an old llama.cpp on physical
    Apple Silicon (or any non-Mac) must keep its projector on the GPU."""
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
    audio-encoder probe, so clearing it after any of those would launch a
    projector the guard believes it dropped (or offer audio input that is gone)."""
    src = _load_model_source()
    gate_at = src.index("_pv_mmproj_unpinnable = bool(")
    assert gate_at < src.index('cmd.extend(["--mmproj", launch_mmproj_path])')
    assert gate_at < src.index("self._mmproj_vram_bytes(launch_mmproj_path)")
    assert gate_at < src.index("read_mmproj_audio_capability(launch_mmproj_path)")
    # ...and the session flag the frontend reads follows the same variable.
    assert "self._is_vision = effective_is_vision" in src


# ── a pass-through GPU split mode must not fail the CPU-only load ────


@pytest.mark.parametrize(
    "extra_args, expected",
    [
        # `-sm row` throws "device <X> does not support split buffers" in
        # make_gpu_buft_list for every backend except SYCL, and that runs over
        # model->devices before any layer is assigned, so --gpu-layers 0 does not
        # save it. Metal is always in that list here: the zero-offload mask only
        # writes CUDA/HIP visibility vars.
        (["-sm", "row"], ["--split-mode", "layer"]),
        (["--split-mode", "row"], ["--split-mode", "layer"]),
        # `-sm tensor` throws "not implemented for architecture" for every arch
        # llm_arch_supports_sm_tensor excludes, also independently of ngl.
        (["--top-k", "40", "-sm", "tensor"], ["--split-mode", "layer"]),
        (["-sm", "none"], ["--split-mode", "layer"]),
        (["-sm=row"], ["--split-mode", "layer"]),
        # The negatives: nothing to neutralise means no flag, so a CPU-only launch
        # is not handed a redundant duplicate argument.
        (None, []),
        ([], []),
        (["--top-k", "40"], []),
        (["-sm", "layer"], []),
        (["--split-mode", "LAYER"], []),
        # Last-wins, like the parser: a user who already ends on layer is fine.
        (["-sm", "row", "--split-mode", "layer"], []),
        # ...and one who ends on row is not, however it started.
        (["--split-mode", "layer", "-sm", "row"], ["--split-mode", "layer"]),
        # --tensor-split is genuinely inert at --gpu-layers 0: llama-model.cpp
        # computes the split points but never indexes them once act_gpu_layers is
        # 0, so it must not drag the override in on its own.
        (["-ts", "3,1"], []),
        (["--tensor-split", "3,1"], []),
    ],
)
def test_a_pass_through_split_mode_cannot_fail_the_cpu_only_load(extra_args, expected):
    assert llama_cpp._paravirtual_split_mode_pin(extra_args) == expected


def test_the_split_mode_override_is_scoped_to_the_virtualised_mac():
    """The strongest negative: a real Mac (or any non-Mac) must keep the user's
    row/none split, so the call site has to be gated on the detector, not on the
    zero-layer placement that a plain Manual load also has."""
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
    """llama.cpp is last-wins, so an override emitted before the user's extras
    would be undone by the very flag it exists to neutralise."""
    # Overridden, not stripped: the route's duplicate-load comparator compares the
    # stored extras verbatim against the request and the UI does not round-trip the
    # extras box, so a strip here would make every later Apply a real model swap.
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


def _run_clamp_then_fallback(
    *,
    n_parallel,
    extra_args,
    spec_flags,
    cmd,
    asked_for = None,
):
    """Execute the real clamp block, rebuild fallback_cmd the way the MTP retry
    does, then execute the real restore block. Returns the two argvs and the
    slot count that _commit_effective_parallel_slots would receive."""
    scope = {
        "n_parallel": n_parallel,
        "extra_args": extra_args,
        "spec_flags": spec_flags,
        "cmd": list(cmd),
        "_mtp_clamped_slots": 0,
        "model_identifier": "owner/repo",
        "logger": llama_cpp.logger,
        "_extra_args_set_spec_type": llama_cpp._extra_args_set_spec_type,
        "_extra_args_requests_mtp": llama_cpp._extra_args_requests_mtp,
        # The pre-fit ask, which the restore must NOT reach for.
        "_pending_load_kwargs": {"n_parallel": n_parallel if asked_for is None else asked_for},
    }
    exec(ast.unparse(_if_block(_is_mtp_clamp)), scope)
    clamped = list(scope["cmd"])
    # The retry swaps the spec slice for --spec-default; the slice sits at the
    # tail of this synthetic argv.
    spec_at = len(clamped) - len(spec_flags)
    scope["fallback_cmd"] = clamped[:spec_at] + ["--spec-default"]
    exec(ast.unparse(_if_block(_is_slot_restore)), scope)
    return clamped, scope["fallback_cmd"], scope["n_parallel"]


def _cmd(slots: int, spec_flags: list) -> list:
    return ["llama-server", "-m", "/m.gguf", "--parallel", str(slots), "--kv-unified", *spec_flags]


def test_the_mtp_fallback_gets_the_requested_slots_back():
    """MTP aborts at startup, the retry drops speculative decoding entirely, and
    that server serves chats in parallel just fine -- so it must not inherit the
    single slot MTP needed. The KV fit was sized for the full count."""
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


def test_the_fallback_gets_back_what_the_fit_sized_not_what_the_user_asked():
    """_slots_that_fit_on_gpu can already have cut the count before the clamp, so
    restoring the raw ask would launch a server the VRAM budget never covered."""
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
    """The pre-flight clamp already reduced this load before the KV fit, so the
    fit is sized for one slot and the backstop never records a debt."""
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
    """No clamp, so the restore has nothing to rewrite and must not touch a
    count that was already correct."""
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
    """A successful MTP launch, and the FA-off retry that keeps MTP, must both
    stay at one slot: the restore belongs to the --spec-default retry alone."""
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
    assert src.index("fallback_cmd = (") < src.index("_mtp_clamped_slots > 1")
    assert src.index("_mtp_clamped_slots > 1") < src.index("_spawn_and_wait(fallback_cmd")
