# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression guards for silent tensor-parallel downgrades in load_model.

PR #6416 blanket-disabled tensor parallelism for vision models to dodge a
--split-mode tensor + --mmproj GGML_ASSERT (#6415), which silently single-GPU'd
any mmproj/MTP GGUF that fit on one card. The fix makes the skip self-healing:
tensor is tried by default and recorded per (binary, model) only on a real abort.

load_model is too entangled to drive end-to-end, so these tests inspect the
source / drive the pure helpers. The headline test pins the set of TP-drop
conditions, so a new silent drop fails CI. No GPU; fully deterministic.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import os
import sys
import textwrap
import types as _types
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# External-dep stubs so importing the backend doesn't require structlog / httpx /
# loggers -- but only when the real module is missing, so a lightweight stub never
# shadows the real package (or `loggers.handlers` submodule) for tests collected
# later in the same pytest process.
try:
    import structlog  # noqa: F401
except ImportError:
    _structlog_stub = _types.ModuleType("structlog")
    _structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    sys.modules["structlog"] = _structlog_stub
try:
    import loggers  # noqa: F401
except ImportError:
    _loggers_stub = _types.ModuleType("loggers")
    _loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules["loggers"] = _loggers_stub
try:
    import httpx as _httpx_real  # noqa: F401
except ImportError:
    _httpx_stub = _types.ModuleType("httpx")
    for _exc in (
        "ConnectError",
        "TimeoutException",
        "ReadTimeout",
        "ReadError",
        "RemoteProtocolError",
        "CloseError",
        "HTTPError",
        "RequestError",
    ):
        setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
    _httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
    _httpx_stub.Response = type("Response", (), {})
    _httpx_stub.Client = type(
        "C",
        (),
        {
            "__init__": lambda s, **kw: None,
            "__enter__": lambda s: s,
            "__exit__": lambda s, *a: None,
        },
    )
    sys.modules["httpx"] = _httpx_stub

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_GB = 1024**3


def _load_inference_routes_module():
    """Load routes/inference.py directly, bypassing routes/__init__.py (which imports
    every router, dragging in unrelated deps like python-multipart) (Codex #6659)."""
    route_path = Path(_BACKEND_DIR) / "routes" / "inference.py"
    spec = importlib.util.spec_from_file_location(
        "tp_vision_regression_inference_routes", route_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_model_ast() -> ast.FunctionDef:
    """Parse load_model into an AST FunctionDef (no import side effects)."""
    src = textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model))
    return ast.parse(src).body[0]


def _tensor_parallel_false_drop_guards() -> list[str]:
    """Source of the guard expression for every `if ...: tensor_parallel = False`
    (the LOCAL variable, not self._tensor_parallel) inside load_model."""
    fn = _load_model_ast()

    def _body_drops_tp(body) -> bool:
        for n in body:
            if (
                isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "tensor_parallel" for t in n.targets)
                and isinstance(n.value, ast.Constant)
                and n.value.value is False
            ):
                return True
        return False

    return [
        ast.unparse(node.test)
        for node in ast.walk(fn)
        if isinstance(node, ast.If) and _body_drops_tp(node.body)
    ]


# Every condition that may flip a requested tensor_parallel back to False. Adding
# one must be conscious: update this allowlist and keep multi-GPU where possible.
_ALLOWED_TP_DROP_GUARDS = {
    # Capability: --split-mode tensor aborted for this (binary, model) (#6415).
    # Self-healing -- tried by default, skipped only after a real abort (vs #6416).
    "tensor_parallel and self._tensor_split_aborts(binary, model_identifier)",
    # Capacity: tensor needs >= 2 GPUs clearing the compute-buffer reserve. Gated
    # on plan_tp (not raw tensor_parallel) so manual mode skips this planner (#6414).
    "plan_tp and len(tp_gpus) < 2",
    # Capacity: pooled usable VRAM can't hold weights + MTP reserve -> layer split.
    "_tp_weight_budget_mib <= _tp_required_mib",
    # Manual mode, Auto layers: --fit owns memory and is incompatible with a
    # tensor split, so TP is dropped (surfaced via logger.info) before the
    # cache-drop, so a quantized KV survives into the --fit load (#6414).
    "tensor_parallel and gpu_memory_mode == 'manual' and (gpu_layers < 0)",
    # Manual mode, explicit layers: a tensor split still needs >= 2 GPUs in use.
    "tensor_parallel and gpu_memory_mode == 'manual' and (gpu_layers >= 0) and (self._effective_gpu_count(sorted(gpu_ids) if gpu_ids else None) < 2)",
    # Manual mode, zero layers: nothing to split on the GPU, and a tensor-mode
    # launch under the CPU-only GPU mask (no visible devices) aborts the server
    # instead of the intended CPU-only load (#6414).
    "gpu_memory_mode == 'manual' and gpu_layers == 0",
}


def test_tensor_parallel_drop_sites_match_allowlist():
    """The set of reasons a requested TP can be dropped is fixed and reviewed: a new
    drop site fails this set-equality until consciously allowlisted (would catch #6416)."""
    found = set(_tensor_parallel_false_drop_guards())
    assert found == _ALLOWED_TP_DROP_GUARDS, (
        "tensor_parallel drop sites changed.\n"
        f"  unexpected (new) : {sorted(found - _ALLOWED_TP_DROP_GUARDS)}\n"
        f"  missing (removed): {sorted(_ALLOWED_TP_DROP_GUARDS - found)}\n"
        "A new drop means a user's TP request is ignored for a new reason -- "
        "review it, keep multi-GPU where possible, surface it, then update "
        "_ALLOWED_TP_DROP_GUARDS."
    )


def test_every_tp_drop_is_logged_not_silent():
    """Each tensor_parallel downgrade must log why, so it never disappears silently."""
    fn = _load_model_ast()

    def _body_drops_tp(body):
        return any(
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "tensor_parallel" for t in n.targets)
            and isinstance(n.value, ast.Constant)
            and n.value.value is False
            for n in body
        )

    def _body_logs(body) -> bool:
        for n in ast.walk(ast.Module(body = list(body), type_ignores = [])):
            if (
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "logger"
            ):
                return True
        return False

    for node in ast.walk(fn):
        if isinstance(node, ast.If) and _body_drops_tp(node.body):
            assert _body_logs(node.body), (
                f"TP drop under `{ast.unparse(node.test)}` has no logger call -- "
                "downgrades must explain themselves."
            )


def test_tensor_split_gate_is_self_healing_not_blanket():
    """Skip is conditional on a recorded (binary, model) abort, not a blanket
    is_vision disable (the #6416 regression)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "self._tensor_split_aborts(binary, model_identifier)" in src
    assert "if tensor_parallel and is_vision:" not in src
    assert "if tensor_parallel and effective_is_vision:" not in src


def test_tensor_split_skip_documents_layer_split_fallback():
    """When the skip fires (known-bad binary+model), it states the fallback."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    gate = src.find("self._tensor_split_aborts(binary, model_identifier)")
    assert gate != -1
    block = src[gate : gate + 600]
    assert "layer split" in block, "the skip should state it falls back to layer split"


def test_tensor_split_abort_recorded_early_on_first_spawn():
    """Recorded on the first spawn showing the marker, before the flash-attn-off
    retry (which can't run tensor so drops the marker) -- else it loops (oobabooga, #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    idx = src.find("_record_tensor_split_abort(binary, model_identifier)")
    assert idx != -1, "load_model must record a (binary, model) tensor-split abort"
    guard = src[max(0, idx - 600) : idx]
    assert "self._tensor_parallel" in guard
    assert (
        "_should_record_tensor_split_abort" in guard
    ), "record must be gated on the marker-plus-hard-crash decision helper"
    # Recorded before the flash-attn-off retry, not after the full ladder.
    fa_off = src.find("_with_flash_attn_off")
    assert 0 <= idx < fa_off, "recording must latch on the first spawn, before flash-off"


def test_vision_downgrade_preserves_multi_gpu_intent():
    """The vision downgrade raises _layer_min_gpus and threads it into both the
    _select_gpus and auto-context layer paths, so a fitting model still spreads."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "_layer_min_gpus = max(_layer_min_gpus, len(gpus))" in src
    assert src.count("min_gpus = _layer_min_gpus") >= 2
    assert "range(_auto_min_gpus, len(ranked) + 1)" in src
    auto = src.find("_auto_min_gpus = max(")
    assert auto != -1 and "_layer_min_gpus" in src[auto : auto + 200]


# ── per-binary capability cache (pure) ───────────────────────────────


def test_tensor_attempted_by_default_for_unknown_binary():
    """A (binary, model) not seen to abort -> tensor is attempted (not skipped)."""
    assert LlamaCppBackend._tensor_split_aborts("/never/seen/llama-server", "m") is False
    assert LlamaCppBackend._tensor_split_aborts(None, "m") is False
    assert LlamaCppBackend._tensor_split_aborts("/x", None) is False


def test_recorded_tensor_abort_is_per_model():
    """A recorded (binary, model) abort trips the gate for that model only -- a
    different model on the same binary still attempts tensor (oobabooga, #6659)."""
    b = f"/tmp/llama-server-{id(object())}"
    try:
        assert LlamaCppBackend._tensor_split_aborts(b, "model-a") is False
        LlamaCppBackend._record_tensor_split_abort(b, "model-a")
        assert LlamaCppBackend._tensor_split_aborts(b, "model-a") is True
        # a different model on the same binary is unaffected
        assert LlamaCppBackend._tensor_split_aborts(b, "model-b") is False
    finally:
        LlamaCppBackend._tensor_split_abort_keys.discard(
            LlamaCppBackend._tensor_split_cache_key(b, "model-a")
        )


# ── _select_gpus: single-GPU collapse vs honored multi-GPU intent (pure) ──


def test_select_gpus_collapses_to_single_gpu_when_model_fits():
    """Default (min_gpus=1): a 39 GB model on four 183 GB GPUs pins ONE GPU -- the
    'single GPU' symptom once TP drops, and why the downgrade needs min_gpus."""
    gpus = [(0, 180000), (1, 180000), (2, 180000), (3, 180000)]  # (idx, free MiB)
    gpu_indices, _use_fit = LlamaCppBackend._select_gpus(int(39 * _GB), gpus)
    assert gpu_indices is not None and len(gpu_indices) == 1


def test_select_gpus_min_gpus_keeps_multi_gpu_for_fitting_model():
    """min_gpus>=2 must NOT collapse to one GPU for a model that fits on one."""
    gpus = [(0, 180000), (1, 180000), (2, 180000), (3, 180000)]
    gpu_indices, _ = LlamaCppBackend._select_gpus(int(39 * _GB), gpus, min_gpus = 2)
    assert gpu_indices is not None and len(gpu_indices) >= 2


def test_select_gpus_min_gpus_capped_to_available():
    """min_gpus larger than the GPU count is capped, not an error."""
    gpus = [(0, 180000), (1, 180000)]
    gi, _ = LlamaCppBackend._select_gpus(int(10 * _GB), gpus, min_gpus = 8)
    assert gi is not None and len(gi) == 2


def test_select_gpus_uses_multiple_gpus_when_model_does_not_fit():
    """Sanity: selection spreads across GPUs when one card can't hold the model."""
    gpus = [(0, 40000), (1, 40000), (2, 40000), (3, 40000)]  # 40 GB free each
    gpu_indices, _use_fit = LlamaCppBackend._select_gpus(int(120 * _GB), gpus)
    assert gpu_indices is not None and len(gpu_indices) >= 2


def test_select_gpus_min_gpus_excludes_unusable_gpu():
    """min_gpus caps to usable cards: 2 free + 1 nearly-full -> 2-GPU split, not
    forcing the full card (OOM) or tripping --fit (#6659)."""
    gpus = [(0, 180000), (1, 180000), (2, 500)]  # GPU 2 is nearly full
    total = {0: 180000, 1: 180000, 2: 180000}
    gi, _ = LlamaCppBackend._select_gpus(
        int(39 * _GB),
        gpus,
        min_gpus = 3,
        total_by_idx = total,
        per_device_overhead_bytes = int(1 * _GB),
    )
    assert gi is not None
    assert 2 not in gi, "a nearly-full GPU must not be forced in to satisfy min_gpus"
    assert len(gi) == 2


def test_tensor_abort_cache_invalidated_on_binary_mtime_change(tmp_path):
    """Cache keys on (path, mtime, model), so a binary swapped in place (in-app
    update, no restart) is re-probed instead of inheriting the old abort (#6659)."""
    binp = tmp_path / "llama-server"
    binp.write_text("v1")
    p = str(binp)
    try:
        LlamaCppBackend._record_tensor_split_abort(p, "m")
        assert LlamaCppBackend._tensor_split_aborts(p, "m") is True
        # Simulate an in-place update bumping the binary's mtime.
        st = binp.stat()
        os.utime(p, (st.st_atime, st.st_mtime + 10))
        assert (
            LlamaCppBackend._tensor_split_aborts(p, "m") is False
        ), "a binary swapped in place (new mtime) must be re-probed"
        # A same-second replacement (sub-second mtime bump) must also re-probe:
        # second-resolution mtime would inherit the stale abort (reviewer.py P2).
        sec_ns = (binp.stat().st_mtime_ns // 1_000_000_000) * 1_000_000_000
        os.utime(p, ns = (sec_ns, sec_ns))
        LlamaCppBackend._record_tensor_split_abort(p, "m")
        binp.write_text("v2")
        os.utime(p, ns = (sec_ns, sec_ns + 1))
        assert (
            LlamaCppBackend._tensor_split_aborts(p, "m") is False
        ), "a same-second in-place swap (ns mtime bump) must be re-probed"
    finally:
        for key in list(LlamaCppBackend._tensor_split_abort_keys):
            if key and key[0] == p:
                LlamaCppBackend._tensor_split_abort_keys.discard(key)


def test_tensor_split_abort_raises_early_to_layer_fallback():
    """The first-spawn abort raises to the route's layer fallback (not the text-only
    mmproj strip), before the flash-attn-off retry, preserving the projector (#6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    raise_idx = src.find("(split-axis geometry); retrying with layer split")
    assert raise_idx != -1, "the split-axis abort must raise to trigger a layer retry"
    # raises before both the flash-attn-off retry and the text-only mmproj strip
    assert raise_idx < src.find("_with_flash_attn_off")
    assert raise_idx < src.find("_strip_mmproj_args(_last_spawn_cmd)")
    # gated on the marker-plus-crash helper, which also drives the record just above
    guard = src[max(0, raise_idx - 600) : raise_idx]
    assert "_should_record_tensor_split_abort" in guard
    rec_idx = src.find("_record_tensor_split_abort(binary, model_identifier)")
    assert rec_idx != -1 and rec_idx < raise_idx


def test_budget_downgrade_preserves_multi_gpu_intent():
    """The pooled-VRAM downgrade raises _layer_min_gpus from the usable tensor GPUs
    too, symmetric with the vision downgrade (reviewer.py asymmetric fix, #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    budget = src.find("_tp_weight_budget_mib <= _tp_required_mib")
    assert budget != -1
    block = src[budget : budget + 1000]
    assert "tensor_parallel = False" in block
    assert (
        "_layer_min_gpus = max(_layer_min_gpus, len(tp_gpus))" in block
    ), "the budget downgrade must preserve multi-GPU intent like the vision gate"


def test_compute_buffer_downgrade_preserves_multi_gpu_intent():
    """The len(tp_gpus) < 2 compute-buffer downgrade raises _layer_min_gpus from the
    full GPU set too, so it is symmetric with the budget/geometry downgrades and
    doesn't collapse a multi-GPU layer load to one card (reviewer.py P1 on #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    gate = src.find("plan_tp and len(tp_gpus) < 2")
    assert gate != -1
    # Bound to exactly this block: from its gate to the next (budget) downgrade.
    nxt = src.find("_tp_weight_budget_mib <= _tp_required_mib", gate)
    assert nxt != -1
    block = src[gate:nxt]
    assert "tensor_parallel = False" in block
    assert (
        "_layer_min_gpus = max(_layer_min_gpus, len(gpus))" in block
    ), "the compute-buffer downgrade must preserve multi-GPU intent like the others"


def test_tensor_split_layer_min_gpus_bump_requires_tensor_request():
    """Every guard that bumps _layer_min_gpus off the abort cache also tests
    tensor_parallel, so a non-tensor load on a known-bad binary doesn't grab every
    GPU for a fitting model (#6659)."""
    fn = _load_model_ast()
    checked = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            test_src = ast.unparse(node.test)
            if "self._tensor_split_aborts(binary, model_identifier)" not in test_src:
                continue
            body = "\n".join(ast.unparse(n) for n in node.body)
            if "_layer_min_gpus" in body:
                checked += 1
                assert "tensor_parallel" in test_src, (
                    "the cached _layer_min_gpus bump must require a current tensor "
                    f"request, but fires under `{test_src}`"
                )
    assert checked >= 1, "expected an abort-cache guard that bumps _layer_min_gpus"


# ── round-2 follow-up: route-fallback retry + auto-context cap + assert marker ──


def test_layer_fallback_retry_preserves_multi_gpu_intent():
    """load_model takes a preserve_multi_gpu_on_layer hint and raises _layer_min_gpus
    for it, so the tensor-off fallback retry still spreads a fitting model (#6659)."""
    sig = inspect.signature(LlamaCppBackend.load_model)
    assert "preserve_multi_gpu_on_layer" in sig.parameters
    assert sig.parameters["preserve_multi_gpu_on_layer"].default is False
    fn = _load_model_ast()
    found = any(
        isinstance(n, ast.If)
        and "preserve_multi_gpu_on_layer" in ast.unparse(n.test)
        and "_layer_min_gpus" in "\n".join(ast.unparse(b) for b in n.body)
        for n in ast.walk(fn)
    )
    assert found, "preserve_multi_gpu_on_layer must raise _layer_min_gpus"


def test_auto_context_layer_loops_capped_to_usable_gpus():
    """The auto-context loops bypass _select_gpus, so they apply its cap: a card
    counts only if usable VRAM clears the per-device layer overhead (#6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert (
        "range(max(1, _layer_min_gpus), len(ranked) + 1)" not in src
    ), "auto-context loops must cap _layer_min_gpus to usable GPUs, not use it raw"
    assert "_auto_min_gpus" in src
    assert "range(_auto_min_gpus, len(ranked) + 1)" in src
    # the eligibility threshold is the per-device layer overhead, not bare > 0
    auto = src.find("_auto_min_gpus = max(")
    assert auto != -1
    block = src[auto : auto + 400]
    assert "_pipeline_overhead_mib" in block, (
        "a card must clear the per-device layer overhead to count, mirroring "
        "_select_gpus, so a nearly-full GPU is not exposed and OOMs"
    )


def test_fallback_hint_uses_effective_tensor_request_not_just_toggle():
    """Tensor intent keys off _effective_tensor_parallel (toggle + extras + env), not
    just the toggle, so extra/env-driven tensor users keep multi-GPU (#6659)."""
    route = Path(_BACKEND_DIR) / "routes" / "inference.py"
    src = route.read_text()
    idx = src.find("_tensor_intent_overall = _effective_tensor_parallel(")
    assert idx != -1, "the GGUF load closure must compute tensor intent"
    block = src[idx : idx + 300]
    assert "extra_llama_args, request.tensor_parallel" in block
    pres = src.find("preserve_multi_gpu_on_layer = bool(")
    assert (
        "_effective_tensor_parallel(attempt_extra_args, tensor_parallel)" in src[pres : pres + 200]
    )
    # not the toggle-only form this replaced
    assert (
        "bool(\n                        request.tensor_parallel and not tensor_parallel" not in src
    )


def test_carry_preserved_tensor_intent_truth_table():
    """Behavioral check of the carry-forward decision: carried only for the SAME
    model, preserved, and not an explicit drop. Catches a `not` inversion (ctx-only
    collapse) and a missing same-model guard (cross-model leak) (#6659)."""
    inference_routes = _load_inference_routes_module()
    f = inference_routes._carry_preserved_tensor_intent
    assert f(preserved = True, same_model = True, explicit_drop = False) is True
    assert f(preserved = True, same_model = True, explicit_drop = True) is False  # explicit drop
    assert f(preserved = True, same_model = False, explicit_drop = False) is False  # model switch
    assert f(preserved = False, same_model = True, explicit_drop = False) is False  # not a fallback


def test_preserved_fallback_carried_across_non_drop_reload():
    """The hint carries the preserved fallback via _carry_preserved_tensor_intent,
    gated on the same model loaded, so a ctx-only reload keeps multi-GPU but a model
    switch / explicit drop doesn't inherit it (#6659)."""
    route = Path(_BACKEND_DIR) / "routes" / "inference.py"
    src = route.read_text()
    idx = src.find("_tensor_intent_overall = _effective_tensor_parallel(")
    assert idx != -1
    block = src[idx : idx + 400]
    assert "_carry_preserved_tensor_intent(" in block
    assert "preserved = llama_backend.layer_preserves_tensor_intent" in block
    assert "same_model = _same_model_loaded" in block
    assert "explicit_drop = _explicit_tensor_drop" in block


def test_same_model_guard_checks_path_and_variant():
    """The same-model guard matches the resolved config.identifier (what load_model
    stores, after from_identifier normalizes shorthands) -- not the raw request id --
    and also matches the loaded quant by path (local multi-variant dir) else variant (HF
    repo), so a reload keeps the carry-forward and a different variant doesn't inherit
    the prior one's preserved tensor intent (#6659)."""
    route = Path(_BACKEND_DIR) / "routes" / "inference.py"
    src = route.read_text()
    idx = src.find("_same_model_loaded = (")
    assert idx != -1
    block = src[idx : idx + 1300]
    # Identity compares the normalized config.identifier, not the raw model_identifier.
    head = src[idx : idx + 200]
    assert "config.identifier" in head and "== (model_identifier" not in head
    assert "llama_backend.gguf_path" in block and "config.gguf_file" in block
    assert "llama_backend.hf_variant" in block and "config.gguf_variant" in block


def test_diffusion_load_clears_preserved_tensor_flag():
    """The diffusion early-return path (skips the command builder) clears the
    preserved-fallback flag, so a prior tensor fallback doesn't churn it (#6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    diff = src.find("if self._is_diffusion:")
    assert diff != -1
    start = src.find("return self._start_diffusion_server", diff)
    assert start != -1
    assert "self._layer_preserves_tensor_intent = False" in src[diff:start]


def test_is_tensor_split_assert_marker():
    """Matches the specific #6415 split-axis assert, not any ggml assert/abort, so
    an unrelated invariant a corrupt GGUF/projector trips isn't cached (#6659)."""
    f = LlamaCppBackend._is_tensor_split_assert
    # the real #6415 warmup assert (split-axis enum, in ggml-backend-meta)
    assert (
        f(
            "ggml-backend-meta.cpp:541: GGML_ASSERT(src_ss[0].axis != "
            "GGML_BACKEND_SPLIT_AXIS_0) failed"
        )
        is True
    )
    # the split-axis token alone (file path elided / reworded) still matches
    assert f("GGML_ASSERT(x.axis != GGML_BACKEND_SPLIT_AXIS_1) failed") is True
    # UNRELATED asserts must NOT match -- including a different invariant from the
    # same multi-assert source file (matched on the token, not the file name).
    assert f("ggml-backend-meta.cpp:99: GGML_ASSERT(buf != NULL) failed") is False
    assert f("/x/ggml.c:1234: GGML_ASSERT(ne == 1) failed") is False
    assert f("ggml_abort: something else entirely") is False
    assert f("Segmentation fault (core dumped)") is False
    assert f("") is False
    assert f(None) is False


def test_layer_preserve_hint_replayed_on_respawn():
    """The preserve hint is in the replay snapshot (_pending_load_kwargs), so a
    respawn keeps the downgraded model multi-GPU (Codex review on #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    pend = src.find("_pending_load_kwargs = {")
    assert pend != -1
    block = src[pend : src.find("}", pend) + 1]
    assert '"preserve_multi_gpu_on_layer": preserve_multi_gpu_on_layer' in block, (
        "the layer-preserve hint must be in the replay snapshot so _respawn_if_dead "
        "keeps the multi-GPU placement"
    )


def test_should_record_tensor_split_abort_decision():
    """Behavioral check of marker AND (signal crash OR Windows abort), so an
    or->and typo or caching a generic crash fails here, not just the source pins."""
    f = LlamaCppBackend._should_record_tensor_split_abort
    marker = "ggml-backend-meta.cpp:541: GGML_ASSERT(x.axis != GGML_BACKEND_SPLIT_AXIS_0) failed"
    # marker + a hard crash records, across every platform's abort encoding
    assert f(-6, marker) is True  # POSIX SIGABRT
    assert f(-11, marker) is True  # POSIX SIGSEGV
    assert f(3, marker) is True  # Windows CRT abort() exit (not a signal)
    assert f(0xC0000005, marker) is True  # Windows NTSTATUS access violation
    # marker present but no hard crash -> not recorded
    assert f(0, marker) is False  # clean exit
    assert f(-9, marker) is False  # SIGKILL (OOM / unload), not a fault
    assert f(None, marker) is False  # still running
    # hard crash but not the split-axis marker -> not recorded (no over-caching)
    assert f(3, "some other failure") is False
    assert f(-6, "GGML_ASSERT(buf != NULL) failed") is False
    assert f(0xC0000005, "") is False


def test_fit_off_retry_skipped_on_split_axis_abort():
    """The fit-independent --fit off retry is skipped on the split-axis marker, else
    the model crashes a second time before the latch records it (reviewer.py, #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    retry = src.find('run_cmd = [*run_cmd, "--fit", "off"]')
    assert retry != -1
    guard = src[max(0, retry - 1000) : retry]
    assert "_fit_retry_allowed" in guard and "_startup_crashed" in guard
    assert (
        "not _split_axis_crash" in guard
    ), "the fit-off retry must be skipped when the crash is a split-axis abort"


def test_is_abort_exit_recognizes_windows_crt_abort():
    """exit code 3 (MSVC abort()) counts as a crash; signals / clean exits do not."""
    f = LlamaCppBackend._is_abort_exit
    assert f(3) is True
    assert f(0) is False
    assert f(-6) is False  # POSIX SIGABRT is handled by _is_signal_crash, not here
    assert f(None) is False


# ── tensor-off after a multi-GPU fallback forces a reload (route dedup) ─


class _NoopProcess:
    """Stand-in for Popen so is_loaded is True and atexit cleanup doesn't crash."""

    def terminate(self):
        pass

    def wait(self, timeout = None):
        return 0

    def kill(self):
        pass

    def poll(self):
        return 0


def _fallback_loaded_backend(layer_preserves_tensor_intent: bool) -> LlamaCppBackend:
    """A loaded backend in the tensor->layer fallback state (tensor off, --split-mode
    layer stored), differing only in the preserved-multi-GPU flag."""
    b = LlamaCppBackend()
    b._model_identifier = "owner/repo"
    b._requested_n_ctx = 0
    b._cache_type_kv = None
    b._tensor_parallel = False
    b._layer_preserves_tensor_intent = layer_preserves_tensor_intent
    b._extra_args = ["--split-mode", "layer"]
    b._requested_spec_mode = "auto"
    b._chat_template_override = None
    b._gguf_path = None
    return b


def test_tensor_off_echo_preserves_multi_gpu_fallback():
    """The Unsloth UI always sends tensor_parallel and echoes the /load response's
    resolved value, so after a fallback a ctx/settings reload carries tensor_parallel=
    false even though the user never changed it. That echo must NOT collapse the
    preserved multi-GPU placement -- it dedupes (Codex #6659)."""
    from models.inference import LoadRequest

    inference_routes = _load_inference_routes_module()

    req = LoadRequest(model_path = "owner/repo", tensor_parallel = False)
    assert "tensor_parallel" in req.model_fields_set, "the UI always sends the field"

    # Preserved fallback + bare tensor=false echo: dedupe, keep multi-GPU (no collapse).
    assert (
        inference_routes._request_matches_loaded_settings(
            req, _fallback_loaded_backend(layer_preserves_tensor_intent = True)
        )
        is True
    )
    # A genuine layer load (no preserved intent): tensor-off also dedupes, no churn.
    assert (
        inference_routes._request_matches_loaded_settings(
            req, _fallback_loaded_backend(layer_preserves_tensor_intent = False)
        )
        is True
    )


def test_explicit_split_mode_layer_extras_reloads_after_multi_gpu_fallback():
    """Tensor intent can be dropped via extras too: an explicit --split-mode layer
    matches the stored fallback extras but must still reload (reviewer.py P1, #6659)."""
    from models.inference import LoadRequest

    inference_routes = _load_inference_routes_module()

    req = LoadRequest(model_path = "owner/repo", llama_extra_args = ["--split-mode", "layer"])
    assert "llama_extra_args" in req.model_fields_set
    assert (
        inference_routes._request_matches_loaded_settings(
            req, _fallback_loaded_backend(layer_preserves_tensor_intent = True)
        )
        is False
    )


def test_tensor_off_reload_requires_explicit_toggle():
    """An Apply that doesn't touch the toggle (e.g. a context change) isn't churned
    by the preserved-fallback reload -- the working server is kept (Codex #6659)."""
    from models.inference import LoadRequest

    inference_routes = _load_inference_routes_module()

    req = LoadRequest(model_path = "owner/repo")  # tensor_parallel left unset
    assert "tensor_parallel" not in req.model_fields_set
    assert (
        inference_routes._request_matches_loaded_settings(
            req, _fallback_loaded_backend(layer_preserves_tensor_intent = True)
        )
        is True
    )


def test_tensor_off_under_env_tensor_does_not_reload_loop(monkeypatch):
    """With LLAMA_ARG_SPLIT_MODE=tensor set, a tensor-off request can't drop tensor
    intent, so the env-aware guard dedupes instead of reload-looping (Codex #6659)."""
    from models.inference import LoadRequest

    inference_routes = _load_inference_routes_module()
    monkeypatch.setenv("LLAMA_ARG_SPLIT_MODE", "tensor")

    req = LoadRequest(model_path = "owner/repo", tensor_parallel = False)
    assert "tensor_parallel" in req.model_fields_set
    # env still forces tensor -> not a real drop -> dedupe (no reload loop).
    assert (
        inference_routes._request_matches_loaded_settings(
            req, _fallback_loaded_backend(layer_preserves_tensor_intent = True)
        )
        is True
    )


def test_is_explicit_tensor_drop_truth_table():
    """Only an explicit non-tensor --split-mode override is a drop. A bare
    tensor_parallel field (the UI always sends it and echoes the fallback's false), an
    empty clear, an unrelated extra (--top-k), or inherit (None) must NOT collapse a
    preserved fallback; --split-mode tensor / tensor_parallel=true re-engage (Codex
    #6659)."""
    from models.inference import LoadRequest

    f = _load_inference_routes_module()._is_explicit_tensor_drop
    # A non-tensor split-mode override is the one deliberate departure -> drop.
    assert (
        f(LoadRequest(model_path = "owner/repo", llama_extra_args = ["--split-mode", "layer"])) is True
    )
    # tensor / retry re-engages, never a drop.
    assert (
        f(LoadRequest(model_path = "owner/repo", llama_extra_args = ["--split-mode", "tensor"]))
        is False
    )
    # A bare tensor_parallel field is the UI echo, not a drop (would collapse on reload).
    assert f(LoadRequest(model_path = "owner/repo", tensor_parallel = False)) is False
    assert f(LoadRequest(model_path = "owner/repo", tensor_parallel = True)) is False
    # Unrelated extra / empty clear / inherit all keep the preserved placement.
    assert f(LoadRequest(model_path = "owner/repo", llama_extra_args = ["--top-k", "20"])) is False
    assert f(LoadRequest(model_path = "owner/repo", llama_extra_args = [])) is False
    assert f(LoadRequest(model_path = "owner/repo")) is False


def test_explicit_tensor_drop_uses_shared_helper_in_both_readers():
    """Both the already-loaded dedup and the load carry-forward derive the drop from
    _is_explicit_tensor_drop, so they agree on what counts as a drop -- a reload for
    an unrelated extra still carries the preserved intent rather than collapsing to one
    GPU (Codex #6659)."""
    src = (Path(_BACKEND_DIR) / "routes" / "inference.py").read_text()
    # Dedup reader (the preserved-fallback reload guard).
    assert "layer_preserves_tensor_intent and _is_explicit_tensor_drop(request)" in src
    # Load carry-forward reader feeds the same decision into the carry-forward.
    assert "_explicit_tensor_drop = _is_explicit_tensor_drop(request)" in src


def test_layer_preserves_tensor_intent_set_only_on_preserved_downgrade():
    """load_model latches the flag from _layer_min_gpus (raised only when a tensor
    request is downgraded but kept multi-GPU), and clears it when tensor stays on."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    on = src.find("self._tensor_parallel = True")
    off = src.find("self._tensor_parallel = False")
    assert 0 <= on and 0 <= off
    assert "self._layer_preserves_tensor_intent = False" in src[on : on + 120]
    assert "self._layer_preserves_tensor_intent = _layer_min_gpus > 1" in src[off : off + 400]


def test_layer_min_gpus_bound_before_gpu_selection_try():
    """_layer_min_gpus is bound before the GPU-selection try, so the --fit-on except
    path can't UnboundLocalError when the command builder reads it (Codex #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert src.count("_layer_min_gpus = 1") == 1, "exactly one init, before the try"
    init = src.find("_layer_min_gpus = 1")
    try_body = src.find("gguf_size = self._get_gguf_size_bytes")
    fit_except = src.find("GPU selection failed")
    use_after = src.find("self._layer_preserves_tensor_intent = _layer_min_gpus > 1")
    assert (
        -1 < init < try_body < fit_except < use_after
    ), "the init must precede the try body, the except, and the command-builder use"


def test_already_in_target_state_reloads_on_tensor_off_after_fallback():
    """The backend fast path mirrors the route dedup: a preserved fallback reloads on
    an EXPLICIT tensor-off request, but an implicit same-settings reload (carry-forward
    preserve_multi_gpu_on_layer=True) still dedupes (Codex #6659)."""

    def _backend(layer_preserves: bool) -> LlamaCppBackend:
        b = _fallback_loaded_backend(layer_preserves_tensor_intent = layer_preserves)
        b._process = _NoopProcess()
        b._healthy = True
        return b

    kwargs = dict(
        gguf_path = None,
        mtp_draft_path = None,
        model_identifier = "owner/repo",
        hf_variant = None,
        n_ctx = 0,
        cache_type_kv = None,
        speculative_type = None,
        spec_draft_n_max = None,
        tensor_parallel = False,
        chat_template_override = None,
        extra_args = ["--split-mode", "layer"],
        is_vision = False,
    )
    # Preserved fallback + EXPLICIT tensor drop -> reload (not already in target state).
    assert _backend(True)._already_in_target_state(**kwargs) is False
    # Same preserved fallback but an implicit reload that carries the intent forward
    # (HF auto-pick / local-dir flows skip the route guard and reach here) -> dedupe.
    assert (
        _backend(True)._already_in_target_state(**kwargs, preserve_multi_gpu_on_layer = True) is True
    )
    # A genuine layer load (no preserved intent) -> dedupe, no churn.
    assert _backend(False)._already_in_target_state(**kwargs) is True


# ── route dedup: host-residency memory mode + gpu_ids device strip (#7164/#7188) ─


def _mem_loaded_backend(
    *,
    memory_mode,
    extra_args,
    launched_with_inherited_mem_env = False,
):
    """A loaded GGUF backend in a given memory-placement state, for the route dedup."""
    b = LlamaCppBackend()
    b._model_identifier = "owner/repo"
    b._requested_n_ctx = 0
    b._cache_type_kv = None
    b._tensor_parallel = False
    b._layer_preserves_tensor_intent = False
    b._extra_args = list(extra_args) if extra_args else None
    b._requested_spec_mode = "auto"
    b._chat_template_override = None
    b._gguf_path = None
    b._gpu_ids = None
    b._memory_mode = memory_mode
    b._launched_with_inherited_mem_env = launched_with_inherited_mem_env
    return b


def test_explicit_auto_reloads_over_passthrough_mlock_in_extras():
    """A server loaded with no memory_mode keeps a pass-through --mlock and is still
    mlocked. An explicit "auto" repeating --mlock must NOT dedupe: stripping only the
    request side keeps the backend's --mlock visible, so the matcher reloads and the
    scrub runs (Codex #7164)."""
    from models.inference import LoadRequest

    inference_routes = _load_inference_routes_module()

    req = LoadRequest(
        model_path = "owner/repo",
        gguf_memory_mode = "auto",
        llama_extra_args = ["--mlock"],
    )
    assert "gguf_memory_mode" in req.model_fields_set
    backend = _mem_loaded_backend(memory_mode = None, extra_args = ["--mlock"])
    assert inference_routes._request_matches_loaded_settings(req, backend) is False


def test_explicit_null_memory_mode_dedupes_over_passthrough_mlock():
    """An explicit gguf_memory_mode=null (a client echoing the status response) is "no
    opinion", not a mode change. Pydantic marks it set, but the dedup gates the strip on
    the VALUE: null must NOT strip the request's --mlock, so a status-hydrated Apply
    dedupes to the running server (#7188)."""
    from models.inference import LoadRequest

    inference_routes = _load_inference_routes_module()

    req = LoadRequest(
        model_path = "owner/repo",
        gguf_memory_mode = None,
        llama_extra_args = ["--mlock"],
    )
    # Pydantic marks an explicit null as set -- why gating on model_fields_set was wrong.
    assert "gguf_memory_mode" in req.model_fields_set
    backend = _mem_loaded_backend(memory_mode = None, extra_args = ["--mlock"])
    assert inference_routes._request_matches_loaded_settings(req, backend) is True


def test_explicit_pinned_dedupes_when_flags_already_applied():
    """A server loaded WITH memory_mode="pinned" already had --mlock stripped from its
    extras. A repeat pinned Apply re-sending --mlock must still dedupe: the request-side
    strip makes it compare equal to the empty backend extras (#7164)."""
    from models.inference import LoadRequest

    inference_routes = _load_inference_routes_module()

    req = LoadRequest(
        model_path = "owner/repo",
        gguf_memory_mode = "pinned",
        llama_extra_args = ["--mlock"],
    )
    backend = _mem_loaded_backend(memory_mode = "pinned", extra_args = None)
    assert inference_routes._request_matches_loaded_settings(req, backend) is True


def test_explicit_gpu_ids_dedupes_when_device_already_stripped():
    """A GGUF loaded with explicit gpu_ids had a user --device stripped from its stored
    extras. A repeat identical request re-sending --device must still dedupe: the request-
    side strip (gated on gpu_ids) compares equal to the stripped backend extras, so the
    load hits the fast path instead of a needless reload / training 409 (#7188)."""
    from models.inference import LoadRequest

    inference_routes = _load_inference_routes_module()

    req = LoadRequest(
        model_path = "owner/repo",
        gpu_ids = [0],
        llama_extra_args = ["--device", "Vulkan3", "--top-k", "5"],
    )
    backend = _mem_loaded_backend(memory_mode = None, extra_args = ["--top-k", "5"])
    backend._gpu_ids = [0]
    assert inference_routes._request_matches_loaded_settings(req, backend) is True


def test_empty_gpu_ids_dedupes_without_stripping_device():
    """gpu_ids=[] is documented as auto (same as omitting): the load path normalizes it to
    None and keeps its --device, so the request-side strip must be gated on an EFFECTIVE pin.
    An empty list re-sending --device must still dedupe against a server loaded with no pin
    that kept its --device, not force a needless reload / training 409 (#7188)."""
    from models.inference import LoadRequest

    inference_routes = _load_inference_routes_module()

    req = LoadRequest(
        model_path = "owner/repo",
        gpu_ids = [],
        llama_extra_args = ["--device", "Vulkan3", "--top-k", "5"],
    )
    backend = _mem_loaded_backend(
        memory_mode = None, extra_args = ["--device", "Vulkan3", "--top-k", "5"]
    )
    backend._gpu_ids = None
    assert inference_routes._request_matches_loaded_settings(req, backend) is True
