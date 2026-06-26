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
    # Capability/policy: --split-mode tensor aborted for this model on this binary
    # (#6415 split-axis geometry). Self-healing -- attempted by default, skipped
    # only for a (binary, model) already seen to abort (replaces #6416's blanket
    # vision skip).
    "tensor_parallel and self._tensor_split_aborts(binary, model_identifier)",
    # Capacity: tensor parallelism needs >= 2 GPUs that clear the compute-buffer reserve.
    "tensor_parallel and len(tp_gpus) < 2",
    # Capacity: pooled usable VRAM can't hold weights + MTP reserve -> layer split.
    "_tp_weight_budget_mib <= _tp_required_mib",
}


def test_tensor_parallel_drop_sites_match_allowlist():
    """The set of reasons a requested TP can be dropped is fixed and reviewed.

    This is the guard that would have flagged #6416: it added a brand-new
    `tensor_parallel = False` site, which would make this set-equality fail until
    a reviewer consciously allowlisted it (and preserved multi-GPU).
    """
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
    """The tensor skip is conditional on a recorded (binary, model) abort, never a
    blanket disable on is_vision / effective_is_vision (the #6416 regression that
    single-GPU'd every mmproj GGUF)."""
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
    """The abort is recorded on the FIRST spawn that shows the split-axis signature,
    BEFORE the flash-attn-off retry -- which can't run tensor (SPLIT_MODE_TENSOR
    requires flash_attn) so its output drops the :541 marker. Recording after the
    ladder never sees it and the crash loop repeats every load (oobabooga, #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    idx = src.find("_record_tensor_split_abort(binary, model_identifier)")
    assert idx != -1, "load_model must record a (binary, model) tensor-split abort"
    guard = src[max(0, idx - 400) : idx]
    assert "self._tensor_parallel" in guard
    assert "_is_signal_crash" in guard, "record must be gated on a hard signal crash"
    assert "_is_tensor_split_assert" in guard, "record must be confirmed by the :541 marker"
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
        assert LlamaCppBackend._tensor_split_aborts(p, "m") is False, (
            "a binary swapped in place (new mtime) must be re-probed"
        )
    finally:
        for key in list(LlamaCppBackend._tensor_split_abort_keys):
            if key and key[0] == p:
                LlamaCppBackend._tensor_split_abort_keys.discard(key)


def test_tensor_split_abort_raises_early_to_layer_fallback():
    """The first-spawn split-axis abort raises (route fallback retries layer split)
    rather than falling through to the text-only mmproj-strip path, and it does so
    BEFORE the flash-attn-off retry, so the projector/vision is preserved and the
    futile retry ladder is skipped (#6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    raise_idx = src.find("(split-axis geometry); retrying with layer split")
    assert raise_idx != -1, "the split-axis abort must raise to trigger a layer retry"
    # raises before both the flash-attn-off retry and the text-only mmproj strip
    assert raise_idx < src.find("_with_flash_attn_off")
    assert raise_idx < src.find("_strip_mmproj_args(_last_spawn_cmd)")
    # gated on the :541 signature, which also drives the record just above
    guard = src[max(0, raise_idx - 400) : raise_idx]
    assert "_is_tensor_split_assert" in guard
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
    assert "_layer_min_gpus = max(_layer_min_gpus, len(tp_gpus))" in block, (
        "the budget downgrade must preserve multi-GPU intent like the vision gate"
    )


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
    assert "range(max(1, _layer_min_gpus), len(ranked) + 1)" not in src, (
        "auto-context loops must cap _layer_min_gpus to usable GPUs, not use it raw"
    )
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
    """The hint keys off _effective_tensor_parallel (toggle + extras + env), not
    just the toggle, so extra/env-driven tensor users keep multi-GPU (#6659)."""
    route = Path(_BACKEND_DIR) / "routes" / "inference.py"
    src = route.read_text()
    idx = src.find("preserve_multi_gpu_on_layer = bool(")
    assert idx != -1, "the GGUF load closure must pass the hint"
    block = src[idx : idx + 300]
    assert "_effective_tensor_parallel(extra_llama_args, request.tensor_parallel)" in block
    assert "_effective_tensor_parallel(attempt_extra_args, tensor_parallel)" in block
    # not the toggle-only form this replaced
    assert (
        "bool(\n                        request.tensor_parallel and not tensor_parallel" not in src
    )


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
    """The preserve_multi_gpu_on_layer hint is in the replay snapshot
    (_pending_load_kwargs), so a respawn after llama-server dies keeps the
    downgraded model multi-GPU instead of silently coming back single-GPU
    (Codex review on #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    pend = src.find("_pending_load_kwargs = {")
    assert pend != -1
    block = src[pend : src.find("}", pend) + 1]
    assert '"preserve_multi_gpu_on_layer": preserve_multi_gpu_on_layer' in block, (
        "the layer-preserve hint must be in the replay snapshot so _respawn_if_dead "
        "keeps the multi-GPU placement"
    )


def test_tensor_split_record_requires_signal_crash_and_marker():
    """The record is gated on both a hard signal crash and the split-axis marker, so
    a projector that SIGSEGVs independent of split mode, or an unrelated assert, is
    not cached (Codex review on #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    idx = src.find("_record_tensor_split_abort(binary, model_identifier)")
    assert idx != -1
    guard = src[max(0, idx - 400) : idx]
    assert "_is_signal_crash" in guard
    assert "_is_tensor_split_assert" in guard


# ── tensor-off after a multi-GPU fallback forces a reload (route dedup) ─


def _fallback_loaded_backend(layer_preserves_tensor_intent: bool) -> LlamaCppBackend:
    """A loaded backend in the tensor->layer fallback state: tensor reports off and
    --split-mode layer is stored, differing only in whether the placement was kept
    multi-GPU to honor a (now-downgraded) tensor request."""
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


def test_explicit_tensor_off_reloads_after_multi_gpu_fallback():
    """A tensor->layer fallback (preserve_multi_gpu_on_layer) spans all GPUs while
    reporting tensor=off. An explicit tensor-off Apply must reload so placement
    re-selects (single GPU for a 1-GPU-fit model), not dedupe to the all-GPU mask;
    a genuine layer load (no preserved intent) still dedupes (Codex review #6659)."""
    from models.inference import LoadRequest
    from routes import inference as inference_routes

    req = LoadRequest(model_path = "owner/repo", tensor_parallel = False)
    assert "tensor_parallel" in req.model_fields_set, "the toggle must be explicit"

    # Preserved multi-GPU fallback: the toggle going off drops that intent -> reload.
    assert (
        inference_routes._request_matches_loaded_settings(
            req, _fallback_loaded_backend(layer_preserves_tensor_intent = True)
        )
        is False
    )
    # A deliberate layer load (not a tensor fallback): tensor-off dedupes, no churn.
    assert (
        inference_routes._request_matches_loaded_settings(
            req, _fallback_loaded_backend(layer_preserves_tensor_intent = False)
        )
        is True
    )


def test_tensor_off_reload_requires_explicit_toggle():
    """An Apply that does not set the tensor toggle (e.g. only a context change)
    must not be churned by the preserved-fallback reload -- the working multi-GPU
    layer server is kept (Codex review #6659)."""
    from models.inference import LoadRequest
    from routes import inference as inference_routes

    req = LoadRequest(model_path = "owner/repo")  # tensor_parallel left unset
    assert "tensor_parallel" not in req.model_fields_set
    assert (
        inference_routes._request_matches_loaded_settings(
            req, _fallback_loaded_backend(layer_preserves_tensor_intent = True)
        )
        is True
    )


def test_layer_preserves_tensor_intent_set_only_on_preserved_downgrade():
    """load_model latches the flag from _layer_min_gpus (raised only when a tensor
    request is downgraded but kept multi-GPU), and clears it when tensor stays on."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    on = src.find("self._tensor_parallel = True")
    off = src.find("self._tensor_parallel = False")
    assert 0 <= on and 0 <= off
    assert "self._layer_preserves_tensor_intent = False" in src[on : on + 120]
    assert "self._layer_preserves_tensor_intent = _layer_min_gpus > 1" in src[off : off + 400]
