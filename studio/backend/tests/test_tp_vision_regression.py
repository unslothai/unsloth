# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression guards for silent tensor-parallel downgrades in load_model.

Context
-------
PR #6416 ("skip tensor-parallel for vision models") added a blanket

    if tensor_parallel and effective_is_vision:
        tensor_parallel = False

to dodge a `GGML_ASSERT` crash with `--split-mode tensor` + `--mmproj` on an
older llama.cpp build + consumer Blackwell (sm_120). Side effect: ANY GGUF repo
that ships an mmproj projector (e.g. `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`) was
treated as vision, so an explicit `tensor_parallel=true` was silently dropped and
-- on hardware where the model fits on one GPU -- collapsed to a SINGLE GPU.

The fix makes the skip *self-healing*: tensor is attempted for vision models
(it works on current builds, verified on B200/sm_100), and a binary is recorded
as incompatible only after it actually aborts (`_record_vision_tensor_split_abort`
/ `_vision_tensor_split_aborts`), so the upfront skip applies just where needed.
`_select_gpus(min_gpus=...)` lets a downgraded TP request keep multiple GPUs.

`load_model` is too entangled (subprocess + GPU probe) to drive end-to-end, so
like the rest of the suite these tests inspect the source / drive the pure
helpers. The headline test pins the *set* of conditions under which a user's
tensor-parallel request may be dropped, so a NEW silent drop (the shape of the
#6416 regression) fails CI and forces review. No GPU; fully deterministic.
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

# Same external-dep stubs as the other llama_cpp unit tests so importing the
# backend doesn't drag in structlog / httpx / loggers.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)
_httpx_stub = _types.ModuleType("httpx")
for _exc in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
_httpx_stub.Timeout = type("T", (), {"__init__": lambda s, *a, **k: None})
_httpx_stub.Client = type(
    "C",
    (),
    {"__init__": lambda s, **kw: None, "__enter__": lambda s: s, "__exit__": lambda s, *a: None},
)
sys.modules.setdefault("httpx", _httpx_stub)

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


# Every condition under which load_model may flip a requested tensor_parallel=True
# back to False. Adding a new one (as #6416 did with the blanket vision gate) MUST
# be a conscious change: update this allowlist AND keep multi-GPU execution where
# possible (or surface it), then extend the assertions below.
_ALLOWED_TP_DROP_GUARDS = {
    # Capability/policy: --split-mode tensor + --mmproj aborted on some builds
    # (#6415). Self-healing -- attempted by default, skipped only on a binary
    # already seen to abort this session (replaces #6416's blanket skip).
    "tensor_parallel and effective_is_vision and self._vision_tensor_split_aborts(binary)",
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


def test_vision_gate_is_self_healing_not_blanket():
    """The vision skip must be conditional on a recorded per-binary abort, never a
    blanket disable on is_vision / effective_is_vision alone.

    Guards against regressing to #6416's behavior (single-GPU for every mmproj GGUF).
    """
    src = inspect.getsource(LlamaCppBackend.load_model)
    assert "effective_is_vision" in src
    assert "self._vision_tensor_split_aborts(binary)" in src
    # not a blanket disable on bare is_vision or effective_is_vision alone
    assert "if tensor_parallel and is_vision:" not in src
    assert "if tensor_parallel and effective_is_vision:" not in src


def test_vision_skip_documents_layer_split_fallback():
    """When the vision skip does fire (known-bad binary), it states the fallback."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    gate = src.find("self._vision_tensor_split_aborts(binary)")
    assert gate != -1
    block = src[gate : gate + 600]
    assert "layer split" in block, "the vision skip should state it falls back to layer split"


def test_vision_tensor_abort_recorded_only_after_retries_and_signature():
    """The binary is recorded vision-tensor incompatible only AFTER every startup
    retry fails (fit-off, flash-attn-off, MTP-drop) and the crash is a hard signal
    fault with no non-tensor cause (OOM / unknown arch). This excludes the benign
    `--fit` step abort (the fit-off retry resolves it, so we never reach the record)
    and unrelated crashes -- the two ways the first naive version mis-cached a
    capable binary (Codex review on #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    idx = src.find("_record_vision_tensor_split_abort(binary)")
    assert idx != -1, "load_model must record a binary that aborts on vision + tensor"
    guard = src[max(0, idx - 500) : idx]
    assert "self._tensor_parallel" in guard
    assert "launched_with_mmproj" in guard
    assert "_is_signal_crash" in guard, "record must be gated on a hard signal crash"
    assert "_is_tensor_split_assert" in guard, (
        "record must be confirmed by the ggml tensor-split assert marker, not a bare "
        "signal crash shared with the projector-incompat branch"
    )
    assert (
        "not self._output_has_nonprojector_diagnostic" in guard
    ), "record must exclude OOM / unknown-arch crashes"
    # The record sits in the post-all-retries failure block: after the MTP-drop retry.
    mtp_retry = src.find("retrying without speculative decoding")
    assert 0 <= mtp_retry < idx, "recording must come after the startup-retry ladder"


def test_vision_downgrade_preserves_multi_gpu_intent():
    """When the cached vision gate downgrades a tensor request, it raises
    _layer_min_gpus and threads it into the layer GPU selection, so a model that
    fits on one GPU is still spread instead of collapsing to a single device."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    # gate records the multi-GPU intent
    assert "_layer_min_gpus = max(_layer_min_gpus, len(gpus))" in src
    # and the layer selection consumes it: explicit/file-size paths via _select_gpus,
    # the auto-context loops via the usable-capped _auto_min_gpus derived from it.
    assert src.count("min_gpus = _layer_min_gpus") >= 2
    assert "range(_auto_min_gpus, len(ranked) + 1)" in src
    # the auto-context minimum is derived from _layer_min_gpus (capped to usable)
    auto = src.find("_auto_min_gpus = max(")
    assert auto != -1 and "_layer_min_gpus" in src[auto : auto + 200]


# ── per-binary capability cache (pure) ───────────────────────────────


def test_vision_tensor_attempted_by_default_for_unknown_binary():
    """A binary not seen to abort -> tensor parallelism is attempted (not skipped)."""
    assert LlamaCppBackend._vision_tensor_split_aborts("/never/seen/llama-server") is False
    assert LlamaCppBackend._vision_tensor_split_aborts(None) is False


def test_recorded_vision_tensor_abort_makes_gate_skip():
    """After a vision+tensor abort is recorded, the gate predicate trips for it."""
    b = f"/tmp/llama-server-{id(object())}"
    try:
        assert LlamaCppBackend._vision_tensor_split_aborts(b) is False
        LlamaCppBackend._record_vision_tensor_split_abort(b)
        assert LlamaCppBackend._vision_tensor_split_aborts(b) is True
    finally:
        LlamaCppBackend._vision_tensor_abort_binaries.discard(
            LlamaCppBackend._vision_binary_cache_key(b)
        )


# ── _select_gpus: single-GPU collapse vs honored multi-GPU intent (pure) ──


def test_select_gpus_collapses_to_single_gpu_when_model_fits():
    """Default (min_gpus=1): a 39 GB model on four 183 GB GPUs pins ONE GPU.

    This is the runtime root of the 'loads on a single GPU' symptom once TP is
    dropped -- and why the downgrade needs min_gpus to honor a TP request.
    """
    gpus = [(0, 180000), (1, 180000), (2, 180000), (3, 180000)]  # (idx, free MiB)
    gpu_indices, _use_fit = LlamaCppBackend._select_gpus(int(39 * _GB), gpus)
    assert gpu_indices is not None and len(gpu_indices) == 1


def test_select_gpus_min_gpus_keeps_multi_gpu_for_fitting_model():
    """Fix: an explicit multi-GPU/TP intent (min_gpus>=2) must NOT collapse to one
    GPU for a model that happens to fit on one. (Was xfail before the fix.)"""
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
    """min_gpus must not force a nearly-full card just to hit the count.

    A downgraded multi-GPU request on a host with 2 free + 1 nearly-full GPU caps
    to the 2 usable cards rather than pulling in the full one (which would OOM) or
    tripping --fit (Codex review on #6659: base the layer minimum on usable GPUs).
    """
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


def test_vision_abort_cache_invalidated_on_binary_mtime_change(tmp_path):
    """The vision tensor-abort cache keys on (path, mtime), so a binary swapped in
    place by the in-app updater (POST /api/llama/update, no backend restart) is
    re-probed instead of inheriting the old build's abort (Codex review on #6659).
    """
    binp = tmp_path / "llama-server"
    binp.write_text("v1")
    p = str(binp)
    try:
        LlamaCppBackend._record_vision_tensor_split_abort(p)
        assert LlamaCppBackend._vision_tensor_split_aborts(p) is True
        # Simulate an in-place update bumping the binary's mtime.
        st = binp.stat()
        os.utime(p, (st.st_atime, st.st_mtime + 10))
        assert LlamaCppBackend._vision_tensor_split_aborts(p) is False, (
            "a binary swapped in place (new mtime) must be re-probed, not inherit "
            "the old build's abort"
        )
    finally:
        for key in list(LlamaCppBackend._vision_tensor_abort_binaries):
            if key and key[0] == p:
                LlamaCppBackend._vision_tensor_abort_binaries.discard(key)


def test_tensor_mmproj_abort_retries_layer_split_not_text_only():
    """A tensor + --mmproj GGML_ASSERT hands back to the route-level tensor->layer
    fallback (raise) so the projector is retried with layer split, instead of
    stripping --mmproj and silently loading text-only on the first load.

    Unanimous P1 (reviewer.py [5/5] + Codex on #6659): the abort signature also
    matches the projector-incompatibility branch, so without raising first the
    first vision load on a bad binary loses vision until the next (cached) load.
    """
    src = inspect.getsource(LlamaCppBackend.load_model)
    raise_idx = src.find("retrying with layer split (projector preserved)")
    assert raise_idx != -1, "the tensor+mmproj abort must raise to trigger a layer retry"
    strip_idx = src.find("_strip_mmproj_args(_last_spawn_cmd)")
    assert strip_idx != -1
    # The raise precedes the text-only mmproj-strip path: the abort never falls
    # through to stripping vision.
    assert raise_idx < strip_idx
    # The raise is gated on the tensor+mmproj crash signature, not any failure,
    # and the same signature drives the cache record.
    guard = src[max(0, raise_idx - 400) : raise_idx]
    assert "vision_tensor_split_crash" in guard
    rec_idx = src.find("_record_vision_tensor_split_abort(binary)")
    assert rec_idx != -1 and rec_idx < raise_idx


def test_budget_downgrade_preserves_multi_gpu_intent():
    """The pooled-VRAM tensor downgrade raises _layer_min_gpus from the usable
    tensor GPUs, symmetric with the vision downgrade, so a multi-GPU request that
    can't fit weights+reserve in tensor mode still layer-splits across cards
    instead of collapsing to one (reviewer.py asymmetric-fix finding on #6659).
    """
    src = inspect.getsource(LlamaCppBackend.load_model)
    budget = src.find("_tp_weight_budget_mib <= _tp_required_mib")
    assert budget != -1
    block = src[budget : budget + 1000]
    assert "tensor_parallel = False" in block
    assert (
        "_layer_min_gpus = max(_layer_min_gpus, len(tp_gpus))" in block
    ), "the budget downgrade must preserve multi-GPU intent like the vision gate"


def test_vision_layer_min_gpus_bump_requires_tensor_request():
    """The cached-vision _layer_min_gpus bump must be gated on a CURRENT tensor
    request, not the abort cache alone.

    Otherwise a plain non-tensor vision load on a binary that aborted earlier this
    session would grab every GPU (min_gpus = len(gpus)) for a model that fits on
    one card, just because an unrelated TP attempt failed before (Codex review on
    #6659). So every guard that bumps _layer_min_gpus off the vision abort cache
    also tests tensor_parallel.
    """
    fn = _load_model_ast()
    checked = 0
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            test_src = ast.unparse(node.test)
            if "self._vision_tensor_split_aborts(binary)" not in test_src:
                continue
            body = "\n".join(ast.unparse(n) for n in node.body)
            if "_layer_min_gpus" in body:
                checked += 1
                assert "tensor_parallel" in test_src, (
                    "the cached-vision _layer_min_gpus bump must require a current "
                    f"tensor request, but fires under `{test_src}`"
                )
    assert checked >= 1, "expected a vision-cache guard that bumps _layer_min_gpus"


# ── round-2 follow-up: route-fallback retry + auto-context cap + assert marker ──


def test_layer_fallback_retry_preserves_multi_gpu_intent():
    """The route-level tensor->layer fallback retry runs tensor-off, so load_model
    takes a preserve_multi_gpu_on_layer hint and raises _layer_min_gpus when set --
    otherwise the first successful fallback of a fits-on-one-card model single-GPUs
    despite the user's tensor request (Codex review on #6659)."""
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
    """The auto-context layer loops bypass _select_gpus, so they apply their own
    usable-GPU cap: a raised _layer_min_gpus must not force a card that can't pay
    its per-device layer overhead into the subset (Codex review on #6659). The cap
    mirrors _select_gpus: usable VRAM must exceed the per-device overhead, not just
    be positive."""
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
    """The preserve_multi_gpu_on_layer hint keys off the EFFECTIVE tensor request
    (toggle, --split-mode tensor in extras, or inherited env) via
    _effective_tensor_parallel, not just request.tensor_parallel -- otherwise an
    extra/env-driven tensor user's downgraded load single-GPUs (Codex review on
    #6659)."""
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
    """The marker matches the SPECIFIC #6415 split-axis assertion, not any ggml
    assert/abort -- so an unrelated invariant a corrupt GGUF/projector trips with
    --mmproj present is not cached as tensor/mmproj-incompatible (Codex review on
    #6659)."""
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
    # an UNRELATED ggml assert with --mmproj present must NOT match
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


def test_vision_tensor_abort_requires_assert_marker_for_record_and_raise():
    """Both the abort cache record and the layer-retry raise are gated on the ggml
    assert marker, so a projector that SIGSEGVs independent of split mode is not
    cached as tensor/mmproj-incompatible nor sent through the tensor layer retry
    (Codex review on #6659)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    idx = src.find("vision_tensor_split_crash = (")
    assert idx != -1
    block = src[idx : idx + 400]
    assert "_is_tensor_split_assert" in block
    assert "_is_signal_crash" in block
