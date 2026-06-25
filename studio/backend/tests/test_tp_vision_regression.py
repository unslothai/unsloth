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


def test_vision_tensor_abort_is_recorded_on_startup_crash():
    """A tensor + --mmproj launch that crashes at startup records the binary, so
    later vision loads skip tensor upfront instead of crashing every time (#6415)."""
    src = inspect.getsource(LlamaCppBackend.load_model)
    idx = src.find("_record_vision_tensor_split_abort(binary)")
    assert idx != -1, "load_model must record a binary that aborts on vision + tensor"
    guard = src[max(0, idx - 300) : idx]
    assert "_startup_crashed" in guard
    assert "self._tensor_parallel" in guard
    assert "launched_with_mmproj" in guard


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
        LlamaCppBackend._vision_tensor_abort_binaries.discard(b)


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
