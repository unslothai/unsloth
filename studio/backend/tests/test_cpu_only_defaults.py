# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Guards for CPU-only safe llama-server defaults in load_model.

On a CPU-only host (no discrete GPU, not Apple Metal) none of the GPU/Apple fit
branches run, so the launch used to keep GPU defaults: `--flash-attn on`, `--fit on`,
and the model's full native context. For large MLA/sparse-attention/MTP GGUFs (e.g.
GLM-5.2) `--fit`'s graph-reserve estimator aborts (llama.cpp #21932) and CPU flash
attention is unsafe. These tests pin that the launch now derives `_cpu_only` and uses
it to disable --fit and default flash-attn off, while leaving GPU/Apple behaviour and
user overrides intact. Source/AST level: load_model is too entangled to drive E2E.
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

try:
    import structlog  # noqa: F401
except ImportError:
    _s = _types.ModuleType("structlog")
    _s.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    _s.BoundLogger = type("BoundLogger", (), {})
    sys.modules["structlog"] = _s
# Importing core.inference.* runs core/inference/__init__.py (orchestrator + loggers +
# httpx); stub those when absent so a dependency-light run can still collect this file.
try:
    import loggers  # noqa: F401
except ImportError:
    _loggers_stub = _types.ModuleType("loggers")
    _loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
    sys.modules["loggers"] = _loggers_stub
try:
    import httpx  # noqa: F401
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


def _load_model_src() -> str:
    return textwrap.dedent(inspect.getsource(LlamaCppBackend.load_model))


def test_cpu_only_flag_derived_from_no_gpu_and_not_apple():
    src = _load_model_src()
    assert "_cpu_only = (not gpus) and not _is_apple_silicon()" in src


def test_flash_attn_default_is_off_on_cpu_only():
    """The base cmd must use a computed flash default, off for CPU-only."""
    src = _load_model_src()
    assert '_flash_default = "off" if _cpu_only else "on"' in src
    # And the base cmd must NOT hardcode flash-attn on anymore.
    assert '"on",  # Force flash attention for speed' not in src
    # The --flash-attn argument in the base cmd list is the computed default.
    assert "_flash_default,  # CPU-only" in src


def test_fit_disabled_on_cpu_only():
    src = _load_model_src()
    assert "if _cpu_only and use_fit:" in src
    fn = ast.parse(src).body[0]
    # There is an `if _cpu_only and use_fit:` whose body sets use_fit = False.
    found = False
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.BoolOp)
            and any(isinstance(v, ast.Name) and v.id == "_cpu_only" for v in ast.walk(node.test))
        ):
            for n in node.body:
                if (
                    isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "use_fit" for t in n.targets)
                    and isinstance(n.value, ast.Constant)
                    and n.value.value is False
                ):
                    found = True
    assert found, "expected `if _cpu_only and use_fit: ... use_fit = False`"


def test_gpu_path_still_emits_fit_on():
    """Backwards compat: the GPU branch still emits --fit on (only CPU changes)."""
    src = _load_model_src()
    assert 'cmd.extend(["--fit", "on"])' in src


def test_cpu_only_emits_explicit_fit_off():
    """--fit defaults to on in llama.cpp, so CPU-only must pass --fit off explicitly,
    not just skip --fit on (PR review fix)."""
    src = _load_model_src()
    assert "elif _cpu_only:" in src
    assert 'cmd.extend(["--fit", "off"])' in src


# ---- Phase 3: CPU context cap + RAM preflight ------------------------------


def test_cpu_context_ceiling_constant_is_sane():
    from core.inference import llama_cpp as m

    assert hasattr(m, "_CPU_CTX_AUTO_CEILING")
    # A safe chat ceiling, far below a model's million-token native context.
    assert 4096 <= m._CPU_CTX_AUTO_CEILING <= 131072
    assert 0.5 < m._CPU_RAM_BUDGET_FRAC <= 1.0


def test_cpu_only_caps_auto_context_but_honors_explicit():
    src = _load_model_src()
    # The fit runs for any auto context (requested_ctx <= 0), against a ceiling of
    # min(native, 32k); an explicit -c (requested_ctx > 0) is honored untouched.
    assert "requested_ctx <= 0 and effective_ctx > 0" in src
    assert "_ctx_ceiling = min(effective_ctx, _CPU_CTX_AUTO_CEILING)" in src
    assert "effective_ctx = _cpu_cap" in src


def test_cpu_context_fit_runs_below_ceiling_too():
    """A large GGUF with a native context already <= 32k must still be RAM-fit, not
    skipped, so it can be reduced toward 4096 instead of OS-killed (PR review fix)."""
    src = _load_model_src()
    # The gate is `> 0`, not `> _CPU_CTX_AUTO_CEILING`, and the fit ceiling is clamped.
    assert "effective_ctx > _CPU_CTX_AUTO_CEILING" not in src
    assert "requested_ctx = _ctx_ceiling" in src


def _nows(s: str) -> str:
    """Whitespace-stripped source, so assertions survive the formatter wrapping a line."""
    return "".join(s.split())


def test_cpu_context_fit_accounts_for_mtp():
    """mtp_engaged alone is a no-op once budget_frac is set, so the CPU fit must pass the
    byte-accurate MTP overhead fn to actually reserve MTP KV (PR review fix)."""
    src = _nows(_load_model_src())
    assert _nows("mtp_overhead_fn = (_mtp_bytes if _mtp_will_engage_cpu else None)") in src


def test_cpu_context_cap_reuses_fit_helper_against_ram():
    """The cap reuses _fit_context_to_vram with the system-RAM budget, not VRAM."""
    src = _load_model_src()
    assert "_available_system_memory_mib()" in src
    assert "self._fit_context_to_vram(" in src
    assert "_cpu_budget = _CPU_RAM_BUDGET_FRAC" in src
    assert "budget_frac = _cpu_budget" in src


def test_cpu_ram_preflight_warns_when_weights_exceed_ram():
    src = _load_model_src()
    assert "CPU-only memory preflight" in src


def test_cpu_context_floors_to_min_when_weights_exceed_budget():
    """When the fixed footprint exceeds the RAM budget, _fit_context_to_vram returns
    the ceiling unchanged; the cap must floor to the minimum instead (PR review fix)."""
    src = _load_model_src()
    # The check uses the fitted footprint (weights + compute buffer), not raw weights.
    assert "_fixed = model_size_fit or model_size" in src
    assert "_fixed >= _budget_b" in src
    assert "_cpu_cap = 4096" in src


def test_cpu_context_fit_uses_fitted_footprint():
    """The CPU RAM fit must pass the fitted footprint (weights + compute buffer), so a
    context that only fits when the buffer is ignored can't slip through (PR review fix)."""
    src = _load_model_src()
    assert "model_size_bytes = _fixed" in src


def test_numa_decision_uses_footprint_not_just_weights():
    """The NUMA interleave decision must use the full resident footprint (weights +
    compute buffer + KV at the launched parallel slots + MTP reserve), so a model whose
    weights fit one node but whose footprint does not still interleaves (PR review fixes)."""
    src = _load_model_src()
    assert "_numa_footprint" in src
    assert "decide_interleave(_numa_footprint" in src
    # Footprint = fitted weights (incl. compute buffer) + KV + MTP reserve.
    assert "_resident = model_size_fit or model_size" in src
    # MTP is recomputed at the post-cap context, not the stale pre-cap reserve.
    assert _nows("_numa_mtp = _mtp_bytes(effective_ctx) if _mtp_will_engage_cpu else 0") in _nows(
        src
    )
    # KV must be sized for the launched --parallel slots, not the n_parallel=1 default.
    assert "effective_ctx, cache_type_kv, n_parallel = n_parallel" in src


def test_numa_surfaces_total_ram_failure():
    """When the footprint exceeds total RAM across all nodes, decide_interleave returns
    an actionable 'interleave cannot help' reason; the caller must surface it, not only
    the missing-numactl case (PR review fix)."""
    src = _load_model_src()
    assert '"interleave cannot help" in _numa.reason' in src


def test_explicit_user_numa_skips_auto_interleave_prefix():
    """An explicit user --numa must skip the numactl argv prefix (which user extra args
    can't override), not just the --numa distribute flag (PR review fix)."""
    src = _load_model_src()
    assert 'if _numa.interleave and _extra_args_set_any_flag(extra_args, {"--numa"}):' in src
    assert "leaving auto-interleave off" in src


def test_extra_args_forces_cpu_offload_helper():
    """The CPU-force detector: zero GPU layers or --device none, via CLI or the inherited
    LLAMA_ARG_* env (PR review fixes)."""
    from core.inference.llama_cpp import _extra_args_forces_cpu_offload as f

    E: dict = {}  # explicit empty env so cases ignore the ambient environment
    assert f(["-ngl", "0"], env = E)
    assert f(["--n-gpu-layers", "0"], env = E)
    assert f(["--gpu-layers", "0"], env = E)
    assert f(["-ngl=0"], env = E)
    assert not f(["-ngl", "99"], env = E)
    assert not f([], env = E)
    assert not f(None, env = E)
    assert not f(["--flash-attn", "on"], env = E)
    # Each flag's last occurrence wins, matching llama-server's own parsing.
    assert f(["-ngl", "99", "-ngl", "0"], env = E)
    assert not f(["-ngl", "0", "-ngl", "99"], env = E)
    # --device/-dev none also forces CPU, independently of -ngl.
    assert f(["--device", "none"], env = E)
    assert f(["-dev", "none"], env = E)
    assert f(["--device=none"], env = E)
    assert not f(["--device", "CUDA0"], env = E)
    # The two controls are independent: -ngl 0 stays CPU even with a device named.
    assert f(["-ngl", "0", "--device", "CUDA0"], env = E)
    assert f(["--device", "none", "-ngl", "99"], env = E)
    # Inherited env forces CPU when the CLI does not set the control.
    assert f([], env = {"LLAMA_ARG_N_GPU_LAYERS": "0"})
    assert f([], env = {"LLAMA_ARG_DEVICE": "none"})
    assert not f([], env = {"LLAMA_ARG_N_GPU_LAYERS": "99"})
    assert not f([], env = {"LLAMA_ARG_DEVICE": "CUDA0"})
    # CLI wins over env.
    assert not f(["-ngl", "99"], env = {"LLAMA_ARG_N_GPU_LAYERS": "0"})
    assert f(["-ngl", "0"], env = {"LLAMA_ARG_N_GPU_LAYERS": "99"})


def test_cpu_cap_lowers_advertised_ceiling():
    """When the CPU cap reduces the launched context, max_available_ctx must drop too,
    so /status and the UI safe-zone reflect the real window, not native (PR review fix)."""
    src = _load_model_src()
    assert "max_available_ctx = min(max_available_ctx, _cpu_cap)" in src


def test_cpu_fit_skips_mtp_reserve_when_mla_auto_drops():
    """Auto drops embedded MTP for MLA models, so the CPU cap / NUMA footprint must not
    reserve a target-KV copy for a drafter that won't launch (PR review fix)."""
    src = _nows(_load_model_src())
    assert _nows("_mtp_will_engage_cpu = _mtp_will_engage and not (") in src
    assert _nows("not _mla_mtp_auto_enabled()") in src
    # The CPU cap and NUMA recompute use the gated flag, not the raw _mtp_will_engage.
    assert _nows("mtp_overhead_fn = (_mtp_bytes if _mtp_will_engage_cpu else None)") in src
    assert _nows("_numa_mtp = _mtp_bytes(effective_ctx) if _mtp_will_engage_cpu else 0") in src


def test_cpu_fit_reserves_flat_mtp_when_draft_unsized():
    """When MTP engages but the draft KV can't be byte-sized (mtp_overhead_fn is None),
    budget_frac skips the flat reserve, so the CPU budget is trimmed to still hold MTP
    RAM back instead of fitting a context that OOMs once the draft allocates (PR review)."""
    src = _load_model_src()
    assert "if _mtp_will_engage_cpu and mtp_overhead_fn is None:" in src
    assert "_cpu_budget -= _MTP_VRAM_RESERVE_FRAC" in src
    assert "budget_frac = _cpu_budget" in src


def test_zero_offload_folds_into_cpu_only():
    """A visible GPU plus a user -ngl 0 must be treated as CPU-only: the GPU list is
    dropped before _cpu_only is computed so the CPU safe defaults apply (PR review fix)."""
    src = _load_model_src()
    assert "_extra_args_forces_cpu_offload(extra_args)" in src
    assert "gpus, total_by_idx = [], {}" in src
