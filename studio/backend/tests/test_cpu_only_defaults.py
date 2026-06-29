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
    # Cap only fires for an auto context (requested_ctx <= 0) over the ceiling;
    # an explicit -c (requested_ctx > 0) is honored.
    assert "requested_ctx <= 0 and effective_ctx > _CPU_CTX_AUTO_CEILING" in src
    assert "effective_ctx = _cpu_cap" in src


def test_cpu_context_cap_reuses_fit_helper_against_ram():
    """The cap reuses _fit_context_to_vram with the system-RAM budget, not VRAM."""
    src = _load_model_src()
    assert "_available_system_memory_mib()" in src
    assert "self._fit_context_to_vram(" in src
    assert "budget_frac = _CPU_RAM_BUDGET_FRAC" in src


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
    assert "_mtp_reserve_bytes" in src
    # KV must be sized for the launched --parallel slots, not the n_parallel=1 default.
    assert "effective_ctx, cache_type_kv, n_parallel = n_parallel" in src


def test_extra_args_forces_cpu_offload_helper():
    """The zero-offload detector: -ngl 0 / --n-gpu-layers 0 / --gpu-layers 0 (last wins)."""
    from core.inference.llama_cpp import _extra_args_forces_cpu_offload as f

    assert f(["-ngl", "0"])
    assert f(["--n-gpu-layers", "0"])
    assert f(["--gpu-layers", "0"])
    assert f(["-ngl=0"])
    assert not f(["-ngl", "99"])
    assert not f([])
    assert not f(None)
    assert not f(["--flash-attn", "on"])
    # Last occurrence wins, matching llama-server's own parsing.
    assert f(["-ngl", "99", "-ngl", "0"])
    assert not f(["-ngl", "0", "-ngl", "99"])


def test_zero_offload_folds_into_cpu_only():
    """A visible GPU plus a user -ngl 0 must be treated as CPU-only: the GPU list is
    dropped before _cpu_only is computed so the CPU safe defaults apply (PR review fix)."""
    src = _load_model_src()
    assert "_extra_args_forces_cpu_offload(extra_args)" in src
    assert "gpus, total_by_idx = [], {}" in src
