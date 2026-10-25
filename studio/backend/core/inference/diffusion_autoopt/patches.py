# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One registry and one lifecycle for every diffusion source / forward patch.

Each per-architecture module (eager fusions, arch ``_SPECS``, Qwen RoPE, Q2.1 fast step / RoPE / fused, FLUX.2 RoPE,
ROCm fused, Z-Image fused, Wan fused) used to be installed from its own call site with its own teardown list. They now
register a :class:`PatchSpec` here and are installed by stage, uninstalled by group, and report one outcome each into
the load's audit record. The modules keep their internals (fingerprints, eligibility, stashes); the registry owns
WHEN they run, the kill switches, and the record.

Fingerprints share one helper, :func:`source_digest` (the two identical ``_digest`` copies it replaces stay importable
as aliases), plus :func:`body_has` / :func:`body_has_once` for needle guards. :func:`splice_source` is the safe in-memory
source rewrite of unsloth-zoo's ``fused_losses/forward_install.py``: exactly-once needles, an unresolved-globals check,
exec in the original module globals, a digest gate, and the original kept for an exact uninstall.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import inspect
import io
import os
import textwrap
import threading
import tokenize
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

# Stages, in load order. ``load_eager``: shared compile-safe fusions (speed tiers only). ``load``: per-pipe installs
# that gate themselves (dtype / device / class). ``pre_int8`` / ``pre_compile``: per-denoiser installs inside the
# regional compile, before and after the int8 GEMM swap (the order the compiled graph depends on).
STAGE_LOAD_EAGER = "load_eager"
STAGE_LOAD = "load"
STAGE_PRE_INT8 = "pre_int8"
STAGE_PRE_COMPILE = "pre_compile"
STAGES = (STAGE_LOAD_EAGER, STAGE_LOAD, STAGE_PRE_INT8, STAGE_PRE_COMPILE)

# Teardown groups.
GROUP_EAGER = "eager"
GROUP_FUSED_DIT = "fused_dit"
GROUP_VIDEO_FUSED = "video_fused"

_FALSE_TOKENS = ("0", "off", "false", "no")


# ------------------------------------------------------------------------------------------------------ fingerprints
def source_digest(fn: Any) -> Optional[str]:
    """16-hex digest of ``fn``'s source minus comments and docstrings. Source text, not ``ast.dump`` (which changes
    between Python versions). None when the source cannot be read."""
    try:
        src = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
        tree = ast.parse(src)
        comments = [
            tok.start
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type == tokenize.COMMENT
        ]
    except (OSError, TypeError, SyntaxError, ValueError, tokenize.TokenError):
        return None
    lines = src.splitlines()
    for row, col in comments:
        lines[row - 1] = lines[row - 1][:col]
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            for row in range(body[0].lineno, body[0].end_lineno + 1):
                lines[row - 1] = ""
    text = "\n".join(line.rstrip() for line in lines if line.strip())
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _source(fn: Any) -> Optional[str]:
    try:
        return inspect.getsource(fn)
    except (OSError, TypeError):
        return None


def body_has(fn: Callable, *needles: str) -> bool:
    """True iff every ``needle`` is in ``fn``'s source: a body-drift guard that self-disables a patch when diffusers
    changed the rewritten lines."""
    src = _source(fn)
    return src is not None and all(n in src for n in needles)


def body_has_once(fn: Callable, *needles: str) -> bool:
    """Like :func:`body_has`, but each needle must occur exactly once (a rewrite target must be unambiguous)."""
    src = _source(fn)
    return src is not None and all(src.count(n) == 1 for n in needles)


# ------------------------------------------------------------------------------------------------- source splicing
class SpliceError(ValueError):
    """Why a source splice declined; the original stays in place."""


def _unresolved_names(tree: ast.AST, fn_globals: dict) -> list[str]:
    """Names the rewritten function loads that neither its arguments, its own assignments, its globals nor builtins
    provide (a NameError waiting for the first call)."""
    bound: set[str] = set()
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (loaded if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return sorted(n for n in loaded - bound if n not in fn_globals and not hasattr(builtins, n))


def splice_source(
    fn: Callable,
    edits: Iterable[tuple[str, str]],
    *,
    fingerprints: Optional[Iterable[str]] = None,
    extra_globals: Optional[dict] = None,
) -> Callable:
    """Return ``fn`` rebuilt from its source with each ``(old, new)`` edit applied. Every ``old`` must occur exactly
    once; ``fingerprints`` (when given) must contain the stock :func:`source_digest`; the result must resolve every
    global it loads; closures are refused (their cells cannot be rebuilt). Executed in ``fn.__globals__`` (plus
    ``extra_globals`` merged into a copy) so module-level helpers resolve exactly as in the original. Raises
    :class:`SpliceError` with the reason; never mutates anything."""
    raw = inspect.unwrap(fn)
    code = getattr(raw, "__code__", None)
    if code is None:
        raise SpliceError("not a Python function")
    if code.co_freevars:
        raise SpliceError(f"closure over {', '.join(code.co_freevars)}")
    if fingerprints is not None:
        digest = source_digest(raw)
        if digest not in set(fingerprints):
            raise SpliceError(
                f"source differs from the version this was written against ({digest})"
            )
    src = _source(raw)
    if src is None:
        raise SpliceError("source unavailable")
    src = textwrap.dedent(src)
    for old, new in edits:
        count = src.count(old)
        if count != 1:
            raise SpliceError(f"needle found {count} times, expected exactly once: {old[:60]!r}")
        src = src.replace(old, new, 1)
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        raise SpliceError(f"rewritten source does not parse: {exc}") from exc
    funcs = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(funcs) != 1:
        raise SpliceError("rewritten source is not a single function")
    funcs[0].decorator_list = []  # decorators already ran on the original; never re-apply them
    namespace = dict(raw.__globals__)
    if extra_globals:
        namespace.update(extra_globals)
    missing = _unresolved_names(tree, namespace)
    if missing:
        raise SpliceError(f"unresolved globals: {', '.join(missing)}")
    try:
        compiled = compile(tree, filename = f"<unsloth splice {raw.__qualname__}>", mode = "exec")
        exec(compiled, namespace)  # noqa: S102 - the source is the library's own, checked above
    except Exception as exc:  # noqa: BLE001
        raise SpliceError(f"rewritten source failed to build: {exc}") from exc
    new_fn = namespace[funcs[0].name]
    new_fn.__defaults__ = raw.__defaults__
    new_fn.__kwdefaults__ = raw.__kwdefaults__
    new_fn.__qualname__ = raw.__qualname__
    new_fn.__module__ = raw.__module__
    new_fn.__doc__ = raw.__doc__
    new_fn.__unsloth_spliced__ = True
    new_fn.__wrapped_original__ = raw
    return new_fn


# ------------------------------------------------------------------------------------------------------- registry
@dataclass
class PatchContext:
    """What a spec may read when it installs. ``transformer`` is set for per-denoiser stages."""

    pipe: Any = None
    transformer: Any = None
    target: Any = None
    family: Any = None
    dtype: Any = None
    device: Any = None
    speed_mode: Optional[str] = None
    offload_active: bool = False
    denoiser_offloaded: Optional[bool] = None
    backend: str = "image"
    logger: Any = None


@dataclass
class PatchSpec:
    """One patch. ``install(ctx)`` returns truthy when engaged (it owns its own fingerprint / eligibility checks and
    must never raise for a mere mismatch); ``uninstall()`` restores it exactly and is idempotent.

    ``classes``: denoiser class names the spec applies to (empty = any). ``speed_only``: install only on a non-``off``
    tier, and uninstall on ``off`` (the eager fusions). ``env``: the module's existing kill switch, recorded as the
    decline reason (the module also honours it itself). Every spec also answers ``UNSLOTH_DIFFUSION_PATCH_<NAME>=0``.
    ``warn``: the label of the warning a failure logs (kept identical to the pre-registry call sites)."""

    name: str
    stage: str
    install: Callable[[PatchContext], Any]
    uninstall: Optional[Callable[[], None]] = None
    group: Optional[str] = None
    classes: tuple[str, ...] = ()
    backends: tuple[str, ...] = ("image", "video")
    speed_only: bool = False
    env: Optional[str] = None
    warn: Optional[str] = None
    fingerprints: tuple[str, ...] = field(default_factory = tuple)

    @property
    def kill_env(self) -> str:
        return (
            "UNSLOTH_DIFFUSION_PATCH_"
            + "".join(c if c.isalnum() else "_" for c in self.name).upper()
        )


@dataclass
class PatchOutcome:
    name: str
    status: str  # engaged | declined | failed
    reason: Optional[str] = None


_LOCK = threading.RLock()
_REGISTRY: list[PatchSpec] = []


def register(spec: PatchSpec) -> PatchSpec:
    """Add (or replace by name) a spec; registration order is install order within a stage."""
    if spec.stage not in STAGES:
        raise ValueError(f"unknown patch stage {spec.stage!r}")
    with _LOCK:
        for i, existing in enumerate(_REGISTRY):
            if existing.name == spec.name:
                _REGISTRY[i] = spec
                return spec
        _REGISTRY.append(spec)
    return spec


def specs(stage: Optional[str] = None, *, group: Optional[str] = None) -> list[PatchSpec]:
    _ensure_builtin()
    with _LOCK:
        return [
            s
            for s in _REGISTRY
            if (stage is None or s.stage == stage) and (group is None or s.group == group)
        ]


def _env_off(name: Optional[str]) -> bool:
    return bool(name) and (os.environ.get(name) or "").strip().lower() in _FALSE_TOKENS


def _class_name(module: Any) -> str:
    return type(module).__name__


def install_stage(
    stage: str,
    ctx: PatchContext,
    *,
    audit: Any = None,
) -> dict[str, Any]:
    """Run every spec of ``stage`` against ``ctx`` in registration order. Returns ``{name: install result}`` for the
    specs that ran (a declined-by-gate spec is absent). Never raises: a failing spec logs its warning, records
    ``failed`` and the next spec runs."""
    results: dict[str, Any] = {}
    off = str(ctx.speed_mode or "").strip().lower() == "off"
    for spec in specs(stage):
        if ctx.backend not in spec.backends:
            continue
        if spec.classes:
            subject = (
                ctx.transformer
                if ctx.transformer is not None
                else getattr(ctx.pipe, "transformer", None)
            )
            if _class_name(subject) not in spec.classes:
                continue
        if _env_off(spec.kill_env):
            _record(audit, spec.name, "declined", f"{spec.kill_env}=0")
            if spec.uninstall is not None:
                _safe_uninstall(spec)
            continue
        if spec.speed_only and off:
            if spec.uninstall is not None:
                _safe_uninstall(spec)
            _record(audit, spec.name, "declined", "speed off")
            continue
        try:
            result = spec.install(ctx)
        except Exception as exc:  # noqa: BLE001 - optimisation only; the stock code keeps running
            _warn(ctx.logger, spec.warn or spec.name, exc)
            _record(audit, spec.name, "failed", f"{type(exc).__name__}: {exc}")
            continue
        results[spec.name] = result
        if _engaged(result):
            _record(audit, spec.name, "engaged", None)
        elif _env_off(spec.env):
            _record(audit, spec.name, "declined", f"{spec.env}=0")
        else:
            _record(audit, spec.name, "declined", "not applicable or fingerprint mismatch")
    return results


def _engaged(result: Any) -> bool:
    if isinstance(result, dict):
        return any(bool(v) for v in result.values())
    return bool(result)


def _safe_uninstall(spec: PatchSpec) -> None:
    try:
        spec.uninstall()  # type: ignore[misc]
    except Exception:  # noqa: BLE001 - teardown is best effort
        pass


def uninstall_group(group: str) -> None:
    """Restore every spec in ``group`` (idempotent, best effort, reverse registration order)."""
    for spec in reversed(specs(group = group)):
        if spec.uninstall is not None:
            _safe_uninstall(spec)


def uninstall_stage(stage: str) -> None:
    for spec in reversed(specs(stage)):
        if spec.uninstall is not None:
            _safe_uninstall(spec)


def _record(audit: Any, name: str, status: str, reason: Optional[str]) -> None:
    if audit is None:
        return
    try:
        audit.record(f"patch:{name}", status, reason)
    except Exception:  # noqa: BLE001 - the audit never fails a load
        pass


def _warn(logger: Any, what: str, exc: Exception) -> None:
    if logger is not None:
        logger.warning("diffusion.speed: %s failed: %s", what, exc)


# ------------------------------------------------------------------------------------------------ built-in specs
_BUILTIN_DONE = False


def _ensure_builtin() -> None:
    global _BUILTIN_DONE
    if _BUILTIN_DONE:
        return
    with _LOCK:
        if _BUILTIN_DONE:
            return
        _BUILTIN_DONE = True
        for spec in _builtin_specs():
            register(spec)


def _builtin_specs() -> list[PatchSpec]:
    """The pre-registry install sites, in their original order. Imports are lazy: these modules import torch."""

    def eager_install(ctx: PatchContext) -> int:
        from ..diffusion_eager_patches import install_compile_safe_patches
        return install_compile_safe_patches()

    def eager_uninstall() -> None:
        from ..diffusion_eager_patches import uninstall_patches
        uninstall_patches()

    def arch_install(ctx: PatchContext) -> int:
        from ..diffusion_arch_patches import install_arch_patches
        return install_arch_patches()

    def arch_uninstall() -> None:
        from ..diffusion_arch_patches import uninstall_arch_patches
        uninstall_arch_patches()

    def q21_fast_step(ctx: PatchContext) -> bool:
        from ..diffusion_qwenimage21 import install_for_pipe
        return install_for_pipe(ctx.pipe, ctx.logger)

    def flux2_rope(ctx: PatchContext) -> bool:
        from ..diffusion_flux2_rope import install_for_pipe
        return install_for_pipe(ctx.pipe, ctx.dtype, ctx.device, ctx.logger)

    def flux2_rope_uninstall() -> None:
        from ..diffusion_flux2_rope import uninstall
        uninstall()

    def rocm_fused(ctx: PatchContext) -> dict:
        from ..diffusion_rocm_fused import install_for_pipe
        return install_for_pipe(ctx.pipe, ctx.dtype, ctx.device, ctx.logger)

    def rocm_fused_uninstall() -> None:
        from ..diffusion_rocm_fused import uninstall
        uninstall()

    def wan_fused(ctx: PatchContext) -> bool:
        from ..video_wan_fused import install_for_pipe
        return install_for_pipe(ctx.pipe, ctx.dtype, ctx.device, logger = ctx.logger)

    def wan_fused_uninstall() -> None:
        from ..video_wan_fused import uninstall
        uninstall()

    def q21_real_rope(ctx: PatchContext) -> bool:
        from ..diffusion_qwenimage21_rope import install
        return install(ctx.logger)

    def qwen_real_rope(ctx: PatchContext) -> bool:
        from ..diffusion_qwenimage_rope import install
        return install(ctx.transformer, ctx.logger)

    def qwen_real_rope_uninstall() -> None:
        from ..diffusion_qwenimage_rope import uninstall
        uninstall()

    def zimage_fused(ctx: PatchContext) -> bool:
        from ..diffusion_zimage_fused import install
        return install(ctx.transformer, ctx.logger, offload_active = ctx.offload_active)

    def zimage_fused_uninstall() -> None:
        from ..diffusion_zimage_fused import uninstall
        uninstall()

    def q21_fused(ctx: PatchContext) -> Any:
        from ..diffusion_qwenimage21_fused import install
        return install(
            ctx.transformer,
            ctx.logger,
            offload_active = ctx.offload_active
            if ctx.denoiser_offloaded is None
            else bool(ctx.denoiser_offloaded),
        )

    def q21_fused_uninstall() -> None:
        from ..diffusion_qwenimage21_fused import uninstall
        uninstall()

    def moss_fused(ctx: PatchContext) -> Any:
        from ..diffusion_moss_fused import install
        return install(
            ctx.transformer,
            ctx.logger,
            offload_active = ctx.offload_active
            if ctx.denoiser_offloaded is None
            else bool(ctx.denoiser_offloaded),
        )

    return [
        # Shared compile-safe fusions (RMSNorm / AdaLN) and the per-arch addcmul fusions: speed tiers only.
        PatchSpec(
            "eager_fusions",
            STAGE_LOAD_EAGER,
            eager_install,
            eager_uninstall,
            group = GROUP_EAGER,
            backends = ("image",),
            speed_only = True,
            env = "UNSLOTH_DIFFUSION_EAGER_PATCHES",
        ),
        PatchSpec(
            "arch_fusions",
            STAGE_LOAD_EAGER,
            arch_install,
            arch_uninstall,
            group = GROUP_EAGER,
            backends = ("image",),
            speed_only = True,
            env = "UNSLOTH_DIFFUSION_ARCH_PATCHES",
        ),
        # Per-pipe installs that gate themselves on class / dtype / device, on every tier.
        PatchSpec(
            "qwenimage21_fast_step",
            STAGE_LOAD,
            q21_fast_step,
            backends = ("image",),
            env = "UNSLOTH_DIFFUSION_Q21_FAST_STEP",
        ),
        PatchSpec(
            "flux2_rope",
            STAGE_LOAD,
            flux2_rope,
            flux2_rope_uninstall,
            group = GROUP_FUSED_DIT,
            backends = ("image",),
            env = "UNSLOTH_DIFFUSION_FLUX2_FUSED_ROPE",
        ),
        PatchSpec(
            "rocm_fused",
            STAGE_LOAD,
            rocm_fused,
            rocm_fused_uninstall,
            group = GROUP_FUSED_DIT,
            backends = ("image",),
            env = "UNSLOTH_DIFFUSION_FUSED_ROPE",
        ),
        PatchSpec(
            "wan_fused",
            STAGE_LOAD,
            wan_fused,
            wan_fused_uninstall,
            group = GROUP_VIDEO_FUSED,
            backends = ("video",),
            speed_only = True,
            env = "UNSLOTH_DIFFUSION_WAN_FUSED_ADALN",
        ),
        # Inside the regional compile, before the int8 swaps.
        PatchSpec(
            "qwenimage21_real_rope",
            STAGE_PRE_INT8,
            q21_real_rope,
            classes = ("QwenImage21Transformer2DModel",),
            warn = "qwen-image-2.1 real rope",
            env = "UNSLOTH_DIFFUSION_Q21_REAL_ROPE",
        ),
        # Inside the regional compile, after the int8 GEMM swap (whose leftovers the Q2.1 fused install reads).
        PatchSpec(
            "qwenimage_real_rope",
            STAGE_PRE_COMPILE,
            qwen_real_rope,
            qwen_real_rope_uninstall,
            group = GROUP_FUSED_DIT,
            classes = ("QwenImageTransformer2DModel",),
            warn = "qwen-image real rope",
            env = "UNSLOTH_DIFFUSION_QWEN_REAL_ROPE",
        ),
        PatchSpec(
            "zimage_fused",
            STAGE_PRE_COMPILE,
            zimage_fused,
            zimage_fused_uninstall,
            group = GROUP_FUSED_DIT,
            classes = ("ZImageTransformer2DModel",),
            warn = "z-image fused attention",
            env = "UNSLOTH_DIFFUSION_ZIMAGE_FUSED",
        ),
        PatchSpec(
            "qwenimage21_fused",
            STAGE_PRE_COMPILE,
            q21_fused,
            q21_fused_uninstall,
            group = GROUP_FUSED_DIT,
            classes = ("QwenImage21Transformer2DModel",),
            warn = "qwen-image-2.1 convrot sharing",
            env = "UNSLOTH_DIFFUSION_Q21_CONVROT_FUSED",
        ),
        PatchSpec(
            "moss_fused",
            STAGE_PRE_COMPILE,
            moss_fused,
            None,
            group = GROUP_FUSED_DIT,
            classes = ("MOSSImageTransformer",),
            warn = "moss-image fused projections",
            env = "UNSLOTH_DIFFUSION_MOSS_FUSED",
        ),
    ]
