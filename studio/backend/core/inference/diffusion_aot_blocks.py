# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persist the regional block compile with dynamo included, so a restart's first render skips the dynamo re-trace.

The compile bundle (``diffusion_compile_cache``) restores inductor's FX graphs, AOTAutograd entries, Triton kernels and
autotune picks, but not dynamo: every restart still re-traces each distinct repeated block from Python bytecode and
rebuilds its guards before it can look those caches up (1.0-1.8 s of a B200 restart's first render). torch's
``aot_compile`` (``torch._dynamo.aot_compile``, torch >= 2.10) serialises the traced block WITH its guards and the
compiled inductor graph; ``load_compiled_function`` brings it back without tracing.

Lifecycle, all on the render thread (the one thread every regional compile already runs on, so nothing here compiles
on a second thread):
- ``install`` puts a dispatcher in front of every compiled repeated block, inert until ``bind`` gives it the load's
  compile-bundle dir. It serves a call from a loaded artifact whose guards pass, else from the normal
  ``torch.compile`` path, unchanged. Artifacts load lazily on the first block call, or earlier from ``preload`` (a
  job queued on the render thread while the load is still running).
- The first call of a (block class, input signature) the artifacts do not cover is RECORDED (detached clones).
  ``schedule_save`` queues one render-thread job per recording after the render; each job yields to a queued render,
  so a render never waits behind more than one of them.
- A save ``aot_compile``s the block from the recorded inputs and keeps the artifact ONLY when inductor answered every
  graph it compiled from the FX graph cache with zero misses: the proof that the artifact runs exactly the kernels
  the normal path just ran. Anything else is dropped and that signature keeps the normal path.

Artifacts live under the compile bundle's key dir (its LRU eviction removes them), in a sub-dir keyed on the Studio
diffusion sources, the env knobs and the compile kwargs: ``aot_compile`` drops guards on globals, so a patch that
changed what a block calls must also change the directory. Kill switch: ``UNSLOTH_DIFFUSION_AOT_BLOCKS=0``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

_ENV = "UNSLOTH_DIFFUSION_AOT_BLOCKS"
_FALSE = ("0", "false", "no", "off")
_MANIFEST = "manifest.json"
_FORMAT = 1
# Distinct signatures recorded per transformer; past this a new signature is simply not persisted.
_MAX_RECORDED = 16
_DISPATCH_MARK = "_unsloth_aot_dispatch"

_SOURCE_DIGEST: Optional[str] = None
_SOURCE_LOCK = threading.Lock()


def enabled() -> bool:
    return (os.environ.get(_ENV) or "").strip().lower() not in _FALSE


def supported() -> bool:
    """torch exposes aot_compile + load_compiled_function (2.10+), CUDA, not ROCm."""
    try:
        import torch

        if getattr(torch.version, "hip", None) or not torch.cuda.is_available():
            return False
        from torch._dynamo import aot_compile as _aot

        return (
            hasattr(_aot, "aot_compile_fullgraph")
            and hasattr(_aot, "AOTCompiledFunction")
            and hasattr(torch.compiler, "load_compiled_function")
            and hasattr(torch._dynamo.config, "enable_aot_compile")
        )
    except Exception:  # noqa: BLE001
        return False


def _sources_digest() -> str:
    """Hash of every Studio diffusion source: a patch edit must not reuse an artifact traced before it."""
    global _SOURCE_DIGEST
    with _SOURCE_LOCK:
        if _SOURCE_DIGEST is None:
            h = hashlib.sha256()
            here = Path(__file__).resolve().parent
            for p in sorted(here.glob("diffusion*.py")):
                try:
                    h.update(p.name.encode())
                    h.update(p.read_bytes())
                except OSError:
                    h.update(b"?")
            _SOURCE_DIGEST = h.hexdigest()[:24]
        return _SOURCE_DIGEST


def _versions() -> dict[str, Optional[str]]:
    out: dict[str, Optional[str]] = {}
    try:
        import importlib.metadata as md
    except Exception:  # noqa: BLE001
        return out
    for pkg in ("torch", "triton", "torchao", "diffusers", "peft", "transformers"):
        try:
            out[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = None
    return out


def subkey(compile_kwargs: dict[str, Any]) -> str:
    knobs = {
        k: v
        for k, v in sorted(os.environ.items())
        if k.startswith(("UNSLOTH_", "TORCHINDUCTOR_", "TORCH_"))
    }
    for noisy in ("UNSLOTH_STUDIO_HOME", "TORCHINDUCTOR_CACHE_DIR", "UNSLOTH_DIFFUSION_COMPILE_CACHE_DIR"):
        knobs.pop(noisy, None)
    payload = {
        "format": _FORMAT,
        "sources": _sources_digest(),
        "versions": _versions(),
        "env": knobs,
        "compile_kwargs": {k: str(v) for k, v in sorted((compile_kwargs or {}).items())},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys = True, default = str).encode()).hexdigest()[:20]


def _signature(x: Any) -> Any:
    import torch

    if isinstance(x, torch.Tensor):
        return ("T", tuple(x.shape), tuple(x.stride()), str(x.dtype), str(x.device), bool(x.is_inference()))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__,) + tuple(_signature(y) for y in x)
    if isinstance(x, dict):
        return ("D",) + tuple((str(k), _signature(v)) for k, v in sorted(x.items(), key = lambda kv: str(kv[0])))
    if x is None or isinstance(x, (bool, int, float, str)):
        return ("V", x)
    return ("O", type(x).__name__)


def _clone(x: Any) -> Any:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().clone()
    if isinstance(x, tuple):
        return tuple(_clone(y) for y in x)
    if isinstance(x, list):
        return [_clone(y) for y in x]
    if isinstance(x, dict):
        return {k: _clone(v) for k, v in x.items()}
    return x


def _recordable(x: Any) -> bool:
    import torch

    if isinstance(x, torch.Tensor):
        return not x.requires_grad
    if isinstance(x, (list, tuple)):
        return all(_recordable(y) for y in x)
    if isinstance(x, dict):
        return all(_recordable(v) for v in x.values())
    return x is None or isinstance(x, (bool, int, float, str))


def _global_state() -> list:
    """Global modes a dynamo guard would check but ``aot_compile`` does not serialise (it drops global guards)."""
    import torch

    try:
        autocast = bool(torch.is_autocast_enabled("cuda"))
    except TypeError:  # torch < 2.4 signature
        autocast = bool(torch.is_autocast_enabled())
    return [
        bool(torch.is_grad_enabled()),
        bool(torch.is_inference_mode_enabled()),
        autocast,
        bool(torch.are_deterministic_algorithms_enabled()),
    ]


def _hook_free(module: Any) -> bool:
    """No hook anywhere in the block or registered globally: a serialised artifact would silently skip one."""
    import torch.nn.modules.module as mm

    for name in ("_global_forward_hooks", "_global_forward_pre_hooks", "_global_backward_hooks",
                 "_global_backward_pre_hooks"):
        if getattr(mm, name, None):
            return False
    for m in module.modules():
        for name in ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks"):
            if getattr(m, name, None):
                return False
    return True


def _digest(sig: Any) -> str:
    return hashlib.sha256(repr(sig).encode()).hexdigest()[:32]


def _read_json(path: Path) -> Optional[dict]:
    try:
        out = json.loads(path.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001
        return None
    return out if isinstance(out, dict) else None


def _atomic_write(path: Path, data: bytes) -> None:
    from .diffusion_compile_cache import _atomic_write as write

    write(path, data)


class _Sample:
    __slots__ = ("module", "cls", "args", "kwargs", "inference", "dynamic_sources", "unbacked_sources", "device",
                 "global_state")

    def __init__(self, module: Any, args: tuple, kwargs: dict) -> None:
        import torch

        self.module = module
        self.cls = type(module).__name__
        self.args = _clone(args)
        self.kwargs = _clone(kwargs)
        self.inference = bool(torch.is_inference_mode_enabled())
        cfg = getattr(getattr(torch, "compiler", None), "config", None)
        self.dynamic_sources = getattr(cfg, "dynamic_sources", None)
        self.unbacked_sources = getattr(cfg, "unbacked_sources", None)
        self.device = torch.cuda.current_device()
        self.global_state = _global_state()


class Registry:
    """Per-transformer state: loaded artifacts, recordings waiting for a save, counters for status / tests."""

    def __init__(self, compile_kwargs: dict[str, Any], logger: Any = None) -> None:
        # Set by ``bind`` once the load knows its compile-bundle key; until then every call takes the normal path.
        self.dir: Optional[Path] = None
        self.compile_kwargs = dict(compile_kwargs or {})
        self.logger = logger
        self.lock = threading.Lock()
        self.loaded = False
        self.failed: Optional[str] = None
        self.entries: dict[str, list[Any]] = {}
        self.known: set = set()
        self.recorded: dict[str, _Sample] = {}
        self.queued: set = set()
        self.stats: dict[str, Any] = {"hits": 0, "misses": 0, "loaded": 0, "load_s": 0.0, "saved": 0, "rejected": 0}

    # ---- loading -----------------------------------------------------------------------------------------------
    def load(self) -> None:
        with self.lock:
            if self.loaded or self.dir is None:
                return
            self.loaded = True
        t0 = time.perf_counter()
        man = _read_json(self.dir / _MANIFEST)
        if not man or man.get("format") != _FORMAT:
            return
        import torch

        n = 0
        for ent in man.get("entries", []):
            path = self.dir / Path(str(ent.get("file", ""))).name
            try:
                with open(path, "rb") as fh:
                    fn = torch.compiler.load_compiled_function(fh)
                fn.disable_guard_check()
            except Exception as exc:  # noqa: BLE001 - a stale or foreign artifact is a miss, never a failure
                self._log("warning", "diffusion.aot_blocks: could not load %s (%s: %s)", path.name,
                          type(exc).__name__, str(exc)[:200])
                continue
            self.entries.setdefault(str(ent.get("cls")), []).append((ent.get("global"), fn))
            self.known.add(str(ent.get("sig")))
            n += 1
        self.stats["loaded"] = n
        self.stats["load_s"] = round(time.perf_counter() - t0, 3)
        if n:
            self._log("info", "diffusion.aot_blocks: loaded %d block graph(s) in %.0f ms; dynamo skips them",
                      n, self.stats["load_s"] * 1000)

    # ---- dispatch ----------------------------------------------------------------------------------------------
    def dispatch(self, module: Any, inner: Callable, args: tuple, kwargs: dict) -> Any:
        if self.failed is not None or self.dir is None:
            return inner(*args, **kwargs)
        if not self.loaded:
            self.load()
        fns = self.entries.get(type(module).__name__)
        if fns and _hook_free(module):
            state = _global_state()
            try:
                for want, fn in fns:
                    if want == state and fn.guard_check(module, *args, **kwargs):
                        self.stats["hits"] += 1
                        return fn(module, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                from .diffusion_batched import is_oom_error

                if is_oom_error(exc):
                    raise
                # Never again this load: the normal path serves every block from here on.
                self.failed = f"{type(exc).__name__}: {str(exc)[:200]}"
                self._log("warning", "diffusion.aot_blocks: disabled for this load (%s)", self.failed)
                exc.__traceback__ = None
        self.stats["misses"] += 1
        out = inner(*args, **kwargs)
        self._maybe_record(module, args, kwargs)
        return out

    def _maybe_record(self, module: Any, args: tuple, kwargs: dict) -> None:
        try:
            import torch

            if torch.cuda.is_current_stream_capturing():
                return
            if len(self.recorded) >= _MAX_RECORDED or not _recordable((args, kwargs)):
                return
            if not _hook_free(module):
                return
            key = _digest((type(module).__name__, _signature(args), _signature(kwargs)))
            if key in self.known or key in self.recorded:
                return
            self.recorded[key] = _Sample(module, args, kwargs)
        except Exception:  # noqa: BLE001 - a missed recording only means no artifact for that signature
            pass

    # ---- saving ------------------------------------------------------------------------------------------------
    def pending(self) -> list[str]:
        return [k for k in list(self.recorded) if k not in self.known]

    def save_one(self, key: str) -> Optional[bool]:
        """aot_compile one recording; True saved, False rejected, None nothing to do. Runs on the render thread."""
        sample = self.recorded.get(key)
        if sample is None or key in self.known or self.dir is None or self.failed is not None:
            return None
        self.known.add(key)
        try:
            ok, data, why = _aot_compile_sample(sample, self.compile_kwargs)
        except Exception as exc:  # noqa: BLE001
            ok, data, why = False, None, f"{type(exc).__name__}: {str(exc)[:200]}"
        finally:
            self.recorded.pop(key, None)
        if not ok or data is None:
            self.stats["rejected"] += 1
            self._log("info", "diffusion.aot_blocks: not persisting a %s graph (%s)", sample.cls, why)
            return False
        try:
            self.dir.mkdir(parents = True, exist_ok = True)
            fname = f"b-{key[:16]}.aot"
            _atomic_write(self.dir / fname, data)
            with self.lock:
                man = _read_json(self.dir / _MANIFEST) or {}
                if man.get("format") != _FORMAT:
                    man = {"format": _FORMAT, "entries": []}
                man["entries"] = [e for e in man.get("entries", []) if e.get("sig") != key]
                man["entries"].append(
                    {"cls": sample.cls, "sig": key, "file": fname, "bytes": len(data), "global": sample.global_state}
                )
                _atomic_write(self.dir / _MANIFEST, json.dumps(man, indent = 1).encode())
            self.stats["saved"] += 1
            self._log("info", "diffusion.aot_blocks: persisted a %s graph (%d KB)", sample.cls, len(data) // 1024)
            return True
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            self._log("warning", "diffusion.aot_blocks: save failed (%s: %s)", type(exc).__name__, exc)
            return False

    def describe(self) -> dict:
        return dict(self.stats, failed = self.failed, pending = len(self.pending()))

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.logger is not None:
            try:
                getattr(self.logger, level)(msg, *args)
            except Exception:  # noqa: BLE001
                pass


@contextlib.contextmanager
def _pristine_dynamo_state(sample: _Sample):
    """Trace as a first compile would: automatic-dynamic history isolated and restored, the recorded compiler config."""
    import torch
    from torch._dynamo import pgo

    saved_state = pgo._CODE_STATE
    cfg = torch.compiler.config
    saved_cfg = (getattr(cfg, "dynamic_sources", None), getattr(cfg, "unbacked_sources", None))
    saved_aot = torch._dynamo.config.enable_aot_compile
    try:
        pgo._CODE_STATE = None
        if hasattr(cfg, "dynamic_sources"):
            cfg.dynamic_sources = sample.dynamic_sources
        if hasattr(cfg, "unbacked_sources"):
            cfg.unbacked_sources = sample.unbacked_sources
        torch._dynamo.config.enable_aot_compile = True
        yield
    finally:
        pgo._CODE_STATE = saved_state
        if hasattr(cfg, "dynamic_sources"):
            cfg.dynamic_sources = saved_cfg[0]
        if hasattr(cfg, "unbacked_sources"):
            cfg.unbacked_sources = saved_cfg[1]
        torch._dynamo.config.enable_aot_compile = saved_aot


def _aot_compile_sample(sample: _Sample, compile_kwargs: dict[str, Any]) -> tuple[bool, Optional[bytes], str]:
    import torch
    from torch._dynamo.utils import counters

    kwargs: dict[str, Any] = {"fullgraph": True, "dynamic": compile_kwargs.get("dynamic")}
    if compile_kwargs.get("mode"):
        kwargs["mode"] = compile_kwargs["mode"]
    cf = torch.compile(torch.nn.Module._call_impl, **kwargs)
    before = {k: counters["inductor"].get(k, 0) for k in ("fxgraph_cache_hit", "fxgraph_cache_miss",
                                                           "fxgraph_cache_bypass")}
    mode = torch.inference_mode() if sample.inference else torch.no_grad()
    torch.cuda.set_device(sample.device)
    with _pristine_dynamo_state(sample), mode:
        fn = cf.aot_compile(((sample.module,) + tuple(sample.args), sample.kwargs))
        guard_ok = bool(fn.guard_check(sample.module, *sample.args, **sample.kwargs))
    delta = {k: counters["inductor"].get(k, 0) - v for k, v in before.items()}
    if not guard_ok:
        return False, None, "its guards reject the inputs it was traced from"
    if delta["fxgraph_cache_miss"] or delta["fxgraph_cache_bypass"] or delta["fxgraph_cache_hit"] < 1:
        # A miss means this trace lowered a graph the normal path did not: other kernels, possibly other bits.
        return False, None, "FX cache " + ", ".join(f"{k[13:]}={v}" for k, v in delta.items())
    res = type(fn).serialize(fn)
    data = getattr(res, "serialized_data", res)  # torch 2.10 returns the bytes themselves
    return True, data, "ok"


# ---- wiring --------------------------------------------------------------------------------------------------------
def install(transformer: Any, compile_kwargs: dict[str, Any], logger: Any = None) -> Optional[Registry]:
    """Dispatcher in front of every compiled repeated block of ``transformer`` (inert until ``bind``). Never raises.

    Called right after the regional compile + compile guard, before anything that wraps a block's call from the
    outside (``diffusion_block_restride``), so the dispatcher sees the inputs the compiled graph sees."""
    if not enabled() or not supported():
        return None
    if not (compile_kwargs or {}).get("fullgraph"):
        if logger is not None:
            logger.info("diffusion.aot_blocks: off for this load (the block compile is not fullgraph)")
        return None
    try:
        names = set(getattr(transformer, "_repeated_blocks", None) or ())
        if not names:
            return None
        reg = Registry(compile_kwargs, logger)
        count = 0
        for module in transformer.modules():
            if type(module).__name__ not in names:
                continue
            inner = getattr(module, "_compiled_call_impl", None)
            if inner is None or getattr(inner, _DISPATCH_MARK, None) is not None:
                continue

            def dispatch(*args: Any, _m: Any = module, _inner: Any = inner, **kwargs: Any) -> Any:
                return reg.dispatch(_m, _inner, args, kwargs)

            setattr(dispatch, _DISPATCH_MARK, reg)
            guard = getattr(inner, "_unsloth_compile_guard", None)
            if guard is not None:
                dispatch._unsloth_compile_guard = guard  # type: ignore[attr-defined]
            module._compiled_call_impl = dispatch
            count += 1
        if not count:
            return None
        transformer._unsloth_aot_blocks = reg
        if logger is not None:
            logger.info("diffusion.aot_blocks: armed on %d compiled block(s) of %s", count, type(transformer).__name__)
        return reg
    except Exception as exc:  # noqa: BLE001 - optimisation only
        if logger is not None:
            logger.warning("diffusion.aot_blocks: not installed (%s: %s)", type(exc).__name__, exc)
        return None


def registries(pipe: Any) -> list[Registry]:
    out = []
    for name in ("transformer", "transformer_2"):
        reg = getattr(getattr(pipe, name, None), "_unsloth_aot_blocks", None)
        if isinstance(reg, Registry):
            out.append(reg)
    return out


def bind(pipe: Any, compile_ctx: Any) -> int:
    """Point every installed registry at the load's compile-bundle dir. Without a bundle key they stay inert."""
    count = 0
    cdir = getattr(compile_ctx, "dir", None)
    for reg in registries(pipe):
        if cdir is None:
            reg.failed = "no compile-cache key for this load"
            continue
        reg.dir = Path(cdir) / "aot" / subkey(reg.compile_kwargs)
        count += 1
    return count


def schedule_save(pipe: Any, name: str = "diffusion") -> int:
    """Queue one render-thread job per pending recording. Each yields to a queued render. Returns jobs queued."""
    from . import diffusion_render_thread as render_thread

    queued = 0
    for reg in registries(pipe):
        for key in reg.pending():
            sample = reg.recorded.get(key)
            if sample is None or key in reg.queued:
                continue
            if render_thread.submit_idle(name, lambda r = reg, k = key: r.save_one(k), device = sample.device):
                reg.queued.add(key)
                queued += 1
        if queued:
            reg._log("info", "diffusion.aot_blocks: queued %d block graph(s) to persist after this render", queued)
    return queued


def warm_compiler(pool: bool) -> None:
    """The compiler's once-per-process start-up, minus any compile: inductor imports, the torch-source hash every
    inductor / AOT cache key starts from and, with ``pool``, inductor's compile-worker subprocesses (a load that
    will compile for real; a restart served from artifacts never needs them). Runs on the render thread."""
    try:
        import torch._inductor.codecache as codecache
        import torch._inductor.compile_fx  # noqa: F401
        import torch._inductor.runtime.triton_heuristics  # noqa: F401

        codecache.torch_key()
        if pool:
            from torch._inductor.async_compile import maybe_warm_pool

            maybe_warm_pool()
    except Exception:  # noqa: BLE001 - a warm, never a failure
        pass


def preload(pipe: Any, device: Optional[int], name: str = "diffusion") -> bool:
    """Queue the compiler warm-up and the artifact load on the render thread now, so they overlap the rest of the
    load instead of opening render 1. Never blocks."""
    from . import diffusion_render_thread as render_thread

    regs = [r for r in registries(pipe) if not r.loaded and r.dir is not None]
    if not regs or device is None:
        return False

    def job() -> None:
        warm_compiler(pool = not any((r.dir / _MANIFEST).is_file() for r in regs))
        for r in regs:
            r.load()

    return render_thread.submit_idle(name, job, device = device, yield_to_renders = False)


def describe(pipe: Any) -> Optional[list[dict]]:
    regs = registries(pipe)
    return [r.describe() for r in regs] if regs else None
