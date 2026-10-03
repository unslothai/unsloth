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
  compile-bundle dir. It serves a call from a loaded artifact whose guards pass. Artifacts load lazily on the first
  block call, or earlier from ``preload`` (a job queued on the render thread while the load is still running).
- The FIRST graph of each block class (every graph, for a static compile) is compiled through ``aot_compile`` of the
  class's ``forward`` instead of the module's ``torch.compile`` wrapper: the same code, inputs and settings, so the
  same graph and the same dynamo bookkeeping, and the result is persisted right away (no idle time needed).
- A later new signature under automatic dynamic takes the normal ``torch.compile`` path, so dynamo generalises it
  exactly as before, and a restart re-traces it as before. Persisting it would need a second trace on the render
  thread between renders, which a queued render would then wait for.

Guards: ``aot_compile`` keeps the guards on the block's inputs, its parameters, buffers and attribute values, and
drops the ones it cannot serialise: guards on globals, and identity guards on functions and code objects (a diffusers
attention processor's ``__call__`` is one; with them left in, every FLUX / Qwen-Image block fails to serialise).
What those dropped guards protected is covered instead by
- the sub-dir key: Studio diffusion sources, package versions, env knobs, compile kwargs, under the compile bundle's
  key (GPU, dtype, quant, attention backend);
- a code fingerprint stored with each artifact: the class and the bytecode of every submodule's ``forward`` and every
  attention processor's ``__call__``, compared before a block uses the artifact (a patched class takes the normal
  path).
A class whose graph still cannot be serialised is recorded as refused in the manifest, so neither this process nor a
later start pays a second ``aot_compile`` for it.

Artifacts live under the compile bundle's key dir (its LRU eviction removes them). Kill switch:
``UNSLOTH_DIFFUSION_AOT_BLOCKS=0``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
import weakref
from pathlib import Path
from typing import Any, Callable, Optional

_ENV = "UNSLOTH_DIFFUSION_AOT_BLOCKS"
_FALSE = ("0", "false", "no", "off")
_MANIFEST = "manifest.json"
_FORMAT = 2
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


def _plain_inputs(x: Any) -> bool:
    """Tensors that need no grad, containers of them, plain constants: what a block call can be traced from."""
    import torch

    if isinstance(x, torch.Tensor):
        return not x.requires_grad
    if isinstance(x, (list, tuple)):
        return all(_plain_inputs(y) for y in x)
    if isinstance(x, dict):
        return all(_plain_inputs(v) for v in x.values())
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


def _hook_free_global() -> bool:
    """No module hook registered globally: a serialised artifact would silently skip one (per-block hooks are checked
    by ``Registry._hook_free``)."""
    import torch.nn.modules.module as mm

    return not any(getattr(mm, name, None) for name in ("_global_forward_hooks", "_global_forward_pre_hooks",
                                                         "_global_backward_hooks", "_global_backward_pre_hooks"))


def _capturing() -> bool:
    import torch

    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _code_digest(fn: Any, h: Any, depth: int = 0) -> None:
    fn = getattr(fn, "__func__", fn)
    h.update(f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', type(fn).__name__)}".encode())
    code = getattr(fn, "__code__", None)
    if code is None:
        return
    _code_object_digest(code, h, depth)


def _code_object_digest(code: Any, h: Any, depth: int = 0) -> None:
    h.update(code.co_code)
    h.update(repr(code.co_names).encode())
    for const in code.co_consts:
        if hasattr(const, "co_code") and depth < 8:
            _code_object_digest(const, h, depth + 1)
        else:
            h.update(_stable_repr(const).encode())


def _stable_repr(value: Any) -> str:
    """``repr`` that is the same in every process: a frozenset constant (``x in {"a", "b"}``) iterates in string-hash
    order, which ``PYTHONHASHSEED`` randomises per process."""
    if isinstance(value, (frozenset, set)):
        return "{" + ",".join(sorted(_stable_repr(v) for v in value)) + "}"
    if isinstance(value, tuple):
        return "(" + ",".join(_stable_repr(v) for v in value) + ")"
    return repr(value)


def _tensor_digest(name: str, t: Any, h: Any, depth: int = 0) -> None:
    h.update(f"{name}:{type(t).__qualname__}:{t.dtype}:{tuple(t.shape)}:{tuple(t.stride())}:{t.device.type}".encode())
    flatten = getattr(t, "__tensor_flatten__", None)
    if flatten is None or depth > 8:
        return
    try:
        attrs, _ctx = flatten()
    except Exception:  # noqa: BLE001
        return
    for a in attrs:
        inner = getattr(t, a, None)
        if inner is not None and hasattr(inner, "dtype"):
            _tensor_digest(f"{name}.{a}", inner, h, depth + 1)


def _code_fingerprint(module: Any) -> str:
    """Class and bytecode of every submodule's ``forward`` (class or instance-bound) and every attention processor's
    ``__call__`` (what the artifact inlined but whose identity guards ``aot_compile`` cannot keep), plus the type,
    dtype, shape, stride and device of every weight (guards on torchao weights cannot be serialised either)."""
    h = hashlib.sha256()
    for name, t in list(module.named_parameters()) + list(module.named_buffers()):
        # A guard on a weight the pickler cannot copy is dropped (``_guard_filter``): keep its metadata here.
        _tensor_digest(name, t, h)
    for name, m in module.named_modules():
        cls = type(m)
        h.update(f"{name}:{cls.__module__}.{cls.__qualname__}".encode())
        _code_digest(getattr(cls, "forward", None), h)
        inst = vars(m).get("forward")
        if inst is not None:  # an instance-bound forward (its guards are dropped by ``_guard_filter``)
            h.update(b"instance-forward")
            _code_digest(inst, h)
        proc = vars(m).get("processor")
        if proc is not None:
            pcls = type(proc)
            h.update(f"p:{pcls.__module__}.{pcls.__qualname__}".encode())
            _code_digest(getattr(pcls, "__call__", None), h)
            if "__call__" in vars(proc):
                h.update(b"instance-call")
    return h.hexdigest()[:24]


def _guard_filter(entries: Any) -> list[bool]:
    """Keep every guard ``aot_compile`` can serialise; drop globals and identity guards (also when derived, e.g. a
    constant match on a processor's code object). What they covered is keyed or fingerprinted (module docstring)."""
    try:
        from torch._dynamo.guards import CheckFunctionManager

        bad = set(CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES)
    except Exception:  # noqa: BLE001
        bad = {"DICT_VERSION", "NN_MODULE", "ID_MATCH", "FUNCTION_MATCH", "CLASS_MATCH", "MODULE_MATCH",
               "CLOSURE_MATCH", "WEAKREF_ALIVE"}
    # A torchao weight the guard pickler cannot copy: every guard on it, on its owning module (the pickler copies a
    # guarded module's attributes, and torchao gives a quantised Linear an ``extra_repr`` it cannot copy either) or
    # reached through them is dropped. Their classes, code and weight metadata are fingerprinted instead.
    # Same for a submodule whose ``forward`` is an instance attribute (Studio's int8 GEMM binds one to each int8
    # Linear): the pickler cannot copy a bound method of a module-level function; its code is fingerprinted.
    opaque: set = set()
    for g in entries:
        if not getattr(g, "has_value", False):
            continue
        value = getattr(g, "value", None)
        name = str(g.name)
        if not _meta_picklable(value):
            opaque.add(name)
            for marker in ("._parameters[", "._buffers["):
                if marker in name:
                    opaque.add(name.split(marker, 1)[0])
        elif _has_instance_forward(value):
            opaque.add(name)
    opaque = tuple(sorted(opaque))
    out = []
    for g in entries:
        types = {getattr(g, "guard_type", None), *(getattr(g, "derived_guard_types", None) or ())}
        keep = not (getattr(g, "is_global", False) or bool(types & bad))
        name = str(g.name)
        if keep and opaque and name.startswith(opaque):
            keep = False
        out.append(keep)
    return out


def _has_instance_forward(value: Any) -> bool:
    try:
        import torch

        return isinstance(value, torch.nn.Module) and "forward" in vars(value)
    except Exception:  # noqa: BLE001
        return False


_UNSET = object()


@contextlib.contextmanager
def _picklable_instance_forwards(module: Any):
    """While ``aot_compile`` pickles its guard state, give every method bound on a block submodule's instance the
    ``__name__`` it is bound under.

    A block's parameters are graph inputs, so their modules stay in the pickled guard tree whatever the filter drops,
    and the pickler reduces a bound method by looking its function up on the instance under the function's
    ``__name__``. Studio's int8 GEMM binds ``_linear_forward`` as ``module.forward`` and torchao binds a
    ``functools.partial`` as ``extra_repr``: both lookups fail. Under the name they are bound as, the lookup finds the
    method itself and pickles it by reference. Only ``__name__`` changes (``__qualname__``, which pickling a function
    by reference uses, does not), and it is restored right after."""
    import types

    renamed = []
    try:
        for m in module.modules():
            for attr, value in list(vars(m).items()):
                if not isinstance(value, types.MethodType) or value.__self__ is not m:
                    continue
                fn = value.__func__
                old = getattr(fn, "__name__", _UNSET)
                if old == attr:
                    continue
                try:
                    fn.__name__ = attr
                except (AttributeError, TypeError):
                    continue
                renamed.append((fn, old))
        yield
    finally:
        for fn, old in reversed(renamed):
            try:
                if old is _UNSET:
                    del fn.__name__
                else:
                    fn.__name__ = old
            except (AttributeError, TypeError):
                pass


def _meta_picklable(value: Any, depth: int = 0) -> bool:
    """Can dynamo's guard pickler copy this value (it rebuilds tensor subclasses from ``empty_like(device="meta")``
    of every inner tensor; torchao's int8 ``PlainAQTTensorImpl`` refuses that op)."""
    import torch

    if not isinstance(value, torch.Tensor) or depth > 8:
        return True
    try:
        from torch.utils._python_dispatch import is_traceable_wrapper_subclass

        if not is_traceable_wrapper_subclass(value):
            return True
        torch.empty_like(value, device = "meta")
        attrs, _ctx = value.__tensor_flatten__()
        return all(_meta_picklable(getattr(value, a), depth + 1) for a in attrs)
    except Exception:  # noqa: BLE001
        return False


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


class Registry:
    """Per-transformer state: loaded artifacts, refused classes, counters for logs / tests."""

    def __init__(self, compile_kwargs: dict[str, Any], logger: Any = None) -> None:
        # Set by ``bind`` once the load knows its compile-bundle key; until then every call takes the normal path.
        self.dir: Optional[Path] = None
        self.compile_kwargs = dict(compile_kwargs or {})
        self.logger = logger
        self.lock = threading.Lock()
        self.loaded = False
        self.failed: Optional[str] = None
        self.entries: dict[str, list[Any]] = {}
        # Block classes whose first graph in THIS process went through _compile_now (or was refused by it). A class
        # with loaded artifacts that serve no call still compiles its first graph through aot_compile: that hits the
        # FX / autograd caches its first start filled, where the normal path's own key would miss them.
        self.compiled_classes: set = set()
        # Block classes aot_compile could not serialise: never retried (persisted, so a restart does not either).
        self.refused: set = set()
        self._fingerprints: "weakref.WeakKeyDictionary[Any, str]" = weakref.WeakKeyDictionary()
        self._hook_dicts: "weakref.WeakKeyDictionary[Any, tuple]" = weakref.WeakKeyDictionary()
        self._noted: set = set()
        self.stats: dict[str, Any] = {"hits": 0, "misses": 0, "loaded": 0, "load_s": 0.0, "saved": 0}

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
        for ent in man.get("refused", []):
            self.refused.add(str(ent))
            self.compiled_classes.add(str(ent))
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
            self.entries.setdefault(str(ent.get("cls")), []).append((ent.get("global"), ent.get("code"), fn))
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
        # Artifacts trace the class's forward with no hooks: an instance forward or a hook means the normal path.
        if fns and "forward" not in vars(module) and self._hook_free(module):
            state = _global_state()
            try:
                code = self.fingerprint(module)
                for want, want_code, fn in fns:
                    if want == state and want_code == code and fn.guard_check(module, *args, **kwargs):
                        self.stats["hits"] += 1
                        return fn(module, *args, **kwargs)
                self._note_miss(module, fns, state, code)
            except Exception as exc:  # noqa: BLE001
                from .diffusion_batched import is_oom_error

                if is_oom_error(exc):
                    raise
                # Never again this load: the normal path serves every block from here on.
                self.failed = f"{type(exc).__name__}: {str(exc)[:200]}"
                self._log("warning", "diffusion.aot_blocks: disabled for this load (%s)", self.failed)
                exc.__traceback__ = None
        if self._compile_first_allowed(module):
            done, out = self._compile_now(module, args, kwargs)
            if done:
                return out
        self.stats["misses"] += 1
        return inner(*args, **kwargs)

    def _hook_free(self, module: Any) -> bool:
        """No hook in the block or globally, checked on every call without walking the block: hooks are registered
        into the same per-module dicts, so those are collected once per block instance and only tested for emptiness."""
        dicts = self._hook_dicts.get(module)
        if dicts is None:
            names = ("_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks")
            dicts = self._hook_dicts[module] = tuple(
                d for m in module.modules() for d in (getattr(m, n, None) for n in names) if d is not None
            )
        return not any(dicts) and _hook_free_global()

    def _note_miss(self, module: Any, fns: list, state: list, code: str) -> None:
        """Log once per class why no loaded artifact served it (a new shape is expected; anything else is not)."""
        cls = type(module).__name__
        if cls in self._noted:
            return
        self._noted.add(cls)
        why = ("global state" if all(w != state for w, _c, _f in fns)
               else "code fingerprint" if all(c != code for _w, c, _f in fns)
               else "guards (new input shape or attribute)")
        self.stats.setdefault("miss_reasons", {})[cls] = why
        self._log("info", "diffusion.aot_blocks: no loaded %s graph serves this call (%s)", cls, why)

    def fingerprint(self, module: Any) -> str:
        """``_code_fingerprint`` once per block instance (a class patched after a block's first call is not seen)."""
        fp = self._fingerprints.get(module)
        if fp is None:
            fp = self._fingerprints[module] = _code_fingerprint(module)
        return fp

    def _compile_first_allowed(self, module: Any) -> bool:
        """The first graph of a block class (or any graph of a static compile) is compiled through ``aot_compile``.

        That IS the compile the normal path would run (same forward code, same inputs, so the same automatic-dynamic
        bookkeeping), and it leaves a serialisable result. A later new signature of a class under automatic dynamic
        takes the normal path, so dynamo generalises it exactly as before."""
        if self.failed is not None or self.dir is None or "forward" in vars(module):
            return False
        cls = type(module).__name__
        if cls in self.refused:
            return False
        return self.compile_kwargs.get("dynamic") is False or cls not in self.compiled_classes

    def _compile_now(self, module: Any, args: tuple, kwargs: dict) -> tuple[bool, Any]:
        import torch

        try:
            if _capturing() or not _plain_inputs((args, kwargs)) or not self._hook_free(module):
                return False, None
            key = _digest((type(module).__name__, _signature(args), _signature(kwargs)))
            state = _global_state()
            code = self.fingerprint(module)
            t0 = time.perf_counter()
            saved_aot = torch._dynamo.config.enable_aot_compile
            torch._dynamo.config.enable_aot_compile = True
            try:
                with _picklable_instance_forwards(module):
                    fn = _compiler(type(module).forward, self.compile_kwargs).aot_compile(
                        ((module,) + tuple(args), kwargs)
                    )
            finally:
                torch._dynamo.config.enable_aot_compile = saved_aot
            if not fn.guard_check(module, *args, **kwargs):
                return False, None
            fn.disable_guard_check()
        except Exception as exc:  # noqa: BLE001 - the normal path compiles this one (and its guard falls back to eager)
            from .diffusion_batched import is_oom_error

            if is_oom_error(exc):
                raise
            self._log("info", "diffusion.aot_blocks: %s compiles on the normal path (%s: %s)", type(module).__name__,
                      type(exc).__name__, str(exc).splitlines()[0][:200] if str(exc) else "")
            self.compiled_classes.add(type(module).__name__)
            self._refuse(type(module).__name__)
            exc.__traceback__ = None
            return False, None
        cls = type(module).__name__
        self.compiled_classes.add(cls)
        self.entries.setdefault(cls, []).append((state, code, fn))
        self.stats["compiled"] = self.stats.get("compiled", 0) + 1
        self._log("info", "diffusion.aot_blocks: compiled a %s graph in %.1f s", cls, time.perf_counter() - t0)
        try:
            with _picklable_instance_forwards(module):
                res = type(fn).serialize(fn)
            self._persist(key, cls, getattr(res, "serialized_data", res), state, code)
        except Exception as exc:  # noqa: BLE001 - this process still uses it; only the next start pays the trace
            self._log("warning", "diffusion.aot_blocks: could not serialise a %s graph (%s: %s)", cls,
                      type(exc).__name__, str(exc)[:200])
            self._refuse(cls)
        return True, fn(module, *args, **kwargs)

    def _update_manifest(self, edit: Callable[[dict], None]) -> None:
        self.dir.mkdir(parents = True, exist_ok = True)
        with self.lock:
            man = _read_json(self.dir / _MANIFEST) or {}
            if man.get("format") != _FORMAT:
                man = {"format": _FORMAT, "entries": []}
            edit(man)
            _atomic_write(self.dir / _MANIFEST, json.dumps(man, indent = 1).encode())

    def _refuse(self, cls: str) -> None:
        """Remember (also on disk) that ``cls`` does not serialise, so nothing pays its aot_compile again."""
        self.refused.add(cls)
        self.stats["refused"] = sorted(self.refused)
        if self.dir is None:
            return
        try:
            self._update_manifest(
                lambda man: man.__setitem__("refused", sorted(set(man.get("refused", [])) | {cls}))
            )
        except Exception:  # noqa: BLE001 - best-effort
            pass

    def _persist(self, key: str, cls: str, data: bytes, global_state: list, code: Optional[str]) -> bool:
        try:
            self.dir.mkdir(parents = True, exist_ok = True)
            fname = f"b-{key[:16]}.aot"
            _atomic_write(self.dir / fname, data)

            def edit(man: dict) -> None:
                man["entries"] = [e for e in man.get("entries", []) if e.get("sig") != key]
                man["entries"].append({"cls": cls, "sig": key, "file": fname, "bytes": len(data),
                                       "global": global_state, "code": code})

            self._update_manifest(edit)
            self.stats["saved"] += 1
            self._log("info", "diffusion.aot_blocks: persisted a %s graph (%d KB)", cls, len(data) // 1024)
            return True
        except Exception as exc:  # noqa: BLE001 - persistence is best-effort
            self._log("warning", "diffusion.aot_blocks: save failed (%s: %s)", type(exc).__name__, exc)
            return False

    def describe(self) -> dict:
        return dict(self.stats, failed = self.failed)

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.logger is not None:
            try:
                getattr(self.logger, level)(msg, *args)
            except Exception:  # noqa: BLE001
                pass


def _compiler(fn: Callable, compile_kwargs: dict[str, Any]) -> Any:
    """``torch.compile`` with the regional compile's settings, around the block class's ``forward``: the code object
    the normal path traces, so dynamo's per-code bookkeeping lands where it would have."""
    import torch

    dynamic = compile_kwargs.get("dynamic")
    # torch.compile takes the guard filter only through ``options``, and refuses ``mode`` next to ``options``: a mode
    # is passed as the inductor options it stands for (the same config patch, so the same graphs and cache keys).
    options: dict[str, Any] = {}
    if compile_kwargs.get("mode"):
        from torch._inductor import list_mode_options

        options.update(list_mode_options(compile_kwargs["mode"], dynamic))
    options["guard_filter_fn"] = _guard_filter
    return torch.compile(fn, fullgraph = True, dynamic = dynamic, options = options)


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
