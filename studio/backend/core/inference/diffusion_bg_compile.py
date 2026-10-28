# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dense denoisers (offloaded too) compile in background while renders run eager; a render mid-
compile waits for it; any failure ends the attempt and the next render compiles inline;
UNSLOTH_DIFFUSION_BG_COMPILE=0 disables"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE"
OFFLOADED_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_OFFLOADED"
_MAX_SAMPLES = 4

# ContextVars, not thread-locals: diffusion_render_thread.run copies the caller's context onto the render thread.
_FORCE_EAGER: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_force_eager", default = False
)
_NO_CAPTURE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_no_capture", default = False
)
_NO_RECORD: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_no_record", default = False
)
_AUXILIARY: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "unsloth_diffusion_auxiliary_pass", default = False
)


def enabled() -> bool:
    raw = (os.environ.get(_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def offloaded_enabled() -> bool:
    raw = (os.environ.get(OFFLOADED_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


FLOAT_SEEDS_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_FLOAT_SEEDS"
FLOAT_SEEDS_FILE = "bg_compile_float_specializations.json"
_FLOAT_SEEDS_VERSION = 1
_SYMFLOAT_NAME = re.compile(r"^zf\d+$")


def float_seeds_enabled() -> bool:
    raw = (os.environ.get(FLOAT_SEEDS_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _tensorify_state() -> Any:
    """dynamo's set of SymFloat names to specialise on a (re)trace, or None on a torch without it."""
    try:
        from torch._dynamo.symbolic_convert import TensorifyState
    except Exception:  # noqa: BLE001 - older / newer torch: nothing to seed
        return None
    names = getattr(TensorifyState, "force_specializations", None)
    if not isinstance(names, set) or not callable(getattr(TensorifyState, "specialize", None)):
        return None
    return TensorifyState


def _holds_frozenset(const: Any) -> bool:
    if isinstance(const, frozenset):
        return True
    return isinstance(const, tuple) and any(_holds_frozenset(item) for item in const)


def _const_repr(const: Any) -> str:
    """repr, except a frozenset (`x in {"a", "b"}` compiles to one) lists its items sorted.

    Its repr follows PYTHONHASHSEED, so each process would get its own signature and
    never read the seeds an earlier one saved. Anything without one keeps plain repr,
    so seed files written before this stay valid."""
    if not _holds_frozenset(const):
        return repr(const)
    if isinstance(const, frozenset):
        return "frozenset({" + ", ".join(sorted(_const_repr(item) for item in const)) + "})"
    return "(" + "".join(_const_repr(item) + ", " for item in const) + ")"


def _code_digest(code: Any) -> str:
    """Bytecode plus constants; a nested code object (lambda, genexpr) by its own digest, never its address."""
    h = hashlib.sha256(code.co_code)
    for const in code.co_consts:
        if hasattr(const, "co_code"):
            h.update(_code_digest(const).encode("utf-8"))
        else:
            h.update(_const_repr(const).encode("utf-8"))
    return h.hexdigest()


def _code_signature(module: Any) -> Optional[str]:
    """The traced code a seed list belongs to: torch build plus every repeated block class's forward bytecode."""
    try:
        import torch

        from .diffusion_block_graph import repeated_blocks

        parts = [str(torch.__version__), type(module).__qualname__]
        for cls in sorted({type(b) for b in repeated_blocks(module)}, key = lambda c: c.__qualname__):
            code = getattr(getattr(cls, "forward", None), "__code__", None)
            if code is None:
                return None
            name = getattr(code, "co_qualname", code.co_name)
            parts.append(f"{cls.__module__}.{cls.__qualname__}:{name}:{_code_digest(code)}")
        if len(parts) < 3:
            return None
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    except Exception:  # noqa: BLE001 - no signature, no seeds
        return None


def _modules_signature(modules: Any) -> Optional[str]:
    modules = list(modules) if isinstance(modules, (list, tuple)) else [modules]
    sigs = [_code_signature(m) for m in modules]
    if not sigs or any(s is None for s in sigs):
        return None
    return sigs[0] if len(sigs) == 1 else hashlib.sha256("|".join(sigs).encode("utf-8")).hexdigest()


def _read_float_seeds(path: Path, signature: str) -> set:
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except Exception:  # noqa: BLE001 - missing / torn file: learn again
        return set()
    if not isinstance(data, dict) or data.get("version") != _FLOAT_SEEDS_VERSION:
        return set()
    if data.get("signature") != signature:
        return set()
    names = data.get("names")
    if not isinstance(names, list):
        return set()
    return {n for n in names if isinstance(n, str) and _SYMFLOAT_NAME.match(n)}


def _write_float_seeds(path: Path, signature: str, names: set) -> bool:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        tmp.write_text(
            json.dumps(
                {"version": _FLOAT_SEEDS_VERSION, "signature": signature, "names": sorted(names)},
                indent = 1,
            ),
            encoding = "utf-8",
        )
        os.replace(tmp, path)
        return True
    except Exception:  # noqa: BLE001 - bookkeeping for the next load only
        with contextlib.suppress(Exception):
            tmp.unlink()
        return False


def _seed_floats(modules: Any, seed_dir: Optional[Path]) -> tuple[Optional[tuple], int]:
    """Pre-seed the SymFloat names the last compile of this key learned; returns (keep state, names added)."""
    if seed_dir is None or not float_seeds_enabled():
        return None, 0
    try:
        state = _tensorify_state()
        signature = _modules_signature(modules)
        if state is None or signature is None:
            return None, 0
        path = Path(seed_dir) / FLOAT_SEEDS_FILE
        before = set(state.force_specializations)
        seeds = _read_float_seeds(path, signature)
        for name in sorted(seeds - before):
            state.specialize(name)
        return (state, path, signature, before, seeds), len(seeds - before)
    except Exception:  # noqa: BLE001 - no seeds: the compile restarts as before
        return None, 0


def _keep_floats(seeded: Optional[tuple]) -> int:
    """Save this compile's specialised names if changed; returns how many are new."""
    if seeded is None:
        return 0
    state, path, signature, before, seeds = seeded
    try:
        names = {
            n for n in set(state.force_specializations) - before if _SYMFLOAT_NAME.match(str(n))
        }
        names |= seeds & set(state.force_specializations)
        if names and names != seeds:
            _write_float_seeds(path, signature, names)
        return len(names - seeds)
    except Exception:  # noqa: BLE001 - bookkeeping only
        return 0


class InlineFloatSeeds:
    """Float seeds for a compile inside a render (video's first clip): ``seed`` at denoise entry, ``keep`` at decode
    entry, so only the denoiser's names are saved. Never raises."""

    def __init__(
        self,
        modules: Any,
        seed_dir: Any,
        *,
        logger: Any = None,
    ) -> None:
        self.modules = list(modules) if isinstance(modules, (list, tuple)) else [modules]
        self.seed_dir = Path(seed_dir) if seed_dir else None
        self.logger = logger
        self.float_seeds = 0
        self.float_seeds_learned = 0
        self.seeded = False
        self.kept = False
        self._seeds: Optional[tuple] = None

    def seed(self) -> int:
        if self.seeded:
            return 0
        self.seeded = True
        self._seeds, self.float_seeds = _seed_floats(self.modules, self.seed_dir)
        if self.float_seeds and self.logger is not None:
            with contextlib.suppress(Exception):
                self.logger.info(
                    "diffusion.float_seeds: %d float specialisation(s) pre-seeded for the %s compile",
                    self.float_seeds,
                    "/".join(type(m).__name__ for m in self.modules),
                )
        return self.float_seeds

    def keep(self) -> None:
        if not self.seeded or self.kept:
            return
        self.kept = True
        self.float_seeds_learned = _keep_floats(self._seeds)
        self._seeds = None
        self.modules = []

    def done(self) -> bool:
        return self.kept

    def describe(self) -> dict:
        return {"float_seeds": self.float_seeds, "float_seeds_learned": self.float_seeds_learned}


def load_time_enabled() -> bool:
    """Opt-in (``=1``): render 1 eager and render 2 compiled would break "same seed twice repeats"."""
    raw = (os.environ.get(_ENV) or "").strip().lower()
    return raw in ("1", "true", "yes", "on", "all")


def eager_forced() -> bool:
    """True inside a generation that must not touch the compiled callable (its compile is still in flight)."""
    return _FORCE_EAGER.get()


def capture_suppressed() -> bool:
    """True on the background compile thread: run compiled, never record a CUDA graph."""
    return _NO_CAPTURE.get()


@contextlib.contextmanager
def no_record():
    token = _NO_RECORD.set(True)
    try:
        yield
    finally:
        _NO_RECORD.reset(token)


def auxiliary() -> bool:
    return _AUXILIARY.get()


@contextlib.contextmanager
def auxiliary_pass():
    """Detailer crop inpaints: eager, uncaptured, unrecorded, uncounted. Each box is its own size, so compiling
    them recompiled per detection and used up the per-module graph cap the main render needs."""
    tokens = (_AUXILIARY.set(True), _NO_RECORD.set(True), _FORCE_EAGER.set(True))
    try:
        yield
    finally:
        _FORCE_EAGER.reset(tokens[2])
        _NO_RECORD.reset(tokens[1])
        _AUXILIARY.reset(tokens[0])


def gate_compiled_eager(modules: Any) -> Callable[[], None]:
    """Route ``Module.compile``d modules to eager ``_call_impl`` under ``force_eager``; returns the restore.
    Without it force_eager only reaches regional blocks and graph wrappers, which still call the compiled U-Net."""
    installed: list = []
    for module in modules or ():
        inner = getattr(module, "_compiled_call_impl", None)
        if inner is None:
            continue
        eager = getattr(module, "_call_impl", None)
        if eager is None:
            continue

        def gate(
            *args: Any,
            _inner: Any = inner,
            _eager: Any = eager,
            **kwargs: Any,
        ) -> Any:
            if _FORCE_EAGER.get():
                return _eager(*args, **kwargs)
            return _inner(*args, **kwargs)

        try:
            module._compiled_call_impl = gate
        except Exception:  # noqa: BLE001 - not a settable module
            continue
        installed.append((module, gate, inner))

    def restore() -> None:
        for module, gate, inner in installed:
            try:
                if getattr(module, "_compiled_call_impl", None) is gate:
                    module._compiled_call_impl = inner
            except Exception:  # noqa: BLE001
                pass
        installed.clear()

    return restore


@contextlib.contextmanager
def force_eager(on: bool = True):
    token = _FORCE_EAGER.set(bool(on))
    try:
        yield
    finally:
        _FORCE_EAGER.reset(token)


def recording(module: Any) -> bool:
    """True when a call into ``module`` now records inputs for its pending background compile; callers that reshape
    compiled calls (Z-Image CFG split) must reshape this one too, or the compile builds a graph nothing runs."""
    if not eager_forced() or _NO_RECORD.get():
        return False
    for hook in tuple((getattr(module, "_forward_pre_hooks", None) or {}).values()):
        owner = getattr(hook, "__self__", None)
        if (
            isinstance(owner, BackgroundCompile)
            and owner.module is module
            and owner.wants_samples()
        ):
            return True
    return False


def _cuda_graph_helpers() -> tuple[Callable, Callable, Callable]:
    from .diffusion_cuda_graph import graph_key
    return _flatten_sample, _rebuild_sample, graph_key


# Plain state objects in a call (Qwen-Image-2.1's ``QwenImage21KVCache``) are recorded as fresh instances of cloned tensors.
OBJECTS_ENV = "UNSLOTH_DIFFUSION_BG_COMPILE_OBJECTS"
_OBJECT_DEPTH = 3
# A prefix holding a condition image is ~2.2 GiB: such a call keeps compiling on the render.
_OBJECT_MAX_BYTES = 1 << 30


def _objects_enabled() -> bool:
    raw = (os.environ.get(OBJECTS_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _plain_state(obj: Any) -> Optional[dict]:
    import torch

    if isinstance(obj, (torch.nn.Module, torch.Tensor, type)) or callable(obj):
        return None
    cls = type(obj)
    if str(cls.__module__).split(".")[0] in ("builtins", "torch") or hasattr(cls, "__slots__"):
        return None
    state = getattr(obj, "__dict__", None)
    return state if isinstance(state, dict) else None


def _flatten_sample(
    obj: Any,
    out: list,
    depth: int = 0,
    budget: Optional[list] = None,
) -> tuple:
    import torch

    if budget is None:
        budget = [0]
    if torch.is_tensor(obj):
        out.append(obj)
        return ("t", len(out) - 1)
    if isinstance(obj, (list, tuple)):
        return ("l", isinstance(obj, tuple), [_flatten_sample(o, out, depth, budget) for o in obj])
    if isinstance(obj, dict):
        return ("d", [(k, _flatten_sample(v, out, depth, budget)) for k, v in obj.items()])
    if isinstance(obj, (int, float, bool, str, bytes)) or obj is None:
        return ("v", obj)
    state = _plain_state(obj) if _objects_enabled() and depth < _OBJECT_DEPTH else None
    if state is None:
        raise TypeError(f"cannot make a static buffer for a {type(obj).__name__} in the call tree")
    first = len(out)
    spec = (
        "o",
        type(obj),
        [(k, _flatten_sample(v, out, depth + 1, budget)) for k, v in state.items()],
    )
    held = budget[1] if len(budget) > 1 else None
    if held is not None:
        held.update(range(first, len(out)))
    if depth == 0:
        # The outermost object's slice holds every nested one's: charging each level would count a tensor twice.
        budget[0] += sum(t.numel() * t.element_size() for t in out[first:])
    if budget[0] > _OBJECT_MAX_BYTES:
        raise TypeError(
            f"{type(obj).__name__} holds {budget[0] >> 20} MiB, above the {_OBJECT_MAX_BYTES >> 20} MiB a sample copies"
        )
    return spec


def _rebuild_sample(spec: tuple, tensors: list) -> Any:
    kind = spec[0]
    if kind == "t":
        return tensors[spec[1]]
    if kind == "l":
        values = [_rebuild_sample(s, tensors) for s in spec[2]]
        return tuple(values) if spec[1] else values
    if kind == "d":
        return {k: _rebuild_sample(s, tensors) for k, s in spec[1]}
    if kind == "o":
        obj = object.__new__(spec[1])
        for k, s in spec[2]:
            obj.__dict__[k] = _rebuild_sample(s, tensors)
        return obj
    return spec[1]


def _capture_armed(module: Any) -> bool:
    """Whether a live CUDA-graph wrapper sits on ``module`` (forward slot for a DiT, compiled slot for a U-Net)."""
    try:
        from .diffusion_cuda_graph import GraphedForward, _outer_layer
    except Exception:  # noqa: BLE001
        return False
    candidates = [getattr(module, "_compiled_call_impl", None), module.__dict__.get("forward")]
    gate = candidates[0]
    pair = getattr(gate, "_unsloth_bg_inner", None)
    if pair is not None:
        candidates.append(pair)
    try:
        outer = _outer_layer(module)
        if outer is not None:
            candidates.append(getattr(outer, "inner", None))
    except Exception:  # noqa: BLE001
        pass
    return any(
        isinstance(c, GraphedForward) and c.enabled and not c.poisoned and not c.bypassed
        for c in candidates
    )


class BackgroundCompile:
    """One armed denoiser: records eager inputs, then compiles from them on a daemon thread."""

    def __init__(
        self,
        module: Any,
        *,
        logger: Any = None,
        max_samples: int = _MAX_SAMPLES,
        seed_dir: Any = None,
    ) -> None:
        self.module = module
        self.logger = logger
        self.seed_dir = Path(seed_dir) if seed_dir else None
        self.float_seeds = 0
        self.float_seeds_learned = 0
        self.max_samples = int(max_samples)
        self.samples: list[tuple] = []
        self._keys: set = set()
        self._handle: Any = None
        self._gate: Optional[tuple] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._closed = threading.Event()
        # recording -> compiling -> done | failed
        self.state = "recording"
        self.error: Optional[str] = None
        self.compile_s: Optional[float] = None
        self.eager_generations = 0

    def install(self) -> bool:
        try:
            self._install_gate()
            self._handle = self.module.register_forward_pre_hook(self._record, with_kwargs = True)
            return True
        except Exception as exc:  # noqa: BLE001 - no hook, no background compile
            self._finish("failed", f"pre-hook: {type(exc).__name__}: {exc}")
            self._remove_gate()
            return False

    def _install_gate(self) -> None:
        """``Module.compile`` serves every call from ``_compiled_call_impl``: route it eager under ``force_eager``."""
        inner = getattr(self.module, "_compiled_call_impl", None)
        if inner is None:
            return
        eager = self.module._call_impl

        def gate(*args: Any, **kwargs: Any) -> Any:
            if _FORCE_EAGER.get():
                return eager(*args, **kwargs)
            return inner(*args, **kwargs)

        gate._unsloth_bg_gate = True  # type: ignore[attr-defined]
        gate._unsloth_bg_inner = inner  # type: ignore[attr-defined]
        self.module._compiled_call_impl = gate
        self._gate = (gate, inner)

    def _remove_gate(self) -> None:
        pair, self._gate = self._gate, None
        if pair is None or self.module is None:
            return
        gate, inner = pair
        try:
            if getattr(self.module, "_compiled_call_impl", None) is gate:
                self.module._compiled_call_impl = inner
        except Exception:  # noqa: BLE001
            pass

    def _record(self, module: Any, args: tuple, kwargs: dict) -> None:
        if (
            not eager_forced()
            or _NO_RECORD.get()
            or len(self.samples) >= self.max_samples
            or self.state != "recording"
        ):
            return None
        try:
            import torch

            _flatten, _rebuild, graph_key = _cuda_graph_helpers()
            key = graph_key((args, kwargs))
            if key in self._keys:
                return None
            live: list = []
            held: set = set()
            spec = _flatten((args, kwargs), live, 0, [0, held])
            if _capture_armed(module):
                # Like the capture's static buffers (made outside inference mode): dynamo guards on inference-ness.
                # Tensors inside a state object are not graph inputs: they keep the inference-ness they arrived with.
                clones = []
                for index, t in enumerate(live):
                    with torch.inference_mode(index in held and t.is_inference()):
                        clones.append(torch.empty_like(t))
                for dst, src in zip(clones, live):
                    dst.copy_(src)
            else:
                clones = [t.detach().clone() for t in live]
            self._keys.add(key)
            self.samples.append(
                (
                    spec,
                    clones,
                    bool(torch.is_inference_mode_enabled()),
                    bool(torch.is_grad_enabled()),
                    torch.cuda.current_device() if torch.cuda.is_available() else None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - an uncloneable tree just means no background compile
            self.state = "failed"
            self.error = f"record: {type(exc).__name__}: {exc}"
            # Not _finish: the hook is running, and removing it mid-dispatch would mutate the dict torch iterates.
            self.samples = []
            if self.logger is not None:
                self.logger.info(
                    "diffusion.bg_compile: %s cannot be recorded (%s); the next render compiles inline",
                    type(module).__name__,
                    self.error,
                )
        return None

    def wants_samples(self) -> bool:
        return (
            self.state == "recording"
            and len(self.samples) < self.max_samples
            and self._handle is not None
        )

    def _remove_hook(self) -> None:
        handle, self._handle = self._handle, None
        if handle is not None:
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass

    def pending(self) -> bool:
        return self.state in ("recording", "compiling")

    def note_eager_generation(self) -> None:
        self.eager_generations += 1

    def compiling(self) -> bool:
        thread = self._thread
        return self.state == "compiling" and thread is not None and thread.is_alive()

    def wait(
        self,
        cancel: Any = None,
        poll_s: float = 0.25,
    ) -> float:
        """Block until an in-flight background compile ends; returns the seconds waited. ``cancel`` aborts the wait."""
        t0 = time.perf_counter()
        thread = self._thread
        if thread is None or thread is threading.current_thread():
            return 0.0
        while thread.is_alive():
            if cancel is not None and cancel.is_set():
                break
            thread.join(poll_s)
        return time.perf_counter() - t0

    def kick(self) -> bool:
        """Start the background compile once samples exist. Idempotent; returns whether a thread is running."""
        with self._lock:
            if self.state == "compiling":
                return True
            if self.state != "recording":
                self._remove_hook()
                return False
            if not self.samples:
                # The generation never reached the denoiser (cancelled / failed early): keep recording.
                return False
            # Before the thread starts: the compiled trace must not contain the hook.
            self._remove_hook()
            self.state = "compiling"
            self._thread = threading.Thread(
                target = self._run, name = "unsloth-diffusion-bg-compile", daemon = True
            )
            self._thread.start()
            return True

    def abandon(self, reason: str) -> bool:
        """Give up while still recording, so the next render compiles inline. False once a compile started or ended."""
        with self._lock:
            if self.state != "recording":
                return False
            self._finish("failed", reason)
            return True

    def _finish(
        self,
        state: str,
        error: Optional[str] = None,
    ) -> None:
        self.state = state
        self.error = error
        self.samples = []
        self._remove_hook()

    def _run(self) -> None:
        t0 = time.perf_counter()
        token = _NO_CAPTURE.set(True)
        try:
            import torch

            from . import diffusion_compile_config

            # torch 2.12+ keeps compile config per context: a fresh thread would compile without the recorded knobs.
            diffusion_compile_config.apply()
            seeds = self._seed_floats()
            _flatten, _rebuild, graph_key = _cuda_graph_helpers()
            for spec, clones, inference, grad, device in list(self.samples):
                if self._closed.is_set():
                    self._finish("failed", "closed before the compile finished")
                    return
                if device is not None:
                    torch.cuda.set_device(device)
                args, kwargs = _rebuild(spec, clones)
                mode = torch.inference_mode() if inference else torch.set_grad_enabled(grad)
                with mode:
                    self.module(*args, **kwargs)
                if torch.cuda.is_available():
                    from .diffusion_cuda_graph import synchronize_outside_capture
                    synchronize_outside_capture()
            self.compile_s = time.perf_counter() - t0
            self._keep_floats(seeds)
            self._finish("done")
            if self.logger is not None:
                self.logger.info(
                    "diffusion.bg_compile: %s compiled in the background in %.1f s%s; the next render runs compiled",
                    type(self.module).__name__,
                    self.compile_s,
                    f" ({self.float_seeds} float specialisation(s) pre-seeded)"
                    if self.float_seeds
                    else "",
                )
        except BaseException as exc:  # noqa: BLE001 - never kills the process; the next render compiles inline
            self.compile_s = time.perf_counter() - t0
            self._finish(
                "failed",
                f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}",
            )
            if self.logger is not None:
                self.logger.warning(
                    "diffusion.bg_compile: background compile of %s failed after %.1f s (%s); the next render "
                    "compiles inline",
                    type(self.module).__name__,
                    self.compile_s,
                    self.error,
                )
            exc.__traceback__ = None
        finally:
            _NO_CAPTURE.reset(token)

    def _seed_floats(self) -> Optional[tuple]:
        """Pre-seed the SymFloat specialisations this key's last compile learned; returns what ``_keep_floats`` needs."""
        seeded, self.float_seeds = _seed_floats(self.module, self.seed_dir)
        return seeded

    def _keep_floats(self, seeded: Optional[tuple]) -> None:
        self.float_seeds_learned = _keep_floats(seeded)

    def close(self, timeout: Optional[float] = None) -> None:
        """Stop recording and wait for an in-flight compile (it holds the module); used on unload."""
        self._closed.set()
        self._remove_hook()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            if self.logger is not None:
                self.logger.info(
                    "diffusion.bg_compile: waiting for the background compile before unloading"
                )
            thread.join(timeout)
        if self.pending():
            self._finish("failed", "closed")
        # Unload uninstalls the CUDA-graph layer by identity: the gate in front of it goes first.
        self._remove_gate()
        self.module = None

    def describe(self) -> dict:
        return {
            "state": self.state,
            "samples": len(self.samples),
            "compile_s": None if self.compile_s is None else round(self.compile_s, 2),
            "eager_generations": self.eager_generations,
            "float_seeds": self.float_seeds,
            "error": self.error,
        }


def select_module(
    pipe: Any,
    *,
    speed_optims: Any,
    default_tier: bool,
    quantized: bool,
    gguf: bool,
    step_cache: bool,
    device: Any,
    backend: Any,
    denoiser_hooked: bool,
) -> Any:
    """The one dense default-tier CUDA denoiser whose compile may move off the render, or None.

    Not torchao (~30x slower eager), GGUF (compiles only its dequant), a step cache (toggles graphs per step) or
    max-autotune (an explicit request to pay the compile). An offloaded denoiser only with ``offloaded_enabled``."""
    if not enabled():
        return None
    if "compiled" not in tuple(speed_optims or ()) or not default_tier:
        return None
    if quantized or gguf or step_cache or (denoiser_hooked and not offloaded_enabled()):
        return None
    if device != "cuda" or backend == "rocm":
        return None
    try:
        from .diffusion_cuda_graph import _denoiser_modules
        modules = _denoiser_modules(pipe)
    except Exception:  # noqa: BLE001
        return None
    return modules[0] if len(modules) == 1 else None


def arm(
    module: Any,
    *,
    logger: Any = None,
    seed_dir: Any = None,
) -> Optional[BackgroundCompile]:
    """A recording ``BackgroundCompile`` on ``module``, or None (disabled / no hook)."""
    if module is None or not enabled():
        return None
    bg = BackgroundCompile(module, logger = logger, seed_dir = seed_dir)
    if not bg.install():
        return None
    return bg
