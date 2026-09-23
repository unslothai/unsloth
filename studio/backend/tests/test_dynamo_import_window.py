# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The partially-initialised ``torch._dynamo`` window (#10350, #10963).

``torch/__init__.py`` makes ``_dynamo`` a lazy submodule, so ``torch._dynamo.X`` imports it
on demand and returns whatever ``sys.modules`` holds, a still-initialising module included.
``diffusers.hooks`` opens that window on every diffusion load with an offload policy, because
it evaluates ``@torch.compiler.disable()`` at class-body time and that is ``import
torch._dynamo``. Nothing on the offload path is guarded, so a read from another thread in
that window fails the whole load with a bare one-line error.

These tests pin the three things the fix depends on, none of which need a real torch:

  1. ``ensure_dynamo_imported`` resolves ``.utils`` BY ATTRIBUTE, so the state an ``import``
     cannot see (submodule in sys.modules, never bound on the parent) is still caught.
  2. It is idempotent and single-flight, which is the property that closes the window.
  3. The load path calls it BEFORE ``apply_memory_plan``, and the load failure handler logs
     with ``exc_info``. Both are asserted against the source, since reaching them for real
     needs a GPU and a model.
"""

from __future__ import annotations

import ast
import builtins
import sys
import threading
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent


@pytest.fixture
def warm(monkeypatch):
    """``utils.torch_warmup`` with its dynamo latch reset, so each test starts cold."""
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "_dynamo_done", False, raising=False)
    return torch_warmup


def _fake_torch(monkeypatch, *, bind_utils: bool):
    """A ``torch`` whose ``_dynamo`` may or may not have ``.utils`` bound on it.

    ``bind_utils=False`` is the reported state: ``torch._dynamo`` and ``torch._dynamo.utils``
    are both in ``sys.modules``, so both imports succeed, but the parent never got the
    attribute -- which is how the compile stack reads it.
    """
    torch = types.ModuleType("torch")
    dynamo = types.ModuleType("torch._dynamo")
    utils = types.ModuleType("torch._dynamo.utils")
    if bind_utils:
        dynamo.utils = utils
    torch._dynamo = dynamo
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._dynamo", dynamo)
    monkeypatch.setitem(sys.modules, "torch._dynamo.utils", utils)
    return torch


def test_reports_false_when_utils_is_not_bound_on_the_parent(warm, monkeypatch):
    """The failure an ``import`` cannot observe. Both imports succeed here; only the
    attribute read distinguishes the broken state, so a probe that merely imported would
    report success in exactly the case that matters."""
    _fake_torch(monkeypatch, bind_utils=False)
    assert warm.ensure_dynamo_imported() is False


def test_reports_true_and_latches_once_dynamo_is_whole(warm, monkeypatch):
    _fake_torch(monkeypatch, bind_utils=True)
    assert warm.ensure_dynamo_imported() is True
    assert warm._dynamo_done is True


def test_absent_torch_is_not_fatal(warm, monkeypatch):
    """A --no-torch host reports False rather than raising: callers keep their eager paths."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert warm.ensure_dynamo_imported() is False


def test_concurrent_callers_import_once(warm, monkeypatch):
    """Single-flight is the whole mechanism: the window closes because exactly ONE thread
    performs the first import while the rest wait on our lock rather than racing CPython's.

    Asserted by counting the IMPORTS, not the lock entries. Threads that arrive before the
    first one latches legitimately queue on the lock, so a bound on lock entries is a race
    against thread scheduling; the double-checked flag inside the lock is what guarantees
    the import body runs once, and that is the property worth pinning."""
    _fake_torch(monkeypatch, bind_utils=True)

    real_import = builtins.__import__
    imports = []
    lock = threading.Lock()

    def _counting_import(name, *args, **kwargs):
        if name.startswith("torch._dynamo"):
            with lock:
                imports.append(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _counting_import)

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(warm.ensure_dynamo_imported()))
        for _ in range(12)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == [True] * 12
    # `import torch._dynamo` plus `import torch._dynamo.utils`, from one thread only.
    assert len(imports) == 2, f"the import body ran more than once: {imports}"


def test_the_warm_closes_the_window_before_it_starts_a_thread():
    """The warm is what actually fixes this: it gets the first import done while the process
    is still single-threaded. _prime_nvlink_topology starts the first thread this module
    spawns, so warming after it would forfeit that.

    Asserted on source order rather than by running the stage, which would need real hardware
    detection."""
    import inspect
    from utils import torch_warmup

    body = inspect.getsource(torch_warmup._warm_inference_backend)
    assert "ensure_dynamo_imported()" in body, "the warm no longer closes the dynamo window"
    assert body.index("ensure_dynamo_imported()") < body.index(
        "_prime_nvlink_topology()"
    ), "dynamo must be imported before the warm starts any thread"


def test_the_stage_list_and_its_purge_contract_are_untouched():
    """The window is closed INSIDE an existing stage, not by adding one. _STAGES carries a
    purge-on-failure mapping whose only plausible entry for a dynamo stage would be torch,
    and purging torch is never right. This pins that decision."""
    from utils import torch_warmup

    assert [name for name, _ in torch_warmup._STAGES] == [
        "hardware",
        "inference_backend",
        "transformers",
        "datasets",
    ]


def _load_pipeline_body():
    tree = ast.parse((_BACKEND / "core/inference/diffusion.py").read_text(encoding="utf-8"))
    return next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "load_pipeline"
    )


def test_load_path_closes_the_window_before_every_dynamo_consumer():
    """Order is the point, not presence, and there is more than one consumer.

    ``apply_memory_plan`` imports ``diffusers.hooks``, but the speed path gets there FIRST on the
    default GGUF profile: ``apply_speed_optims`` reads ``torch._dynamo.config`` and
    ``compile_cache.begin`` enters the compile stack. Whichever runs first is the one that can
    lose the race, and the speed path's own best-effort handler would swallow it, quietly
    disabling compile while leaving the module poisoned for the offload below. So the pre-import
    has to precede all three, not just the offload."""
    body = _load_pipeline_body()
    # Keyed on lineno, not on ast.walk order: walk is breadth-first, so wrapping a call in a
    # try/except would silently reorder it and make this assertion meaningless.
    lines = {}
    for node in ast.walk(body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            lines.setdefault(node.func.id, node.lineno)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            lines.setdefault(node.func.attr, node.lineno)
    assert "close_dynamo_import_window" in lines, "the load path never closes the dynamo window"
    for consumer in (
        "hidream_te4_kwargs",  # FP8 text-encoder cast -> diffusion_precision -> diffusers.hooks
        "apply_step_cache",  # diffusion_cache -> diffusers.hooks
        "begin",  # compile_cache.begin -> the compile stack
        "apply_speed_optims",  # reads torch._dynamo.config
        "apply_memory_plan",  # offload -> diffusers.hooks
    ):
        assert consumer in lines, f"{consumer} is no longer on this path; re-check the ordering"
        assert (
            lines["close_dynamo_import_window"] < lines[consumer]
        ), f"the dynamo pre-import runs after {consumer}, which can reach dynamo first"

    # And ahead of the plain `import diffusers` too, which pulls dynamo in by itself: every
    # module in diffusers.hooks evaluates @torch.compiler.disable() at class-body time.
    src = (_BACKEND / "core/inference/diffusion.py").read_text(encoding="utf-8").splitlines()
    first_diffusers = next(
        n for n, line in enumerate(src, 1) if line.strip() == "import diffusers" and n > body.lineno
    )
    assert (
        lines["close_dynamo_import_window"] < first_diffusers
    ), "the pre-import runs after `import diffusers`, which triggers the dynamo import itself"


def test_the_video_path_closes_the_window_before_each_of_its_diffusers_imports():
    """The image path is not the only one that reaches diffusers.

    ``core/inference/video.py`` imports it twice: once during modular validation, which runs on
    the REQUEST thread, and once during pipeline assembly. The server accepts requests as soon as
    the socket binds, while the background warm may still be inside ``import torch._dynamo``, so
    a video load issued right after startup can recreate exactly the race this PR closes for
    images. Every ``import diffusers`` in this file must be preceded by the guard.
    """
    src = (_BACKEND / "core/inference/video.py").read_text(encoding="utf-8").splitlines()
    # A call to assert_pipeline_class_available counts: it closes the window itself, ahead of its
    # own `import diffusers`, so an import below one is already protected.
    guards = [
        n
        for n, line in enumerate(src, 1)
        if "close_dynamo_import_window(" in line
        or ("assert_pipeline_class_available(" in line and "import" not in line)
    ]
    imports = [n for n, line in enumerate(src, 1) if line.strip() == "import diffusers"]
    assert imports, "video.py no longer imports diffusers; re-check this test"
    assert guards, "the video path never closes the dynamo window"
    for imp in imports:
        assert any(
            g < imp for g in guards
        ), f"`import diffusers` at video.py:{imp} has no dynamo guard above it"
        # Above it in the same block, not merely somewhere earlier in a 6000-line file.
        assert (
            imp - max(g for g in guards if g < imp) < 40
        ), f"the guard for video.py:{imp} is too far above it to be the one protecting it"


def test_the_pipeline_class_probe_closes_the_window_itself():
    """The guard belongs where the import is, not at each call site.

    ``assert_pipeline_class_available`` does its own ``import diffusers`` and then a ``hasattr``
    that imports the pipeline's submodule, and all three request-thread entry points reach it:
    image validation (diffusion.validate_load_request), video validation, and the training
    preflight (_assert_family_pipeline_available -> DiffusionLoraConfig.normalized). Guarding it
    once here covers every caller, including any added later, which chasing call sites does not.
    """
    tree = ast.parse(
        (_BACKEND / "core/inference/diffusion_families.py").read_text(encoding="utf-8")
    )
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "assert_pipeline_class_available"
    )
    guard = next(
        (
            n.lineno
            for n in ast.walk(fn)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "close_dynamo_import_window"
        ),
        None,
    )
    assert guard is not None, "the pipeline-class probe never closes the dynamo window"

    imports = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Import) and any(a.name == "diffusers" for a in n.names)
    ]
    assert imports, "assert_pipeline_class_available no longer imports diffusers"
    assert guard < min(imports), "the guard runs after the import it is supposed to precede"


def test_every_request_thread_entry_point_reaches_that_probe():
    """Pins the three callers the guard is there for, so a path that stops routing through
    assert_pipeline_class_available and imports diffusers directly fails here."""
    for rel, caller in (
        ("core/inference/diffusion.py", "image validation"),
        ("core/inference/video.py", "video validation"),
        ("core/training/diffusion_train_common.py", "training preflight"),
    ):
        src = (_BACKEND / rel).read_text(encoding="utf-8")
        assert (
            "assert_pipeline_class_available(" in src
        ), f"{caller} ({rel}) no longer routes through the guarded probe; it needs its own guard"


def test_load_failure_is_logged_with_a_traceback():
    """The client only ever receives str(exc), so without exc_info here a one-line failure
    cannot be attributed to any call site. Both issues stalled for exactly this reason."""
    source = (_BACKEND / "core/inference/diffusion.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        args = [a for a in node.args if isinstance(a, ast.Constant)]
        if not any(str(a.value).startswith("diffusion.load_failed") for a in args):
            continue
        assert any(
            kw.arg == "exc_info" for kw in node.keywords
        ), "diffusion.load_failed must log exc_info, or no reporter can supply a traceback"
        return
    pytest.fail("no diffusion.load_failed log call found")


def test_the_dynamo_failure_is_rewritten_into_something_actionable():
    """The raw text names a private torch module and reads as a bug in the model, while the
    only remedy is a restart. Measured on torch 2.10: a process that has lost this import race
    does not recover, so "try again" in the same process is wrong advice."""
    from core.inference.diffusion import dynamo_partial_init_message

    exc = AttributeError(
        "partially initialized module 'torch._dynamo' has no attribute 'utils' "
        "(most likely due to a circular import)"
    )
    msg = dynamo_partial_init_message(exc)
    assert msg and "Restart Unsloth" in msg
    assert "torch._dynamo" in msg, "name the module so the log and the toast can be tied together"


def test_the_rewrite_finds_the_failure_through_an_exception_chain():
    """It surfaces from inside diffusers, so it arrives wrapped."""
    from core.inference.diffusion import dynamo_partial_init_message

    try:
        try:
            raise AttributeError("module 'torch._dynamo' has no attribute 'utils'")
        except AttributeError as inner:
            raise RuntimeError("Failed to import diffusers.hooks") from inner
    except RuntimeError as outer:
        assert dynamo_partial_init_message(outer) is not None


def test_a_suppressed_context_is_not_followed():
    """``raise ... from None`` means the raiser deliberately hid the inner error, so following
    __context__ past it would answer a visible, unrelated failure with restart advice that does
    not apply. Same walk as _gated_in_chain, which is what this file's neighbour already does."""
    from core.inference.diffusion import dynamo_partial_init_message

    try:
        try:
            raise AttributeError("module 'torch._dynamo' has no attribute 'utils'")
        except AttributeError:
            raise RuntimeError("Model weights are corrupt: checksum mismatch") from None
    except RuntimeError as visible:
        assert visible.__suppress_context__ is True
        assert (
            dynamo_partial_init_message(visible) is None
        ), "a hidden dynamo error replaced an unrelated visible failure"


def test_an_explicit_cause_is_still_followed():
    """``raise ... from inner`` sets __suppress_context__ too, but the cause is explicit: the
    raiser is pointing AT the inner error, so the rewrite must still find it."""
    from core.inference.diffusion import dynamo_partial_init_message

    try:
        try:
            raise AttributeError("module 'torch._dynamo' has no attribute 'utils'")
        except AttributeError as inner:
            raise RuntimeError("Failed to import diffusers.hooks") from inner
    except RuntimeError as outer:
        assert outer.__suppress_context__ is True
        assert dynamo_partial_init_message(outer) is not None


def test_a_cyclic_exception_chain_terminates():
    """The walk is guarded by identity, not a depth counter, so a self-referential chain
    cannot spin."""
    from core.inference.diffusion import dynamo_partial_init_message

    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert dynamo_partial_init_message(a) is None


def test_an_unrelated_load_failure_keeps_its_own_text():
    """Same contract as hub_access_message: rewrite only what it recognises, or a real error
    would be replaced by advice that does not apply to it."""
    from core.inference.diffusion import dynamo_partial_init_message

    assert dynamo_partial_init_message(RuntimeError("CUDA out of memory")) is None
    assert dynamo_partial_init_message(FileNotFoundError("no such file")) is None


def test_the_rewrite_is_wired_into_the_load_failure_handler():
    """Asserted on source, since reaching the handler needs a GPU and a model."""
    body = ast.unparse(_load_pipeline_failure_handler())
    assert (
        "dynamo_partial_init_message" in body
    ), "the load failure handler no longer rewrites the dynamo error"
    assert body.index("hub_access_message") < body.index(
        "dynamo_partial_init_message"
    ), "a gated-repo message must keep priority; it is the more specific diagnosis"


def _load_pipeline_failure_handler():
    tree = ast.parse((_BACKEND / "core/inference/diffusion.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            seg = ast.unparse(node)
            if "hub_access_message" in seg and "redact_native_paths" not in seg:
                return node
    raise AssertionError("could not find the load failure handler")


def test_trainer_reads_dynamo_config_defensively():
    """A bare ``torch._dynamo.config`` at module scope triggers the lazy import and can bind a
    half-built module, raising at import time where nothing handles it."""
    # AST, not a substring search: the prose explaining why this form is wrong necessarily
    # contains the wrong form, so a text match would fail on its own comment.
    tree = ast.parse((_BACKEND / "core/training/trainer.py").read_text(encoding="utf-8"))
    bare = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr == "config"
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "_dynamo"
        and isinstance(node.value.value, ast.Name)
        and node.value.value.id == "torch"
    ]
    assert not bare, (
        f"trainer.py reads torch._dynamo.config directly at line(s) "
        f"{[n.lineno for n in bare]}; use the getattr form"
    )
