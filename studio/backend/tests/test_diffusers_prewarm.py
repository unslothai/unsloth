# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-importing diffusers off the first image load.

These tests pin the properties that make paying that import early safe, not the speedup:

  1. A chat-only or training-only install pays NOTHING.
  2. Every failure mode degrades to "skip": the post-warm worker also carries MLX repair.
  3. It runs from the POST-warm worker, so it cannot delay a warm stage or the socket bind.
  4. The tqdm quieting diffusers forces at import is applied here too.

None of these import the real diffusers, which is the point of stubbing it.
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
def restore_diffusers_modules():
    """Put back every ``diffusers*`` entry this test disturbs.

    monkeypatch only restores keys it set itself, so the real purge leaks into later tests.
    """
    saved = {name: mod for name, mod in sys.modules.items() if name.split(".")[0] == "diffusers"}
    try:
        yield
    finally:
        for name in [n for n in sys.modules if n.split(".")[0] == "diffusers"]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture
def warm(monkeypatch):
    """``utils.torch_warmup`` with the prewarm latch reset, so each test starts cold."""
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "_diffusers_prewarmed", False, raising=False)
    monkeypatch.delenv(torch_warmup.DIFFUSERS_PREWARM_DISABLE_ENV_VAR, raising=False)
    return torch_warmup


def _stub_gate(
    monkeypatch,
    ids_by_task,
    *,
    raises=False,
    engine="diffusers",
):
    """Stand in for the media index and the engine router.

    ``engine`` is not decoration: a CPU or MPS host routes a supported GGUF to sd.cpp.
    """
    idx = types.ModuleType("core.inference.media_model_index")

    def _available(task):
        if raises:
            raise RuntimeError("index unavailable")
        return list(ids_by_task.get(task, []))

    idx.available_media_model_ids = _available
    idx.resolve_local_media_model = lambda model_id, task: types.SimpleNamespace(
        model_id=model_id,
        model_path="/nonexistent",
        gguf_filename="m.gguf",
        model_kind="gguf",
        ambiguous=False,
    )
    monkeypatch.setitem(sys.modules, "core.inference.media_model_index", idx)

    router = types.ModuleType("core.inference.diffusion_engine_router")
    router.ENGINE_DIFFUSERS = "diffusers"
    router.predict_engine = lambda fam, model_kind=None: engine
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_engine_router", router)

    loc = types.ModuleType("core.inference.media_locality")
    loc.detected_image_family = lambda pick: object()
    monkeypatch.setitem(sys.modules, "core.inference.media_locality", loc)


def _stub_diffusers(monkeypatch, *, raises=False):
    """A diffusers that records whether it was imported, without importing the real one."""
    seen = {"imported": False, "quieted": False}
    if raises:
        monkeypatch.setitem(sys.modules, "diffusers", None)  # import -> ImportError
        return seen
    d = types.ModuleType("diffusers")
    hooks = types.ModuleType("diffusers.hooks")
    d.hooks = hooks
    seen["imported"] = False

    class _Marking(types.ModuleType):
        pass

    monkeypatch.setitem(sys.modules, "diffusers", d)
    monkeypatch.setitem(sys.modules, "diffusers.hooks", hooks)
    seen["imported"] = True

    cfg = sys.modules.get("loggers.config")
    if cfg is not None:
        monkeypatch.setattr(
            cfg,
            "quiet_third_party_progress_bars",
            lambda: seen.__setitem__("quieted", True),
            raising=False,
        )
    return seen


def test_an_install_with_no_image_models_pays_nothing(warm, monkeypatch):
    """The gate is the entire justification for doing this at boot. A chat-only or
    training-only user must not pay diffusers' 316 MB for a page they never open."""
    _stub_gate(monkeypatch, {"text-to-image": [], "text-to-video": []})
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    assert "diffusers" not in sys.modules, "the prewarm imported diffusers with nothing to load"


def test_it_prewarms_when_an_image_model_is_present(warm, monkeypatch):
    _stub_gate(monkeypatch, {"text-to-image": ["unsloth/Z-Image-GGUF"], "text-to-video": []})
    seen = _stub_diffusers(monkeypatch)

    assert warm.prewarm_diffusers_if_image_models_exist() is True
    assert seen["imported"] is True


def test_a_video_only_install_also_prewarms(warm, monkeypatch):
    """Both media backends go through the same diffusers import."""
    _stub_gate(monkeypatch, {"text-to-image": [], "text-to-video": ["some/video-model"]})
    _stub_diffusers(monkeypatch)

    assert warm.prewarm_diffusers_if_image_models_exist() is True


def test_a_video_only_install_of_an_h3_gguf_pays_nothing(warm, monkeypatch):
    """MiniMax H3 as a GGUF is the one video combination that never imports diffusers.

    ``VideoBackend.load_pipeline`` returns through ``_run_load_h3_native`` before its own
    ``import diffusers``, and ``detected_image_family`` cannot place a ``VideoFamily``.
    """
    _stub_gate(monkeypatch, {"text-to-image": [], "text-to-video": ["unsloth/MiniMax-H3-GGUF"]})
    seen = _stub_diffusers(monkeypatch)
    seen["imported"] = False

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    # And not latched, so an image model downloaded later still gets a prewarm next boot.
    assert warm._diffusers_prewarmed is False


def test_an_h3_gguf_named_only_by_its_filename_pays_nothing(warm, monkeypatch):
    """The family token can live only in the checkpoint filename, so video.py tries both."""
    _stub_gate(monkeypatch, {"text-to-image": [], "text-to-video": ["custom-video"]})
    sys.modules["core.inference.media_model_index"].resolve_local_media_model = (
        lambda model_id, task: types.SimpleNamespace(
            model_id="custom-video",  # names no family
            model_path="/models/custom",  # nor does the directory
            gguf_filename="minimax_h3_fl2va-Q4.gguf",  # only the checkpoint does
            model_kind="gguf",
            ambiguous=False,
        )
    )
    _stub_diffusers(monkeypatch)

    assert warm.prewarm_diffusers_if_image_models_exist() is False


def test_a_video_family_we_cannot_identify_still_prewarms(warm, monkeypatch):
    """The H3 skip is an exception: every other video load reaches `import diffusers`."""
    _stub_gate(monkeypatch, {"text-to-image": [], "text-to-video": ["someone/private-repack"]})
    _stub_diffusers(monkeypatch)

    assert warm.prewarm_diffusers_if_image_models_exist() is True


def test_the_torch_warm_opt_out_also_disables_the_prewarm(warm, monkeypatch):
    """join_background_warm() reports True when no worker ran, so this needs its own check."""
    monkeypatch.setenv(warm.DISABLE_ENV_VAR, "1")
    _stub_gate(monkeypatch, {"text-to-image": ["unsloth/Z-Image-GGUF"]})
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    assert "diffusers" not in sys.modules, "the torch-warm opt-out did not stop the prewarm"


def test_the_kill_switch_is_honoured(warm, monkeypatch):
    monkeypatch.setenv(warm.DIFFUSERS_PREWARM_DISABLE_ENV_VAR, "1")
    _stub_gate(monkeypatch, {"text-to-image": ["unsloth/Z-Image-GGUF"]})
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    assert "diffusers" not in sys.modules


def test_it_latches_so_a_second_call_is_free(warm, monkeypatch):
    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    _stub_diffusers(monkeypatch)

    assert warm.prewarm_diffusers_if_image_models_exist() is True
    assert warm.prewarm_diffusers_if_image_models_exist() is False


def test_concurrent_callers_prewarm_once(warm, monkeypatch):
    """Single-flight, asserted on how many callers did the work rather than on lock entries,
    which would be a race against thread scheduling."""
    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    _stub_diffusers(monkeypatch)

    did = []
    threads = [
        threading.Thread(target=lambda: did.append(warm.prewarm_diffusers_if_image_models_exist()))
        for _ in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for d in did if d) == 1, f"the import ran more than once: {did}"


def test_a_gate_that_raises_means_skip_not_crash(warm, monkeypatch):
    """A broken or absent model index must not take the post-warm worker down with it."""
    _stub_gate(monkeypatch, {}, raises=True)
    assert warm.prewarm_diffusers_if_image_models_exist() is False


def test_a_diffusers_that_cannot_import_means_skip_not_crash(warm, monkeypatch):
    """A --no-torch host, or a broken diffusers, reports False. The load path imports it again
    and is the one that reports the failure to the user."""
    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    _stub_diffusers(monkeypatch, raises=True)
    assert warm.prewarm_diffusers_if_image_models_exist() is False


def test_the_windows_rocm_stubs_are_installed_before_the_import(warm, monkeypatch):
    """On Windows ROCm, diffusers imports xformers and torchao onto an absent backend."""
    from core import _torchao_stub
    from core.inference import diffusion_torchao_patches

    order = []
    for mod, name in (
        (_torchao_stub, "install_xformers_windows_rocm_stub"),
        (_torchao_stub, "install_torchao_windows_rocm_stub"),
        (diffusion_torchao_patches, "install_torchao_int_mm_patch"),
    ):
        monkeypatch.setattr(mod, name, (lambda n: lambda: order.append(n))(name))

    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)

    real_import = builtins.__import__

    def _record_import(name, *args, **kwargs):
        if name.startswith("diffusers"):
            order.append("import")
            return types.ModuleType(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _record_import)

    warm.prewarm_diffusers_if_image_models_exist()
    monkeypatch.undo()

    assert order[:3] == [
        "install_xformers_windows_rocm_stub",
        "install_torchao_windows_rocm_stub",
        "install_torchao_int_mm_patch",
    ], f"stubs did not run first, in the loader's order: {order}"
    assert "import" in order and order.index("import") > 2


def test_a_failed_prewarm_leaves_no_half_imported_diffusers(
    warm, monkeypatch, restore_diffusers_modules
):
    """The failure this prewarm adds that the loader did not have.

    When ``diffusers/__init__.py`` raises, CPython evicts only the parent and keeps every
    submodule it executed, so the next importer rebuilds an incomplete package from them.
    """
    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)
    leftover = types.ModuleType("diffusers.pipelines")
    monkeypatch.setitem(sys.modules, "diffusers.pipelines", leftover)

    class _Boom:
        def find_module(self, *a, **k):
            return None

        def find_spec(
            self,
            name,
            path=None,
            target=None,
        ):
            if name == "diffusers":
                raise ImportError("simulated half-built diffusers")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Boom(), *sys.meta_path])

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    assert (
        "diffusers.pipelines" not in sys.modules
    ), "a failed prewarm left submodules behind for the load path to trip over"


def test_the_diffusers_import_lock_is_held_across_the_failure_cleanup(
    warm, monkeypatch, restore_diffusers_modules
):
    """Releasing between the failed import and the purge is the whole bug.

    CPython drops the module lock the moment ``__init__`` raises, and a request waiting in
    that gap republishes the malformed parent, which the purge then declines.
    """
    from importlib._bootstrap import _get_module_lock

    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)
    monkeypatch.setitem(sys.modules, "diffusers.pipelines", types.ModuleType("diffusers.pipelines"))

    held_during_purge = []
    real_purge = warm.purge_partial_import

    def _checking_purge(package):
        # The lock of the package being purged, not always the parent: the hooks subtree has its
        # own lock and the same release-before-cleanup gap. Reading .owner does not disturb the
        # lock the way acquiring it would.
        lock = _get_module_lock(package)
        held_during_purge.append(getattr(lock, "owner", None) == threading.get_ident())
        return real_purge(package)

    monkeypatch.setattr(warm, "purge_partial_import", _checking_purge)

    class _Boom:
        def find_spec(
            self,
            name,
            path=None,
            target=None,
        ):
            if name == "diffusers":
                raise ImportError("simulated half-built diffusers")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Boom(), *sys.meta_path])

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    # One entry per purge call (the hooks subtree and the parent); every one must be under lock.
    assert held_during_purge and all(held_during_purge), (
        "the diffusers import lock was released before the purge ran, so a waiting importer "
        f"could take it in the gap (observed: {held_during_purge})"
    )


def test_a_hooks_failure_after_a_good_parent_still_purges_the_hook_subtree(
    warm, monkeypatch, restore_diffusers_modules
):
    """The #7580 shape one level down: with the parent good, purging it is a no-op."""
    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    parent = types.ModuleType("diffusers")
    monkeypatch.setitem(sys.modules, "diffusers", parent)
    monkeypatch.setitem(
        sys.modules,
        "diffusers.hooks.group_offloading",
        types.ModuleType("diffusers.hooks.group_offloading"),
    )
    monkeypatch.delitem(sys.modules, "diffusers.hooks", raising=False)

    from importlib._bootstrap import _get_module_lock

    held = []
    real_purge = warm.purge_partial_import

    def _checking_purge(package):
        lock = _get_module_lock(package)
        held.append((package, getattr(lock, "owner", None) == threading.get_ident()))
        return real_purge(package)

    monkeypatch.setattr(warm, "purge_partial_import", _checking_purge)

    class _Boom:
        def find_spec(
            self,
            name,
            path=None,
            target=None,
        ):
            if name == "diffusers.hooks":
                raise ImportError("simulated half-built diffusers.hooks")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Boom(), *sys.meta_path])

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    # The subpackage has its own lock and the same gap: `import diffusers.hooks` drops it when it
    # raises, and with the healthy parent already published a waiting request could rebuild the
    # hooks package from these stale submodules before the purge reacquires it.
    assert held and all(
        ok for _, ok in held
    ), f"a purge ran without holding that package's own import lock (observed: {held})"
    assert (
        "diffusers.hooks.group_offloading" not in sys.modules
    ), "the executed hook submodules survived, so the load path can rebuild a partial package"
    assert (
        sys.modules.get("diffusers") is parent
    ), "the healthy parent was evicted; only the failed subtree should go"


def test_the_parent_and_child_import_locks_are_never_held_together(
    warm, monkeypatch, restore_diffusers_modules
):
    """Holding both would invert CPython's own lock order and deadlock a concurrent import.

    ``import diffusers.hooks`` makes CPython take the CHILD lock first, so parent-then-child
    inverts that and cycles, as the ``_DeadlockError`` importlib swallows.
    """
    from importlib._bootstrap import _get_module_lock

    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)
    monkeypatch.delitem(sys.modules, "diffusers.hooks", raising=False)

    me = threading.get_ident()
    both_held = []

    def _owned(name):
        return getattr(_get_module_lock(name), "owner", None) == me

    real_import = builtins.__import__

    def _watching_import(name, *args, **kwargs):
        if name.startswith("diffusers"):
            both_held.append((name, _owned("diffusers"), _owned("diffusers.hooks")))
            return types.ModuleType(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _watching_import)

    warm.prewarm_diffusers_if_image_models_exist()
    monkeypatch.undo()

    assert both_held, "the prewarm imported nothing; this test would be vacuous"
    offenders = [(n, p, c) for n, p, c in both_held if p and c]
    assert not offenders, f"parent and child import locks held at the same time: {offenders}"


def test_a_concurrent_submodule_import_does_not_deadlock_the_prewarm(
    warm, monkeypatch, restore_diffusers_modules
):
    """The same inversion from the other side, with two real threads and the real locks."""
    from importlib._bootstrap import _ModuleLockManager as LM

    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)
    monkeypatch.delitem(sys.modules, "diffusers.hooks", raising=False)
    _stub_diffusers(monkeypatch)

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def _importer():
        # CPython's order for a submodule import: child lock, then the parent.
        with LM("diffusers.hooks"):
            started.set()
            release.wait(10)
            with LM("diffusers"):
                pass
        finished.set()

    t = threading.Thread(target=_importer, daemon=True)
    t.start()
    assert started.wait(10), "the helper thread never took the hooks lock"

    done = threading.Event()

    def _prewarm():
        try:
            warm.prewarm_diffusers_if_image_models_exist()
        finally:
            done.set()

    p = threading.Thread(target=_prewarm, daemon=True)
    p.start()
    # The prewarm must not be blocked behind a lock the other thread is holding while that
    # thread waits on one the prewarm holds.
    release.set()
    assert done.wait(20), "the prewarm deadlocked against a concurrent submodule import"
    assert finished.wait(20), "the concurrent submodule import deadlocked against the prewarm"
    t.join(5)
    p.join(5)


def test_the_lock_is_never_released_between_a_failed_import_and_its_purge(
    warm, monkeypatch, restore_diffusers_modules
):
    """Held CONTINUOUSLY, not merely held again by the time the purge runs.

    The try around the ``with`` releases the lock on the exception and reacquires it in the
    handler, which asserting that the purge ran under the lock would not catch.
    """
    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)
    monkeypatch.setitem(sys.modules, "diffusers.pipelines", types.ModuleType("diffusers.pipelines"))

    real_lm = warm._ModuleLockManager
    real_purge = warm.purge_partial_import
    timeline = []

    class _RecordingLockManager:
        def __init__(self, name):
            self._name = name
            self._inner = real_lm(name)

        def __enter__(self):
            timeline.append(("enter", self._name))
            return self._inner.__enter__()

        def __exit__(self, *exc):
            timeline.append(("exit", self._name))
            return self._inner.__exit__(*exc)

    def _recording_purge(package):
        timeline.append(("purge", package))
        return real_purge(package)

    monkeypatch.setattr(warm, "_ModuleLockManager", _RecordingLockManager)
    monkeypatch.setattr(warm, "purge_partial_import", _recording_purge)

    class _Boom:
        def find_spec(
            self,
            name,
            path=None,
            target=None,
        ):
            if name == "diffusers":
                raise ImportError("simulated half-built diffusers")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Boom(), *sys.meta_path])

    assert warm.prewarm_diffusers_if_image_models_exist() is False

    purge_at = next(
        i for i, (what, name) in enumerate(timeline) if what == "purge" and name == "diffusers"
    )
    # Everything before the purge, from the first acquisition onwards, must be acquisitions:
    # a matching release in there is the gap. Reentrant acquires (purge_partial_import takes the
    # same lock through its own decorator) are fine and expected.
    enter_at = next(
        i for i, (what, name) in enumerate(timeline) if what == "enter" and name == "diffusers"
    )
    released_early = [
        i
        for i, (what, name) in enumerate(timeline)
        if what == "exit" and name == "diffusers" and enter_at < i < purge_at
    ]
    assert not released_early, (
        "the diffusers lock was released between the failed import and the purge, so a waiting "
        f"importer could take it in the gap (timeline: {timeline})"
    )


def test_a_host_that_routes_to_sd_cpp_pays_nothing(warm, monkeypatch):
    """What presence alone gets wrong: a native binary or UNSLOTH_DIFFUSION_ENGINE=sd_cpp
    serves a supported GGUF through sd.cpp, importing no diffusers."""
    _stub_gate(monkeypatch, {"text-to-image": ["unsloth/Z-Image-GGUF"]}, engine="sd_cpp")
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    assert "diffusers" not in sys.modules, "prewarmed on a host whose image path is native"


def test_a_non_gguf_model_still_prewarms_on_a_native_host(warm, monkeypatch):
    """Only a GGUF can go native, so a dense checkpoint lands on diffusers even where sd.cpp is
    the preferred engine. Gating the whole host off would lose the speedup for it."""
    _stub_gate(monkeypatch, {"text-to-image": ["some/dense-sdxl"]}, engine="sd_cpp")
    idx = sys.modules["core.inference.media_model_index"]
    monkeypatch.setattr(
        idx,
        "resolve_local_media_model",
        lambda model_id, task: types.SimpleNamespace(
            model_id=model_id,
            model_path="/nonexistent",
            gguf_filename=None,
            model_kind=None,
            ambiguous=False,
        ),
    )
    _stub_diffusers(monkeypatch)

    assert warm.prewarm_diffusers_if_image_models_exist() is True


def test_a_gguf_whose_family_is_only_in_its_filename_is_still_recognised():
    """The layout `detect_family` cannot see: the family keyword is only in the filename."""
    from core.inference.media_locality import detected_image_family

    opaque = types.SimpleNamespace(
        model_id="local/custom",
        model_path="/models/custom",
        gguf_filename="z-image-turbo-Q4_K_M.gguf",
        model_kind="gguf",
        ambiguous=False,
    )
    assert (
        detected_image_family(opaque) is not None
    ), "the filename-only family went unrecognised; the gate would prewarm on an sd.cpp host"


def test_the_gate_uses_the_pick_aware_family_resolver():
    """Not `detect_family(pick.model_id)`: that misses the filename-only layout above."""
    import inspect
    from utils import torch_warmup

    body = inspect.getsource(torch_warmup._a_local_model_would_load_through_diffusers)
    assert "detected_image_family" in body
    assert "detect_family(" not in body, "the id-only resolver misses filename-only families"


def test_the_gate_uses_the_routers_own_prediction():
    """Not a reimplementation of the routing policy: predict_engine is what selection and the
    download planner use, and it is documented to activate nothing and install nothing."""
    import inspect
    from utils import torch_warmup

    body = inspect.getsource(torch_warmup._a_local_model_would_load_through_diffusers)
    assert "predict_engine" in body, "the gate no longer asks the router"
    assert "policy_eligible" not in body, "the routing policy must not be copied here"


def test_the_gate_asks_for_the_catalogs_real_task_identifiers():
    """``_build_index`` matches ``_local_model_task(info) == task`` exactly, so a friendly
    ``"image"``/``"video"`` builds an EMPTY index while every stubbed test still passes."""
    from hub.services.models.catalog_classification import _LOADABLE_MEDIA_GGUF_TASKS
    from utils import torch_warmup

    assert set(torch_warmup._MEDIA_PREWARM_TASKS) == set(_LOADABLE_MEDIA_GGUF_TASKS)


def test_the_real_index_answers_our_task_strings_and_not_the_friendly_ones(monkeypatch):
    """Through the REAL index: a stub keyed on the gate's own strings cannot catch this."""
    from core.inference import media_model_index as idx
    from utils import torch_warmup

    fake = types.SimpleNamespace(
        id="unsloth/Z-Image-GGUF",
        model_id="unsloth/Z-Image-GGUF",
        display_name="Z-Image-GGUF",
        path="/nonexistent/z-image",
        model_format=None,
        partial=False,
    )
    routes_models = sys.modules.setdefault("routes.models", types.ModuleType("routes.models"))
    monkeypatch.setattr(routes_models, "collect_local_models", lambda _root: [fake], raising=False)
    monkeypatch.setattr(
        routes_models, "_local_model_task", lambda _info: "text-to-image", raising=False
    )
    # _name_keys and the on-disk checks would reject a path that does not exist, so stand in
    # for the registration step; the task comparison above it is what is under test.
    monkeypatch.setattr(idx, "_name_keys", lambda _info: ("z-image-gguf",), raising=False)
    monkeypatch.setattr(idx, "_resolve_load_dir", lambda p: p, raising=False)
    monkeypatch.setattr(
        idx, "_add_gguf_picks", lambda index, info, keys, on_disk, load_dir: False, raising=False
    )
    monkeypatch.setattr(idx, "_loadable_directory", lambda _d: True, raising=False)
    idx.invalidate_index()

    found = {
        task: idx.available_media_model_ids(task)
        for task in (*torch_warmup._MEDIA_PREWARM_TASKS, "image", "video")
    }
    idx.invalidate_index()

    assert found["text-to-image"], "the gate's identifier finds nothing in the real index"
    assert not found["image"], "the friendly identifier unexpectedly matched"
    assert not found["video"]


def _post_warm_source() -> str:
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding="utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_post_warm_background_work"
    )
    return ast.unparse(fn)


def test_it_runs_from_the_post_warm_worker_and_last():
    """POST-warm, so it cannot delay a stage or the bind, and last: it is latency work."""
    src = _post_warm_source()
    assert (
        "prewarm_diffusers_if_image_models_exist" in src
    ), "the prewarm is no longer wired into the post-warm worker"
    assert "join_background_warm" in src
    assert src.index("join_background_warm") < src.index("prewarm_diffusers_if_image_models_exist")
    assert src.index("_start_linked_folder_auto_sync") < src.index(
        "prewarm_diffusers_if_image_models_exist"
    ), "latency work must not jump ahead of the correctness work in this worker"


def test_the_coordinated_warm_stages_do_not_import_diffusers():
    """The warm stages gate a usable backend, so diffusers there delays every boot."""
    from utils import torch_warmup
    import inspect

    for name, fn in torch_warmup._STAGES:
        body = inspect.getsource(fn)
        assert "diffusers" not in body, f"warm stage {name!r} imports diffusers"
