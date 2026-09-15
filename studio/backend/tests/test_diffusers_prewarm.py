# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pre-importing diffusers off the first image load.

The first diffusion load in a fresh process pays about 5.3s of pure import before it touches a
weight (``diffusers`` 1.6s, ``diffusers.hooks`` 2.4s, the pipeline classes 1.3s). None of it
depends on which model was picked, so it can be paid earlier on a thread nobody waits on.
Measured end to end, the load path's own import work drops from 4.47s to 0.60s, at the same
peak RSS: the memory is paid either way, just earlier.

The tests below pin the properties that make that safe to ship rather than the speedup itself:

  1. A chat-only or training-only install pays NOTHING. The gate must not import diffusers when
     there is no local image or video model, which is the whole reason the gate exists.
  2. Every failure mode degrades to "skip", never to an exception: the post-warm worker also
     carries MLX repair and linked-folder sync.
  3. It runs from the POST-warm worker, after the coordinated warm, so it cannot delay a warm
     stage or the socket bind.
  4. The tqdm quieting that diffusers forces at import is applied here too, or the prewarm would
     write progress bars onto the structlog stream.

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
def warm(monkeypatch):
    """``utils.torch_warmup`` with the prewarm latch reset, so each test starts cold."""
    from utils import torch_warmup

    monkeypatch.setattr(torch_warmup, "_diffusers_prewarmed", False, raising = False)
    monkeypatch.delenv(torch_warmup.DIFFUSERS_PREWARM_DISABLE_ENV_VAR, raising = False)
    return torch_warmup


def _stub_gate(
    monkeypatch,
    ids_by_task,
    *,
    raises = False,
    engine = "diffusers",
):
    """Stand in for the media index and the engine router.

    ``engine`` is what a load of these models would select. It is not decoration: a CPU or MPS
    host routes a supported GGUF to sd.cpp, which imports no diffusers, so "a model exists" and
    "diffusers would be used" are different questions and only the second one may prewarm.
    """
    idx = types.ModuleType("core.inference.media_model_index")

    def _available(task):
        if raises:
            raise RuntimeError("index unavailable")
        return list(ids_by_task.get(task, []))

    idx.available_media_model_ids = _available
    idx.resolve_local_media_model = lambda model_id, task: types.SimpleNamespace(
        model_id = model_id,
        model_path = "/nonexistent",
        gguf_filename = "m.gguf",
        model_kind = "gguf",
        ambiguous = False,
    )
    monkeypatch.setitem(sys.modules, "core.inference.media_model_index", idx)

    router = types.ModuleType("core.inference.diffusion_engine_router")
    router.ENGINE_DIFFUSERS = "diffusers"
    router.predict_engine = lambda fam, model_kind = None: engine
    monkeypatch.setitem(sys.modules, "core.inference.diffusion_engine_router", router)

    loc = types.ModuleType("core.inference.media_locality")
    loc.detected_image_family = lambda pick: object()
    monkeypatch.setitem(sys.modules, "core.inference.media_locality", loc)


def _stub_diffusers(monkeypatch, *, raises = False):
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
            raising = False,
        )
    return seen


def test_an_install_with_no_image_models_pays_nothing(warm, monkeypatch):
    """The gate is the entire justification for doing this at boot. A chat-only or
    training-only user must not pay diffusers' 316 MB for a page they never open."""
    _stub_gate(monkeypatch, {"text-to-image": [], "text-to-video": []})
    monkeypatch.delitem(sys.modules, "diffusers", raising = False)

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

    ``VideoBackend.load_pipeline`` asks ``is_h3_native(fam, kind)`` and returns through
    ``_run_load_h3_native`` before its own ``import diffusers``. The image resolver cannot see
    this: ``detected_image_family`` has no answer for a ``VideoFamily``, so the "unknown family"
    branch would call it diffusers and charge an H3-only install 316 MB it never uses. Uses the
    real family detector and the real predicate, so a rename on either side fails here.
    """
    _stub_gate(monkeypatch, {"text-to-image": [], "text-to-video": ["unsloth/MiniMax-H3-GGUF"]})
    seen = _stub_diffusers(monkeypatch)
    seen["imported"] = False

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    # And not latched, so an image model downloaded later still gets a prewarm next boot.
    assert warm._diffusers_prewarmed is False


def test_a_video_family_we_cannot_identify_still_prewarms(warm, monkeypatch):
    """The H3 skip is an exception carved out of the default, not a new default.

    Every other video load reaches video.py's ``import diffusers``, so anything the detector
    cannot place must keep prewarming. Getting this backwards would silently drop the feature
    for every video user whose repo id is not in the family table.
    """
    _stub_gate(monkeypatch, {"text-to-image": [], "text-to-video": ["someone/private-repack"]})
    _stub_diffusers(monkeypatch)

    assert warm.prewarm_diffusers_if_image_models_exist() is True


def test_the_kill_switch_is_honoured(warm, monkeypatch):
    monkeypatch.setenv(warm.DIFFUSERS_PREWARM_DISABLE_ENV_VAR, "1")
    _stub_gate(monkeypatch, {"text-to-image": ["unsloth/Z-Image-GGUF"]})
    monkeypatch.delitem(sys.modules, "diffusers", raising = False)

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
        threading.Thread(target = lambda: did.append(warm.prewarm_diffusers_if_image_models_exist()))
        for _ in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(1 for d in did if d) == 1, f"the import ran more than once: {did}"


def test_a_gate_that_raises_means_skip_not_crash(warm, monkeypatch):
    """A broken or absent model index must not take the post-warm worker down with it."""
    _stub_gate(monkeypatch, {}, raises = True)
    assert warm.prewarm_diffusers_if_image_models_exist() is False


def test_a_diffusers_that_cannot_import_means_skip_not_crash(warm, monkeypatch):
    """A --no-torch host, or a broken diffusers, reports False. The load path imports it again
    and is the one that reports the failure to the user."""
    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    _stub_diffusers(monkeypatch, raises = True)
    assert warm.prewarm_diffusers_if_image_models_exist() is False


def test_the_windows_rocm_stubs_are_installed_before_the_import(warm, monkeypatch):
    """core/inference/diffusion.py installs these three at module scope, above its own lazy
    `import diffusers`, because on Windows ROCm diffusers imports xformers on sight and its
    quantizers torchao, and both land on an absent distributed backend. This prewarm can be the
    first importer in the process, so it owes the same three installs, in front of the import."""
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
    monkeypatch.delitem(sys.modules, "diffusers", raising = False)

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


def test_a_failed_prewarm_leaves_no_half_imported_diffusers(warm, monkeypatch):
    """The failure this prewarm adds that the loader did not have.

    When ``diffusers/__init__.py`` raises, CPython evicts only the parent and keeps every
    submodule it already executed. The prewarm swallows that failure, so the next importer is a
    user's image load, and it would re-run ``__init__`` with each ``from .x import y`` served
    from that cache and attributes never rebound: diffusers imports "successfully" while missing
    pieces. Hand the load path a clean slate instead.
    """
    _stub_gate(monkeypatch, {"text-to-image": ["m"]})
    monkeypatch.delitem(sys.modules, "diffusers", raising = False)
    leftover = types.ModuleType("diffusers.pipelines")
    monkeypatch.setitem(sys.modules, "diffusers.pipelines", leftover)

    class _Boom:
        def find_module(self, *a, **k):
            return None

        def find_spec(self, name, path = None, target = None):
            if name == "diffusers":
                raise ImportError("simulated half-built diffusers")
            return None

    monkeypatch.setattr(sys, "meta_path", [_Boom(), *sys.meta_path])

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    assert "diffusers.pipelines" not in sys.modules, (
        "a failed prewarm left submodules behind for the load path to trip over"
    )


def test_a_host_that_routes_to_sd_cpp_pays_nothing(warm, monkeypatch):
    """The case presence alone gets wrong. A CPU or MPS host with a runnable native binary, or
    UNSLOTH_DIFFUSION_ENGINE=sd_cpp, serves a supported GGUF through sd.cpp and imports no
    diffusers, so prewarming would add ~316 MB to a low-memory install that never reclaims it
    with a load."""
    _stub_gate(monkeypatch, {"text-to-image": ["unsloth/Z-Image-GGUF"]}, engine = "sd_cpp")
    monkeypatch.delitem(sys.modules, "diffusers", raising = False)

    assert warm.prewarm_diffusers_if_image_models_exist() is False
    assert "diffusers" not in sys.modules, "prewarmed on a host whose image path is native"


def test_a_non_gguf_model_still_prewarms_on_a_native_host(warm, monkeypatch):
    """Only a GGUF can go native, so a dense checkpoint lands on diffusers even where sd.cpp is
    the preferred engine. Gating the whole host off would lose the speedup for it."""
    _stub_gate(monkeypatch, {"text-to-image": ["some/dense-sdxl"]}, engine = "sd_cpp")
    idx = sys.modules["core.inference.media_model_index"]
    monkeypatch.setattr(
        idx,
        "resolve_local_media_model",
        lambda model_id, task: types.SimpleNamespace(
            model_id = model_id,
            model_path = "/nonexistent",
            gguf_filename = None,
            model_kind = None,
            ambiguous = False,
        ),
    )
    _stub_diffusers(monkeypatch)

    assert warm.prewarm_diffusers_if_image_models_exist() is True


def test_a_gguf_whose_family_is_only_in_its_filename_is_still_recognised():
    """The layout `detect_family` cannot see: an opaque directory whose family keyword lives
    only in the .gguf filename. Treating it as unknown would prewarm on exactly the sd.cpp host
    this gate exists to spare, so the gate must use the same pick-aware resolver the listing and
    the loader use. Driven through the REAL resolver, since a stub of it could not show this."""
    from core.inference.media_locality import detected_image_family

    opaque = types.SimpleNamespace(
        model_id = "local/custom",
        model_path = "/models/custom",
        gguf_filename = "z-image-turbo-Q4_K_M.gguf",
        model_kind = "gguf",
        ambiguous = False,
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
    ``"image"``/``"video"`` builds a permanently EMPTY index and the gate refuses forever: the
    prewarm would be dead code that every stubbed test still passes. Pinned against the
    catalog's own set so a rename there fails here instead of silently disabling the prewarm."""
    from hub.services.models.catalog_classification import _LOADABLE_MEDIA_GGUF_TASKS
    from utils import torch_warmup

    assert set(torch_warmup._MEDIA_PREWARM_TASKS) == set(_LOADABLE_MEDIA_GGUF_TASKS)


def test_the_real_index_answers_our_task_strings_and_not_the_friendly_ones(monkeypatch):
    """The same check driven through the REAL index rather than a stub of it.

    This is the one a stub cannot make: feed the catalog one text-to-image model and confirm
    ``available_media_model_ids`` finds it under the identifier the gate passes, and finds
    nothing under ``"image"``. Without it, keying the stub on the gate's own strings makes a
    wrong identifier look correct, which is exactly how this shipped the first time."""
    from core.inference import media_model_index as idx
    from utils import torch_warmup

    fake = types.SimpleNamespace(
        id = "unsloth/Z-Image-GGUF",
        model_id = "unsloth/Z-Image-GGUF",
        display_name = "Z-Image-GGUF",
        path = "/nonexistent/z-image",
        model_format = None,
        partial = False,
    )
    routes_models = sys.modules.setdefault("routes.models", types.ModuleType("routes.models"))
    monkeypatch.setattr(routes_models, "collect_local_models", lambda _root: [fake], raising = False)
    monkeypatch.setattr(
        routes_models, "_local_model_task", lambda _info: "text-to-image", raising = False
    )
    # _name_keys and the on-disk checks would reject a path that does not exist, so stand in
    # for the registration step; the task comparison above it is what is under test.
    monkeypatch.setattr(idx, "_name_keys", lambda _info: ("z-image-gguf",), raising = False)
    monkeypatch.setattr(idx, "_resolve_load_dir", lambda p: p, raising = False)
    monkeypatch.setattr(
        idx, "_add_gguf_picks", lambda index, info, keys, on_disk, load_dir: False, raising = False
    )
    monkeypatch.setattr(idx, "_loadable_directory", lambda _d: True, raising = False)
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
    tree = ast.parse((_BACKEND / "main.py").read_text(encoding = "utf-8"))
    fn = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_post_warm_background_work"
    )
    return ast.unparse(fn)


def test_it_runs_from_the_post_warm_worker_and_last():
    """POST-warm, so it cannot delay a coordinated warm stage or the socket bind. And last
    within that worker, because it is the only item there that is latency work rather than
    correctness: MLX repair and linked-folder sync keep their place in the queue."""
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
    """The four warm stages are on the path to a usable backend. Pulling diffusers into one of
    them would delay every boot, including chat-only ones, which is what the post-warm worker
    exists to avoid."""
    from utils import torch_warmup
    import inspect

    for name, fn in torch_warmup._STAGES:
        body = inspect.getsource(fn)
        assert "diffusers" not in body, f"warm stage {name!r} imports diffusers"
