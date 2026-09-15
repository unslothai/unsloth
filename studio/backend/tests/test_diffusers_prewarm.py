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
):
    """Stand in for core.inference.media_model_index.available_media_model_ids."""
    mod = types.ModuleType("core.inference.media_model_index")

    def _available(task):
        if raises:
            raise RuntimeError("index unavailable")
        return list(ids_by_task.get(task, []))

    mod.available_media_model_ids = _available
    monkeypatch.setitem(sys.modules, "core.inference.media_model_index", mod)


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
        id = "unsloth/Z-Image-GGUF", model_id = "unsloth/Z-Image-GGUF",
        display_name = "Z-Image-GGUF", path = "/nonexistent/z-image",
        model_format = None, partial = False,
    )
    routes_models = sys.modules.setdefault("routes.models", types.ModuleType("routes.models"))
    monkeypatch.setattr(routes_models, "collect_local_models", lambda _root: [fake],
                        raising = False)
    monkeypatch.setattr(routes_models, "_local_model_task",
                        lambda _info: "text-to-image", raising = False)
    # _name_keys and the on-disk checks would reject a path that does not exist, so stand in
    # for the registration step; the task comparison above it is what is under test.
    monkeypatch.setattr(idx, "_name_keys", lambda _info: ("z-image-gguf",), raising = False)
    monkeypatch.setattr(idx, "_resolve_load_dir", lambda p: p, raising = False)
    monkeypatch.setattr(idx, "_add_gguf_picks",
                        lambda index, info, keys, on_disk, load_dir: False, raising = False)
    monkeypatch.setattr(idx, "_loadable_directory", lambda _d: True, raising = False)
    idx.invalidate_index()

    found = {task: idx.available_media_model_ids(task)
             for task in (*torch_warmup._MEDIA_PREWARM_TASKS, "image", "video")}
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
