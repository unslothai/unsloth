# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in media auto-switch: the name resolver and the switch it drives.

The local-model scan is replaced with fixture entries pointing at real (empty) files, and
the load routes with fakes, so these exercise resolution, the drain/load sequencing and the
error envelopes without torch, diffusers, weights or a GPU.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import time
import types

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import core.inference.gpu_arbiter as arb
import core.inference.media_auto_switch as mas
import core.inference.media_keepwarm as mk
import core.inference.media_locality as locality
import core.inference.media_model_index as index
import core.inference.media_switch_backends as backends
import core.inference.media_switch_errors as errors
import routes.models as models_route
import utils.openai_auto_switch_settings as settings
from auth.authentication import get_current_subject
from core.inference.openai_auto_download import preferred_quant
from utils.api_errors import install_api_error_handlers


def _a_real_video_family(name = "wan2.2-ti2v-5b"):
    """A real ``VideoFamily``, for tests that just need the route to carry one.

    A bare ``object()`` stands in for a forty-field frozen dataclass, so the first field
    nobody hand-copied fails the test on an AttributeError about nothing it asserts.
    """
    from core.inference.video_families import detect_video_family

    # By name, not repo id: which tokens a repo id matches is its own rule, tested elsewhere.
    fam = detect_video_family("", override = name)
    assert fam is not None, f"the video family registry no longer has {name!r}"
    return fam


def _video_load_backend(**overrides):
    """A real ``VideoBackend`` with only the asserted methods replaced.

    Every other call the route makes runs the real implementation, which for the load
    route's preflight and reservation helpers is pure registry resolution: no hub, no GPU,
    no weights, and ``__init__`` only allocates locks. A hand-rolled stub instead needs
    extending every time the route grows a call, and until it is, unrelated tests fail on a
    missing attribute rather than on what they assert.

    ``overrides`` are checked against the class, so a stub for a method that does not exist
    fails loudly instead of quietly never being called.
    """
    from core.inference.video import VideoBackend

    unknown = sorted(name for name in overrides if not hasattr(VideoBackend, name))
    assert not unknown, f"not part of the video backend's surface: {unknown}"

    backend = VideoBackend()
    for name, impl in overrides.items():
        setattr(backend, name, impl)
    return backend


def _info(
    model_id,
    path,
    *,
    task,
    display_name = None,
    model_format = None,
    source = "models_dir",
):
    """One local-model scan row, in the shape ``collect_local_models`` returns."""
    return types.SimpleNamespace(
        id = model_id,
        model_id = model_id,
        display_name = display_name or model_id,
        path = str(path),
        model_format = model_format,
        source = source,
        task = task,
    )


def _hf_cache_repo(root, repo_id, *, files):
    """A minimal HF cache repo: ``models--org--name/snapshots/<sha>/<file> -> ../../blobs/<sha>``.

    The symlinks are the point. Both bugs this layout covers only appear once the files are
    links into ``blobs/`` and the entry path is the repo root rather than the snapshot.
    """
    sha = "a" * 40
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / sha
    blobs = repo_dir / "blobs"
    snapshot.mkdir(parents = True)
    blobs.mkdir(parents = True)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text(sha)
    for index, name in enumerate(files):
        blob = blobs / f"blob{index}"
        blob.write_bytes(b"")
        (snapshot / name).symlink_to(blob)
    return repo_dir, snapshot


@pytest.fixture(autouse = True)
def _clean_index():
    mas.invalidate_index()
    yield
    mas.invalidate_index()


@pytest.fixture
def catalog(monkeypatch):
    """An empty local-model catalog the tests fill, with the task read off each row."""
    rows: list = []
    monkeypatch.setattr(models_route, "collect_local_models", lambda root: list(rows))
    monkeypatch.setattr(models_route, "_local_model_task", lambda info: info.task)
    return rows


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(settings, "get_media_auto_switch_enabled", lambda: True)


@pytest.fixture
def takes_the_gpu(monkeypatch):
    """Pin the load to the GPU-taking path instead of inheriting the host's device.

    Whether the switch waits on chat and on the other media backend, and which gates it holds,
    is decided by ``load_takes_the_gpu`` -- so a test about that wait reads the running host
    unless it says otherwise, and passes on a CUDA box while failing on every CPU-only CI
    runner. Both bindings, like the CPU test below: the drain sizes its wait with one, the
    switch decides on the gpu lock with the other.
    """
    monkeypatch.setattr(backends, "load_takes_the_gpu", lambda: True)
    monkeypatch.setattr(mas, "load_takes_the_gpu", lambda: True)


# ── resolving a name ────────────────────────────────────────────────


def test_a_diffusers_directory_resolves_as_a_kindless_pick(flux):
    pick = mas.resolve_local_media_model("black-forest-labs/FLUX.1-dev", task = mas.IMAGE_TASK)

    # No kind and no filename: the load route detects those, including the single-file case.
    assert pick == mas.MediaModelPick("black-forest-labs/FLUX.1-dev", str(flux))


def test_a_gguf_repo_resolves_bare_and_per_quant(catalog, tmp_path):
    repo = tmp_path / "flux-gguf"
    repo.mkdir()
    (repo / "model_index.json").write_text("{}")
    for quant in ("Q4_K_M", "Q8_0"):
        (repo / f"flux1-dev-{quant}.gguf").write_bytes(b"")
    catalog.append(_info("city96/FLUX.1-dev-gguf", repo, task = mas.IMAGE_TASK))

    qualified = mas.resolve_local_media_model("city96/FLUX.1-dev-gguf:Q8_0", task = mas.IMAGE_TASK)
    assert qualified.gguf_filename == "flux1-dev-Q8_0.gguf"
    assert qualified.model_kind == "gguf"
    assert qualified.model_path == str(repo)
    # The id stays bare, so a repo with eight quants is one row in a "not found" error.
    assert qualified.model_id == "city96/FLUX.1-dev-gguf"
    assert mas.available_media_model_ids(mas.IMAGE_TASK) == ["city96/FLUX.1-dev-gguf"]

    # A bare id means the quant a plain load would take, so both surfaces agree on one answer.
    bare = mas.resolve_local_media_model("city96/FLUX.1-dev-gguf", task = mas.IMAGE_TASK)
    assert bare.gguf_filename == f"flux1-dev-{preferred_quant(['Q4_K_M', 'Q8_0'])}.gguf"


def test_a_cached_gguf_repo_resolves_to_its_repo_id(catalog, tmp_path):
    # A snapshot entry is a symlink into blobs/, and the loader resolves a symlink before its
    # containment check, so a snapshot directory refuses its own file ("gguf_filename must
    # resolve to a file inside the repo"). Name the repo the way the picker names a Hub pick.
    repo_dir, _snapshot = _hf_cache_repo(
        tmp_path, "unsloth/Z-Image-Turbo-GGUF", files = ["z-image-turbo-Q4_K_S.gguf"]
    )
    catalog.append(
        _info(
            "unsloth/Z-Image-Turbo-GGUF",
            repo_dir,
            task = mas.IMAGE_TASK,
            model_format = "gguf",
            source = "hf_cache",
        )
    )

    pick = mas.resolve_local_media_model("unsloth/Z-Image-Turbo-GGUF", task = mas.IMAGE_TASK)

    assert pick.model_path == "unsloth/Z-Image-Turbo-GGUF"
    assert pick.gguf_filename == "z-image-turbo-Q4_K_S.gguf"


def test_a_cached_pipeline_resolves_to_its_snapshot_directory(catalog, tmp_path):
    # The scan reports the repo ROOT, which holds no model_index.json ("Local pipeline
    # directory has no model_index.json"). The pipeline lives one level down.
    repo_dir, snapshot = _hf_cache_repo(
        tmp_path, "Tongyi-MAI/Z-Image-Turbo", files = ["model_index.json"]
    )
    catalog.append(
        _info("Tongyi-MAI/Z-Image-Turbo", repo_dir, task = mas.IMAGE_TASK, source = "hf_cache")
    )

    pick = mas.resolve_local_media_model("Tongyi-MAI/Z-Image-Turbo", task = mas.IMAGE_TASK)

    assert pick.model_path == str(snapshot)
    assert pick.gguf_filename is None and pick.model_kind is None


def test_a_standalone_gguf_resolves_to_its_directory(catalog, tmp_path):
    weights = tmp_path / "z-image-Q4_K_M.gguf"
    weights.write_bytes(b"")
    catalog.append(_info("z-image", weights, task = mas.IMAGE_TASK, model_format = "gguf"))

    pick = mas.resolve_local_media_model("z-image", task = mas.IMAGE_TASK)

    assert pick == mas.MediaModelPick("z-image", str(tmp_path), "z-image-Q4_K_M.gguf", "gguf")


def test_the_index_is_keyed_by_task(catalog, tmp_path):
    clip = tmp_path / "wan"
    clip.mkdir()
    (clip / "model_index.json").write_text("{}")
    (clip / "model_index.json").write_text("{}")
    catalog.append(_info("unsloth/Wan2.2", clip, task = mas.VIDEO_TASK))

    assert mas.resolve_local_media_model("unsloth/Wan2.2", task = mas.VIDEO_TASK) is not None
    # An image request must not be answered by a video model, or the load 400s after eviction.
    assert mas.resolve_local_media_model("unsloth/Wan2.2", task = mas.IMAGE_TASK) is None


def test_what_the_index_refuses_to_advertise(catalog, tmp_path):
    # A cancelled pull still lists and fails predictably; several checkpoints with no
    # model_index.json is a directory both load routes reject rather than choose between; and an
    # absolute path is what the ./models and LM Studio scanners report as the id, not something
    # an API caller should have to send.
    half = tmp_path / "half"
    half.mkdir()
    (half / "model_index.json").write_text("{}")
    partial = _info("org/half", half, task = mas.IMAGE_TASK)
    partial.partial = True
    catalog.append(partial)

    ambiguous = tmp_path / "two-checkpoints"
    ambiguous.mkdir()
    for name in ("a.safetensors", "b.safetensors"):
        (ambiguous / name).write_bytes(b"")
    catalog.append(_info("org/ambiguous", ambiguous, task = mas.IMAGE_TASK))

    local = tmp_path / "local-only"
    local.mkdir()
    (local / "model_index.json").write_text("{}")
    by_path = _info(str(local), local, task = mas.IMAGE_TASK, display_name = "Local Only")
    by_path.model_id = None
    catalog.append(by_path)

    for name in ("someone/else", "", "org/half", "org/ambiguous", str(local)):
        assert mas.resolve_local_media_model(name, task = mas.IMAGE_TASK) is None
    # The path-named model is still reachable by its label, which is a name a caller can send.
    assert mas.resolve_local_media_model("Local Only", task = mas.IMAGE_TASK) is not None
    assert mas.available_media_model_ids(mas.IMAGE_TASK) == ["Local Only"]


# ── the switch ──────────────────────────────────────────────────────


class _FakeMediaBackend:
    def __init__(
        self,
        repo_id = None,
        gguf_variant = None,
        model_kind = None,
        h3_task = None,
    ):
        self.repo_id = repo_id
        self.gguf_variant = gguf_variant
        self.model_kind = model_kind
        self.h3_task = h3_task
        self.loading: tuple[str, ...] = ()
        self.active = False
        self.phase = "ready" if repo_id else None
        # Bytes the planner reports as still to fetch; the guard refuses anything above 0.
        self.missing_bytes = 0

    def status(self):
        return {
            "loaded": self.repo_id is not None,
            "repo_id": self.repo_id,
            "base_repo": None,
            "gguf_variant": self.gguf_variant,
            "model_kind": self.model_kind,
            "h3_task": self.h3_task,
        }

    def loading_repo_ids(self):
        return self.loading

    def generate_progress(self):
        return {"active": self.active}

    def load_progress(self):
        return {"phase": self.phase}

    def download_plan(self, model_path, **kwargs):
        return {"total_bytes": self.missing_bytes}


@pytest.fixture
def backend(monkeypatch):
    fake = _FakeMediaBackend()
    monkeypatch.setattr(mas, "backend_for", lambda owner: fake)
    # The real planner resolves the engine router and reaches the Hub; this suite is offline.
    monkeypatch.setattr(locality, "planners_for", lambda owner, pick: [fake])
    return fake


@pytest.fixture
def loads(monkeypatch, backend):
    """Record every load the switch starts, and bring the fake up as the loader would."""
    started: list = []

    async def _start(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        started.append((owner, pick))
        backend.repo_id = pick.model_path
        # The real backends publish extract_quant_token, not the lister label.
        backend.gguf_variant = index.published_token(pick) or None
        backend.model_kind = pick.model_kind
        # A switch sends no h3_task, so the load comes up on the family default.
        backend.h3_task = None
        backend.phase = "ready"
        mk.note_load_origin(owner, pick.model_path, user_action = False)

    monkeypatch.setattr(mas, "_start_load", _start)
    return started


@pytest.fixture
def flux(catalog, tmp_path):
    """A local FLUX pipeline in the catalog: the stand-in target for most switch tests."""
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    return pipeline


@pytest.fixture
def cached_gguf(catalog, tmp_path):
    """A GGUF repo in the HF cache, whose snapshot entries are symlinks into blobs/."""
    repo, _snapshot = _hf_cache_repo(tmp_path, "city96/FLUX.1-dev-gguf", files = ["f-Q4_K_M.gguf"])
    catalog.append(
        _info(
            "city96/FLUX.1-dev-gguf",
            repo,
            task = mas.IMAGE_TASK,
            model_format = "gguf",
            source = "hf_cache",
        )
    )
    return repo


@pytest.fixture
def two_bpw(catalog, tmp_path):
    """Two builds of one quant, which the backend's published token cannot tell apart."""
    for bpw in ("3.53", "3.97"):
        (tmp_path / f"z-IQ4_XS-{bpw}bpw.gguf").write_bytes(b"")
        catalog.append(
            _info(
                f"z-IQ4_XS-{bpw}bpw",
                tmp_path / f"z-IQ4_XS-{bpw}bpw.gguf",
                task = mas.IMAGE_TASK,
                model_format = "gguf",
            )
        )
    return tmp_path


@pytest.fixture
def hidream(catalog, tmp_path):
    """A local HiDream pipeline, whose Llama encoder lives outside its own directory."""
    pipeline = tmp_path / "hidream"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("HiDream-ai/HiDream-I1-Dev", pipeline, task = mas.IMAGE_TASK))
    return pipeline


@pytest.fixture
def h3_modular(catalog, tmp_path):
    """A dense MiniMax-H3 directory, which carries modular_model_index.json instead."""
    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text("{}")
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))
    return modular


def _client(router, prefix):
    """A TestClient over one router, with the error handlers the real app installs."""
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = prefix)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _switch(
    name,
    *,
    owner = arb.DIFFUSION,
    openai_errors = True,
):
    return asyncio.run(
        mas.maybe_auto_switch_media_model(
            name,
            owner = owner,
            current_subject = "test-user",
            openai_errors = openai_errors,
        )
    )


def test_the_switch_is_inert_unless_it_is_on_and_named(catalog, enabled, loads, monkeypatch):
    _switch(None)
    _switch("   ")
    monkeypatch.setattr(settings, "get_media_auto_switch_enabled", lambda: False)
    # Not even an unknown name is refused: `model` keeps its old informational meaning.
    _switch("someone/else")
    assert loads == []


def test_an_unresolvable_name_is_refused_and_lists_what_is_downloaded(flux, enabled, loads):
    with pytest.raises(HTTPException) as excinfo:
        _switch("someone/else")

    assert excinfo.value.status_code == 404
    error = excinfo.value.detail["error"]
    assert error["code"] == "model_not_found"
    assert error["param"] == "model"
    assert "black-forest-labs/FLUX.1-dev" in error["message"]
    assert loads == []


def test_a_video_refusal_carries_a_plain_detail(catalog, enabled, loads):
    # /api/inference/video/generate is not an OpenAI surface, so it must not grow an envelope.
    with pytest.raises(HTTPException) as excinfo:
        _switch("someone/else", owner = arb.VIDEO, openai_errors = False)

    assert isinstance(excinfo.value.detail, str)


@pytest.mark.parametrize("resident", ["repo id", "path"])
def test_a_resident_model_is_not_reloaded(flux, enabled, backend, loads, resident):
    # A model loaded from the Images page reports its repo id while one this module loaded
    # reports the local path it was given, and either has to count as already serving.
    backend.repo_id = "black-forest-labs/FLUX.1-dev" if resident == "repo id" else str(flux)

    _switch("black-forest-labs/FLUX.1-dev")

    assert loads == []


def test_a_different_model_is_loaded_before_the_request_proceeds(flux, enabled, backend, loads):
    backend.repo_id = "Qwen/Qwen-Image"

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]
    # The load came from the API, so "only unload models loaded by the API" may free it.
    assert mk.loaded_by_user_action(arb.DIFFUSION) is False


def test_a_busy_backend_is_not_swapped_out_from_under_its_generation(
    flux, enabled, backend, loads, monkeypatch
):
    monkeypatch.setattr(mas, "_DRAIN_WAIT_S", 0.0)
    backend.repo_id = "Qwen/Qwen-Image"
    backend.active = True

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 409
    assert excinfo.value.headers["Retry-After"] == "15"
    assert loads == []


def test_a_load_still_running_at_the_deadline_asks_for_a_retry(flux, enabled, backend, monkeypatch):
    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.0)

    async def _start(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        backend.phase = "downloading"

    monkeypatch.setattr(mas, "_start_load", _start)

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 503
    assert excinfo.value.headers["Retry-After"] == "15"


def test_a_failed_load_surfaces_its_error(flux, enabled, backend, monkeypatch):
    async def _start(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        backend.phase = "error"
        backend.load_progress = lambda: {"phase": "error", "error": "not enough VRAM"}

    monkeypatch.setattr(mas, "_start_load", _start)

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    # A 503 naming the reason, not a bare 500 carrying the loader's exception.
    assert excinfo.value.status_code == 503
    assert "not enough VRAM" in excinfo.value.detail["error"]["message"]


# ── route wiring ────────────────────────────────────────────────────


def test_a_companion_base_repo_does_not_count_as_serving(
    catalog, enabled, tmp_path, backend, loads
):
    # A GGUF borrows the full pipeline as its text-encoder/VAE base; a request for that
    # pipeline must load it, not be answered by the GGUF that borrows it.
    pipeline = tmp_path / "flux-dev"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = "city96/FLUX.1-dev-gguf"

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_the_video_route_switches_before_it_touches_the_backend(monkeypatch):
    import core.inference.video as video_module
    from routes.video import router as video_router

    calls: list = []

    async def _switch_stub(
        requested_model,
        *,
        owner,
        current_subject,
        openai_errors,
        hf_token = None,
        before_switch = None,
    ):
        calls.append((requested_model, owner, openai_errors, hf_token))

    monkeypatch.setattr(mas, "maybe_auto_switch_media_model", _switch_stub)

    class _Backend:
        def begin_generate(self, **kwargs):
            return None

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Backend())

    resp = _client(video_router, "/api/inference").post(
        "/api/inference/video/generate",
        json = {"prompt": "a sloth", "model": "unsloth/Wan2.2"},
        headers = {"X-Unsloth-HF-Token": "hf_abc"},
    )

    assert resp.status_code == 200
    # Not the OpenAI envelope: this route is a Studio surface and its errors are plain details.
    assert calls == [("unsloth/Wan2.2", arb.VIDEO, False, "hf_abc")]


def test_a_sibling_loose_gguf_is_not_treated_as_already_serving(
    catalog, enabled, tmp_path, backend, loads
):
    # Loose .gguf files in one scan folder share that folder as their model_path, so the path
    # alone would report a sibling as serving and generate on the wrong weights.
    for quant in ("Q4_K_M", "Q8_0"):
        (tmp_path / f"z-image-{quant}.gguf").write_bytes(b"")
        catalog.append(
            _info(
                f"z-image-{quant}",
                tmp_path / f"z-image-{quant}.gguf",
                task = mas.IMAGE_TASK,
                model_format = "gguf",
            )
        )
    backend.repo_id = str(tmp_path)
    backend.gguf_variant = "Q4_K_M"
    backend.model_kind = "gguf"

    _switch("z-image-Q8_0")

    assert [pick.gguf_filename for _owner, pick in loads] == ["z-image-Q8_0.gguf"]
    # And the sibling that IS now loaded still short-circuits.
    loads.clear()
    _switch("z-image-Q8_0")
    assert loads == []


@pytest.mark.parametrize(
    ("plan", "fragment"),
    [
        # Sized companions the loader would fetch: the resolver indexes checkpoints only, and a
        # GGUF still loads its encoders and VAE from a base repo.
        ({"total_bytes": 4_300_000_000}, "4.3 GB"),
        # Both planners coerce an unknown sibling size to zero while keeping the entry, so bytes
        # alone would read a pending multi-GB fetch as nothing to do.
        ({"total_bytes": 0, "entries": [{"repo_id": "x", "files": ["a"], "bytes": 0}]}, ""),
        # Cached in full and still unloadable (a FLUX.2 GGUF on a different-size base). The
        # route's cheap validation misses it, so only the background loader would have found out.
        ({"total_bytes": 0, "entries": [], "incompatible_reason": "needs the 32B base"}, ""),
    ],
)
def test_a_pick_the_plan_cannot_clear_is_refused_rather_than_downloaded(
    cached_gguf, enabled, backend, loads, plan, fragment
):
    backend.download_plan = lambda model_path, **kw: plan

    with pytest.raises(HTTPException) as excinfo:
        _switch("city96/FLUX.1-dev-gguf")

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail["error"]["code"] == "model_not_downloaded"
    assert fragment in excinfo.value.detail["error"]["message"]
    assert loads == []


def test_a_request_queued_on_the_switch_lock_does_not_block_the_drain(
    flux, enabled, backend, loads, monkeypatch
):
    # Two concurrent requests for the same absent model are both counted by the middleware.
    # Counting the queued one as work to drain made each wait the other out and both 409.
    backend.repo_id = "Qwen/Qwen-Image"
    monkeypatch.setattr(mas, "_DRAIN_WAIT_S", 0.5)
    # Two tracked requests in flight, and both parked on the switch: only one holds the lock.
    monkeypatch.setattr(
        mk,
        "other_request_count",
        lambda owner, current_request_counted = False, count_pending = True: 1,
    )
    monkeypatch.setattr(backends, "waiter_count", lambda owner: 2)

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_the_drain_and_the_load_share_one_budget(flux, enabled, backend, monkeypatch):
    # Separate budgets added up past the ~100s tunnel window, so a slow switch lost the socket
    # instead of returning the retryable 503 the bounds exist to produce.
    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.6)
    monkeypatch.setattr(mas, "_DRAIN_WAIT_S", 30.0)
    busy_until = [True]

    def _busy(_backend):
        # Busy for the first poll only, so the drain eats part of the shared budget.
        was = busy_until[0]
        busy_until[0] = False
        return was

    monkeypatch.setattr(backends, "backend_busy", _busy)

    async def _start(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        backend.phase = "downloading"

    monkeypatch.setattr(mas, "_start_load", _start)

    began = time.monotonic()
    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 503
    # The load wait inherits what the drain left, so the whole switch stays inside the budget.
    assert time.monotonic() - began < 2.0


def test_a_load_that_lands_while_draining_is_not_repeated(flux, enabled, backend, loads):
    # A retry can acquire the switch lock while the earlier attempt's load is still running.
    # Draining waits that out, and without a recheck the retry tears down what just landed.
    backend.repo_id = "Qwen/Qwen-Image"
    backend.loading = ("black-forest-labs/FLUX.1-dev",)

    async def _drain_lands_the_model(_owner, _backend, _deadline, **kwargs):
        backend.loading = ()
        backend.repo_id = str(flux)
        return True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mas, "drain", _drain_lands_the_model)
        _switch("black-forest-labs/FLUX.1-dev")

    assert loads == []


def test_a_replacement_load_is_not_reported_as_the_requested_model(
    flux, enabled, backend, monkeypatch
):
    # A user load accepted between two polls supersedes ours. Returning success on "something
    # is resident" would generate on the replacement while naming the requested model.

    async def _start(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        backend.repo_id = "Qwen/Qwen-Image"
        backend.phase = "ready"

    monkeypatch.setattr(mas, "_start_load", _start)

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 503
    assert "replaced" in excinfo.value.detail["error"]["message"]


def test_the_download_plan_asks_the_engine_that_will_load_the_pick(cached_gguf, monkeypatch):
    # The resident engine can be native sd.cpp while the target loads through diffusers; its
    # planner refuses the pick, and that refusal would read as nothing missing.
    pick = mas.resolve_local_media_model("city96/FLUX.1-dev-gguf", task = mas.IMAGE_TASK)
    asked: list = []

    class _Planner:
        def download_plan(self, model_path, **kwargs):
            asked.append(model_path)
            return {"total_bytes": 7}

    monkeypatch.setattr(locality, "planners_for", lambda owner, p: [_Planner()])

    assert mas.missing_download_bytes(arb.DIFFUSION, pick) == 7
    assert asked == [pick.model_path]


def test_two_bpw_builds_of_one_quant_are_not_confused(two_bpw, enabled, backend, loads):
    # The backend's published token collapses IQ4_XS-3.53bpw and -3.97bpw to IQ4_XS, so a
    # resident sibling can never be assumed to be ours and the skip stays refused. The load
    # this request starts is its own, though, and rejecting that would 503 what just landed.
    backend.repo_id = str(two_bpw)
    backend.gguf_variant = "IQ4_XS"
    backend.model_kind = "gguf"

    _switch("z-IQ4_XS-3.97bpw")
    assert [pick.gguf_filename for _owner, pick in loads] == ["z-IQ4_XS-3.97bpw.gguf"]

    loads.clear()
    _switch("z-IQ4_XS-3.97bpw")
    assert [pick.gguf_filename for _owner, pick in loads] == ["z-IQ4_XS-3.97bpw.gguf"]


def test_an_unverifiable_download_plan_refuses_rather_than_loading(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # The video planner reports zero bytes when its own metadata call failed, because its normal
    # caller falls back to an inline pull. Zero there is "unknown", not "nothing to fetch".
    repo, _snapshot = _hf_cache_repo(tmp_path, "unsloth/Wan2.2-GGUF", files = ["wan-Q4_K_M.gguf"])
    catalog.append(
        _info(
            "unsloth/Wan2.2-GGUF",
            repo,
            task = mas.VIDEO_TASK,
            model_format = "gguf",
            source = "hf_cache",
        )
    )
    monkeypatch.setattr(
        backend, "download_plan", lambda model_path, **kw: {"total_bytes": 0, "plan_failed": True}
    )

    with pytest.raises(HTTPException) as excinfo:
        _switch("unsloth/Wan2.2-GGUF", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409
    assert "Could not verify" in excinfo.value.detail
    assert loads == []


def test_an_exact_resident_match_does_not_need_the_index(catalog, enabled, backend, loads):
    # A scan that failed caches an empty index for a few seconds; the model that is loaded and
    # named exactly must stay servable through that window.
    backend.repo_id = "black-forest-labs/FLUX.1-dev"

    _switch("black-forest-labs/FLUX.1-dev")

    assert loads == []


def test_an_alias_two_models_share_resolves_to_neither(catalog, tmp_path):
    # Cached repos advertise their final component, so org-a/model and org-b/model both offer
    # "model". Binding whichever the scan reached first would load arbitrary weights.
    for org in ("org-a", "org-b"):
        pipeline = tmp_path / org
        pipeline.mkdir()
        (pipeline / "model_index.json").write_text("{}")
        row = _info(f"{org}/model", pipeline, task = mas.IMAGE_TASK, display_name = "model")
        catalog.append(row)

    assert mas.resolve_local_media_model("model", task = mas.IMAGE_TASK) is None
    # The unambiguous full ids still resolve.
    assert mas.resolve_local_media_model("org-a/model", task = mas.IMAGE_TASK) is not None
    assert mas.resolve_local_media_model("org-b/model", task = mas.IMAGE_TASK) is not None
    assert mas.available_media_model_ids(mas.IMAGE_TASK) == ["org-a/model", "org-b/model"]


def test_the_exact_resident_shortcut_never_answers_for_a_gguf(
    catalog, enabled, tmp_path, backend, loads
):
    # A bare repo id means the preferred quant, which the shortcut cannot check, so a resident
    # non-preferred quant would have served the request.
    repo = tmp_path / "flux-gguf"
    repo.mkdir()
    (repo / "model_index.json").write_text("{}")
    for quant in ("Q4_K_M", "Q8_0"):
        (repo / f"flux1-dev-{quant}.gguf").write_bytes(b"")
    catalog.append(_info("city96/FLUX.1-dev-gguf", repo, task = mas.IMAGE_TASK))
    backend.repo_id = "city96/FLUX.1-dev-gguf"
    backend.gguf_variant = "Q8_0"
    backend.model_kind = "gguf"

    _switch("city96/FLUX.1-dev-gguf")

    preferred = preferred_quant(["Q4_K_M", "Q8_0"])
    if preferred != "Q8_0":
        assert [p.gguf_filename for _o, p in loads] == [f"flux1-dev-{preferred}.gguf"]


def test_a_bare_single_file_directory_is_planned_as_the_load_reads_it(
    catalog, tmp_path, monkeypatch
):
    # The load routes reinterpret such a directory as single_file and then resolve that family's
    # companions; planning it as a local pipeline reports nothing to fetch.
    checkpoint = tmp_path / "solo"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"")
    catalog.append(_info("some/solo", checkpoint, task = mas.IMAGE_TASK))
    pick = mas.resolve_local_media_model("some/solo", task = mas.IMAGE_TASK)
    seen: list = []

    class _Planner:
        def download_plan(self, model_path, **kwargs):
            seen.append((kwargs.get("gguf_filename"), kwargs.get("model_kind")))
            return {"total_bytes": 0, "entries": []}

    monkeypatch.setattr(locality, "planners_for", lambda owner, p: [_Planner()])
    monkeypatch.setattr(locality, "plan_gpu_ordinal", lambda: None)

    assert mas.missing_download_bytes(arb.DIFFUSION, pick) == 0
    assert seen == [("model.safetensors", "single_file")]


def test_a_local_pipeline_is_complete_without_asking_the_hub(flux, enabled, backend, loads):
    # The planner asks HfApi about an absolute path and fails, which now reads as unverifiable.
    # A directory on disk is what from_pretrained loads, so it needs no plan at all.

    def _explode(model_path, **kwargs):
        raise AssertionError("a local flux must not be planned against the Hub")

    backend.download_plan = _explode

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_paths_differing_only_in_case_are_different_models(
    catalog, enabled, tmp_path, backend, loads
):
    # Case-sensitive filesystems allow /models/Foo and /models/foo side by side. Folding them
    # would report one as already serving the other, and lowercasing the path in the ambiguity
    # scan merged their builds into one quant token so neither could ever be reused.
    for name in ("Foo", "foo"):
        directory = tmp_path / name
        if not directory.exists():
            directory.mkdir()
        (directory / f"{name}-Q4_K_M.gguf").write_bytes(b"")
        catalog.append(_info(f"org/{name}", directory, task = mas.IMAGE_TASK, model_format = "gguf"))
    upper = mas.resolve_local_media_model("org/Foo", task = mas.IMAGE_TASK)
    lower = mas.resolve_local_media_model("org/foo", task = mas.IMAGE_TASK)
    if upper is None or lower is None or upper.model_path == lower.model_path:
        pytest.skip("this filesystem folds case, so the two models cannot coexist")
    assert not upper.ambiguous and not lower.ambiguous
    backend.repo_id = lower.model_path
    backend.gguf_variant = "Q4_K_M"
    backend.model_kind = "gguf"

    _switch("org/Foo")

    assert [pick.model_path for _owner, pick in loads] == [upper.model_path]


def test_a_native_prediction_also_verifies_the_diffusers_fallback(cached_gguf, monkeypatch):
    # predict_engine calls sd.cpp available whenever its install is allowed, while activation
    # falls back to diffusers when that install produces nothing runnable.
    pick = mas.resolve_local_media_model("city96/FLUX.1-dev-gguf", task = mas.IMAGE_TASK)

    class _Planner:
        def __init__(self, missing):
            self.missing = missing

        def download_plan(self, model_path, **kwargs):
            return {"total_bytes": self.missing, "entries": []}

    # The predicted engine sees nothing missing; the fallback's companion set is incomplete.
    monkeypatch.setattr(locality, "planners_for", lambda owner, p: [_Planner(0), _Planner(9_000)])

    assert mas.missing_download_bytes(arb.DIFFUSION, pick) == 9_000


def test_a_cache_tree_inside_a_scan_folder_still_loads_by_repo_id(catalog, tmp_path):
    # collect_local_models rewrites such a row's source to "custom" while the snapshot entries
    # stay symlinks into blobs/, which the loader's containment check refuses.
    repo, _snapshot = _hf_cache_repo(
        tmp_path, "unsloth/Z-Image-Turbo-GGUF", files = ["z-Q4_K_S.gguf"]
    )
    catalog.append(
        _info(
            "unsloth/Z-Image-Turbo-GGUF",
            repo,
            task = mas.IMAGE_TASK,
            model_format = "gguf",
            source = "custom",
        )
    )

    pick = mas.resolve_local_media_model("unsloth/Z-Image-Turbo-GGUF", task = mas.IMAGE_TASK)

    assert pick.model_path == "unsloth/Z-Image-Turbo-GGUF"


def test_the_native_fallback_is_only_verified_when_a_binary_must_be_installed(monkeypatch):
    # With a runnable sd.cpp binary the load stays native, so demanding the diffusers shards
    # would refuse a model the selected engine can serve.
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    pick = mas.MediaModelPick("x/y", "x/y", "y-Q4_K_M.gguf", "gguf")
    monkeypatch.setattr(mas, "backend_for", lambda owner: object())
    router = __import__("core.inference.diffusion_engine_router", fromlist = ["x"])
    families = __import__("core.inference.diffusion_families", fromlist = ["x"])
    monkeypatch.setattr(families, "detect_family_for_pick", lambda *a, **k: object())
    monkeypatch.setattr(router, "predict_engine", lambda fam, model_kind = None: ENGINE_SD_CPP)
    monkeypatch.setattr(router, "engine_for", lambda name: name)

    monkeypatch.setattr(router, "native_binary_installed", lambda: True)
    assert locality.planners_for(arb.DIFFUSION, pick) == [ENGINE_SD_CPP]

    monkeypatch.setattr(router, "native_binary_installed", lambda: False)
    assert len(locality.planners_for(arb.DIFFUSION, pick)) == 2


def test_load_setup_that_stalls_returns_inside_the_budget(flux, enabled, backend, monkeypatch):
    # Preflight and a first-run sd.cpp install both run before begin_load registers, while the
    # admission gate is held, so an unbounded await blocks the backend as well as the caller.
    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.3)

    async def _stalls(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        await asyncio.sleep(5)

    monkeypatch.setattr(mas, "_start_load", _stalls)

    began = time.monotonic()
    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 503
    assert time.monotonic() - began < 3.0


def test_a_modular_pipeline_index_is_discoverable(h3_modular):
    # A dense MiniMax-H3 directory carries modular_model_index.json, which the video loader
    # opens; rejecting it here 404'd every named request for a fully downloaded model.
    assert mas.resolve_local_media_model("MiniMaxAI/MiniMax-H3", task = mas.VIDEO_TASK) is not None


@pytest.mark.parametrize("resident", ["repo id", "path"])
def test_a_resident_h3_reference_partition_does_not_answer_a_plain_request(
    h3_modular, enabled, backend, loads, resident
):
    # An auto-load of this name takes the default keyframe denoiser, so a resident ref2va is a
    # different build and serving it accepts a generation that then fails for missing references.
    # The pre-index shortcut returns before resident_is_pick runs, so it needs the same test.
    backend.repo_id = "MiniMaxAI/MiniMax-H3" if resident == "repo id" else str(h3_modular)
    backend.h3_task = "ref2va"

    _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert [pick.model_id for _owner, pick in loads] == ["MiniMaxAI/MiniMax-H3"]


def test_a_bare_id_takes_a_root_variant_over_a_qualified_one(catalog, tmp_path):
    # A plain local load resolves non-recursively and always takes the root, so a bare id must
    # not select a distilled subdirectory build the picker and chat resolver never would.
    repo = tmp_path / "flux-gguf"
    (repo / "distilled").mkdir(parents = True)
    (repo / "flux1-Q4_K_M.gguf").write_bytes(b"")
    (repo / "distilled" / "flux1-Q8_0.gguf").write_bytes(b"")
    catalog.append(_info("city96/FLUX.1-dev-gguf", repo, task = mas.IMAGE_TASK))

    bare = mas.resolve_local_media_model("city96/FLUX.1-dev-gguf", task = mas.IMAGE_TASK)

    assert bare.gguf_filename == "flux1-Q4_K_M.gguf"


def test_setup_keeps_the_gate_and_lock_after_the_caller_gives_up(
    flux, enabled, backend, monkeypatch
):
    # Shielding alone let the caller unwind both contexts while setup was still before
    # begin_load, so a newly admitted generation could be cut short by the orphaned switch.
    import core.inference.media_keepwarm as keepwarm

    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_setup(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        started.set()
        await release.wait()

    async def _drive():
        # One warm pass over the same path, at the production budget, with a setup that
        # returns instead of blocking. Reaching _start_load the FIRST time measured 1.98s
        # here, and media_auto_switch counts that against the switch budget on purpose
        # ("the cold scan is part of the wait the caller experiences"). At the production
        # 90s that is nothing. This test shrinks the budget to 0.3s because a large one
        # has to be waited out -- at 30s the test took 31.8s -- and a cold path then blows
        # 0.3s before setup is entered: the 503 comes from the scan rather than from the
        # block under test, the status-code assertion still passes, and the failure lands
        # further down on `started.is_set()` explaining nothing. That is how it failed on
        # the 3.13 leg while 3.10 passed the same commit. Warming is what makes the small
        # budget measure the block and only the block.
        warmed = asyncio.Event()

        async def _quick_setup(
            owner,
            pick,
            current_subject,
            hf_token = None,
        ):
            warmed.set()

        monkeypatch.setattr(mas, "_start_load", _quick_setup)
        with contextlib.suppress(HTTPException):
            await mas.maybe_auto_switch_media_model(
                "black-forest-labs/FLUX.1-dev",
                owner = arb.DIFFUSION,
                current_subject = "test-user",
                openai_errors = True,
            )
        assert warmed.is_set(), (
            "the warm pass never reached _start_load at the production budget, so the "
            "timed run below cannot be measuring what this test is about"
        )

        monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.3)
        monkeypatch.setattr(mas, "_start_load", _slow_setup)
        task = asyncio.ensure_future(
            mas.maybe_auto_switch_media_model(
                "black-forest-labs/FLUX.1-dev",
                owner = arb.DIFFUSION,
                current_subject = "test-user",
                openai_errors = True,
            )
        )
        with pytest.raises(HTTPException) as excinfo:
            await task
        assert excinfo.value.status_code == 503
        # The caller gave up, but setup owns the gate until it reaches registration.
        assert started.is_set()
        assert keepwarm._TRACKERS[arb.DIFFUSION].gate.locked()
        assert mas.switch_lock(arb.DIFFUSION).locked()
        release.set()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if not keepwarm._TRACKERS[arb.DIFFUSION].gate.locked():
                break
        assert not keepwarm._TRACKERS[arb.DIFFUSION].gate.locked()
        assert not mas.switch_lock(arb.DIFFUSION).locked()

    asyncio.run(_drive())


def test_a_request_parked_on_the_held_gate_does_not_abort_the_switch(
    flux, enabled, backend, loads, monkeypatch
):
    # A newcomer arriving while the gated task owns the gate is counted pending and then blocks
    # on that gate, so counting it aborted an otherwise idle switch.
    backend.repo_id = "Qwen/Qwen-Image"
    tracker = mk._TRACKERS[arb.DIFFUSION]
    monkeypatch.setattr(tracker, "_pending", 1)

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_a_ref2va_checkpoint_expects_its_own_partition(catalog, enabled, tmp_path, backend, loads):
    # The native backend publishes ref2va for a minimax_h3_ref2va denoiser, so assuming the
    # keyframe default rejected the checkpoint that had just loaded.
    (tmp_path / "minimax_h3_ref2va-Q4_K_M.gguf").write_bytes(b"")
    catalog.append(
        _info(
            "minimax_h3_ref2va-Q4_K_M",
            tmp_path / "minimax_h3_ref2va-Q4_K_M.gguf",
            task = mas.VIDEO_TASK,
            model_format = "gguf",
        )
    )
    backend.repo_id = str(tmp_path)
    backend.gguf_variant = "Q4_K_M"
    backend.model_kind = "gguf"
    backend.h3_task = "ref2va"

    _switch("minimax_h3_ref2va-Q4_K_M", owner = arb.VIDEO, openai_errors = False)

    # Already serving: the resident partition is the one this checkpoint brings up.
    assert loads == []


def test_a_local_video_pipeline_is_still_planned(h3_modular, enabled, backend, loads):
    # A local MiniMax-H3 modular pipeline substitutes a hosted quantized conditioner during
    # assembly, so the image shortcut's "on disk means complete" does not hold for video.
    backend.missing_bytes = 27_000_000_000

    with pytest.raises(HTTPException) as excinfo:
        _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409
    assert loads == []


def test_every_backend_call_the_video_route_makes_exists_on_the_backend():
    """``backend.<name>`` in routes/video.py must name something ``VideoBackend`` has.

    ``get_video_backend()`` is untyped at the call site, so a method renamed on the class or
    misspelled in the route is no syntax error, no lint finding, and invisible to any double
    stubbing the old name. It is an AttributeError on a real load, and the route's own tests
    mock the backend out. Read off the parse tree, not the text.
    """
    import ast
    import inspect

    import routes.video as video_route
    from core.inference.video import VideoBackend

    tree = ast.parse(inspect.getsource(video_route))
    asked = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "backend"
        and isinstance(node.ctx, ast.Load)
    }
    assert asked, "no backend calls found -- the route was restructured, so this guard is blind"
    missing = sorted(name for name in asked if not hasattr(VideoBackend, name))
    assert not missing, f"routes/video.py calls backend methods that do not exist: {missing}"


def test_the_video_load_route_records_provenance_without_raising(monkeypatch):
    # The provenance call is made positionally from both load routes, so a signature that drifts
    # from them 500s every load after the background work has already been accepted.
    import core.inference.video as video_module
    from routes.video import router as video_router

    backend = _video_load_backend(
        validate_load_request = lambda *a, **k: _a_real_video_family(),
        begin_load = lambda *a, **k: {"loaded": False, "repo_id": None},
    )
    monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
    monkeypatch.setattr(video_module, "resolve_video_model_kind", lambda *a, **k: "gguf")
    monkeypatch.setattr(video_module, "assert_video_precision_available", lambda *a, **k: None)
    monkeypatch.setattr("routes.video._guard_video_load_against_training", lambda: None)
    monkeypatch.setattr("routes.video._selected_gpu_ordinal", _async_none)
    import core.inference.diffusion_device as device_module

    monkeypatch.setattr(
        device_module,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(device = "cpu"),
    )

    resp = _client(video_router, "/api/inference").post(
        "/api/inference/video/load",
        json = {
            "model_path": "unsloth/Wan2.2-GGUF",
            "gguf_filename": "wan-Q4_K_M.gguf",
            "model_kind": "gguf",
        },
    )

    assert resp.status_code == 200, resp.text
    assert mk.loaded_by_user_action(arb.VIDEO, "unsloth/Wan2.2-GGUF", "Q4_K_M") is True


async def _async_none(*args, **kwargs):
    return None


async def _lock_is_held(owner):
    return mas.switch_lock(owner).locked()


@pytest.mark.parametrize("busy", ["other media backend", "chat"])
def test_a_switch_waits_for_work_the_gpu_handoff_would_cancel(
    flux, enabled, backend, loads, monkeypatch, takes_the_gpu, busy
):
    # The load takes the GPU through the arbiter, whose cross-owner handoff unloads whoever
    # holds it, cancelling a video generation or a streaming completion this request never met.
    monkeypatch.setattr(mas, "_DRAIN_WAIT_S", 0.3)
    if busy == "chat":
        monkeypatch.setattr(backends, "chat_busy", lambda *a, **k: True)
    else:
        monkeypatch.setattr(backends, "other_backend_busy", lambda owner: True)

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_local_hidream_is_refused_while_its_encoder_is_incomplete(
    hidream, enabled, backend, loads, monkeypatch
):
    # HiDream loads a separate ~16 GB encoder repo, so a directory on disk is no evidence that
    # nothing will be downloaded, and one linked shard is not the repository from_pretrained opens.
    monkeypatch.setattr(locality, "encoder_repo_complete", lambda repo_id: False)

    with pytest.raises(HTTPException) as excinfo:
        _switch("HiDream-ai/HiDream-I1-Dev")

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_local_hidream_is_accepted_once_its_encoder_is_cached(
    hidream, enabled, backend, loads, monkeypatch
):
    # The planner cannot be handed an absolute pipeline path, so the dependency is checked
    # against the cache directly rather than by refusing the model outright.
    monkeypatch.setattr(locality, "encoder_repo_complete", lambda repo_id: True)

    def _explode(model_path, **kwargs):
        raise AssertionError("a local pipeline must not be planned against the Hub")

    backend.download_plan = _explode

    _switch("HiDream-ai/HiDream-I1-Dev")

    assert [pick.model_id for _owner, pick in loads] == ["HiDream-ai/HiDream-I1-Dev"]


def test_a_cpu_load_does_not_wait_on_chat_or_the_other_backend(
    flux, enabled, backend, loads, monkeypatch
):
    # A CPU diffusion device releases ownership instead of acquiring it, so such a switch
    # evicts nobody and owes no cross-owner wait.
    # both bindings: the drain sizes its wait with it, the switch decides on the gpu lock with it
    monkeypatch.setattr(backends, "load_takes_the_gpu", lambda: False)
    monkeypatch.setattr(mas, "load_takes_the_gpu", lambda: False)
    monkeypatch.setattr(backends, "chat_busy", lambda *a, **k: True)
    monkeypatch.setattr(backends, "other_backend_busy", lambda owner: True)
    # And a real tracked request on the other backend, which the count must also ignore.
    monkeypatch.setattr(mk._TRACKERS[arb.VIDEO], "_inflight", 1)

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_a_stalled_gate_does_not_pin_the_switch_lock(
    flux, enabled, backend, monkeypatch, takes_the_gpu
):
    # A gate held elsewhere must not keep the setup task, and with it the switch lock, alive
    # past the budget: acquisition happens before the non-cancellable phase.
    import core.inference.media_keepwarm as keepwarm

    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.4)
    keepwarm._TRACKERS[arb.VIDEO].gate.acquire()
    try:
        began = time.monotonic()
        with pytest.raises(HTTPException) as excinfo:
            _switch("black-forest-labs/FLUX.1-dev")
        assert excinfo.value.status_code == 503
        assert time.monotonic() - began < 3.0
        # The lock is per running loop, so its release is asserted inside one.
        assert not asyncio.run(_lock_is_held(arb.DIFFUSION))
    finally:
        keepwarm._TRACKERS[arb.VIDEO].gate.release()


def test_two_switches_on_different_backends_do_not_refuse_each_other(
    flux, enabled, backend, loads, monkeypatch
):
    # Each switcher is counted by the middleware on its own backend, so without discounting
    # them both saw the other as cross-owner work and both returned busy.
    backend.repo_id = "Qwen/Qwen-Image"
    monkeypatch.setattr(mas, "_DRAIN_WAIT_S", 0.5)
    # A video switch is in flight and its request is tracked on the video backend.
    monkeypatch.setattr(mk._TRACKERS[arb.VIDEO], "_inflight", 1)

    with mas.note_switcher(arb.VIDEO):
        _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_the_ltx23_extras_check_names_the_exact_companions(tmp_path, monkeypatch):
    # The extras repo also holds checkpoints, so any-weight-file evidence proves nothing about
    # the three variant-specific companions the assembly reads.
    import core.inference.diffusion_families as families
    import core.inference.video_ltx2 as ltx2
    import core.inference.video_families as video_families

    checkpoint = tmp_path / "generic-Q4_K_M.gguf"
    checkpoint.write_bytes(b"")
    pick = mas.MediaModelPick("local/ltx", str(tmp_path), checkpoint.name, "gguf")
    monkeypatch.setattr(
        video_families, "detect_video_family", lambda *a, **k: types.SimpleNamespace(name = "ltx-2")
    )
    monkeypatch.setattr(ltx2, "is_ltx23_checkpoint", lambda path: True)
    monkeypatch.setattr(ltx2, "ltx23_extras_files", lambda path: ("a.safetensors", "b.safetensors"))
    asked: list = []

    def _holds(repo_id, files):
        asked.append(tuple(files))
        return False

    monkeypatch.setattr(families, "cache_holds_files", _holds)

    assert locality.hidden_ltx23_extras(arb.VIDEO, pick) is True
    assert asked == [("a.safetensors", "b.safetensors")]


def test_the_chat_probe_counts_a_parked_switcher_once(monkeypatch):
    # A waiter is marked inside its own switch, so counting both discounted it twice and an
    # active chat stream read as idle, which the final gated drain no longer re-checks.
    import core.inference.llama_keepwarm as chat
    monkeypatch.setattr(chat, "other_inference_request_count", lambda **kw: 2)
    with mas.note_switcher(arb.DIFFUSION), mas.note_switcher(arb.VIDEO):
        with mas.note_waiter(arb.VIDEO):
            assert backends.chat_busy() is True


def test_a_sharded_encoder_without_its_index_is_incomplete(monkeypatch):
    # One shard and no index is an interrupted pull of a repo that is always sharded, so
    # from_pretrained would fetch the index and the rest.
    import core.inference.diffusion_families as families

    monkeypatch.setattr(families, "_upstream_is_cached", lambda *a, **k: True)
    monkeypatch.setattr(locality, "_cached_snapshot_file", lambda repo, name: None)

    assert locality.encoder_repo_complete("unsloth/Meta-Llama-3.1-8B-Instruct") is False
    # An unsharded repo keeps the single-file reading.
    assert locality.encoder_repo_complete("org/unsharded-encoder") is True


def test_an_edit_only_single_file_directory_is_refused(catalog, enabled, tmp_path, backend, loads):
    # The catalog identifies the family from the checkpoint name, and the load route
    # reinterprets the directory as single_file, so the guard has to see that filename too.
    directory = tmp_path / "mystuff"
    directory.mkdir()
    (directory / "qwen_image_edit_2509_fp8.safetensors").write_bytes(b"")
    catalog.append(_info("local/edit", directory, task = mas.IMAGE_TASK))
    backend.repo_id = "Qwen/Qwen-Image"

    with pytest.raises(HTTPException) as excinfo:
        _switch("local/edit")

    assert excinfo.value.status_code == 400
    assert loads == []


def test_a_model_unloaded_during_resolution_is_not_reported_as_resident(
    flux, enabled, backend, loads, monkeypatch
):
    # The pre-resolution snapshot can be a whole budget old, so an idle unload landing during
    # the index build would otherwise leave the route 503ing on nothing.
    # Resident under its path, so the pre-index shortcut misses and resolution decides.
    backend.repo_id = str(flux)
    original = mas.resolve_local_media_model

    def _unload_midway(name, *, task):
        backend.repo_id = None
        return original(name, task = task)

    monkeypatch.setattr(mas, "resolve_local_media_model", _unload_midway)

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_the_images_route_switches_before_it_checks_what_is_loaded(monkeypatch):
    import core.inference.diffusion as diffusion_module
    import core.inference.image_gallery as gallery_module
    from routes.inference import router

    calls: list = []

    async def _switch_stub(
        requested_model,
        *,
        owner,
        current_subject,
        openai_errors,
        hf_token = None,
        before_switch = None,
    ):
        calls.append((requested_model, owner, openai_errors, hf_token))

    monkeypatch.setattr(mas, "maybe_auto_switch_media_model", _switch_stub)

    class _Backend:
        is_loaded = True

        def status(self):
            return {"loaded": True, "repo_id": "unsloth/Z-Image-Turbo-GGUF", "base_repo": None}

        def generate(self, **kwargs):
            return {"images": [object()], "seed": 1, "repo_id": "unsloth/Z-Image-Turbo-GGUF"}

    backend = _Backend()
    monkeypatch.setattr(diffusion_module, "get_diffusion_backend", lambda: backend)
    monkeypatch.setattr(
        gallery_module,
        "save",
        lambda image, meta: {
            **meta,
            "id": "img0",
            "url": "/api/inference/images/gallery/img0/file",
        },
    )

    resp = _client(router, "/v1").post(
        "/v1/images/generations",
        json = {"prompt": "p", "size": "256x256", "model": "black-forest-labs/FLUX.1-dev"},
        headers = {"X-Unsloth-HF-Token": "hf_abc"},
    )

    assert resp.status_code == 200
    assert calls == [("black-forest-labs/FLUX.1-dev", arb.DIFFUSION, True, "hf_abc")]


def test_a_lone_unlabelled_gguf_is_not_reloaded_on_every_request(
    catalog, enabled, tmp_path, backend, loads
):
    # An unlabelled file publishes an empty quant token and so does the backend, so the two
    # match; marking the only build under its path ambiguous reloaded it on every request.
    checkpoint = tmp_path / "plain.gguf"
    checkpoint.write_bytes(b"")
    catalog.append(_info("local/plain", checkpoint, task = mas.IMAGE_TASK, model_format = "gguf"))

    _switch("local/plain")
    _switch("local/plain")

    assert [pick.model_id for _owner, pick in loads] == ["local/plain"]


def test_a_setup_refusal_after_the_caller_gives_up_is_not_reported_as_unretrieved(
    flux, enabled, backend, monkeypatch, caplog
):
    # Setup keeps running after the budget expires, and it refuses on ordinary paths, so its
    # exception has to be consumed or the loop reports a traceback for a handled 409.
    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.3)
    release = asyncio.Event()

    async def _refuses_late(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        await release.wait()
        raise HTTPException(status_code = 409, detail = "busy")

    monkeypatch.setattr(mas, "_start_load", _refuses_late)

    async def _drive():
        with pytest.raises(HTTPException) as excinfo:
            await mas.maybe_auto_switch_media_model(
                "black-forest-labs/FLUX.1-dev",
                owner = arb.DIFFUSION,
                current_subject = "test-user",
                openai_errors = True,
            )
        assert excinfo.value.status_code == 503
        release.set()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if not mas.switch_lock(arb.DIFFUSION).locked():
                break

    with caplog.at_level(logging.ERROR, logger = "asyncio"):
        asyncio.run(_drive())
        gc.collect()

    assert "never retrieved" not in caplog.text


def test_a_pre_switch_refusal_evicts_nothing(flux, enabled, backend, loads):
    # The caller's last say on the resolved pick runs while the resident model is still up, so
    # a request the target could never serve costs no eviction and no load.
    seen: list = []

    def _refuse(pick):
        seen.append(pick.model_id)
        raise HTTPException(status_code = 422, detail = "no")

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(
            mas.maybe_auto_switch_media_model(
                "black-forest-labs/FLUX.1-dev",
                owner = arb.DIFFUSION,
                current_subject = "test-user",
                openai_errors = True,
                before_switch = _refuse,
            )
        )

    assert excinfo.value.status_code == 422
    assert seen == ["black-forest-labs/FLUX.1-dev"]
    assert loads == []


@pytest.mark.parametrize(
    ("pick", "request_body", "status", "fragment"),
    [
        # 512x512 is not one of the A14B presets.
        (
            ("unsloth/Wan2.2-T2V-A14B-GGUF", None),
            {"width": 512, "height": 512},
            422,
            "not a supported resolution",
        ),
        # Ref2VA conditions on references, so a first frame is unservable on it.
        (
            ("unsloth/MiniMax-H3-GGUF", "ref2va/minimax_h3_ref2va-Q4_K_M.gguf"),
            {"first_frame": "data:image/png;base64,AAAA"},
            400,
            "Ref2VA partition",
        ),
    ],
)
def test_the_video_route_refuses_what_the_target_cannot_serve(
    monkeypatch, pick, request_body, status, fragment
):
    # begin_generate would say the same thing, but only after the switch had evicted the
    # resident model and spent minutes loading a target that was never going to serve this.
    import core.inference.video as video_module
    from routes.video import router as video_router

    generated: list = []
    path, filename = pick

    async def _switch_stub(
        requested_model,
        *,
        before_switch = None,
        **kwargs,
    ):
        before_switch(index.MediaModelPick(requested_model, path, filename, "gguf"))

    monkeypatch.setattr(mas, "maybe_auto_switch_media_model", _switch_stub)

    class _Backend:
        def begin_generate(self, **kwargs):
            generated.append(kwargs)

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Backend())

    resp = _client(video_router, "/api/inference").post(
        "/api/inference/video/generate",
        json = {"prompt": "a sloth", "model": path, **request_body},
    )

    assert resp.status_code == status
    assert fragment in resp.json()["detail"]
    assert generated == []


def test_an_encoder_missing_its_config_is_incomplete(monkeypatch):
    # Every shard on disk and no config.json still reaches the Hub: the pipeline builds the
    # encoder with from_pretrained on the whole repository, not on the weights alone.
    import core.inference.diffusion_families as families

    monkeypatch.setattr(families, "_upstream_is_cached", lambda *a, **k: True)
    monkeypatch.setattr(locality, "_cached_snapshot_file", lambda repo, name: None)
    held: list = []
    monkeypatch.setattr(families, "cache_holds_files", lambda repo, names: held == names)

    held = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    assert locality.encoder_repo_complete("org/unsharded-encoder") is True
    held = ["config.json"]
    assert locality.encoder_repo_complete("org/unsharded-encoder") is False


def test_a_generically_named_ltx23_checkpoint_is_still_header_checked(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # Neither the repo nor the filename carries a family token, so only the loader's
    # general.architecture fallback resolves LTX and reaches the 2.3 extras check.
    import core.inference.video as video_module
    import core.inference.video_ltx2 as ltx2
    from core.inference.video_families import detect_video_family

    checkpoint = tmp_path / "model-Q4_K_M.gguf"
    checkpoint.write_bytes(b"")
    catalog.append(_info("local/generic", checkpoint, task = mas.VIDEO_TASK, model_format = "gguf"))
    monkeypatch.setattr(video_module, "_picked_gguf_arch", lambda repo_id, gguf_filename: "ltxv")
    monkeypatch.setattr(ltx2, "is_ltx23_checkpoint", lambda path: True)
    monkeypatch.setattr(ltx2, "ltx23_extras_files", lambda path: ("vae.safetensors",))
    monkeypatch.setattr(
        "core.inference.diffusion_families.cache_holds_files", lambda repo, names: False
    )
    # The family token really is absent from both names, so only the header can find it.
    assert detect_video_family("local/generic") is None

    with pytest.raises(HTTPException) as excinfo:
        _switch("local/generic", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_pipeline_is_planned_by_its_index_not_its_directory_name(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # detect_family_for_pick reads model_index.json ahead of any guess made from a name, so
    # asking about the catalog id first answered FLUX for a HiDream pipeline in a flux.1
    # directory while the loader answered HiDream and fetched its 16 GB encoder.
    pipeline = tmp_path / "flux.1"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text('{"_class_name": "HiDreamImagePipeline"}')
    catalog.append(_info("local/flux.1", pipeline, task = mas.IMAGE_TASK))
    monkeypatch.setattr(locality, "encoder_repo_complete", lambda repo_id: False)

    with pytest.raises(HTTPException) as excinfo:
        _switch("local/flux.1")

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_gguf_the_loader_cannot_open_is_not_advertised(catalog, tmp_path):
    # A non-active HF cache is scanned with the snapshot directory as the entry path, so it is
    # never unwrapped to a repo id. Its entries are symlinks into blobs/, which both load
    # validators refuse for escaping the directory, and the repo id would send the loader to the
    # active cache and download the model again.
    _repo_dir, snapshot = _hf_cache_repo(
        tmp_path, "unsloth/Z-Image-Turbo-GGUF", files = ["z-image-turbo-Q4_K_S.gguf"]
    )
    catalog.append(
        _info(
            "unsloth/Z-Image-Turbo-GGUF",
            snapshot,
            task = mas.IMAGE_TASK,
            model_format = "gguf",
            source = "hf_cache",
        )
    )

    assert mas.resolve_local_media_model("unsloth/Z-Image-Turbo-GGUF", task = mas.IMAGE_TASK) is None
    assert mas.available_media_model_ids(mas.IMAGE_TASK) == []


def test_a_cpu_switch_does_not_hold_the_gates_it_cannot_evict_behind(
    flux, enabled, backend, loads, monkeypatch
):
    # A CPU load releases GPU ownership instead of taking it, so waiting on chat's lifecycle gate
    # let an unrelated teardown time the switch out, and holding it blocked new chat requests for
    # as long as the re-plan and the load registration took.
    import core.inference.llama_keepwarm as chat

    monkeypatch.setattr(backends, "load_takes_the_gpu", lambda: False)
    monkeypatch.setattr(mas, "load_takes_the_gpu", lambda: False)
    held: list = []

    def _gate():
        held.append(True)
        raise AssertionError("a cpu switch must not wait on the chat lifecycle gate")

    monkeypatch.setattr(chat, "inference_lifecycle_gate", _gate)

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]
    assert held == []


def test_the_video_route_refuses_a_flow_control_the_target_does_not_expose(monkeypatch):
    # LTX-2 exposes no audio_flow_shift, and _resolve_flow_shifts would only say so after the
    # switch had evicted the resident pipeline and loaded the target.
    import core.inference.video as video_module
    from routes.video import router as video_router

    generated: list = []

    async def _switch_stub(
        requested_model,
        *,
        before_switch = None,
        **kwargs,
    ):
        before_switch(index.MediaModelPick(requested_model, "Lightricks/LTX-Video-2", None, None))

    monkeypatch.setattr(mas, "maybe_auto_switch_media_model", _switch_stub)

    class _Backend:
        def begin_generate(self, **kwargs):
            generated.append(kwargs)

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Backend())

    resp = _client(video_router, "/api/inference").post(
        "/api/inference/video/generate",
        json = {
            "prompt": "a sloth",
            "model": "Lightricks/LTX-Video-2",
            "audio_flow_shift": 3.0,
        },
    )

    assert resp.status_code == 400
    assert "audio_flow_shift" in resp.json()["detail"]
    assert generated == []


def test_an_incomplete_local_pipeline_is_refused_before_the_resident_model_goes(
    catalog, enabled, tmp_path, backend, loads
):
    # A hand-copied or interrupted pipeline still carries model_index.json, and treating the
    # directory as complete let the loader tear the resident pipeline down and only then find
    # that from_pretrained has no weights for one of the components.
    pipeline = tmp_path / "z-image"
    (pipeline / "vae").mkdir(parents = True)
    (pipeline / "vae" / "config.json").write_text("{}")
    (pipeline / "model_index.json").write_text(
        '{"_class_name": "ZImagePipeline", "vae": ["diffusers", "AutoencoderKL"],'
        ' "transformer": ["diffusers", "ZImageTransformer2DModel"]}'
    )
    catalog.append(_info("Tongyi-MAI/Z-Image-Turbo", pipeline, task = mas.IMAGE_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("Tongyi-MAI/Z-Image-Turbo")

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail["error"]["code"] == "model_not_downloaded"
    assert loads == []


def test_a_complete_local_pipeline_still_needs_no_plan(catalog, enabled, tmp_path, backend, loads):
    # The completeness check must not turn every on-disk pipeline into a planned one: a
    # directory whose components are all present is still complete by definition.
    pipeline = tmp_path / "z-image"
    for component in ("vae", "transformer"):
        (pipeline / component).mkdir(parents = True)
        (pipeline / component / "config.json").write_text("{}")
        (pipeline / component / "model.safetensors").write_bytes(b"")
    (pipeline / "model_index.json").write_text(
        '{"_class_name": "ZImagePipeline", "vae": ["diffusers", "AutoencoderKL"],'
        ' "transformer": ["diffusers", "ZImageTransformer2DModel"],'
        ' "safety_checker": [null, null]}'
    )
    catalog.append(_info("Tongyi-MAI/Z-Image-Turbo", pipeline, task = mas.IMAGE_TASK))

    def _explode(model_path, **kwargs):
        raise AssertionError("a complete local pipeline must not be planned against the Hub")

    backend.download_plan = _explode

    _switch("Tongyi-MAI/Z-Image-Turbo")

    assert [pick.model_id for _owner, pick in loads] == ["Tongyi-MAI/Z-Image-Turbo"]


def test_a_sharded_component_missing_a_shard_is_refused(catalog, enabled, tmp_path, backend, loads):
    # The component directory exists and is not empty, but its own weight index names a shard
    # that is not there, which from_pretrained would fetch.
    pipeline = tmp_path / "z-image"
    (pipeline / "transformer").mkdir(parents = True)
    (pipeline / "transformer" / "config.json").write_text("{}")
    (pipeline / "transformer" / "model-00001-of-00002.safetensors").write_bytes(b"")
    (pipeline / "transformer" / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors",'
        ' "b": "model-00002-of-00002.safetensors"}}'
    )
    (pipeline / "model_index.json").write_text(
        '{"_class_name": "ZImagePipeline",'
        ' "transformer": ["diffusers", "ZImageTransformer2DModel"]}'
    )
    catalog.append(_info("Tongyi-MAI/Z-Image-Turbo", pipeline, task = mas.IMAGE_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("Tongyi-MAI/Z-Image-Turbo")

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_native_h3_target_refuses_an_audio_shift_sd_cpp_cannot_apply(monkeypatch):
    # A MiniMax-H3 GGUF always loads through sd.cpp, which derives the audio schedule against a
    # fixed shift, so the engine rule is knowable before the switch rather than only after it.
    import core.inference.video as video_module
    from routes.video import router as video_router

    generated: list = []

    async def _switch_stub(
        requested_model,
        *,
        before_switch = None,
        **kwargs,
    ):
        before_switch(
            index.MediaModelPick(
                requested_model,
                "unsloth/MiniMax-H3-GGUF",
                "minimax_h3_fl2va-Q4_K_M.gguf",
                "gguf",
            )
        )

    monkeypatch.setattr(mas, "maybe_auto_switch_media_model", _switch_stub)

    class _Backend:
        def begin_generate(self, **kwargs):
            generated.append(kwargs)

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Backend())

    resp = _client(video_router, "/api/inference").post(
        "/api/inference/video/generate",
        json = {
            "prompt": "a sloth",
            "model": "unsloth/MiniMax-H3-GGUF",
            "audio_flow_shift": 4.0,
        },
    )

    assert resp.status_code == 400
    assert "Diffusers engine" in resp.json()["detail"]
    assert generated == []


def test_a_load_route_refusal_still_carries_the_openai_envelope(monkeypatch):
    # _start_load calls the internal load route, which raises a plain-string HTTPException; the
    # /v1 surface must still answer in the OpenAI error shape rather than FastAPI's detail body.
    from routes.inference import router as images_router

    async def _refuses(requested_model, **kwargs):
        raise HTTPException(status_code = 400, detail = "'x/y' is not a supported diffusion model.")

    monkeypatch.setattr(mas, "maybe_auto_switch_media_model", _refuses)

    resp = _client(images_router, "/v1").post(
        "/v1/images/generations",
        json = {"prompt": "p", "size": "256x256", "model": "x/y"},
    )

    assert resp.status_code == 400
    assert resp.json()["error"]["message"] == "'x/y' is not a supported diffusion model."


def test_a_component_holding_only_its_config_is_refused(catalog, enabled, tmp_path, backend, loads):
    # A partially copied pipeline leaves the component directory in place with its config and
    # none of its weights, which the nonempty-directory test alone accepted.
    pipeline = tmp_path / "z-image"
    (pipeline / "transformer").mkdir(parents = True)
    (pipeline / "transformer" / "config.json").write_text("{}")
    (pipeline / "model_index.json").write_text(
        '{"_class_name": "ZImagePipeline",'
        ' "transformer": ["diffusers", "ZImageTransformer2DModel"]}'
    )
    catalog.append(_info("Tongyi-MAI/Z-Image-Turbo", pipeline, task = mas.IMAGE_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("Tongyi-MAI/Z-Image-Turbo")

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_weightless_component_is_still_accepted(catalog, enabled, tmp_path, backend, loads):
    # Schedulers and processors ship no weights at all and declare their own *_config.json
    # rather than config.json, so requiring a weight file must not refuse them. A tokenizer
    # ships no weights either but is useless without its vocabulary, so that one is required.
    pipeline = tmp_path / "z-image"
    (pipeline / "scheduler").mkdir(parents = True)
    (pipeline / "scheduler" / "scheduler_config.json").write_text("{}")
    (pipeline / "tokenizer").mkdir(parents = True)
    (pipeline / "tokenizer" / "tokenizer_config.json").write_text("{}")
    (pipeline / "tokenizer" / "vocab.json").write_text("{}")
    (pipeline / "model_index.json").write_text(
        '{"_class_name": "ZImagePipeline", "scheduler": ["diffusers", "FlowMatchEulerScheduler"],'
        ' "tokenizer": ["transformers", "Qwen2Tokenizer"]}'
    )
    catalog.append(_info("Tongyi-MAI/Z-Image-Turbo", pipeline, task = mas.IMAGE_TASK))

    _switch("Tongyi-MAI/Z-Image-Turbo")

    assert [pick.model_id for _owner, pick in loads] == ["Tongyi-MAI/Z-Image-Turbo"]


def test_a_tokenizer_without_its_vocabulary_is_refused(catalog, enabled, tmp_path, backend, loads):
    # tokenizer_config.json alone builds no tokenizer: from_pretrained fetches the vocabulary,
    # and by then the resident pipeline is already gone.
    pipeline = tmp_path / "z-image"
    (pipeline / "tokenizer").mkdir(parents = True)
    (pipeline / "tokenizer" / "tokenizer_config.json").write_text("{}")
    (pipeline / "model_index.json").write_text(
        '{"_class_name": "ZImagePipeline", "tokenizer": ["transformers", "Qwen2Tokenizer"]}'
    )
    catalog.append(_info("Tongyi-MAI/Z-Image-Turbo", pipeline, task = mas.IMAGE_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("Tongyi-MAI/Z-Image-Turbo")

    assert excinfo.value.status_code == 409
    assert loads == []


def test_an_incomplete_local_video_pipeline_is_refused_too(
    catalog, enabled, tmp_path, backend, loads
):
    # The video planner omits the base files whenever the local path exists, so an incomplete
    # directory reported nothing missing and the load worker tore the resident pipeline down.
    pipeline = tmp_path / "wan"
    (pipeline / "transformer").mkdir(parents = True)
    (pipeline / "transformer" / "config.json").write_text("{}")
    (pipeline / "model_index.json").write_text(
        '{"_class_name": "WanPipeline", "transformer": ["diffusers", "WanTransformer3DModel"]}'
    )
    catalog.append(_info("Wan-AI/Wan2.2-T2V-A14B", pipeline, task = mas.VIDEO_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("Wan-AI/Wan2.2-T2V-A14B", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_single_file_hidream_still_checks_its_encoder(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # normalized_pick gives a standalone checkpoint a gguf_filename, which skipped the encoder
    # check, while the single-file assembly still loads the Llama repo unconditionally.
    directory = tmp_path / "hidream-i1-dev"
    directory.mkdir()
    (directory / "hidream-i1-dev.safetensors").write_bytes(b"")
    catalog.append(_info("HiDream-ai/HiDream-I1-Dev", directory, task = mas.IMAGE_TASK))
    monkeypatch.setattr(locality, "encoder_repo_complete", lambda repo_id: False)

    with pytest.raises(HTTPException) as excinfo:
        _switch("HiDream-ai/HiDream-I1-Dev")

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail["error"]["code"] == "model_not_downloaded"
    assert loads == []


def test_a_native_h3_target_refuses_max_reference_sizing(monkeypatch):
    # stable-diffusion.cpp scales every reference to the generation's pixel area, so 'max' needs
    # the Diffusers engine, and an H3 GGUF is always native.
    import core.inference.video as video_module
    from routes.video import router as video_router

    generated: list = []

    async def _switch_stub(
        requested_model,
        *,
        before_switch = None,
        **kwargs,
    ):
        before_switch(
            index.MediaModelPick(
                requested_model,
                "unsloth/MiniMax-H3-GGUF",
                "ref2va/minimax_h3_ref2va-Q4_K_M.gguf",
                "gguf",
            )
        )

    monkeypatch.setattr(mas, "maybe_auto_switch_media_model", _switch_stub)

    class _Backend:
        def begin_generate(self, **kwargs):
            generated.append(kwargs)

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Backend())

    resp = _client(video_router, "/api/inference").post(
        "/api/inference/video/generate",
        json = {
            "prompt": "a sloth",
            "model": "unsloth/MiniMax-H3-GGUF",
            "reference_images": ["data:image/png;base64,AAAA"],
            "reference_image_size": "max",
        },
    )

    assert resp.status_code == 400
    assert "Diffusers engine" in resp.json()["detail"]
    assert generated == []


def test_a_modular_component_the_index_keeps_locally_is_checked(
    catalog, enabled, tmp_path, backend, loads
):
    # A modular entry is [library, class, spec], so a two-element test skipped every component
    # of a MiniMax-H3 pipeline. One whose spec names another repo is the planner's business; one
    # with no source is a local subfolder this directory is expected to hold.
    modular = tmp_path / "h3"
    (modular / "transformer").mkdir(parents = True)
    (modular / "transformer" / "config.json").write_text("{}")
    (modular / "modular_model_index.json").write_text(
        '{"transformer": ["diffusers", "MiniMaxH3Transformer3DModel", {}],'
        ' "text_encoder": ["transformers", "Qwen3VLModel",'
        ' {"pretrained_model_name_or_path": "unsloth/MiniMax-H3"}]}'
    )
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409
    assert loads == []


@pytest.mark.parametrize("cached", [True, False])
def test_a_hosted_modular_component_is_checked_against_the_cache(
    catalog, enabled, tmp_path, backend, loads, monkeypatch, cached
):
    # load_components pulls each repository the modular index names, and the video planner omits
    # its base manifest whenever the local path exists, so a hosted component that is not on
    # disk would be downloaded after the resident pipeline had already gone.
    import core.inference.diffusion as diffusion_module

    cache = tmp_path / "hub"
    component = cache / "models--unsloth--MiniMax-H3" / "snapshots" / ("a" * 40) / "vae"
    if cached:
        component.mkdir(parents = True)
        (component / "config.json").write_text("{}")
        (component / "model.safetensors").write_bytes(b"")
    monkeypatch.setattr(diffusion_module, "hub_cache_dir", lambda: str(cache))

    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text(
        '{"vae": ["diffusers", "AutoencoderKLMiniMaxH3",'
        ' {"pretrained_model_name_or_path": "unsloth/MiniMax-H3", "subfolder": "vae"}]}'
    )
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))

    if cached:
        _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)
        assert [pick.model_id for _owner, pick in loads] == ["MiniMaxAI/MiniMax-H3"]
        return

    with pytest.raises(HTTPException) as excinfo:
        _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)
    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_cached_single_checkpoint_the_loader_cannot_open_is_not_advertised(catalog, tmp_path):
    # The directory holds exactly one checkpoint, so both load routes reinterpret it as a
    # single_file load and resolve the name through the same containment check a GGUF gets,
    # which a snapshot's symlink into blobs/ fails.
    _repo_dir, snapshot = _hf_cache_repo(
        tmp_path, "unsloth/Z-Image-Turbo", files = ["z-image-turbo.safetensors"]
    )
    catalog.append(_info("unsloth/Z-Image-Turbo", snapshot, task = mas.IMAGE_TASK, source = "hf_cache"))

    assert mas.resolve_local_media_model("unsloth/Z-Image-Turbo", task = mas.IMAGE_TASK) is None


def test_a_stalled_load_probe_still_answers_inside_the_budget(flux, enabled, backend, monkeypatch):
    # load_progress walks cache directories to count bytes, so one poll on a stalled filesystem
    # outlived the budget the check at the bottom of the loop is there to enforce. Timed inside
    # the loop: asyncio.run joins the executor on the way out, so the worker's own sleep would
    # otherwise be counted against a request that had already answered.
    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.4)

    async def _start(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        backend.phase = "downloading"
        backend.load_progress = lambda: time.sleep(3) or {"phase": "downloading"}

    monkeypatch.setattr(mas, "_start_load", _start)

    async def _drive():
        began = time.monotonic()
        with pytest.raises(HTTPException) as excinfo:
            await mas.maybe_auto_switch_media_model(
                "black-forest-labs/FLUX.1-dev",
                owner = arb.DIFFUSION,
                current_subject = "test-user",
                openai_errors = True,
            )
        assert excinfo.value.status_code == 503
        assert time.monotonic() - began < 2.0

    asyncio.run(_drive())


def test_setup_that_never_registers_gives_the_gates_back(flux, enabled, backend, monkeypatch):
    # A first-run native install runs for minutes before begin_load, and holding both media
    # admission gates and chat's that long blocks every unrelated request.
    import core.inference.media_keepwarm as keepwarm

    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.3)
    monkeypatch.setattr(mas, "_SETUP_GRACE_S", 0.5)
    release = asyncio.Event()

    async def _never_registers(
        owner,
        pick,
        current_subject,
        hf_token = None,
    ):
        await release.wait()

    monkeypatch.setattr(mas, "_start_load", _never_registers)

    async def _drive():
        with pytest.raises(HTTPException) as excinfo:
            await mas.maybe_auto_switch_media_model(
                "black-forest-labs/FLUX.1-dev",
                owner = arb.DIFFUSION,
                current_subject = "test-user",
                openai_errors = True,
            )
        assert excinfo.value.status_code == 503
        for _ in range(200):
            await asyncio.sleep(0.01)
            if not keepwarm._TRACKERS[arb.DIFFUSION].gate.locked():
                break
        assert not keepwarm._TRACKERS[arb.DIFFUSION].gate.locked()
        assert not mas.switch_lock(arb.DIFFUSION).locked()
        release.set()

    asyncio.run(_drive())


def test_a_superseded_snapshot_does_not_vouch_for_the_active_one(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # refs/main names the revision from_pretrained resolves to, so a complete component in an
    # older snapshot must not answer for the partial one the load will actually read.
    import core.inference.diffusion as diffusion_module

    cache = tmp_path / "hub"
    repo = cache / "models--unsloth--MiniMax-H3"
    old = repo / "snapshots" / ("a" * 40) / "vae"
    old.mkdir(parents = True)
    (old / "config.json").write_text("{}")
    (old / "model.safetensors").write_bytes(b"")
    (repo / "snapshots" / ("b" * 40)).mkdir(parents = True)
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text("b" * 40)
    monkeypatch.setattr(diffusion_module, "hub_cache_dir", lambda: str(cache))

    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text(
        '{"vae": ["diffusers", "AutoencoderKLMiniMaxH3",'
        ' {"pretrained_model_name_or_path": "unsloth/MiniMax-H3", "subfolder": "vae"}]}'
    )
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_root_level_hosted_component_needs_more_than_one_shard(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # A component named as a whole repo went through _upstream_is_cached, whose no-manifest
    # branch is satisfied by a single weight file an interrupted sharded pull leaves behind.
    import core.inference.diffusion as diffusion_module

    cache = tmp_path / "hub"
    snapshot = cache / "models--unsloth--MiniMax-H3-VAE" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"")
    (snapshot / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00001-of-00002.safetensors",'
        ' "b": "model-00002-of-00002.safetensors"}}'
    )
    monkeypatch.setattr(diffusion_module, "hub_cache_dir", lambda: str(cache))

    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text(
        '{"vae": ["diffusers", "AutoencoderKLMiniMaxH3",'
        ' {"pretrained_model_name_or_path": "unsloth/MiniMax-H3-VAE"}]}'
    )
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_split_gguf_missing_a_shard_is_not_advertised(catalog, tmp_path):
    # The loader opens the sibling shards implicitly and the planners read a local checkpoint as
    # present, so half a split set would evict the resident model and then fail at startup.
    (tmp_path / "z-image-Q4_K_M-00001-of-00002.gguf").write_bytes(b"")
    catalog.append(
        _info(
            "z-image",
            tmp_path / "z-image-Q4_K_M-00001-of-00002.gguf",
            task = mas.IMAGE_TASK,
            model_format = "gguf",
        )
    )

    assert mas.resolve_local_media_model("z-image", task = mas.IMAGE_TASK) is None

    (tmp_path / "z-image-Q4_K_M-00002-of-00002.gguf").write_bytes(b"")
    mas.invalidate_index()

    assert mas.resolve_local_media_model("z-image", task = mas.IMAGE_TASK) is not None


def test_a_pinned_component_revision_is_the_one_checked(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # ComponentSpec.load is handed the whole spec, so a pinned revision is what gets fetched;
    # a complete default snapshot must not answer for it.
    import core.inference.diffusion as diffusion_module

    cache = tmp_path / "hub"
    repo = cache / "models--unsloth--MiniMax-H3"
    main = repo / "snapshots" / ("a" * 40) / "vae"
    main.mkdir(parents = True)
    (main / "config.json").write_text("{}")
    (main / "model.safetensors").write_bytes(b"")
    (repo / "refs").mkdir(parents = True)
    (repo / "refs" / "main").write_text("a" * 40)
    monkeypatch.setattr(diffusion_module, "hub_cache_dir", lambda: str(cache))

    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text(
        '{"vae": ["diffusers", "AutoencoderKLMiniMaxH3",'
        ' {"pretrained_model_name_or_path": "unsloth/MiniMax-H3", "subfolder": "vae",'
        ' "revision": "' + "c" * 40 + '"}]}'
    )
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_pinned_component_variant_needs_its_own_weights(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # from_pretrained asks for the named variant's files rather than falling back to the
    # default ones, so a component holding only the default weights is not complete for it.
    import core.inference.diffusion as diffusion_module

    cache = tmp_path / "hub"
    component = cache / "models--unsloth--MiniMax-H3" / "snapshots" / ("a" * 40) / "vae"
    component.mkdir(parents = True)
    (component / "config.json").write_text("{}")
    (component / "diffusion_pytorch_model.safetensors").write_bytes(b"")
    monkeypatch.setattr(diffusion_module, "hub_cache_dir", lambda: str(cache))

    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text(
        '{"vae": ["diffusers", "AutoencoderKLMiniMaxH3",'
        ' {"pretrained_model_name_or_path": "unsloth/MiniMax-H3", "subfolder": "vae",'
        ' "variant": "fp16"}]}'
    )
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409

    (component / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"")
    mas.invalidate_index()
    loads.clear()

    _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert [pick.model_id for _owner, pick in loads] == ["MiniMaxAI/MiniMax-H3"]


def test_a_local_component_source_is_checked_without_a_subfolder(
    catalog, enabled, tmp_path, backend, loads
):
    # A spec pointing straight at a local directory was accepted for existing at all, so an
    # empty or half-copied one passed and load_components discovered it after the eviction.
    empty = tmp_path / "vae-src"
    empty.mkdir()
    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text(
        '{"vae": ["diffusers", "AutoencoderKLMiniMaxH3",'
        ' {"pretrained_model_name_or_path": "' + str(empty) + '"}]}'
    )
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_shard_index_declaring_nothing_is_not_proof(catalog, enabled, tmp_path, backend, loads):
    # An empty weight_map names no shard, so nothing is missing by that reading, and the mere
    # presence of the index was being taken as the component being complete.
    pipeline = tmp_path / "z-image"
    (pipeline / "transformer").mkdir(parents = True)
    (pipeline / "transformer" / "config.json").write_text("{}")
    (pipeline / "transformer" / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    (pipeline / "model_index.json").write_text(
        '{"_class_name": "ZImagePipeline",'
        ' "transformer": ["diffusers", "ZImageTransformer2DModel"]}'
    )
    catalog.append(_info("Tongyi-MAI/Z-Image-Turbo", pipeline, task = mas.IMAGE_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("Tongyi-MAI/Z-Image-Turbo")

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_cached_split_gguf_missing_a_shard_is_not_advertised(catalog, tmp_path, monkeypatch):
    # An active cache is loaded by repo id, which skipped the split check entirely, while the
    # planners look only at the selected first shard and report nothing missing.
    import core.inference.diffusion as diffusion_module

    repo_dir, snapshot = _hf_cache_repo(
        tmp_path,
        "unsloth/Z-Image-Turbo-GGUF",
        files = ["z-image-Q4_K_M-00001-of-00002.gguf"],
    )
    monkeypatch.setattr(diffusion_module, "hub_cache_dir", lambda: str(tmp_path))
    catalog.append(
        _info(
            "unsloth/Z-Image-Turbo-GGUF",
            repo_dir,
            task = mas.IMAGE_TASK,
            model_format = "gguf",
            source = "hf_cache",
        )
    )

    assert mas.resolve_local_media_model("unsloth/Z-Image-Turbo-GGUF", task = mas.IMAGE_TASK) is None

    (snapshot / "z-image-Q4_K_M-00002-of-00002.gguf").write_bytes(b"")
    mas.invalidate_index()

    assert (
        mas.resolve_local_media_model("unsloth/Z-Image-Turbo-GGUF", task = mas.IMAGE_TASK) is not None
    )


def test_chat_admitted_before_the_gate_still_stops_the_switch(
    flux, enabled, backend, loads, monkeypatch, takes_the_gpu
):
    # A chat request that passed the lifecycle gate after the outer drain's last probe is
    # already running when the switch takes that gate, and the GPU handoff would terminate it.
    import core.inference.llama_keepwarm as chat

    monkeypatch.setattr(mas, "_DRAIN_WAIT_S", 0.3)
    outer = {"done": False}

    def _counted(current_request_counted = True, *, include_pending = True):
        # idle until the outer drain has passed, then in flight for the in-gate check
        return 1 if outer["done"] else 0

    def _pass_outer_drain(owner, backend_obj, deadline, **kwargs):
        if kwargs.get("count_pending", True):
            outer["done"] = True
        return _real_drain(owner, backend_obj, deadline, **kwargs)

    _real_drain = mas.drain
    monkeypatch.setattr(chat, "other_inference_request_count", _counted)
    monkeypatch.setattr(mas, "drain", _pass_outer_drain)

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 409
    assert loads == []


def test_a_scheduler_without_its_config_is_refused(catalog, enabled, tmp_path, backend, loads):
    # A metadata-only component IS its config, so a directory holding a stray file and nothing
    # else builds nothing and would be fetched after the resident model had gone.
    pipeline = tmp_path / "z-image"
    (pipeline / "scheduler").mkdir(parents = True)
    (pipeline / "scheduler" / "README.md").write_text("hello")
    (pipeline / "model_index.json").write_text(
        '{"_class_name": "ZImagePipeline", "scheduler": ["diffusers", "FlowMatchEulerScheduler"]}'
    )
    catalog.append(_info("Tongyi-MAI/Z-Image-Turbo", pipeline, task = mas.IMAGE_TASK))

    with pytest.raises(HTTPException) as excinfo:
        _switch("Tongyi-MAI/Z-Image-Turbo")

    assert excinfo.value.status_code == 409
    assert loads == []


def test_two_h3_partitions_of_one_quant_are_told_apart(catalog, enabled, tmp_path, backend, loads):
    # Both denoisers share a directory and a quant token, but status publishes h3_task and
    # partition_matches reads it, so marking them indistinguishable reloaded on every request.
    for partition in ("fl2va", "ref2va"):
        name = f"minimax_h3_{partition}-Q4_K_M.gguf"
        (tmp_path / name).write_bytes(b"")
        catalog.append(
            _info(
                f"minimax_h3_{partition}-Q4_K_M",
                tmp_path / name,
                task = mas.VIDEO_TASK,
                model_format = "gguf",
            )
        )
    backend.repo_id = str(tmp_path)
    backend.gguf_variant = "Q4_K_M"
    backend.model_kind = "gguf"
    backend.h3_task = "ref2va"

    _switch("minimax_h3_ref2va-Q4_K_M", owner = arb.VIDEO, openai_errors = False)

    # The resident partition is the one this checkpoint brings up, so nothing is reloaded.
    assert loads == []


def _capture_begin_load(monkeypatch, route):
    """Record the local_files_only begin_load is called with, for either media route."""
    seen: list = []

    def _begin_load(*a, **k):
        seen.append(k.get("local_files_only"))
        return {"loaded": False, "repo_id": None}

    if route == "images":
        import core.inference.diffusion_engine_router as router_module

        class _Backend:
            def validate_load_request(self, *a, **k):
                return types.SimpleNamespace(name = "z-image", base_repo = None)

            begin_load = staticmethod(_begin_load)

        monkeypatch.setattr(router_module, "select_and_activate_engine", lambda *a, **k: _Backend())
        monkeypatch.setattr(router_module, "get_active_diffusion_engine", lambda: _Backend())
    else:
        import core.inference.video as video_module

        backend = _video_load_backend(
            validate_load_request = lambda *a, **k: _a_real_video_family(),
            begin_load = _begin_load,
        )
        monkeypatch.setattr(video_module, "get_video_backend", lambda: backend)
        monkeypatch.setattr(video_module, "resolve_video_model_kind", lambda *a, **k: "gguf")
        monkeypatch.setattr(video_module, "assert_video_precision_available", lambda *a, **k: None)
        monkeypatch.setattr("routes.video._guard_video_load_against_training", lambda: None)
        monkeypatch.setattr("routes.video._selected_gpu_ordinal", _async_none)
    return seen


@pytest.mark.parametrize("user_initiated", [False, True])
def test_only_a_load_nobody_asked_for_is_kept_off_the_hub(monkeypatch, user_initiated):
    # The switch verifies locality from the outside, and this is what makes that promise the
    # loader's own rule rather than a prediction about which files it will open. The picker's
    # own load is what downloads a model in the first place, so it keeps its access.
    from models.inference import VideoLoadRequest
    from routes.video import load_video_model_gated

    seen = _capture_begin_load(monkeypatch, "video")
    import core.inference.diffusion_device as device_module

    monkeypatch.setattr(
        device_module,
        "resolve_diffusion_device_target",
        lambda: types.SimpleNamespace(device = "cpu"),
    )

    asyncio.run(
        load_video_model_gated(
            VideoLoadRequest(model_path = "unsloth/Wan2.2-GGUF", gguf_filename = "wan-Q4_K_M.gguf"),
            "test-user",
            user_initiated = user_initiated,
        )
    )

    assert seen == [not user_initiated]


def test_a_non_h3_sibling_stays_in_the_ambiguity_group(catalog, enabled, tmp_path, backend, loads):
    # An H3 denoiser and another family's build sharing a directory and a quant token are
    # indistinguishable to status, and partition_matches reads a resident fl2va as answering
    # for the non-H3 one, so neither may be treated as already serving the other.
    for name in ("minimax_h3_fl2va-Q4_K_M.gguf", "wan-Q4_K_M.gguf"):
        (tmp_path / name).write_bytes(b"")
        catalog.append(
            _info(
                name.removesuffix(".gguf"),
                tmp_path / name,
                task = mas.VIDEO_TASK,
                model_format = "gguf",
            )
        )
    backend.repo_id = str(tmp_path)
    backend.gguf_variant = "Q4_K_M"
    backend.model_kind = "gguf"
    backend.h3_task = "fl2va"

    _switch("wan-Q4_K_M", owner = arb.VIDEO, openai_errors = False)

    # The resident H3 checkpoint must not answer for the Wan request.
    assert [pick.gguf_filename for _owner, pick in loads] == ["wan-Q4_K_M.gguf"]


async def _until(predicate, *, timeout = 5.0):
    """Wait for a real condition rather than guessing a sleep length."""
    limit = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < limit, "condition never held"
        await asyncio.sleep(0.01)


def test_a_media_request_parked_on_a_gate_is_not_read_as_running_chat_work(monkeypatch):
    # Every media generation route is counted on chat's in-flight counter as well as its own.
    # With the media gates taken before chat's, a request arriving in between passed the still
    # open chat gate, incremented chat's _inflight, and only then blocked on the held media
    # gate: the in-gate drain discounted it on the media side but chat_busy(count_pending=False)
    # read the same blocked request as running chat work, so an otherwise idle switch answered
    # 409 and loaded nothing. Driven with the real gates and the real middleware.
    import core.inference.llama_keepwarm as chat

    class _Backend:
        def status(self):
            return {"loaded": True, "repo_id": "resident/model"}

        def loading_repo_ids(self):
            return []

        def generate_progress(self):
            return {"active": False}

    monkeypatch.setattr(backends, "load_takes_the_gpu", lambda: True)
    monkeypatch.setattr(backends, "other_backend_busy", lambda owner: False)
    monkeypatch.setattr(backends, "backend_for", lambda owner: _Backend())
    monkeypatch.setattr(mas, "backend_for", lambda owner: _Backend())
    monkeypatch.setattr(mas, "satisfied_by", lambda status, name, pick: False)

    started: list = []

    async def _start_load(owner, pick, subject, token):
        started.append(pick)

    async def _require_local(*args, **kwargs):
        return None

    monkeypatch.setattr(mas, "_start_load", _start_load)
    monkeypatch.setattr(mas, "_require_local", _require_local)

    pick = index.MediaModelPick("org/target", "/models/target", None, None)
    image = mk._TRACKERS[arb.DIFFUSION]
    video = mk._TRACKERS[arb.VIDEO]

    async def _app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200})
        await send({"type": "http.response.body", "body": b"", "more_body": False})

    async def _scenario():
        # The switching request is itself tracked by the middleware, on both counters.
        chat._note_pending()
        chat._note_start()
        image.note_pending()
        image.note_start()
        # An ordinary holder of the other backend's admission gate: the media idle tick keeps
        # it across status(), the busy probe and unload(). A real lock, not a patched count.
        assert video.gate.acquire(blocking = False)
        newcomer = None
        try:
            switcher = asyncio.ensure_future(
                mas._gated_start_load(
                    arb.DIFFUSION,
                    "org/target",
                    pick,
                    "test-user",
                    [],
                    time.monotonic() + 30.0,
                    kind = "image",
                    openai_errors = True,
                    hf_token = None,
                    takes_the_gpu = True,
                )
            )
            # it is parked on the video gate, having taken everything ahead of it
            await _until(lambda: image.gate.locked())

            # a second /v1/images/generations arrives now, through the real middleware
            newcomer = asyncio.ensure_future(
                chat.LlamaKeepWarmMiddleware(_app)(
                    {
                        "type": "http",
                        "method": "POST",
                        "path": "/v1/images/generations",
                        "headers": [(b"authorization", b"Bearer token")],
                    },
                    None,
                    lambda message: asyncio.sleep(0),
                )
            )
            # counted by the middleware, on whichever gate it came to rest on
            await _until(lambda: chat._inflight + chat._pending > 1)

            # it is waiting, so it is neither running chat work nor touching the backend
            assert image.outstanding(count_pending = False) == 1
            assert backends.chat_busy(False) is False

            video.gate.release()
            assert await asyncio.wait_for(switcher, 10) is False
        finally:
            with contextlib.suppress(RuntimeError):
                video.gate.release()
            if newcomer is not None:
                newcomer.cancel()
                with contextlib.suppress(BaseException):
                    await newcomer

    try:
        asyncio.run(_scenario())
    finally:
        chat._inflight = chat._pending = 0
        for tracker in mk._TRACKERS.values():
            tracker._inflight = tracker._pending = 0

    assert [entry.model_id for entry in started] == ["org/target"]
