# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Opt-in media auto-switch: the name resolver and the switch it drives.

The local-model scan is replaced with fixture entries pointing at real (empty) files, and
the load routes with fakes, so these exercise resolution, the drain/load sequencing and the
error envelopes without torch, diffusers, weights or a GPU.
"""

from __future__ import annotations

import asyncio
import time
import types

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import core.inference.gpu_arbiter as arb
import core.inference.media_auto_switch as mas
import core.inference.media_keepwarm as mk
import routes.models as models_route
import utils.openai_auto_switch_settings as settings
from auth.authentication import get_current_subject
from core.inference.openai_auto_download import preferred_quant
from utils.api_errors import install_api_error_handlers


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


# ── resolving a name ────────────────────────────────────────────────


def test_a_diffusers_directory_resolves_as_a_kindless_pick(catalog, tmp_path):
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))

    pick = mas.resolve_local_media_model("black-forest-labs/FLUX.1-dev", task = mas.IMAGE_TASK)

    # No kind and no filename: the load route detects those, including the single-file case.
    assert pick == mas.MediaModelPick("black-forest-labs/FLUX.1-dev", str(pipeline))


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

    assert pick == mas.MediaModelPick(
        "z-image", str(tmp_path), "z-image-Q4_K_M.gguf", "gguf", quant = "Q4_K_M"
    )


def test_the_index_is_keyed_by_task(catalog, tmp_path):
    clip = tmp_path / "wan"
    clip.mkdir()
    (clip / "model_index.json").write_text("{}")
    (clip / "model_index.json").write_text("{}")
    catalog.append(_info("unsloth/Wan2.2", clip, task = mas.VIDEO_TASK))

    assert mas.resolve_local_media_model("unsloth/Wan2.2", task = mas.VIDEO_TASK) is not None
    # An image request must not be answered by a video model, or the load 400s after eviction.
    assert mas.resolve_local_media_model("unsloth/Wan2.2", task = mas.IMAGE_TASK) is None


def test_an_unknown_name_resolves_to_nothing(catalog):
    assert mas.resolve_local_media_model("someone/else", task = mas.IMAGE_TASK) is None
    assert mas.resolve_local_media_model("", task = mas.IMAGE_TASK) is None


def test_an_absolute_path_is_not_an_advertised_name(catalog, tmp_path):
    pipeline = tmp_path / "local-only"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    row = _info(str(pipeline), pipeline, task = mas.IMAGE_TASK, display_name = "Local Only")
    row.model_id = None
    catalog.append(row)

    # The scanners report the on-disk path as the id; a caller should not have to send one.
    assert mas.resolve_local_media_model(str(pipeline), task = mas.IMAGE_TASK) is None
    assert mas.resolve_local_media_model("Local Only", task = mas.IMAGE_TASK) is not None


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
    monkeypatch.setattr(mas, "_backend_for", lambda owner: fake)
    # The real planner resolves the engine router and reaches the Hub; this suite is offline.
    monkeypatch.setattr(mas, "_planners_for", lambda owner, pick: [fake])
    return fake


@pytest.fixture
def loads(monkeypatch, backend):
    """Record every load the switch starts, and bring the fake up as the loader would."""
    started: list = []

    async def _start(owner, pick, current_subject):
        started.append((owner, pick))
        backend.repo_id = pick.model_path
        # The real backends publish extract_quant_token, not the lister label.
        backend.gguf_variant = mas._published_token(pick) or None
        backend.model_kind = pick.model_kind
        # A switch sends no h3_task, so the load comes up on the family default.
        backend.h3_task = None
        backend.phase = "ready"
        mk.note_load_origin(owner, pick.model_path, user_action = False)

    monkeypatch.setattr(mas, "_start_load", _start)
    return started


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


def test_the_switch_is_inert_while_the_setting_is_off(catalog, monkeypatch, loads):
    monkeypatch.setattr(settings, "get_media_auto_switch_enabled", lambda: False)
    # Not even an unknown name is refused: `model` keeps its old informational meaning.
    _switch("someone/else")
    assert loads == []


def test_no_model_named_is_a_no_op(catalog, enabled, loads):
    _switch(None)
    _switch("   ")
    assert loads == []


def test_an_unresolvable_name_is_refused_and_lists_what_is_downloaded(
    catalog, enabled, tmp_path, loads
):
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))

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


def test_a_resident_model_is_not_reloaded(catalog, enabled, tmp_path, backend, loads):
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = "black-forest-labs/FLUX.1-dev"

    _switch("black-forest-labs/FLUX.1-dev")

    assert loads == []


def test_a_model_loaded_by_path_still_counts_as_serving(catalog, enabled, tmp_path, backend, loads):
    # A model this module loaded reports the local path it was given, not the repo id.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = str(pipeline)

    _switch("black-forest-labs/FLUX.1-dev")

    assert loads == []


def test_a_different_model_is_loaded_before_the_request_proceeds(
    catalog, enabled, tmp_path, backend, loads
):
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = "Qwen/Qwen-Image"

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]
    # The load came from the API, so "only unload models loaded by the API" may free it.
    assert mk.loaded_by_user_action(arb.DIFFUSION) is False


def test_a_busy_backend_is_not_swapped_out_from_under_its_generation(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    monkeypatch.setattr(mas, "_DRAIN_WAIT_S", 0.0)
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = "Qwen/Qwen-Image"
    backend.active = True

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 409
    assert excinfo.value.headers["Retry-After"] == "15"
    assert loads == []


def test_a_load_still_running_at_the_deadline_asks_for_a_retry(
    catalog, enabled, tmp_path, backend, monkeypatch
):
    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.0)
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))

    async def _start(owner, pick, current_subject):
        backend.phase = "downloading"

    monkeypatch.setattr(mas, "_start_load", _start)

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 503
    assert excinfo.value.headers["Retry-After"] == "15"


def test_a_failed_load_surfaces_its_error(catalog, enabled, tmp_path, backend, monkeypatch):
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))

    async def _start(owner, pick, current_subject):
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

    async def _switch_stub(requested_model, *, owner, current_subject, openai_errors):
        calls.append((requested_model, owner, openai_errors))

    monkeypatch.setattr(mas, "maybe_auto_switch_media_model", _switch_stub)

    class _Backend:
        def begin_generate(self, **kwargs):
            return None

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Backend())

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(video_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    resp = TestClient(app).post(
        "/api/inference/video/generate",
        json = {"prompt": "a sloth", "model": "unsloth/Wan2.2"},
    )

    assert resp.status_code == 200
    # Not the OpenAI envelope: this route is a Studio surface and its errors are plain details.
    assert calls == [("unsloth/Wan2.2", arb.VIDEO, False)]


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


def test_a_pick_whose_companions_are_missing_is_refused_rather_than_downloaded(
    catalog, enabled, tmp_path, backend, loads
):
    # The resolver only indexes downloaded checkpoints; a GGUF still loads its encoders and VAE
    # from a base repo the loader would fetch. Auto-switch promises it never downloads.
    # A cached GGUF, not a local pipeline: the latter is complete by definition and never planned.
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
    backend.missing_bytes = 4_300_000_000

    with pytest.raises(HTTPException) as excinfo:
        _switch("city96/FLUX.1-dev-gguf")

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail["error"]["code"] == "model_not_downloaded"
    assert "4.3 GB" in excinfo.value.detail["error"]["message"]
    assert loads == []


def test_a_request_queued_on_the_switch_lock_does_not_block_the_drain(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # Two concurrent requests for the same absent model are both counted by the middleware.
    # Counting the queued one as work to drain made each wait the other out and both 409.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = "Qwen/Qwen-Image"
    monkeypatch.setattr(mas, "_DRAIN_WAIT_S", 0.5)
    # Two tracked requests in flight, and both parked on the switch: only one holds the lock.
    monkeypatch.setattr(
        mk,
        "other_request_count",
        lambda owner, current_request_counted = False, count_pending = True: 1,
    )
    monkeypatch.setattr(mas, "_waiter_count", lambda owner: 2)

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_the_drain_and_the_load_share_one_budget(catalog, enabled, tmp_path, backend, monkeypatch):
    # Separate budgets added up past the ~100s tunnel window, so a slow switch lost the socket
    # instead of returning the retryable 503 the bounds exist to produce.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.6)
    monkeypatch.setattr(mas, "_DRAIN_WAIT_S", 30.0)
    busy_until = [True]

    def _busy(_backend):
        # Busy for the first poll only, so the drain eats part of the shared budget.
        was = busy_until[0]
        busy_until[0] = False
        return was

    monkeypatch.setattr(mas, "_backend_busy", _busy)

    async def _start(owner, pick, current_subject):
        backend.phase = "downloading"

    monkeypatch.setattr(mas, "_start_load", _start)

    began = time.monotonic()
    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 503
    # The load wait inherits what the drain left, so the whole switch stays inside the budget.
    assert time.monotonic() - began < 2.0


def test_a_load_that_lands_while_draining_is_not_repeated(
    catalog, enabled, tmp_path, backend, loads
):
    # A retry can acquire the switch lock while the earlier attempt's load is still running.
    # Draining waits that out, and without a recheck the retry tears down what just landed.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = "Qwen/Qwen-Image"
    backend.loading = ("black-forest-labs/FLUX.1-dev",)

    async def _drain_lands_the_model(_owner, _backend, _deadline):
        backend.loading = ()
        backend.repo_id = str(pipeline)
        return True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(mas, "_drain", _drain_lands_the_model)
        _switch("black-forest-labs/FLUX.1-dev")

    assert loads == []


def test_a_replacement_load_is_not_reported_as_the_requested_model(
    catalog, enabled, tmp_path, backend, monkeypatch
):
    # A user load accepted between two polls supersedes ours. Returning success on "something
    # is resident" would generate on the replacement while naming the requested model.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))

    async def _start(owner, pick, current_subject):
        backend.repo_id = "Qwen/Qwen-Image"
        backend.phase = "ready"

    monkeypatch.setattr(mas, "_start_load", _start)

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 503
    assert "replaced" in excinfo.value.detail["error"]["message"]


def test_the_download_plan_asks_the_engine_that_will_load_the_pick(catalog, tmp_path, monkeypatch):
    # The resident engine can be native sd.cpp while the target loads through diffusers; its
    # planner refuses the pick, and that refusal would read as nothing missing.
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
    pick = mas.resolve_local_media_model("city96/FLUX.1-dev-gguf", task = mas.IMAGE_TASK)
    asked: list = []

    class _Planner:
        def download_plan(self, model_path, **kwargs):
            asked.append(model_path)
            return {"total_bytes": 7}

    monkeypatch.setattr(mas, "_planners_for", lambda owner, p: [_Planner()])

    assert mas._missing_download_bytes(arb.DIFFUSION, pick) == 7
    assert asked == [pick.model_path]


def test_two_bpw_builds_of_one_quant_are_not_confused(catalog, enabled, tmp_path, backend, loads):
    # The backend's published token collapses IQ4_XS-3.53bpw and -3.97bpw to IQ4_XS, so a token
    # comparison would report either as serving the other. The full lister label does not.
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
    backend.repo_id = str(tmp_path)
    backend.gguf_variant = "IQ4_XS"
    backend.model_kind = "gguf"

    _switch("z-IQ4_XS-3.97bpw")

    # The published token cannot separate them, so the resident one is never assumed to be ours.
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


def test_a_slow_resolve_refuses_inside_the_budget(catalog, enabled, monkeypatch):
    # The budget has to cover the scan and the plan, not just the drain and the load: either can
    # outlive the response window on its own.
    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.2)

    def _slow(name, *, task):
        time.sleep(1.0)
        return None

    monkeypatch.setattr(mas, "resolve_local_media_model", _slow)

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 503
    assert excinfo.value.detail["error"]["code"] == "model_loading"


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


def test_a_load_that_lands_is_accepted_even_when_the_token_is_ambiguous(
    catalog, enabled, tmp_path, backend, loads
):
    # The skip check must stay conservative for builds sharing a token, but the load this
    # request started is its own: rejecting it there would 503 the model that just loaded.
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

    _switch("z-IQ4_XS-3.53bpw")

    assert [pick.gguf_filename for _owner, pick in loads] == ["z-IQ4_XS-3.53bpw.gguf"]


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


def test_a_plan_with_unsized_entries_is_refused(catalog, enabled, tmp_path, backend, loads):
    # Both planners coerce an unknown sibling size to zero while keeping the entry, so bytes
    # alone would read a pending multi-GB fetch as nothing to do.
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
    unsized = {"total_bytes": 0, "entries": [{"repo_id": "x", "files": ["a"], "bytes": 0}]}
    backend.download_plan = lambda model_path, **kw: unsized

    with pytest.raises(HTTPException) as excinfo:
        _switch("city96/FLUX.1-dev-gguf")

    assert excinfo.value.status_code == 409
    assert excinfo.value.detail["error"]["code"] == "model_not_downloaded"
    assert loads == []


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

    monkeypatch.setattr(mas, "_planners_for", lambda owner, p: [_Planner()])
    monkeypatch.setattr(mas, "_plan_gpu_ordinal", lambda: None)

    assert mas._missing_download_bytes(arb.DIFFUSION, pick) == 0
    assert seen == [("model.safetensors", "single_file")]


def test_a_local_pipeline_is_complete_without_asking_the_hub(
    catalog, enabled, tmp_path, backend, loads
):
    # The planner asks HfApi about an absolute path and fails, which now reads as unverifiable.
    # A directory on disk is what from_pretrained loads, so it needs no plan at all.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))

    def _explode(model_path, **kwargs):
        raise AssertionError("a local pipeline must not be planned against the Hub")

    backend.download_plan = _explode

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_paths_differing_only_in_case_are_different_models(
    catalog, enabled, tmp_path, backend, loads
):
    # Case-sensitive filesystems allow /models/Foo and /models/foo side by side; folding them
    # would report one as already serving the other.
    for name in ("Foo", "foo"):
        directory = tmp_path / name
        if not directory.exists():
            directory.mkdir()
        (directory / "model_index.json").write_text("{}")
        catalog.append(_info(f"org/{name}", directory, task = mas.IMAGE_TASK))
    upper = mas.resolve_local_media_model("org/Foo", task = mas.IMAGE_TASK)
    lower = mas.resolve_local_media_model("org/foo", task = mas.IMAGE_TASK)
    if upper is None or lower is None or upper.model_path == lower.model_path:
        pytest.skip("this filesystem folds case, so the two models cannot coexist")
    backend.repo_id = lower.model_path

    _switch("org/Foo")

    assert [pick.model_path for _owner, pick in loads] == [upper.model_path]


def test_a_native_prediction_also_verifies_the_diffusers_fallback(catalog, tmp_path, monkeypatch):
    # predict_engine calls sd.cpp available whenever its install is allowed, while activation
    # falls back to diffusers when that install produces nothing runnable.
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
    pick = mas.resolve_local_media_model("city96/FLUX.1-dev-gguf", task = mas.IMAGE_TASK)

    class _Planner:
        def __init__(self, missing):
            self.missing = missing

        def download_plan(self, model_path, **kwargs):
            return {"total_bytes": self.missing, "entries": []}

    # The predicted engine sees nothing missing; the fallback's companion set is incomplete.
    monkeypatch.setattr(mas, "_planners_for", lambda owner, p: [_Planner(0), _Planner(9_000)])

    assert mas._missing_download_bytes(arb.DIFFUSION, pick) == 9_000


def test_a_partial_download_is_not_an_available_model(catalog, tmp_path):
    # A cancelled pull still lists; loading it fails predictably, and the 404 listing should not
    # advertise it either.
    pipeline = tmp_path / "half"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    row = _info("org/half", pipeline, task = mas.IMAGE_TASK)
    row.partial = True
    catalog.append(row)

    assert mas.resolve_local_media_model("org/half", task = mas.IMAGE_TASK) is None
    assert mas.available_media_model_ids(mas.IMAGE_TASK) == []


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


def test_an_edit_only_model_is_refused_before_anything_is_evicted(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # The catalog tags edit families text-to-image, so the load would finish and only then be
    # refused for lacking txt2img, with the previously useful model already gone.
    pipeline = tmp_path / "kontext"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-Kontext-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = "Qwen/Qwen-Image"
    monkeypatch.setattr(mas, "_is_edit_only", lambda pick: True)

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-Kontext-dev")

    assert excinfo.value.status_code == 400
    assert loads == []


def test_the_native_fallback_is_only_verified_when_a_binary_must_be_installed(monkeypatch):
    # With a runnable sd.cpp binary the load stays native, so demanding the diffusers shards
    # would refuse a model the selected engine can serve.
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    pick = mas.MediaModelPick("x/y", "x/y", "y-Q4_K_M.gguf", "gguf", quant = "Q4_K_M")
    monkeypatch.setattr(mas, "_backend_for", lambda owner: object())
    router = __import__("core.inference.diffusion_engine_router", fromlist = ["x"])
    families = __import__("core.inference.diffusion_families", fromlist = ["x"])
    monkeypatch.setattr(families, "detect_family_for_pick", lambda *a, **k: object())
    monkeypatch.setattr(router, "predict_engine", lambda fam, model_kind = None: ENGINE_SD_CPP)
    monkeypatch.setattr(router, "engine_for", lambda name: name)

    monkeypatch.setattr(router, "native_binary_installed", lambda: True)
    assert mas._planners_for(arb.DIFFUSION, pick) == [ENGINE_SD_CPP]

    monkeypatch.setattr(router, "native_binary_installed", lambda: False)
    assert len(mas._planners_for(arb.DIFFUSION, pick)) == 2


def test_a_directory_the_load_route_would_reject_is_not_advertised(catalog, tmp_path):
    # Several checkpoints and no model_index.json is ambiguous; both routes reject it rather
    # than choose, so offering it would only cost a failed switch.
    ambiguous = tmp_path / "two-checkpoints"
    ambiguous.mkdir()
    for name in ("a.safetensors", "b.safetensors"):
        (ambiguous / name).write_bytes(b"")
    catalog.append(_info("org/ambiguous", ambiguous, task = mas.IMAGE_TASK))

    assert mas.resolve_local_media_model("org/ambiguous", task = mas.IMAGE_TASK) is None
    assert mas.available_media_model_ids(mas.IMAGE_TASK) == []


def test_an_incompatible_plan_is_refused_before_the_resident_model_is_torn_down(
    catalog, enabled, tmp_path, backend, loads
):
    # A FLUX.2 GGUF paired with a different-size base is fully cached and still unloadable; the
    # route's cheap validation misses it, so only the background loader would find out.
    repo, _snapshot = _hf_cache_repo(tmp_path, "city96/FLUX.2-gguf", files = ["f-Q4_K_M.gguf"])
    catalog.append(
        _info(
            "city96/FLUX.2-gguf",
            repo,
            task = mas.IMAGE_TASK,
            model_format = "gguf",
            source = "hf_cache",
        )
    )
    backend.download_plan = lambda model_path, **kw: {
        "total_bytes": 0,
        "entries": [],
        "incompatible_reason": "this GGUF needs the 32B base",
    }

    with pytest.raises(HTTPException) as excinfo:
        _switch("city96/FLUX.2-gguf")

    assert excinfo.value.status_code == 409
    assert loads == []


def test_load_setup_that_stalls_returns_inside_the_budget(
    catalog, enabled, tmp_path, backend, monkeypatch
):
    # Preflight and a first-run sd.cpp install both run before begin_load registers, while the
    # admission gate is held, so an unbounded await blocks the backend as well as the caller.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.3)

    async def _stalls(owner, pick, current_subject):
        await asyncio.sleep(5)

    monkeypatch.setattr(mas, "_start_load", _stalls)

    began = time.monotonic()
    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

    assert excinfo.value.status_code == 503
    assert time.monotonic() - began < 3.0


def test_a_modular_pipeline_index_is_discoverable(catalog, tmp_path):
    # A dense MiniMax-H3 directory carries modular_model_index.json, which the video loader
    # opens; rejecting it here 404'd every named request for a fully downloaded model.
    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text("{}")
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))

    assert mas.resolve_local_media_model("MiniMaxAI/MiniMax-H3", task = mas.VIDEO_TASK) is not None


def test_a_resident_h3_reference_partition_does_not_answer_a_plain_request(
    catalog, enabled, tmp_path, backend, loads
):
    # An auto-load of this name takes the default keyframe denoiser, so a resident ref2va is a
    # different build; serving it accepts a generation that then fails for missing references.
    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text("{}")
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))
    backend.repo_id = str(modular)
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


def test_the_resident_shortcut_also_checks_the_h3_partition(
    catalog, enabled, tmp_path, backend, loads
):
    # The pre-index shortcut returns before _resident_is_pick runs, so it needs the same
    # partition test or a resident ref2va answers a plain request.
    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text("{}")
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))
    backend.repo_id = "MiniMaxAI/MiniMax-H3"
    backend.h3_task = "ref2va"

    _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert [pick.model_id for _owner, pick in loads] == ["MiniMaxAI/MiniMax-H3"]


def test_setup_keeps_the_gate_and_lock_after_the_caller_gives_up(
    catalog, enabled, tmp_path, backend, monkeypatch
):
    # Shielding alone let the caller unwind both contexts while setup was still before
    # begin_load, so a newly admitted generation could be cut short by the orphaned switch.
    import core.inference.media_keepwarm as keepwarm

    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    monkeypatch.setattr(mas, "_SWITCH_BUDGET_S", 0.3)
    started = asyncio.Event()
    release = asyncio.Event()

    async def _slow_setup(owner, pick, current_subject):
        started.set()
        await release.wait()

    monkeypatch.setattr(mas, "_start_load", _slow_setup)

    async def _drive():
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
        assert mas._switch_lock(arb.DIFFUSION).locked()
        release.set()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if not keepwarm._TRACKERS[arb.DIFFUSION].gate.locked():
                break
        assert not keepwarm._TRACKERS[arb.DIFFUSION].gate.locked()
        assert not mas._switch_lock(arb.DIFFUSION).locked()

    asyncio.run(_drive())


def test_an_expired_budget_refuses_instead_of_crashing(
    catalog, enabled, tmp_path, backend, monkeypatch
):
    # _bounded receives a Future from shield(), which cancels rather than closes; calling
    # close() on it raised AttributeError instead of the intended retryable 503.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))

    async def _expired(coro):
        return await mas._bounded(coro, time.monotonic() - 1, kind = "image", openai_errors = True)

    async def _drive():
        with pytest.raises(HTTPException) as excinfo:
            await _expired(asyncio.shield(asyncio.ensure_future(asyncio.sleep(0))))
        assert excinfo.value.status_code == 503

    asyncio.run(_drive())


def test_a_request_parked_on_the_held_gate_does_not_abort_the_switch(
    catalog, enabled, tmp_path, backend, loads, monkeypatch
):
    # A newcomer arriving while the gated task owns the gate is counted pending and then blocks
    # on that gate, so counting it aborted an otherwise idle switch.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    (pipeline / "model_index.json").write_text("{}")
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = "Qwen/Qwen-Image"
    tracker = mk._TRACKERS[arb.DIFFUSION]
    monkeypatch.setattr(tracker, "_pending", 1)

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_paths_differing_only_in_case_are_not_merged_by_the_ambiguity_scan(catalog, tmp_path):
    # Lowercasing the path merged two directories whose builds publish the same quant token,
    # marking both ambiguous so the resident one could never be reused.
    for name in ("Foo", "foo"):
        directory = tmp_path / name
        if not directory.exists():
            directory.mkdir()
        (directory / f"{name}-Q4_K_M.gguf").write_bytes(b"")
        catalog.append(_info(f"org/{name}", directory, task = mas.IMAGE_TASK, model_format = "gguf"))
    upper = mas.resolve_local_media_model("org/Foo", task = mas.IMAGE_TASK)
    lower = mas.resolve_local_media_model("org/foo", task = mas.IMAGE_TASK)
    if upper is None or lower is None or upper.model_path == lower.model_path:
        pytest.skip("this filesystem folds case, so the two directories cannot coexist")

    assert not upper.ambiguous and not lower.ambiguous


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


def test_a_local_video_pipeline_is_still_planned(catalog, enabled, tmp_path, backend, loads):
    # A local MiniMax-H3 modular pipeline substitutes a hosted quantized conditioner during
    # assembly, so the image shortcut's "on disk means complete" does not hold for video.
    modular = tmp_path / "h3"
    modular.mkdir()
    (modular / "modular_model_index.json").write_text("{}")
    catalog.append(_info("MiniMaxAI/MiniMax-H3", modular, task = mas.VIDEO_TASK))
    backend.missing_bytes = 27_000_000_000

    with pytest.raises(HTTPException) as excinfo:
        _switch("MiniMaxAI/MiniMax-H3", owner = arb.VIDEO, openai_errors = False)

    assert excinfo.value.status_code == 409
    assert loads == []


def test_the_video_load_route_records_provenance_without_raising(monkeypatch):
    # The provenance call is made positionally from both load routes, so a signature that drifts
    # from them 500s every load after the background work has already been accepted.
    import core.inference.video as video_module
    from routes.video import router as video_router

    class _Backend:
        def validate_load_request(self, *a, **k):
            return object()

        def begin_load(self, *a, **k):
            return {"loaded": False, "repo_id": None}

    monkeypatch.setattr(video_module, "get_video_backend", lambda: _Backend())
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

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(video_router, prefix = "/api/inference")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    resp = TestClient(app).post(
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


def test_the_images_route_switches_before_it_checks_what_is_loaded(monkeypatch):
    import core.inference.diffusion as diffusion_module
    import core.inference.image_gallery as gallery_module
    from routes.inference import router

    calls: list = []

    async def _switch_stub(requested_model, *, owner, current_subject, openai_errors):
        calls.append((requested_model, owner, openai_errors))

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

    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(router, prefix = "/v1")
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    resp = TestClient(app).post(
        "/v1/images/generations",
        json = {"prompt": "p", "size": "256x256", "model": "black-forest-labs/FLUX.1-dev"},
    )

    assert resp.status_code == 200
    assert calls == [("black-forest-labs/FLUX.1-dev", arb.DIFFUSION, True)]
