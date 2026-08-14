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
    ):
        self.repo_id = repo_id
        self.gguf_variant = gguf_variant
        self.model_kind = model_kind
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
    monkeypatch.setattr(mas, "_planner_for", lambda owner, pick: fake)
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
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = "black-forest-labs/FLUX.1-dev"

    _switch("black-forest-labs/FLUX.1-dev")

    assert loads == []


def test_a_model_loaded_by_path_still_counts_as_serving(catalog, enabled, tmp_path, backend, loads):
    # A model this module loaded reports the local path it was given, not the repo id.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = str(pipeline)

    _switch("black-forest-labs/FLUX.1-dev")

    assert loads == []


def test_a_different_model_is_loaded_before_the_request_proceeds(
    catalog, enabled, tmp_path, backend, loads
):
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
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
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.missing_bytes = 4_300_000_000

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

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
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    backend.repo_id = "Qwen/Qwen-Image"
    monkeypatch.setattr(mas, "_DRAIN_WAIT_S", 0.5)
    # Two tracked requests in flight, and both parked on the switch: only one holds the lock.
    monkeypatch.setattr(mk, "other_request_count", lambda owner, current_request_counted = False: 1)
    monkeypatch.setattr(mas, "_waiter_count", lambda owner: 2)

    _switch("black-forest-labs/FLUX.1-dev")

    assert [pick.model_id for _owner, pick in loads] == ["black-forest-labs/FLUX.1-dev"]


def test_the_drain_and_the_load_share_one_budget(catalog, enabled, tmp_path, backend, monkeypatch):
    # Separate budgets added up past the ~100s tunnel window, so a slow switch lost the socket
    # instead of returning the retryable 503 the bounds exist to produce.
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
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
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    pick = mas.resolve_local_media_model("black-forest-labs/FLUX.1-dev", task = mas.IMAGE_TASK)
    asked: list = []

    class _Planner:
        def download_plan(self, model_path, **kwargs):
            asked.append(model_path)
            return {"total_bytes": 7}

    monkeypatch.setattr(mas, "_planner_for", lambda owner, p: _Planner())

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
    clip = tmp_path / "wan"
    clip.mkdir()
    catalog.append(_info("unsloth/Wan2.2", clip, task = mas.VIDEO_TASK))
    monkeypatch.setattr(
        backend, "download_plan", lambda model_path, **kw: {"total_bytes": 0, "plan_failed": True}
    )

    with pytest.raises(HTTPException) as excinfo:
        _switch("unsloth/Wan2.2", owner = arb.VIDEO, openai_errors = False)

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
    pipeline = tmp_path / "my-flux"
    pipeline.mkdir()
    catalog.append(_info("black-forest-labs/FLUX.1-dev", pipeline, task = mas.IMAGE_TASK))
    monkeypatch_plan = {"total_bytes": 0, "entries": [{"repo_id": "x", "files": ["a"], "bytes": 0}]}
    backend.download_plan = lambda model_path, **kw: monkeypatch_plan

    with pytest.raises(HTTPException) as excinfo:
        _switch("black-forest-labs/FLUX.1-dev")

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

    monkeypatch.setattr(mas, "_planner_for", lambda owner, p: _Planner())
    monkeypatch.setattr(mas, "_plan_gpu_ordinal", lambda: None)

    assert mas._missing_download_bytes(arb.DIFFUSION, pick) == 0
    assert seen == [("model.safetensors", "single_file")]


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
