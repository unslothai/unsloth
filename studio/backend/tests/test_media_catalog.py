# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.inference import diffusion_families, media_catalog, video_families
from core.inference.diffusion_inference_info import family_inference_infos
from hub.services.models import account_access
from utils.api_errors import install_api_error_handlers


@pytest.fixture
def selectable(monkeypatch):
    # Selectability needs diffusers; the registry decides the rows here.
    hidden = set()

    def pick(fam):
        return fam.name not in hidden

    monkeypatch.setattr(diffusion_families, "family_selectable", pick)
    monkeypatch.setattr(media_catalog, "family_selectable", pick)
    return hidden


def _rows():
    return [row.model_dump() for row in media_catalog.media_catalog_payload().models]


def test_every_selectable_family_is_a_row_sized_from_its_registry(selectable):
    rows = _rows()
    bf16 = {i["family"]: i["estimated_resident_gb"]["bf16"] for i in family_inference_infos()}
    images = [row for row in rows if row["kind"] == "image"]
    assert [row["family"] for row in images] == [f.name for f in diffusion_families._FAMILIES]
    for row in images:
        expected = bf16.get(row["family"])
        assert row["size_bytes"] == (None if expected is None else round(expected * 1e9))
        assert row["low_vram"] is False
    videos = [row for row in rows if row["kind"] == "video" and not row["low_vram"]]
    assert [row["family"] for row in videos] == [f.name for f in video_families._FAMILIES]
    sizes = {f.name: f.bf16_components_gb for f in video_families._FAMILIES}
    for row in videos:
        assert row["size_bytes"] == round(sum(sizes[row["family"]]) * 1e9)
    assert {"id", "kind", "family", "display_name", "size_bytes", "low_vram"} == set(rows[0])
    # Images first, then videos, each in registry order.
    kinds = [row["kind"] for row in rows]
    assert kinds == sorted(kinds)


def test_video_gguf_repos_are_low_vram_rows_without_a_size(selectable):
    ggufs = [row for row in _rows() if row["low_vram"]]
    expected = [(f.gguf_repo, f.name) for f in video_families._FAMILIES if f.gguf_repo]
    assert [(row["id"], row["family"]) for row in ggufs] == expected
    assert {row["size_bytes"] for row in ggufs} == {None}
    assert {row["kind"] for row in ggufs} == {"video"}


def test_mirror_ids_resolve_back_to_their_family(selectable):
    rows = _rows()
    by_id = {row["id"]: row for row in rows}
    assert by_id["unsloth/FLUX.1-schnell"]["family"] == "flux.1"
    assert by_id["unsloth/Wan2.2-TI2V-5B-GGUF"]["display_name"] == "Wan2.2-TI2V-5B-GGUF"
    for row in rows:
        detect = diffusion_families.detect_family
        if row["kind"] == "video":
            detect = video_families.detect_video_family
        assert detect(row["id"]).name == row["family"], row["id"]


def test_an_unselectable_family_is_left_out(selectable):
    selectable.update({"flux.1", "wan2.2-ti2v-5b"})
    families = {row["family"] for row in _rows()}
    assert "flux.1" not in families and "wan2.2-ti2v-5b" not in families
    assert "z-image" in families and "ltx-2" in families


@pytest.fixture
def client():
    from routes import inference

    app = FastAPI()
    install_api_error_handlers(app)
    app.dependency_overrides[get_current_subject] = lambda: "tester"
    app.include_router(inference.router, prefix = "/api/inference")
    app.include_router(inference.studio_router, prefix = "/api/inference")
    app.include_router(inference.router, prefix = "/v1")
    yield TestClient(app)


PAYLOAD = media_catalog.MediaCatalogResponse(
    models = [
        {"id": "unsloth/Z-Image-Turbo", "kind": "image", "family": "z-image", "display_name": "Z"},
        {"id": "unsloth/FLUX.2-dev", "kind": "image", "family": "flux.2-dev", "display_name": "F"},
    ]
)


def test_the_route_serves_every_row_to_the_owner(client, monkeypatch):
    monkeypatch.setattr(media_catalog, "media_catalog_payload", lambda: PAYLOAD)
    response = client.get("/api/inference/media/catalog")
    assert response.status_code == 200
    assert media_catalog.MediaCatalogResponse.model_validate(response.json()) == PAYLOAD
    assert client.get("/v1/media/catalog").status_code == 404


def test_a_managed_account_sees_only_the_rows_it_may_use(client, monkeypatch):
    monkeypatch.setattr(media_catalog, "media_catalog_payload", lambda: PAYLOAD)
    seen = []
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "model_grants", lambda: set())
    monkeypatch.setattr(account_access, "_warm_public_repos", lambda *a: None)
    monkeypatch.setattr(
        account_access,
        "model_visible",
        lambda reference, **_k: seen.append(reference) or "FLUX" not in reference,
    )
    response = client.get("/api/inference/media/catalog")
    assert [row["id"] for row in response.json()["models"]] == ["unsloth/Z-Image-Turbo"]
    assert seen == ["unsloth/Z-Image-Turbo", "unsloth/FLUX.2-dev"]
    assert len(PAYLOAD.models) == 2


def test_a_builder_failure_answers_503_without_its_text(client, monkeypatch):
    def build():
        raise RuntimeError("/home/me/.cache/huggingface exploded")

    monkeypatch.setattr(media_catalog, "media_catalog_payload", build)
    response = client.get("/api/inference/media/catalog")
    assert response.status_code == 503
    assert "Unsloth Studio could not list image and video models right now." in response.text
    assert "exploded" not in response.text
