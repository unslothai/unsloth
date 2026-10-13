# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The disk-only requests used by On Device retain authorization and completeness checks."""

import asyncio
from collections import OrderedDict
import struct
import time
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from auth.authentication import authenticated_via_api_key, get_current_subject
from hub.dependencies import get_request_hf_token
from hub.routes import inventory
from hub.services.models import gguf_variants as GV
from hub.utils import download_manifest, hf_tokens, inventory_scan
from routes import models
from utils import hf_cache_settings

REPO = "Org/OnDevice-GGUF"
QUANT = "Q4_K_M"
MAIN = f"model-{QUANT}.gguf"
TOKEN = "hf_on_device_test_owner"


@pytest.fixture
def cached_model(monkeypatch, tmp_path):
    cache = tmp_path / "hub"
    monkeypatch.setitem(hf_cache_settings._EXPLICIT_CACHE_ENV, "HF_HUB_CACHE", str(cache))
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [cache])
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(cache))
    monkeypatch.setattr(GV, "_VARIANT_REQUIREMENT_LAST_KNOWN", OrderedDict())
    monkeypatch.setattr(GV, "_VARIANT_REQUIREMENT_FORGOTTEN", False)
    hf_tokens.reset_repo_access_cache()
    monkeypatch.setattr(hf_tokens, "_hub_offline", lambda: False)
    monkeypatch.setattr(hf_tokens, "_host_hf_credentials", lambda: (True, [TOKEN]))
    monkeypatch.setattr(hf_tokens, "_no_other_credential_ever_held", lambda _: True)
    monkeypatch.setattr(hf_tokens, "_recorded_request_token_repos", lambda: {})
    monkeypatch.setattr(hf_tokens, "_provenance_record_is_missing", lambda *_: False)
    calls = []

    def unreachable(*args, **kwargs):
        calls.append(args)
        raise ConnectionError("Hugging Face unavailable")

    monkeypatch.setattr(
        "huggingface_hub.utils.get_session", lambda: SimpleNamespace(get = unreachable)
    )
    monkeypatch.setattr("huggingface_hub.HfApi.model_info", unreachable)
    monkeypatch.setattr("huggingface_hub.model_info", unreachable)
    monkeypatch.setattr(GV, "_gguf_all_variant_requirements", unreachable)
    monkeypatch.setattr(GV, "list_gguf_variants", unreachable)
    repo = cache / "models--Org--OnDevice-GGUF"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "refs/main").write_text(snapshot.name)
    (snapshot / MAIN).write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + b"\0" * 232)

    def plan(*files):
        assert download_manifest.write_manifest(
            "model",
            REPO,
            QUANT,
            [download_manifest.ExpectedFile(name, size) for name, size in files],
            hub_cache = cache,
        )
        inventory_scan.invalidate_hf_cache_scans()

    plan((MAIN, 256))
    yield SimpleNamespace(repo = repo, snapshot = snapshot, plan = plan, calls = calls)
    hf_tokens.reset_repo_access_cache()
    inventory_scan.invalidate_hf_cache_scans()


def query(
    model,
    prefix,
    *,
    token = TOKEN,
    local_path = True,
    offline = True,
):
    app = FastAPI()
    app.include_router(inventory.router if prefix == "/api/hub" else models.router, prefix = prefix)
    app.dependency_overrides[get_current_subject] = lambda: "test-owner"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    app.dependency_overrides[get_request_hf_token] = lambda: token
    params = {
        "repo_id": REPO,
        "prefer_local_cache": str(offline).lower(),
        "offline": str(offline).lower(),
        "include_cache_locations": "true",
    }
    if local_path:
        params["local_path"] = str(model.snapshot)
    with TestClient(app) as client:
        return client.get(f"{prefix}/gguf-variants", params = params)


@pytest.mark.parametrize("prefix", ["/api/hub", "/api/models"])
def test_cached_quant_resolves_without_hub_even_when_host_reports_online(cached_model, prefix):
    response = query(cached_model, prefix)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["dependencies_resolved"] is True
    assert [v["quant"] for v in data["variants"] if v["downloaded"]] == [QUANT]
    assert cached_model.calls == []


@pytest.mark.parametrize("missing", ["shard", "companion"])
@pytest.mark.parametrize("prefix", ["/api/hub", "/api/models"])
def test_local_discovery_keeps_incomplete_quants_non_loadable(cached_model, prefix, missing):
    if missing == "companion":
        cached_model.plan((MAIN, 256), ("mmproj-F16.gguf", 128))
    else:
        shard = "model-Q4_K_M-00001-of-00002.gguf"
        (cached_model.snapshot / MAIN).rename(cached_model.snapshot / shard)
        cached_model.plan((shard, 256), ("model-Q4_K_M-00002-of-00002.gguf", 256))
    response = query(cached_model, prefix)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["dependencies_resolved"] is False
    assert all(not v["downloaded"] or v["partial"] for v in data["variants"])
    assert cached_model.calls == []


@pytest.mark.parametrize("token", ["hf_another_credential", False])
@pytest.mark.parametrize("local_path", [False, True])
def test_local_discovery_withholds_private_cache_from_another_identity(
    cached_model, token, local_path
):
    (cached_model.snapshot / "Q8_0").mkdir()
    response = query(cached_model, "/api/hub", token = token, local_path = local_path)
    assert response.status_code == 404 or (
        response.status_code == 200 and response.json()["variants"] == []
    )
    assert cached_model.calls == []


def test_offline_managed_account_cannot_probe_for_a_missing_grant(cached_model, monkeypatch):
    access = GV.account_access
    monkeypatch.setattr(access, "managed_account", lambda: True)
    monkeypatch.setattr(access, "account_hf_token", lambda token: token)

    def denied(*args, **kwargs):
        raise HTTPException(403, "Model is not granted to this account")

    def authorize(*args):
        cached_model.calls.append("authorize_download")
        denied()

    monkeypatch.setattr(access, "require_model_access", denied)
    monkeypatch.setattr(access, "authorize_download", authorize)
    with pytest.raises(HTTPException) as error:
        asyncio.run(
            GV.get_gguf_variants_response(
                REPO, hf_token = TOKEN, offline = True, prefer_local_cache = True
            )
        )
    assert error.value.status_code == 403
    assert cached_model.calls == []


@pytest.mark.parametrize("prefix", ["/api/hub", "/api/models"])
@pytest.mark.parametrize("access", ["grant", "public-proof", "none"])
def test_managed_account_authorizes_from_disk_without_public_repo_probe(
    cached_model, monkeypatch, access, prefix
):
    policy = GV.account_access
    monkeypatch.setattr(policy, "managed_account", lambda: True)
    monkeypatch.setattr(
        policy, "model_grants", lambda: {f"model:{REPO.lower()}"} if access == "grant" else set()
    )
    monkeypatch.setattr(policy, "_public_repos", {})
    key = policy._public_key(REPO, "model")
    name = f"{key[0]}|model:{REPO.lower()}"
    monkeypatch.setattr(
        policy,
        "_load_public_verdicts",
        lambda: {name: time.time()} if access == "public-proof" else {},
    )

    def no_probe(*args, **kwargs):
        cached_model.calls.append("public_repo_probe")
        raise ConnectionError("offline")

    monkeypatch.setattr(policy, "_hub_public_answer", no_probe)
    response = query(cached_model, prefix)
    assert response.status_code == (404 if access == "none" else 200), response.text
    if response.status_code == 200:
        assert response.json()["dependencies_resolved"] is True
    assert cached_model.calls == []


def test_an_explicit_non_hub_folder_keeps_its_local_listing(cached_model, tmp_path):
    local = tmp_path / "local-model"
    local.mkdir()
    (local / MAIN).write_bytes((cached_model.snapshot / MAIN).read_bytes())
    response = asyncio.run(
        GV.get_gguf_variants_response(REPO, local_path = str(local), offline = True, hf_token = False)
    )
    assert [v.quant for v in response.variants if v.downloaded] == [QUANT]
    assert cached_model.calls == []


@pytest.mark.parametrize("prefix", ["/api/hub", "/api/models"])
def test_online_remote_discovery_still_lists_published_quants(cached_model, monkeypatch, prefix):
    from hub.utils.gguf import GgufVariantInfo

    listed = []

    def remote(repo_id, **kwargs):
        listed.append(repo_id)
        return [GgufVariantInfo(filename = MAIN, quant = QUANT, size_bytes = 256)], False, []

    monkeypatch.setattr(GV, "list_gguf_variants", remote)
    monkeypatch.setattr(GV, "_gguf_all_variant_requirements", lambda *_a, **_kw: {})
    response = query(cached_model, prefix, local_path = False, offline = False)
    assert response.status_code == 200, response.text
    assert listed == [REPO]
    assert response.json()["variants"]
