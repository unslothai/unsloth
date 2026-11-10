# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A managed account's image load clears the base it resolves, the same check generate runs on it (#16178)."""

from __future__ import annotations

import os
import sys

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import core.inference.diffusion as diffusion_module  # noqa: E402
from hub.services.models import account_access  # noqa: E402
from test_diffusion_routes import client  # noqa: E402,F401 -- the shared fake-backend client fixture

KREA_GGUF = {
    "model_path": "realrebelai/KREA-2_GGUFs",
    "gguf_filename": "TURBO/Krea-2-Turbo-Q8_0.gguf",
}
PUBLIC = {"realrebelai/KREA-2_GGUFs", "unsloth/Qwen-Image-2512-GGUF", "Qwen/Qwen-Image"}


@pytest.fixture
def managed(monkeypatch):
    grants: set[str] = set()
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    monkeypatch.setattr(account_access, "require_media_references", lambda *a, **k: None)
    monkeypatch.setattr(account_access, "model_grants", lambda: set(grants))
    monkeypatch.setattr(account_access, "model_allowlist", lambda: None)
    monkeypatch.setattr(
        account_access, "repo_is_public", lambda repo_id, *a, **k: repo_id in PUBLIC
    )
    # The GGUF repo's card names the gated base, as realrebelai/KREA-2_GGUFs does.
    monkeypatch.setattr(
        diffusion_module,
        "_hf_base_model",
        lambda repo_id, token: {
            "realrebelai/KREA-2_GGUFs": "krea/Krea-2-Turbo",
            "unsloth/Qwen-Image-2512-GGUF": "Qwen/Qwen-Image",
        }.get(repo_id),
    )
    return grants


def _generate(client):
    return client.post("/api/inference/images/generate", json = {"prompt": "p"})


def test_an_ungranted_gated_base_is_refused_at_load_before_anything_loads(client, managed):
    backend = diffusion_module.get_diffusion_backend()
    resp = client.post("/api/inference/images/load", json = KREA_GGUF)
    assert resp.status_code == 403, resp.text
    assert "krea/Krea-2-Turbo" in resp.json()["detail"]
    assert backend.loaded is False
    assert not hasattr(backend, "last_load_kwargs")


def test_a_load_the_account_may_make_can_also_generate(client, managed, monkeypatch):
    managed.add(account_access._grant_key("krea/Krea-2-Turbo", "model"))
    backend = diffusion_module.get_diffusion_backend()
    real_begin = backend.begin_load
    monkeypatch.setattr(
        backend,
        "begin_load",
        lambda model_path, **kw: {**real_begin(model_path, **kw), "base_repo": "krea/Krea-2-Turbo"},
    )
    monkeypatch.setattr(
        backend,
        "status",
        lambda: {
            "loaded": True,
            "repo_id": KREA_GGUF["model_path"],
            "family": "krea-2",
            "base_repo": "krea/Krea-2-Turbo",
        },
    )
    assert client.post("/api/inference/images/load", json = KREA_GGUF).status_code == 200
    assert _generate(client).status_code == 200


def test_the_same_resident_is_refused_by_generate_without_the_grant(client, managed, monkeypatch):
    """The policy the load now applies early: generate already refuses this base for the account."""
    backend = diffusion_module.get_diffusion_backend()
    backend.loaded = True
    monkeypatch.setattr(
        backend,
        "status",
        lambda: {
            "loaded": True,
            "repo_id": KREA_GGUF["model_path"],
            "family": "krea-2",
            "base_repo": "krea/Krea-2-Turbo",
        },
    )
    assert _generate(client).status_code == 404


def test_a_public_base_still_loads_for_a_managed_account(client, managed):
    resp = client.post(
        "/api/inference/images/load",
        json = {
            "model_path": "unsloth/Qwen-Image-2512-GGUF",
            "gguf_filename": "qwen-image-2512-Q4_K_M.gguf",
        },
    )
    assert resp.status_code == 200, resp.text


def test_the_installation_owner_is_not_asked_for_a_grant(client, monkeypatch):
    monkeypatch.setattr(account_access, "managed_account", lambda: False)
    monkeypatch.setattr(
        diffusion_module,
        "_hf_base_model",
        lambda *a, **k: pytest.fail("the owner path must not resolve the base for an access check"),
    )
    resp = client.post("/api/inference/images/load", json = KREA_GGUF)
    assert resp.status_code == 200, resp.text


def _req(**kw):
    import types
    return types.SimpleNamespace(
        model_path = kw.get("model_path", "org/some-GGUF"),
        base_repo = kw.get("base_repo"),
        hf_token = None,
        gguf_filename = kw.get("gguf_filename", "m-Q4_K_M.gguf"),
    )


@pytest.fixture
def allowed(monkeypatch):
    from fastapi import HTTPException

    ok: set[str] = set()
    monkeypatch.setattr(account_access, "managed_account", lambda: True)

    def _check(repo):
        if repo not in ok:
            raise HTTPException(status_code = 404, detail = "Not found")

    monkeypatch.setattr(account_access, "require_model_access", _check)
    return ok


def _family(name):
    from core.inference.diffusion_families import _FAMILIES
    return next(f for f in _FAMILIES if f.name == name)


def test_an_unknown_engine_is_not_checked_against_a_guessed_base(allowed, monkeypatch):
    from routes import inference as routes
    monkeypatch.setattr(
        diffusion_module, "_resolve_base_repo", lambda *a, **k: pytest.fail("no engine, no guess")
    )
    routes._require_resolved_image_base_access(_req(), _family("krea-2"), "gguf", None)


def test_mlx_checks_the_base_its_own_resolver_publishes(allowed, monkeypatch):
    from fastapi import HTTPException
    from routes import inference as routes
    import core.inference.mlx_diffusion_backend as mlx

    fam = _family("z-image")
    monkeypatch.setattr(mlx, "resolve_variant_base", lambda *a, **k: "Tongyi-MAI/Z-Image")
    allowed.add("Tongyi-MAI/Z-Image")
    routes._require_resolved_image_base_access(_req(), fam, "gguf", "mlx")
    allowed.clear()
    with pytest.raises(HTTPException) as exc:
        routes._require_resolved_image_base_access(_req(), fam, "gguf", "mlx")
    assert exc.value.status_code == 403 and "Tongyi-MAI/Z-Image" in exc.value.detail


def test_sd_cpp_grid_family_passes_when_any_base_it_could_publish_is_allowed(allowed, monkeypatch):
    from fastapi import HTTPException
    from routes import inference as routes

    fam = _family("qwen-image-2.1")
    monkeypatch.setattr(
        diffusion_module, "_resolve_base_repo", lambda *a, **k: "Qwen/Qwen-Image-2.1-Turbo"
    )
    allowed.add("Qwen/Qwen-Image-2.1-Turbo")
    routes._require_resolved_image_base_access(_req(), fam, "gguf", "sd_cpp")
    allowed.clear()
    with pytest.raises(HTTPException) as exc:
        routes._require_resolved_image_base_access(_req(), fam, "gguf", "sd_cpp")
    assert exc.value.status_code == 403
