# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A model folder granted to the jails whole must not be (or hold) an HF home."""

import os

import pytest

from core.inference import os_sandbox, tool_path_approval


@pytest.fixture
def roots(monkeypatch, tmp_path):
    def register(*folders, hf_homes = ()):
        monkeypatch.setattr(
            tool_path_approval, "_scan_folder_roots", lambda: tuple(map(str, folders))
        )
        monkeypatch.setattr(
            "utils.paths.storage_roots.well_known_model_dirs", lambda: (), raising = False
        )
        monkeypatch.setattr(
            "utils.hf_cache_settings.known_hf_cache_homes", lambda: list(hf_homes), raising = False
        )
        return os_sandbox.model_library_roots()

    return register


def test_a_registered_hf_home_is_not_granted(roots, tmp_path):
    hf_home = tmp_path / "hf"
    (hf_home / "hub").mkdir(parents = True)
    (hf_home / "token").write_text("hf_FAKE")
    assert roots(hf_home, hf_homes = [hf_home]) == ()


def test_a_folder_holding_an_hf_home_is_not_granted(roots, tmp_path):
    cache = tmp_path / "cache"
    hf_home = cache / "huggingface"
    hf_home.mkdir(parents = True)
    (hf_home / "stored_tokens").write_text("[]")
    assert roots(cache, hf_homes = [hf_home]) == ()


def test_a_folder_holding_a_cache_only_home_is_still_granted(roots, tmp_path):
    models = tmp_path / "models"
    (models / "hf" / "hub").mkdir(parents = True)
    assert roots(models, hf_homes = [models / "hf"]) == (os.path.realpath(models),)


@pytest.mark.parametrize("name", ["token", "stored_tokens", ".env", ".git-credentials", ".netrc"])
def test_a_folder_with_a_credential_file_at_its_top_is_not_granted(roots, tmp_path, name):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / name).write_text("FAKE")
    assert roots(folder) == ()


def test_a_folder_holding_the_hf_token_path_is_not_granted(roots, tmp_path, monkeypatch):
    folder = tmp_path / "models"
    (folder / "secrets").mkdir(parents = True)
    (folder / "secrets" / "hf-token").write_text("hf_FAKE")
    monkeypatch.setenv("HF_TOKEN_PATH", str(folder / "secrets" / "hf-token"))
    assert roots(folder) == ()


def _native(*parts):
    return os.path.join(os.sep, *parts)


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_an_hf_home_spelled_in_another_case_is_still_inside(monkeypatch, platform):
    # macOS realpath keeps the spelling it was given, so /users/alice/models must still hold /Users/Alice/....
    monkeypatch.setattr(os_sandbox.sys, "platform", platform)
    home, library = (
        _native("Users", "Alice", "models", "huggingface"),
        _native("users", "alice", "models"),
    )
    assert os_sandbox._is_within(home, library)


def test_case_still_counts_on_linux(monkeypatch):
    monkeypatch.setattr(os_sandbox.sys, "platform", "linux")
    home, library = (
        _native("Users", "Alice", "models", "huggingface"),
        _native("users", "alice", "models"),
    )
    assert not os_sandbox._is_within(home, library)


def test_an_ordinary_model_folder_and_the_hub_under_an_hf_home_are_granted(roots, tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    (models / "model.gguf").write_text("x")
    hub = tmp_path / "hf" / "hub"
    hub.mkdir(parents = True)
    granted = roots(models, hub, hf_homes = [tmp_path / "hf"])
    assert granted == (os.path.realpath(models), os.path.realpath(hub))


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_the_capability_text_says_loopback_is_reachable(monkeypatch, platform):
    # Imported under the real platform first, so nothing caches the faked one.
    from core.inference import sandbox_linux, sandbox_macos, sandbox_probe  # noqa: F401

    monkeypatch.setattr(os_sandbox.sys, "platform", platform)
    monkeypatch.setattr(os_sandbox, "_runtime_identity", lambda: "id")
    monkeypatch.setattr(sandbox_probe, "probe", lambda backend, force = False: (True, "ok"))
    capability = os_sandbox.capability_snapshot()
    assert capability.available
    assert "loopback" in capability.remediation


@pytest.mark.parametrize("name", ["credentials", "auth.db", ".pypirc", ".npmrc", "id_ed25519"])
def test_a_folder_holding_any_approval_gated_credential_is_not_granted(tmp_path, name):
    from core.inference import os_sandbox
    (tmp_path / name).write_text("secret")
    assert os_sandbox._holds_a_credential_file(str(tmp_path))
