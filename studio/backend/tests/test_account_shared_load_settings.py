# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Install-wide load policy is the owner's, so a managed account's load obeys it too; an account-keyed read would silently fall back to defaults."""

from __future__ import annotations

import pytest

from auth import policy
from utils import (
    hf_cache_settings,
    llama_cpp_path_settings,
    model_memory_settings,
    openai_auto_switch_settings,
    preview_sharing_settings,
    upload_limits,
    vram_budget_settings,
)
from utils.account_context import OWNER, AccountContext, run_as

ALICE = AccountContext("a" * 32, "alice")


@pytest.fixture(autouse = True)
def studio_home(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(policy, "installation_is_multi_user", lambda: True)
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
    for module in (model_memory_settings, vram_budget_settings, openai_auto_switch_settings):
        monkeypatch.setattr(module, "_cache", {})
        if hasattr(module, "_generation"):
            monkeypatch.setattr(module, "_generation", {})
    yield


def test_a_managed_load_reads_the_owners_model_memory_policy():
    run_as(OWNER, model_memory_settings.set_model_memory_settings, True, False)
    assert run_as(ALICE, model_memory_settings.get_model_memory_settings) == (True, False)
    assert run_as(ALICE, model_memory_settings.should_mlock) is True


def test_a_managed_load_reads_the_owners_vram_budget():
    run_as(OWNER, vram_budget_settings.set_vram_budget_fraction, 0.85)
    assert run_as(ALICE, vram_budget_settings.get_vram_budget_fraction) == pytest.approx(0.85)


def test_a_managed_request_reads_the_owners_auto_switch_policy():
    run_as(OWNER, openai_auto_switch_settings.set_openai_auto_switch, True, None)
    assert run_as(ALICE, openai_auto_switch_settings.get_openai_auto_switch_enabled) is True


def test_a_managed_upload_reads_the_owners_upload_limit():
    run_as(OWNER, upload_limits.set_upload_limit_mb, 2048)
    assert run_as(ALICE, upload_limits.get_upload_limit_mb) == 2048
    assert run_as(ALICE, upload_limits.get_upload_limit_bytes) == 2048 * 1024 * 1024


def test_a_managed_request_reads_the_owners_preview_sharing_switch():
    run_as(OWNER, preview_sharing_settings.set_preview_sharing_enabled, False)
    assert run_as(ALICE, preview_sharing_settings.get_preview_sharing_enabled) is False


def test_a_managed_load_reads_the_owners_llama_cpp_path(tmp_path, monkeypatch):
    binary = tmp_path / "llama" / llama_cpp_path_settings.llama_server_binary_name()
    binary.parent.mkdir(parents = True, exist_ok = True)
    binary.write_text("")
    binary.chmod(0o755)
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    run_as(OWNER, llama_cpp_path_settings.set_custom_llama_cpp_path, str(binary.parent))
    assert run_as(ALICE, llama_cpp_path_settings.get_stored_custom_llama_cpp_path) is not None
    assert run_as(ALICE, llama_cpp_path_settings.custom_llama_cpp_path_source) == "studio"


def test_a_managed_scan_keys_on_the_owners_cache_home(tmp_path):
    custom = tmp_path / "external" / "huggingface"
    custom.parent.mkdir(parents = True, exist_ok = True)
    run_as(OWNER, hf_cache_settings.set_hf_cache_home, str(custom))
    # The key separates in-flight scans per cache volume, so it must name the home actually scanned.
    assert run_as(ALICE, hf_cache_settings.get_hf_cache_paths).cache_home == custom
    assert run_as(ALICE, hf_cache_settings.configured_cache_key) == run_as(
        OWNER, hf_cache_settings.configured_cache_key
    )
    assert run_as(ALICE, hf_cache_settings.configured_cache_key) == "studio:" + str(custom)
