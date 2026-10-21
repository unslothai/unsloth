# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from utils import llama_cpp_path_settings as path_settings


@pytest.fixture()
def settings_store(monkeypatch):
    store = {}
    monkeypatch.delenv("LLAMA_SERVER_PATH", raising = False)
    monkeypatch.delenv("UNSLOTH_LLAMA_CPP_PATH", raising = False)
    monkeypatch.delenv(path_settings.MANAGED_LLAMA_CPP_PATH_MARKER, raising = False)
    monkeypatch.setattr(
        "storage.studio_db.get_app_setting",
        lambda key, fallback = None: store.get(key, fallback),
    )
    monkeypatch.setattr(
        "storage.studio_db.upsert_app_settings",
        lambda values: store.update(values) or values,
    )
    return store


def _binary(
    root: Path,
    *,
    platform: str | None = None,
    layout: str = "build",
) -> Path:
    name = path_settings.llama_server_binary_name(platform)
    if layout == "root":
        binary = root / name
    elif layout == "release":
        binary = root / "build" / "bin" / "Release" / name
    else:
        binary = root / "build" / "bin" / name
    binary.parent.mkdir(parents = True, exist_ok = True)
    binary.write_bytes(b"test llama-server")
    binary.chmod(0o755)
    return binary


@pytest.mark.parametrize(
    ("platform", "layout"),
    [
        ("linux", "root"),
        ("linux", "build"),
        ("darwin", "build"),
        ("win32", "root"),
        ("win32", "build"),
        ("win32", "release"),
    ],
)
def test_supported_build_layouts_resolve(tmp_path, platform, layout):
    root = tmp_path / f"{platform}-{layout}"
    expected = _binary(root, platform = platform, layout = layout)

    assert path_settings.resolve_llama_server_binary(root, platform = platform) == expected


def test_setting_round_trip_and_reset(settings_store, tmp_path):
    root = tmp_path / "custom llama.cpp"
    binary = _binary(root)
    initial_revision = path_settings.custom_llama_cpp_path_revision()

    selected = path_settings.set_custom_llama_cpp_path(str(root))
    status = path_settings.custom_llama_cpp_path_status()

    assert selected == root.resolve()
    assert settings_store[path_settings.CUSTOM_LLAMA_CPP_PATH_SETTING_KEY] == str(root.resolve())
    assert status == {
        "path": str(root.resolve()),
        "source": "studio",
        "editable": True,
        "available": True,
        "resolved_binary": str(binary),
        "environment_variable": None,
    }
    assert path_settings.custom_llama_cpp_path_revision() == initial_revision + 1

    assert path_settings.set_custom_llama_cpp_path(None) is None
    assert settings_store[path_settings.CUSTOM_LLAMA_CPP_PATH_SETTING_KEY] is None
    assert path_settings.custom_llama_cpp_path_status()["source"] == "default"
    assert path_settings.custom_llama_cpp_path_revision() == initial_revision + 2


def test_setting_rejects_missing_folder_and_folder_without_server(settings_store, tmp_path):
    with pytest.raises(ValueError, match = "does not exist"):
        path_settings.set_custom_llama_cpp_path(str(tmp_path / "missing"))

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match = "No executable llama-server"):
        path_settings.set_custom_llama_cpp_path(str(empty))

    assert path_settings.CUSTOM_LLAMA_CPP_PATH_SETTING_KEY not in settings_store


def test_environment_path_is_displayed_and_locks_the_setting(settings_store, monkeypatch, tmp_path):
    stored = tmp_path / "stored"
    _binary(stored)
    settings_store[path_settings.CUSTOM_LLAMA_CPP_PATH_SETTING_KEY] = str(stored)

    env_root = tmp_path / "environment"
    env_binary = _binary(env_root)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(env_root))

    status = path_settings.custom_llama_cpp_path_status()
    assert status["source"] == "environment"
    assert status["editable"] is False
    assert status["path"] == str(env_root)
    assert status["resolved_binary"] == str(env_binary)
    assert status["environment_variable"] == "UNSLOTH_LLAMA_CPP_PATH"
    with pytest.raises(RuntimeError, match = "UNSLOTH_LLAMA_CPP_PATH"):
        path_settings.set_custom_llama_cpp_path(str(stored))


def test_direct_environment_binary_has_highest_priority(settings_store, monkeypatch, tmp_path):
    env_dir = tmp_path / "directory-env"
    _binary(env_dir)
    direct = _binary(tmp_path / "direct")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(env_dir))
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(direct))

    status = path_settings.custom_llama_cpp_path_status()

    assert status["path"] == str(direct)
    assert status["resolved_binary"] == str(direct)
    assert status["environment_variable"] == "LLAMA_SERVER_PATH"


def test_studio_managed_environment_path_does_not_lock_out_the_ui(
    settings_store, monkeypatch, tmp_path
):
    from core.inference.llama_cpp import LlamaCppBackend

    managed = tmp_path / "managed"
    _binary(managed)
    selected = tmp_path / "selected"
    selected_binary = _binary(selected)
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(managed))
    monkeypatch.setenv(path_settings.MANAGED_LLAMA_CPP_PATH_MARKER, "1")

    path_settings.set_custom_llama_cpp_path(str(selected))
    status = path_settings.custom_llama_cpp_path_status()

    assert status["source"] == "studio"
    assert status["editable"] is True
    assert status["resolved_binary"] == str(selected_binary)
    assert LlamaCppBackend._find_llama_server_binary() == str(selected_binary)


def test_inherited_managed_path_is_marked_even_when_launcher_exported_it(
    settings_store, monkeypatch, tmp_path
):
    managed = tmp_path / "not-installed-yet" / "llama.cpp"
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(managed))

    assert path_settings.mark_managed_llama_cpp_path(managed) is True
    assert os.environ[path_settings.MANAGED_LLAMA_CPP_PATH_MARKER] == "1"
    assert path_settings.custom_llama_cpp_path_status()["editable"] is True

    explicit = tmp_path / "user-owned-llama.cpp"
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", str(explicit))
    assert path_settings.mark_managed_llama_cpp_path(managed) is False
    assert path_settings.MANAGED_LLAMA_CPP_PATH_MARKER not in os.environ
    assert path_settings.custom_llama_cpp_path_status()["editable"] is False


def test_runtime_skips_non_executable_root_entrypoint_for_valid_build_layout(
    settings_store, monkeypatch, tmp_path
):
    from core.inference.llama_cpp import LlamaCppBackend

    root = tmp_path / "custom"
    root_binary = _binary(root, platform = "linux", layout = "root")
    build_binary = _binary(root, platform = "linux", layout = "build")
    monkeypatch.setattr(path_settings.sys, "platform", "linux")
    monkeypatch.setattr(
        path_settings.os,
        "access",
        lambda path, mode: Path(path) != root_binary,
    )

    path_settings.set_custom_llama_cpp_path(str(root))

    assert path_settings.resolve_llama_server_binary(root) == build_binary
    assert LlamaCppBackend._find_llama_server_binary() == str(build_binary)


def test_runtime_resolver_uses_studio_path_and_does_not_silently_fallback(
    settings_store, monkeypatch, tmp_path
):
    from core.inference.llama_cpp import LlamaCppBackend

    selected = tmp_path / "selected"
    binary = _binary(selected)
    settings_store[path_settings.CUSTOM_LLAMA_CPP_PATH_SETTING_KEY] = str(selected)

    assert LlamaCppBackend._find_llama_server_binary() == str(binary)

    binary.unlink()
    monkeypatch.setattr("core.inference.llama_cpp.shutil.which", lambda _name: "fallback")
    assert LlamaCppBackend._find_llama_server_binary() is None


def test_runtime_resolver_rejects_a_selected_binary_that_loses_execute_permission(
    settings_store, monkeypatch, tmp_path
):
    from core.inference.llama_cpp import LlamaCppBackend

    from core.inference import llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import (
        LLAMA_SERVER_NOT_FOUND_DETAIL,
        LlamaCppBackend,
    )

    selected = tmp_path / "selected"
    binary = _binary(selected, platform = "linux")
    settings_store[path_settings.CUSTOM_LLAMA_CPP_PATH_SETTING_KEY] = str(selected)
    real_access = os.access

    monkeypatch.setattr(llama_cpp_module.sys, "platform", "linux")
    monkeypatch.setattr(
        os,
        "access",
        lambda path, mode: False if Path(path) == binary else real_access(path, mode),
    )

    assert path_settings.custom_llama_cpp_path_status()["available"] is False
    assert LlamaCppBackend._find_llama_server_binary() is None
    assert LlamaCppBackend._find_llama_server_binary(include_denied = True) is None
    assert "execute permission" in LLAMA_SERVER_NOT_FOUND_DETAIL


def test_backend_updater_never_replaces_a_studio_selected_tree(monkeypatch):
    from utils import llama_cpp_update

    monkeypatch.setattr(llama_cpp_update, "_studio_custom_path_active", lambda: True)
    monkeypatch.setattr(llama_cpp_update, "_find_binary", lambda: "custom/llama-server")
    monkeypatch.setattr(
        llama_cpp_update,
        "read_install_marker",
        lambda _binary: {"backend": "cuda", "backend_request": "auto"},
    )

    status = llama_cpp_update.get_backend_status()
    plan = llama_cpp_update._plan_llama_phase("cpu")

    assert status["supported"] is False
    assert status["reason"] == "custom_path"
    assert plan["skip_reason"] == "custom_path"
    assert plan["refusal"]["reason"] == "custom_path"


def test_selected_checkout_is_not_given_managed_runtime_repair_advice(settings_store, tmp_path):
    from core.inference.llama_cpp import LlamaCppBackend

    root = tmp_path / "llama.cpp"
    binary = _binary(root)
    path_settings.set_custom_llama_cpp_path(str(root))

    assert LlamaCppBackend._is_unsloth_managed_binary(str(binary)) is False
    message = LlamaCppBackend._missing_library_message("libllama.so", str(binary))
    assert "unsloth studio update" not in message
    assert "custom install" in message


def test_settings_route_round_trips_the_selected_folder(settings_store, monkeypatch, tmp_path):
    from routes import settings as settings_route

    root = tmp_path / "route-selected"
    binary = _binary(root)
    monkeypatch.setattr(settings_route, "_llama_cpp_path_reload_required", lambda: False)

    saved = settings_route.update_llama_cpp_path(
        settings_route.LlamaCppPathPayload(path = str(root)),
        current_subject = "studio-user",
        via_api_key = False,
    )
    loaded = settings_route.get_llama_cpp_path(current_subject = "studio-user")

    assert saved.path == str(root.resolve())
    assert saved.source == "studio"
    assert saved.resolved_binary == str(binary)
    assert loaded == saved


def test_settings_route_reports_reload_while_old_binary_launch_is_pending(monkeypatch):
    from routes import settings as settings_route

    from routes import inference as inference_route

    class _PendingBackend:
        is_active = False
        _binary_revision_pending = ("old-binary",)

        @staticmethod
        def _binary_changed_since_revision(revision):
            return revision == ("old-binary",)

    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: _PendingBackend())

    assert settings_route._llama_cpp_path_reload_required() is True


def test_settings_route_rejects_api_key_writes_before_mutation(monkeypatch):
    from routes import settings as settings_route

    from fastapi import HTTPException

    mutated = False

    def _unexpected_mutation(_path):
        nonlocal mutated
        mutated = True

    monkeypatch.setattr(settings_route, "set_custom_llama_cpp_path", _unexpected_mutation)

    with pytest.raises(HTTPException) as exc_info:
        settings_route.update_llama_cpp_path(
            settings_route.LlamaCppPathPayload(path = None),
            current_subject = "api-client",
            via_api_key = True,
        )

    assert exc_info.value.status_code == 403
    assert mutated is False


def test_settings_route_returns_the_specific_validation_error(settings_store, tmp_path):
    from routes import settings as settings_route

    from fastapi import HTTPException

    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(HTTPException) as exc_info:
        settings_route.update_llama_cpp_path(
            settings_route.LlamaCppPathPayload(path = str(empty)),
            current_subject = "studio-user",
            via_api_key = False,
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        f"No executable {path_settings.llama_server_binary_name()} was found in that folder "
        "or its build/bin directory."
    )


# The account that does not exist anywhere, which is what a rename leaves behind in a
# service unit or a .env.
_UNKNOWN_USER = "~no-such-account-anywhere/llama.cpp"


def test_an_unresolvable_named_user_does_not_raise_out_of_the_expansion():
    """Codex 3962583754, P2. Path.expanduser() raises RuntimeError on a name it cannot
    resolve, unlike os.path.expanduser, which hands the value straight back.

    Measured on this host:

        Path("~no-such-account-anywhere/llama.cpp").expanduser()
        RuntimeError: Could not determine home directory.

    Every reader of UNSLOTH_LLAMA_CPP_PATH used the raising form, so a stale pin made
    runtime discovery itself throw rather than treating the override as unusable and
    continuing down the documented search order.
    """
    with pytest.raises(RuntimeError):
        Path(_UNKNOWN_USER).expanduser()
    assert path_settings.expanded_user_path(_UNKNOWN_USER) == Path(_UNKNOWN_USER)


def test_the_resolvable_forms_still_expand(monkeypatch, tmp_path):
    """The guard must not cost the expansion that was the point of adding it."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert path_settings.expanded_user_path("~/llama.cpp") == tmp_path / "llama.cpp"
    assert path_settings.expanded_user_path("~") == tmp_path
    absolute = tmp_path / "elsewhere" / "llama.cpp"
    assert path_settings.expanded_user_path(absolute) == absolute
    assert path_settings.expanded_user_path("relative/llama.cpp") == Path("relative/llama.cpp")


def test_the_status_payload_survives_an_unresolvable_override(settings_store, monkeypatch):
    """The UI reads this, and a raise here is a broken settings page rather than a
    message saying the pin is unusable."""
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", _UNKNOWN_USER)
    status = path_settings.custom_llama_cpp_path_status()
    assert status["source"] == "environment"
    assert status["path"] == _UNKNOWN_USER
    assert status["resolved_binary"] is None
    assert status["available"] is False


def test_discovery_continues_past_an_unresolvable_override(monkeypatch, tmp_path):
    """The failure the item is actually about: _find_llama_server_binary threw instead
    of falling through, and several callers have no exception boundary, so a model
    operation reported the expansion error rather than using another runtime."""
    llama_cpp = pytest.importorskip("core.inference.llama_cpp")
    monkeypatch.setenv("UNSLOTH_LLAMA_CPP_PATH", _UNKNOWN_USER)
    monkeypatch.delenv("UNSLOTH_STUDIO_MANAGED_LLAMA_CPP_PATH", raising = False)
    # Not asserting which runtime is found: the machine decides that. The claim under
    # test is only that the search reaches its own answer instead of raising.
    result = llama_cpp.LlamaCppBackend._find_llama_server_binary()
    assert result is None or isinstance(result, (str, Path, tuple))
