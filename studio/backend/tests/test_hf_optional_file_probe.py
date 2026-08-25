# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Optional Hub file probes must not mutate the cache.

Cached 404s can leave refs pointing to absent snapshots, causing cache scans to omit the repo.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

from utils.hf_probe import hf_file_definitely_absent


def _http_error(name: str, *, fallback: str | None = None) -> Exception:
    """Build an HTTP-shaped error for either supported Hub exception layout."""
    import requests

    from huggingface_hub import errors

    cls = getattr(errors, name, None)
    if cls is None:
        assert fallback is not None, f"{name} is missing and no fallback was named"
        cls = getattr(errors, fallback)
    response = requests.Response()
    response.status_code = 404 if "Entry" in name else 401
    try:
        return cls(name, response = response)
    except TypeError:
        # The plain-Exception base takes a message and nothing else.
        return cls(name)


_BACKEND = Path(__file__).resolve().parents[1]

# Optional-file readers and the guard each must call before downloading. The template reader
# reuses its existing path lookup instead of adding another request.
_GUARDED = {
    "core/inference/llama_cpp.py": {"_fetch_swa_entry_from_hf": "hf_file_definitely_absent"},
    "picker/service.py": {"read_default_chat_template": "get_paths_info"},
    "utils/models/model_config.py": {
        "_raw_config_has_vision_config": "hf_file_definitely_absent",
        "get_base_model_from_lora_identifier": "hf_file_definitely_absent",
    },
    "utils/security/consent.py": {"_load_remote_code_configs": "hf_file_definitely_absent"},
    "utils/security/file_security.py": {"_indexed_shard_paths": "hf_file_definitely_absent"},
    "utils/security/remote_code_scan.py": {
        "external_auto_map_repos": "hf_file_definitely_absent",
        "repo_remote_code_files": "hf_file_definitely_absent",
    },
}


def _raise(exc):
    def _fn(*_args, **_kwargs):
        raise exc

    return _fn


def _patch_metadata(monkeypatch, behavior):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", behavior)


# --- what counts as absent ---------------------------------------------------


def test_the_real_remote_404_reads_as_absent(monkeypatch):
    """Both supported Hub exception layouts report a remote 404 as absent."""
    _patch_metadata(
        monkeypatch,
        _raise(_http_error("RemoteEntryNotFoundError", fallback = "EntryNotFoundError")),
    )

    assert hf_file_definitely_absent("Org/Model", "adapter_config.json") is True


def test_a_remote_404_is_the_only_absent_answer(monkeypatch):
    _patch_metadata(monkeypatch, _raise(EntryNotFoundError("no such file")))

    assert hf_file_definitely_absent("Org/Model", "adapter_config.json") is True


def test_offline_is_not_absence(monkeypatch):
    """A local cache miss means offline, not remote absence."""
    _patch_metadata(monkeypatch, _raise(LocalEntryNotFoundError("offline")))

    assert hf_file_definitely_absent("Org/Model", "adapter_config.json") is False


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: _http_error("GatedRepoError"),
        lambda: _http_error("RepositoryNotFoundError"),
        lambda: TimeoutError("slow"),
        lambda: ValueError("nonsense"),
    ],
    ids = ["gated", "missing-repo", "timeout", "unexpected"],
)
def test_every_other_failure_falls_through_to_the_caller(monkeypatch, make_exc):
    """Only confirmed remote 404s may short-circuit caller behavior."""
    _patch_metadata(monkeypatch, _raise(make_exc()))

    assert hf_file_definitely_absent("Org/Model", "config.json") is False


def test_a_present_file_is_not_absent(monkeypatch):
    _patch_metadata(monkeypatch, lambda *_a, **_k: object())

    assert hf_file_definitely_absent("Org/Model", "config.json") is False


def test_an_unimportable_hub_is_not_an_answer(monkeypatch):
    """An import failure is not proof of absence."""
    import builtins

    real_import = builtins.__import__

    def _fail(name, *args, **kwargs):
        if name == "huggingface_hub":
            raise ImportError("no hub")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail)

    assert hf_file_definitely_absent("Org/Model", "config.json") is False


# --- the cache is never touched ----------------------------------------------


def test_the_probe_writes_nothing_to_the_cache(monkeypatch, tmp_path):
    """A 404 probe leaves refs, snapshots, and no-exist markers unchanged."""
    repo_dir = tmp_path / "models--Org--Model"
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "snapshots" / ("a" * 40)).mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("a" * 40, encoding = "utf-8")

    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    _patch_metadata(monkeypatch, _raise(EntryNotFoundError("no such file")))

    assert hf_file_definitely_absent("Org/Model", "adapter_config.json") is True
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before
    assert (repo_dir / "refs" / "main").read_text(encoding = "utf-8") == "a" * 40


def test_the_lora_base_probe_skips_the_download_when_the_file_is_absent(monkeypatch):
    import huggingface_hub

    from utils.models import model_config

    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *args, **kwargs: calls.append(args) or "/dev/null",
    )
    _patch_metadata(monkeypatch, _raise(EntryNotFoundError("no such file")))

    assert model_config.get_base_model_from_lora_identifier("unsloth/Qwen3-1.7B-GGUF") is None
    assert calls == [], "a file the Hub says is absent must never reach the cache"


def test_a_present_adapter_config_still_resolves_its_base(monkeypatch, tmp_path):
    import huggingface_hub

    from utils.models import model_config

    cfg = tmp_path / "adapter_config.json"
    cfg.write_text('{"base_model_name_or_path": "unsloth/Qwen3-1.7B"}', encoding = "utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *_a, **_k: str(cfg))
    _patch_metadata(monkeypatch, lambda *_a, **_k: object())

    assert model_config.get_base_model_from_lora_identifier("Org/Adapter") == "unsloth/Qwen3-1.7B"


def test_the_chat_template_search_skips_paths_the_listing_does_not_name(monkeypatch):
    """The existing path lookup must gate absent template downloads."""
    import huggingface_hub

    from picker import service

    listed: list[str] = []
    downloads: list[str] = []

    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "get_paths_info",
        lambda self, repo_id, paths, **kwargs: listed.extend(paths) or [],
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda *args, **kwargs: downloads.append(args) or "/dev/null",
    )

    assert service.read_default_chat_template("Org/Model") is None
    assert listed, "the listing must still run; it is what answers both questions"
    assert downloads == [], "a path the listing does not name must never reach the cache"


# --- the guard cannot be dropped ---------------------------------------------


def _functions(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_names(node: ast.AST) -> set[str]:
    """Every function and method name called anywhere inside *node*, nested defs included."""
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            names.add(child.func.attr)
    return names


@pytest.mark.parametrize("rel", sorted(_GUARDED))
def test_every_optional_file_read_on_the_load_path_probes_first(rel):
    defined = _functions(_BACKEND / rel)
    for name, guard in sorted(_GUARDED[rel].items()):
        assert name in defined, f"{rel}::{name} was renamed; update _GUARDED"
        called = _called_names(defined[name])
        assert (
            "hf_hub_download" in called
        ), f"{rel}::{name} no longer downloads; drop it from _GUARDED"
        assert guard in called, (
            f"{rel}::{name} downloads an optional file without asking {guard} first, so a 404 "
            "there rewrites refs/main and hides the repo from the Hub cached inventory"
        )
