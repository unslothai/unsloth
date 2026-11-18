# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pins the ollama-manifest: load hand-off in _resolve_model_identifier_for_request."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import HTTPException


def _write_ollama_store(root: Path, *, extra_layers: tuple[str, ...] = ()) -> Path:
    """A minimal Ollama layout whose optional extra layers resolve to real blobs."""
    digest_value = "a" * 64
    blob = root / "blobs" / f"sha256-{digest_value}"
    blob.parent.mkdir(parents = True)
    blob.write_bytes(b"GGUF-not-really")

    layers = [
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": f"sha256:{digest_value}",
        }
    ]
    for index, media_type in enumerate(extra_layers, start = 1):
        layer_digest = f"{index:064x}"
        (root / "blobs" / f"sha256-{layer_digest}").write_text("{}", encoding = "utf-8")
        layers.append(
            {
                "mediaType": media_type,
                "digest": f"sha256:{layer_digest}",
            }
        )

    tag_file = root / "manifests" / "registry.ollama.ai" / "library" / "llama3" / "latest"
    tag_file.parent.mkdir(parents = True)
    tag_file.write_text(
        json.dumps(
            {
                "config": {},
                "layers": layers,
            }
        ),
        encoding = "utf-8",
    )
    return tag_file


def _manifest_ref(tmp_path: Path, monkeypatch) -> str:
    from hub.services.models import ollama

    root = tmp_path / "ollama"
    _write_ollama_store(root)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])

    rows = ollama.scan_ollama_dir(root)
    assert len(rows) == 1
    ref = rows[0].load_id
    assert ollama.is_ollama_manifest_ref(ref)
    return ref


def test_custom_folder_scan_preserves_ollama_rows(tmp_path):
    from hub.services.models import local_inventory

    root = tmp_path / "custom-ollama"
    _write_ollama_store(root)

    rows = [
        local_inventory._promote_to_custom_source(row)
        for row in local_inventory._scan_custom_folder(root)
    ]

    assert len(rows) == 1
    assert rows[0].source == "ollama"
    assert rows[0].load_id.startswith("ollama-manifest:")


def test_registered_custom_ollama_ref_can_be_materialized(tmp_path, monkeypatch):
    from hub.services.models import ollama

    from hub.storage import scan_folders

    root = tmp_path / "registered-ollama"
    _write_ollama_store(root)
    ref = ollama.scan_ollama_dir(root)[0].load_id
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [])
    monkeypatch.setattr(scan_folders, "list_scan_folders", lambda: [{"path": str(root)}])

    assert Path(ollama.materialize_ollama_model_ref(ref)).is_file()


def test_load_resolves_a_manifest_ref_to_a_gguf_link(tmp_path, monkeypatch):
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    ref = _manifest_ref(tmp_path, monkeypatch)
    identifier, label, native_grant_backed = _resolve_model_identifier_for_request(
        LoadRequest(model_path = ref), operation = "load-model"
    )

    assert identifier.endswith(".gguf")
    assert Path(identifier).is_file(), "the .gguf link must be materialized"
    assert label == Path(identifier).name, "logs get the link name, not the opaque ref"
    assert native_grant_backed is False


def test_public_identity_keeps_the_manifest_ref():
    from inspect import getsource

    from routes.inference import _public_model_identifier, validate_model

    ref = "ollama-manifest:%2Fhome%2Fu%2F.ollama%2Fmanifests%2Fllama3"
    assert _public_model_identifier(ref, "/tmp/.studio_links/llama3.gguf") == ref
    assert _public_model_identifier("owner/model", "owner/model") == "owner/model"
    assert "_public_model_identifier(request.model_path, model_identifier)" in getsource(
        validate_model
    )


def test_retagged_manifest_replaces_one_hardlink_and_invalidates_loaded_identity(
    tmp_path, monkeypatch
):
    from hub.services.models import ollama
    from types import SimpleNamespace

    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

    root = tmp_path / "ollama-retagged"
    tag_file = _write_ollama_store(root)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])

    def deny_symlink(*_args, **_kwargs):
        raise OSError("symlinks disabled")

    monkeypatch.setattr(Path, "symlink_to", deny_symlink)
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    first_path = ollama.materialize_ollama_model_ref(ref)
    first_resolved = Path(first_path).resolve()
    first_stat = first_resolved.stat()
    first_identity = (
        (
            str(first_resolved),
            first_stat.st_dev,
            first_stat.st_ino,
            first_stat.st_size,
            first_stat.st_mtime_ns,
        ),
    )
    assert Path(first_path).read_bytes() == b"GGUF-not-really"

    replacement_digest = "b" * 64
    replacement_blob = root / "blobs" / f"sha256-{replacement_digest}"
    replacement_blob.write_bytes(b"GGUF-replacement")
    manifest = json.loads(tag_file.read_text(encoding = "utf-8"))
    manifest["layers"][0]["digest"] = f"sha256:{replacement_digest}"

    replacement_config_digest = "c" * 64
    (root / "blobs" / f"sha256-{replacement_config_digest}").write_text(
        json.dumps({"file_type": "Q8_0"}), encoding = "utf-8"
    )
    manifest["config"] = {"digest": f"sha256:{replacement_config_digest}"}
    tag_file.write_text(json.dumps(manifest), encoding = "utf-8")

    second_path = ollama.materialize_ollama_model_ref(ref)

    assert second_path == first_path
    assert Path(second_path).read_bytes() == b"GGUF-replacement"
    assert list((root / ".studio_links").rglob("*.gguf")) == [Path(second_path)]

    resident = SimpleNamespace(
        is_loaded = True,
        _model_identifier = ref,
        _gguf_path = first_path,
        _gguf_load_identity = first_identity,
    )
    replacement_intent = GgufLoadIntent(model_identifier = ref, gguf_path = second_path)
    assert not LlamaCppBackend.matches_load_source(resident, replacement_intent)


def test_missing_projector_retag_is_rejected_before_main_link_changes(tmp_path, monkeypatch):
    from hub.services.models import ollama

    root = tmp_path / "ollama-missing-projector"
    tag_file = _write_ollama_store(
        root,
        extra_layers = ("application/vnd.ollama.image.projector",),
    )
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    model_path = ollama.materialize_ollama_model_ref(ref)
    projector_path = next(Path(model_path).parent.glob("*-mmproj.gguf"))

    replacement_digest = "b" * 64
    (root / "blobs" / f"sha256-{replacement_digest}").write_bytes(b"GGUF-replacement")
    manifest = json.loads(tag_file.read_text(encoding = "utf-8"))
    manifest["layers"][0]["digest"] = f"sha256:{replacement_digest}"
    manifest["layers"][1]["digest"] = f"sha256:{'c' * 64}"
    tag_file.write_text(json.dumps(manifest), encoding = "utf-8")

    with pytest.raises(ValueError, match = "projector"):
        ollama.materialize_ollama_model_ref(ref)

    assert Path(model_path).read_bytes() == b"GGUF-not-really"
    assert projector_path.read_bytes() == b"{}"


@pytest.mark.parametrize("remove_projector", [False, True])
def test_dangling_projector_link_can_be_replaced_or_removed(
    tmp_path, monkeypatch, remove_projector
):
    from hub.services.models import ollama

    root = tmp_path / "ollama-dangling-projector"
    tag_file = _write_ollama_store(
        root,
        extra_layers = ("application/vnd.ollama.image.projector",),
    )
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"
    model_path = ollama.materialize_ollama_model_ref(ref)
    projector_path = next(Path(model_path).parent.glob("*-mmproj.gguf"))

    manifest = json.loads(tag_file.read_text(encoding = "utf-8"))
    if remove_projector:
        manifest["layers"] = manifest["layers"][:1]
    else:
        projector_digest = "c" * 64
        (root / "blobs" / f"sha256-{projector_digest}").write_bytes(b"GGUF-projector-replacement")
        manifest["layers"][1]["digest"] = f"sha256:{projector_digest}"
    tag_file.write_text(json.dumps(manifest), encoding = "utf-8")

    real_is_symlink = Path.is_symlink
    real_resolve = Path.resolve

    def is_symlink(path):
        return path == projector_path or real_is_symlink(path)

    def resolve(path, *args, **kwargs):
        if path == projector_path and kwargs.get("strict"):
            raise FileNotFoundError(projector_path)
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)
    monkeypatch.setattr(Path, "resolve", resolve)

    assert ollama.materialize_ollama_model_ref(ref) == model_path
    if remove_projector:
        assert not projector_path.exists()
    else:
        assert projector_path.read_bytes() == b"GGUF-projector-replacement"


def test_failed_main_retag_restores_the_previous_projector(tmp_path, monkeypatch):
    from hub.services.models import ollama

    root = tmp_path / "ollama-pair-rollback"
    tag_file = _write_ollama_store(
        root,
        extra_layers = ("application/vnd.ollama.image.projector",),
    )

    def deny_symlink(*_args, **_kwargs):
        raise OSError("symlinks disabled")

    monkeypatch.setattr(Path, "symlink_to", deny_symlink)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"
    model_path = ollama.materialize_ollama_model_ref(ref)
    projector_path = next(Path(model_path).parent.glob("*-mmproj.gguf"))

    model_digest = "b" * 64
    projector_digest = "c" * 64
    model_blob = root / "blobs" / f"sha256-{model_digest}"
    projector_blob = root / "blobs" / f"sha256-{projector_digest}"
    model_blob.write_bytes(b"GGUF-replacement")
    projector_blob.write_bytes(b"GGUF-projector-replacement")
    manifest = json.loads(tag_file.read_text(encoding = "utf-8"))
    manifest["layers"][0]["digest"] = f"sha256:{model_digest}"
    manifest["layers"][1]["digest"] = f"sha256:{projector_digest}"
    tag_file.write_text(json.dumps(manifest), encoding = "utf-8")

    make_link = ollama._make_ollama_blob_link

    def fail_replacement_model(link_dir, link_name, target):
        if target == model_blob:
            return None
        return make_link(link_dir, link_name, target)

    monkeypatch.setattr(ollama, "_make_ollama_blob_link", fail_replacement_model)
    with pytest.raises(ValueError, match = "model blob"):
        ollama.materialize_ollama_model_ref(ref)

    assert Path(model_path).read_bytes() == b"GGUF-not-really"
    assert projector_path.read_bytes() == b"{}"


def test_materialization_lease_blocks_a_concurrent_retag(tmp_path, monkeypatch):
    from hub.services.models import ollama

    import threading

    root = tmp_path / "ollama-materialization-lease"
    tag_file = _write_ollama_store(root)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    alternate_ref = ref.replace("%2F", "%2f").replace("%5C", "%5c")  # POSIX or Windows separators
    assert alternate_ref != ref
    lease = ollama.acquire_ollama_model_ref(ref)

    replacement_digest = "b" * 64
    (root / "blobs" / f"sha256-{replacement_digest}").write_bytes(b"GGUF-replacement")
    manifest = json.loads(tag_file.read_text(encoding = "utf-8"))
    manifest["layers"][0]["digest"] = f"sha256:{replacement_digest}"
    tag_file.write_text(json.dumps(manifest), encoding = "utf-8")

    started = threading.Event()
    finished = threading.Event()

    def retag():
        started.set()
        ollama.materialize_ollama_model_ref(alternate_ref)
        finished.set()

    worker = threading.Thread(target = retag)
    worker.start()
    assert started.wait(1)
    assert not finished.wait(0.1)
    lease.release()
    worker.join(timeout = 1)

    assert finished.is_set()
    assert Path(lease.path).read_bytes() == b"GGUF-replacement"


def test_waiting_route_lease_does_not_starve_the_default_executor(tmp_path, monkeypatch):
    from hub.services.models import ollama

    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack

    from models.inference import ValidateModelRequest
    from routes import inference

    root = tmp_path / "ollama-route-lease"
    tag_file = _write_ollama_store(root)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"
    request = ValidateModelRequest(model_path = ref)

    real_acquire = inference.acquire_ollama_model_ref
    waiter_started = threading.Event()
    calls_lock = threading.Lock()
    calls = 0

    def observed_acquire(model_ref):
        nonlocal calls
        with calls_lock:
            calls += 1
            is_waiter = calls == 2
        if is_waiter:
            waiter_started.set()
        return real_acquire(model_ref)

    monkeypatch.setattr(inference, "acquire_ollama_model_ref", observed_acquire)

    async def scenario():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers = 1))
        first_stack = ExitStack()
        second_stack = ExitStack()
        await inference._lease_ollama_model_ref(
            request, operation = "validate-model", stack = first_stack
        )
        waiter = asyncio.create_task(
            inference._lease_ollama_model_ref(
                request, operation = "validate-model", stack = second_stack
            )
        )
        for _ in range(100):
            if waiter_started.is_set():
                break
            await asyncio.sleep(0.01)
        assert waiter_started.is_set()
        try:
            progressed = await asyncio.wait_for(asyncio.to_thread(lambda: True), timeout = 1)
        except TimeoutError:
            progressed = False
        finally:
            first_stack.close()
        await asyncio.wait_for(waiter, timeout = 1)
        second_stack.close()
        return progressed

    assert asyncio.run(scenario())


def test_projector_removal_deletes_the_stale_link(tmp_path, monkeypatch):
    from hub.services.models import ollama

    root = tmp_path / "ollama-projector-removed"
    tag_file = _write_ollama_store(
        root,
        extra_layers = ("application/vnd.ollama.image.projector",),
    )
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    model_path = ollama.materialize_ollama_model_ref(ref)
    projector_path = next(Path(model_path).parent.glob("*-mmproj.gguf"))
    manifest = json.loads(tag_file.read_text(encoding = "utf-8"))
    manifest["layers"] = manifest["layers"][:1]
    tag_file.write_text(json.dumps(manifest), encoding = "utf-8")

    assert ollama.materialize_ollama_model_ref(ref) == model_path
    assert not projector_path.exists()


def test_projector_only_retag_invalidates_loaded_identity(tmp_path, monkeypatch):
    from hub.services.models import ollama
    from types import SimpleNamespace

    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

    root = tmp_path / "ollama-projector-retagged"
    tag_file = _write_ollama_store(
        root,
        extra_layers = ("application/vnd.ollama.image.projector",),
    )
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    model_path = ollama.materialize_ollama_model_ref(ref)
    projector_path = next(Path(model_path).parent.glob("*-mmproj.gguf"))
    first_identity = LlamaCppBackend._gguf_load_source_identity(model_path, str(projector_path))

    replacement_digest = "c" * 64
    replacement_blob = root / "blobs" / f"sha256-{replacement_digest}"
    replacement_blob.write_bytes(b"GGUF-projector-replacement")
    manifest = json.loads(tag_file.read_text(encoding = "utf-8"))
    manifest["layers"][1]["digest"] = f"sha256:{replacement_digest}"
    tag_file.write_text(json.dumps(manifest), encoding = "utf-8")

    assert ollama.materialize_ollama_model_ref(ref) == model_path
    assert Path(projector_path).read_bytes() == b"GGUF-projector-replacement"

    resident = SimpleNamespace(
        is_loaded = True,
        _model_identifier = ref,
        _gguf_path = model_path,
        _gguf_load_identity = first_identity,
    )
    replacement_intent = GgufLoadIntent(
        model_identifier = ref,
        gguf_path = model_path,
        mmproj_path = str(projector_path),
    )
    assert not LlamaCppBackend.matches_load_source(resident, replacement_intent)


def test_ollama_intent_loads_the_link_but_keeps_the_manifest_identity(tmp_path, monkeypatch):
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request
    from types import SimpleNamespace

    from routes.inference import (
        _active_gguf_intent,
        _LoadPlacement,
        _llama_status_model_ids,
        _resolve_gguf_load_intent,
        _resolve_model_identifier_for_request,
    )

    ref = _manifest_ref(tmp_path, monkeypatch)
    resolved, _, _ = _resolve_model_identifier_for_request(
        LoadRequest(model_path = ref), operation = "load-model"
    )
    config = SimpleNamespace(
        identifier = resolved,
        gguf_hf_repo = None,
        gguf_file = resolved,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
        gguf_variant = None,
        is_vision = False,
    )

    intent = _resolve_gguf_load_intent(
        config,
        LoadRequest(model_path = ref),
        native_grant_backed = False,
        chat_template_override = None,
        extra_args = None,
        placement = _LoadPlacement(None, None, False, None),
        n_parallel = 1,
    )

    assert intent.gguf_path == resolved
    assert intent.model_identifier == ref

    backend = SimpleNamespace(
        model_identifier = intent.model_identifier,
        _native_grant_backed = False,
        _native_display_label = None,
        _openai_advertised_id = None,
    )
    assert _llama_status_model_ids(backend) == (ref, ref)

    active_backend = SimpleNamespace(
        extra_args = None,
        last_load_intent = intent,
        hf_repo = None,
        gguf_path = resolved,
        hf_variant = None,
        layer_preserves_tensor_intent = False,
    )
    active_intent = _active_gguf_intent(
        LoadRequest(model_path = ref),
        active_backend,
        model_identifier = ref,
        chat_template_override = None,
        n_parallel = 1,
        native_grant_backed = False,
    )
    assert active_intent.model_identifier == ref
    assert active_intent.gguf_path == resolved


_METADATA_LAYERS = (
    "application/vnd.ollama.image.params",
    "application/vnd.ollama.image.template",
    "application/vnd.ollama.image.system",
    "application/vnd.ollama.image.messages",
    "application/vnd.ollama.image.prompt",
)

_UNSUPPORTED_RUNTIME_LAYERS = (
    "application/vnd.ollama.image.adapter",
    "application/vnd.ollama.image.future-runtime",
)

_IGNORABLE_LAYERS = (
    "application/vnd.ollama.image.draft",
    "application/vnd.ollama.image.embed",
)


def _rich_manifest_ref(tmp_path: Path, monkeypatch) -> tuple[Path, str]:
    from hub.services.models import ollama

    root = tmp_path / "ollama-rich"
    tag_file = _write_ollama_store(
        root, extra_layers = _METADATA_LAYERS + _UNSUPPORTED_RUNTIME_LAYERS
    )
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"
    return root, ref


def test_rich_manifest_is_withheld_from_inventory(tmp_path, monkeypatch):
    from hub.services.models import ollama
    root, _ = _rich_manifest_ref(tmp_path, monkeypatch)

    assert ollama.scan_ollama_dir(root) == []


def test_a_normally_pulled_model_is_listed(tmp_path, monkeypatch):
    from hub.services.models import ollama

    root = tmp_path / "ollama-pulled"
    _write_ollama_store(root, extra_layers = _METADATA_LAYERS)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])

    rows = ollama.scan_ollama_dir(root)
    assert len(rows) == 1
    assert ollama.is_ollama_manifest_ref(rows[0].load_id)


def test_a_normally_pulled_model_resolves_for_a_load(tmp_path, monkeypatch):
    from hub.services.models import ollama
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    root = tmp_path / "ollama-pulled-load"
    tag_file = _write_ollama_store(root, extra_layers = _METADATA_LAYERS)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    resolved, _link_name, _is_dir = _resolve_model_identifier_for_request(
        LoadRequest(model_path = ref), operation = "load-model"
    )
    assert resolved.endswith(".gguf")


@pytest.mark.parametrize("ignorable", _IGNORABLE_LAYERS)
def test_an_ignorable_layer_does_not_hide_the_primary_model(tmp_path, monkeypatch, ignorable):
    # Neither layer stops llama.cpp loading the model; ollama ignores embed since 0.1.2.

    from hub.services.models import ollama

    root = tmp_path / f"ollama-{ignorable.rpartition('.')[2]}"
    _write_ollama_store(root, extra_layers = _METADATA_LAYERS + (ignorable,))
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])

    rows = ollama.scan_ollama_dir(root)
    assert len(rows) == 1
    assert ollama.is_ollama_manifest_ref(rows[0].load_id)


@pytest.mark.parametrize("unsupported", _UNSUPPORTED_RUNTIME_LAYERS)
def test_one_unsupported_layer_still_withholds_beside_the_metadata(
    tmp_path, monkeypatch, unsupported
):
    # Pin each unsupported type alone; the combined test would miss one being re-admitted.

    from hub.services.models import ollama

    root = tmp_path / f"ollama-{unsupported.rpartition('.')[2]}"
    _write_ollama_store(root, extra_layers = _METADATA_LAYERS + (unsupported,))
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])

    assert ollama.scan_ollama_dir(root) == []


def test_rich_manifest_ref_is_rejected_without_creating_links(tmp_path, monkeypatch):
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    root, ref = _rich_manifest_ref(tmp_path, monkeypatch)

    with pytest.raises(HTTPException) as excinfo:
        _resolve_model_identifier_for_request(LoadRequest(model_path = ref), operation = "load-model")

    assert excinfo.value.status_code == 400
    assert "unsupported runtime layers" in str(excinfo.value.detail).lower()
    for media_type in _UNSUPPORTED_RUNTIME_LAYERS:
        assert media_type in str(excinfo.value.detail)
    links_root = root / ".studio_links"
    assert not links_root.exists() or not any(path.is_file() for path in links_root.rglob("*"))


def test_license_metadata_does_not_hide_a_plain_manifest(tmp_path, monkeypatch):
    from hub.services.models import ollama

    root = tmp_path / "ollama-licensed"
    _write_ollama_store(
        root,
        extra_layers = ("application/vnd.ollama.image.license",),
    )
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])

    rows = ollama.scan_ollama_dir(root)
    assert len(rows) == 1
    assert ollama.is_ollama_manifest_ref(rows[0].load_id)


def test_non_object_manifest_ref_is_a_400(tmp_path, monkeypatch):
    from hub.services.models import ollama
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    root = tmp_path / "ollama-non-object-manifest"
    tag_file = _write_ollama_store(root)
    tag_file.write_text("[]", encoding = "utf-8")
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    assert ollama.scan_ollama_dir(root) == []
    with pytest.raises(HTTPException) as excinfo:
        _resolve_model_identifier_for_request(LoadRequest(model_path = ref), operation = "load-model")
    assert excinfo.value.status_code == 400
    assert "manifest" in str(excinfo.value.detail).lower()


def test_non_object_config_blob_ref_is_a_400(tmp_path, monkeypatch):
    from hub.services.models import ollama
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    root = tmp_path / "ollama-non-object-config"
    tag_file = _write_ollama_store(root)
    config_digest = "b" * 64
    (root / "blobs" / f"sha256-{config_digest}").write_text("[]", encoding = "utf-8")
    manifest = json.loads(tag_file.read_text(encoding = "utf-8"))
    manifest["config"] = {"digest": f"sha256:{config_digest}"}
    tag_file.write_text(json.dumps(manifest), encoding = "utf-8")
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    assert ollama.scan_ollama_dir(root) == []
    with pytest.raises(HTTPException) as excinfo:
        _resolve_model_identifier_for_request(LoadRequest(model_path = ref), operation = "load-model")
    assert excinfo.value.status_code == 400
    assert "config blob" in str(excinfo.value.detail).lower()


def test_a_ref_outside_known_ollama_dirs_is_a_400(tmp_path, monkeypatch):
    from hub.services.models import ollama
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [tmp_path / "elsewhere"])

    outside = tmp_path / "not-ollama" / "manifests" / "x" / "y" / "latest"
    ref = f"ollama-manifest:{outside}"
    with pytest.raises(HTTPException) as excinfo:
        _resolve_model_identifier_for_request(LoadRequest(model_path = ref), operation = "load-model")
    assert excinfo.value.status_code == 400


def test_non_ollama_paths_take_the_existing_path(tmp_path):
    from models.inference import LoadRequest
    from routes.inference import _resolve_model_identifier_for_request

    identifier, label, native_grant_backed = _resolve_model_identifier_for_request(
        LoadRequest(model_path = "unsloth/model-GGUF"), operation = "load-model"
    )
    assert (identifier, label, native_grant_backed) == (
        "unsloth/model-GGUF",
        "unsloth/model-GGUF",
        False,
    )


def _safetensors_blob(
    root: Path,
    tensors: dict,
    metadata: dict | None = None,
) -> dict:
    """A one-blob safetensors file as Ollama writes each MLX tensor; returns its layer digest."""
    import hashlib
    import struct

    import numpy as np

    header, data = ({"__metadata__": metadata} if metadata else {}), b""
    for name, array in tensors.items():
        dtype = {np.uint32: "U32", np.uint8: "U8", np.float32: "F32", np.float16: "F16"}[
            array.dtype.type
        ]
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(data), len(data) + array.nbytes],
        }
        data += array.tobytes()
    text = json.dumps(header).encode()
    payload = struct.pack("<Q", len(text)) + text + data
    digest = hashlib.sha256(payload).hexdigest()
    (root / "blobs" / f"sha256-{digest}").write_bytes(payload)
    return {"digest": f"sha256:{digest}", "size": len(payload)}


def _write_ollama_mlx_store(
    root: Path,
    *,
    capabilities = ("completion", "vision"),
    extra_tensor: bool = False,
):
    import numpy as np

    rng = np.random.default_rng(0)
    (root / "blobs").mkdir(parents = True)
    u32 = lambda *shape: rng.integers(0, 2**32, shape, dtype = np.uint64).astype(np.uint32)
    u8 = lambda *shape: rng.integers(0, 256, shape, dtype = np.uint8)
    down, gate_up, embed = (
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.embed_tokens",
    )
    tensors = {
        # Small ModelOpt NVFP4 linear, packed into the shared shard.
        f"{down}.weight": (
            {
                f"{down}.weight": u32(4, 8),
                f"{down}.weight.scale": u8(4, 4),
                f"{down}.weight.global_scale": np.array(0.5, np.float32),
            },
            {"quant_type": "nvfp4", "group_size": "16"},
        ),
        # Fused experts carry no ".weight".
        gate_up: (
            {gate_up: u32(2, 4, 8), f"{gate_up}.scale": u8(2, 4, 4)},
            {"quant_type": "nvfp4", "group_size": "16"},
        ),
        # Over the packing threshold: an APFS clone with a rewritten header.
        f"{embed}.weight": (
            {f"{embed}.weight": u32(512, 640), f"{embed}.weight.scale": u8(512, 80)},
            {"quant_type": "mxfp8", "group_size": "32"},
        ),
        # Over the threshold and unchanged: a clone of the blob.
        "model.language_model.norm.weight": (
            {"model.language_model.norm.weight": rng.random(600_000).astype(np.float16)},
            None,
        ),
        "draft.model.norm.weight": ({"draft.model.norm.weight": np.ones(4, np.float16)}, None),
        "draft.model.embed_tokens.weight": (
            {"draft.model.embed_tokens.weight": rng.random(600_000).astype(np.float16)},
            None,
        ),
        "mtp.fc.weight": ({"mtp.fc.weight": np.ones(4, np.float16)}, None),
    }
    if extra_tensor:
        tensors["model.language_model.extra.weight"] = (
            {"model.language_model.extra.weight": np.ones(4, np.float16)},
            None,
        )
    layers = [
        {
            "mediaType": "application/vnd.ollama.image.tensor",
            "name": name,
            **_safetensors_blob(root, arrays, metadata),
        }
        for name, (arrays, metadata) in tensors.items()
    ]
    config = {
        "model_type": "gemma4",
        "text_config": {"hidden_size_per_layer_input": 256},
        "quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"},
    }
    for name, content in {
        "config.json": config,
        "tokenizer_config.json": {"bos_token": "<bos>"},
        "hf_quant_config.json": {},
        "draft/config.json": {
            "model_type": "gemma4_assistant",
            "architectures": ["Gemma4AssistantForCausalLM"],
        },
        "draft/tokenizer.json": {},
    }.items():
        text = json.dumps(content).encode()
        digest = __import__("hashlib").sha256(text).hexdigest()
        (root / "blobs" / f"sha256-{digest}").write_bytes(text)
        layers.append(
            {
                "mediaType": "application/vnd.ollama.image.json",
                "name": name,
                "digest": f"sha256:{digest}",
                "size": len(text),
            }
        )
    ollama_config = json.dumps(
        {
            "model_format": "safetensors",
            "capabilities": list(capabilities),
            "renderer": "gemma4",
            "file_type": "nvfp4",
        }
    ).encode()
    config_digest = __import__("hashlib").sha256(ollama_config).hexdigest()
    (root / "blobs" / f"sha256-{config_digest}").write_bytes(ollama_config)
    tag_file = root / "manifests" / "registry.ollama.ai" / "library" / "gemma4" / "e4b-mlx"
    tag_file.parent.mkdir(parents = True, exist_ok = True)
    tag_file.write_text(
        json.dumps({"config": {"digest": f"sha256:{config_digest}"}, "layers": layers}),
        encoding = "utf-8",
    )
    return tag_file, tensors


# The checkpoint is built from APFS clones, so only on macOS.
_APFS = pytest.mark.skipif(sys.platform != "darwin", reason = "APFS clonefile")


@pytest.fixture
def mlx_store(tmp_path, monkeypatch):
    from hub.services.models import ollama, ollama_mlx

    root = tmp_path / "ollama"
    tag_file, tensors = _write_ollama_mlx_store(root)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    monkeypatch.setattr(ollama_mlx, "mlx_host", lambda: True)
    return root, tag_file, tensors


def test_an_mlx_tag_lists_as_a_chat_only_safetensors_row(mlx_store, monkeypatch):
    from hub.services.models import ollama, ollama_mlx

    root, _, tensors = mlx_store
    (row,) = ollama.scan_ollama_dir(root)
    # Named by the checkpoint path a load builds, which the scan itself does not create.
    checkpoint = Path(os.path.realpath(root)) / ".studio_links"
    assert row.id == row.load_id and Path(row.id).relative_to(checkpoint).parts[1:] == (
        "mlx",
        "gemma4-e4b-mlx",
    )
    assert not checkpoint.exists()
    (alias := root.parent / "alias").symlink_to(root)
    assert [r.id for r in ollama.scan_ollama_dir(alias)] == [row.id]
    assert (row.model_id, row.model_format, row.display_name) == (
        "ollama/gemma4:e4b-mlx",
        "safetensors",
        "gemma4:e4b-mlx (MLX nvfp4)",
    )
    assert row.capabilities.can_chat and not row.capabilities.can_train
    assert row.size_bytes == sum(
        layer["size"]
        for layer in json.loads(mlx_store[1].read_text())["layers"]
        if layer["mediaType"].endswith("tensor")
        and not layer["name"].startswith(("draft.", "mtp."))
    )
    ref = ollama.ollama_mlx_ref_for_path(row.id)
    assert ollama.is_ollama_mlx_ref(ref) and ollama.ollama_model_ref_public_id(ref) == row.model_id
    with pytest.raises(ValueError, match = "MLX format"):
        ollama.materialize_ollama_model_ref(ref.replace("ollama-mlx:", "ollama-manifest:"))
    monkeypatch.setattr(ollama_mlx, "mlx_host", lambda: False)
    assert ollama.scan_ollama_dir(root) == []


def test_an_unloaded_mlx_tag_resolves_for_the_api(mlx_store):
    from core.inference import local_model_resolver as resolver
    from hub.services.models import ollama

    (row,) = ollama.scan_ollama_dir(mlx_store[0])
    entry = resolver._local_servable_entry(row.model_id, row)
    assert (entry.load_path, entry.is_gguf) == (row.id, False)
    assert resolver.local_servable_model(row) == (False, ())
    assert resolver._local_servable_entry(row.model_id, row.model_copy(update = {"id": "/x"})) is None
    # Nor once a tensor blob is gone, since the build would fail.
    tensor = next(
        layer
        for layer in json.loads(mlx_store[1].read_text())["layers"]
        if layer["mediaType"].endswith("tensor")
    )
    (mlx_store[0] / "blobs" / tensor["digest"].replace(":", "-")).unlink()
    assert resolver._local_servable_entry(row.model_id, row) is None


@pytest.mark.parametrize("key", ["auto_map", "model_file"])
def test_an_mlx_tag_needing_custom_code_is_not_offered_to_the_api(tmp_path, monkeypatch, key):
    import hashlib

    from core.inference import local_model_resolver as resolver
    from hub.services.models import ollama, ollama_mlx

    root = tmp_path / "ollama"
    tag_file = _write_ollama_mlx_store(root)[0]
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    monkeypatch.setattr(ollama_mlx, "mlx_host", lambda: True)
    manifest = json.loads(tag_file.read_text())
    layer = next(layer for layer in manifest["layers"] if layer.get("name") == "config.json")
    text = json.dumps({"model_type": "gemma4", key: {"AutoModel": "m.Model"}}).encode()
    digest = hashlib.sha256(text).hexdigest()
    (root / "blobs" / f"sha256-{digest}").write_bytes(text)
    layer.update(digest = f"sha256:{digest}", size = len(text))
    tag_file.write_text(json.dumps(manifest))
    (row,) = ollama.scan_ollama_dir(root)
    assert resolver._local_servable_entry(row.model_id, row) is None


def test_an_mlx_tag_the_checkpoint_cannot_serve_as_pulled_is_not_listed(tmp_path, monkeypatch):
    from hub.services.models import ollama, ollama_mlx

    monkeypatch.setattr(ollama_mlx, "mlx_host", lambda: True)
    _write_ollama_mlx_store(tmp_path / "ollama", capabilities = ("image",))
    assert ollama.scan_ollama_dir(tmp_path / "ollama") == []
    # Nor a tag whose SYSTEM prompt or preset messages the checkpoint would silently drop.
    for kind in ("system", "messages"):
        tag_file = _write_ollama_mlx_store(tmp_path / kind)[0]
        manifest = json.loads(tag_file.read_text())
        manifest["layers"].append(
            {
                "mediaType": f"application/vnd.ollama.image.{kind}",
                "digest": "sha256:" + "0" * 64,
                "size": 1,
            }
        )
        tag_file.write_text(json.dumps(manifest))
        assert ollama.scan_ollama_dir(tmp_path / kind) == []


@_APFS
def test_an_mlx_tag_materializes_as_a_standard_mlx_checkpoint(mlx_store):
    import numpy as np
    from safetensors.numpy import load_file

    from core.inference.chat_templates import load_bundled_chat_template
    from core.inference.mlx_speculative import _bundled_companion
    from hub.services.models import ollama

    root, tag_file, tensors = mlx_store
    row_id = ollama.scan_ollama_dir(root)[0].id
    ref = ollama.ollama_mlx_ref_for_path(row_id)
    checkpoint = Path(ollama.materialize_ollama_mlx_ref(ref))
    assert str(checkpoint) == row_id and checkpoint.parent.is_symlink()
    weight_map = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    down, gate_up, embed = (
        "model.language_model.layers.0.mlp.down_proj",
        "model.language_model.layers.0.mlp.experts.gate_up_proj",
        "model.language_model.embed_tokens",
    )
    renamed = {
        f"{down}.weight.scale": f"{down}.scales",
        f"{down}.weight.global_scale": f"{down}.weight_scale_2",
        f"{gate_up}.scale": f"{gate_up}.scales",
        f"{embed}.weight.scale": f"{embed}.scales",
    }
    expected = {
        renamed.get(name, name): array
        for arrays, _ in tensors.values()
        for name, array in arrays.items()
        if not name.startswith("draft.")
    }
    assert weight_map.keys() == expected.keys() and "mtp.fc.weight" in weight_map
    loaded = {}
    for shard in set(weight_map.values()):
        loaded.update(load_file(str(checkpoint / shard)))
    assert all(np.array_equal(loaded[name], array) for name, array in expected.items())
    # The assistant is a checkpoint of its own, named as its standalone release names it.
    draft = checkpoint / "draft"
    draft_map = json.loads((draft / "model.safetensors.index.json").read_text())["weight_map"]
    shard = load_file(str(draft / draft_map["model.embed_tokens.weight"]))
    assert draft_map.keys() == {"model.norm.weight", "model.embed_tokens.weight"}
    assert np.array_equal(*shard.values(), *tensors["draft.model.embed_tokens.weight"][0].values())
    assert (draft / "tokenizer.json").is_symlink()
    assert _bundled_companion(str(checkpoint)).path == str(draft)
    assert (
        weight_map[f"{down}.scales"] == weight_map[f"{gate_up}.scales"]
    )  # both small: one packed shard
    assert not (checkpoint / weight_map["model.language_model.norm.weight"]).is_symlink()
    assert not (
        checkpoint / weight_map[f"{embed}.scales"]
    ).is_symlink()  # renamed in a clone, never in the blob
    config = json.loads((checkpoint / "config.json").read_text())
    assert (
        config["quantization"] == {"group_size": 16, "bits": 4, "mode": "nvfp4"}
        and "quantization_config" not in config
    )
    assert not (checkpoint / "hf_quant_config.json").exists()
    assert (checkpoint / "chat_template.jinja").read_text() == load_bundled_chat_template(
        "gemma-4-edge.jinja"
    )
    assert ollama.ollama_manifest_ref_for_path(str(checkpoint)) is None
    built = (checkpoint / "config.json").stat().st_ino
    assert (
        ollama.materialize_ollama_mlx_ref(ref) == str(checkpoint)
        and (checkpoint / "config.json").stat().st_ino == built
    )

    # A re-pull builds anew behind the same path and keeps the build a resident model may read.
    first = Path(os.path.realpath(checkpoint))
    _write_ollama_mlx_store(root.parent / "repull", extra_tensor = True)
    for blob in (root.parent / "repull" / "blobs").iterdir():
        if not (root / "blobs" / blob.name).exists():
            blob.rename(root / "blobs" / blob.name)
    tag_file.write_text((root.parent / "repull" / tag_file.relative_to(root)).read_text())
    assert ollama.materialize_ollama_mlx_ref(ref) == str(checkpoint)
    second = Path(os.path.realpath(checkpoint))
    assert ollama.materialize_ollama_mlx_ref(ref) == str(checkpoint)  # again, as a validate does
    assert second != first and first.exists()
    assert (
        "model.language_model.extra.weight"
        in json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    )
    # The one before it goes.
    tag_file.write_text(json.dumps(json.loads(tag_file.read_text()), indent = 1))
    assert ollama.materialize_ollama_mlx_ref(ref) == str(checkpoint)
    assert not first.exists() and second.exists()
    assert os.path.realpath(checkpoint) not in (str(first), str(second))
    # The build kept for a resident model still reads once Ollama reclaims a blob it shared.
    norm = "model.language_model.norm.weight"
    layers = json.loads(tag_file.read_text())["layers"]
    digest = next(layer["digest"] for layer in layers if layer["name"] == norm)
    (root / "blobs" / digest.replace(":", "-")).unlink()
    kept = second
    shard = json.loads((kept / "model.safetensors.index.json").read_text())["weight_map"][norm]
    assert np.array_equal(load_file(str(kept / shard))[norm], tensors[norm][0][norm])


@_APFS
@pytest.mark.parametrize("name", ["ollama_tag.txt", "payload.py", "Config.json"])
def test_an_mlx_layer_named_as_a_generated_file_cannot_write_into_a_blob(mlx_store, name):
    import contextlib

    from hub.services.models import ollama

    root, tag_file, _ = mlx_store
    manifest = json.loads(tag_file.read_text(encoding = "utf-8"))
    shared = next(layer for layer in manifest["layers"] if layer["name"] == "tokenizer_config.json")
    manifest["layers"].append({**shared, "name": name})
    tag_file.write_text(json.dumps(manifest), encoding = "utf-8")
    blob = root / "blobs" / shared["digest"].replace(":", "-")
    before = blob.read_bytes()
    if not name.endswith(".json"):
        # Nor is a tag whose JSON layer could land as a non-JSON file (``model.py``) listed.
        assert ollama.scan_ollama_dir(root) == []
        return
    row_id = ollama.scan_ollama_dir(root)[0].id
    # On a case-sensitive volume the alias is just another file, so the build may succeed.
    with contextlib.suppress(OSError):
        ollama.materialize_ollama_mlx_ref(ollama.ollama_mlx_ref_for_path(row_id))
    assert blob.read_bytes() == before


@_APFS
def test_an_mlx_load_builds_the_checkpoint_it_names_and_advertises_the_tag(mlx_store):
    from hub.services.models import ollama
    from models.inference import LoadRequest
    import threading

    from routes.inference import _acquire_ollama_lease, _as_ollama_mlx_request

    root = mlx_store[0]
    row_id = ollama.scan_ollama_dir(root)[0].id
    # A lease is verified only after this runs, so a leased request builds nothing.
    leased = LoadRequest(model_path = row_id, native_path_lease = "unverified")
    assert _as_ollama_mlx_request(leased) is leased and not Path(row_id).exists()
    resolved = _as_ollama_mlx_request(LoadRequest(model_path = row_id))
    assert resolved.model_path == row_id and resolved._advertised_id == "ollama/gemma4:e4b-mlx"
    # The load's lease builds it, on the lease executor.
    assert not Path(row_id).exists()
    lease = _acquire_ollama_lease(row_id, None)
    assert (Path(row_id) / "config.json").is_file()
    # A second request does not wait on the lease a running load holds.
    worker = threading.Thread(target = _as_ollama_mlx_request, args = (LoadRequest(model_path = row_id),))
    worker.start()
    worker.join(timeout = 30)
    lease.release()
    assert not worker.is_alive()
    # Another spelling of the same place is the same tag.
    (alias := root.parent / "alias").symlink_to(root)
    respelled = alias / Path(row_id).relative_to(os.path.realpath(root))
    assert _as_ollama_mlx_request(LoadRequest(model_path = str(respelled)))._advertised_id == (
        "ollama/gemma4:e4b-mlx"
    )
    # Neither another name beside it, the same name under another links root, nor a copy is the tag.
    assert ollama.ollama_mlx_ref_for_path(str(Path(row_id).with_name("other"))) is None
    elsewhere = ollama._ollama_links_roots(root)[1] / Path(row_id).relative_to(
        Path(row_id).parents[2]
    )
    assert ollama.ollama_mlx_ref_for_path(str(elsewhere)) is None
    copied = shutil.copytree(row_id, root.parent / "copy" / "gemma4-e4b-mlx", symlinks = True)
    assert _as_ollama_mlx_request(LoadRequest(model_path = str(copied)))._advertised_id is None


@_APFS
@pytest.mark.parametrize("value", [None, "", {}])
def test_an_unset_tokenizer_template_still_gets_the_bundled_one(tmp_path, monkeypatch, value):
    import hashlib

    from core.inference.chat_templates import load_bundled_chat_template
    from hub.services.models import ollama, ollama_mlx

    root = tmp_path / "ollama"
    tag_file = _write_ollama_mlx_store(root)[0]
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    monkeypatch.setattr(ollama_mlx, "mlx_host", lambda: True)
    manifest = json.loads(tag_file.read_text())
    layer = next(
        layer for layer in manifest["layers"] if layer.get("name") == "tokenizer_config.json"
    )
    text = json.dumps({"bos_token": "<bos>", "chat_template": value}).encode()
    digest = hashlib.sha256(text).hexdigest()
    (root / "blobs" / f"sha256-{digest}").write_bytes(text)
    layer.update(digest = f"sha256:{digest}", size = len(text))
    tag_file.write_text(json.dumps(manifest))
    row_id = ollama.scan_ollama_dir(root)[0].id
    checkpoint = Path(ollama.materialize_ollama_mlx_ref(ollama.ollama_mlx_ref_for_path(row_id)))
    assert (checkpoint / "chat_template.jinja").read_text() == load_bundled_chat_template(
        "gemma-4-edge.jinja"
    )


@_APFS
def test_an_mlx_load_holds_the_tag_until_it_ends(mlx_store):
    from hub.services.models import ollama
    from routes.inference import _acquire_ollama_lease

    root, tag_file, _ = mlx_store
    row_id = ollama.scan_ollama_dir(root)[0].id
    lease = _acquire_ollama_lease(row_id, None)
    lock = ollama._materialization_lock(tag_file, root)
    try:
        assert lease.path == row_id and (Path(row_id) / "config.json").is_file()
        assert not lock.acquire(blocking = False)  # a delete waits for the load
    finally:
        lease.release()
    assert lock.acquire(blocking = False)
    lock.release()
    assert _acquire_ollama_lease(row_id, "unverified") is None


@_APFS
def test_an_mlx_config_naming_a_model_file_is_refused(mlx_store):
    import hashlib

    from hub.services.models import ollama

    root, tag_file, _ = mlx_store
    manifest = json.loads(tag_file.read_text())
    layer = next(layer for layer in manifest["layers"] if layer.get("name") == "config.json")
    text = json.dumps({"model_type": "gemma4", "model_file": "model.py"}).encode()
    digest = hashlib.sha256(text).hexdigest()
    (root / "blobs" / f"sha256-{digest}").write_bytes(text)
    layer.update(digest = f"sha256:{digest}", size = len(text))
    tag_file.write_text(json.dumps(manifest))
    row_id = ollama.scan_ollama_dir(root)[0].id
    with pytest.raises(ValueError, match = "model_file"):
        ollama.materialize_ollama_mlx_ref(ollama.ollama_mlx_ref_for_path(row_id))


def test_an_mlx_tag_builds_only_after_the_access_check():
    import inspect

    from routes import inference

    for route in (inference._load_model_impl, inference.validate_model):
        source = inspect.getsource(route)
        assert source.index("require_model_access") < source.index("_as_ollama_mlx_request")
    assert "_as_ollama_mlx_request" not in inspect.getsource(inference.load_model_gated)


def test_an_unrelated_load_does_not_take_the_ollama_lease_pool(monkeypatch):
    import asyncio
    from contextlib import ExitStack

    from models.inference import LoadRequest
    from routes import inference

    class Refusing:
        def submit(self, *args, **kwargs):
            raise AssertionError("an unrelated load queued on the Ollama lease pool")

    monkeypatch.setattr(inference, "_OLLAMA_LEASE_EXECUTOR", Refusing())
    for path in ("unsloth/Qwen3-0.6B", "/models/qwen/config-dir"):
        with ExitStack() as stack:
            lease = inference._lease_ollama_model_ref(
                LoadRequest(model_path = path), operation = "load-model", stack = stack
            )
            assert asyncio.run(lease) is None


@_APFS
def test_an_image_request_probes_an_unbuilt_mlx_tag_from_its_checkpoint(mlx_store, monkeypatch):
    from hub.services.models import ollama
    from routes import inference

    row_id = ollama.scan_ollama_dir(mlx_store[0])[0].id
    seen = []
    monkeypatch.setattr(
        inference,
        "_target_is_vision",
        lambda path, *_a: seen.append(path) or (Path(path) / "config.json").is_file(),
    )
    assert not Path(row_id).exists()
    assert inference._target_accepts_request_input(row_id, False, True, False)
    assert seen == [row_id]
