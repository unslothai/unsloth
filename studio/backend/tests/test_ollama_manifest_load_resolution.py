# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""POST /load resolves opaque ``ollama-manifest:`` inventory references.

The read-only hub inventory scan returns Ollama rows whose load id is an
``ollama-manifest:`` reference (hub/services/models/ollama.py); the load path
owns the filesystem write that turns one into a loadable ``.gguf`` link. These
tests pin that hand-off at ``_resolve_model_identifier_for_request``, which both
the load and validate routes go through.
"""

from __future__ import annotations

import json
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
    from types import SimpleNamespace

    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend
    from hub.services.models import ollama

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
    import threading

    from hub.services.models import ollama

    root = tmp_path / "ollama-materialization-lease"
    tag_file = _write_ollama_store(root)
    monkeypatch.setattr(ollama, "ollama_model_dirs", lambda: [root])
    ref = f"ollama-manifest:{quote(str(tag_file), safe = '')}"

    alternate_ref = ref.replace("%2F", "%2f")
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
    import asyncio
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import ExitStack

    from hub.services.models import ollama
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
    from types import SimpleNamespace

    from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend
    from hub.services.models import ollama

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
    from types import SimpleNamespace

    from models.inference import LoadRequest
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


# Modelfile metadata nearly every pulled model carries.
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


@pytest.mark.parametrize("unsupported", _UNSUPPORTED_RUNTIME_LAYERS)
def test_one_unsupported_layer_still_withholds_beside_the_metadata(
    tmp_path, monkeypatch, unsupported
):
    # Pins each unsupported type on its own. The withholding test above carries both at
    # once, so admitting just one of them back into the loadable set would still pass it.
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
