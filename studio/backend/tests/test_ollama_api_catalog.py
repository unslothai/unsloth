# SPDX-License-Identifier: AGPL-3.0-only

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from core.inference import local_model_resolver as resolver
from hub.services.models import account_access, ollama
from models.inference import LoadRequest
from routes import inference as inf, models as models_route
from storage import studio_db
from utils import paths, hf_cache_settings
from studio.backend.tests.test_legacy_ollama_source import _write_ollama_store
from studio.backend.tests.test_openai_auto_switch import (
    _FakeBackend,
    _reset_keepwarm,
    _run_hook,
    _wired,
)


_NO_ORCHESTRATOR = SimpleNamespace(active_model_name=None, models={})


def _refusing_to_scan():
    raise AssertionError("the alias lookup must not scan")


def _retag(store: Path, tag: str, digest: str) -> None:
    (store / "blobs" / f"sha256-{digest}").write_bytes(b"GGUF-not-really-" + digest[:4].encode())
    layer = {"mediaType": "application/vnd.ollama.image.model", "digest": f"sha256:{digest}"}
    (store / "manifests/registry.ollama.ai/library/llama3" / tag).write_text(
        json.dumps({"config": {}, "layers": [layer]}), encoding="utf-8"
    )


@pytest.fixture
def store(tmp_path, monkeypatch):
    root = tmp_path / "ollama"
    _write_ollama_store(root)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(tmp_path)
    for module in (paths, ollama):
        monkeypatch.setattr(module, "ollama_model_dirs", lambda: [root])
    for name in ("lmstudio_model_dirs", "hermes_model_dirs"):
        monkeypatch.setattr(paths, name, lambda: [])
    for name in ("legacy_hf_cache_dir", "hf_default_cache_dir"):
        monkeypatch.setattr(paths, name, lambda: empty)
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: empty)
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [])
    monkeypatch.setattr(inf, "_CATALOG_CACHE", {"at": 0.0, "models": []})
    monkeypatch.setattr(inf, "_SERVABLE_SCAN_CACHE", {"entry": None})
    monkeypatch.setattr(inf, "_peek_inference_backend", lambda: None)
    for empty_index in ("_media_model_objects", "_stt_model_objects"):
        monkeypatch.setattr(inf, empty_index, lambda *args: [])
    resolver.invalidate_index()
    _reset_keepwarm()
    yield root
    resolver.invalidate_index()
    _reset_keepwarm()


@pytest.fixture
def aliased(store):
    tags = store / "manifests/registry.ollama.ai/library/llama3"
    (tags / "8b").write_bytes((tags / "latest").read_bytes())
    return {r.model_id: r.id for r in models_route._scan_ollama_dir(store, materialize_links=False)}


@pytest.fixture
def serving(monkeypatch):
    def _pin(backend):
        monkeypatch.setattr(inf, "get_llama_cpp_backend", lambda: backend)
        monkeypatch.setattr(inf, "get_inference_backend", lambda: _NO_ORCHESTRATOR)
        return backend

    return _pin


def test_the_compat_inventory_still_hands_out_a_gguf_path(store):
    """Its readers treat a row's id and path as filenames; a suffixless blob is not loadable."""
    (row,) = models_route.collect_local_models(Path("models").resolve())
    assert row.model_id == "ollama/llama3:latest"
    assert row.id == row.path
    assert row.path.endswith(".gguf")
    assert Path(row.path).is_file()


def _store_contents(store):
    def _entry(p):
        stat = p.lstat()
        return str(p.relative_to(store)), stat.st_size, stat.st_mtime_ns, p.is_symlink()

    return sorted(_entry(p) for p in store.rglob("*"))


def test_catalog_and_resolver_are_read_only_and_share_the_public_id(store, monkeypatch):
    before = _store_contents(store)
    (row,) = models_route.collect_local_models(
        Path("models").resolve(), materialize_ollama_links=False
    )
    assert row.model_id == "ollama/llama3:latest"
    assert row.id.startswith("ollama-manifest:")
    assert row.model_format == "gguf"
    assert resolver.local_servable_model(row) == (True, ())
    assert resolver.resolve_local_gguf(row.model_id) == (row.id, None, row.model_id)
    assert _store_contents(store) == before
    monkeypatch.setattr(studio_db, "list_scan_folders", lambda: [{"path": str(store)}])
    rows = models_route.collect_local_models(
        Path("models").resolve(), materialize_ollama_links=False
    )
    assert len(rows) == 1


def test_a_tag_being_loaded_stays_in_the_inventory(store):
    ref = models_route._scan_ollama_dir(store, materialize_links=False)[0].id
    lease = ollama.acquire_ollama_model_ref(ref)
    try:
        rows = models_route._scan_ollama_dir(store, materialize_links=True)
    finally:
        lease.release()
    assert [row.path for row in rows] == [lease.path]
    assert Path(lease.path).is_file()


def _resident(ref: str, *, advertised: str = "") -> _FakeBackend:
    from core.inference.llama_cpp import LlamaCppBackend

    materialized = ollama.materialize_ollama_model_ref(ref)
    backend = _FakeBackend(ref, advertised_id=advertised or ollama.ollama_model_ref_public_id(ref))
    backend.gguf_path = materialized
    backend._is_audio, backend._audio_type = False, None
    for field in ("context_length", "max_context_length", "native_context_length"):
        setattr(backend, field, 4096)
    backend._gguf_load_identity = LlamaCppBackend._gguf_load_source_identity(materialized)
    return backend


def test_a_repulled_tag_is_not_reported_loaded_while_it_cannot_be_answered(store, serving):
    ref = models_route._scan_ollama_dir(store, materialize_links=False)[0].id
    backend = serving(_resident(ref))
    assert inf._loaded_satisfies("ollama/llama3:latest") is True
    assert inf._loaded_satisfies("ollama/llama3:latest:Q8_0") is False
    assert [m["id"] for m in inf._openai_model_objects()] == ["ollama/llama3:latest"]
    # The load identity is recorded before the server answers and outlives a failed load.
    backend.is_loaded = False
    assert inf._resolves_to_resident(ref, llama_only=True) is False
    backend.is_loaded = True
    _retag(store, "latest", "c" * 64)
    assert inf._loaded_satisfies("ollama/llama3:latest") is False
    assert inf._openai_model_objects() == []
    # A manifest mid-pull reads as nothing: withhold the row, do not fail the listing.
    (store / "manifests/registry.ollama.ai/library/llama3/latest").write_text("truncated")
    assert inf._loaded_satisfies("ollama/llama3:latest") is False
    assert inf._openai_model_objects() == []


def _rewritten(model_path: str) -> str:
    return inf._as_ollama_manifest_request(LoadRequest(model_path=model_path)).model_path


def test_a_link_is_loaded_as_the_tag_that_materialized_it(store, monkeypatch, serving):
    """Its directory names the tag, so a selection stored under an older link name resolves too."""
    ref = models_route._scan_ollama_dir(store, materialize_links=False)[0].id
    link = Path(ollama.materialize_ollama_model_ref(ref))
    data = json.loads((store / "manifests/registry.ollama.ai/library/llama3/latest").read_text())
    for named in (link, link.with_name("llama3-latest-Q4_K_M.gguf")):
        assert _rewritten(str(named)) == ref
    assert _rewritten(str(store / "blobs")) == str(store / "blobs")
    # An authorized request keeps its own path: the tag may name blobs it never judged.
    monkeypatch.setattr(account_access, "managed_account", lambda: True)
    assert _rewritten(str(link)) == str(link)
    monkeypatch.setattr(account_access, "managed_account", lambda: False)
    leased = LoadRequest(model_path=str(link), native_path_lease="lease")
    assert inf._as_ollama_manifest_request(leased).model_path == str(link)
    # A tag that no longer reads leaves a link that still resolves alone.
    manifest = store / "manifests/registry.ollama.ai/library/llama3/latest"
    manifest.write_text("truncated")
    assert _rewritten(str(link)) == str(link)
    manifest.write_text(json.dumps(data))
    serving(_resident(ref))
    assert [(m["id"], m["loaded"]) for m in asyncio.run(inf._openai_catalog_objects())] == [
        ("ollama/llama3:latest", True)
    ]


def test_the_switch_gate_follows_the_blobs_under_an_alias(store, aliased, monkeypatch):
    """A tag is a mutable name: it answers for the resident model only while it still names it."""
    alias = aliased["ollama/llama3:8b"]
    backend = _resident(aliased["ollama/llama3:latest"])
    _, recorder = _wired(monkeypatch, backend, (alias, None, "ollama/llama3:8b"))
    loader = inf._load_model_impl

    async def _load_and_record_identity(request, *args, **kwargs):
        from core.inference.llama_cpp import LlamaCppBackend

        await loader(request, *args, **kwargs)
        backend._gguf_load_identity = LlamaCppBackend._gguf_load_source_identity(
            ollama.materialize_ollama_model_ref(request.model_path)
        )

    monkeypatch.setattr(inf, "_load_model_impl", _load_and_record_identity)
    _run_hook("ollama/llama3:8b")
    assert recorder.calls == []
    _retag(store, "8b", "c" * 64)
    _run_hook("ollama/llama3:8b")
    assert [call.model_path for call in recorder.calls] == [alias]


def test_an_alias_answers_for_the_resident_blobs_until_its_own_tag_moves(
    store, aliased, monkeypatch, serving
):
    """Adoption records a name, and a re-pull replaces the weights under a name it keeps."""
    link = Path(ollama.materialize_ollama_model_ref(aliased["ollama/llama3:latest"]))
    alias_link = Path(ollama.materialize_ollama_model_ref(aliased["ollama/llama3:8b"]))
    # The advertised id a request adopting the resident model under another name leaves behind.
    serving(_resident(aliased["ollama/llama3:latest"], advertised="ollama/llama3:8b"))

    def _satisfied() -> bool:
        resolver.invalidate_index()
        resolver.resolve_local_gguf("ollama/llama3:8b")
        with monkeypatch.context() as warm_only:
            # Scanning here would run ahead of the bounded cold-index wait its callers apply.
            warm_only.setattr(resolver, "_index", _refusing_to_scan)
            return inf._loaded_satisfies("ollama/llama3:8b")

    assert _satisfied() is True
    assert [m["id"] for m in inf._openai_model_objects()] == ["ollama/llama3:8b"]
    # Whichever spelling the inventory gave a recipe: this link, an older link name, the alias.
    for stored in (link, link.with_name("llama3-latest-Q4_K_M.gguf"), alias_link):
        assert _validate(str(stored)).resident is True
    _retag(store, "latest", "d" * 64)
    assert _satisfied() is True
    assert [m["id"] for m in inf._openai_model_objects()] == ["ollama/llama3:8b"]

    # Re-pulled: the spelling is the same and the weights under it are not.
    _retag(store, "8b", "c" * 64)
    assert _satisfied() is False
    assert inf._openai_model_objects() == []
    assert _validate(str(alias_link)).resident is False


def _validate(
    model_path: str,
    *,
    variant=None,
    gguf_file=...,
):
    from models.inference import ValidateModelRequest

    config = SimpleNamespace(
        identifier=model_path,
        display_name=Path(model_path).name,
        is_gguf=True,
        gguf_file=model_path if gguf_file is ... else gguf_file,
        is_lora=False,
        is_vision=False,
    )
    request = ValidateModelRequest(model_path=model_path, gguf_variant=variant)
    with patch.object(inf.ModelConfig, "from_identifier", return_value=config):
        return asyncio.run(inf.validate_model(request, current_subject="t"))


def test_validate_answers_for_the_artifact_it_resolved(store, monkeypatch, tmp_path, serving):
    """The variant picks the quant and a lease picks the file, so the identifier decides neither."""
    from hub.utils import gguf as gguf_utils

    def quant(name):
        return str(tmp_path / f"model-{name}.gguf")

    def loaded(
        identifier,
        path,
        variant=None,
    ):
        return SimpleNamespace(
            is_loaded=True,
            model_identifier=identifier,
            _openai_advertised_id=None,
            hf_variant=variant,
            gguf_path=path,
        )

    monkeypatch.setattr(gguf_utils, "resolve_local_gguf_path", lambda _id, name: quant(name))
    # By repo id, then out of the directory the quants share. Q8_0 is loaded both times.
    for identifier, by_file in (("org/model-GGUF", False), (str(tmp_path), True)):
        serving(loaded(identifier, quant("Q8_0"), variant="Q8_0"))
        for name, expected in (("Q8_0", True), ("Q4_K_M", False)):
            asked = quant(name) if by_file else None
            assert _validate(identifier, variant=name, gguf_file=asked).resident is expected

    one, other = (tmp_path / name for name in ("loaded.gguf", "other.gguf"))
    for path in (one, other):
        path.write_bytes(b"GGUF")
    serving(loaded(str(one), str(one)))
    for granted, expected in ((other, False), (one, True)):
        monkeypatch.setattr(
            inf,
            "_resolve_model_identifier_for_request",
            lambda request, granted=granted, **kwargs: (str(granted), granted.name, True),
        )
        assert _validate(str(one)).resident is expected


def test_a_symlinked_manifests_dir_still_resolves(tmp_path, monkeypatch):
    """A reference carries the canonical path, a scan the spelling it walked."""
    staging, elsewhere, root = (tmp_path / name for name in ("staging", "elsewhere", "ollama"))
    _write_ollama_store(staging)
    elsewhere.mkdir()
    root.mkdir()
    (staging / "manifests").rename(elsewhere / "manifests")
    (staging / "blobs").rename(root / "blobs")
    (root / "manifests").symlink_to(elsewhere / "manifests", target_is_directory=True)
    for module in (paths, ollama):
        monkeypatch.setattr(module, "ollama_model_dirs", lambda: [root])

    (row,) = models_route._scan_ollama_dir(root, materialize_links=False)
    assert ollama.ollama_model_ref_files(row.id)[0].startswith(str(root / "blobs"))


def test_stop_loading_reaches_a_tag_the_load_renamed(store):
    """Chat holds the link it picked; the load runs as the tag. Stop must still find it."""
    ref = models_route._scan_ollama_dir(store, materialize_links=False)[0].id
    link = models_route._scan_ollama_dir(store, materialize_links=True)[0].id
    assert link != ref
    assert inf._names_the_resident_model(ref, link)
    assert inf._names_the_loading_model(ref, link)
    assert not inf._names_the_resident_model(ref, str(store / "blobs" / "nothing.gguf"))
