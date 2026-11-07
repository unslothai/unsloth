# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
import asyncio
from dataclasses import replace
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
import typer
from typer.testing import CliRunner

import unsloth_cli.commands.start as start
from unsloth_cli import _model_catalog as catalog
from unsloth_cli._inference import ensure_studio_backend_path
from unsloth_cli.tests.test_start import fake_studio  # noqa: F401
from unsloth_cli.tests.test_start_model_picker import (  # noqa: F401
    _autostart,
    _empty_server,
    _loads,
    _route,
    picker,
)

REAL_CATALOG = catalog.list_chat_models
MODES = ["idle-server", "auto-start"]


def _gguf(path: Path) -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_bytes(b"GGUF" + bytes(4092))
    return path


def _only_cache_and(monkeypatch, **sources):
    monkeypatch.setattr(catalog, "list_chat_models", REAL_CATALOG)
    for source in ("trained_entries", "exported_entries", "local_folder_entries"):
        monkeypatch.setattr(catalog, source, sources.get(source, lambda **kwargs: []))


@pytest.fixture()
def cached_gguf_pick(tmp_path, monkeypatch, picker):
    ensure_studio_backend_path(seed_cache_env = False)
    repo = tmp_path / "models--Org--Tiny-GGUF"
    snapshot = repo / "snapshots" / ("a" * 40)
    for quant in ("Q4_K_M", "Q8_0"):
        _gguf(snapshot / f"tiny-{quant}.gguf")
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    row = {"repo_id": "Org/Tiny-GGUF", "cache_path": str(repo)}
    target = catalog._cached_gguf_load_id(row)
    assert target == str(snapshot / "tiny-Q4_K_M.gguf")
    monkeypatch.setattr(catalog, "_cached_catalog_rows", lambda: ([row], []))
    _only_cache_and(monkeypatch)
    return target


def _pick(mode, variant, fake_studio, monkeypatch, picker):
    """Press Enter in the picker; return the (model, gguf_variant) the load would use."""
    _empty_server(fake_studio, monkeypatch)
    picker["answers"] = [""]
    load = start.LoadOptions(gguf_variant = variant)
    if mode == "idle-server":
        start._connect(None, None, load, launch = False, gguf_only = True)
        [payload] = _loads(fake_studio)
        assert payload["require_idle"] is True
        return payload["model_path"], payload.get("gguf_variant")
    attempts = _autostart(monkeypatch)
    with pytest.raises(typer.Exit) as error:
        start._connect(None, None, load, serve = True, launch = True, gguf_only = True)
    assert error.value.exit_code == 7
    [(model, used)] = attempts
    assert used._replace(gguf_variant = variant) == load
    return model, used.gguf_variant


@pytest.mark.parametrize("mode", MODES)
def test_picked_file_rejects_a_conflicting_quant_before_load(
    cached_gguf_pick, picker, fake_studio, monkeypatch, capsys, mode
):
    # A row for an individual file must never switch to a sibling.
    entry = catalog.ModelEntry("Downloaded", "tiny", "gguf", cached_gguf_pick)
    monkeypatch.setattr(catalog, "list_chat_models", lambda **kwargs: [entry])
    _empty_server(fake_studio, monkeypatch)
    picker["answers"] = [""]
    auto = mode == "auto-start"
    attempts = _autostart(monkeypatch) if auto else []
    load = start.LoadOptions(gguf_variant = "Q8_0")

    with pytest.raises(typer.Exit) as error:
        start._connect(None, None, load, serve = auto, launch = auto, gguf_only = True)

    assert error.value.exit_code == 1
    assert not attempts and not _loads(fake_studio)
    output = "".join(capsys.readouterr())
    assert "--gguf-variant Q8_0" in output and "--model" in output


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("variant", ["Q4_K_M", "q4_k_m", "tiny-Q4_K_M", "tiny-Q4_K_M.gguf", "Q8_0"])
def test_picked_repo_resolves_the_requested_quant(
    cached_gguf_pick, picker, fake_studio, monkeypatch, mode, variant
):
    expected = Path(cached_gguf_pick)
    if variant == "Q8_0":
        expected = expected.with_name("tiny-Q8_0.gguf")
    assert _pick(mode, variant, fake_studio, monkeypatch, picker) == (str(expected), variant)


@pytest.mark.parametrize("source", ["models_dir", "custom", "export"])
@pytest.mark.parametrize("mode", MODES)
def test_picked_directory_resolves_the_requested_quant(
    tmp_path, picker, fake_studio, monkeypatch, mode, source
):
    ensure_studio_backend_path(seed_cache_env = False)
    from hub.services.models.local_inventory import _classify_local_path, _scan_models_dir
    from utils import models

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    folder = tmp_path / ("exports" if source == "export" else "models") / "tiny"
    default, requested = (_gguf(folder / f"tiny-{q}.gguf") for q in ("Q4_K_M", "Q8_0"))
    exports, rows = [], []
    if source == "export":
        exports = models.scan_exported_models()
    else:
        rows = (
            _scan_models_dir(folder.parent)
            if source == "models_dir"
            else _classify_local_path(folder, "custom")
        )
    monkeypatch.setattr(models, "scan_exported_models", lambda: exports)
    monkeypatch.setattr(catalog, "_local_catalog_rows", lambda: rows)
    monkeypatch.setattr(catalog, "trained_entries", lambda: [])
    monkeypatch.setattr(catalog, "_cached_catalog_rows", lambda: ([], []))
    monkeypatch.setattr(catalog, "list_chat_models", REAL_CATALOG)
    assert [entry.model for entry in REAL_CATALOG()] == [str(default)]

    assert _pick(mode, "Q8_0", fake_studio, monkeypatch, picker) == (str(requested), "Q8_0")


def test_cached_picker_lists_only_complete_matching_quants(cached_gguf_pick):
    snapshot = Path(cached_gguf_pick).parent
    _gguf(snapshot / "tiny-Q6_K-00001-of-00002.gguf")
    assert catalog.list_chat_models(gguf_variant = "Q6_K") == []
    assert catalog.list_chat_models(gguf_variant = "Q5_K_M") == []

    build = _gguf(snapshot / "other-build" / "tiny-Q8_0.gguf")
    [entry] = catalog.list_chat_models(gguf_variant = "other-build/tiny-Q8_0.gguf")
    assert (entry.model, entry.file_variant) == (str(build), "tiny-Q8_0.gguf")


def test_cached_picker_checks_only_the_selected_shard_family(cached_gguf_pick):
    snapshot = Path(cached_gguf_pick).parent
    empty = snapshot / "a-Q8_0.gguf"
    empty.touch()
    first, second = (_gguf(snapshot / f"b-Q8_0-0000{i}-of-00002.gguf") for i in (1, 2))

    assert [e.model for e in REAL_CATALOG(gguf_variant = first.name)] == [str(first)]
    assert REAL_CATALOG(gguf_variant = empty.name) == []
    second.write_bytes(b"")
    assert REAL_CATALOG(gguf_variant = first.name) == []


def test_auto_start_keeps_the_resolved_duplicate_family(
    cached_gguf_pick, picker, fake_studio, monkeypatch
):
    directory = Path(cached_gguf_pick).parent / "Q8_0"
    _gguf(directory / "a-Q8_0.gguf")
    picked = _gguf(directory / "b-Q8_0.gguf")

    model, variant = _pick("auto-start", "Q8_0/b-Q8_0.gguf", fake_studio, monkeypatch, picker)

    assert (model, variant) == (str(picked), "b-Q8_0.gguf")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("location", ["remembered-cache", "sibling-snapshot"])
def test_picked_repo_finds_the_requested_quant_outside_main(
    tmp_path, monkeypatch, picker, fake_studio, mode, location
):
    ensure_studio_backend_path(seed_cache_env = False)
    from huggingface_hub import scan_cache_dir
    from hub.services.models import cache_inventory
    from hub.utils.download_manifest import ExpectedFile, write_manifest
    from utils import hf_cache_settings

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "home"))
    active, remembered = tmp_path / "active", tmp_path / "remembered"
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [active, remembered])
    paths = hf_cache_settings.get_hf_cache_paths()
    monkeypatch.setattr(
        hf_cache_settings, "get_hf_cache_paths", lambda: replace(paths, hub_cache = active)
    )
    q8_root = remembered if location == "remembered-cache" else active
    for root, revision, quant in [(q8_root, "b", "Q8_0"), (active, "a", "Q4_K_M")]:
        repo = root / "models--Org--Tiny-GGUF"
        _gguf(repo / "snapshots" / (revision * 40) / f"tiny-{quant}.gguf")
        (repo / "refs").mkdir(exist_ok = True)
        (repo / "refs" / "main").write_text(revision * 40)
    expected = str(q8_root / "models--Org--Tiny-GGUF" / "snapshots" / ("b" * 40) / "tiny-Q8_0.gguf")
    scans = [scan_cache_dir(root) for root in (active, remembered) if root.is_dir()]
    monkeypatch.setattr(cache_inventory, "all_hf_cache_scans", lambda: scans)
    [row], _ = catalog._cached_catalog_rows()
    assert Path(row["cache_path"]).parent == active
    assert row["load_id"] == row["repo_id"] == "Org/Tiny-GGUF"
    # A row pinned to one snapshot must not borrow a quant from another.
    pinned = Path(row["cache_path"]) / "snapshots" / ("a" * 40)
    assert catalog._cached_variant_entry({**row, "load_id": str(pinned)}, "Q8_0") is None
    _only_cache_and(monkeypatch)

    assert _pick(mode, "Q8_0", fake_studio, monkeypatch, picker) == (expected, "Q8_0")

    # A positive-sized file can still be shorter than its recorded download.
    manifest = [ExpectedFile(Path(expected).name, 8192)]
    commit = Path(expected).parent.name
    assert write_manifest(
        "model", row["repo_id"], "Q8_0", manifest, hub_cache = q8_root, commit_hash = commit
    )
    assert catalog._cached_variant_entry(row, "Q8_0") is None


@pytest.mark.parametrize(
    ("agent", "filename", "selector"),
    [
        ("claude", "tiny-UD-NVFP4.gguf", "UD-NVFP4"),
        ("codex", "tiny-NVFP4-4.25bpw.gguf", "NVFP4-4.25bpw"),
        ("claude", "NVFP4/model.gguf", "NVFP4"),
        ("codex", "Q6_K/build/model.gguf", "Q6_K"),
        ("claude", "Q8_0/b-Q8_0.gguf", "Q8_0/b-Q8_0.gguf"),
        ("codex", "other-build/tiny-Q8_0.gguf", "other-build/tiny-Q8_0.gguf"),
    ],
)
def test_agent_preload_agrees_with_the_local_loader(
    cached_gguf_pick, fake_studio, monkeypatch, picker, agent, filename, selector
):
    from hub.services.models.gguf_variants import get_gguf_variants_response
    from utils.models.model_config import _find_local_gguf_by_variant

    snapshot = Path(cached_gguf_pick).parent
    expected = _gguf(snapshot / filename)
    if filename == "Q8_0/b-Q8_0.gguf":
        _gguf(expected.parent / "a-Q8_0.gguf")
    assert _find_local_gguf_by_variant(str(snapshot), selector) == str(expected)
    _empty_server(fake_studio, monkeypatch)

    def variants(url, key, payload):
        repo_id = parse_qs(urlparse(url).query)["repo_id"][0]
        return asyncio.run(get_gguf_variants_response(repo_id)).model_dump()

    _route(monkeypatch, {"/api/models/gguf-variants": variants})
    picker["answers"] = [""]

    result = CliRunner().invoke(start.start_app, [agent, "--gguf-variant", selector, "--no-launch"])

    assert result.exit_code == 0, result.output
    [payload] = _loads(fake_studio)
    assert payload["model_path"] == str(expected)
    assert _find_local_gguf_by_variant(payload["model_path"], payload["gguf_variant"]) == str(
        expected
    )


def test_loose_gguf_files_follow_the_requested_quant(tmp_path, monkeypatch):
    ensure_studio_backend_path(seed_cache_env = False)
    from hub.services.models.local_inventory import _scan_models_dir
    from utils.models.model_config import _direct_gguf_for_variant

    root = tmp_path / "models"
    for name in ("tiny-Q4_K_M.gguf", "tiny-Q8_0.gguf"):
        _gguf(root / name)
    first, second = (_gguf(root / f"big-Q6_K-0000{i}-of-00002.gguf") for i in (1, 2))
    rows = _scan_models_dir(root)
    monkeypatch.setattr(catalog, "_local_catalog_rows", lambda: rows)

    assert str(root / "tiny-Q4_K_M.gguf") in [e.model for e in catalog.local_folder_entries()]
    q8 = catalog.local_folder_entries(gguf_variant = "Q8_0")
    assert [entry.model for entry in q8] == [str(root / "tiny-Q8_0.gguf")]
    # Naming any shard yields the loadable first shard.
    [entry] = catalog.local_folder_entries(gguf_variant = second.name)
    assert entry.model == str(first)
    assert _direct_gguf_for_variant(entry.model, entry.file_variant) == str(first)
