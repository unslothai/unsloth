# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""A picked GGUF file must not silently override an explicit quantization choice."""

from dataclasses import replace
from pathlib import Path

import pytest
import typer

import unsloth_cli.commands.start as start
from unsloth_cli import _model_catalog as catalog
from unsloth_cli._inference import ensure_studio_backend_path
from unsloth_cli.tests.test_start import fake_studio
from unsloth_cli.tests.test_start_model_picker import picker, _empty_server, _loads


REAL_CATALOG = catalog.list_chat_models


@pytest.fixture()
def cached_gguf_pick(tmp_path, monkeypatch, picker):
    ensure_studio_backend_path(seed_cache_env = False)
    repo = tmp_path / "models--Org--Tiny-GGUF"
    snapshot = repo / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("a" * 40)
    for quant in ("Q4_K_M", "Q8_0"):
        (snapshot / f"tiny-{quant}.gguf").write_bytes(b"GGUF" + b"\0" * 4092)
    target = catalog._cached_gguf_load_id({"repo_id": "Org/Tiny-GGUF", "cache_path": str(repo)})
    assert target == str(snapshot / "tiny-Q4_K_M.gguf")
    row = {"repo_id": "Org/Tiny-GGUF", "cache_path": str(repo)}
    monkeypatch.setattr(catalog, "_cached_catalog_rows", lambda: ([row], []))
    monkeypatch.setattr(catalog, "list_chat_models", REAL_CATALOG)
    for source in ("trained_entries", "exported_entries", "local_folder_entries"):
        monkeypatch.setattr(catalog, source, lambda **kwargs: [])
    return target


def _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts):
    _empty_server(fake_studio, monkeypatch)
    picker["answers"] = [""]
    if mode == "auto-start":
        monkeypatch.setattr(start, "find_studio_server", lambda: None)
        monkeypatch.delenv("UNSLOTH_STUDIO_URL", raising = False)

        def require(model, load, **kwargs):
            attempts.append((model, load))
            raise typer.Exit(code = 7)

        monkeypatch.setattr(start, "_require_studio", require)
    return start._connect(
        None,
        None,
        load,
        serve = mode == "auto-start",
        launch = mode == "auto-start",
        gguf_only = True,
    )


@pytest.mark.parametrize("mode", ["idle-server", "auto-start"])
def test_picked_file_rejects_conflicting_quant_before_load(
    cached_gguf_pick, picker, fake_studio, monkeypatch, capsys, mode
):
    from utils.models.model_config import _direct_gguf_for_variant, detect_gguf_model

    # A row for an individual file must never switch to a sibling.
    monkeypatch.setattr(
        catalog,
        "list_chat_models",
        lambda **kwargs: [catalog.ModelEntry("Downloaded", "tiny", "gguf", cached_gguf_pick)],
    )
    assert detect_gguf_model(cached_gguf_pick) == cached_gguf_pick
    assert _direct_gguf_for_variant(cached_gguf_pick, "Q8_0") is None
    attempts = []
    with pytest.raises(typer.Exit) as error:
        _connect_picked(
            mode,
            start.LoadOptions(gguf_variant = "Q8_0"),
            picker,
            fake_studio,
            monkeypatch,
            attempts,
        )
    assert error.value.exit_code == 1
    assert not attempts
    assert not _loads(fake_studio)
    output = capsys.readouterr()
    assert "--gguf-variant Q8_0" in output.out + output.err
    assert "--model" in output.out + output.err


@pytest.mark.parametrize("mode", ["idle-server", "auto-start"])
@pytest.mark.parametrize("variant", ["Q4_K_M", "q4_k_m", "tiny-Q4_K_M", "tiny-Q4_K_M.gguf", "Q8_0"])
def test_picked_repo_resolves_requested_quant(
    cached_gguf_pick, picker, fake_studio, monkeypatch, mode, variant
):
    from utils.models.model_config import _direct_gguf_for_variant

    expected = (
        str(Path(cached_gguf_pick).with_name("tiny-Q8_0.gguf"))
        if variant == "Q8_0"
        else cached_gguf_pick
    )
    assert _direct_gguf_for_variant(expected, variant) == expected
    attempts = []
    load = start.LoadOptions(gguf_variant = variant)
    if mode == "auto-start":
        with pytest.raises(typer.Exit) as error:
            _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts)
        assert error.value.exit_code == 7
        assert attempts == [(expected, load)]
    else:
        _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts)
        assert _loads(fake_studio) == [
            {"require_idle": True, "model_path": expected, "gguf_variant": variant}
        ]


@pytest.mark.parametrize("source", ["models_dir", "custom", "export"])
@pytest.mark.parametrize("mode", ["idle-server", "auto-start"])
def test_picked_directory_resolves_requested_quant(
    tmp_path, picker, fake_studio, monkeypatch, mode, source
):
    ensure_studio_backend_path(seed_cache_env = False)
    from hub.services.models.local_inventory import _classify_local_path, _scan_models_dir
    from utils import models

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    root = tmp_path / ("exports" if source == "export" else "models")
    folder = root / "tiny"
    folder.mkdir(parents = True)
    default = folder / "tiny-Q4_K_M.gguf"
    requested = folder / "tiny-Q8_0.gguf"
    for file in (default, requested):
        file.write_bytes(b"GGUF" + bytes(4092))

    if source == "export":
        exports, rows = models.scan_exported_models(), []
    else:
        exports = []
        rows = (
            _scan_models_dir(root)
            if source == "models_dir"
            else _classify_local_path(folder, "custom")
        )
    monkeypatch.setattr(models, "scan_exported_models", lambda: exports)
    monkeypatch.setattr(catalog, "_local_catalog_rows", lambda: rows)
    monkeypatch.setattr(catalog, "trained_entries", lambda: [])
    monkeypatch.setattr(catalog, "_cached_catalog_rows", lambda: ([], []))
    monkeypatch.setattr(catalog, "list_chat_models", REAL_CATALOG)
    assert [entry.model for entry in REAL_CATALOG()] == [str(default)]

    attempts = []
    load = start.LoadOptions(gguf_variant = "Q8_0")
    if mode == "auto-start":
        with pytest.raises(typer.Exit) as error:
            _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts)
        assert error.value.exit_code == 7
        assert attempts == [(str(requested), load)]
    else:
        _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts)
        assert _loads(fake_studio) == [
            {"require_idle": True, "model_path": str(requested), "gguf_variant": "Q8_0"}
        ]


def test_cached_picker_excludes_missing_and_incomplete_quants(cached_gguf_pick):
    snapshot = Path(cached_gguf_pick).parent
    (snapshot / "tiny-Q6_K-00001-of-00002.gguf").write_bytes(b"GGUF" + bytes(4092))
    assert catalog.list_chat_models(gguf_variant = "Q6_K") == []
    assert catalog.list_chat_models(gguf_variant = "Q5_K_M") == []


def test_cached_picker_preserves_path_qualified_selector(cached_gguf_pick):
    snapshot = Path(cached_gguf_pick).parent
    build = snapshot / "other-build"
    build.mkdir()
    (build / "tiny-Q8_0.gguf").write_bytes(b"GGUF" + bytes(4092))
    [entry] = catalog.list_chat_models(gguf_variant = "other-build/tiny-Q8_0.gguf")
    assert entry.model == str(build / "tiny-Q8_0.gguf")
    assert entry.file_variant == "tiny-Q8_0.gguf"


def test_auto_start_keeps_the_resolved_duplicate_family(
    cached_gguf_pick, picker, fake_studio, monkeypatch
):
    directory = Path(cached_gguf_pick).parent / "Q8_0"
    directory.mkdir()
    for name in ("a-Q8_0.gguf", "b-Q8_0.gguf"):
        (directory / name).write_bytes(b"GGUF" + bytes(4092))
    load = start.LoadOptions(gguf_variant = "Q8_0/b-Q8_0.gguf")
    attempts = []
    with pytest.raises(typer.Exit) as error:
        _connect_picked("auto-start", load, picker, fake_studio, monkeypatch, attempts)
    assert error.value.exit_code == 7
    assert attempts == [(str(directory / "b-Q8_0.gguf"), load._replace(gguf_variant = "b-Q8_0.gguf"))]


@pytest.mark.parametrize("mode", ["idle-server", "auto-start"])
@pytest.mark.parametrize("location", ["remembered-cache", "sibling-snapshot"])
def test_picked_repo_finds_requested_quant_outside_main(
    tmp_path, monkeypatch, picker, fake_studio, mode, location
):
    ensure_studio_backend_path(seed_cache_env = False)
    from huggingface_hub import scan_cache_dir
    from hub.services.models import cache_inventory
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
        snapshot = repo / "snapshots" / (revision * 40)
        snapshot.mkdir(parents = True)
        (repo / "refs").mkdir(exist_ok = True)
        (repo / "refs" / "main").write_text(snapshot.name)
        file = snapshot / f"tiny-{quant}.gguf"
        file.write_bytes(b"GGUF" + bytes(4092))
        if quant == "Q8_0":
            expected = str(file)

    scans = [scan_cache_dir(root) for root in (active, remembered) if root.is_dir()]
    monkeypatch.setattr(cache_inventory, "all_hf_cache_scans", lambda: scans)
    [row], _ = catalog._cached_catalog_rows()
    assert Path(row["cache_path"]).parent == active
    assert row["load_id"] == row["repo_id"] == "Org/Tiny-GGUF"
    pinned = Path(row["cache_path"]) / "snapshots" / ("a" * 40)
    assert catalog._cached_variant_entry({**row, "load_id": str(pinned)}, "Q8_0") is None
    monkeypatch.setattr(catalog, "list_chat_models", REAL_CATALOG)
    for source in ("trained_entries", "exported_entries", "local_folder_entries"):
        monkeypatch.setattr(catalog, source, lambda **kwargs: [])

    load = start.LoadOptions(gguf_variant = "Q8_0")
    attempts = []
    if mode == "auto-start":
        with pytest.raises(typer.Exit) as error:
            _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts)
        assert error.value.exit_code == 7
        assert attempts == [(expected, load)]
    else:
        _connect_picked(mode, load, picker, fake_studio, monkeypatch, attempts)
        assert _loads(fake_studio) == [
            {"require_idle": True, "model_path": expected, "gguf_variant": "Q8_0"}
        ]

    # A positive-sized file can still be shorter than its recorded download.
    from hub.utils.download_manifest import ExpectedFile, write_manifest

    assert write_manifest(
        "model",
        row["repo_id"],
        "Q8_0",
        [ExpectedFile(Path(expected).name, 8192)],
        hub_cache = q8_root,
        commit_hash = Path(expected).parent.name,
    )
    assert catalog._cached_variant_entry(row, "Q8_0") is None


@pytest.mark.parametrize("agent", ["claude", "codex"])
@pytest.mark.parametrize(
    ("filename", "selector"),
    [
        ("tiny-UD-NVFP4.gguf", "UD-NVFP4"),
        ("tiny-NVFP4-4.25bpw.gguf", "NVFP4-4.25bpw"),
        ("NVFP4/model.gguf", "NVFP4"),
        ("Q6_K/build/model.gguf", "Q6_K"),
        ("Q8_0/b-Q8_0.gguf", "Q8_0/b-Q8_0.gguf"),
        ("other-build/tiny-Q8_0.gguf", "other-build/tiny-Q8_0.gguf"),
    ],
)
def test_agent_picker_preload_agrees_with_local_loader(
    cached_gguf_pick, fake_studio, monkeypatch, picker, filename, selector, agent
):
    import asyncio
    from urllib.parse import parse_qs, urlparse
    from typer.testing import CliRunner
    from hub.services.models.gguf_variants import get_gguf_variants_response
    from utils.models.model_config import _find_local_gguf_by_variant

    snapshot = Path(cached_gguf_pick).parent
    expected = snapshot / filename
    expected.parent.mkdir(parents = True, exist_ok = True)
    expected.write_bytes(b"GGUF" + bytes(4092))
    if filename == "Q8_0/b-Q8_0.gguf":
        (expected.parent / "a-Q8_0.gguf").write_bytes(b"GGUF" + bytes(4092))
    assert _find_local_gguf_by_variant(str(snapshot), selector) == str(expected)
    _empty_server(fake_studio, monkeypatch)
    previous = start._http_json

    def http(method, url, key, *args, **kwargs):
        if "/api/models/gguf-variants?" in url:
            model = parse_qs(urlparse(url).query)["repo_id"][0]
            return asyncio.run(get_gguf_variants_response(model)).model_dump()
        return previous(method, url, key, *args, **kwargs)

    monkeypatch.setattr(start, "_http_json", http)
    picker["answers"] = [""]
    result = CliRunner().invoke(start.start_app, [agent, "--gguf-variant", selector, "--no-launch"])
    assert result.exit_code == 0, result.output
    [payload] = _loads(fake_studio)
    assert payload["model_path"] == str(expected)
    assert _find_local_gguf_by_variant(payload["model_path"], payload["gguf_variant"]) == str(
        expected
    )


def test_cached_picker_checks_only_the_selected_shard_family(cached_gguf_pick):
    snapshot = Path(cached_gguf_pick).parent
    empty = snapshot / "a-Q8_0.gguf"
    empty.touch()
    first = snapshot / "b-Q8_0-00001-of-00002.gguf"
    second = snapshot / "b-Q8_0-00002-of-00002.gguf"
    for file in (first, second):
        file.write_bytes(b"GGUF" + bytes(4092))

    assert [e.model for e in REAL_CATALOG(gguf_variant = first.name)] == [str(first)]
    assert REAL_CATALOG(gguf_variant = empty.name) == []
    second.write_bytes(b"")
    assert REAL_CATALOG(gguf_variant = first.name) == []


def test_loose_gguf_files_follow_requested_quant(tmp_path, monkeypatch):
    ensure_studio_backend_path(seed_cache_env = False)
    from hub.services.models.local_inventory import _scan_models_dir

    root = tmp_path / "models"
    root.mkdir()
    for name in ("tiny-Q4_K_M.gguf", "tiny-Q8_0.gguf"):
        (root / name).write_bytes(b"GGUF" + bytes(4092))
    rows = _scan_models_dir(root)
    monkeypatch.setattr(catalog, "_local_catalog_rows", lambda: rows)
    assert len(catalog.local_folder_entries()) == 2
    assert [entry.model for entry in catalog.local_folder_entries(gguf_variant = "Q8_0")] == [
        str(root / "tiny-Q8_0.gguf")
    ]


def test_loose_shard_filename_yields_the_loadable_first_shard(tmp_path, monkeypatch):
    ensure_studio_backend_path(seed_cache_env = False)
    from hub.services.models.local_inventory import _scan_models_dir
    from utils.models.model_config import _direct_gguf_for_variant

    paths = [tmp_path / f"tiny-Q8_0-{i:05d}-of-00002.gguf" for i in (1, 2)]
    for path in paths:
        path.write_bytes(b"GGUF" + bytes(4092))
    first, second = paths
    rows = _scan_models_dir(tmp_path)
    monkeypatch.setattr(catalog, "_local_catalog_rows", lambda: rows)
    assert _direct_gguf_for_variant(str(second), second.name) == str(first)
    [entry] = catalog.local_folder_entries(gguf_variant = second.name)
    assert entry.model == str(first)
    assert _direct_gguf_for_variant(entry.model, entry.file_variant) == str(first)
