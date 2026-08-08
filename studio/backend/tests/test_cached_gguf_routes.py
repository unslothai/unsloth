# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import json
import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

# Keep this test runnable without optional logging deps.
if "structlog" not in sys.modules:

    class _DummyLogger:
        def __getattr__(self, _name):
            return lambda *args, **kwargs: None

    sys.modules["structlog"] = types.SimpleNamespace(
        BoundLogger = _DummyLogger,
        get_logger = lambda *args, **kwargs: _DummyLogger(),
    )

import routes.models as models_route
from hub.services.models import gguf_variants as GV


def _answer(
    repo_id,
    variants = (),
    *,
    default_variant = None,
    source = None,
):
    """The (listing, source) pair the route consumes; *source* is the copy it came from."""
    return GV.VariantsAnswer(
        SimpleNamespace(
            repo_id = repo_id,
            variants = list(variants),
            has_vision = False,
            default_variant = default_variant,
        ),
        source,
    )


def _repo(
    repo_id: str,
    files: list[SimpleNamespace],
    repo_path: Path,
    *,
    revisions: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        repo_id = repo_id,
        repo_type = "model",
        repo_path = repo_path,
        revisions = revisions or [SimpleNamespace(files = files)],
    )


def _file(
    name: str,
    size_on_disk: int,
    *,
    blob_path: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        file_name = name,
        size_on_disk = size_on_disk,
        blob_path = blob_path,
    )


def test_iter_gguf_paths_matches_extension_case_insensitively(tmp_path):
    nested = tmp_path / "snapshots" / "rev"
    nested.mkdir(parents = True)
    lower = nested / "Q4_K_M.gguf"
    upper = nested / "Q8_0.GGUF"
    other = nested / "README.md"
    lower.write_text("a")
    upper.write_text("b")
    other.write_text("c")

    result = sorted(path.name for path in models_route._iter_gguf_paths(tmp_path))

    assert result == ["Q4_K_M.gguf", "Q8_0.GGUF"]


def test_legacy_hf_scan_uses_snapshot_path_for_inactive_cache(tmp_path):
    repo = tmp_path / "models--Org--Model"
    snapshot = repo / "snapshots" / "revision"
    snapshot.mkdir(parents = True)

    [row] = models_route._scan_hf_cache(tmp_path, active_cache = False)

    assert row.model_id == "Org/Model"
    assert row.id == str(snapshot.resolve())
    assert row.path == str(snapshot.resolve())


def test_collect_local_models_scans_previous_cache(monkeypatch, tmp_path):
    active = tmp_path / "active"
    previous = tmp_path / "previous"
    active.mkdir()
    snapshot = previous / "models--Org--Previous" / "snapshots" / "revision"
    snapshot.mkdir(parents = True)

    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr("utils.paths.lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [active, previous])
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [])

    rows = models_route.collect_local_models(tmp_path / "models")

    previous_row = next(row for row in rows if row.model_id == "Org/Previous")
    assert previous_row.id == str(snapshot.resolve())


def test_collect_local_models_prefers_complete_previous_copy(monkeypatch, tmp_path):
    from hub.utils import download_manifest

    active = tmp_path / "active"
    previous = tmp_path / "previous"
    active_partial = active / "models--Org--Model" / "blobs" / "abc.incomplete"
    active_partial.parent.mkdir(parents = True)
    active_partial.write_bytes(b"partial")
    snapshot = previous / "models--Org--Model" / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    (snapshot / "model.safetensors").write_bytes(b"complete")
    build_calls = []
    real_build = download_manifest.build_variant_state_index

    def record_build(repositories, **kwargs):
        repositories = tuple(repositories)
        build_calls.append(repositories)
        return real_build(repositories, **kwargs)

    monkeypatch.setattr("hub.utils.download_manifest.build_variant_state_index", record_build)
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr("utils.paths.lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [active, previous])
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [])

    rows = models_route.collect_local_models(tmp_path / "models")
    [row] = [row for row in rows if row.model_id == "Org/Model"]
    assert row.id == str(snapshot.resolve())
    assert row.partial is False
    assert row.active_cache is False
    assert len(build_calls) == 1
    assert set(build_calls[0]) == {
        ("model", "Org/Model", active),
        ("model", "Org/Model", previous),
    }


def test_compat_local_inventory_requests_share_scan(monkeypatch, tmp_path):
    # Total and stable on hostile input is the whole contract here; the exact
    # string is platform-dependent. POSIX realpath() rejects an embedded NUL with
    # ValueError so the raw string is normcased through, while Windows non-strict
    # realpath falls back to abspath and joins the cwd.
    for hostile in ("\0", "\ud800"):
        identity = models_route._compat_inventory_path_identity(hostile)
        assert identity == models_route._compat_inventory_path_identity(hostile)
        assert identity.endswith(hostile)
    sources = models_route._CompatLocalInventorySources(
        tmp_path / "active",
        tmp_path / "legacy",
        tmp_path / "default",
        (tmp_path / "lm",),
        (tmp_path / "known",),
    )
    folders = [[{"id": 1, "path": "/custom", "created_at": "now"}]]
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: folders[0])
    started, release, workers = [asyncio.Event(), asyncio.Event()], asyncio.Event(), []

    async def fake_to_thread(function, *args, **kwargs):
        if function is models_route.collect_local_models:
            workers.append((args, kwargs))
            started[len(workers) - 1].set()
            await release.wait()
            return []
        return function(*args, **kwargs)

    monkeypatch.setattr(models_route.asyncio, "to_thread", fake_to_thread)

    async def run_requests():
        request = lambda root = tmp_path: models_route._shared_compat_local_inventory_scan(
            root, sources
        )
        first = asyncio.create_task(request(tmp_path))
        await asyncio.wait_for(started[0].wait(), 1)
        second = asyncio.create_task(request(tmp_path / "alias" / ".."))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert len(workers) == 1
        key = next(iter(models_route._compat_local_inventory_flights))[1]
        assert sources in key and isinstance(key[-1], int)
        first.cancel()
        second.cancel()
        await asyncio.gather(first, second, return_exceptions = True)
        second = asyncio.create_task(request())
        await asyncio.sleep(0)
        folders[0] = [{"id": 2, "path": "/changed", "created_at": "later"}]
        changed = asyncio.create_task(request())
        await asyncio.wait_for(started[1].wait(), 1)
        assert [worker[1]["custom_folders"] for worker in workers] == [
            [{"id": 1, "path": "/custom", "created_at": "now"}],
            folders[0],
        ]
        release.set()
        return await asyncio.gather(second, changed)

    assert (
        asyncio.run(run_requests()) == [[], []] and not models_route._compat_local_inventory_flights
    )


def test_legacy_custom_inventory_filters_registered_mtp_root(tmp_path, monkeypatch):
    root = tmp_path / "MTP"
    root.mkdir()
    main = root / "Qwen3.6-27B-MTP-Q6_K.gguf"
    companion = root / "gemma-4-12b-it-Q8_0-MTP.gguf"
    model_dir = root / "model"
    model_dir.mkdir()
    main.write_bytes(b"x")
    companion.write_bytes(b"x")
    model_main = model_dir / "Qwen3.6-27B-MTP-Q8_0.gguf"
    model_main.write_bytes(b"x")
    real_is_dir = Path.is_dir

    def locked_is_dir(path):
        if path == model_main:
            raise OSError("locked")
        return real_is_dir(path)

    monkeypatch.setattr(Path, "is_dir", locked_is_dir)
    companion_dir = root / "companion-only"
    (companion_dir / "other").mkdir(parents = True)
    (companion_dir / "mtp-gemma-4-12b-it-Q8_0.gguf").write_bytes(b"x")
    (companion_dir / "other" / "Qwen3.6-27B-Q8_0.gguf").write_bytes(b"x")
    snapshot = root / "models--Org--Nested" / "snapshots" / "revision"
    quant_dir = snapshot / "BF16"
    quant_dir.mkdir(parents = True)
    (quant_dir / "Qwen3.6-27B-MTP-BF16.gguf").write_bytes(b"x")

    def fail_recursive_variant_scan(*_args, **_kwargs):
        raise AssertionError("unexpected recursive variant scan")

    monkeypatch.setattr(
        "utils.models.model_config._iter_gguf_files",
        fail_recursive_variant_scan,
    )
    monkeypatch.setattr(
        "storage.studio_db.list_scan_folders",
        lambda: pytest.fail("captured custom folders were reloaded"),
    )
    monkeypatch.setattr("utils.paths.lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr("utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path / "active")

    rows = models_route.collect_local_models(
        tmp_path / "models", custom_folders = [{"path": str(root)}]
    )
    paths = {row.path for row in rows}

    assert {str(main), str(model_dir), str(snapshot.resolve())} <= paths
    assert str(companion_dir) not in paths
    assert str(companion) not in paths


def test_list_cached_gguf_reports_snapshot_load_id_for_inactive_cache(monkeypatch, tmp_path):
    """Only a repo outside the active cache needs a snapshot load_id."""
    active = tmp_path / "active"
    snapshot = tmp_path / "legacy" / "models--Org--Away" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "Q4_K_M.gguf").write_bytes(b"\0")
    away = _repo(
        "Org/Away",
        [],
        tmp_path / "legacy" / "models--Org--Away",
        revisions = [
            SimpleNamespace(files = [_file("Q4_K_M.gguf", 5_000)], snapshot_path = snapshot),
        ],
    )
    here = _repo("Org/Here", [_file("Q4_K_M.gguf", 6_000)], active / "models--Org--Here")

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [away, here])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    rows = {
        c["repo_id"]: c
        for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]
    }

    assert rows["Org/Away"]["load_id"] == str(snapshot)
    assert "load_id" not in rows["Org/Here"]


def test_list_cached_gguf_pins_a_snapshot_for_a_recovered_active_cache_repo(monkeypatch, tmp_path):
    """Being in the active cache normally makes the repo id the load target, but not while refs/main
    names a commit with no directory: an offline client would follow the dangling ref and fail."""
    active = tmp_path / "active"
    repo_dir = active / "models--Org--Recovered"
    snapshot = repo_dir / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("c" * 40, encoding = "utf-8")

    recovered = _repo(
        "Org/Recovered",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(files = [_file("Model-Q4_K_M.gguf", 256)], snapshot_path = snapshot),
        ],
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [recovered])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    rows = {
        c["repo_id"]: c
        for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]
    }
    assert rows["Org/Recovered"]["load_id"] == str(snapshot)

    # Control: the same repo with a resolving ref keeps the repo id.
    (repo_dir / "refs" / "main").write_text("a" * 40, encoding = "utf-8")
    rows = {
        c["repo_id"]: c
        for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]
    }
    assert "load_id" not in rows["Org/Recovered"]


def test_list_cached_gguf_load_id_follows_snapshot_dir_mtime(monkeypatch, tmp_path):
    """Pick the snapshot variant discovery reads: newest directory, not newest blob."""
    import os

    active = tmp_path / "active"
    repo_dir = tmp_path / "legacy" / "models--Org--Multi"
    older, newer = repo_dir / "snapshots" / "rev-a", repo_dir / "snapshots" / "rev-b"
    for path in (older, newer):
        path.mkdir(parents = True)
    (older / "Q4_K_M.gguf").write_bytes(b"\0")
    (newer / "Q8_0.gguf").write_bytes(b"\0")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    repo = _repo(
        "Org/Multi",
        [],
        repo_dir,
        revisions = [
            # The older directory holds the newer blob, which is what diverges.
            SimpleNamespace(
                files = [_file("Q4_K_M.gguf", 5_000, blob_path = "b1")], snapshot_path = older
            ),
            SimpleNamespace(files = [_file("Q8_0.gguf", 6_000, blob_path = "b2")], snapshot_path = newer),
        ],
    )

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr(
        models_route, "_blob_mtime", lambda f: 9_000 if f.blob_path == "b1" else 1.0
    )

    rows = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]

    assert rows[0]["load_id"] == str(newer)


@pytest.mark.parametrize("reverse", [False, True])
def test_list_cached_gguf_load_id_breaks_mtime_ties_like_variant_discovery(
    reverse, monkeypatch, tmp_path
):
    """Equal snapshot mtimes must not leave the load id to iteration order.

    ``repo_info.revisions`` is a ``frozenset``, so on a coarse-timestamp filesystem this route
    published whichever snapshot the hash seed reached first while ``/api/models/gguf-variants``
    named the one ``snapshot_selection_key`` picks. Both revision orders are driven for that reason.
    """
    import os

    from hub.utils.gguf import iter_hf_cache_snapshots
    from hub.utils.hf_cache_state import snapshot_selection_key

    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    repo_dir = legacy / "models--Org--Tied"
    low, high = repo_dir / "snapshots" / "rev-a", repo_dir / "snapshots" / "rev-b"
    for path in (low, high):
        path.mkdir(parents = True)
    (low / "Model-Q4_K_M.gguf").write_bytes(b"\0")
    (high / "Model-Q5_K_M.gguf").write_bytes(b"\0")
    # One timestamp for both: the tie is the case.
    for path in (low, high):
        os.utime(path, (1_700_000_000, 1_700_000_000))

    revisions = [
        SimpleNamespace(files = [_file("Model-Q4_K_M.gguf", 5_000)], snapshot_path = low),
        SimpleNamespace(files = [_file("Model-Q5_K_M.gguf", 6_000)], snapshot_path = high),
    ]
    repo = _repo("Org/Tied", [], repo_dir, revisions = revisions[::-1] if reverse else revisions)

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda: [legacy], raising = False)

    rows = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]

    # The shared key's answer, and the directory variant discovery walks first.
    expected = max((low, high), key = snapshot_selection_key)
    assert rows[0].get("load_id") == str(expected)
    assert next(iter(iter_hf_cache_snapshots("Org/Tied"))) == expected


def test_list_cached_gguf_load_id_skips_partial_split_snapshot(monkeypatch, tmp_path):
    """A half-downloaded split quant must not beat an older snapshot that can load."""
    import os

    active = tmp_path / "active"
    repo_dir = tmp_path / "legacy" / "models--Org--Split"
    older, newer = repo_dir / "snapshots" / "rev-a", repo_dir / "snapshots" / "rev-b"
    for path in (older, newer):
        path.mkdir(parents = True)
    (older / "Model-Q8_0.gguf").write_bytes(b"\0")
    # Only part 1 of 3 landed before the download was interrupted.
    (newer / "Model-Q4_K_M-00001-of-00003.gguf").write_bytes(b"\0")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    repo = _repo(
        "Org/Split",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(files = [_file("Model-Q8_0.gguf", 5_000)], snapshot_path = older),
            SimpleNamespace(
                files = [_file("Model-Q4_K_M-00001-of-00003.gguf", 6_000)], snapshot_path = newer
            ),
        ],
    )

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    rows = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]

    assert rows[0]["load_id"] == str(older)


def test_list_cached_gguf_omits_load_id_when_no_snapshot_is_complete(monkeypatch, tmp_path):
    """With only a half-downloaded split quant, fall back to the repo id, not a path."""
    active = tmp_path / "active"
    repo_dir = tmp_path / "legacy" / "models--Org--Torn"
    snapshot = repo_dir / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    (snapshot / "Model-Q4_K_M-00001-of-00003.gguf").write_bytes(b"\0")

    repo = _repo(
        "Org/Torn",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(
                files = [_file("Model-Q4_K_M-00001-of-00003.gguf", 6_000)], snapshot_path = snapshot
            ),
        ],
    )

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    rows = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]

    assert "load_id" not in rows[0]


def test_list_cached_gguf_load_id_takes_the_snapshot_holding_a_whole_quant(monkeypatch, tmp_path):
    """One whole quant beside a half-downloaded one is still a safe load target:
    the lister behind /gguf-variants trims its offer to the completed subset."""
    import os

    active = tmp_path / "active"
    repo_dir = tmp_path / "legacy" / "models--Org--Mixed"
    older, newer = repo_dir / "snapshots" / "rev-a", repo_dir / "snapshots" / "rev-b"
    for path in (older, newer):
        path.mkdir(parents = True)
    (older / "Model-Q4_K_M.gguf").write_bytes(b"\0")
    # rev-b has a complete Q8_0 AND a half-downloaded split Q4_K_M.
    (newer / "Model-Q8_0.gguf").write_bytes(b"\0")
    (newer / "Model-Q4_K_M-00001-of-00003.gguf").write_bytes(b"\0")
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    repo = _repo(
        "Org/Mixed",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(files = [_file("Model-Q4_K_M.gguf", 5_000)], snapshot_path = older),
            SimpleNamespace(
                files = [
                    _file("Model-Q8_0.gguf", 5_000),
                    _file("Model-Q4_K_M-00001-of-00003.gguf", 6_000),
                ],
                snapshot_path = newer,
            ),
        ],
    )

    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    rows = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]

    assert rows[0].get("load_id") == str(newer)
    # Every quant that offer advertises is on disk in the snapshot it pinned.
    from hub.utils.gguf import list_local_gguf_variants
    from hub.utils.inventory_scan import complete_snapshot_variants

    pinned = rows[0]["load_id"]
    offered = {v.quant for v in list_local_gguf_variants(pinned)[0] if v.quant}
    advertised = offered & complete_snapshot_variants(pinned)
    assert advertised == {"Q8_0"}
    assert advertised - {v.quant for v in list_local_gguf_variants(str(older))[0] if v.quant}


def test_a_zero_byte_first_gguf_leaves_the_quant_incomplete(tmp_path):
    """The resolver takes the lexicographically first file under a label. Skipping a zero-byte one
    handed the label to the next file, so the quant read complete while the load opened the empty
    file it names."""
    from hub.utils.inventory_scan import complete_snapshot_variants
    from utils.models.model_config import _find_local_gguf_by_variant

    snapshot = tmp_path / "models--Org--Quant" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "A-Q4_K_M.gguf").write_bytes(b"")
    (snapshot / "B-Q4_K_M.gguf").write_bytes(b"\0" * 256)

    assert Path(_find_local_gguf_by_variant(str(snapshot), "Q4_K_M")).name == "A-Q4_K_M.gguf"
    assert complete_snapshot_variants(str(snapshot)) == set()

    # Control: the same pair with the empty file sorting second, which the resolver never opens.
    (snapshot / "A-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    (snapshot / "B-Q4_K_M.gguf").write_bytes(b"")
    assert complete_snapshot_variants(str(snapshot)) == {"Q4_K_M"}


def test_a_snapshot_scope_from_another_repo_is_not_used(tmp_path):
    """The response carries the requested repo's identity, so a local_path pointing into a
    different repo's cache must not become the scope its files are counted in."""
    other = tmp_path / "models--Org--Other" / "snapshots" / ("a" * 40)
    mine = tmp_path / "models--Org--Quant" / "snapshots" / ("b" * 40)
    for path in (other, mine):
        path.mkdir(parents = True)

    assert GV._snapshot_scope_for_request("Org/Quant", str(other)) is None
    assert GV._snapshot_scope_for_request("Org/Quant", str(mine)) == mine
    # Casing follows the cache directory, which need not match the requested id.
    assert GV._snapshot_scope_for_request("org/quant", str(mine)) == mine


@pytest.mark.parametrize(
    ("files", "expected_default", "expected_ready"),
    [
        pytest.param(
            {"Model-Q8_0.gguf": b"\0" * 256, "Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 256},
            "Q8_0",
            {"Q8_0"},
            id = "the-preferred-quant-is-short-a-shard",
        ),
        pytest.param(
            {"Model-Q8_0.gguf": b"\0" * 256, "Model-Q4_K_M.gguf": b"\0" * 128},
            "Q4_K_M",
            {"Q4_K_M", "Q8_0"},
            id = "both-whole-so-the-usual-preference-stands",
        ),
        pytest.param(
            {"Model-Q4_K_M-00001-of-00002.gguf": b"\0" * 256},
            "Q4_K_M",
            set(),
            id = "nothing-ready-so-the-default-is-a-download-target",
        ),
    ],
)
def test_the_local_default_variant_is_one_that_can_load(
    tmp_path, files, expected_default, expected_ready
):
    """The picker re-checks only that default_variant fits in memory, never that it is downloaded,
    so a quant short a shard gets recommended over a whole one sitting beside it."""
    snapshot = tmp_path / "models--Org--Model" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents = True)
    for name, blob in files.items():
        (snapshot / name).write_bytes(blob)

    response = asyncio.run(GV.get_gguf_variants_response(str(snapshot), hf_token = None))

    assert response.default_variant == expected_default
    assert {v.quant for v in response.variants if v.downloaded} == expected_ready


def test_variant_readiness_is_counted_in_the_snapshot_the_row_pinned(monkeypatch, tmp_path):
    """A pinned row resolves inside one directory. Counting readiness across the repo offered a
    quant living in a sibling revision, which the pinned load then cannot find."""
    import os

    active = tmp_path / "active"
    repo_dir = active / "models--Org--Quant"
    pinned, sibling = repo_dir / "snapshots" / ("d" * 40), repo_dir / "snapshots" / ("e" * 40)
    for path in (pinned, sibling):
        path.mkdir(parents = True)
    (pinned / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    (sibling / "Model-Q8_0.gguf").write_bytes(b"\0" * 256)
    os.utime(pinned, (1_000, 1_000))
    os.utime(sibling, (2_000, 2_000))

    monkeypatch.setattr(
        "hub.utils.gguf.iter_hf_cache_snapshots",
        lambda repo_id, root = None: [sibling, pinned],
    )
    monkeypatch.setattr(
        "hub.services.models.gguf_variants.iter_hf_cache_snapshots",
        lambda repo_id, root = None: [sibling, pinned],
    )

    scoped = asyncio.run(
        GV.get_gguf_variants_response("Org/Quant", prefer_local_cache = True, local_path = str(pinned))
    )
    assert {v.quant for v in scoped.variants if v.downloaded} == {"Q4_K_M"}

    # Control: naming the sibling counts that directory instead.
    other = asyncio.run(
        GV.get_gguf_variants_response("Org/Quant", prefer_local_cache = True, local_path = str(sibling))
    )
    assert {v.quant for v in other.variants if v.downloaded} == {"Q8_0"}


def test_a_later_attempts_cancel_marker_does_not_break_the_pinned_quant(monkeypatch, tmp_path):
    """A marker carries no revision and is rewritten by each attempt, so it belongs to the newest
    snapshot. The row already attributes it that way; the variants endpoint has to agree, or the
    one quant the pinned snapshot can load is hidden."""
    import os

    from hub.utils import download_manifest
    from hub.utils.gguf import GgufVariantInfo

    active = tmp_path / "active"
    repo_dir = active / "models--Org--Quant"
    pinned, newer = repo_dir / "snapshots" / ("d" * 40), repo_dir / "snapshots" / ("e" * 40)
    for path in (pinned, newer):
        path.mkdir(parents = True)
    (pinned / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    os.utime(pinned, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    (repo_dir / "refs").mkdir(parents = True)
    # Dangling, so the row pins the older complete snapshot.
    (repo_dir / "refs" / "main").write_text("c" * 40, encoding = "utf-8")

    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(
            hub_cache = active, hf_home = tmp_path, source = "studio", cache_home = tmp_path
        ),
    )
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda **kw: [active])
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.hf_cache_root",
        lambda create = False, root = None: root if root is not None else active,
    )
    monkeypatch.setattr(
        GV,
        "list_gguf_variants",
        lambda repo_id, hf_token = None: (
            [
                GgufVariantInfo(
                    filename = "Model-Q4_K_M.gguf",
                    quant = "Q4_K_M",
                    display_label = "Q4_K_M",
                    size_bytes = 256,
                )
            ],
            False,
            [],
        ),
    )
    assert download_manifest.write_cancel_marker("model", "Org/Quant", "Q4_K_M", hub_cache = active)

    response = asyncio.run(GV.get_gguf_variants_response("Org/Quant", local_path = str(pinned)))
    assert {v.quant for v in response.variants if v.downloaded} == {"Q4_K_M"}
    assert not any(v.partial for v in response.variants)

    # Control: with refs/main resolving, the marker describes what a repo-id load reads.
    (repo_dir / "refs" / "main").write_text("e" * 40, encoding = "utf-8")
    unpinned = asyncio.run(GV.get_gguf_variants_response("Org/Quant"))
    assert all(v.partial for v in unpinned.variants)


def test_the_pins_excuse_covers_only_the_quants_it_holds(monkeypatch, tmp_path):
    """The pinned snapshot excuses a revision-less signal because it holds that quant whole. A quant
    it does not hold is still the cancelled download the marker describes, and hiding that leaves it
    listed as a plain undownloaded row with nothing to resume or delete."""
    import os

    from hub.utils import download_manifest
    from hub.utils.gguf import GgufVariantInfo

    active = tmp_path / "active"
    repo_dir = active / "models--Org--Quant"
    pinned, newer = repo_dir / "snapshots" / ("d" * 40), repo_dir / "snapshots" / ("e" * 40)
    for path in (pinned, newer):
        path.mkdir(parents = True)
    (pinned / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    (newer / "Model-Q8_0-00001-of-00002.gguf").write_bytes(b"\0" * 16)
    os.utime(pinned, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("e" * 40, encoding = "utf-8")

    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(
            hub_cache = active, hf_home = tmp_path, source = "studio", cache_home = tmp_path
        ),
    )
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda **kw: [active])
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.hf_cache_root",
        lambda create = False, root = None: root if root is not None else active,
    )
    monkeypatch.setattr(
        GV,
        "list_gguf_variants",
        lambda repo_id, hf_token = None: (
            [
                GgufVariantInfo(filename = "Model-Q4_K_M.gguf", quant = "Q4_K_M", size_bytes = 256),
                GgufVariantInfo(filename = "Model-Q8_0.gguf", quant = "Q8_0", size_bytes = 512),
            ],
            False,
            [],
        ),
    )
    assert download_manifest.write_cancel_marker("model", "Org/Quant", "Q8_0", hub_cache = active)

    response = asyncio.run(GV.get_gguf_variants_response("Org/Quant", local_path = str(pinned)))
    by_quant = {v.quant: v for v in response.variants}
    assert by_quant["Q4_K_M"].downloaded is True and by_quant["Q4_K_M"].partial is False
    assert by_quant["Q8_0"].partial is True


def test_a_later_attempts_incomplete_blob_does_not_break_the_pinned_quant(monkeypatch, tmp_path):
    """blobs/ is repo-wide and each attempt rewrites it, so a retry's .incomplete belongs to the
    newest snapshot exactly as a cancel marker does. Judging the pinned quant by it hid the one
    copy that loads."""
    import os

    from hub.utils import download_registry
    from hub.utils.download_manifest import ExpectedFile
    from hub.utils.gguf import GgufVariantInfo
    from hub.utils.gguf_plan import plan_from_expected_files

    sha = "f" * 64
    active = tmp_path / "active"
    repo_dir = active / "models--Org--Quant"
    pinned, newer = repo_dir / "snapshots" / ("d" * 40), repo_dir / "snapshots" / ("e" * 40)
    for path in (pinned, newer):
        path.mkdir(parents = True)
    (pinned / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    (newer / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0" * 16)
    os.utime(pinned, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    (repo_dir / "blobs").mkdir(parents = True)
    (repo_dir / "blobs" / f"{sha}.incomplete").write_bytes(b"\0" * 8)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("e" * 40, encoding = "utf-8")

    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(
            hub_cache = active, hf_home = tmp_path, source = "studio", cache_home = tmp_path
        ),
    )
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda **kw: [active])
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.hf_cache_root",
        lambda create = False, root = None: root if root is not None else active,
    )
    variant = GgufVariantInfo(
        filename = "Model-Q4_K_M.gguf",
        quant = "Q4_K_M",
        display_label = "Q4_K_M",
        size_bytes = 256,
    )
    monkeypatch.setattr(
        GV, "list_gguf_variants", lambda repo_id, hf_token = None: ([variant], False, [])
    )
    monkeypatch.setattr(
        GV,
        "_gguf_all_variant_requirements",
        lambda repo_id, hf_token, siblings = None: {
            "q4_k_m": plan_from_expected_files(
                "Q4_K_M", [ExpectedFile(path = "Model-Q4_K_M.gguf", size = 256, sha256 = sha)]
            )
        },
    )
    monkeypatch.setattr(GV, "_variant_requirement_cache_get", lambda key: None)
    monkeypatch.setattr(download_registry, "incomplete_blob_hashes", lambda *a, **kw: {sha})

    response = asyncio.run(GV.get_gguf_variants_response("Org/Quant", local_path = str(pinned)))
    assert {v.quant for v in response.variants if v.downloaded} == {"Q4_K_M"}
    assert not any(v.partial for v in response.variants)

    # Controls: unpinned, and pinned to the snapshot the blob does belong to.
    for target in (None, str(newer)):
        other = asyncio.run(
            GV.get_gguf_variants_response(
                "Org/Quant", **({} if target is None else {"local_path": target})
            )
        )
        assert all(v.partial for v in other.variants)


def test_a_copy_that_loads_beats_a_bigger_one_that_does_not(monkeypatch, tmp_path):
    """Two caches holding one repo are two directories, and only one of them loads. Keeping the
    larger download put an unusable copy on the row while a whole one sat in the other cache."""
    active, legacy = tmp_path / "active", tmp_path / "legacy"
    torn = active / "models--Org--Quant" / "snapshots" / ("d" * 40)
    whole = legacy / "models--Org--Quant" / "snapshots" / ("e" * 40)
    for path in (torn, whole):
        path.mkdir(parents = True)
    # Bigger, and half a split: size alone would keep this one.
    (torn / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0" * 4096)
    for shard in ("00001", "00002"):
        (whole / f"Model-Q4_K_M-{shard}-of-00002.gguf").write_bytes(b"\0" * 256)
    for repo_dir, ref in ((torn.parent.parent, "c" * 40), (whole.parent.parent, "e" * 40)):
        (repo_dir / "refs").mkdir(parents = True)
        (repo_dir / "refs" / "main").write_text(ref, encoding = "utf-8")

    def _repo_for(snapshot, size, repo_dir):
        return _repo(
            "Org/Quant",
            [],
            repo_dir,
            revisions = [
                SimpleNamespace(
                    files = [_file(f.name, size) for f in sorted(snapshot.iterdir())],
                    snapshot_path = snapshot,
                )
            ],
        )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [
            SimpleNamespace(repos = [_repo_for(torn, 4096, torn.parent.parent)]),
            SimpleNamespace(repos = [_repo_for(whole, 256, whole.parent.parent)]),
        ],
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    row = {
        c["repo_id"]: c
        for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]
    }["Org/Quant"]
    assert row["cache_path"] == str(whole.parent.parent)
    assert row["load_id"] == str(whole)


def test_list_cached_gguf_pins_a_snapshot_when_the_default_ref_quant_is_torn(monkeypatch, tmp_path):
    """The repo id resolving is not enough: refs/main can land on a revision holding half a split
    while an older one is whole. The compat schema carries no partial flag, so a client loading the
    id follows the torn ref and fails with a complete copy one directory away."""
    import os

    active = tmp_path / "active"
    repo_dir = active / "models--Org--Quant"
    older, newer = repo_dir / "snapshots" / ("d" * 40), repo_dir / "snapshots" / ("e" * 40)
    for path in (older, newer):
        path.mkdir(parents = True)
    for shard in ("00001", "00002"):
        (older / f"Model-Q4_K_M-{shard}-of-00002.gguf").write_bytes(b"\0" * 256)
    (newer / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0" * 256)
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("e" * 40, encoding = "utf-8")

    repo = _repo(
        "Org/Quant",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(
                files = [_file("Model-Q4_K_M-00001-of-00002.gguf", 256)], snapshot_path = newer
            ),
            SimpleNamespace(
                files = [
                    _file("Model-Q4_K_M-00001-of-00002.gguf", 256),
                    _file("Model-Q4_K_M-00002-of-00002.gguf", 256),
                ],
                snapshot_path = older,
            ),
        ],
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    rows = {
        c["repo_id"]: c
        for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]
    }
    assert rows["Org/Quant"]["load_id"] == str(older)

    # Control: point the ref at the whole copy and the repo id is what loads again.
    (repo_dir / "refs" / "main").write_text("d" * 40, encoding = "utf-8")
    rows = {
        c["repo_id"]: c
        for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]
    }
    assert "load_id" not in rows["Org/Quant"]


def test_vision_is_read_from_the_snapshot_the_row_pins(monkeypatch, tmp_path):
    """A projector in a revision the row does not point at is one the loader never opens, so
    advertising vision off it offers a capability the pinned copy cannot deliver."""
    import os

    active = tmp_path / "active"
    repo_dir = active / "models--Org--Vision"
    older, newer = repo_dir / "snapshots" / ("d" * 40), repo_dir / "snapshots" / ("e" * 40)
    for path in (older, newer):
        path.mkdir(parents = True)
    for shard in ("00001", "00002"):
        (older / f"Model-Q4_K_M-{shard}-of-00002.gguf").write_bytes(b"\0" * 256)
    (newer / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0" * 256)
    (newer / "mmproj-F16.gguf").write_bytes(b"\0" * 256)
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("e" * 40, encoding = "utf-8")

    repo = _repo(
        "Org/Vision",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(
                files = [
                    _file("Model-Q4_K_M-00001-of-00002.gguf", 256),
                    _file("mmproj-F16.gguf", 256),
                ],
                snapshot_path = newer,
            ),
            SimpleNamespace(
                files = [
                    _file("Model-Q4_K_M-00001-of-00002.gguf", 256),
                    _file("Model-Q4_K_M-00002-of-00002.gguf", 256),
                ],
                snapshot_path = older,
            ),
        ],
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)

    def _row():
        return {
            c["repo_id"]: c
            for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))[
                "cached"
            ]
        }["Org/Vision"]

    row = _row()
    assert row["load_id"] == str(older)
    assert row["has_vision"] is False

    # Control: put a projector beside the copy that loads and the capability is real again.
    (older / "mmproj-F16.gguf").write_bytes(b"\0" * 256)
    repo.revisions[1].files = list(repo.revisions[1].files) + [_file("mmproj-F16.gguf", 256)]
    row = _row()
    assert row["load_id"] == str(older)
    assert row["has_vision"] is True


def test_vision_is_read_from_the_snapshot_a_repo_id_load_resolves(monkeypatch, tmp_path):
    """A row that pins nothing still loads from one snapshot, the one the resolver returns. A
    projector in a revision that resolution never reaches is not vision support either."""
    import os

    active = tmp_path / "active"
    repo_dir = active / "models--Org--Vision"
    main, other = repo_dir / "snapshots" / ("d" * 40), repo_dir / "snapshots" / ("e" * 40)
    for path in (main, other):
        path.mkdir(parents = True)
    for shard in ("00001", "00002"):
        (main / f"Model-Q4_K_M-{shard}-of-00002.gguf").write_bytes(b"\0" * 256)
    # Newest, and holds only the projector: nothing here for a load to land on.
    (other / "mmproj-F16.gguf").write_bytes(b"\0" * 256)
    os.utime(main, (1_000, 1_000))
    os.utime(other, (2_000, 2_000))
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("d" * 40, encoding = "utf-8")

    repo = _repo(
        "Org/Vision",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(files = [_file("mmproj-F16.gguf", 256)], snapshot_path = other),
            SimpleNamespace(
                files = [
                    _file("Model-Q4_K_M-00001-of-00002.gguf", 256),
                    _file("Model-Q4_K_M-00002-of-00002.gguf", 256),
                ],
                snapshot_path = main,
            ),
        ],
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr(
        "hub.utils.gguf.iter_hf_cache_snapshots", lambda repo_id, root = None: [other, main]
    )

    def _row():
        return {
            c["repo_id"]: c
            for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))[
                "cached"
            ]
        }["Org/Vision"]

    row = _row()
    assert "load_id" not in row
    assert row["has_vision"] is False

    # Control: the projector beside the quant that resolves is reachable.
    (main / "mmproj-F16.gguf").write_bytes(b"\0" * 256)
    row = _row()
    assert row["has_vision"] is True


def test_a_projector_at_the_snapshot_root_serves_a_quant_in_a_subdirectory(monkeypatch, tmp_path):
    """Split quants live in a per quant subdirectory while the projector stays at the snapshot
    root, so vision has to be judged on the snapshot rather than on the file's own directory."""
    active = tmp_path / "active"
    repo_dir = active / "models--Org--Nested"
    snapshot = repo_dir / "snapshots" / ("d" * 40)
    (snapshot / "UD-Q4_K_XL").mkdir(parents = True)
    for shard in ("00001", "00002"):
        (snapshot / "UD-Q4_K_XL" / f"Model-UD-Q4_K_XL-{shard}-of-00002.gguf").write_bytes(
            b"\0" * 256
        )
    (snapshot / "mmproj-F16.gguf").write_bytes(b"\0" * 256)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("d" * 40, encoding = "utf-8")

    repo = _repo(
        "Org/Nested",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(
                files = [
                    _file("Model-UD-Q4_K_XL-00001-of-00002.gguf", 256),
                    _file("Model-UD-Q4_K_XL-00002-of-00002.gguf", 256),
                    _file("mmproj-F16.gguf", 256),
                ],
                snapshot_path = snapshot,
            )
        ],
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr(
        "hub.utils.gguf.iter_hf_cache_snapshots", lambda repo_id, root = None: [snapshot]
    )

    row = {
        c["repo_id"]: c
        for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]
    }["Org/Nested"]
    assert "load_id" not in row
    assert row["has_vision"] is True


def test_vision_is_read_from_the_cache_root_holding_the_row(monkeypatch, tmp_path):
    """The same repo can sit in several cache roots, and the row describes one copy: a duplicate
    elsewhere is a download the load never reaches, so it cannot answer for this row's projector."""
    import os

    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    repo_dir = active / "models--Org--Split"
    here = repo_dir / "snapshots" / ("a" * 40)
    there = legacy / "models--Org--Split" / "snapshots" / ("b" * 40)
    here.mkdir(parents = True)
    there.mkdir(parents = True)
    for name in ("Model-Q4_K_M.gguf", "mmproj-F16.gguf"):
        (here / name).write_bytes(b"\0" * 256)
    (there / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    os.utime(here, (1_000, 1_000))
    os.utime(there, (2_000, 2_000))
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("a" * 40, encoding = "utf-8")

    repo = _repo(
        "Org/Split",
        [],
        repo_dir,
        revisions = [
            SimpleNamespace(
                files = [_file("Model-Q4_K_M.gguf", 256), _file("mmproj-F16.gguf", 256)],
                snapshot_path = here,
            )
        ],
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda: [active, legacy])

    row = {
        c["repo_id"]: c
        for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]
    }["Org/Split"]
    assert row["has_vision"] is True


def test_vision_is_not_invented_for_a_copy_that_ships_no_projector(monkeypatch, tmp_path):
    """Control for the test above: scoping the lookup must not turn every row vision capable."""
    active = tmp_path / "active"
    legacy = tmp_path / "legacy"
    repo_dir = active / "models--Org--Plain"
    here = repo_dir / "snapshots" / ("a" * 40)
    there = legacy / "models--Org--Plain" / "snapshots" / ("b" * 40)
    here.mkdir(parents = True)
    there.mkdir(parents = True)
    (here / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    for name in ("Model-Q4_K_M.gguf", "mmproj-F16.gguf"):
        (there / name).write_bytes(b"\0" * 256)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("a" * 40, encoding = "utf-8")

    repo = _repo(
        "Org/Plain",
        [],
        repo_dir,
        revisions = [SimpleNamespace(files = [_file("Model-Q4_K_M.gguf", 256)], snapshot_path = here)],
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda: [active, legacy])

    row = {
        c["repo_id"]: c
        for c in asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))["cached"]
    }["Org/Plain"]
    assert row["has_vision"] is False


def test_metadata_resolves_from_the_snapshot_holding_the_whole_quant(monkeypatch, tmp_path):
    """The lister and the load take the whole copy, so mtime order alone would read metadata out of
    a newer half download nothing loads."""
    import os

    from hub.utils.gguf import iter_snapshots_preferring_whole

    repo_dir = tmp_path / "models--Org--Quant"
    older, newer = repo_dir / "snapshots" / ("d" * 40), repo_dir / "snapshots" / ("e" * 40)
    for path in (older, newer):
        path.mkdir(parents = True)
    for shard in ("00001", "00002"):
        (older / f"Model-Q4_K_M-{shard}-of-00002.gguf").write_bytes(b"\0" * 256)
    (newer / "Model-Q4_K_M-00001-of-00002.gguf").write_bytes(b"\0" * 256)
    os.utime(older, (1_000, 1_000))
    os.utime(newer, (2_000, 2_000))

    monkeypatch.setattr(
        "hub.utils.gguf.iter_hf_cache_snapshots", lambda repo_id, root = None: [newer, older]
    )

    assert iter_snapshots_preferring_whole("Org/Quant", "Q4_K_M") == [older, newer]
    # No variant to judge, so mtime order stands.
    assert iter_snapshots_preferring_whole("Org/Quant", None) == [newer, older]


def test_list_cached_gguf_includes_non_suffix_repo_when_cache_contains_gguf(monkeypatch, tmp_path):
    repo = _repo(
        "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
        [_file("Q4_K_M.gguf", 5_000), _file("README.md", 10)],
        tmp_path / "models--HauhauCS--Gemma",
    )
    scan = SimpleNamespace(repos = [repo])

    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [scan])

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive",
            "size_bytes": 5_000,
            "cache_path": str(repo.repo_path),
            "has_vision": False,
            "task": None,
        }
    ]


def test_list_cached_gguf_matches_extension_case_insensitively(monkeypatch, tmp_path):
    repo = _repo(
        "Org/Model-Without-Suffix",
        [_file("Q8_0.GGUF", 7_000)],
        tmp_path / "models--Org--Model-Without-Suffix",
    )
    scan = SimpleNamespace(repos = [repo])

    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [scan])

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/Model-Without-Suffix",
            "size_bytes": 7_000,
            "cache_path": str(repo.repo_path),
            "has_vision": False,
            "task": None,
        }
    ]


def test_is_hidden_model_hides_validation_probe_everywhere():
    """Every picker (model list, local, cached GGUF, cached models) gates on
    _is_hidden_model, so hiding the probe here hides it in the search menu too.
    Cover both forms callers pass: the reconstructed repo id and the on-disk
    snapshot path."""
    assert models_route._is_hidden_model("ggml-org/models")
    assert models_route._is_hidden_model("ggml-org/models/tinyllamas/stories260K.gguf")
    assert models_route._is_hidden_model(
        None, "/hf/models--ggml-org--models/snapshots/abc/tinyllamas/stories260K.gguf"
    )
    # A Windows-style snapshot path must match too, even on a POSIX interpreter
    # (the filename check splits on both separators).
    assert models_route._is_hidden_model(
        r"C:\Users\u\.cache\huggingface\hub\models--ggml-org--models\snapshots\abc\tinyllamas\stories260K.gguf"
    )
    assert not models_route._is_hidden_model("unsloth/gemma-3-270m-it-GGUF")
    # The exact-filename needle must not hide a real repo that merely
    # references stories260K in its name.
    assert not models_route._is_hidden_model("user/stories260K-finetune-GGUF")


def test_is_hidden_model_hides_dictation_models(tmp_path):
    assert models_route._is_hidden_model("unsloth/whisper-tiny")
    assert models_route._is_hidden_model("unsloth/whisper-base")
    assert models_route._is_hidden_model("unsloth/whisper-small")
    assert models_route._is_hidden_model("unsloth/whisper-large-v3-turbo")
    assert models_route._is_hidden_model(
        "/hf/models--unsloth--whisper-large-v3/snapshots/abc/model.safetensors"
    )
    assert not models_route._is_hidden_model("user/whisper-finetune")
    assert not models_route._is_hidden_model(
        "C:\\cache\\models--unsloth--whisper-small-finetune\\model.safetensors"
    )
    custom = tmp_path / "custom-whisper"
    custom.mkdir()
    (custom / "config.json").write_text(
        '{"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}'
    )
    (custom / "model.safetensors").write_bytes(b"weights")
    assert models_route._is_hidden_model(
        "user/custom-checkpoint",
        str(custom / "model.safetensors"),
    )
    named_only = tmp_path / "whisper-finetune"
    named_only.mkdir()
    (named_only / "config.json").write_text('{"model_type": "llama"}')
    assert not models_route._is_hidden_model("user/whisper-finetune", str(named_only))


def test_list_cached_models_hides_custom_whisper_by_config(monkeypatch, tmp_path):
    # Regression: the legacy /cached-models picker must pass the snapshot path so
    # the config check hides a custom (non-curated) Whisper checkpoint; a bare
    # repo id cannot ("user/whisper-finetune" is not in the curated set).
    repo_path = tmp_path / "models--user--whisper-finetune"
    snap = repo_path / "snapshots" / "abc"
    snap.mkdir(parents = True)
    (snap / "config.json").write_text(
        '{"model_type": "whisper", "architectures": ["WhisperForConditionalGeneration"]}'
    )
    (snap / "model.safetensors").write_bytes(b"weights")

    captured: list = []
    real_hidden = models_route._is_hidden_model

    def spy(*values):
        captured.append(values)
        return real_hidden(*values)

    monkeypatch.setattr(models_route, "_is_hidden_model", spy)
    repo = _repo(
        "user/whisper-finetune",
        [SimpleNamespace(file_name = "model.safetensors", size_on_disk = 10)],
        repo_path,
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [repo])]
    )

    result = asyncio.run(
        models_route.list_cached_models(current_subject = "test-user", hf_token = None)
    )
    # The route passed the snapshot path (not just the repo id) ...
    assert any(str(repo_path) in values for values in captured)
    # ... so the custom Whisper checkpoint is hidden from the chat picker.
    assert result["cached"] == []


def test_is_hidden_model_matches_repo_ids_exactly(monkeypatch):
    """A custom embedder with a generic basename is hidden by EXACT repo-id
    match only, so unrelated cached repos that merely contain the basename stay
    visible. Regression: substring basename matching hid real chat models like
    ``user/model-chat`` from the On Device inventory."""
    from core.rag import config as rag_config

    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/model")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/model-GGUF")

    # The exact embedder repo and its GGUF companion are hidden.
    assert models_route._is_hidden_model("org/model")
    assert models_route._is_hidden_model("org/model-GGUF")
    # Unrelated repos that merely contain "model" must NOT be hidden.
    assert not models_route._is_hidden_model("user/model-chat")
    assert not models_route._is_hidden_model("org/model-instruct")
    assert not models_route._is_hidden_model("acme/remodelled-chat")
    # The validation probe stays hidden regardless of embedder config.
    assert models_route._is_hidden_model("ggml-org/models")


def test_is_hidden_model_matches_repo_derived_local_paths(monkeypatch):
    """Match exact repo-derived cache and LM Studio paths."""
    from core.rag import config as rag_config

    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/model")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/model-GGUF")

    assert models_route._is_hidden_model(
        "/cache/models--org--model/snapshots/abc/model.safetensors"
    )
    assert models_route._is_hidden_model(
        r"C:\Users\u\.cache\huggingface\hub\models--org--model-GGUF\snapshots\abc"
    )
    assert models_route._is_hidden_model("/lm-studio/org/model-GGUF/model-Q8_0.gguf")
    assert not models_route._is_hidden_model("/lm-studio/user/model-chat/model-Q8_0.gguf")
    assert not models_route._is_hidden_model("/cache/models--org--model-instruct")


def test_is_hidden_model_prefers_existing_relative_path(monkeypatch, tmp_path):
    """Prefer an existing relative path over repo-id syntax."""
    from core.rag import config as rag_config

    embedder = tmp_path / "models" / "embedder"
    embedder.mkdir(parents = True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "models/embedder")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/embedder-GGUF")

    assert models_route._is_hidden_model(str(embedder))


def test_is_hidden_model_keeps_stale_default_embedder_hidden(monkeypatch):
    """Keep default embedders hidden after a settings change."""
    from core.rag import config as rag_config

    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/custom")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/custom-GGUF")

    assert models_route._is_hidden_model("unsloth/bge-small-en-v1.5")
    assert models_route._is_hidden_model("unsloth/bge-small-en-v1.5-GGUF")
    assert models_route._is_hidden_model("/models/bge-small-en-v1.5")
    assert models_route._is_hidden_model("/models/bge-small-en-v1.5-F16.gguf")
    assert models_route._is_hidden_model(r"C:\models\bge-small-en-v1.5-Q8_0.gguf")
    # Repo IDs still use exact matching, and similar local basenames must have
    # a real separator after the static default name.
    assert not models_route._is_hidden_model("user/bge-small-en-v1.5-chat")
    assert not models_route._is_hidden_model("/models/bge-small-en-v1.50")


def test_is_hidden_model_keeps_env_default_hidden_after_override(monkeypatch):
    """A persisted override must not expose the deployment's env default."""
    from core.rag import config as rag_config

    monkeypatch.delenv("RAG_EMBED_GGUF_REPO", raising = False)
    monkeypatch.setattr(rag_config, "EMBEDDING_MODEL", "org/env-default")
    monkeypatch.setattr(rag_config, "effective_embedding_model", lambda: "org/custom")
    monkeypatch.setattr(rag_config, "effective_gguf_repo", lambda: "org/custom-GGUF")

    assert models_route._is_hidden_model("org/env-default")
    assert models_route._is_hidden_model("org/env-default-GGUF")
    assert models_route._is_hidden_model("org/custom")
    assert models_route._is_hidden_model("org/custom-GGUF")
    assert not models_route._is_hidden_model("org/env-default-chat")


def test_hidden_models_importable_without_heavy_model_stack():
    """The hub cache scanner imports ``is_hidden_model`` at module scope, so it
    must not drag in ``utils/models/__init__`` (the model-config + checkpoint
    stack). Verify in a clean interpreter that importing the helper touches
    neither ``utils.models`` nor those heavy submodules, and still classifies
    the probe."""
    import os
    import subprocess
    import textwrap

    backend = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        import sys

        class _Blocker:
            _blocked = (
                "utils.models",
                "utils.models.model_config",
                "utils.models.checkpoints",
            )

            def find_spec(self, name, path=None, target=None):
                if name in self._blocked:
                    raise ImportError("blocked heavy import: " + name)
                return None

        sys.meta_path.insert(0, _Blocker())
        from utils.hidden_models import is_hidden_model

        loaded = sorted(m for m in sys.modules if m.startswith("utils.models"))
        assert not loaded, loaded
        assert is_hidden_model("ggml-org/models") is True
        assert is_hidden_model("unsloth/gemma-3-270m-it-GGUF") is False
        print("HIDDEN_MODELS_IMPORT_OK")
        """
    )
    env = dict(os.environ, PYTHONPATH = str(backend))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output = True,
        text = True,
        env = env,
    )
    assert proc.returncode == 0, proc.stderr
    assert "HIDDEN_MODELS_IMPORT_OK" in proc.stdout


def test_list_cached_gguf_hides_llama_validation_probe(monkeypatch, tmp_path):
    """The ggml-org/models / stories260K install validation probe can land in
    the HF cache as a side effect of installing the prebuilt llama-server.
    It is not a chat model (it sorts smallest and would be auto-selected), so
    pickers must hide it while keeping real cached models."""
    probe = _repo(
        "ggml-org/models",
        [_file("tinyllamas/stories260K.gguf", 1_000)],
        tmp_path / "models--ggml-org--models",
    )
    real = _repo(
        "unsloth/gemma-3-270m-it-GGUF",
        [_file("gemma-3-270m-it-UD-Q4_K_XL.gguf", 200_000)],
        tmp_path / "models--unsloth--gemma-3-270m-it-GGUF",
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [probe, real])]
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    repo_ids = [c["repo_id"] for c in result["cached"]]
    assert "ggml-org/models" not in repo_ids
    assert "unsloth/gemma-3-270m-it-GGUF" in repo_ids


def test_list_cached_gguf_skips_repos_without_positive_gguf_size(monkeypatch, tmp_path):
    missing = _repo(
        "Org/ReadmeOnly",
        [_file("README.md", 10)],
        tmp_path / "models--Org--ReadmeOnly",
    )
    zero = _repo(
        "Org/ZeroSize",
        [_file("Q4_K_M.gguf", 0)],
        tmp_path / "models--Org--ZeroSize",
    )
    scan = SimpleNamespace(repos = [missing, zero])

    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [scan])

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == []


def test_list_cached_gguf_keeps_largest_duplicate_repo_across_scans(monkeypatch, tmp_path):
    smaller = _repo(
        "Org/Dupe",
        [_file("Q4_K_M.gguf", 2_000)],
        tmp_path / "models--Org--Dupe-a",
    )
    larger = _repo(
        "org/dupe",
        [_file("Q4_K_M.gguf", 5_000), _file("Q6_K.gguf", 1_000)],
        tmp_path / "models--Org--Dupe-b",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [
            SimpleNamespace(repos = [smaller]),
            SimpleNamespace(repos = [larger]),
        ],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "org/dupe",
            "size_bytes": 6_000,
            "cache_path": str(larger.repo_path),
            "has_vision": False,
            "task": None,
        }
    ]


def test_list_cached_gguf_dedupes_shared_blobs_across_revisions(monkeypatch, tmp_path):
    shared = "blobs/shared-q4"
    repo = _repo(
        "Org/SharedBlobRepo",
        [],
        tmp_path / "models--Org--SharedBlobRepo",
        revisions = [
            SimpleNamespace(files = [_file("Q4_K_M.gguf", 5_000, blob_path = shared)]),
            SimpleNamespace(files = [_file("Q4_K_M.gguf", 5_000, blob_path = shared)]),
        ],
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [repo])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/SharedBlobRepo",
            "size_bytes": 5_000,
            "cache_path": str(repo.repo_path),
            "has_vision": False,
            "task": None,
        }
    ]


def test_list_cached_models_skips_non_suffix_repo_when_gguf_files_exist(monkeypatch, tmp_path):
    mixed = _repo(
        "Org/MixedRepo",
        [
            _file("Q4_K_M.gguf", 5_000),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MixedRepo",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mixed])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))

    assert result["cached"] == []


def test_list_cached_models_prefers_complete_over_larger_partial(monkeypatch, tmp_path):
    # The same repo cached in two roots: a LARGER but PARTIAL copy must not shadow a SMALLER but COMPLETE one, or the picker hides a usable model.
    complete = _repo(
        "Org/Dup",
        [_file("model.safetensors", 10_000)],
        tmp_path / "root_a" / "models--Org--Dup",
    )
    partial = _repo(
        "Org/Dup",
        [_file("model.safetensors", 15_000)],
        tmp_path / "root_b" / "models--Org--Dup",
    )

    # The larger copy (root_b) is the partial one; the smaller (root_a) is complete.
    monkeypatch.setattr(
        models_route,
        "_cached_repo_partial",
        lambda repo_id, repo_cache_dir = None: "root_b" in str(repo_cache_dir),
    )
    monkeypatch.setattr(models_route, "_cached_repo_task", lambda repo_info: None)
    # List the partial (larger) FIRST, so the old size-only rule would have picked it.
    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [partial, complete])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))

    assert len(result["cached"]) == 1
    row = result["cached"][0]
    assert row["repo_id"] == "Org/Dup"
    # The COMPLETE (smaller) copy won.
    assert row.get("partial") is not True
    assert row["size_bytes"] == 10_000


def test_list_cached_gguf_includes_mixed_repo_with_gguf_and_safetensors(monkeypatch, tmp_path):
    """Mixed repo still surfaces in cached-gguf as a GGUF download."""
    mixed = _repo(
        "Org/MixedRepo",
        [
            _file("Q4_K_M.gguf", 5_000),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MixedRepo",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mixed])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/MixedRepo",
            "size_bytes": 5_000,
            "cache_path": str(mixed.repo_path),
            "has_vision": False,
            "task": None,
        }
    ]


def test_list_cached_gguf_handles_none_size_on_disk(monkeypatch, tmp_path):
    """``size_on_disk = None`` (partial download) is treated as zero, not a
    TypeError from ``sum()`` that wipes the response."""
    partial = _repo(
        "Org/PartialDownload",
        [_file("Q4_K_M.gguf", None), _file("Q6_K.gguf", 5_000)],
        tmp_path / "models--Org--PartialDownload",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [partial])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/PartialDownload",
            "size_bytes": 5_000,
            "cache_path": str(partial.repo_path),
            "has_vision": False,
            "task": None,
        }
    ]


def test_list_cached_gguf_skips_malformed_repo_without_wiping_response(monkeypatch, tmp_path):
    """One repo raising during classification must not poison the response."""

    class _ExplodingRepo:
        repo_id = "Org/Broken"
        repo_type = "model"
        repo_path = tmp_path / "models--Org--Broken"

        @property
        def revisions(self):
            raise RuntimeError("boom")

    healthy = _repo(
        "Org/Healthy",
        [_file("Q4_K_M.gguf", 5_000)],
        tmp_path / "models--Org--Healthy",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [_ExplodingRepo(), healthy])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/Healthy",
            "size_bytes": 5_000,
            "cache_path": str(healthy.repo_path),
            "has_vision": False,
            "task": None,
        }
    ]


def test_list_cached_gguf_skips_repo_with_only_mmproj_gguf(monkeypatch, tmp_path):
    """A repo whose only ``.gguf`` is an mmproj vision adapter is not a GGUF
    repo: mmproj is filtered out, leaving zero variants."""
    mmproj_only = _repo(
        "Org/MmprojOnly",
        [
            _file("mmproj-Q8_0.gguf", 5_000),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MmprojOnly",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mmproj_only])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == []


def test_list_cached_models_includes_repo_with_only_mmproj_gguf(monkeypatch, tmp_path):
    """A safetensors repo with an auxiliary mmproj adapter still surfaces in
    cached-models as a normal model."""
    mmproj_aux = _repo(
        "Org/MmprojAux",
        [
            _file("mmproj-Q8_0.gguf", 5_000),
            _file("model.safetensors", 10_000),
        ],
        tmp_path / "models--Org--MmprojAux",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [mmproj_aux])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))

    assert result["cached"] == [{"repo_id": "Org/MmprojAux", "size_bytes": 15_000, "task": None}]


def test_list_cached_models_tags_diffusers_pipeline_as_text_to_image(monkeypatch, tmp_path):
    """A cached diffusers pipeline repo (model_index.json present) is tagged
    text-to-image so the chat picker hides it, while a plain checkpoint isn't."""
    diffusion = _repo(
        "Tongyi-MAI/Z-Image-Turbo",
        [
            _file("model_index.json", 1_000),
            _file("text_encoder/model.safetensors", 9_000),
            _file("transformer/diffusion_pytorch_model.safetensors", 9_000),
        ],
        tmp_path / "models--Tongyi-MAI--Z-Image-Turbo",
    )
    checkpoint = _repo(
        "unsloth/Llama-3.2-1B-Instruct",
        [_file("config.json", 1_000), _file("model.safetensors", 9_000)],
        tmp_path / "models--unsloth--Llama-3.2-1B-Instruct",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [diffusion, checkpoint])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))
    by_repo = {c["repo_id"]: c["task"] for c in result["cached"]}
    assert by_repo == {
        "Tongyi-MAI/Z-Image-Turbo": "text-to-image",
        "unsloth/Llama-3.2-1B-Instruct": None,
    }


def test_list_cached_models_marks_companion_only_pipeline_partial(monkeypatch, tmp_path):
    """A companion-only prefetch (VAE / text-encoder / model_index.json but no transformer) carries
    a root model_index.json yet is not a loadable pipeline, so it must be marked partial. A sibling
    repo that DOES ship its transformer shards stays complete."""
    companion_only = _repo(
        "black-forest-labs/FLUX.1-dev",
        [
            _file("model_index.json", 1_000),
            _file("vae/diffusion_pytorch_model.safetensors", 9_000),
            _file("text_encoder/model.safetensors", 9_000),
        ],
        tmp_path / "models--black-forest-labs--FLUX.1-dev",
    )
    complete = _repo(
        "Tongyi-MAI/Z-Image-Turbo",
        [
            _file("model_index.json", 1_000),
            _file("text_encoder/model.safetensors", 9_000),
            _file("transformer/diffusion_pytorch_model.safetensors", 9_000),
        ],
        tmp_path / "models--Tongyi-MAI--Z-Image-Turbo",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [companion_only, complete])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))
    by_repo = {c["repo_id"]: c for c in result["cached"]}
    assert by_repo["black-forest-labs/FLUX.1-dev"].get("partial") is True
    assert by_repo["Tongyi-MAI/Z-Image-Turbo"].get("partial") is None


def test_list_cached_gguf_includes_vision_repo_with_main_gguf_and_mmproj(monkeypatch, tmp_path):
    """A vision GGUF repo (main weight + mmproj) is a GGUF repo; reported size
    is the main weight only, since mmproj is filtered at classification."""
    vision_repo = _repo(
        "Org/VisionGguf",
        [
            _file("Q4_K_M.gguf", 5_000),
            _file("mmproj-Q8_0.gguf", 1_000),
        ],
        tmp_path / "models--Org--VisionGguf",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [vision_repo])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert result["cached"] == [
        {
            "repo_id": "Org/VisionGguf",
            "size_bytes": 5_000,
            "cache_path": str(vision_repo.repo_path),
            "has_vision": True,
            "task": None,
        }
    ]


def _gfile(name: str, size: int, mtime: float) -> SimpleNamespace:
    """A cached file carrying a Hugging Face ``blob_last_modified`` timestamp."""
    return SimpleNamespace(
        file_name = name,
        size_on_disk = size,
        blob_path = None,
        blob_last_modified = mtime,
    )


def test_all_hf_cache_scans_uses_shared_inventory(monkeypatch, tmp_path):
    from hub.utils import inventory_scan

    active = SimpleNamespace(
        repos = [_repo("Org/Active", [_file("Q4_K_M.gguf", 5_000)], tmp_path / "active")]
    )

    monkeypatch.setattr(inventory_scan, "all_hf_cache_scans", lambda: [active])

    scans = models_route._all_hf_cache_scans()
    assert scans == [active]

    # End-to-end: the endpoint still returns the active cache's repo.
    monkeypatch.setattr(models_route, "_all_hf_cache_scans", lambda: [active])
    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))
    assert result["cached"] == [
        {
            "repo_id": "Org/Active",
            "size_bytes": 5_000,
            "cache_path": str(tmp_path / "active"),
            "has_vision": False,
            "task": None,
        }
    ]


def test_list_cached_gguf_sorts_newest_first_grouping_by_latest_quant(monkeypatch, tmp_path):
    """Downloaded is ordered newest-first, and a multi-quant repo is placed by
    its most recently downloaded quant (``last_modified`` = newest quant)."""
    older = _repo(
        "Org/Older",
        [_gfile("Older-Q4_K_M.gguf", 5_000, 1_000.0)],
        tmp_path / "models--Org--Older",
    )
    newer = _repo(
        "Org/Newer",
        [
            _gfile("Newer-Q4_K_M.gguf", 5_000, 2_000.0),
            _gfile("Newer-Q8_0.gguf", 9_000, 3_000.0),  # newest quant in the repo
        ],
        tmp_path / "models--Org--Newer",
    )

    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [older, newer])],
    )

    result = asyncio.run(models_route.list_cached_gguf(current_subject = "test-user"))

    assert [c["repo_id"] for c in result["cached"]] == ["Org/Newer", "Org/Older"]
    assert result["cached"][0]["last_modified"] == 3_000.0
    assert result["cached"][1]["last_modified"] == 1_000.0


def test_list_cached_gguf_dedupe_keeps_newest_timestamp(monkeypatch, tmp_path):
    """Same repo in two caches with equal size keeps the newest last_modified,
    regardless of scan order."""
    older = _repo("org/dupe", [_gfile("dupe-Q4_K_M.gguf", 5_000, 1_000.0)], tmp_path / "a")
    newer = _repo("org/dupe", [_gfile("dupe-Q4_K_M.gguf", 5_000, 9_000.0)], tmp_path / "b")
    for scans in ([older, newer], [newer, older]):  # both orders
        monkeypatch.setattr(
            models_route,
            "_all_hf_cache_scans",
            lambda s = scans: [SimpleNamespace(repos = [s[0]]), SimpleNamespace(repos = [s[1]])],
        )
        result = asyncio.run(models_route.list_cached_gguf(current_subject = "t"))
        assert len(result["cached"]) == 1
        assert result["cached"][0]["last_modified"] == 9_000.0


def test_gguf_variants_mmproj_does_not_mark_quant_downloaded(monkeypatch, tmp_path):
    """The per-quant 'downloaded' flag is driven by the real weight file in a
    single snapshot; an mmproj vision adapter (matching a quant label) must
    not make that quant appear downloaded."""
    variants = [
        SimpleNamespace(
            filename = "model-Q4_K_M.gguf",
            quant = "Q4_K_M",
            display_label = None,
            size_bytes = 10_000,
        ),
        SimpleNamespace(
            filename = "model-F16.gguf",
            quant = "F16",
            display_label = None,
            size_bytes = 20_000,
        ),
    ]
    monkeypatch.setattr(
        GV,
        "list_gguf_variants",
        lambda repo_id, hf_token = None: (variants, True, []),
    )
    monkeypatch.setattr(
        GV,
        "_local_main_gguf_blobs_by_quant",
        lambda _repo_id, repo_cache_dir = None: {},
    )

    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 10_000)  # real weight, fully present
    (snap / "mmproj-F16.gguf").write_bytes(b"y" * 20_000)  # mmproj adapter, label "F16"
    monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id, root = None: [snap])

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo", hf_token = None, current_subject = "test-user"
        )
    )

    flags = {v.quant: v.downloaded for v in result.variants}
    assert flags["Q4_K_M"] is True
    assert flags["F16"] is False


def test_gguf_variants_route_scopes_local_probe_to_selected_cache(monkeypatch, tmp_path):
    snapshot = tmp_path / "inactive" / "models--org--repo" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)
    calls = []

    async def scoped_variants(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return _answer(repo_id)

    context_calls = []
    monkeypatch.setattr(GV, "get_gguf_variants_answer", scoped_variants)
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((model, is_local)) or 8192,
    )

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            prefer_local_cache = True,
            local_path = str(snapshot),
            hf_token = None,
            current_subject = "test-user",
        )
    )

    assert calls == [
        (
            "org/repo",
            {
                "prefer_local_cache": True,
                "offline": False,
                "local_path": str(snapshot),
                "hf_token": None,
            },
        )
    ]
    assert context_calls == [(str(snapshot), True)]
    assert result.context_length == 8192


def test_gguf_variants_route_reads_context_from_the_pinned_snapshot(monkeypatch, tmp_path):
    """Enumeration may be repo wide, but the native context must come from the pinned snapshot
    or the dialog offers a length the model cannot serve."""
    snapshot = tmp_path / "active" / "models--org--repo" / "snapshots" / "rev"
    snapshot.mkdir(parents = True)

    async def scoped_variants(repo_id, **kwargs):
        return _answer(repo_id)

    context_calls = []
    monkeypatch.setattr(GV, "get_gguf_variants_answer", scoped_variants)
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((model, is_local)) or 4096,
    )

    asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            prefer_local_cache = False,
            local_path = str(snapshot),
            hf_token = None,
            current_subject = "test-user",
        )
    )

    assert context_calls == [(str(snapshot.resolve()), True)]


def test_gguf_variants_route_ignores_a_pin_naming_another_repo(monkeypatch, tmp_path):
    """Control: a pin naming another repo falls back to the repo id rather than reporting a
    stranger's metadata."""
    other = tmp_path / "active" / "models--org--other" / "snapshots" / "rev"
    other.mkdir(parents = True)

    async def scoped_variants(repo_id, **kwargs):
        return _answer(repo_id)

    context_calls = []
    monkeypatch.setattr(GV, "get_gguf_variants_answer", scoped_variants)
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((model, is_local)) or 4096,
    )

    asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            prefer_local_cache = False,
            local_path = str(other),
            hf_token = None,
            current_subject = "test-user",
        )
    )

    assert context_calls == [("org/repo", False)]


def test_gguf_variants_route_forwards_offline(monkeypatch):
    """Parity with /api/hub/gguf-variants: without this an unreachable Hub still sends the
    picker down the remote path."""
    calls = []

    async def scoped_variants(repo_id, **kwargs):
        calls.append(kwargs)
        return _answer(repo_id)

    monkeypatch.setattr(GV, "get_gguf_variants_answer", scoped_variants)
    monkeypatch.setattr(
        models_route, "_read_native_context_length", lambda model, *, is_local: None
    )

    asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            offline = True,
            hf_token = None,
            current_subject = "test-user",
        )
    )

    assert calls == [
        {"prefer_local_cache": False, "offline": True, "local_path": None, "hf_token": None}
    ]


def test_native_context_read_gives_up_when_the_cache_walk_drags(monkeypatch, tmp_path):
    """Unbounded, this walk held the variant listing open, leaving the picker on
    "Loading variants…" with no quant to click. It reports None and stops walking instead."""
    visited = []

    def dragging_walk(root, deadline = None):
        for index in range(200):
            time.sleep(0.01)
            visited.append(index)
            yield Path(root) / f"model-{index}.gguf"

    monkeypatch.setattr(models_route, "_iter_gguf_paths", dragging_walk)
    monkeypatch.setattr(models_route, "_NATIVE_CONTEXT_READ_TIMEOUT_SECONDS", 0.1)

    started = time.monotonic()
    result = models_route._read_native_context_length(str(tmp_path), is_local = True)
    elapsed = time.monotonic() - started

    assert result is None
    # A signature drift here raises inside the broad except and returns instantly, which
    # would pass every other assertion without walking anything.
    assert visited, "the walk never ran, so this proves nothing"
    assert elapsed < 2
    assert len(visited) < 200


def test_native_context_read_budget_binds_on_a_walk_that_yields_nothing(monkeypatch, tmp_path):
    """_iter_gguf_paths yields only .gguf files, so a large cache can walk a long time
    yielding nothing. Checking the budget per yield alone would never check it at all."""
    handed = []

    def walk(root, deadline = None):
        handed.append(deadline)
        for _ in range(200):
            time.sleep(0.005)
            if deadline is not None and time.monotonic() >= deadline:
                return
        return
        yield  # pragma: no cover

    monkeypatch.setattr(models_route, "_iter_gguf_paths", walk)
    monkeypatch.setattr(models_route, "_NATIVE_CONTEXT_READ_TIMEOUT_SECONDS", 0.05)

    started = time.monotonic()
    assert models_route._read_native_context_length(str(tmp_path), is_local = True) is None
    assert handed and handed[0] is not None, "the walker was given no deadline"
    assert time.monotonic() - started < 1


def test_native_context_read_budget_is_checked_between_caches(monkeypatch, tmp_path):
    """A repo present in several caches must not restart the budget per cache."""
    walked = []

    def walk(root, deadline = None):
        walked.append(str(root))
        time.sleep(0.2)
        return iter(())

    monkeypatch.setattr(models_route, "_iter_gguf_paths", walk)
    monkeypatch.setattr(models_route, "_NATIVE_CONTEXT_READ_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(models_route, "_is_valid_repo_id", lambda _r: True)
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.iter_repo_cache_dirs",
        lambda _kind, _repo: [tmp_path / "a", tmp_path / "b", tmp_path / "c"],
    )

    models_route._read_native_context_length("org/repo", is_local = False)
    assert len(walked) < 3, f"every cache was walked despite the budget: {walked}"


def test_native_context_read_budget_covers_cache_discovery(monkeypatch, tmp_path):
    """Cache enumeration touches the filesystem too. Started after it, the budget would hand
    the walk a full fresh allowance on top of whatever discovery already cost."""

    def slow_discovery(_kind, _repo):
        time.sleep(0.4)
        return [tmp_path / "a", tmp_path / "b", tmp_path / "c"]

    def slow_walk(root, deadline = None):
        time.sleep(0.3)
        return iter(())

    monkeypatch.setattr(models_route, "_is_valid_repo_id", lambda _r: True)
    monkeypatch.setattr("hub.utils.hf_cache_state.iter_repo_cache_dirs", slow_discovery)
    monkeypatch.setattr(models_route, "_iter_gguf_paths", slow_walk)
    monkeypatch.setattr(models_route, "_NATIVE_CONTEXT_READ_TIMEOUT_SECONDS", 0.05)

    started = time.monotonic()
    assert models_route._read_native_context_length("org/repo", is_local = False) is None
    # Discovery itself is not interruptible; a walk on top of it means the budget restarted.
    assert time.monotonic() - started < 0.55


def test_gguf_variants_route_answers_when_a_header_read_never_returns(monkeypatch, tmp_path):
    """One syscall that never returns cannot be interrupted from inside the walk, so the
    route bounds it. Without that the listing waits on it and the picker has nothing to click."""
    (tmp_path / "model-Q4_K_M.gguf").write_bytes(b"x")

    def hung_read(_path):
        time.sleep(5)
        return 8192

    monkeypatch.setattr("utils.models.gguf_metadata.read_gguf_context_length", hung_read)
    monkeypatch.setattr(models_route, "_NATIVE_CONTEXT_READ_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(models_route, "_NATIVE_CONTEXT_HARD_TIMEOUT_SECONDS", 0.2)

    async def scoped_variants(repo_id, **kwargs):
        return _answer(
            repo_id,
            [
                SimpleNamespace(
                    filename = "model-Q4_K_M.gguf",
                    quant = "Q4_K_M",
                    size_bytes = 10,
                    download_size_bytes = 10,
                    downloaded = True,
                )
            ],
            default_variant = "Q4_K_M",
        )

    monkeypatch.setattr(GV, "get_gguf_variants_answer", scoped_variants)

    async def drive():
        began = time.monotonic()
        answer = await models_route.get_gguf_variants(
            repo_id = str(tmp_path), hf_token = None, current_subject = "test-user"
        )
        return answer, time.monotonic() - began

    result, elapsed = asyncio.run(drive())
    assert [v.quant for v in result.variants] == ["Q4_K_M"]
    assert result.context_length is None
    assert elapsed < 3


def test_native_context_read_runs_on_a_daemon_thread(monkeypatch):
    """A thread pool's workers are joined at interpreter exit, so a read abandoned on a hung
    mount would hold up shutdown for as long as the mount stays hung (measured: the full
    length of the read). A daemon thread does not."""
    observed = {}
    entered = threading.Event()

    def stalled(model, *, is_local):
        observed["daemon"] = threading.current_thread().daemon
        entered.set()
        time.sleep(0.5)
        return 8192

    monkeypatch.setattr(models_route, "_read_native_context_length", stalled)
    monkeypatch.setattr(models_route, "_NATIVE_CONTEXT_HARD_TIMEOUT_SECONDS", 0.1)

    async def drive():
        return await models_route._read_native_context_length_bounded("/tmp", True)

    assert asyncio.run(drive()) is None
    assert entered.wait(3)
    assert observed["daemon"] is True


def _live_context_threads() -> int:
    return sum(1 for thread in threading.enumerate() if thread.name == "native-ctx")


def _drain_context_threads(timeout: float = 5.0) -> None:
    """Reads abandoned by an earlier test outlive it, so wait them out before counting."""
    end = time.monotonic() + timeout
    while _live_context_threads() and time.monotonic() < end:
        time.sleep(0.02)


def test_native_context_reads_stop_starting_once_every_slot_is_stranded(monkeypatch):
    """Retries against a hung mount must not start a thread apiece; they wait for a slot
    and give up inside the bound."""
    release = threading.Event()

    def stalled(model, *, is_local):
        release.wait(5)
        return 8192

    monkeypatch.setattr(models_route, "_read_native_context_length", stalled)
    monkeypatch.setattr(models_route, "_NATIVE_CONTEXT_HARD_TIMEOUT_SECONDS", 0.1)
    monkeypatch.setattr(models_route, "_NATIVE_CONTEXT_MAX_CONCURRENT_READS", 2)
    _drain_context_threads()

    async def drive():
        for _ in range(2):  # strand every slot
            assert await models_route._read_native_context_length_bounded("/tmp", True) is None
        live_before = _live_context_threads()
        began = time.monotonic()
        answer = await models_route._read_native_context_length_bounded("/tmp", True)
        return answer, time.monotonic() - began, live_before, _live_context_threads()

    try:
        answer, elapsed, before, after = asyncio.run(drive())
        assert answer is None
        assert elapsed < 1  # gave up inside the bound rather than waiting on the mount
        assert before == 2  # the cap held
        assert after <= before  # and nothing new was started
    finally:
        release.set()


def test_concurrent_native_context_reads_all_keep_their_length(monkeypatch):
    """Ordinary concurrency must queue for a slot, not skip the read. Giving up when no slot
    was free on the spot dropped most lengths on a healthy cache (measured 4 of 64)."""
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: (time.sleep(0.002), 8192)[1],
    )

    async def drive():
        return await asyncio.gather(
            *[models_route._read_native_context_length_bounded("/tmp", True) for _ in range(64)]
        )

    assert asyncio.run(drive()) == [8192] * 64


def test_offline_reads_context_from_the_copy_the_variants_came_from(monkeypatch, tmp_path):
    """The length has to come from the copy the listing came from. Which copy that is cannot
    be read off the request: the HF cache answers before local_path, so the service reports
    it and the route follows."""
    context_calls = []

    async def scoped_variants(repo_id, **kwargs):
        return _answer(repo_id, source = kwargs["local_path"])

    monkeypatch.setattr(GV, "get_gguf_variants_answer", scoped_variants)
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((model, is_local)) or 4096,
    )

    asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            offline = True,
            prefer_local_cache = False,
            local_path = str(tmp_path),
            hf_token = None,
            current_subject = "test-user",
        )
    )
    assert context_calls == [(str(tmp_path), True)]


def test_native_context_read_still_reports_a_length_within_budget(monkeypatch, tmp_path):
    """Control: the bound only trims a walk that drags; a header reached in time still answers."""
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"x")

    monkeypatch.setattr(models_route, "_iter_gguf_paths", lambda root, deadline = None: iter([gguf]))
    monkeypatch.setattr("utils.models.gguf_metadata.read_gguf_context_length", lambda _path: 8192)

    assert models_route._read_native_context_length(str(tmp_path), is_local = True) == 8192


def test_gguf_variants_ignore_big_endian_siblings(monkeypatch, tmp_path):
    siblings = [
        SimpleNamespace(rfilename = "model-Q4_K_M-be.gguf", size = 100),
        SimpleNamespace(rfilename = "model-Q4_K_M.gguf", size = 10),
    ]
    monkeypatch.setattr(
        GV,
        "list_gguf_variants",
        lambda repo_id, hf_token = None: (
            [
                SimpleNamespace(
                    filename = "model-Q4_K_M.gguf",
                    quant = "Q4_K_M",
                    display_label = None,
                    size_bytes = 10,
                )
            ],
            False,
            siblings,
        ),
    )
    monkeypatch.setattr(
        GV,
        "_local_main_gguf_blobs_by_quant",
        lambda _repo_id, repo_cache_dir = None: {},
    )

    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M.gguf").write_bytes(b"x" * 10)
    monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id, root = None: [snap])

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo", hf_token = None, current_subject = "test-user"
        )
    )

    assert [(v.quant, v.filename, v.size_bytes, v.downloaded) for v in result.variants] == [
        ("Q4_K_M", "model-Q4_K_M.gguf", 10, True)
    ]


def test_gguf_variants_cached_big_endian_does_not_satisfy_variant(monkeypatch, tmp_path):
    variants = [
        SimpleNamespace(
            filename = "model-Q4_K_M.gguf",
            quant = "Q4_K_M",
            display_label = None,
            size_bytes = 10,
        ),
    ]
    monkeypatch.setattr(
        GV,
        "list_gguf_variants",
        lambda repo_id, hf_token = None: (variants, False, []),
    )
    monkeypatch.setattr(
        GV,
        "_local_main_gguf_blobs_by_quant",
        lambda _repo_id, repo_cache_dir = None: {},
    )

    snap = tmp_path / "models--org--repo" / "snapshots" / "rev"
    snap.mkdir(parents = True)
    (snap / "model-Q4_K_M-be.gguf").write_bytes(b"x" * 10)
    monkeypatch.setattr(GV, "iter_hf_cache_snapshots", lambda _repo_id, root = None: [snap])

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo", hf_token = None, current_subject = "test-user"
        )
    )

    assert result.variants[0].downloaded is False


def test_legacy_gguf_progress_delegates_to_shared_service(monkeypatch):
    calls = []

    async def shared(repo_id, *, variant, expected_bytes, hf_token):
        calls.append((repo_id, variant, expected_bytes, hf_token))
        return {"downloaded_bytes": 10, "expected_bytes": 20, "progress": 0.5}

    monkeypatch.setattr(
        "hub.services.models.downloads.get_gguf_download_progress_response",
        shared,
    )

    result = asyncio.run(
        models_route.get_gguf_download_progress(
            repo_id = "org/repo",
            variant = "Q4_K_M",
            expected_bytes = 20,
            hf_token = "token",
            current_subject = "test-user",
        )
    )

    assert result["progress"] == 0.5
    assert calls == [("org/repo", "Q4_K_M", 20, "token")]


def test_legacy_model_progress_delegates_to_shared_service(monkeypatch):
    calls = []

    async def shared(repo_id, *, hf_token):
        calls.append((repo_id, hf_token))
        return {"downloaded_bytes": 10, "expected_bytes": 20, "progress": 0.5}

    monkeypatch.setattr(
        "hub.services.models.downloads.get_download_progress_response",
        shared,
    )

    result = asyncio.run(
        models_route.get_download_progress(
            repo_id = "org/repo",
            hf_token = "token",
            current_subject = "test-user",
        )
    )

    assert result["progress"] == 0.5
    assert calls == [("org/repo", "token")]


def test_legacy_delete_delegates_to_shared_service(monkeypatch):
    calls = []

    async def shared(
        repo_id,
        variant,
        hf_token,
        cache_path = None,
    ):
        calls.append((repo_id, variant, hf_token, cache_path))
        return {"status": "deleted", "repo_id": repo_id}

    monkeypatch.setattr(
        "hub.services.models.deletion.delete_cached_model_response",
        shared,
    )

    result = asyncio.run(
        models_route.delete_cached_model(
            repo_id = "org/repo",
            variant = None,
            cache_path = "/data/hf/hub",
            hf_token = "token",
            current_subject = "test-user",
        )
    )

    assert result == {"status": "deleted", "repo_id": "org/repo"}
    assert calls == [("org/repo", None, "token", "/data/hf/hub")]


def test_arch_to_task_hides_unsupported_diffusion_from_chat():
    assert models_route._arch_to_task("flux") == "text-to-image"
    assert models_route._arch_to_task("z_image") == "text-to-image"
    assert models_route._arch_to_task("qwen_image") == "text-to-image"
    assert models_route._arch_to_task("llama") == "text-generation"
    assert models_route._arch_to_task(None) is None
    # Known-but-unsupported diffusion archs get a task that is neither chat nor a loadable image task, so both pickers skip them.
    for arch in ("sdxl", "sd1", "sd3", "lumina2", "hidream", "cosmos", "hyvid"):
        task = models_route._arch_to_task(arch)
        assert task == models_route._UNSUPPORTED_DIFFUSION_TASK
        assert task not in ("text-generation", "text-to-image")
    # A video arch with a REGISTERED VideoFamily surfaces with the Video-picker task.
    assert models_route._arch_to_task("ltxv") == models_route._VIDEO_GEN_TASK
    assert models_route._arch_to_task("ltxv") not in ("text-generation", "text-to-image")
    # A video arch that does not resolve from the bare arch alone ("wan" covers TI2V-5B and the A14B MoE) stays unsupported.
    assert models_route._arch_to_task("wan") == models_route._UNSUPPORTED_DIFFUSION_TASK
    assert models_route._arch_to_task("wan") not in ("text-generation", "text-to-image")
    # With a repo/file name hint the loadable TI2V-5B resolves to Video while the A14B MoE stays unsupported, matching the loader.
    assert (
        models_route._arch_to_task("wan", ("unsloth/Wan2.2-TI2V-5B-GGUF",))
        == models_route._VIDEO_GEN_TASK
    )
    assert (
        models_route._arch_to_task("wan", (None, "Wan2.2-TI2V-5B-Q4_K_M.gguf"))
        == models_route._VIDEO_GEN_TASK
    )
    assert (
        models_route._arch_to_task("wan", ("QuantStack/Wan2.2-T2V-A14B-GGUF",))
        == models_route._UNSUPPORTED_DIFFUSION_TASK
    )
    # Drift guard: every diffusion arch llama.cpp rejects as a chat model must classify here as some non-chat task.
    from core.inference.llama_cpp import LlamaCppBackend

    classified = (
        models_route._DIFFUSION_GGUF_ARCHS
        | models_route._UNSUPPORTED_DIFFUSION_GGUF_ARCHS
        | models_route._AMBIGUOUS_DIFFUSION_GGUF_ARCHS
        | models_route._VIDEO_GGUF_ARCHS
    )
    missing = {a for a in LlamaCppBackend._DIFFUSION_ARCHES if a.lower() not in classified}
    assert not missing, f"diffusion archs would still show in chat: {missing}"


def test_arch_to_task_tags_the_h3_gguf_bundle_as_video():
    # The published MiniMax-H3 GGUFs carry kv_count 0, so general.architecture is absent and the
    # arch read alone leaves the downloaded repo without a task -- dropped from the Video picker's
    # On Device list and offered to chat instead. Both bundle repo ids must resolve to Video.
    from core.inference.video_minimax_h3 import H3_GGUF_REPO

    for repo in (H3_GGUF_REPO, "leejet/MiniMax-H3-GGUF"):
        assert (
            models_route._arch_to_task(None, (repo, "minimax_h3_fl2va-Q4_K_M.gguf"))
            == models_route._VIDEO_GEN_TASK
        )
    # No hint is still unknown, and an unrelated repo is untouched.
    assert models_route._arch_to_task(None) is None
    assert models_route._arch_to_task(None, ("unsloth/Qwen3-GGUF", "q.gguf")) is None


def test_arch_to_task_resolves_z_image_gguf_tagged_lumina2():
    # Z-Image's DiT is a Lumina2 derivative, so both Z-Image GGUF repos declare general.architecture = "lumina2". Reading
    # the arch alone tagged the whole line unsupported and hid it, even though validate_load_request loads it happily.
    for repo, fname in (
        ("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf"),
        ("unsloth/Z-Image-GGUF", "z-image-Q8_0.gguf"),
    ):
        assert models_route._arch_to_task("lumina2", (repo, fname)) == "text-to-image"
        # The filename alone carries the family for a bare local .gguf pick.
        assert models_route._arch_to_task("lumina2", (None, fname)) == "text-to-image"
    # An unrecognised repo on the shared arch stays hidden rather than being guessed loadable.
    assert (
        models_route._arch_to_task("lumina2", ("someone/mystery-gguf", "model-Q4_K.gguf"))
        == models_route._UNSUPPORTED_DIFFUSION_TASK
    )


def test_arch_to_task_agrees_with_the_loader_on_ambiguous_archs():
    # The picker and the loader must not disagree: whatever _arch_to_task advertises as loadable, validate_load_request
    # must accept, and whatever it hides must be rejected. Otherwise the Images list hides a working model or offers a 400.
    from core.inference.diffusion import DiffusionBackend
    from core.inference.diffusion_families import _FAMILIES

    backend = DiffusionBackend.__new__(DiffusionBackend)  # validation touches no state
    for fam in _FAMILIES:
        repo = f"unsloth/{fam.name}-GGUF"
        fname = f"{fam.name}-Q4_K_M.gguf"
        task = models_route._arch_to_task("lumina2", (repo, fname))
        try:
            backend.validate_load_request(repo, gguf_filename = fname, model_kind = "gguf")
            loader_accepts = True
        except (ValueError, FileNotFoundError):
            loader_accepts = False
        assert (
            task == "text-to-image"
        ) == loader_accepts, f"{fam.name}: picker task={task} but loader accepts={loader_accepts}"


def _clear_chat_delete_guards(monkeypatch):
    """Report chat + orchestrator idle so only the Images / Video guards can refuse a delete."""
    import core.inference as core_inference
    import routes.inference as routes_inference

    monkeypatch.setattr(
        routes_inference,
        "get_llama_cpp_backend",
        lambda: SimpleNamespace(
            is_active = False,
            is_loaded = False,
            model_identifier = None,
            hf_variant = None,
        ),
    )
    monkeypatch.setattr(
        core_inference,
        "get_inference_backend",
        lambda: SimpleNamespace(active_model_name = None),
    )


def _idle_video_backend():
    return SimpleNamespace(
        status = lambda: {"loaded": False, "repo_id": None},
        loading_repo_ids = lambda: (),
    )


def _idle_diffusion_engine():
    return SimpleNamespace(
        status = lambda: {"loaded": False, "repo_id": None},
        loaded_repo_ids = lambda: (),
        loading_repo_ids = lambda: (),
    )


def test_delete_cached_refuses_diffusion_loaded_repo(monkeypatch):
    # The cached-delete guard refuses deleting a repo the Images backend has loaded, so its GGUF cannot vanish from under a live pipeline.
    from fastapi import HTTPException
    from hub.services.models import deletion
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod

    _clear_chat_delete_guards(monkeypatch)
    monkeypatch.setattr(
        der,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": True, "repo_id": "org/Z-Image-GGUF"},
            loaded_repo_ids = lambda: (),
            loading_repo_ids = lambda: (),
        ),
    )
    monkeypatch.setattr(video_mod, "get_video_backend", _idle_video_backend)

    try:
        asyncio.run(deletion.delete_cached_model_response("org/Z-Image-GGUF"))
        assert False, "expected HTTPException refusing the delete"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_delete_cached_refuses_video_loaded_repo(monkeypatch):
    # Same for the Video backend, which shares the On-Device GGUF delete UI with chat/Images.
    from fastapi import HTTPException
    from hub.services.models import deletion
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod

    _clear_chat_delete_guards(monkeypatch)
    monkeypatch.setattr(der, "get_active_diffusion_engine", _idle_diffusion_engine)
    monkeypatch.setattr(
        video_mod,
        "get_video_backend",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": True, "repo_id": "unsloth/LTX-2.3-GGUF"},
            loading_repo_ids = lambda: (),
        ),
    )

    try:
        asyncio.run(deletion.delete_cached_model_response("unsloth/LTX-2.3-GGUF"))
        assert False, "expected HTTPException refusing the delete"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_delete_cached_refuses_loaded_native_companion_repo(monkeypatch):
    # The native sd.cpp one-shot engine re-reads its companion VAE / text-encoder files every generation, so deleting a
    # companion repo while a FLUX GGUF is loaded must be refused. The repo_id does not match, so the guard needs loaded_repo_ids().
    from fastapi import HTTPException
    from hub.services.models import deletion
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod

    _clear_chat_delete_guards(monkeypatch)
    monkeypatch.setattr(
        der,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": True, "repo_id": "unsloth/FLUX.1-dev-GGUF"},
            loaded_repo_ids = lambda: (
                "unsloth/FLUX.1-dev-GGUF",
                "black-forest-labs/FLUX.1-dev",
                "unsloth/flux-text-encoders",
            ),
            loading_repo_ids = lambda: (),
        ),
    )
    monkeypatch.setattr(video_mod, "get_video_backend", _idle_video_backend)

    try:
        asyncio.run(deletion.delete_cached_model_response("unsloth/flux-text-encoders"))
        assert False, "expected HTTPException refusing the in-use companion delete"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_delete_cached_refuses_repo_a_diffusion_load_is_downloading(monkeypatch):
    # status().loaded is still False while a background Images load downloads the repo, so loading_repo_ids() must refuse the delete.
    from fastapi import HTTPException
    from hub.services.models import deletion
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod

    _clear_chat_delete_guards(monkeypatch)
    monkeypatch.setattr(
        der,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": False, "repo_id": None},
            loaded_repo_ids = lambda: (),
            loading_repo_ids = lambda: ("unsloth/Qwen-Image-2512-GGUF",),
        ),
    )
    monkeypatch.setattr(video_mod, "get_video_backend", _idle_video_backend)

    try:
        asyncio.run(deletion.delete_cached_model_response("unsloth/Qwen-Image-2512-GGUF"))
        assert False, "expected HTTPException refusing the delete mid-download"
    except HTTPException as e:
        assert e.status_code == 400
        assert "An Images model load is using this repo" in e.detail


def test_delete_cached_allows_sibling_of_loaded_diffusion_repo(monkeypatch):
    # A loaded Images repo must not block deleting a different cached repo sharing a name prefix; the guard is `/`-boundary aware.
    from fastapi import HTTPException
    from hub.services.models import deletion
    import core.inference.diffusion_engine_router as der
    import core.inference.video as video_mod

    _clear_chat_delete_guards(monkeypatch)
    monkeypatch.setattr(
        der,
        "get_active_diffusion_engine",
        lambda: SimpleNamespace(
            status = lambda: {"loaded": True, "repo_id": "Qwen/Qwen-Image-2512"},
            loaded_repo_ids = lambda: (),
            loading_repo_ids = lambda: (),
        ),
    )
    monkeypatch.setattr(video_mod, "get_video_backend", _idle_video_backend)
    # Stub the destructive stage: this test is about the guard boundary, not the cache walk.
    monkeypatch.setattr(
        deletion,
        "_delete_cached_model_blocking",
        lambda repo_id, variant, hf_token, cache_path = None: {
            "status": "deleted",
            "repo_id": repo_id,
        },
    )

    # The sibling repo clears every guard and reaches the delete.
    result = asyncio.run(deletion.delete_cached_model_response("Qwen/Qwen-Image"))
    assert result == {"status": "deleted", "repo_id": "Qwen/Qwen-Image"}

    # The loaded repo itself is still refused (exact match).
    try:
        asyncio.run(deletion.delete_cached_model_response("Qwen/Qwen-Image-2512"))
        assert False, "expected HTTPException refusing delete of the loaded repo"
    except HTTPException as e:
        assert e.status_code == 400
        assert "Unload the model before deleting" in e.detail


def test_cached_repo_partial_scopes_probe_to_snapshot_dir(monkeypatch):
    # The partial probe must be scoped to the snapshot row being listed: unscoped, a stale .incomplete copy in one cache root would flag a complete copy in another.
    import hub.utils.inventory_scan as scan

    calls = []

    def _fake(
        repo_type,
        repo_id,
        repo_cache_dir = None,
    ):
        calls.append((repo_type, repo_id, repo_cache_dir))
        return False

    monkeypatch.setattr(scan, "is_snapshot_partial", _fake)
    snapshot_dir = Path("/root_a/models--Org--Repo/snapshots/abc")
    assert models_route._cached_repo_partial("Org/Repo", snapshot_dir) is False
    assert calls == [("model", "Org/Repo", snapshot_dir)]

    monkeypatch.setattr(scan, "is_snapshot_partial", lambda *a, **k: True)
    assert models_route._cached_repo_partial("Org/Repo", snapshot_dir) is True

    # A probe error is swallowed (never hides a usable repo over a scan glitch).
    def _boom(*a, **k):
        raise RuntimeError("scan glitch")

    monkeypatch.setattr(scan, "is_snapshot_partial", _boom)
    assert models_route._cached_repo_partial("Org/Repo", snapshot_dir) is False


def test_repo_has_pipeline_index_requires_root_model_index(tmp_path):
    # Only a ROOT model_index.json makes a repo pipeline-loadable, so a nested subdir one must NOT clear the single_file flag; the helper scopes by snapshot path.
    snap = tmp_path / "snapshots" / "abc"
    nested = SimpleNamespace(
        file_name = "model_index.json",
        file_path = snap / "prior" / "model_index.json",
    )
    repo_nested = SimpleNamespace(
        repo_id = "unsloth/nested-index",
        revisions = [SimpleNamespace(files = [nested], snapshot_path = snap)],
    )
    assert models_route._repo_has_pipeline_index(repo_nested) is False

    root = SimpleNamespace(
        file_name = "model_index.json",
        file_path = snap / "model_index.json",
    )
    repo_root = SimpleNamespace(
        repo_id = "unsloth/root-index",
        revisions = [SimpleNamespace(files = [root], snapshot_path = snap)],
    )
    assert models_route._repo_has_pipeline_index(repo_root) is True


def test_pipeline_scans_read_the_snapshot_the_loader_will_open(tmp_path):
    # A repo cached twice (an older complete snapshot plus a newer companion-only one, the shape a GGUF load leaves) must be judged on the
    # snapshot from_pretrained resolves, the newest by mtime. Scanning every revision let the OLD transformer satisfy completeness.
    import os

    import hub.utils.inventory_scan as scan

    repo_dir = tmp_path / "models--Org--Repo"
    old_snap = repo_dir / "snapshots" / "old"
    new_snap = repo_dir / "snapshots" / "new"
    for d in (old_snap / "transformer", new_snap / "vae"):
        d.mkdir(parents = True)
    # A real manifest, not "{}": the scan reads the denoiser names off it, and one declaring none
    # has nothing it can prove absent.
    manifest = json.dumps(
        {
            "_class_name": "FluxPipeline",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
        }
    )
    (old_snap / "model_index.json").write_text(manifest, encoding = "utf-8")
    (new_snap / "model_index.json").write_text(manifest, encoding = "utf-8")
    # Real weights, not just the dirs: an empty transformer/ is a torn denoiser, not a present one.
    (old_snap / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 256)
    (new_snap / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"\0" * 256)
    # Make "new" unambiguously newer than "old" for the mtime rule both this and the loader use.
    os.utime(old_snap, (1_000_000, 1_000_000))
    os.utime(new_snap, (2_000_000, 2_000_000))

    def _rev(snap, files):
        return SimpleNamespace(
            snapshot_path = snap,
            last_modified = float(snap.stat().st_mtime),
            files = [SimpleNamespace(file_name = Path(f).name, file_path = snap / f) for f in files],
        )

    info = SimpleNamespace(
        repo_id = "Org/Repo",
        repo_path = repo_dir,
        revisions = [
            _rev(old_snap, ["model_index.json", "transformer/diffusion_pytorch_model.safetensors"]),
            _rev(new_snap, ["model_index.json", "vae/diffusion_pytorch_model.safetensors"]),
        ],
    )
    assert scan.repo_has_pipeline_index(info) is True
    assert scan.repo_pipeline_missing_denoiser(info) is True

    # The reverse cache (the complete snapshot is the newer one) still reports complete.
    os.utime(old_snap, (3_000_000, 3_000_000))
    info.revisions = [
        _rev(old_snap, ["model_index.json", "transformer/diffusion_pytorch_model.safetensors"]),
        _rev(new_snap, ["model_index.json", "vae/diffusion_pytorch_model.safetensors"]),
    ]
    assert scan.repo_pipeline_missing_denoiser(info) is False


@pytest.mark.parametrize(
    "extra_files, manifest_extra",
    [
        # A dual-denoiser pipeline whose second expert never landed.
        ({}, {"transformer_2": ["diffusers", "WanTransformer3DModel"]}),
        # A single denoiser whose shard index names two shards and got one.
        (
            {
                "transformer/diffusion_pytorch_model.safetensors.index.json": json.dumps(
                    {
                        "weight_map": {
                            "a": "diffusion_pytorch_model-00001-of-00002.safetensors",
                            "b": "diffusion_pytorch_model-00002-of-00002.safetensors",
                        }
                    }
                ).encode(),
                "transformer/diffusion_pytorch_model-00001-of-00002.safetensors": b"\0" * 256,
            },
            {},
        ),
    ],
    ids = ["dual-denoiser", "half-sharded"],
)
def test_both_cached_listings_agree_on_a_torn_pipeline(extra_files, manifest_extra, tmp_path):
    # /api/models/cached-models ORs the repo-wide helper while the hub inventory calls the snapshot
    # one, so a disagreement leaves a row the hub hides as partial still advertised as runnable.
    import hub.utils.inventory_scan as scan

    manifest = {
        "_class_name": "WanPipeline",
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
    }
    manifest.update(manifest_extra)
    repo_dir = tmp_path / "models--Org--Repo"
    snapshot = repo_dir / "snapshots" / "abc"
    snapshot.mkdir(parents = True)
    files = {
        "model_index.json": json.dumps(manifest).encode(),
        "vae/diffusion_pytorch_model.safetensors": b"\0" * 256,
    }
    if not extra_files:
        files["transformer/diffusion_pytorch_model.safetensors"] = b"\0" * 256
    files.update(extra_files)
    for name, blob in files.items():
        target = snapshot / name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(blob)
    info = SimpleNamespace(
        repo_id = "Org/Repo",
        repo_path = repo_dir,
        revisions = [
            SimpleNamespace(
                snapshot_path = snapshot,
                last_modified = float(snapshot.stat().st_mtime),
                files = [
                    SimpleNamespace(file_name = Path(n).name, file_path = snapshot / n) for n in files
                ],
            )
        ],
    )
    assert scan.snapshot_pipeline_missing_denoiser(snapshot) is True
    assert scan.repo_pipeline_missing_denoiser(info) is True


def test_list_cached_models_flags_single_file_diffusion_repos(monkeypatch, tmp_path):
    # A diffusion-tagged repo with NO top-level model_index.json is a single-file checkpoint (single_file=True); a full pipeline or chat repo carries no flag.
    single = _repo(
        "unsloth/Qwen-Image-fp8-single",
        [_file("qwen-image-fp8.safetensors", 10_000)],
        tmp_path / "models--unsloth--Qwen-Image-fp8-single",
    )
    pipeline = _repo(
        "unsloth/Qwen-Image-pipeline",
        [_file("model_index.json", 10), _file("transformer/model.safetensors", 10_000)],
        tmp_path / "models--unsloth--Qwen-Image-pipeline",
    )
    chat = _repo(
        "Org/ChatRepo",
        [_file("model.safetensors", 10_000)],
        tmp_path / "models--Org--ChatRepo",
    )

    monkeypatch.setattr(
        models_route,
        "_cached_repo_task",
        lambda repo_info: "text-to-image" if "Qwen-Image" in repo_info.repo_id else None,
    )
    monkeypatch.setattr(
        models_route,
        "_all_hf_cache_scans",
        lambda: [SimpleNamespace(repos = [single, pipeline, chat])],
    )

    result = asyncio.run(models_route.list_cached_models(current_subject = "test-user"))

    rows = {r["repo_id"]: r for r in result["cached"]}
    assert rows["unsloth/Qwen-Image-fp8-single"].get("single_file") is True
    assert "single_file" not in rows["unsloth/Qwen-Image-pipeline"]
    assert "single_file" not in rows["Org/ChatRepo"]


def _pipeline_repo(repo_id: str, tmp_path: Path) -> SimpleNamespace:
    return _repo(
        repo_id,
        [
            _file("model_index.json", 1_000),
            _file("transformer/diffusion_pytorch_model.safetensors", 5_000_000),
        ],
        tmp_path / f"models--{repo_id.replace('/', '--')}",
    )


def test_cached_repo_task_gates_an_image_pipeline_on_the_load_path_trust_rule(tmp_path):
    """Every advertised row must be loadable. A cached community pipeline has a model_index.json
    like any other, so tagging it text-to-image put a row in the Images picker that the loader's
    trust check refuses -- the pick 400s. Gate the tag on the same rule."""
    assert models_route._cached_repo_task(_pipeline_repo("unsloth/Qwen-Image", tmp_path)) == (
        "text-to-image"
    )
    assert (
        models_route._cached_repo_task(_pipeline_repo("someone/their-sdxl-mix", tmp_path)) is None
    )


def test_cached_repo_task_never_offers_an_sd_cpp_companion_repo_as_a_model(tmp_path):
    """The single-file VAE / text-encoder repos hold no denoiser, so none of them is a pick.

    Their unsloth mirrors clear the trust gate the old third-party ids never did, and the ids
    resolve to a family, so without the companion check each would list an unloadable Images row.
    """
    from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids

    for repo_id in (
        "unsloth/Z-Image-Turbo-ComfyUI",
        "unsloth/Qwen-Image-ComfyUI",
        "unsloth/FLUX.2-dev-ComfyUI",
        "unsloth/FLUX.2-VAE",
        "unsloth/FLUX.2-klein-9B-ComfyUI",
        "unsloth/flux-text-encoders",
    ):
        assert repo_id.lower() in sd_cpp_companion_only_repo_ids(), repo_id
        assert models_route._cached_repo_task(_pipeline_repo(repo_id, tmp_path)) is None, repo_id

    # FLUX.1-schnell also serves a companion VAE, but it is a real base and must stay loadable.
    assert "black-forest-labs/flux.1-schnell" not in sd_cpp_companion_only_repo_ids()
    assert (
        models_route._cached_repo_task(_pipeline_repo("black-forest-labs/FLUX.1-schnell", tmp_path))
        == "text-to-image"
    )


def test_a_companion_mirror_is_listed_but_flagged_so_no_picker_offers_it(monkeypatch, tmp_path):
    """A task of None does NOT drop the row: it is exactly what an unclassified CHAT repo carries,
    so the chat picker showed the companion as loadable. Deleting the row instead would hide tens
    of GB the user can then never find or remove, so the row stays and carries a flag the pickers
    filter on."""
    companion = _repo(
        "unsloth/Z-Image-Turbo-ComfyUI",
        [_file("split_files/vae/ae.safetensors", 300_000)],
        tmp_path / "models--unsloth--Z-Image-Turbo-ComfyUI",
    )
    chat = _repo(
        "unsloth/Qwen3-8B",
        [_file("model.safetensors", 900_000)],
        tmp_path / "models--unsloth--Qwen3-8B",
    )
    monkeypatch.setattr(
        models_route, "_all_hf_cache_scans", lambda: [SimpleNamespace(repos = [companion, chat])]
    )

    rows = {
        r["repo_id"]: r
        for r in asyncio.run(models_route.list_cached_models(current_subject = "test-user"))["cached"]
    }

    # Listed, so it stays visible and deletable...
    assert "unsloth/Z-Image-Turbo-ComfyUI" in rows
    # ...and flagged, which is the part a task of None could never express.
    assert rows["unsloth/Z-Image-Turbo-ComfyUI"]["companion"] is True
    assert rows["unsloth/Z-Image-Turbo-ComfyUI"]["task"] is None
    # An ordinary chat repo carries the same task of None and must NOT be flagged.
    assert rows["unsloth/Qwen3-8B"].get("companion") is None
    assert rows["unsloth/Qwen3-8B"]["task"] is None


def test_the_companion_set_never_hides_a_repo_that_is_a_real_chat_model(tmp_path):
    """sd.cpp borrows unsloth/Qwen2.5-VL-7B-Instruct-GGUF as a text encoder, but it is a genuine
    chat model. It is in the companion set, so the only thing keeping it safe is that the listing
    this set feeds never sees a GGUF-only repo. Pin that, or a future caller takes a downloaded
    model away from the user."""
    from core.inference.diffusion_families import sd_cpp_companion_only_repo_ids

    assert "unsloth/qwen2.5-vl-7b-instruct-gguf" in sd_cpp_companion_only_repo_ids()
    # A GGUF-only repo has no .safetensors / .bin, so list_cached_models drops it before the flag.
    gguf_only = _repo(
        "unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        [_file("Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", 4_000_000)],
        tmp_path / "models--unsloth--Qwen2.5-VL-7B-Instruct-GGUF",
    )
    assert not [
        f
        for rev in gguf_only.revisions
        for f in rev.files
        if f.file_name.endswith((".safetensors", ".bin"))
    ]


def test_cached_repo_task_hides_an_untrusted_video_repo_instead_of_listing_it_under_images(
    monkeypatch, tmp_path
):
    """A detected video pipeline that fails the video trust rule used to fall through to the image
    fallback and show up in the Images picker, where it is just as unloadable."""
    import core.inference.video as video_mod

    repo = _pipeline_repo("someone/their-ltx-fork", tmp_path)
    monkeypatch.setattr(
        "core.inference.video_families.detect_video_family",
        lambda repo_id: object(),
    )
    monkeypatch.setattr(video_mod, "_is_trusted_video_repo", lambda repo_id: False)
    assert models_route._cached_repo_task(repo) is None

    monkeypatch.setattr(video_mod, "_is_trusted_video_repo", lambda repo_id: True)
    assert models_route._cached_repo_task(repo) == models_route._VIDEO_GEN_TASK


def test_hub_cached_rows_carry_the_task_the_pickers_filter_on(monkeypatch, tmp_path):
    """The picker's On Device rows come from the /api/hub inventory, not the models API. Without a
    task on those rows the Images and Video pickers filtered every one of them out, and the chat
    picker's diffusion routing (which reads the same field) never fired."""
    from hub.schemas.inventory import CachedGgufRepo, CachedModelRepo
    from hub.services.models import cache_inventory

    assert "task" in CachedGgufRepo.model_fields
    assert "task" in CachedModelRepo.model_fields

    repo = _pipeline_repo("unsloth/Qwen-Image", tmp_path)
    monkeypatch.setattr(
        "routes.models._cached_repo_task", lambda repo_info: "text-to-image", raising = True
    )
    assert cache_inventory._cached_row_task(repo, gguf = False) == "text-to-image"
    monkeypatch.setattr(
        "routes.models._repo_gguf_task", lambda repo_info: "text-generation", raising = True
    )
    assert cache_inventory._cached_row_task(repo, gguf = True) == "text-generation"


def test_hub_cached_row_task_never_hides_a_row_when_classification_fails(monkeypatch, tmp_path):
    # Best-effort, like the models API: a classifier that raises leaves the row untagged rather than dropping it.
    from hub.services.models import cache_inventory

    def _boom(repo_info):
        raise RuntimeError("unreadable")

    monkeypatch.setattr("routes.models._cached_repo_task", _boom, raising = True)
    assert cache_inventory._cached_row_task(_pipeline_repo("a/b", tmp_path), gguf = False) is None


def test_hub_local_rows_are_tagged_with_their_task():
    """/api/hub/local feeds the same pickers, and its rows were untagged too."""
    import inspect

    from hub.schemas.inventory import LocalModelInfo
    from hub.services.models import local_inventory

    assert "task" in LocalModelInfo.model_fields
    src = inspect.getsource(local_inventory.list_local_models_response)
    assert "_local_model_task" in src
    assert 'model_copy(update = {"task"' in src


def test_pipeline_class_guard_fires_before_any_download():
    # The 0.39-only families used to die with a bare AttributeError deep in the load, after the checkpoint was fetched, on
    # the older diffusers packaging still allows on Python 3.9. Validation refuses first, naming the version and the fix.
    import pytest

    from core.inference.diffusion_families import assert_pipeline_class_available

    stub = types.SimpleNamespace(__version__ = "0.37.0")
    real = sys.modules.get("diffusers")
    sys.modules["diffusers"] = stub
    try:
        # ValueError, like every other unloadable-pick refusal: RuntimeError reached /images/load's 409 and escaped download-plan as a 500.
        with pytest.raises(ValueError) as excinfo:
            assert_pipeline_class_available("ZImagePipeline", "z-image")
    finally:
        if real is not None:
            sys.modules["diffusers"] = real
        else:
            del sys.modules["diffusers"]
    msg = str(excinfo.value)
    assert "z-image" in msg and "ZImagePipeline" in msg
    assert "0.39" in msg and "0.37.0" in msg
    assert "3.10" in msg  # names the Python floor that carries a new enough diffusers


def test_pipeline_class_guard_passes_every_shipped_family():
    # Split out of the guard test above so it can skip on its own rather than take the guard assertion down with it: the backend
    # CI image installs the CPU-only dep set with no diffusers, and the native sd.cpp engine legitimately serves GGUF picks there.
    # The stub-driven refusal above needs no real diffusers and still runs; only this sweep does not.
    import pytest

    pytest.importorskip("diffusers")

    from core.inference.diffusion_families import _FAMILIES, assert_pipeline_class_available

    for fam in _FAMILIES:
        assert_pipeline_class_available(fam.pipeline_class, fam.name)


def test_pipeline_class_guard_is_silent_when_diffusers_is_absent(monkeypatch):
    # Not this check's business: it answers "is the installed diffusers new enough for this family", and with nothing installed
    # there is no version to judge. What must NOT happen is the raise: ModuleNotFoundError is not the ValueError the routes map
    # to 400, so it escaped /images/download-plan as a bare 500 with the message lost, the precise failure this guard prevents.
    # A pick that really does need diffusers fails later, in the loader, with its own message.
    import builtins

    from core.inference.diffusion_families import assert_pipeline_class_available

    monkeypatch.delitem(sys.modules, "diffusers", raising = False)
    real_import = builtins.__import__

    def _blocked(name, *args, **kwargs):
        if name == "diffusers" or name.startswith("diffusers."):
            raise ImportError("No module named 'diffusers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked)
    assert assert_pipeline_class_available("ZImagePipeline", "z-image") is None


def test_cached_pipeline_needs_a_detectable_image_family(monkeypatch):
    # A top-level model_index.json only proves the repo is a diffusers pipeline: an unsloth-hosted pipeline of a class this backend
    # cannot assemble cleared the trust gate, was advertised, then failed validate_load_request. Both gates now, like the video branch.
    monkeypatch.setattr(models_route, "_repo_has_pipeline_index", lambda info: True)

    def _task(repo_id):
        return models_route._cached_repo_task(SimpleNamespace(repo_id = repo_id, repo_path = "/x"))

    # Trusted AND a detected family -> claimed by Images.
    assert _task("unsloth/Z-Image-Turbo") == "text-to-image"
    assert _task("unsloth/FLUX.1-dev") == "text-to-image"
    # Trusted but no image family the loader can detect -> not advertised.
    assert _task("unsloth/some-unsupported-pipeline") is None
    # Untrusted keeps its existing refusal.
    assert _task("someone/random-diffusers-pipeline") is None


def test_cached_repo_task_agrees_with_the_image_loader(monkeypatch):
    # Same invariant as the GGUF arch test: whatever the picker advertises as loadable, validate_load_request must accept.
    from core.inference.diffusion import DiffusionBackend

    monkeypatch.setattr(models_route, "_repo_has_pipeline_index", lambda info: True)
    backend = DiffusionBackend.__new__(DiffusionBackend)
    for repo_id in (
        "unsloth/Z-Image-Turbo",
        "unsloth/FLUX.1-dev",
        "unsloth/some-unsupported-pipeline",
        "unsloth/stable-audio-open-1.0",
    ):
        task = models_route._cached_repo_task(SimpleNamespace(repo_id = repo_id, repo_path = "/x"))
        try:
            backend.validate_load_request(repo_id)
            loader_accepts = True
        except (ValueError, FileNotFoundError, RuntimeError):
            loader_accepts = False
        assert (
            task == "text-to-image"
        ) == loader_accepts, f"{repo_id}: picker task={task} but loader accepts={loader_accepts}"


def test_cached_picker_hides_a_family_this_diffusers_cannot_build(monkeypatch):
    # The newer families exist only from diffusers 0.39, which cannot be installed on Python 3.9 at all, so advertising one
    # there is a pick that can only fail; the picker applies the same availability check validate_load_request does.
    import types

    import routes.models as models_module
    from core.inference.diffusion_families import detect_family, family_pipeline_available

    fam = detect_family("unsloth/Z-Image-Turbo")
    assert fam is not None
    # Present in this environment's diffusers, so the row is offered.
    assert family_pipeline_available(fam) is True

    monkeypatch.setattr(models_module, "_repo_is_diffusers", lambda info: True)
    monkeypatch.setattr("core.inference.diffusion._is_trusted_diffusion_repo", lambda repo_id: True)
    info = types.SimpleNamespace(repo_id = "unsloth/Z-Image-Turbo")
    assert models_module._cached_repo_task(info) == "text-to-image"

    # An older diffusers without the pipeline class hides the row instead.
    monkeypatch.setattr(
        "core.inference.diffusion_families.family_pipeline_available", lambda f: False
    )
    assert models_module._cached_repo_task(info) is None


def test_family_pipeline_available_fails_open_without_diffusers(monkeypatch):
    # No diffusers at all is a different problem the load path reports properly; a listing must not hide every image model over it.
    import sys

    from core.inference.diffusion_families import detect_family, family_pipeline_available

    monkeypatch.setitem(sys.modules, "diffusers", None)
    assert family_pipeline_available(detect_family("unsloth/Z-Image-Turbo")) is True
    assert family_pipeline_available(None) is False


# ── the unbuildable-family gate on the GGUF paths (both engines) ─────────────


def _pretend_old_diffusers(monkeypatch, *, engine):
    """An environment whose diffusers has none of the newer pipeline classes, on a host whose GGUF
    loads route to ``engine``.

    0.36.0 is the real ceiling for a Python 3.9 host (0.37.0 already declares requires-python
    >=3.10), and it ships no Flux2KleinPipeline. Only the diffusers module and the engine prediction
    are substituted: the availability check, the picker and validate_load_request are the real code.
    """
    import core.inference.diffusion_engine_router as router

    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(__version__ = "0.36.0"))
    monkeypatch.setattr(router, "predict_engine", lambda fam, **kwargs: engine)


def test_gguf_picker_hides_a_family_no_engine_here_can_build(monkeypatch):
    # The gate landed on the cached-repo picker only, so the GGUF repos -- the ones the Images picker actually offers for
    # these families -- still showed as text-to-image on a diffusers too old to build them, and every pick died.
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS

    _pretend_old_diffusers(monkeypatch, engine = ENGINE_DIFFUSERS)

    # The flat diffusion-arch branch (FLUX.2) and the ambiguous one (Z-Image ships as "lumina2").
    assert (
        models_route._arch_to_task("flux2", ("unsloth/FLUX.2-klein-4B-GGUF",))
        == models_route._UNSUPPORTED_DIFFUSION_TASK
    )
    assert (
        models_route._arch_to_task("lumina2", ("unsloth/Z-Image-Turbo-GGUF",))
        == models_route._UNSUPPORTED_DIFFUSION_TASK
    )
    # Neither chat nor Images: the row is hidden, not moved to the picker that would also fail.
    assert models_route._arch_to_task("flux2", ("unsloth/FLUX.2-klein-4B-GGUF",)) not in (
        "text-generation",
        "text-to-image",
    )


def test_gguf_picker_keeps_a_family_the_native_engine_serves(monkeypatch):
    # The opposite mistake: on a CPU/MPS or force-native host sd.cpp loads the GGUF and never instantiates a diffusers class, so hiding the row would withhold a working model.
    from core.inference.sd_cpp_engine import ENGINE_SD_CPP

    _pretend_old_diffusers(monkeypatch, engine = ENGINE_SD_CPP)

    assert models_route._arch_to_task("flux2", ("unsloth/FLUX.2-klein-4B-GGUF",)) == "text-to-image"
    assert models_route._arch_to_task("lumina2", ("unsloth/Z-Image-Turbo-GGUF",)) == "text-to-image"


def test_the_loader_demands_the_diffusers_class_only_when_diffusers_loads_it(monkeypatch):
    # Same predicate on the load path: refuse a too-old diffusers before the download, and never when sd.cpp serves the GGUF.
    import pytest

    from core.inference.diffusion import DiffusionBackend
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

    backend = DiffusionBackend.__new__(DiffusionBackend)

    _pretend_old_diffusers(monkeypatch, engine = ENGINE_SD_CPP)
    fam = backend.validate_load_request(
        "unsloth/FLUX.2-klein-4B-GGUF",
        gguf_filename = "flux2-klein-4b-Q4_0.gguf",
        model_kind = "gguf",
    )
    assert fam.name == "flux.2-klein"

    _pretend_old_diffusers(monkeypatch, engine = ENGINE_DIFFUSERS)
    with pytest.raises(ValueError) as excinfo:
        backend.validate_load_request(
            "unsloth/FLUX.2-klein-4B-GGUF",
            gguf_filename = "flux2-klein-4b-Q4_0.gguf",
            model_kind = "gguf",
        )
    # ValueError, not RuntimeError: /images/load maps RuntimeError to 409 and /images/download-plan catches only (ValueError, FileNotFoundError), so the message escaped as a 500.
    assert "Flux2KleinPipeline" in str(excinfo.value)


def test_the_video_picker_hides_a_family_this_diffusers_cannot_build(monkeypatch):
    # Same gap on the video branches: LTX-2's pipeline class is 0.39-only too, and video has no native engine to fall back
    # to, so the load asserts it unconditionally (video.py -> assert_pipeline_class_available).
    monkeypatch.setattr(models_route, "_repo_is_diffusers", lambda info: True)
    info = SimpleNamespace(repo_id = "Lightricks/LTX-2", repo_path = "/x")
    # Offered on this environment's diffusers ...
    assert models_route._arch_to_task("ltxv") == models_route._VIDEO_GEN_TASK
    assert models_route._cached_repo_task(info) == models_route._VIDEO_GEN_TASK

    # ... and hidden on one that has no LTX2Pipeline.
    monkeypatch.setitem(sys.modules, "diffusers", types.SimpleNamespace(__version__ = "0.36.0"))
    assert models_route._arch_to_task("ltxv") == models_route._UNSUPPORTED_DIFFUSION_TASK
    assert models_route._cached_repo_task(info) is None


def test_every_shipped_video_family_resolves_on_this_diffusers():
    # Drift guard: the video picker hides a family whose pipeline class the installed diffusers lacks, so a stale class name here would hide a working model.
    from core.inference.diffusion_families import family_pipeline_available
    from core.inference.video_families import _FAMILIES as _VIDEO_FAMILIES
    for fam in _VIDEO_FAMILIES:
        assert family_pipeline_available(
            fam
        ), f"{fam.name}: {fam.pipeline_class} is not in diffusers"


def test_the_gguf_picker_and_the_image_loader_agree_on_an_old_diffusers(monkeypatch):
    # The invariant test_cached_repo_task_agrees_with_the_image_loader states for cached repos, applied to the GGUF path on
    # both host kinds: whatever the picker advertises must be accepted, and whatever it hides must be refused.
    from core.inference.diffusion import DiffusionBackend
    from core.inference.sd_cpp_engine import ENGINE_DIFFUSERS, ENGINE_SD_CPP

    backend = DiffusionBackend.__new__(DiffusionBackend)
    picks = (
        ("flux2", "unsloth/FLUX.2-klein-4B-GGUF", "flux2-klein-4b-Q4_0.gguf"),
        ("lumina2", "unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q8_0.gguf"),
    )
    for engine in (ENGINE_DIFFUSERS, ENGINE_SD_CPP):
        _pretend_old_diffusers(monkeypatch, engine = engine)
        for arch, repo_id, filename in picks:
            task = models_route._arch_to_task(arch, (repo_id, filename))
            try:
                backend.validate_load_request(repo_id, gguf_filename = filename, model_kind = "gguf")
                loader_accepts = True
            except (ValueError, FileNotFoundError, RuntimeError):
                loader_accepts = False
            assert (
                (task == "text-to-image") == loader_accepts
            ), f"{repo_id} on {engine}: picker task={task} but loader accepts={loader_accepts}"


def test_a_cancelled_siblings_resume_survives_the_local_listing(monkeypatch, tmp_path):
    """A sibling cancelled before any file landed lives only in download state. The disk-only
    listing is built from the cache, which cannot see it, so the repo reads as holding one quant
    and the picker collapses it into a single row, taking the sibling's resume with it."""
    active = tmp_path / "active"
    repo_dir = active / "models--Org--Quant"
    snapshot = repo_dir / "snapshots" / ("d" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("d" * 40, encoding = "utf-8")

    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(
            hub_cache = active, hf_home = tmp_path, source = "studio", cache_home = tmp_path
        ),
    )
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda **kw: [active])
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.hf_cache_root",
        lambda create = False, root = None: root if root is not None else active,
    )

    # Disk-only means disk-only: a remote listing here would be the bug this route avoids.
    def _no_remote(*args, **kwargs):
        raise AssertionError("remote listing attempted")

    monkeypatch.setattr(GV, "list_gguf_variants", _no_remote)

    # Control: nothing cancelled, so the repo holds one quant and nothing else.
    only_complete = asyncio.run(GV.get_gguf_variants_response("Org/Quant", prefer_local_cache = True))
    assert [(v.quant, v.downloaded) for v in only_complete.variants] == [("Q4_K_M", True)]

    from hub.utils import download_manifest

    assert download_manifest.write_cancel_marker("model", "Org/Quant", "Q8_0", hub_cache = active)

    response = asyncio.run(GV.get_gguf_variants_response("Org/Quant", prefer_local_cache = True))
    assert {v.quant for v in response.variants if v.downloaded} == {"Q4_K_M"}
    assert {v.quant for v in response.variants if v.partial} == {"Q8_0"}


def test_a_cancelled_sibling_survives_a_failed_remote_listing(monkeypatch, tmp_path):
    """The expander asks the remote-first route. When the Hub cannot answer, offline or a private
    repo, the fallback reads the cache, which cannot see a sibling cancelled before any file
    landed. That is exactly when a resume has nowhere else to surface."""
    active = tmp_path / "active"
    repo_dir = active / "models--Org--Quant"
    snapshot = repo_dir / "snapshots" / ("d" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("d" * 40, encoding = "utf-8")

    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(
            hub_cache = active, hf_home = tmp_path, source = "studio", cache_home = tmp_path
        ),
    )
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda **kw: [active])
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.hf_cache_root",
        lambda create = False, root = None: root if root is not None else active,
    )

    def _unreachable(*args, **kwargs):
        raise RuntimeError("hub unreachable")

    monkeypatch.setattr(GV, "list_gguf_variants", _unreachable)

    from hub.utils import download_manifest

    assert download_manifest.write_cancel_marker("model", "Org/Quant", "Q8_0", hub_cache = active)

    # No prefer_local_cache: this is the route the expander uses.
    response = asyncio.run(GV.get_gguf_variants_response("Org/Quant"))
    assert {v.quant for v in response.variants if v.downloaded} == {"Q4_K_M"}
    assert {v.quant for v in response.variants if v.partial} == {"Q8_0"}


def test_a_cancelled_siblings_marker_shows_on_the_repo_row(monkeypatch, tmp_path):
    """A sibling cancelled before any file landed moves neither the repo's bytes nor its mtime,
    and its own partial flag stays false while another quant is clean. Without a signal of its
    own on the row, a reader watching for on-disk change cannot see it happen."""
    active = tmp_path / "active"
    repo_dir = active / "models--Org--Quant"
    snapshot = repo_dir / "snapshots" / ("d" * 40)
    snapshot.mkdir(parents = True)
    (snapshot / "Model-Q4_K_M.gguf").write_bytes(b"\0" * 256)
    (repo_dir / "refs").mkdir(parents = True)
    (repo_dir / "refs" / "main").write_text("d" * 40, encoding = "utf-8")

    monkeypatch.setattr(
        "utils.hf_cache_settings.get_hf_cache_paths",
        lambda: SimpleNamespace(
            hub_cache = active, hf_home = tmp_path, source = "studio", cache_home = tmp_path
        ),
    )
    monkeypatch.setattr("hub.utils.hf_cache_state.hf_cache_roots", lambda **kw: [active])
    monkeypatch.setattr(
        "hub.utils.hf_cache_state.hf_cache_root",
        lambda create = False, root = None: root if root is not None else active,
    )

    from hub.services.models import cache_inventory
    from hub.utils import download_manifest, inventory_scan

    def _row():
        inventory_scan.invalidate_hf_cache_scans()
        cache_inventory.invalidate_hf_cache_scans()
        rows = cache_inventory._scan_cached_gguf()
        return next(r for r in rows if r["repo_id"] == "Org/Quant")

    before = _row()
    assert before["has_variant_state"] is False

    assert download_manifest.write_cancel_marker("model", "Org/Quant", "Q8_0", hub_cache = active)

    after = _row()
    # The signals that would otherwise have to carry it.
    assert after["size_bytes"] == before["size_bytes"]
    assert after["last_modified"] == before["last_modified"]
    assert after["partial"] is False
    assert after["has_variant_state"] is True


def test_identical_variant_scans_in_flight_run_once(monkeypatch):
    """Aborting the HTTP request cannot stop the scan already running in its thread, so the
    picker's Retry would start another against a filesystem that is not answering. Measured
    before this: 23 retries filled all 20 default-executor workers and starved unrelated
    offloaded work."""
    scans = []
    release = threading.Event()

    def slow_scan(path):
        scans.append(path)
        release.wait(5)
        return ([], False)

    monkeypatch.setattr(GV, "is_local_path", lambda _p: True)
    monkeypatch.setattr(GV, "list_local_gguf_variants", slow_scan)

    async def drive():
        pending = [
            asyncio.ensure_future(GV.get_gguf_variants_response("/models/x")) for _ in range(8)
        ]
        await asyncio.sleep(0.1)
        release.set()
        return await asyncio.gather(*pending)

    try:
        results = asyncio.run(drive())
    finally:
        release.set()

    assert len(results) == 8
    assert len(scans) == 1, f"the scan ran {len(scans)} times"


def test_variant_scans_for_different_requests_do_not_share(monkeypatch):
    """Coalescing must key on everything that changes the answer."""
    scans = []

    def scan(path):
        scans.append(path)
        return ([], False)

    monkeypatch.setattr(GV, "is_local_path", lambda _p: True)
    monkeypatch.setattr(GV, "list_local_gguf_variants", scan)

    async def drive():
        await GV.get_gguf_variants_response("/models/a")
        await GV.get_gguf_variants_response("/models/b")
        await GV.get_gguf_variants_response("/models/a", offline = True)
        await GV.get_gguf_variants_response("/models/a", local_path = "/other")

    asyncio.run(drive())
    assert len(scans) == 4


def test_a_failed_variant_scan_is_not_pinned(monkeypatch):
    """A failure must reach every waiter and leave nothing cached, or one bad scan would
    answer for the rest of the session."""
    attempts = []

    def failing_scan(path):
        attempts.append(path)
        raise OSError("mount went away")

    monkeypatch.setattr(GV, "is_local_path", lambda _p: True)
    monkeypatch.setattr(GV, "list_local_gguf_variants", failing_scan)

    async def drive():
        for _ in range(3):
            with pytest.raises(Exception):
                await GV.get_gguf_variants_response("/models/x")

    asyncio.run(drive())
    assert len(attempts) == 3, "a failure was reused instead of retried"


def test_one_caller_giving_up_leaves_the_scan_for_the_others(monkeypatch):
    """The picker abandons its request when the row collapses; the caller still waiting
    must still get an answer."""
    release = threading.Event()
    scans = []

    def slow_scan(path):
        scans.append(path)
        release.wait(5)
        return ([], False)

    monkeypatch.setattr(GV, "is_local_path", lambda _p: True)
    monkeypatch.setattr(GV, "list_local_gguf_variants", slow_scan)

    async def drive():
        staying = asyncio.ensure_future(GV.get_gguf_variants_response("/models/x"))
        leaving = asyncio.ensure_future(GV.get_gguf_variants_response("/models/x"))
        await asyncio.sleep(0.1)
        leaving.cancel()
        release.set()
        return await staying

    try:
        answer = asyncio.run(drive())
    finally:
        release.set()

    assert answer.repo_id == "/models/x"
    assert len(scans) == 1


def test_offline_context_follows_the_hf_cache_when_it_answers(monkeypatch, tmp_path):
    """The HF cache answers before local_path, so with both present the length must come
    from the cache. Picking local_path on the offline flag alone attached another copy's
    context to the cache's variants. Real service, no stub."""
    cache_snapshot = tmp_path / "hub" / "models--org--repo" / "snapshots" / "rev"
    cache_snapshot.mkdir(parents = True)
    context_calls = []

    monkeypatch.setattr(
        GV,
        "select_gguf_cache_snapshot",
        lambda repo_id, root = None: (
            [
                SimpleNamespace(
                    filename = "m-Q4_K_M.gguf",
                    quant = "Q4_K_M",
                    display_label = None,
                    size_bytes = 10,
                )
            ],
            False,
            {"q4_k_m"},
            cache_snapshot,
        ),
    )
    monkeypatch.setattr(GV, "list_partial_gguf_variants_from_state", lambda *a, **k: None)
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((model, is_local)) or 4096,
    )

    asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            offline = True,
            local_path = str(tmp_path),  # an ordinary directory, not this repo's cache
            hf_token = None,
            current_subject = "test-user",
        )
    )
    assert context_calls == [(str(cache_snapshot), True)]


def test_offline_context_follows_the_cache_the_variants_were_read_from(monkeypatch, tmp_path):
    """A local_path under a non-active cache scopes the listing to that cache, so the length
    has to come from there too. Falling back to the repo id walks every cache, active one
    first, and can attach another copy's context to these variants. Real service, no stub."""
    legacy_repo = tmp_path / "legacy" / "hub" / "models--org--repo"
    legacy_snapshot = legacy_repo / "snapshots" / "rev"
    legacy_snapshot.mkdir(parents = True)
    context_calls = []

    monkeypatch.setattr(
        GV,
        "select_gguf_cache_snapshot",
        lambda repo_id, root = None: (
            [
                SimpleNamespace(
                    filename = "m-Q4_K_M.gguf",
                    quant = "Q4_K_M",
                    display_label = None,
                    size_bytes = 10,
                )
            ],
            False,
            {"q4_k_m"},
            legacy_snapshot,
        ),
    )
    monkeypatch.setattr(GV, "list_partial_gguf_variants_from_state", lambda *a, **k: None)
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((model, is_local)) or 4096,
    )

    asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            offline = True,
            local_path = str(legacy_repo),
            hf_token = None,
            current_subject = "test-user",
        )
    )
    assert context_calls == [(str(legacy_snapshot), True)]


def test_failed_hub_context_follows_the_cache_the_variants_were_read_from(monkeypatch, tmp_path):
    """When the Hub request fails, the fallback cache that supplied the variants must also
    supply their context metadata. Otherwise the route searches by repo id and can read a
    different cache copy first."""
    legacy_repo = tmp_path / "legacy" / "hub" / "models--org--repo"
    legacy_snapshot = legacy_repo / "snapshots" / "rev"
    legacy_snapshot.mkdir(parents = True)
    context_calls = []

    def _unreachable(*args, **kwargs):
        raise RuntimeError("hub unreachable")

    monkeypatch.setattr(GV, "list_gguf_variants", _unreachable)
    monkeypatch.setattr(
        GV,
        "select_gguf_cache_snapshot",
        lambda repo_id, root = None: (
            [
                SimpleNamespace(
                    filename = "m-Q4_K_M.gguf",
                    quant = "Q4_K_M",
                    display_label = None,
                    size_bytes = 10,
                )
            ],
            False,
            {"q4_k_m"},
            legacy_snapshot,
        ),
    )
    monkeypatch.setattr(GV, "list_partial_gguf_variants_from_state", lambda *a, **k: None)
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((model, is_local)) or 4096,
    )

    asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            local_path = str(legacy_repo),
            hf_token = None,
            current_subject = "test-user",
        )
    )
    assert context_calls == [(str(legacy_snapshot), True)]


def test_switching_cache_storage_does_not_join_a_stuck_scan(monkeypatch, tmp_path):
    """Pointing Studio at another cache has to start a fresh scan. Coalescing on the request
    alone made the new request wait on the scan wedged against the old volume."""
    import storage.studio_db as studio_db
    import utils.hf_cache_settings as hf_cache_settings

    wedged = threading.Event()
    cache_home = [tmp_path / "wedgedvol"]
    scanned = []

    # Only the stored setting picks the cache home here.
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {})
    monkeypatch.setattr(
        studio_db,
        "get_app_setting",
        lambda key, default = None: (
            str(cache_home[0]) if key == hf_cache_settings.CACHE_HOME_SETTING_KEY else default
        ),
    )

    def scan(repo_id, root = None):
        scanned.append(str(root))
        if "wedgedvol" in str(root):
            wedged.wait(5)
        return None

    monkeypatch.setattr(GV, "select_gguf_cache_snapshot", scan)
    monkeypatch.setattr(GV, "list_partial_gguf_variants_from_state", lambda *a, **k: None)
    monkeypatch.setattr(GV, "list_local_gguf_variants", lambda _p: ([], False))
    monkeypatch.setattr(GV, "_snapshot_scope_for_request", lambda *a, **k: None)

    async def drive():
        stuck = asyncio.ensure_future(GV.get_gguf_variants_answer("org/repo", offline = True))
        await asyncio.sleep(0.2)
        cache_home[0] = tmp_path / "healthyvol"
        second = asyncio.ensure_future(GV.get_gguf_variants_answer("org/repo", offline = True))
        done, _ = await asyncio.wait({second}, timeout = 1.5)
        for task in (stuck, second):
            task.cancel()
        return bool(done)

    try:
        answered = asyncio.run(drive())
    finally:
        wedged.set()

    assert any("wedgedvol" in root for root in scanned), scanned
    assert answered, "the new request waited on the scan stuck against the old cache"
    assert any("healthyvol" in root for root in scanned), scanned


def test_gguf_variants_route_carries_local_resolution(monkeypatch, tmp_path):
    # The CLI gate reads resolved_locally, so the legacy constructor must carry it.
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models" / "qwen").mkdir(parents = True)
    (tmp_path / "models" / "qwen" / "q-Q4_K_M.gguf").write_bytes(b"GGUF")

    result = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "models/qwen", hf_token = None, current_subject = "test-user"
        )
    )
    assert result.resolved_locally is True
    assert [v.quant for v in result.variants] == ["Q4_K_M"]


def _write_cached_gguf(
    hub_cache: Path,
    repo_id: str,
    filename: str,
    mtime: float | None = None,
    revision: str = "rev",
) -> Path:
    """One real snapshot under *hub_cache*; *mtime* pins which one a repo-wide walk picks."""
    import os

    repo_dir = hub_cache / ("models--" + repo_id.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents = True, exist_ok = True)
    (snapshot / filename).write_bytes(b"GGUF" + b"\0" * 32)
    if mtime is not None:
        os.utime(snapshot, (mtime, mtime))
    return repo_dir


def _pin_caches(monkeypatch, active: Path, roots: list[Path]) -> None:
    import utils.hf_cache_settings as hf_cache_settings
    from hub.utils import hf_cache_state

    monkeypatch.setattr(
        hf_cache_settings,
        "get_hf_cache_paths",
        lambda: SimpleNamespace(
            hf_home = active.parent,
            hub_cache = active,
            xet_cache = active.parent / "xet",
            source = "test",
        ),
    )
    monkeypatch.setattr(hf_cache_state, "hf_cache_roots", lambda: list(roots))


def _unreachable_hub(monkeypatch) -> None:
    """Fail only the network call, so the real cache fallback inside the lister runs."""
    import huggingface_hub

    def _boom(self, *args, **kwargs):
        raise OSError("hub unreachable")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", _boom)


def test_failed_hub_lists_the_selected_cache_not_another_one(monkeypatch, tmp_path):
    """A request pinned to one cache must list that cache's quants.

    ``list_gguf_variants`` read the cache repo-wide on an unreachable Hub, so the active
    copy's quants answered for a request pinned elsewhere.
    """
    active = tmp_path / "active" / "hub"
    selected = tmp_path / "selected" / "hub"
    active.mkdir(parents = True)
    selected.mkdir(parents = True)
    # The active copy is newer, so only a scoped lookup can surface the selected one.
    _write_cached_gguf(active, "org/repo", "m-Q4_K_M.gguf", mtime = 2_000_000_000)
    selected_repo = _write_cached_gguf(selected, "org/repo", "m-Q8_0.gguf", mtime = 1_000_000_000)

    _pin_caches(monkeypatch, active, [active, selected])
    _unreachable_hub(monkeypatch)
    monkeypatch.setattr(GV, "list_partial_gguf_variants_from_state", lambda *a, **k: None)

    context_calls = []
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((str(model), is_local)) or 4096,
    )

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            local_path = str(selected_repo),
            hf_token = None,
            current_subject = "test-user",
        )
    )
    assert sorted(v.quant for v in response.variants) == ["Q8_0"]
    # The context length has to come off that same copy, down to the snapshot.
    assert context_calls == [(str(selected_repo / "snapshots" / "rev"), True)]


def test_an_unreadable_cache_root_is_skipped_not_fatal(tmp_path):
    """One unreadable cache must not take down a walk the other roots can answer.

    A bare ``is_dir()`` on ``<repo>/snapshots`` let EACCES escape (up to 3.13) and turned a
    usable listing into a 500. Asserted on the walk, so it pins the guard, not the fallback.
    """
    from hub.utils.gguf import iter_hf_cache_snapshots, list_gguf_variants_from_hf_cache

    blocked_root = tmp_path / "blocked" / "hub"
    readable_root = tmp_path / "readable" / "hub"
    blocked_root.mkdir(parents = True)
    readable_root.mkdir(parents = True)
    blocked_repo = _write_cached_gguf(blocked_root, "org/repo", "m-Q4_K_M.gguf")
    _write_cached_gguf(readable_root, "org/repo", "m-Q8_0.gguf")

    blocked_repo.chmod(0o000)
    try:
        try:
            (blocked_repo / "snapshots").is_dir()
        except OSError:
            pass
        else:
            pytest.skip("filesystem does not enforce the permission (root?)")

        # Scoped at the unreadable root: no snapshots, and no exception.
        assert list(iter_hf_cache_snapshots("org/repo", root = blocked_root)) == []
        assert list_gguf_variants_from_hf_cache("org/repo", root = blocked_root) is None

        # A readable root is still walked normally.
        readable = list(iter_hf_cache_snapshots("org/repo", root = readable_root))
        assert len(readable) == 1
        listed = list_gguf_variants_from_hf_cache("org/repo", root = readable_root)
        assert listed is not None
        assert [v.quant for v in listed[0]] == ["Q8_0"]
    finally:
        blocked_repo.chmod(0o755)


def test_another_caches_quant_is_offered_as_a_download_not_as_downloaded(monkeypatch, tmp_path):
    """Readiness is counted against the cache the request names, never against another one.

    The pinned cache holds the repo but no GGUF, so the lister answers repo-wide. Those
    variants must stay download targets, or the row offers a load that cannot resolve.
    """
    pinned_root = tmp_path / "pinned" / "hub"
    other_root = tmp_path / "other" / "hub"
    pinned_root.mkdir(parents = True)
    other_root.mkdir(parents = True)
    pinned_repo = pinned_root / "models--org--repo"
    (pinned_repo / "snapshots" / "rev").mkdir(parents = True)
    _write_cached_gguf(other_root, "org/repo", "m-Q8_0.gguf")

    _pin_caches(monkeypatch, pinned_root, [pinned_root, other_root])
    _unreachable_hub(monkeypatch)
    monkeypatch.setattr(GV, "list_partial_gguf_variants_from_state", lambda *a, **k: None)
    monkeypatch.setattr(
        models_route, "_read_native_context_length", lambda model, *, is_local: None
    )

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            local_path = str(pinned_repo),
            hf_token = None,
            current_subject = "test-user",
        )
    )
    assert [v.quant for v in response.variants] == ["Q8_0"]
    assert [v.downloaded for v in response.variants] == [False]


def test_context_follows_the_answering_revision_not_a_sibling(monkeypatch, tmp_path):
    """The context read is pinned to the snapshot that answered, not the repo dir.

    The read walks the whole dir, so naming the dir let a skipped revision supply the length.
    """
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir(parents = True)
    # Only the newer snapshot holds a whole quant, so it is the one that answers.
    repo_dir = _write_cached_gguf(
        hub_cache, "org/repo", "m-Q4_K_M.gguf", mtime = 1_000_000_000, revision = "older"
    )
    _write_cached_gguf(hub_cache, "org/repo", "m-Q8_0.gguf", mtime = 2_000_000_000, revision = "newer")

    _pin_caches(monkeypatch, hub_cache, [hub_cache])
    _unreachable_hub(monkeypatch)
    monkeypatch.setattr(GV, "list_partial_gguf_variants_from_state", lambda *a, **k: None)

    context_calls = []
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((str(model), is_local)) or 4096,
    )

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            local_path = str(repo_dir),
            hf_token = None,
            current_subject = "test-user",
        )
    )
    assert [v.quant for v in response.variants] == ["Q8_0"]
    assert context_calls == [(str(repo_dir / "snapshots" / "newer"), True)]


def test_a_case_variant_repo_dir_still_names_its_snapshot(monkeypatch, tmp_path):
    """Provenance survives a repo_id whose case differs from the cached dir's.

    The lister folds case, but the repo dir was rebuilt from repo_id, so it failed ``is_dir()``
    on a case-sensitive filesystem and the length fell back to a repo-wide walk.
    """
    hub_cache = tmp_path / "hub"
    hub_cache.mkdir(parents = True)
    repo_dir = _write_cached_gguf(hub_cache, "org/repo", "m-Q8_0.gguf")

    _pin_caches(monkeypatch, hub_cache, [hub_cache])
    _unreachable_hub(monkeypatch)
    monkeypatch.setattr(GV, "list_partial_gguf_variants_from_state", lambda *a, **k: None)

    context_calls = []
    monkeypatch.setattr(
        models_route,
        "_read_native_context_length",
        lambda model, *, is_local: context_calls.append((str(model), is_local)) or 4096,
    )

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/Repo",  # on disk as models--org--repo
            hf_token = None,
            current_subject = "test-user",
        )
    )
    assert [v.quant for v in response.variants] == ["Q8_0"]
    assert context_calls == [(str(repo_dir / "snapshots" / "rev"), True)]


def test_a_download_scope_is_not_listed_as_a_quant(monkeypatch, tmp_path):
    """A scoped job ("@diffusion") rides the variant slot, so its manifest names the
    same .gguf the real quant does: rebuilding from download state listed one file
    twice, the second permanently partial, costing the picker its single-quant
    collapse. A real cancelled quant must still survive, or it loses its resume."""
    repo_dir = tmp_path / "models--org--repo"
    (repo_dir / "snapshots" / "rev").mkdir(parents = True)

    monkeypatch.setattr(
        GV,
        "select_gguf_cache_snapshot",
        lambda repo_id, root = None: (
            [
                SimpleNamespace(
                    filename = "m-Q4_K_S.gguf",
                    quant = "Q4_K_S",
                    display_label = None,
                    size_bytes = 10,
                )
            ],
            False,
            {"q4_k_s"},
            repo_dir / "snapshots" / "rev",
        ),
    )
    # State holds both: the scope the diffusion load ran under, and a cancelled quant.
    monkeypatch.setattr(
        GV,
        "list_partial_gguf_variants_from_state",
        lambda *a, **k: (
            [
                SimpleNamespace(
                    filename = "m-Q4_K_S.gguf",
                    quant = "@diffusion",
                    display_label = None,
                    size_bytes = 10,
                    download_size_bytes = 10,
                ),
                SimpleNamespace(
                    filename = "m-Q6_K.gguf",
                    quant = "Q6_K",
                    display_label = None,
                    size_bytes = 20,
                    download_size_bytes = 20,
                ),
            ],
            False,
        ),
    )
    monkeypatch.setattr(
        models_route, "_read_native_context_length", lambda model, *, is_local: None
    )

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            offline = True,
            local_path = str(repo_dir),
            hf_token = None,
            current_subject = "test-user",
        )
    )
    quants = [v.quant for v in response.variants]
    assert "@diffusion" not in quants
    assert quants == ["Q4_K_S", "Q6_K"]
    # The scope named the quant's own file, so keeping it would list that file twice.
    filenames = [v.filename for v in response.variants]
    assert len(filenames) == len(set(filenames))


def test_a_scope_alone_in_state_is_not_an_answer(monkeypatch, tmp_path):
    """A scope reaching the state-only fallbacks is worse than a duplicate row: naming
    no .gguf, it reconstructs as "@diffusion.gguf", a file never on disk. So scopes
    alone must make the fallback decline and let the next answer through."""
    repo_dir = tmp_path / "models--org--repo"
    (repo_dir / "snapshots" / "rev").mkdir(parents = True)

    monkeypatch.setattr(GV, "select_gguf_cache_snapshot", lambda repo_id, root = None: None)
    monkeypatch.setattr(
        GV,
        "list_partial_gguf_variants_from_state",
        lambda *a, **k: (
            [
                SimpleNamespace(
                    filename = "@diffusion.gguf",
                    quant = "@diffusion",
                    display_label = None,
                    size_bytes = 0,
                    download_size_bytes = 0,
                )
            ],
            False,
        ),
    )
    monkeypatch.setattr(
        models_route, "_read_native_context_length", lambda model, *, is_local: None
    )

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            offline = True,
            local_path = str(repo_dir),
            hf_token = None,
            current_subject = "test-user",
        )
    )
    assert response.variants == []


def test_a_cancelled_quant_beside_a_scope_still_answers_from_state(monkeypatch, tmp_path):
    """Dropping scopes must not cost the fallback its reason to exist: a cancelled quant
    with no snapshot keeps its row, and its resume."""
    repo_dir = tmp_path / "models--org--repo"
    (repo_dir / "snapshots" / "rev").mkdir(parents = True)

    monkeypatch.setattr(GV, "select_gguf_cache_snapshot", lambda repo_id, root = None: None)
    monkeypatch.setattr(
        GV,
        "list_partial_gguf_variants_from_state",
        lambda *a, **k: (
            [
                SimpleNamespace(
                    filename = "m-Q6_K.gguf",
                    quant = "Q6_K",
                    display_label = None,
                    size_bytes = 20,
                    download_size_bytes = 20,
                ),
                SimpleNamespace(
                    filename = "@diffusion.gguf",
                    quant = "@diffusion",
                    display_label = None,
                    size_bytes = 0,
                    download_size_bytes = 0,
                ),
            ],
            False,
        ),
    )
    monkeypatch.setattr(
        models_route, "_read_native_context_length", lambda model, *, is_local: None
    )

    response = asyncio.run(
        models_route.get_gguf_variants(
            repo_id = "org/repo",
            offline = True,
            local_path = str(repo_dir),
            hf_token = None,
            current_subject = "test-user",
        )
    )
    assert [(v.quant, v.partial) for v in response.variants] == [("Q6_K", True)]
