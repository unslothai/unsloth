# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import sys
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
    active = tmp_path / "active"
    previous = tmp_path / "previous"
    active_partial = active / "models--Org--Model" / "blobs" / "abc.incomplete"
    active_partial.parent.mkdir(parents = True)
    active_partial.write_bytes(b"partial")
    snapshot = previous / "models--Org--Model" / "snapshots" / "revision"
    snapshot.mkdir(parents = True)
    (snapshot / "model.safetensors").write_bytes(b"complete")

    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: active)
    monkeypatch.setattr("utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr("utils.paths.lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr(
        "utils.hf_cache_settings.known_hf_hub_caches",
        lambda: [active, previous],
    )
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [])

    rows = models_route.collect_local_models(tmp_path / "models")

    [row] = [row for row in rows if row.model_id == "Org/Model"]
    assert row.id == str(snapshot.resolve())
    assert row.partial is False
    assert row.active_cache is False


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
    monkeypatch.setattr("storage.studio_db.list_scan_folders", lambda: [{"path": str(root)}])
    monkeypatch.setattr("utils.paths.lmstudio_model_dirs", lambda: [])
    monkeypatch.setattr("utils.paths.legacy_hf_cache_dir", lambda: tmp_path / "legacy")
    monkeypatch.setattr("utils.paths.hf_default_cache_dir", lambda: tmp_path / "default")
    monkeypatch.setattr("utils.hf_cache_settings.known_hf_hub_caches", lambda: [])
    monkeypatch.setattr(models_route, "_resolve_hf_cache_dir", lambda: tmp_path / "active")

    rows = models_route.collect_local_models(tmp_path / "models")
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
        lambda create = False, root = None: (root if root is not None else active),
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
        lambda create = False, root = None: (root if root is not None else active),
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
        lambda create = False, root = None: (root if root is not None else active),
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

    assert result["cached"] == [{"repo_id": "Org/MmprojAux", "size_bytes": 15_000}]


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
        return SimpleNamespace(
            repo_id = repo_id,
            variants = [],
            has_vision = False,
            default_variant = None,
        )

    context_calls = []
    monkeypatch.setattr(GV, "get_gguf_variants_response", scoped_variants)
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
        return SimpleNamespace(repo_id = repo_id, variants = [], has_vision = False, default_variant = None)

    context_calls = []
    monkeypatch.setattr(GV, "get_gguf_variants_response", scoped_variants)
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
        return SimpleNamespace(repo_id = repo_id, variants = [], has_vision = False, default_variant = None)

    context_calls = []
    monkeypatch.setattr(GV, "get_gguf_variants_response", scoped_variants)
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
        return SimpleNamespace(repo_id = repo_id, variants = [], has_vision = False, default_variant = None)

    monkeypatch.setattr(GV, "get_gguf_variants_response", scoped_variants)
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

    def dragging_walk(root):
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
    assert elapsed < 2
    assert len(visited) < 200


def test_native_context_read_still_reports_a_length_within_budget(monkeypatch, tmp_path):
    """Control: the bound only trims a walk that drags; a header reached in time still answers."""
    gguf = tmp_path / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"x")

    monkeypatch.setattr(models_route, "_iter_gguf_paths", lambda root: iter([gguf]))
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
        lambda create = False, root = None: (root if root is not None else active),
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
        lambda create = False, root = None: (root if root is not None else active),
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
        lambda create = False, root = None: (root if root is not None else active),
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
