# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Only caches may be deleted, and only ones this backend resolved itself.

The purge takes a cache KEY and turns it into a directory here, so these tests
cover the two halves that keep that honest: the sizing that decides what the UI
offers, and every refusal that stands between a key and an rmtree.
"""

import os
from pathlib import Path

import pytest

from utils import cache_inventory
from utils.cache_inventory import (
    CACHE_KEYS,
    CachePurgeRefused,
    assert_purgeable_root,
    cache_inventory as build_inventory,
    definition_for,
    describe_cache,
    empty_cache_root,
    purge_caches,
)


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(text, encoding = "utf-8")
    return path


@pytest.fixture
def isolated_caches(tmp_path, monkeypatch):
    """Point every resolver at tmp_path, so no test can reach a real cache.

    The host running these tests has real HF_HOME / XDG_CACHE_HOME values, and
    the resolvers are supposed to honour exactly those, so the isolation has to
    be as complete as it is here.
    """
    from utils import cache_cleanup, hf_cache_settings

    hf_home = tmp_path / "hf"
    (hf_home / "hub").mkdir(parents = True)
    (hf_home / "xet").mkdir()
    paths = hf_cache_settings.HuggingFaceCachePaths(
        hf_home, hf_home / "hub", hf_home / "xet", "studio"
    )
    monkeypatch.setattr(hf_cache_settings, "get_hf_cache_paths", lambda: paths)
    monkeypatch.setattr(hf_cache_settings, "known_hf_cache_homes", lambda: [hf_home])
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [hf_home / "hub"])
    # The compiled cache is cleared through cache_cleanup's own ownership model,
    # which test_cache_cleanup covers; here it must simply not reach a real one.
    monkeypatch.setattr(cache_cleanup, "_cleanable_cache_dirs", lambda: [])
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        monkeypatch.delenv(key, raising = False)
    for key, name in (
        ("HF_DATASETS_CACHE", "hf/datasets"),
        ("HF_ASSETS_CACHE", "hf/assets"),
        ("HF_XET_CACHE", "hf/xet"),
        ("UV_CACHE_DIR", "uv"),
        ("PIP_CACHE_DIR", "pip"),
        ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
        ("TORCH_EXTENSIONS_DIR", "torchext"),
        ("TRITON_CACHE_DIR", "triton"),
        ("CUDA_CACHE_PATH", "cuda"),
        ("NUMBA_CACHE_DIR", "numba"),
        ("VLLM_CACHE_ROOT", "vllm"),
        ("npm_config_cache", "npm"),
        ("BUN_INSTALL_CACHE_DIR", "bun"),
    ):
        monkeypatch.setenv(key, str(tmp_path / name))
    monkeypatch.delenv("MPLCONFIGDIR", raising = False)
    # Sizes are memoized for a minute in production; a test must never read one
    # another test measured.
    monkeypatch.setattr(cache_inventory, "_size_cache", {})
    monkeypatch.setattr(cache_inventory, "_size_epochs", {})
    return hf_home


# --- sizing ---------------------------------------------------------------


def test_a_cache_is_sized_from_its_own_bytes(tmp_path, monkeypatch, isolated_caches):
    root = tmp_path / "uv"
    _write(root / "a.bin", "a" * 100)
    _write(root / "nested" / "b.bin", "b" * 50)
    entry = describe_cache(definition_for("uv"))
    assert entry["present"] is True
    assert entry["size_bytes"] == 150
    # Top-level entries: the file and the directory, not every leaf below it.
    assert entry["entry_count"] == 2
    assert entry["purgeable"] is True


def test_sizing_does_not_follow_a_symlink_out_of_the_cache(tmp_path, isolated_caches):
    outside = tmp_path / "outside"
    _write(outside / "big.bin", "z" * 4096)
    root = tmp_path / "uv"
    root.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory = True)
    _write(root / "real.bin", "r" * 10)
    entry = describe_cache(definition_for("uv"))
    assert entry["size_bytes"] == 10


def test_hardlinked_blobs_are_counted_once(tmp_path, isolated_caches):
    root = tmp_path / "uv"
    blob = _write(root / "blobs" / "sha", "q" * 500)
    (root / "snapshots").mkdir()
    os.link(blob, root / "snapshots" / "model.bin")
    entry = describe_cache(definition_for("uv"))
    assert entry["size_bytes"] == 500


def test_an_absent_cache_reports_no_paths(tmp_path, isolated_caches):
    entry = describe_cache(definition_for("bun"))
    assert entry["present"] is False
    assert entry["paths"] == []
    assert entry["size_bytes"] == 0


def test_the_inventory_totals_exclude_the_opt_in_caches(tmp_path, isolated_caches):
    _write(tmp_path / "uv" / "a.bin", "a" * 100)
    _write(isolated_caches / "hub" / "models--x" / "w.bin", "m" * 900)
    inventory = build_inventory(refresh = True)
    by_key = {entry["key"]: entry for entry in inventory["caches"]}
    assert by_key["hf_hub"]["opt_in"] is True
    assert by_key["hf_hub"]["size_bytes"] == 900
    assert inventory["total_bytes"] == 1000
    # Reclaimable is what a bulk purge would actually free, so the cache whose
    # deletion re-downloads models is not in it.
    assert inventory["reclaimable_bytes"] == 100


def test_every_key_the_api_exposes_has_a_definition():
    assert len(CACHE_KEYS) == len(set(CACHE_KEYS))
    for key in CACHE_KEYS:
        assert definition_for(key).key == key


# --- refusals -------------------------------------------------------------


@pytest.mark.parametrize("raw", ["/", "/home", os.path.expanduser("~")])
def test_a_root_at_or_near_the_filesystem_root_is_refused(raw):
    with pytest.raises(CachePurgeRefused):
        assert_purgeable_root(Path(raw))


def test_a_shallow_root_is_refused_even_when_nothing_protects_it(tmp_path):
    # One component below the anchor ("/mnt", "C:\\Users"): not a protected
    # location, a real directory, and never a cache. The depth rule is the only
    # thing between a variable pointed there and an rmtree of the contents.
    shallow = Path(*tmp_path.parts[:2])
    if not shallow.is_dir() or shallow.is_symlink() or len(shallow.parts) != 2:
        pytest.skip(f"no shallow non-symlink directory to test with ({shallow})")
    with pytest.raises(CachePurgeRefused) as excinfo:
        assert_purgeable_root(shallow, protected = set(), trees = set())
    assert "too close to the filesystem root" in str(excinfo.value)


def test_a_relative_root_is_refused():
    with pytest.raises(CachePurgeRefused):
        assert_purgeable_root(Path("relative/cache"))


def test_a_root_holding_studio_data_is_refused(tmp_path, monkeypatch):
    from utils.paths.storage_roots import studio_root

    home = studio_root()
    home.mkdir(parents = True, exist_ok = True)
    _write(home / "studio.db", "sqlite")
    with pytest.raises(CachePurgeRefused) as excinfo:
        assert_purgeable_root(home)
    assert "protected" in str(excinfo.value)


def test_a_root_inside_the_projects_folder_is_refused(tmp_path, monkeypatch):
    projects = tmp_path / "Projects"
    (projects / "my-run" / "cache").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(projects))
    with pytest.raises(CachePurgeRefused) as excinfo:
        assert_purgeable_root(projects / "my-run" / "cache")
    assert "protected folder" in str(excinfo.value)


def test_a_symlinked_cache_root_is_refused(tmp_path):
    target = tmp_path / "somewhere" / "deep"
    target.mkdir(parents = True)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory = True)
    with pytest.raises(CachePurgeRefused) as excinfo:
        assert_purgeable_root(link)
    assert "symlink" in str(excinfo.value)


def test_the_purge_api_takes_keys_and_never_a_path():
    from routes.settings import CachePurgePayload

    assert set(CachePurgePayload.model_fields) == {"keys"}
    with pytest.raises(ValueError) as excinfo:
        purge_caches(["/"])
    assert "Unknown cache key" in str(excinfo.value)
    with pytest.raises(ValueError):
        purge_caches([os.path.expanduser("~")])
    with pytest.raises(ValueError):
        purge_caches(["../../etc"])


def test_an_unknown_key_deletes_nothing_from_the_valid_ones(tmp_path, isolated_caches):
    kept = _write(tmp_path / "uv" / "a.bin", "a" * 100)
    with pytest.raises(ValueError):
        purge_caches(["uv", "not_a_cache"])
    assert kept.exists()


# --- deletion behaviour ---------------------------------------------------


def test_a_purge_empties_the_root_without_removing_it(tmp_path, isolated_caches):
    root = tmp_path / "uv"
    _write(root / "a.bin", "a" * 100)
    _write(root / "sub" / "b.bin", "b" * 20)
    result = purge_caches(["uv"])
    assert root.is_dir()
    assert list(root.iterdir()) == []
    assert result["freed_bytes"] == 120
    assert result["results"][0]["errors"] == []


def test_a_symlink_inside_the_cache_is_unlinked_and_never_followed(tmp_path, isolated_caches):
    outside = tmp_path / "precious"
    kept_file = _write(outside / "model.safetensors", "weights")
    root = tmp_path / "uv"
    root.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory = True)
    (root / "escape-file").symlink_to(kept_file)
    purge_caches(["uv"])
    assert kept_file.exists()
    assert outside.is_dir()
    assert not (root / "escape").exists()
    assert not (root / "escape").is_symlink()
    assert not (root / "escape-file").is_symlink()


def test_a_purge_leaves_the_database_models_projects_and_token_alone(
    tmp_path, monkeypatch, isolated_caches
):
    from utils.paths.storage_roots import studio_root

    home = studio_root()
    home.mkdir(parents = True, exist_ok = True)
    database = _write(home / "studio.db", "sqlite")
    chat_history = _write(home / "chat" / "thread.json", "hello")
    token = _write(isolated_caches / "token", "hf_secret")
    model = _write(isolated_caches / "hub" / "models--unsloth--x" / "blobs" / "abc", "weights" * 10)
    projects = tmp_path / "Projects"
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(projects))
    project_file = _write(projects / "run" / "adapter.safetensors", "lora")
    dataset = _write(tmp_path / "hf" / "datasets" / "ds" / "data.arrow", "rows")
    cache_file = _write(tmp_path / "uv" / "wheel.whl", "cached")

    bulk = [key for key in CACHE_KEYS if not definition_for(key).opt_in]
    purge_caches(bulk)

    assert not cache_file.exists()
    for survivor in (database, chat_history, token, model, project_file, dataset):
        assert survivor.exists(), f"{survivor} was deleted by a cache purge"


def test_clearing_the_hub_cache_is_asked_for_by_itself(tmp_path, isolated_caches):
    token = _write(isolated_caches / "token", "hf_secret")
    blob = _write(isolated_caches / "hub" / "models--x" / "blobs" / "abc", "weights")
    purge_caches(["hf_hub"])
    assert not blob.exists()
    assert (isolated_caches / "hub").is_dir()
    # The token lives in the cache HOME, one level above the hub cache.
    assert token.exists()


def test_a_refused_root_reports_itself_and_deletes_nothing(tmp_path, monkeypatch, isolated_caches):
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(
        cache_inventory,
        "protected_trees",
        lambda: {Path(os.path.realpath(outputs))},
    )
    kept = _write(outputs / "triton" / "kernel.cubin", "compiled")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(outputs / "triton"))
    entry = describe_cache(definition_for("triton"))
    assert entry["purgeable"] is False
    assert "protected folder" in entry["blocked_reason"]
    result = purge_caches(["triton"])
    assert kept.exists()
    assert result["freed_bytes"] == 0
    assert result["results"][0]["errors"]


@pytest.fixture
def only_the_configured_compiled_cache(monkeypatch, tmp_path):
    """Keep the compiled-cache clear off any install-tree cache of this checkout.

    cache_cleanup's own ownership model stays in force: that is the thing under
    test here, and re-deriving it in cache_inventory is exactly what this
    feature must not do.

    _configured_cache_dirs also offers the CWD, and importing unsloth creates a
    compiled cache there, so these run from a directory that has none. Skipping
    on one instead meant they never ran at all.
    """
    from utils import cache_cleanup

    monkeypatch.setattr(cache_cleanup, "_CACHE_DIRS", [])
    monkeypatch.chdir(tmp_path)


def test_the_compiled_cache_is_cleared_through_the_module_that_owns_it(
    tmp_path, monkeypatch, only_the_configured_compiled_cache
):
    from utils import cache_cleanup

    compiled = tmp_path / "compiled_cache"
    compiled.mkdir()
    (compiled / cache_cleanup.CACHE_MARKER).touch()
    generated = _write(compiled / "unsloth_compiled_module_llama.py", "compiled" * 10)
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(compiled))

    entry = describe_cache(definition_for("unsloth_compiled"))
    assert entry["present"] is True
    assert entry["size_bytes"] >= 80

    result = purge_caches(["unsloth_compiled"])
    assert result["results"][0]["errors"] == []
    assert not generated.exists()
    # The marker is rewritten, so the next cleanup still knows the directory is ours.
    assert (compiled / cache_cleanup.CACHE_MARKER).is_file()


def test_a_shared_compiled_cache_only_loses_the_generated_modules(
    tmp_path, monkeypatch, only_the_configured_compiled_cache
):
    shared = tmp_path / "somebody-elses-folder"
    shared.mkdir()
    generated = _write(shared / "unsloth_compiled_module_llama.py", "compiled")
    mine = _write(shared / "my_notes.py", "print('hello')")
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(shared))

    purge_caches(["unsloth_compiled"])
    assert not generated.exists()
    assert mine.exists()


def test_a_pattern_limited_root_keeps_the_files_that_are_not_cache(tmp_path, monkeypatch):
    config_dir = tmp_path / "mplconfig"
    monkeypatch.setenv("MPLCONFIGDIR", str(config_dir))
    style = _write(config_dir / "matplotlibrc", "font.size: 12")
    fontlist = _write(config_dir / "fontlist-v390.json", "{}")
    empty_cache_root(config_dir, patterns = cache_inventory.MATPLOTLIB_PATTERNS)
    assert style.exists()
    assert not fontlist.exists()


def test_the_root_of_a_purged_cache_is_never_the_deletion_target(tmp_path, isolated_caches):
    root = tmp_path / "uv"
    _write(root / "a.bin", "a")
    before = root.stat().st_ino
    purge_caches(["uv"])
    assert root.is_dir()
    # Same directory, not a recreated one: nothing recreates it with different
    # ownership or permissions.
    assert root.stat().st_ino == before


def test_a_purge_forgets_the_size_it_measured_before_it(tmp_path, isolated_caches):
    _write(tmp_path / "uv" / "a.bin", "a" * 100)
    assert build_inventory()["total_bytes"] == 100
    result = purge_caches(["uv"])
    # The inventory the purge returns cannot still offer the 100 bytes it just
    # deleted, even though the memo was warm.
    assert result["inventory"]["total_bytes"] == 0
    assert build_inventory()["total_bytes"] == 0


def test_the_compiled_cache_is_never_swept_up_by_a_bulk_purge():
    """A bulk purge may cost a recompile or a re-download, not a running job: a
    worker imports generated modules from here long after it started."""
    definition = definition_for("unsloth_compiled")
    assert definition.opt_in is True

    inventory = build_inventory()
    bulk = {
        entry["key"]
        for entry in inventory["caches"]
        if entry["present"] and entry["purgeable"] and not entry["opt_in"]
    }
    assert "unsloth_compiled" not in bulk

    # ...and it is still individually purgeable, or the row would be dead.
    entry = next(e for e in inventory["caches"] if e["key"] == "unsloth_compiled")
    assert entry["opt_in"] is True


def test_a_junctioned_cache_root_is_refused_like_a_symlink(tmp_path, monkeypatch):
    """Only IO_REPARSE_TAG_SYMLINK sets S_IFLNK, so is_symlink() is False for a
    junction while realpath() follows it and the TARGET would be emptied."""
    target = tmp_path / "somewhere" / "deep"
    target.mkdir(parents = True)
    kept = _write(target / "not-a-cache.txt", "mine")
    junction = tmp_path / "cache"
    junction.mkdir()

    # os.path.isjunction is the 3.12+ answer and is always False on POSIX, so
    # the platform test is what a Windows host would report here.
    monkeypatch.setattr(os.path, "isjunction", lambda path: Path(path) == junction, raising = False)
    assert not junction.is_symlink()

    with pytest.raises(CachePurgeRefused) as excinfo:
        assert_purgeable_root(junction)
    assert "junction" in str(excinfo.value)
    assert kept.exists()


def test_a_bulk_clear_cannot_reach_an_opt_in_cache_nested_in_another_root(
    tmp_path, monkeypatch, isolated_caches
):
    """HF_DATASETS_CACHE below UV_CACHE_DIR would otherwise go with the outer root."""
    uv = tmp_path / "uv"
    datasets = uv / "hf-datasets"
    _write(uv / "wheels" / "wheel.whl", "w" * 10)
    dataset_file = _write(datasets / "squad" / "data.arrow", "d" * 10)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(datasets))

    described = describe_cache(definition_for("uv"))
    assert described["purgeable"] is False
    assert "opt-in cache" in (described["blocked_reason"] or "")

    result = purge_caches(["uv"])
    assert dataset_file.exists()
    assert result["freed_bytes"] == 0

    # ...and the nested cache is still clearable when it is the one asked for.
    purge_caches(["hf_datasets"])
    assert not dataset_file.exists()


def test_an_explicit_hub_cache_outside_a_hub_folder_stays_clearable(
    tmp_path, monkeypatch, isolated_caches
):
    """HF_HUB_CACHE=/mnt/hf-cache makes the DISPLAY home the hub itself, and
    protecting it there guards no credential: the token lives in the HF home."""
    from utils import hf_cache_settings

    hub = tmp_path / "hf-cache"
    _write(hub / "models--org--model" / "blob", "m" * 10)
    paths = hf_cache_settings.HuggingFaceCachePaths(
        hub, hub, tmp_path / "xet", "environment", "HF_HUB_CACHE"
    )
    monkeypatch.setattr(hf_cache_settings, "get_hf_cache_paths", lambda: paths)
    monkeypatch.setattr(hf_cache_settings, "known_hf_hub_caches", lambda: [hub])

    described = describe_cache(definition_for("hf_hub"))
    assert described["purgeable"] is True, described["blocked_reason"]

    # ...and the HF home that holds the token is still refused.
    with pytest.raises(CachePurgeRefused):
        assert_purgeable_root(isolated_caches)


# --- resolvers ------------------------------------------------------------


def test_the_inductor_cache_follows_the_account_name_torch_uses(
    tmp_path, monkeypatch, isolated_caches
):
    """A container with no USER puts torch's cache at torchinductor_root, not _0."""
    import tempfile

    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising = False)
    monkeypatch.delenv("USER", raising = False)
    monkeypatch.delenv("USERNAME", raising = False)
    monkeypatch.setenv("LOGNAME", "root")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    root = tmp_path / "torchinductor_root"
    _write(root / "fxgraph" / "entry.bin", "c" * 20)

    entry = describe_cache(definition_for("torch_inductor"))
    assert entry["paths"] == [str(root)]
    assert entry["size_bytes"] == 20


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_the_vllm_cache_is_vllms_own_default_on_every_platform(
    tmp_path, monkeypatch, isolated_caches, platform
):
    """vllm/envs.py is XDG_CACHE_HOME or ~/.cache then "vllm", which the platform
    convention matches on Linux only."""
    import sys as _sys

    monkeypatch.delenv("VLLM_CACHE_ROOT", raising = False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising = False)
    monkeypatch.setattr(_sys, "platform", platform)
    assert cache_inventory._vllm_dirs() == [tmp_path / ".cache" / "vllm"]


def test_the_vllm_cache_honours_xdg_because_vllm_does(tmp_path, monkeypatch, isolated_caches):
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising = False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_inventory._vllm_dirs() == [tmp_path / "xdg" / "vllm"]


# --- what a refusal costs -------------------------------------------------


def test_a_refused_root_is_never_walked(tmp_path, monkeypatch, isolated_caches):
    """Or the Resources tab measures it recursively on every open."""
    outputs = tmp_path / "outputs"
    monkeypatch.setattr(
        cache_inventory, "protected_trees", lambda: {Path(os.path.realpath(outputs))}
    )
    _write(outputs / "triton" / "kernel.cubin", "compiled" * 100)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(outputs / "triton"))

    walked = []
    measure = cache_inventory._measure_root
    monkeypatch.setattr(
        cache_inventory,
        "_measure_root",
        lambda root, **kwargs: (walked.append(root), measure(root, **kwargs))[1],
    )

    entry = describe_cache(definition_for("triton"))
    assert entry["purgeable"] is False
    assert walked == []
    assert entry["size_bytes"] == 0


def test_a_cache_anywhere_under_the_documents_folder_is_refused(
    tmp_path, monkeypatch, isolated_caches
):
    """A protected TREE, not only a protected path: an inherited
    UV_CACHE_DIR=~/Documents/archive would otherwise be reported purgeable."""
    from utils.paths import storage_roots

    documents = tmp_path / "Documents"
    kept = _write(documents / "archive" / "taxes.pdf", "mine")
    monkeypatch.setattr(storage_roots, "documents_root", lambda: documents)
    monkeypatch.setenv("UV_CACHE_DIR", str(documents / "archive"))

    entry = describe_cache(definition_for("uv"))
    assert entry["purgeable"] is False
    assert "protected folder" in (entry["blocked_reason"] or "")
    purge_caches(["uv"])
    assert kept.exists()


def test_a_junction_inside_a_cache_is_not_walked(tmp_path, monkeypatch, isolated_caches):
    """is_dir() sees a directory and is_symlink() no link, so descending sizes the
    junction's target, and one pointing at an ancestor never terminates."""
    root = tmp_path / "uv"
    _write(root / "real.bin", "r" * 10)
    _write(root / "wheels" / "junction" / "elsewhere.bin", "z" * 4096)
    monkeypatch.setattr(cache_inventory, "_is_junction", lambda path: Path(path).name == "junction")

    entry = describe_cache(definition_for("uv"))
    assert entry["size_bytes"] == 10


# --- reporting ------------------------------------------------------------


def test_a_compiled_cache_that_survives_the_clear_says_so(
    tmp_path, monkeypatch, only_the_configured_compiled_cache
):
    """cache_cleanup swallows every failure, so a read-only directory looked
    identical to a clean clear and the UI said so."""
    from utils import cache_cleanup

    compiled = tmp_path / "compiled_cache"
    compiled.mkdir()
    (compiled / cache_cleanup.CACHE_MARKER).touch()
    survivor = _write(compiled / "unsloth_compiled_module_llama.py", "compiled" * 10)
    monkeypatch.setenv("UNSLOTH_COMPILE_LOCATION", str(compiled))
    monkeypatch.setattr(cache_cleanup, "clear_unsloth_compiled_cache", lambda *a, **k: None)

    result = purge_caches(["unsloth_compiled"])["results"][0]
    assert survivor.exists()
    assert result["removed_entries"] == 0
    assert result["errors"]


def test_purging_a_hub_cache_invalidates_the_hugging_face_scans(tmp_path, isolated_caches):
    """Or the picker keeps listing deleted models for the scan's TTL."""
    from hub.utils import inventory_scan

    _write(isolated_caches / "hub" / "models--org--model" / "blob", "m" * 10)
    before = inventory_scan.hf_cache_scans_epoch()
    purge_caches(["hf_hub"])
    assert inventory_scan.hf_cache_scans_epoch() > before

    # ...and a cache that holds no repository does not disturb it.
    steady = inventory_scan.hf_cache_scans_epoch()
    _write(tmp_path / "uv" / "wheel.whl", "w" * 10)
    purge_caches(["uv"])
    assert inventory_scan.hf_cache_scans_epoch() == steady


def test_a_pattern_limited_cache_nested_in_another_root_is_sheltered(
    tmp_path, monkeypatch, isolated_caches
):
    """MPLCONFIGDIR keeps matplotlibrc, which is why its own clear is
    pattern-limited, so an outer root must not take it whole."""
    uv = tmp_path / "uv"
    config = uv / "matplotlib"
    _write(uv / "wheels" / "wheel.whl", "w" * 10)
    kept = _write(config / "matplotlibrc", "backend: Agg")
    fonts = _write(config / "fontlist-v390.json", "[]" * 10)
    monkeypatch.setenv("MPLCONFIGDIR", str(config))

    described = describe_cache(definition_for("uv"))
    assert described["purgeable"] is False
    assert "matplotlib cache" in (described["blocked_reason"] or "")
    result = purge_caches(["uv"])
    assert kept.exists()
    assert result["freed_bytes"] == 0

    # ...and its own clear still takes the cache files and leaves the config.
    purge_caches(["matplotlib"])
    assert kept.exists()
    assert not fonts.exists()


def test_a_cache_home_left_behind_is_protected_but_never_purged(
    tmp_path, monkeypatch, isolated_caches
):
    """An API key may append to that history through an endpoint the purge route
    itself refuses, so only the ACTIVE home is a purge root."""
    from utils import hf_cache_settings

    previous = tmp_path / "old-home"
    theirs = _write(previous / "assets" / "someone-elses.bin", "d" * 10)
    _write(previous / "hub" / "models--org--model" / "blob", "m" * 10)
    mine = _write(isolated_caches / "assets" / "asset.bin", "a" * 10)
    monkeypatch.setattr(
        hf_cache_settings, "known_hf_cache_homes", lambda: [isolated_caches, previous]
    )
    monkeypatch.setattr(
        hf_cache_settings,
        "known_hf_hub_caches",
        lambda: [isolated_caches / "hub", previous / "hub"],
    )
    monkeypatch.delenv("HF_ASSETS_CACHE", raising = False)
    # What initialize_hf_cache_environment leaves behind at startup.
    monkeypatch.setenv("HF_HOME", str(isolated_caches))

    assets = describe_cache(definition_for("hf_assets"))
    assert assets["paths"] == [str(isolated_caches / "assets")]
    hub = describe_cache(definition_for("hf_hub"))
    assert hub["paths"] == [str(isolated_caches / "hub")]

    purge_caches(["hf_assets", "hf_hub"])
    assert not mine.exists()
    assert theirs.exists()

    # ...and a home that was left behind is still refused outright.
    with pytest.raises(CachePurgeRefused):
        assert_purgeable_root(previous)


def test_the_pip_cache_is_the_one_pip_reports(tmp_path, monkeypatch, isolated_caches):
    """pip.conf moves the cache and Studio's pip calls do not pass --isolated."""
    import subprocess as real_subprocess

    from utils import cache_inventory as module

    configured = tmp_path / "corp-pip-cache"
    _write(configured / "wheels" / "cached.whl", "p" * 40)
    monkeypatch.delenv("PIP_CACHE_DIR", raising = False)
    monkeypatch.setattr(module, "_probed_cache_dirs", {})
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: real_subprocess.CompletedProcess(a[0], 0, f"{configured}\n", ""),
    )

    entry = describe_cache(definition_for("pip"))
    assert entry["paths"] == [str(configured)]
    assert entry["size_bytes"] == 40


def test_the_pip_probe_runs_once_and_survives_a_failure(tmp_path, monkeypatch, isolated_caches):
    from utils import cache_inventory as module

    calls = []
    monkeypatch.delenv("PIP_CACHE_DIR", raising = False)
    monkeypatch.setattr(module, "_probed_cache_dirs", {})

    def explode(*args, **kwargs):
        calls.append(args)
        raise OSError("no pip here")

    monkeypatch.setattr(module.subprocess, "run", explode)
    probe = ["python", "-m", "pip", "cache", "dir"]
    assert module._probe_tool_cache_dir("pip", probe) is None
    assert module._probe_tool_cache_dir("pip", probe) is None
    assert len(calls) == 1
    # ...and the platform default still answers, so the row does not vanish.
    assert module._pip_dirs() == [tmp_path / "xdg" / "pip"]


def test_the_child_caches_follow_the_real_home_not_the_displayed_one(
    tmp_path, monkeypatch, isolated_caches
):
    """An explicit HF_HUB_CACHE makes cache_home the hub's PARENT, so the display
    home misses the real assets and datasets and offers somebody else's."""
    from utils import hf_cache_settings

    project = tmp_path / "project"
    hub = project / "hub"
    hub.mkdir(parents = True)
    theirs = _write(project / "assets" / "not-a-cache.bin", "d" * 10)
    real_home = tmp_path / "xdg" / "huggingface"
    mine = _write(real_home / "assets" / "asset.bin", "a" * 10)

    paths = hf_cache_settings.HuggingFaceCachePaths(
        project, hub, real_home / "xet", "environment", "HF_HUB_CACHE"
    )
    monkeypatch.setattr(hf_cache_settings, "get_hf_cache_paths", lambda: paths)
    monkeypatch.setattr(hf_cache_settings, "_EXPLICIT_CACHE_ENV", {"HF_HUB_CACHE": str(hub)})
    monkeypatch.delenv("HF_ASSETS_CACHE", raising = False)
    monkeypatch.setenv("HF_HOME", str(real_home))

    entry = describe_cache(definition_for("hf_assets"))
    assert entry["paths"] == [str(real_home / "assets")]
    purge_caches(["hf_assets"])
    assert theirs.exists()
    assert not mine.exists()


def test_a_measurement_that_began_before_a_purge_is_not_remembered(
    tmp_path, monkeypatch, isolated_caches
):
    """Or it offers space already gone for the rest of the memo window."""
    root = tmp_path / "uv"
    _write(root / "wheel.whl", "w" * 100)
    definition = definition_for("uv")
    real_describe = cache_inventory.describe_cache
    before_the_purge = real_describe(definition)
    assert before_the_purge["size_bytes"] == 100

    def measure_then_purge(_definition):
        # The purge lands while this walk is still running.
        for child in root.iterdir():
            child.unlink()
        cache_inventory.invalidate_cache_size("uv")
        return before_the_purge

    monkeypatch.setattr(cache_inventory, "describe_cache", measure_then_purge)
    assert cache_inventory._described(definition, refresh = True)["size_bytes"] == 100
    monkeypatch.setattr(cache_inventory, "describe_cache", real_describe)
    assert cache_inventory._described(definition, refresh = False)["size_bytes"] == 0


def test_the_pip_probe_asks_the_child_for_utf8(tmp_path, monkeypatch, isolated_caches):
    """A redirected child takes the locale encoding, so a non-ASCII cache path
    comes back mangled and resolves to nothing."""
    import subprocess as real_subprocess

    from utils import cache_inventory as module

    seen = {}
    configured = tmp_path / "caché-pip"
    _write(configured / "wheels" / "cached.whl", "p" * 40)
    monkeypatch.delenv("PIP_CACHE_DIR", raising = False)
    monkeypatch.setattr(module, "_probed_cache_dirs", {})

    def record(*args, **kwargs):
        seen.update(kwargs)
        return real_subprocess.CompletedProcess(args[0], 0, f"{configured}\n", "")

    monkeypatch.setattr(module.subprocess, "run", record)
    entry = describe_cache(definition_for("pip"))

    assert entry["paths"] == [str(configured)]
    assert seen["encoding"] == "utf-8"
    assert seen["env"]["PYTHONIOENCODING"] == "utf-8"


def test_a_scoped_dataset_fallback_override_is_not_a_purge_root(
    tmp_path, monkeypatch, isolated_caches
):
    """cache_safe points HF_DATASETS_CACHE there mid-load, in this process, while
    load_dataset writes Arrow files and lock state into it."""
    from utils.paths import storage_roots

    studio_cache = tmp_path / "studio-cache"
    fallback = studio_cache / "hf-datasets"
    in_flight = _write(fallback / "squad" / "data.arrow", "d" * 10)
    stable = _write(isolated_caches / "datasets" / "cached.arrow", "s" * 10)
    monkeypatch.setattr(storage_roots, "cache_root", lambda: studio_cache)
    monkeypatch.setenv("HF_HOME", str(isolated_caches))
    # The override is live, exactly as it is while the fallback load runs.
    monkeypatch.setenv("HF_DATASETS_CACHE", str(fallback))

    entry = describe_cache(definition_for("hf_datasets"))
    assert entry["paths"] == [str(isolated_caches / "datasets")]
    purge_caches(["hf_datasets"])
    assert in_flight.exists()
    assert not stable.exists()


@pytest.mark.parametrize(
    "platform,name,expected,patterned",
    [
        ("linux", "posix", ".cache/matplotlib", False),
        ("darwin", "posix", ".matplotlib", True),
        ("win32", "nt", ".matplotlib", True),
    ],
)
def test_the_matplotlib_cache_is_where_matplotlib_puts_it(
    tmp_path, monkeypatch, isolated_caches, platform, name, expected, patterned
):
    """get_cachedir takes the XDG branch for linux and freebsd only; elsewhere it
    is ~/.matplotlib, which is get_configdir() too, so only cache entries go."""
    import sys as _sys

    from utils import cache_inventory as module

    monkeypatch.delenv("MPLCONFIGDIR", raising = False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising = False)
    monkeypatch.setattr(_sys, "platform", platform)
    monkeypatch.setattr(module, "_is_windows", lambda: name == "nt")
    (tmp_path / ".matplotlib").mkdir(exist_ok = True)

    assert module._matplotlib_dirs() == [tmp_path / expected]
    assert (module._matplotlib_patterns() is not None) is patterned


def test_the_cuda_cache_is_roaming_appdata_on_windows(tmp_path, monkeypatch, isolated_caches):
    """The CUDA guide's default is %APPDATA%, which is Roaming, not Local."""
    import sys as _sys

    from utils import cache_inventory as module

    monkeypatch.delenv("CUDA_CACHE_PATH", raising = False)
    monkeypatch.setattr(module, "_is_windows", lambda: True)
    monkeypatch.setenv("APPDATA", str(tmp_path / "AppData" / "Roaming"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    assert module._cuda_dirs() == [tmp_path / "AppData" / "Roaming" / "NVIDIA" / "ComputeCache"]

    monkeypatch.setattr(module, "_is_windows", lambda: False)
    monkeypatch.setattr(_sys, "platform", "darwin")
    assert module._cuda_dirs() == [
        tmp_path / "Library" / "Application Support" / "NVIDIA" / "ComputeCache"
    ]


def test_numbas_user_wide_fallback_cache_is_reported(tmp_path, monkeypatch, isolated_caches):
    """__pycache__ next to the source is not ours, but UserWideCacheLocator's
    AppDirs("numba", appauthor = False).user_cache_dir is."""
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising = False)
    root = tmp_path / "xdg" / "numba"
    _write(root / "somemodule.nbi", "n" * 30)

    entry = describe_cache(definition_for("numba"))
    assert entry["paths"] == [str(root)]
    assert entry["size_bytes"] == 30


def test_the_studio_executables_directory_is_refused(tmp_path, monkeypatch, isolated_caches):
    """Descendants of the studio home are deliberately allowed, since the caches
    live there, so the shim and the managed executables need naming."""
    from utils.paths import storage_roots

    studio = tmp_path / "studio"
    binaries = studio / "bin"
    shim = _write(binaries / "unsloth", "#!/bin/sh")
    monkeypatch.setattr(storage_roots, "studio_bin_root", lambda: binaries)
    monkeypatch.setenv("UV_CACHE_DIR", str(binaries))

    entry = describe_cache(definition_for("uv"))
    assert entry["purgeable"] is False
    purge_caches(["uv"])
    assert shim.exists()

    # ...and nothing below it either.
    monkeypatch.setenv("UV_CACHE_DIR", str(binaries / "vendor"))
    (binaries / "vendor").mkdir()
    with pytest.raises(CachePurgeRefused):
        assert_purgeable_root(binaries / "vendor")


def test_the_npm_cache_follows_npmrc(tmp_path, monkeypatch, isolated_caches):
    """No environment variable carries it, and the backend launches npm for MCP."""
    import shutil as real_shutil
    import subprocess as real_subprocess

    from utils import cache_inventory as module

    configured = tmp_path / "corp-npm"
    _write(configured / "_cacache" / "index-v5" / "entry", "n" * 25)
    for key in ("npm_config_cache", "NPM_CONFIG_CACHE"):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setattr(module, "_probed_cache_dirs", {})
    monkeypatch.setattr(real_shutil, "which", lambda name, **kw: "/usr/bin/npm")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: real_subprocess.CompletedProcess(a[0], 0, f"{configured}\n", ""),
    )

    entry = describe_cache(definition_for("npm"))
    assert entry["paths"] == [str(configured / "_cacache")]
    assert entry["size_bytes"] == 25


def test_a_probe_that_answers_with_a_relative_path_is_ignored(monkeypatch, isolated_caches):
    """npm prints "undefined" rather than failing when it has no answer."""
    import subprocess as real_subprocess

    from utils import cache_inventory as module

    monkeypatch.setattr(module, "_probed_cache_dirs", {})
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **k: real_subprocess.CompletedProcess(a[0], 0, "undefined\n", ""),
    )
    assert module._probe_tool_cache_dir("npm", ["npm", "config", "get", "cache"]) is None


def test_the_recorded_install_uv_cache_is_reported(tmp_path, monkeypatch, isolated_caches):
    """The seeded UV_CACHE_DIR is not always the one updates fill: an installer
    that used a warm cache elsewhere records it, and the CLI obeys the record."""
    from utils.paths import storage_roots

    studio_cache = tmp_path / "studio-cache"
    seeded = studio_cache / "uv"
    warm = tmp_path / "warm-uv"
    _write(seeded / "seeded.whl", "s" * 10)
    _write(warm / "archive-v0" / "big.whl", "w" * 900)
    monkeypatch.setattr(storage_roots, "cache_root", lambda: studio_cache)
    (studio_cache / "uv-cache-dir").write_text(f"{warm}\n", encoding = "utf-8")
    monkeypatch.setenv("UV_CACHE_DIR", str(seeded))

    entry = describe_cache(definition_for("uv"))
    assert entry["paths"] == [str(seeded), str(warm)]
    assert entry["size_bytes"] == 910
    purge_caches(["uv"])
    assert not (warm / "archive-v0").exists()
    assert seeded.is_dir()


def test_a_blank_or_missing_uv_marker_adds_nothing(tmp_path, monkeypatch, isolated_caches):
    from utils import cache_inventory as module
    from utils.paths import storage_roots

    studio_cache = tmp_path / "studio-cache"
    studio_cache.mkdir()
    monkeypatch.setattr(storage_roots, "cache_root", lambda: studio_cache)
    assert module._recorded_uv_cache() is None
    (studio_cache / "uv-cache-dir").write_text("   \n", encoding = "utf-8")
    assert module._recorded_uv_cache() is None


def test_the_studio_temporary_workspace_is_refused(tmp_path, monkeypatch, isolated_caches):
    """tmp_root() is not a cache: decoding and training read files back from it."""
    from utils.paths import storage_roots

    workspace = tmp_path / "unsloth-studio"
    in_use = _write(workspace / "decode" / "clip.wav", "a" * 10)
    monkeypatch.setattr(storage_roots, "tmp_root", lambda: workspace)
    monkeypatch.setenv("UV_CACHE_DIR", str(workspace / "decode"))

    entry = describe_cache(definition_for("uv"))
    assert entry["purgeable"] is False
    purge_caches(["uv"])
    assert in_use.exists()
    with pytest.raises(CachePurgeRefused):
        assert_purgeable_root(workspace)


def test_two_cold_probes_do_not_race_into_the_fallback(tmp_path, monkeypatch, isolated_caches):
    """Recording the miss before the probe finished had the second cold request
    read it as a finished failure and show the fallback path."""
    import subprocess as real_subprocess
    import threading

    from utils import cache_inventory as module

    configured = tmp_path / "corp-pip"
    configured.mkdir()
    monkeypatch.delenv("PIP_CACHE_DIR", raising = False)
    monkeypatch.setattr(module, "_probed_cache_dirs", {})
    started = threading.Event()
    release = threading.Event()

    def slow(*args, **kwargs):
        started.set()
        release.wait(5)
        return real_subprocess.CompletedProcess(args[0], 0, f"{configured}\n", "")

    monkeypatch.setattr(module.subprocess, "run", slow)
    answers: list = []
    first = threading.Thread(target = lambda: answers.append(module._pip_dirs()))
    first.start()
    assert started.wait(5)
    second = threading.Thread(target = lambda: answers.append(module._pip_dirs()))
    second.start()
    release.set()
    first.join(10)
    second.join(10)

    assert answers == [[configured], [configured]]


def test_the_inductor_row_ignores_the_diffusion_override(tmp_path, monkeypatch, isolated_caches):
    """begin() repoints TORCHINDUCTOR_CACHE_DIR while a model is resident and
    restore() puts it back, so a row that followed it and the clear disagree."""
    import getpass
    import tempfile

    from utils import cache_inventory as module

    diffusion = tmp_path / "diffusion_compile_cache"
    per_model = diffusion / "abc123" / "inductor"
    transient = _write(per_model / "graph.bin", "d" * 50)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))
    stable = tmp_path / f"torchinductor_{getpass.getuser()}"
    _write(stable / "fx.bin", "s" * 10)
    monkeypatch.setattr(
        module, "_diffusion_compile_root", lambda: Path(os.path.realpath(diffusion))
    )
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(per_model))

    entry = describe_cache(definition_for("torch_inductor"))
    assert entry["paths"] == [str(stable)]
    purge_caches(["torch_inductor"])
    assert transient.exists()

    # ...and an ordinary override is still followed.
    elsewhere = tmp_path / "my-inductor"
    _write(elsewhere / "fx.bin", "e" * 10)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(elsewhere))
    assert describe_cache(definition_for("torch_inductor"))["paths"] == [str(elsewhere)]


def test_a_configured_token_path_is_protected(tmp_path, monkeypatch, isolated_caches):
    """HF_TOKEN_PATH is read instead of <home>/token and can name a file inside a
    clearable cache."""
    assets = tmp_path / "mnt-hf"
    token = _write(assets / "token", "hf_secret")
    _write(assets / "blob", "a" * 10)
    monkeypatch.setenv("HF_ASSETS_CACHE", str(assets))
    monkeypatch.setenv("HF_TOKEN_PATH", str(token))

    entry = describe_cache(definition_for("hf_assets"))
    assert entry["purgeable"] is False
    purge_caches(["hf_assets"])
    assert token.read_text(encoding = "utf-8") == "hf_secret"


def test_a_purge_waits_for_the_downloads_writing_into_the_cache(
    tmp_path, monkeypatch, isolated_caches
):
    """A download writes blobs, locks and partials straight into the hub cache,
    which is why the per-repository deletes refuse for this reason too."""
    from utils import cache_inventory as module

    blob = _write(isolated_caches / "hub" / "models--org--model" / "blob", "m" * 10)
    monkeypatch.setattr(
        module,
        "_reserve_downloads",
        lambda key: ([], "Cancel it.") if key == "hf_hub" else ([], None),
    )
    result = purge_caches(["hf_hub"])["results"][0]
    assert blob.exists()
    assert result["freed_bytes"] == 0
    assert result["errors"] == ["Cancel it."]


def test_the_hub_cache_is_reserved_in_both_registries(monkeypatch):
    """A dataset download snapshot_downloads its datasets-- entries into the same
    hub root, so a job in either registry has to hold the purge off."""
    from hub.utils import download_registry
    from utils import cache_inventory as module

    class _Registry:
        def __init__(self, free = True):
            self.free = free
            self.held = 0

        def begin_cache_purge(self):
            if not self.free:
                return False
            self.held += 1
            return True

        def end_cache_purge(self):
            self.held -= 1

    models, datasets = _Registry(), _Registry()
    monkeypatch.setattr(download_registry, "get_models_registry", lambda: models)
    monkeypatch.setattr(download_registry, "get_datasets_registry", lambda: datasets)

    reserved, busy = module._reserve_downloads("hf_hub")
    assert busy is None
    assert (models.held, datasets.held) == (1, 1)
    module._release_downloads(reserved)
    assert (models.held, datasets.held) == (0, 0)

    # A dataset job alone still holds off the hub clear...
    datasets.free = False
    reserved, busy = module._reserve_downloads("hf_hub")
    assert busy is not None
    # ...and the models reservation it had already taken is handed back.
    assert (models.held, datasets.held) == (0, 0)

    # A cache no download writes into is not gated on one.
    assert module._reserve_downloads("uv") == ([], None)


def test_one_blocked_root_does_not_put_the_others_out_of_reach(
    tmp_path, monkeypatch, isolated_caches
):
    """purge_cache skips a refused root and carries on, so the row has to say so."""
    from utils.paths import storage_roots

    studio_cache = tmp_path / "studio-cache"
    good = studio_cache / "uv"
    documents = tmp_path / "Documents"
    blocked = documents / "uv-cache"
    wheel = _write(good / "wheel.whl", "w" * 40)
    theirs = _write(blocked / "taxes.pdf", "mine")
    monkeypatch.setattr(storage_roots, "cache_root", lambda: studio_cache)
    monkeypatch.setattr(storage_roots, "documents_root", lambda: documents)
    (studio_cache / "uv-cache-dir").write_text(f"{blocked}\n", encoding = "utf-8")
    monkeypatch.setenv("UV_CACHE_DIR", str(good))

    entry = describe_cache(definition_for("uv"))
    assert entry["purgeable"] is True
    assert "protected folder" in (entry["blocked_reason"] or "")
    assert entry["size_bytes"] == 40
    purge_caches(["uv"])
    assert not wheel.exists()
    assert theirs.exists()


def test_two_cold_reads_of_one_cache_walk_it_once(tmp_path, monkeypatch, isolated_caches):
    """Simultaneous misses would each pay a cold walk for the same answer."""
    import threading

    from utils import cache_inventory as module

    _write(tmp_path / "uv" / "wheel.whl", "w" * 10)
    definition = definition_for("uv")
    walks: list = []
    started = threading.Event()
    release = threading.Event()
    real_describe = cache_inventory.describe_cache

    def slow(target):
        walks.append(target.key)
        started.set()
        release.wait(5)
        return real_describe(target)

    monkeypatch.setattr(cache_inventory, "describe_cache", slow)
    answers: list = []
    first = threading.Thread(
        target = lambda: answers.append(module._described(definition, refresh = True))
    )
    first.start()
    assert started.wait(5)
    second = threading.Thread(
        target = lambda: answers.append(module._described(definition, refresh = True))
    )
    second.start()
    release.set()
    first.join(10)
    second.join(10)

    assert walks == ["uv"]
    assert [answer["size_bytes"] for answer in answers] == [10, 10]


def test_the_child_caches_ignore_a_studio_selected_models_folder(
    tmp_path, monkeypatch, isolated_caches
):
    """It moves the hub and xet caches, not HF_HOME, so the token, assets and
    datasets stay at the platform default."""
    from utils import hf_cache_settings

    chosen = tmp_path / "MyModels"
    (chosen / "hub").mkdir(parents = True)
    theirs = _write(chosen / "assets" / "not-a-cache.bin", "d" * 10)
    real_home = tmp_path / "default-hf"
    mine = _write(real_home / "assets" / "asset.bin", "a" * 10)

    paths = hf_cache_settings.HuggingFaceCachePaths(
        chosen, chosen / "hub", real_home / "xet", "studio"
    )
    monkeypatch.setattr(hf_cache_settings, "get_hf_cache_paths", lambda: paths)
    monkeypatch.delenv("HF_ASSETS_CACHE", raising = False)
    monkeypatch.setenv("HF_HOME", str(real_home))

    entry = describe_cache(definition_for("hf_assets"))
    assert entry["paths"] == [str(real_home / "assets")]
    purge_caches(["hf_assets"])
    assert theirs.exists()
    assert not mine.exists()


def test_the_npx_cache_is_cleared_with_the_package_cache(tmp_path, monkeypatch, isolated_caches):
    """<cache>/_npx is npm's cache as much as _cacache is; the logs are not."""
    npm = tmp_path / "npm"
    package = _write(npm / "_cacache" / "index-v5" / "entry", "c" * 20)
    executable = _write(npm / "_npx" / "abc123" / "node_modules" / "server.js", "x" * 30)
    logs = _write(npm / "_logs" / "debug.log", "l" * 5)
    monkeypatch.setenv("npm_config_cache", str(npm))

    entry = describe_cache(definition_for("npm"))
    assert entry["paths"] == [str(npm / "_cacache"), str(npm / "_npx")]
    assert entry["size_bytes"] == 50
    purge_caches(["npm"])
    assert not package.exists()
    assert not executable.exists()
    assert logs.exists()


def test_a_purge_holds_the_registry_against_a_download_claimed_after_the_check(tmp_path):
    """The reservation closes the check-then-delete race, which is why
    begin_delete exists for the per-repository path."""
    from hub.utils.download_registry import DownloadRegistry

    registry = DownloadRegistry()
    assert registry.begin_cache_purge() is True
    claimed, state = registry.claim_repository_owner("org/model", object())
    assert claimed is False
    assert state == "deleting"

    # Counted, so two caches sharing this registry nest rather than releasing early.
    assert registry.begin_cache_purge() is True
    registry.end_cache_purge()
    assert registry.claim_repository_owner("org/model", object())[0] is False
    registry.end_cache_purge()
    assert registry.claim_repository_owner("org/model", object())[0] is True

    # ...and a purge is refused while that owner holds the repository.
    assert registry.begin_cache_purge() is False


def test_the_xet_cache_is_reserved_like_the_hub(monkeypatch):
    """Both download services pass the active xet cache into their claim, so a
    worker is fetching chunks into it while a clear would be emptying it."""
    from hub.utils import download_registry
    from utils import cache_inventory as module

    class _Registry:
        def __init__(self):
            self.held = 0

        def begin_cache_purge(self):
            self.held += 1
            return True

        def end_cache_purge(self):
            self.held -= 1

    models, datasets = _Registry(), _Registry()
    monkeypatch.setattr(download_registry, "get_models_registry", lambda: models)
    monkeypatch.setattr(download_registry, "get_datasets_registry", lambda: datasets)

    reserved, busy = module._reserve_downloads("hf_xet")
    assert busy is None
    assert (models.held, datasets.held) == (1, 1)
    module._release_downloads(reserved)


def test_a_repository_delete_and_a_whole_cache_purge_exclude_each_other():
    """Both are removing files from the same root, in either order."""
    from hub.utils.download_registry import DownloadRegistry

    registry = DownloadRegistry()
    assert registry.begin_delete("org/model") is True
    assert registry.begin_cache_purge() is False
    registry.end_delete("org/model")

    assert registry.begin_cache_purge() is True
    assert registry.begin_delete("org/model") is False
    registry.end_cache_purge()
    assert registry.begin_delete("org/model") is True
