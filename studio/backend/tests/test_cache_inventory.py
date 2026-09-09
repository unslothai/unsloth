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
    """A bulk purge may cost a recompile or a re-download. It may not cost a
    running job.

    Every other cache here is rebuilt by work the user has not started yet. The
    unsloth compiled cache is different: a training or inference worker imports
    generated modules from it lazily, long after the job began, and
    compiled_cache_lock only serialises against a sibling backend, never against
    a spawned worker that holds no lock. So clearing it mid-run forfeits hours of
    compute, and a button labelled "free up space" must not be able to do that
    without the user choosing it specifically.
    """
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
    """A junction is the same hazard as a symlink and fails the same test.

    Since 3.8 only IO_REPARSE_TAG_SYMLINK sets S_IFLNK, so ``is_symlink()`` is
    False for a Windows directory junction while ``realpath()`` follows it to
    its target. Without a tag check, UV_CACHE_DIR pointed at a junction would
    empty whatever the junction names.
    """
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
    """An opt-in cache is only ever cleared when it is asked for by name.

    Nothing stops a variable from putting one inside another cache, and emptying
    the outer root would then delete it anyway: HF_DATASETS_CACHE below
    UV_CACHE_DIR costs the re-download the opt-in exists to prevent.
    """
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
    """HF_HUB_CACHE=/mnt/hf-cache makes the DISPLAY home the hub itself.

    Protecting the display home there would mark the model cache permanently
    unclearable and say so in the row, while protecting no credential: the token
    lives in the HF home, which is protected on its own.
    """
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
    """torch derives it from getpass.getuser(), which a bare uid does not match.

    getpass reads LOGNAME/USER/LNAME/USERNAME and then the pwd account name, so
    a container with none of USER/USERNAME set puts the cache at
    torchinductor_root while a uid fallback looks at torchinductor_0.
    """
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
    """vllm/envs.py resolves XDG_CACHE_HOME or ~/.cache, then "vllm", everywhere.

    The platform cache convention agrees on Linux only: it would look under
    ~/Library/Caches on macOS and LOCALAPPDATA on Windows, where vLLM does not.
    """
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
    """The gate runs before the size walk, not after it.

    A root the gate will refuse (UV_CACHE_DIR=/ or the home directory) would
    otherwise be measured recursively on every open of the Resources tab, before
    the refusal is ever reached.
    """
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
    """Documents is a protected TREE, not only a protected path.

    Every other user-data root is in both lists; without this one, an inherited
    UV_CACHE_DIR=~/Documents/archive is reported purgeable and clearing it takes
    whatever is in that folder.
    """
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
    """A junction is a directory to is_dir() and not a link to is_symlink().

    Descending into one sizes its target instead of the cache, so the row
    advertises bytes a clear will not free, and a junction back to an ancestor
    never terminates.
    """
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
    """cache_cleanup swallows every unlink and rmtree failure.

    A read-only directory or a locked file therefore looks identical to a clean
    clear from here, and the UI showed a success toast over a cache still on
    disk.
    """
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
    """The Hub inventory is invalidated on every app-driven cache mutation.

    Emptying hf_hub or hf_datasets removes the repositories that scan reports,
    so without this the Hub and the model picker keep listing deleted models for
    the scan's TTL.
    """
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
    """MPLCONFIGDIR keeps matplotlibrc, so a broader clear must not take it.

    Its own clear is pattern-limited for exactly that reason. Nothing stops a
    variable from putting it inside another cache, and emptying the outer root
    would delete the configuration the narrower clear was written to keep.
    """
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
    """Only the ACTIVE Hugging Face home is a purge root.

    The history the other homes come from is appended to by an endpoint an API
    key may call, while the purge endpoint refuses one, so resolving purge roots
    through it would let a caller that cannot delete pick what a later clear
    deletes.
    """
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
    """pip.conf can move the cache and Studio's fallback pip calls honour it.

    They do not pass --isolated, so the platform default is the wrong answer
    whenever cache-dir is configured. pip is asked for its own.
    """
    import subprocess as real_subprocess

    from utils import cache_inventory as module

    configured = tmp_path / "corp-pip-cache"
    _write(configured / "wheels" / "cached.whl", "p" * 40)
    monkeypatch.delenv("PIP_CACHE_DIR", raising = False)
    monkeypatch.setattr(module, "_pip_configured", module._UNPROBED)
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
    monkeypatch.setattr(module, "_pip_configured", module._UNPROBED)

    def explode(*args, **kwargs):
        calls.append(args)
        raise OSError("no pip here")

    monkeypatch.setattr(module.subprocess, "run", explode)
    assert module._pip_configured_dir() is None
    assert module._pip_configured_dir() is None
    assert len(calls) == 1
    # ...and the platform default still answers, so the row does not vanish.
    assert module._pip_dirs() == [tmp_path / "xdg" / "pip"]
