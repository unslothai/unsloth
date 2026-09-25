# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the pre-warmed torch.compile cache (``diffusion_compile_cache.py``).

The Mega-cache API (``torch.compiler.save_cache_artifacts`` / ``load_cache_artifacts``)
is monkeypatched with deterministic in-memory fakes so the fingerprint / exact-match /
integrity / fallback / lifecycle logic is exercised without a real compile. The
fingerprint helpers run against the real torch on this box.
"""

from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import pytest

from core.inference import diffusion_compile_cache as cc


def _transformer(blocks = ("FluxTransformerBlock", "FluxSingleTransformerBlock")):
    return types.SimpleNamespace(_repeated_blocks = list(blocks))


_BEGIN_KW = dict(
    family = "flux.1",
    dtype = "torch.bfloat16",
    quant = None,
    attention_backend = "_native_cudnn",
    compile_kwargs = {"fullgraph": True, "dynamic": True},
    shape_bucket = "1024x1024",
)


# --------------------------------------------------------------------------- fingerprint
def test_environment_fingerprint_has_hard_dimensions():
    fp = cc.environment_fingerprint()
    for k in ("torch", "torch_cuda", "triton", "diffusers", "gpu_name", "gpu_capability"):
        assert k in fp


def test_cache_key_stable_across_kwarg_order():
    efp = cc.environment_fingerprint()
    t = _transformer()
    a = cc.model_fingerprint(
        family = "flux.1",
        transformer = t,
        dtype = "bf16",
        quant = None,
        attention_backend = "x",
        compile_kwargs = {"fullgraph": True, "dynamic": True},
    )
    b = cc.model_fingerprint(
        family = "flux.1",
        transformer = t,
        dtype = "bf16",
        quant = None,
        attention_backend = "x",
        compile_kwargs = {"dynamic": True, "fullgraph": True},
    )
    assert cc.cache_key(efp, a) == cc.cache_key(efp, b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("family", "qwen-image"),
        ("dtype", "torch.float16"),
        ("quant", "int8"),
        ("attention_backend", "native"),
        ("shape_bucket", "512x512"),
    ],
)
def test_cache_key_sensitive_to_model_dims(field, value):
    efp = cc.environment_fingerprint()
    t = _transformer()
    base = dict(
        family = "flux.1",
        transformer = t,
        dtype = "bf16",
        quant = None,
        attention_backend = "x",
        compile_kwargs = {"fullgraph": True},
        shape_bucket = "1024x1024",
    )
    k0 = cc.cache_key(efp, cc.model_fingerprint(**base))
    base[field] = value
    assert cc.cache_key(efp, cc.model_fingerprint(**base)) != k0


def test_repeated_blocks_change_key():
    efp = cc.environment_fingerprint()
    k1 = cc.cache_key(
        efp,
        cc.model_fingerprint(
            family = "f",
            transformer = _transformer(("A",)),
            dtype = "bf16",
            quant = None,
            attention_backend = "x",
            compile_kwargs = {},
        ),
    )
    k2 = cc.cache_key(
        efp,
        cc.model_fingerprint(
            family = "f",
            transformer = _transformer(("B",)),
            dtype = "bf16",
            quant = None,
            attention_backend = "x",
            compile_kwargs = {},
        ),
    )
    assert k1 != k2


# ----------------------------------------------------------------------------- env knobs
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0", "off"),
        ("off", "off"),
        ("1", "on"),
        ("on", "on"),
        ("auto", "auto"),
        ("", "auto"),
        ("garbage", "auto"),
    ],
)
def test_cache_mode(monkeypatch, raw, expected):
    monkeypatch.setenv(cc._ENV_MODE, raw)
    assert cc.cache_mode() == expected


def test_cache_mode_default_auto(monkeypatch):
    monkeypatch.delenv(cc._ENV_MODE, raising = False)
    assert cc.cache_mode() == "auto"


# ------------------------------------------------------------------------------ disabled
def test_begin_returns_none_when_disabled(monkeypatch):
    monkeypatch.setenv(cc._ENV_MODE, "0")
    assert cc.begin(transformer = _transformer(), **_BEGIN_KW) is None


def test_begin_returns_none_without_megacache_api(monkeypatch):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    fake_torch = types.ModuleType("torch")
    fake_torch.compiler = types.SimpleNamespace()  # no save/load attrs
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    assert cc.begin(transformer = _transformer(), **_BEGIN_KW) is None


# ----------------------------------------------------------------- megacache fake + flow
@pytest.fixture
def fake_megacache(monkeypatch):
    """Patch torch.compiler save/load with deterministic in-memory behaviour."""
    import torch

    state = {"saved": None, "loaded_with": None}

    def fake_save():
        return (b"ARTIFACT-BYTES", None)

    def fake_load(data: bytes):
        state["loaded_with"] = data
        return object() if data == b"ARTIFACT-BYTES" else None

    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", fake_save, raising = False)
    monkeypatch.setattr(torch.compiler, "load_cache_artifacts", fake_load, raising = False)
    return state


def test_save_then_load_roundtrip(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "on")  # load + save
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))

    # First load: cold (no bundle yet).
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx is not None and ctx.hit is False
    assert cc.save(ctx) is True
    assert ctx.bundle.exists() and ctx.manifest_path.exists()

    # Second load with the SAME fingerprint: warm hit.
    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx2 is not None and ctx2.hit is True
    assert fake_megacache["loaded_with"] == b"ARTIFACT-BYTES"
    assert ctx2.key == ctx.key


def test_auto_mode_saves_by_default(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SAVE, raising = False)
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert cc.save(ctx) is True  # first-run warm: auto saves the bundle
    assert ctx.bundle.exists() and ctx.manifest_path.exists()

    # The next load with the same fingerprint hits the just-saved bundle...
    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx2.hit is True
    # ...and does NOT rewrite it under auto (the artifacts on disk are the ones loaded).
    before = ctx2.bundle.stat().st_mtime_ns
    assert cc.save(ctx2) is False
    assert ctx2.bundle.stat().st_mtime_ns == before


def test_save_env_zero_disables_auto_save(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.setenv(cc._ENV_SAVE, "0")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert cc.save(ctx) is False  # explicit load-only override
    assert not ctx.bundle.exists()


def test_on_mode_resaves_after_hit(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "on")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert cc.save(ctx) is True
    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx2.hit is True
    # Distributor mode refreshes the bundle even on a hit (new variants get captured).
    assert cc.save(ctx2) is True


def test_new_static_shape_redirties_a_hit(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SAVE, raising = False)
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))

    # Cold session at 1024: the save records the shape coverage in the manifest.
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.register_shape(ctx, (1024, 1024, 1), static = True)
    assert cc.save(ctx) is True
    manifest = json.loads(ctx.manifest_path.read_text())
    assert manifest["shapes"] == [[1024, 1024, 1]]

    # Warm session: the covered shape does not dirty the context...
    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx2.hit is True and ctx2.saved is True
    assert ctx2.shapes == {(1024, 1024, 1)}
    cc.register_shape(ctx2, (1024, 1024, 1), static = True)
    assert cc.save(ctx2) is False
    # ...but a NEW static shape (its compile just produced new artifacts) does, and the rewritten manifest covers both.
    cc.register_shape(ctx2, (768, 768, 1), static = True)
    assert ctx2.saved is False
    assert cc.save(ctx2) is True
    manifest = json.loads(ctx2.manifest_path.read_text())
    assert manifest["shapes"] == [[768, 768, 1], [1024, 1024, 1]]


def test_new_batch_size_is_its_own_static_shape(monkeypatch, tmp_path, fake_megacache):
    # A static compile produces one artifact PER (w, h, batch): an unseen batch size (incl. an OOM-backoff half) must re-dirty it, a covered one must not.
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SAVE, raising = False)
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.register_shape(ctx, (1024, 1024, 8), static = True)
    assert cc.save(ctx) is True

    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx2.hit is True
    cc.register_shape(ctx2, (1024, 1024, 8), static = True)
    assert cc.save(ctx2) is False  # covered batch: nothing new
    cc.register_shape(ctx2, (1024, 1024, 32), static = True)
    assert ctx2.saved is False  # new batch size: new artifacts to persist
    assert cc.save(ctx2) is True
    manifest = json.loads(ctx2.manifest_path.read_text())
    assert manifest["shapes"] == [[1024, 1024, 8], [1024, 1024, 32]]


def test_gguf_quant_keys_apart_from_dense():
    # A GGUF transformer compiles a different graph (the dequant chain), so the load path fingerprints it quant="gguf" and bundles never cross-hit.
    efp = cc.environment_fingerprint()
    base = dict(
        family = "flux.1",
        transformer = _transformer(),
        dtype = "torch.bfloat16",
        quant = None,
        attention_backend = "x",
        compile_kwargs = {"fullgraph": True, "dynamic": True},
    )
    dense = cc.model_fingerprint(**base)
    gguf = cc.model_fingerprint(**{**base, "quant": "gguf"})
    assert cc.cache_key(efp, dense) != cc.cache_key(efp, gguf)


def test_dynamic_compile_never_dirties(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.save(ctx)
    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx2.hit is True
    # A dynamic-shape compile reuses one artifact across shapes: no re-save.
    cc.register_shape(ctx2, (768, 768, 1), static = False)
    assert cc.save(ctx2) is False
    cc.register_shape(None, (768, 768, 1), static = True)  # no context: no-op


def test_fingerprint_mismatch_falls_back(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "on")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.save(ctx)

    # Tamper the manifest's env fingerprint: the exact-match guard must reject the bundle.
    manifest = json.loads(ctx.manifest_path.read_text())
    manifest["env"]["torch"] = "0.0.0-other"
    ctx.manifest_path.write_text(json.dumps(manifest))

    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx2.hit is False  # mismatch -> local compile, non-fatal


def test_corrupt_bundle_rejected(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "on")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.save(ctx)
    ctx.bundle.write_bytes(b"CORRUPTED")  # manifest sha256 no longer matches

    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx2.hit is False


# ------------------------------------------------------------------------------- restore
def test_restore_inductor_dir(monkeypatch, tmp_path, fake_megacache):
    import os

    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "/tmp/prior-inductor")
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] != "/tmp/prior-inductor"  # redirected
    cc.restore(ctx)
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == "/tmp/prior-inductor"  # restored


def test_the_import_fallback_matches_the_resolver(monkeypatch):
    """The fallback runs when a CLI entry point reaches this module without studio/backend on
    sys.path. Narrowed to whitespace it made the same path refused or accepted depending only on
    sys.path, which is routing that depends on import order. Swept so the copy cannot drift."""
    import builtins
    import string

    from utils.paths.storage_roots import toolchain_path_unparseable

    real_import = builtins.__import__

    def no_storage_roots(name, *args, **kwargs):
        if name == "utils.paths.storage_roots":
            raise ImportError("simulated: studio/backend is not on sys.path")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_storage_roots)

    disagreed = []
    for char in list(string.printable) + [" ", "é"]:
        path = f"/srv/unsloth{char}root/cache/diffusion_compile_cache/key/inductor"
        if cc._toolchain_path_unparseable(path) != toolchain_path_unparseable(path):
            disagreed.append(char)

    assert disagreed == [], "the fallback and the resolver disagree on: " + ", ".join(
        repr(c) for c in disagreed
    )


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("my studio", id = "a space"),
        pytest.param("o'brien", id = "an apostrophe"),
    ],
)
def test_an_unparseable_cache_root_never_pins_the_unparseable_path(
    name, monkeypatch, tmp_path, fake_megacache
):
    """Startup declines to pin TORCHINDUCTOR_CACHE_DIR into a root the C++ builders cannot
    parse, and this assignment used to overwrite that decision on the first compiled diffusion
    run. Checking the environment just after launch would not have caught it.

    What replaces it has to stay PER KEY. Simply keeping whatever startup left would keep the
    one process-wide fallback, and save_cache_artifacts then serialises that shared cache into
    every fingerprinted bundle, so each model accumulates the others'."""
    import os

    root = tmp_path / name
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.setenv(cc._ENV_DIR, str(root))
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising = False)

    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)

    published = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    assert published != str(ctx.dir / "inductor")
    if published is not None:
        assert not cc._toolchain_path_unparseable(published)
        assert not published.startswith(str(root))
    # The bundle is unaffected either way: only the Inductor pin moves.
    assert ctx.dir.is_dir() and ctx.dir.is_relative_to(root)


def test_two_models_under_an_unparseable_root_do_not_share_one_inductor_cache(
    monkeypatch, tmp_path, fake_megacache
):
    """The isolation the per-key directory exists for has to survive the substitution."""
    import os

    root = tmp_path / "o'brien"
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.setenv(cc._ENV_DIR, str(root))
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising = False)

    first = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    one = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    other = dict(_BEGIN_KW, shape_bucket = "512x512")
    second = cc.begin(transformer = _transformer(), **other)
    two = os.environ.get("TORCHINDUCTOR_CACHE_DIR")

    assert first.key != second.key
    assert one and two and one != two


# -------------------------------------------------------------------------- legacy root
def _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache) -> tuple:
    """Write a bundle under a fake pre-relocation root, then hand back an upgraded install:
    ``(legacy_root, studio_home, legacy_bundle)``, with the environment already pointing at a
    non-portable install whose new default root is empty. The bundle is content-addressed, so its
    name comes from the seeding save rather than from a constant."""
    legacy = tmp_path / "legacy" / "diffusion_compile_cache"
    studio_home = tmp_path / "studio"
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SAVE, raising = False)
    monkeypatch.setenv(cc._ENV_DIR, str(legacy))
    seeded = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert cc.save(seeded) is True

    monkeypatch.delenv(cc._ENV_DIR, raising = False)
    monkeypatch.delenv("UNSLOTH_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_PORTABLE", raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.setattr(cc, "_LEGACY_ROOT", legacy)
    return legacy, studio_home, seeded.bundle


def test_legacy_bundle_is_read_but_the_new_root_takes_the_writes(
    monkeypatch, tmp_path, fake_megacache
):
    # An upgraded install stays warm without the home directory becoming the write root:
    # begin() points TORCHINDUCTOR_CACHE_DIR at ctx.dir and save() writes every later bundle.
    legacy, studio_home, legacy_bundle = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)
    import os

    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)

    assert ctx.hit is True  # warm: the legacy bundle still counts
    assert str(ctx.dir).startswith(str(studio_home))
    assert str(os.environ["TORCHINDUCTOR_CACHE_DIR"]).startswith(str(studio_home))
    assert legacy not in ctx.dir.parents
    # The read fallback migrates: the pair now lives in the write root, byte-identical.
    assert ctx.bundle.read_bytes() == legacy_bundle.read_bytes()
    assert ctx.bundle.parent == ctx.dir
    assert ctx.manifest_path.exists()


def test_an_interrupted_migration_leaves_no_partial_pair(monkeypatch, tmp_path, fake_megacache):
    """The migration copies through a temp file and renames, as every other write here does.

    A plain copyfile onto the live name is visible while it is still partial. The manifest is the
    commit point, so a torn one is read as a miss and costs the cold compile the migration exists
    to avoid; a second backend migrating the same key would interleave its writes into the same
    destination. Interruption is injected at the manifest copy, the later of the two.
    """
    legacy, studio_home, legacy_bundle = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)

    import shutil as _shutil

    real_copyfile = _shutil.copyfile
    calls = {"n": 0}

    def exploding_copyfile(src, dst, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            # Half a manifest on disk, then death, exactly as interpreter exit kills the thread.
            Path(dst).write_bytes(Path(src).read_bytes()[:3])
            raise OSError("interrupted")
        return real_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(cc.shutil, "copyfile", exploding_copyfile)

    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)

    # The read still served this run from the legacy pair.
    assert ctx.hit is True
    # Nothing partial is published under a name a later run reads: the temp file the copy died
    # inside is not the live manifest.
    if ctx.manifest_path.exists():
        assert ctx.manifest_path.read_bytes() == (legacy / ctx.key / cc._MANIFEST_NAME).read_bytes()
    leftovers = [q.name for q in ctx.dir.iterdir() if q.name.endswith(cc._TEMP_SUFFIX)]
    assert leftovers == [], leftovers


def test_migrated_bundle_serves_the_next_run_without_the_legacy_root(
    monkeypatch, tmp_path, fake_megacache
):
    legacy, _, legacy_bundle = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)
    assert cc.begin(transformer = _transformer(), **_BEGIN_KW).hit is True

    import shutil

    shutil.rmtree(legacy)  # the old cache gets cleaned up: the warm start must survive
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx.hit is True


def test_load_only_mode_reads_legacy_without_writing_to_it(monkeypatch, tmp_path, fake_megacache):
    legacy, _, legacy_bundle = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)
    monkeypatch.setenv(cc._ENV_SAVE, "0")

    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)

    assert ctx.hit is True
    assert not (
        ctx.dir / legacy_bundle.name
    ).exists()  # a read-only cache stays read-only, both roots
    assert legacy_bundle.exists()


def test_key_absent_from_legacy_never_writes_into_it(monkeypatch, tmp_path, fake_megacache):
    # The old fallback swapped the WHOLE root, so even an unheld key wrote into the home dir.
    legacy, studio_home, legacy_bundle = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)
    before = {p.name for p in legacy.iterdir()}

    other = dict(_BEGIN_KW, family = "qwen-image")
    ctx = cc.begin(transformer = _transformer(("QwenImageTransformerBlock",)), **other)

    assert ctx.hit is False
    assert cc.save(ctx) is True
    assert str(ctx.bundle).startswith(str(studio_home))
    assert {p.name for p in legacy.iterdir()} == before


def test_portable_mode_never_falls_back_to_the_home_directory(monkeypatch, tmp_path):
    # begin() points TORCHINDUCTOR_CACHE_DIR inside this root, so a fallback here would write
    # GBs into the host machine's home directory.
    monkeypatch.delenv(cc._ENV_DIR, raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "portable"))
    legacy = tmp_path / "legacy" / "diffusion_compile_cache"
    legacy.mkdir(parents = True)
    monkeypatch.setattr(cc, "_LEGACY_ROOT", legacy)

    root = cc.cache_root()

    assert root != legacy
    assert str(root).startswith(str(tmp_path / "portable"))
    assert cc.legacy_cache_root() is None  # not even read: not part of the install


def test_explicit_dir_override_ignores_the_legacy_root(monkeypatch, tmp_path):
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path / "chosen"))
    legacy = tmp_path / "legacy" / "diffusion_compile_cache"
    legacy.mkdir(parents = True)
    monkeypatch.setattr(cc, "_LEGACY_ROOT", legacy)

    assert cc.cache_root() == tmp_path / "chosen"
    assert cc.legacy_cache_root() is None


def test_an_unreadable_legacy_root_is_a_miss_not_a_failure(monkeypatch, tmp_path):
    # The legacy root is the HOST's home, so it can be on a mount the new cache does not need.
    # Path.exists raises for EACCES and EIO before 3.14, and begin() does not catch around here.
    monkeypatch.delenv(cc._ENV_DIR, raising = False)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.delenv("UNSLOTH_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_PORTABLE", raising = False)
    legacy = tmp_path / "legacy" / "diffusion_compile_cache"
    monkeypatch.setattr(cc, "_LEGACY_ROOT", legacy)

    def _raise(self, *a, **k):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(Path, "exists", _raise)

    assert cc.legacy_cache_root() is None


def test_an_unreadable_legacy_bundle_pair_is_a_miss_not_a_failure(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy" / "diffusion_compile_cache"
    legacy.mkdir(parents = True)
    monkeypatch.setattr(cc, "legacy_cache_root", lambda: legacy)
    ctx = cc.CacheContext(
        key = "abc",
        dir = tmp_path / "new" / "abc",
        bundle = tmp_path / "new" / "abc" / "cache.bin",
        manifest_path = tmp_path / "new" / "abc" / "manifest.json",
        env_fp = "e",
        model_fp = "m",
        mode = "auto",
    )

    def _raise(self, *a, **k):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(Path, "exists", _raise)

    assert cc._load_from_legacy(ctx, None) is False


def test_a_manifest_that_is_not_an_object_is_a_miss(monkeypatch, tmp_path, fake_megacache):
    """json.loads returns [] for "[]" and None for "null", and .get() on either raises.

    _try_load's whole contract is that a bad cache entry is a miss, and the legacy root makes
    this reachable in a way it was not before: the manifest being validated was written by an
    older build, on a disk this run has never checked.
    """
    legacy, _, legacy_bundle = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)
    for payload in ("[]", "null", '"a string"', "42"):
        for key_dir in legacy.iterdir():
            (key_dir / cc._MANIFEST_NAME).write_text(payload, encoding = "utf-8")
        ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
        assert ctx.hit is False, payload


def test_an_unreadable_write_root_falls_back_to_legacy(monkeypatch, tmp_path, fake_megacache):
    """Path.exists() raises rather than answering False when a parent denies traversal.

    This branch moves the write root, so it can land on a directory the process does not own on
    some machine. Unguarded, that exception left begin() before the legacy fallback could run,
    which is the fallback the relocation depends on for a warm start.
    """
    legacy, studio_home, legacy_bundle = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)
    real_exists = Path.exists

    def _raise_under_studio(self, *a, **k):
        if str(self).startswith(str(studio_home)):
            raise PermissionError(13, "Permission denied")
        return real_exists(self, *a, **k)

    monkeypatch.setattr(Path, "exists", _raise_under_studio)
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx.hit is True


# ------------------------------------------------------------------ background save worker
# save_async hands the write to a single module-level daemon worker so a generation stops paying for it. These drive
# the worker directly, gating the fake save_cache_artifacts on Events so ordering is asserted, never slept on.
@pytest.fixture
def drained():
    """Leave the shared worker idle for the next test whatever this one did."""
    yield
    cc.wait_for_saves(timeout = 10.0)


def _fake_logger():
    class _L:
        def __init__(self):
            self.warnings: list[str] = []
            self.infos: list[str] = []

        def warning(self, fmt, *args):
            self.warnings.append(fmt % args if args else fmt)

        def info(self, fmt, *args):
            self.infos.append(fmt % args if args else fmt)

    return _L()


def test_async_save_writes_the_same_bytes_as_sync(monkeypatch, tmp_path, fake_megacache, drained):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SYNC, raising = False)

    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path / "sync"))
    sync_ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.register_shape(sync_ctx, (1024, 1024, 1), static = True)
    assert cc.save(sync_ctx) is True

    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path / "async"))
    async_ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.register_shape(async_ctx, (1024, 1024, 1), static = True)
    assert cc.save_async(async_ctx) is True
    assert cc.wait_for_saves(timeout = 10.0) is True

    assert async_ctx.bundle.read_bytes() == sync_ctx.bundle.read_bytes()
    a = json.loads(async_ctx.manifest_path.read_text())
    b = json.loads(sync_ctx.manifest_path.read_text())
    # "created" is a wall clock, everything else must match byte for byte.
    a.pop("created"), b.pop("created")
    assert a == b
    assert async_ctx.saved is True


def test_async_save_is_a_noop_on_a_clean_context(monkeypatch, tmp_path, fake_megacache, drained):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SYNC, raising = False)
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert cc.save_async(ctx) is True
    assert cc.wait_for_saves(timeout = 10.0) is True
    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx2.hit is True and ctx2.saved is True
    assert cc.save_async(ctx2) is False  # a hit has nothing to write


def test_worker_serialises_two_queued_saves(monkeypatch, tmp_path, fake_megacache, drained):
    """One save in flight at a time: the second context waits in the queue, untouched."""
    import threading

    import torch

    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SYNC, raising = False)
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def gated_save():
        calls.append(1)
        started.set()
        release.wait(10)
        return (b"ARTIFACT-BYTES", None)

    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", gated_save, raising = False)

    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path / "one"))
    ctx1 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path / "two"))
    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)

    assert cc.save_async(ctx1) is True
    assert started.wait(10) is True
    assert cc.save_async(ctx2) is True
    # Deterministic, no sleeping: ctx1 holds the worker and ctx2 is still queued behind it.
    assert cc._worker_active is ctx1
    assert [c is ctx2 for c, _ in cc._worker_queue] == [True]
    assert len(calls) == 1
    assert not ctx2.bundle.exists()

    release.set()
    assert cc.wait_for_saves(timeout = 10.0) is True
    assert len(calls) == 2
    assert ctx1.bundle.exists() and ctx2.bundle.exists()


def test_redirtied_context_queues_a_second_save(monkeypatch, tmp_path, fake_megacache, drained):
    """A shape registered WHILE a save runs is not in that bundle, so it must get its own save."""
    import threading

    import torch

    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SYNC, raising = False)
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))

    started = threading.Event()
    release = threading.Event()
    saves: list[int] = []

    def gated_save():
        saves.append(1)
        if len(saves) == 1:
            started.set()
            release.wait(10)
        return (b"ARTIFACT-BYTES", None)

    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", gated_save, raising = False)

    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.register_shape(ctx, (1024, 1024, 1), static = True)
    assert cc.save_async(ctx) is True
    assert started.wait(10) is True

    # Second generation, new static shape, while the first save is mid-flight.
    cc.register_shape(ctx, (768, 768, 1), static = True)
    assert ctx.saved is False
    assert cc.save_async(ctx) is True  # queued behind the running one, not dropped

    release.set()
    assert cc.wait_for_saves(timeout = 10.0) is True
    assert len(saves) == 2
    # The in-flight save must NOT have claimed the context clean, and the rewritten manifest covers both shapes.
    assert ctx.saved is True
    assert json.loads(ctx.manifest_path.read_text())["shapes"] == [[768, 768, 1], [1024, 1024, 1]]


def test_sync_env_switch_writes_inline(monkeypatch, tmp_path, fake_megacache, drained):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.setenv(cc._ENV_SYNC, "1")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    assert cc.sync_saves() is True
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.register_shape(ctx, (1024, 1024, 1), static = True)
    assert cc.save_async(ctx) is True
    # No worker involved: the bundle is on disk the moment save_async returns, as it was before the worker existed.
    assert ctx.bundle.exists() and ctx.manifest_path.exists()
    assert ctx.saved is True
    assert cc._worker_active is None and not cc._worker_queue


def test_worker_failure_is_swallowed_and_logged(monkeypatch, tmp_path, fake_megacache, drained):
    import torch

    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SYNC, raising = False)
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))

    def boom():
        raise RuntimeError("inductor exploded")

    monkeypatch.setattr(torch.compiler, "save_cache_artifacts", boom, raising = False)
    log = _fake_logger()
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.register_shape(ctx, (1024, 1024, 1), static = True)
    assert cc.save_async(ctx, logger = log) is True  # queuing succeeds; the failure is the worker's
    assert cc.wait_for_saves(timeout = 10.0) is True
    assert not ctx.bundle.exists()
    assert ctx.saved is False
    assert any("inductor exploded" in w for w in log.warnings)

    # The worker survives its own failure and takes the next save.
    monkeypatch.setattr(
        torch.compiler, "save_cache_artifacts", lambda: (b"ARTIFACT-BYTES", None), raising = False
    )
    assert cc.save_async(ctx, logger = log) is True
    assert cc.wait_for_saves(timeout = 10.0) is True
    assert ctx.bundle.exists()


def test_atomic_write_never_publishes_a_partial_file(monkeypatch, tmp_path, fake_megacache):
    """A save killed mid-write must leave the PREVIOUS bundle, not a truncated new one."""
    monkeypatch.setenv(cc._ENV_MODE, "on")
    monkeypatch.setenv(cc._ENV_SYNC, "1")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert cc.save(ctx) is True
    good = ctx.bundle.read_bytes()

    # Fail at the exact moment the finished temp file would be published.
    def exploding_replace(src, dst):
        raise OSError("killed mid-write")

    monkeypatch.setattr(cc.os, "replace", exploding_replace)
    ctx.saved = False
    assert cc.save(ctx) is False
    monkeypatch.undo()

    assert ctx.bundle.read_bytes() == good  # the old bundle is still whole and still loadable
    assert [p.name for p in ctx.dir.iterdir() if p.name.endswith(".tmp")] == []


def test_atomic_write_replaces_in_place(tmp_path):
    target = tmp_path / "cache.bin"
    cc._atomic_write(target, b"first")
    cc._atomic_write(target, b"second")
    assert target.read_bytes() == b"second"
    assert list(tmp_path.iterdir()) == [target]  # no temp files left behind


# ------------------------------------------------- interrupted saves and bundle collection
# The manifest is the single commit point and the bundle is content-addressed, so whatever a save is killed in the
# middle of, what is left on disk is a MATCHING pair. These drive each window of that directly.
@pytest.fixture
def mutable_megacache(monkeypatch):
    """Like ``fake_megacache`` but the artifact bytes can change between saves."""
    import torch

    state = {"bytes": b"ARTIFACT-ONE"}
    monkeypatch.setattr(
        torch.compiler, "save_cache_artifacts", lambda: (state["bytes"], None), raising = False
    )
    monkeypatch.setattr(
        torch.compiler,
        "load_cache_artifacts",
        lambda data: object() if data else None,
        raising = False,
    )
    return state


def _cold_pair(monkeypatch, tmp_path):
    """A committed manifest/bundle pair, saved synchronously."""
    monkeypatch.setenv(cc._ENV_MODE, "on")
    monkeypatch.setenv(cc._ENV_SYNC, "1")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert cc.save(ctx) is True
    return ctx


def test_bundles_are_content_addressed_and_named_by_the_manifest(
    monkeypatch, tmp_path, mutable_megacache
):
    ctx = _cold_pair(monkeypatch, tmp_path)
    digest = hashlib.sha256(b"ARTIFACT-ONE").hexdigest()
    assert ctx.bundle.name == cc._bundle_name(digest)
    manifest = json.loads(ctx.manifest_path.read_text())
    assert manifest["bundle"] == ctx.bundle.name
    assert manifest["sha256"] == digest
    # The fixed legacy name is never written any more.
    assert not (ctx.dir / cc._BUNDLE_NAME).exists()


def test_an_interrupted_bundle_write_leaves_the_previous_pair_loadable(
    monkeypatch, tmp_path, mutable_megacache
):
    """Window 1: the manifest is committed and the NEW bundle is only half written."""
    ctx = _cold_pair(monkeypatch, tmp_path)
    live, live_bytes = ctx.bundle, ctx.bundle.read_bytes()

    mutable_megacache["bytes"] = b"ARTIFACT-TWO-IS-A-DIFFERENT-LENGTH"
    doomed = ctx.dir / cc._bundle_name(hashlib.sha256(mutable_megacache["bytes"]).hexdigest())
    real_replace = cc.os.replace

    def die_publishing_the_bundle(src, dst):
        raise OSError("killed mid-write")

    monkeypatch.setattr(cc.os, "replace", die_publishing_the_bundle)
    ctx.saved = False
    assert cc.save(ctx) is False
    monkeypatch.setattr(cc.os, "replace", real_replace)  # not undo(): the env must survive

    # Nothing half written is visible under a name anything reads, and no temp file is left over.
    assert not doomed.exists()
    assert [p.name for p in ctx.dir.iterdir() if p.name.endswith(".tmp")] == []
    # The previous pair is untouched and still a real warm start.
    assert live.read_bytes() == live_bytes
    reopened = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert reopened.hit is True
    assert reopened.bundle == live


def test_a_bundle_published_without_its_manifest_leaves_the_previous_pair_loadable(
    monkeypatch, tmp_path, mutable_megacache
):
    """Window 2: the NEW bundle landed and the manifest never committed.

    This is the case a single fixed ``cache.bin`` got wrong: it left the old manifest describing a file that had
    already been overwritten, so the sha256 check discarded a warm start that was fine.
    """
    ctx = _cold_pair(monkeypatch, tmp_path)
    live, live_bytes = ctx.bundle, ctx.bundle.read_bytes()

    mutable_megacache["bytes"] = b"ARTIFACT-TWO-IS-A-DIFFERENT-LENGTH"
    orphan = ctx.dir / cc._bundle_name(hashlib.sha256(mutable_megacache["bytes"]).hexdigest())
    real_atomic = cc._atomic_write

    def die_before_committing_the_manifest(path, data):
        if path.name == cc._MANIFEST_NAME:
            raise OSError("killed before the manifest committed")
        return real_atomic(path, data)

    monkeypatch.setattr(cc, "_atomic_write", die_before_committing_the_manifest)
    ctx.saved = False
    assert cc.save(ctx) is False
    monkeypatch.setattr(cc, "_atomic_write", real_atomic)

    # The new bundle is on disk but nothing names it; the committed manifest still names the old one, intact.
    assert orphan.exists() and orphan.read_bytes() == b"ARTIFACT-TWO-IS-A-DIFFERENT-LENGTH"
    assert live.read_bytes() == live_bytes
    assert json.loads(ctx.manifest_path.read_text())["bundle"] == live.name
    reopened = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert reopened.hit is True
    assert reopened.bundle == live

    # And orphans do not accumulate. A save of the SAME artifacts adopts the orphan rather than rewriting it
    # (that is what content addressing buys), retiring the old bundle instead...
    monkeypatch.setattr(cc, "_GC_GRACE_SECONDS", -1.0)
    reopened.saved = False
    assert cc.save(reopened) is True
    assert reopened.bundle == orphan and orphan.exists()
    assert not live.exists()

    # ...and a save of DIFFERENT artifacts collects it like any other superseded bundle.
    mutable_megacache["bytes"] = b"ARTIFACT-THREE"
    reopened.saved = False
    assert cc.save(reopened) is True
    assert not orphan.exists()
    assert [p.name for p in reopened.dir.iterdir() if p.name.startswith(cc._BUNDLE_PREFIX)] == [
        reopened.bundle.name
    ]


def test_a_superseded_bundle_is_collected_and_the_live_pair_still_loads(
    monkeypatch, tmp_path, mutable_megacache
):
    """Window 3: a committed new pair retires the old bundle, and what is left still loads."""
    monkeypatch.setattr(cc, "_GC_GRACE_SECONDS", -1.0)
    ctx = _cold_pair(monkeypatch, tmp_path)
    first = ctx.bundle

    mutable_megacache["bytes"] = b"ARTIFACT-TWO-IS-A-DIFFERENT-LENGTH"
    ctx.saved = False
    assert cc.save(ctx) is True
    second = ctx.bundle

    assert second != first
    assert second.exists() and not first.exists()
    assert [p.name for p in ctx.dir.iterdir() if p.name.startswith(cc._BUNDLE_PREFIX)] == [
        second.name
    ]
    reopened = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert reopened.hit is True
    assert reopened.bundle == second


def test_collection_never_removes_the_live_bundle_or_one_a_racing_process_just_wrote(
    monkeypatch, tmp_path, mutable_megacache
):
    import os as _os

    ctx = _cold_pair(monkeypatch, tmp_path)

    # A bundle another process has written but not yet committed a manifest for is named by nothing. Deleting it
    # there would hand that process the broken pair this layout exists to prevent, so the grace window spares it.
    stray = ctx.dir / cc._bundle_name("f" * 64)
    stray.write_bytes(b"written by another process, manifest still pending")
    assert cc._collect_superseded(ctx.dir, None) == []
    assert stray.exists()

    # Once it is old enough to not be anybody's in-flight save, it goes.
    _os.utime(stray, (0, 0))
    assert cc._collect_superseded(ctx.dir, None) == [stray.name]
    assert not stray.exists()

    # The bundle the manifest ON DISK names is never a candidate, however old it is.
    _os.utime(ctx.bundle, (0, 0))
    assert cc._collect_superseded(ctx.dir, None) == []
    assert ctx.bundle.exists()
    assert cc.begin(transformer = _transformer(), **_BEGIN_KW).hit is True


def test_a_manifest_from_before_content_addressing_still_hits(
    monkeypatch, tmp_path, mutable_megacache
):
    """Bundles written by the old layout name no "bundle" key, so they resolve to cache.bin and keep hitting."""
    ctx = _cold_pair(monkeypatch, tmp_path)
    legacy = ctx.dir / cc._BUNDLE_NAME
    ctx.bundle.rename(legacy)
    manifest = json.loads(ctx.manifest_path.read_text())
    manifest.pop("bundle")
    ctx.manifest_path.write_text(json.dumps(manifest))

    reopened = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert reopened.hit is True
    assert reopened.bundle == legacy

    # And the legacy file is collected only once a new pair supersedes it.
    monkeypatch.setattr(cc, "_GC_GRACE_SECONDS", -1.0)
    mutable_megacache["bytes"] = b"ARTIFACT-TWO-IS-A-DIFFERENT-LENGTH"
    reopened.saved = False
    assert cc.save(reopened) is True
    assert not legacy.exists()
    assert cc.begin(transformer = _transformer(), **_BEGIN_KW).hit is True


def test_a_bundle_spared_by_the_grace_window_is_collected_when_the_key_is_opened_again(
    monkeypatch, tmp_path, mutable_megacache
):
    # Two saves for one key inside the grace window: the first bundle is superseded but too young
    # to collect, and the second save is the last thing that would ever have looked at it. Opening
    # the key again is what collects it, so the window costs a delay rather than the disk.
    ctx = _cold_pair(monkeypatch, tmp_path)
    first = ctx.bundle

    mutable_megacache["bytes"] = b"ARTIFACT-TWO-IS-A-DIFFERENT-LENGTH"
    ctx.saved = False
    monkeypatch.setattr(cc, "_GC_GRACE_SECONDS", 3600.0)
    assert cc.save(ctx) is True
    second = ctx.bundle

    # Still there: inside an hour-long grace, the save left it alone.
    assert second != first
    assert first.exists()

    # Next open of the same key, with that grace expired, takes it.
    monkeypatch.setattr(cc, "_GC_GRACE_SECONDS", 60.0)
    import os as _os

    _os.utime(first, (0, 0))
    reopened = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert not first.exists()
    assert second.exists()
    assert reopened.hit is True
    assert reopened.bundle == second


def test_a_bundle_rejected_on_its_checksum_is_rewritten_rather_than_kept(
    monkeypatch, tmp_path, mutable_megacache
):
    # Corruption on disk gives a load that fails the checksum. The recompile that follows produces
    # the same artifacts and therefore the same content-addressed NAME, so the exists() shortcut
    # would leave the corrupt bytes in place under a manifest that names them and the cache could
    # never come back. The rejection is remembered and the file overwritten.
    ctx = _cold_pair(monkeypatch, tmp_path)
    good = ctx.bundle.read_bytes()
    ctx.bundle.write_bytes(b"CORRUPT" + good[7:])

    reopened = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert reopened.hit is False
    assert reopened.rejected_bundle == ctx.bundle.name

    reopened.saved = False
    assert cc.save(reopened) is True
    assert reopened.bundle.read_bytes() == good

    # And the next start hits again, which is the whole point: the cache healed itself.
    assert cc.begin(transformer = _transformer(), **_BEGIN_KW).hit is True


def test_a_temp_file_left_by_a_killed_save_is_collected(monkeypatch, tmp_path, mutable_megacache):
    # _atomic_write's finally does not run when the interpreter tears the daemon save thread down
    # inside fh.write, so its temp file survives, and it is as large as the bundle. Collection now
    # recognises it, under the same grace window that protects a write actually in flight.
    import os as _os

    ctx = _cold_pair(monkeypatch, tmp_path)
    stranded = ctx.dir / f".{ctx.bundle.name}.abcd1234{cc._TEMP_SUFFIX}"
    stranded.write_bytes(b"half of a multi-GB bundle")

    # Fresh, so it could still be somebody's in-flight write.
    assert cc._collect_superseded(ctx.dir, None) == []
    assert stranded.exists()

    _os.utime(stranded, (0, 0))
    assert cc._collect_superseded(ctx.dir, None) == [stranded.name]
    assert not stranded.exists()
    assert ctx.bundle.exists()


def test_mark_recompiled_dirties_a_context_whose_shape_is_already_registered(tmp_path):
    # Automatic dynamic recompiles on the first new text length at an unchanged (width, height, batch); register_shape
    # sees no new key, so the recompiled graphs only reach disk if the context is re-dirtied explicitly.
    ctx = cc.CacheContext(
        key = "abc",
        dir = tmp_path / "abc",
        bundle = tmp_path / "abc" / "cache.bin",
        manifest_path = tmp_path / "abc" / "manifest.json",
        env_fp = "e",
        model_fp = "m",
        mode = "auto",
    )
    cc.register_shape(ctx, (1024, 1024, 1), static = True)
    ctx.saved = True
    seq = ctx.dirty_seq
    cc.register_shape(ctx, (1024, 1024, 1), static = True)
    assert ctx.saved is True and ctx.dirty_seq == seq
    cc.mark_recompiled(ctx)
    assert ctx.saved is False and ctx.dirty_seq == seq + 1
    cc.mark_recompiled(None)  # no context, no error


def test_a_failed_or_cancelled_render_still_dirties_the_bundle_after_a_recompile():
    # The exception path marks the context too: a later render reusing the generalised graph compiles nothing.
    import inspect

    from core.inference import diffusion

    src = inspect.getsource(diffusion.DiffusionBackend)
    before = src.index("graphs_before = fresh_compile_count()")
    handler = src.index("except BaseException:", before)
    reraise = src.index("raise\n", handler)
    assert "compile_cache.mark_recompiled(state.compile_cache_ctx)" in src[handler:reraise]


# ---------------------------------------------------------------------------- eviction
def _key_dir(root: Path, name: str, size: int, age_s: float) -> Path:
    import os
    import time

    d = root / name
    (d / "inductor").mkdir(parents = True)
    (d / "inductor" / "blob").write_bytes(b"x" * size)
    (d / cc._MANIFEST_NAME).write_text("{}")
    used = d / cc._LAST_USED_NAME
    used.write_text("")
    t = time.time() - age_s
    os.utime(used, (t, t))
    return d


def test_max_cache_bytes_env(monkeypatch):
    monkeypatch.delenv(cc._ENV_MAX_GB, raising = False)
    assert cc.max_cache_bytes() == int(cc._DEFAULT_MAX_GB * (1 << 30))
    monkeypatch.setenv(cc._ENV_MAX_GB, "0.5")
    assert cc.max_cache_bytes() == 1 << 29
    monkeypatch.setenv(cc._ENV_MAX_GB, "0")
    assert cc.max_cache_bytes() is None
    monkeypatch.setenv(cc._ENV_MAX_GB, "not-a-number")
    assert cc.max_cache_bytes() == int(cc._DEFAULT_MAX_GB * (1 << 30))
    for raw in ("nan", "inf", "-inf"):
        monkeypatch.setenv(cc._ENV_MAX_GB, raw)
        assert cc.max_cache_bytes() == int(cc._DEFAULT_MAX_GB * (1 << 30))


def test_evict_removes_least_recently_used_keys_until_under_budget(tmp_path):
    old = _key_dir(tmp_path, "a" * 32, 1000, age_s = 3 * 86400)
    mid = _key_dir(tmp_path, "b" * 32, 1000, age_s = 2 * 86400)
    new = _key_dir(tmp_path, "c" * 32, 1000, age_s = 1 * 86400)
    removed = cc.evict(root = tmp_path, max_bytes = 2500)
    assert removed == ["a" * 32]
    assert not old.exists() and mid.exists() and new.exists()


def test_evict_is_a_noop_under_budget_or_disabled(monkeypatch, tmp_path):
    d = _key_dir(tmp_path, "a" * 32, 1000, age_s = 86400)
    assert cc.evict(root = tmp_path, max_bytes = 10_000) == []
    monkeypatch.setenv(cc._ENV_MAX_GB, "0")
    assert cc.evict(root = tmp_path) == []
    assert d.exists()


def test_evict_spares_live_recent_and_foreign_dirs(tmp_path):
    live = _key_dir(tmp_path, "a" * 32, 1000, age_s = 5 * 86400)
    recent = _key_dir(tmp_path, "b" * 32, 1000, age_s = 60)
    foreign = tmp_path / "not-a-key"
    foreign.mkdir()
    (foreign / "big").write_bytes(b"x" * 5000)
    cc._register_live(live)
    try:
        assert cc.evict(root = tmp_path, max_bytes = 1) == []
    finally:
        cc._unregister_live(live)
    assert live.exists() and recent.exists() and foreign.exists()
    assert cc.evict(root = tmp_path, max_bytes = 1) == ["a" * 32]
    assert foreign.exists()


def test_save_evicts_other_keys_but_never_its_own(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SAVE, raising = False)
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    stale = _key_dir(tmp_path, "d" * 32, 4096, age_s = 30 * 86400)
    monkeypatch.setenv(
        cc._ENV_MAX_GB, str(1024 / (1 << 30))
    )
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    try:
        assert cc.save(ctx) is True
        assert not stale.exists()
        assert ctx.dir.exists() and ctx.manifest_path.exists() and ctx.bundle.exists()
    finally:
        cc.restore(ctx)


def test_begin_touches_last_used_and_restore_releases_the_key(
    monkeypatch, tmp_path, fake_megacache
):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert (ctx.dir / cc._LAST_USED_NAME).exists()
    assert str(ctx.dir) in cc._live_dirs
    cc.restore(ctx)
    assert str(ctx.dir) not in cc._live_dirs


def test_removed_key_reads_as_a_miss_not_a_broken_pair(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert cc.save(ctx) is True
    cc.restore(ctx)
    cc._remove_key_dir(ctx.dir)
    ctx2 = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    try:
        assert ctx2.hit is False
    finally:
        cc.restore(ctx2)


def test_fresh_compile_count_ignores_a_retrace_the_cache_serves():
    pytest.importorskip("torch")
    from torch._dynamo.utils import counters

    from core.inference import diffusion_speed as ds_mod

    graphs, misses = ds_mod.dynamo_graph_count(), ds_mod.fresh_compile_count()
    counters["stats"]["unique_graphs"] += 3
    counters["inductor"]["fxgraph_cache_hit"] += 3
    try:
        assert ds_mod.dynamo_graph_count() == graphs + 3
        assert ds_mod.fresh_compile_count() == misses
        counters["inductor"]["fxgraph_cache_miss"] += 1
        assert ds_mod.fresh_compile_count() == misses + 1
    finally:
        counters["stats"]["unique_graphs"] -= 3
        counters["inductor"]["fxgraph_cache_hit"] -= 3
        counters["inductor"]["fxgraph_cache_miss"] -= 1


def test_a_render_marks_its_key_used_before_it_compiles(monkeypatch, tmp_path):
    import os
    import time

    ctx = cc.CacheContext(
        key = "e" * 32,
        dir = tmp_path / ("e" * 32),
        bundle = tmp_path / "b",
        manifest_path = tmp_path / "m",
        env_fp = {},
        model_fp = {},
        mode = "auto",
    )
    d = _key_dir(tmp_path, "e" * 32, 1000, age_s = 5 * 86400)
    ctx.last_touch = time.time() - 2 * cc._TOUCH_INTERVAL_SECONDS
    cc.note_use(ctx)
    assert time.time() - os.stat(d / cc._LAST_USED_NAME).st_mtime < 60
    assert cc.evict(root = tmp_path, max_bytes = 1) == []
    stamp = os.stat(d / cc._LAST_USED_NAME).st_mtime
    cc.note_use(ctx)
    assert os.stat(d / cc._LAST_USED_NAME).st_mtime == stamp
    cc.note_use(None)


def test_generate_marks_the_key_used_before_the_render():
    import inspect

    from core.inference import diffusion

    src = inspect.getsource(diffusion.DiffusionBackend)
    before = src.index("graphs_before = fresh_compile_count()")
    render = src.index("pending = list(chunks)", before)
    assert "compile_cache.note_use(state.compile_cache_ctx)" in src[before:render]


def test_evict_rereads_last_used_before_deleting(monkeypatch, tmp_path):
    import os
    import time

    old = _key_dir(tmp_path, "a" * 32, 1000, age_s = 5 * 86400)
    other = _key_dir(tmp_path, "b" * 32, 1000, age_s = 4 * 86400)
    real_dir_bytes = cc._dir_bytes

    def dir_bytes_then_claim(path):
        size = real_dir_bytes(path)
        if path == old:
            now = time.time()
            os.utime(old / cc._LAST_USED_NAME, (now, now))
        return size

    monkeypatch.setattr(cc, "_dir_bytes", dir_bytes_then_claim)
    assert cc.evict(root = tmp_path, max_bytes = 1500) == ["b" * 32]
    assert old.exists() and not other.exists()


def test_evict_counts_only_what_it_actually_removed(monkeypatch, tmp_path):
    import shutil

    stuck = _key_dir(tmp_path, "a" * 32, 1000, age_s = 5 * 86400)
    nxt = _key_dir(tmp_path, "b" * 32, 1000, age_s = 4 * 86400)
    keep = _key_dir(tmp_path, "c" * 32, 1000, age_s = 3 * 86400)
    real_rmtree = shutil.rmtree

    def rmtree_keeps_the_oldest(path, *args, **kwargs):
        if str(path).startswith(str(stuck)) or "a" * 32 in str(path):
            return
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", rmtree_keeps_the_oldest)
    cc.evict(root = tmp_path, max_bytes = 2500)
    assert not nxt.exists() and keep.exists()
    assert cc._dir_bytes(tmp_path) <= 2500
    monkeypatch.setattr(shutil, "rmtree", real_rmtree)
    cc.evict(root = tmp_path, max_bytes = 2500)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c" * 32]


def test_evict_skips_a_key_it_cannot_take_and_moves_on(monkeypatch, tmp_path):
    import os

    held = _key_dir(tmp_path, "a" * 32, 1000, age_s = 5 * 86400)
    nxt = _key_dir(tmp_path, "b" * 32, 1000, age_s = 4 * 86400)
    real_rename = os.rename

    def rename_refuses_held(src, dst):
        if str(src) == str(held):
            raise PermissionError("in use by another process")
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", rename_refuses_held)
    assert cc.evict(root = tmp_path, max_bytes = 1500) == ["b" * 32]
    assert held.exists() and (held / cc._MANIFEST_NAME).exists() and not nxt.exists()


def test_a_warm_hit_enforces_the_budget_without_a_save(monkeypatch, tmp_path, fake_megacache):
    monkeypatch.setenv(cc._ENV_MODE, "auto")
    monkeypatch.delenv(cc._ENV_SAVE, raising = False)
    monkeypatch.setenv(cc._ENV_SYNC, "1")
    monkeypatch.setenv(cc._ENV_DIR, str(tmp_path))
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert cc.save(ctx) is True
    cc.restore(ctx)
    stale = _key_dir(tmp_path, "d" * 32, 4096, age_s = 30 * 86400)
    monkeypatch.setenv(cc._ENV_MAX_GB, str(1024 / (1 << 30)))
    monkeypatch.setenv(cc._ENV_SAVE, "0")
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    cc.restore(ctx)
    assert stale.exists()
    monkeypatch.delenv(cc._ENV_SAVE)
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    try:
        assert ctx.hit is True and ctx.saved is True
        assert cc.save_async(ctx) is False
        assert not stale.exists()
        assert ctx.manifest_path.exists() and ctx.bundle.exists()
    finally:
        cc.restore(ctx)


def test_max_cache_bytes_overflowing_value_keeps_the_default(monkeypatch):
    monkeypatch.setenv(cc._ENV_MAX_GB, "1e300")
    assert cc.max_cache_bytes() == int(cc._DEFAULT_MAX_GB * (1 << 30))


def test_evict_counts_a_key_another_evictor_took(monkeypatch, tmp_path):
    import os
    import shutil

    taken = _key_dir(tmp_path, "a" * 32, 1000, age_s = 5 * 86400)
    nxt = _key_dir(tmp_path, "b" * 32, 1000, age_s = 4 * 86400)
    real_rename = os.rename

    def another_process_got_there_first(src, dst):
        if str(src) == str(taken):
            shutil.rmtree(taken)
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", another_process_got_there_first)
    assert cc.evict(root = tmp_path, max_bytes = 1500) == []
    assert not taken.exists() and nxt.exists()
