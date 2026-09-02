# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the pre-warmed torch.compile cache (``diffusion_compile_cache.py``).

The Mega-cache API (``torch.compiler.save_cache_artifacts`` / ``load_cache_artifacts``)
is monkeypatched with deterministic in-memory fakes so the fingerprint / exact-match /
integrity / fallback / lifecycle logic is exercised without a real compile. The
fingerprint helpers run against the real torch on this box.
"""

from __future__ import annotations

import json
import types

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


# -------------------------------------------------------------------------- legacy root
def _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache) -> tuple:
    """Write a bundle under a fake pre-relocation root, then hand back an upgraded install.

    Returns ``(legacy_root, studio_home)`` with the environment already pointing at a
    non-portable install whose new default root is empty.
    """
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
    return legacy, studio_home


def test_legacy_bundle_is_read_but_the_new_root_takes_the_writes(
    monkeypatch, tmp_path, fake_megacache
):
    # An upgraded install must stay warm WITHOUT the home directory becoming the
    # permanent write root: begin() points TORCHINDUCTOR_CACHE_DIR at ctx.dir, and
    # save() writes every later bundle there, so that is GBs outside the studio root.
    legacy, studio_home = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)
    import os

    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)

    assert ctx.hit is True  # warm: the legacy bundle still counts
    assert str(ctx.dir).startswith(str(studio_home))
    assert str(os.environ["TORCHINDUCTOR_CACHE_DIR"]).startswith(str(studio_home))
    assert legacy not in ctx.dir.parents
    # The read fallback migrates rather than persisting: the pair now lives in the
    # write root, byte-identical to what just loaded.
    assert ctx.bundle.read_bytes() == (legacy / ctx.key / cc._BUNDLE_NAME).read_bytes()
    assert ctx.manifest_path.exists()


def test_migrated_bundle_serves_the_next_run_without_the_legacy_root(
    monkeypatch, tmp_path, fake_megacache
):
    legacy, _ = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)
    assert cc.begin(transformer = _transformer(), **_BEGIN_KW).hit is True

    import shutil

    shutil.rmtree(legacy)  # the old cache gets cleaned up: the warm start must survive
    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)
    assert ctx.hit is True


def test_load_only_mode_reads_legacy_without_writing_to_it(
    monkeypatch, tmp_path, fake_megacache
):
    legacy, _ = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)
    monkeypatch.setenv(cc._ENV_SAVE, "0")

    ctx = cc.begin(transformer = _transformer(), **_BEGIN_KW)

    assert ctx.hit is True
    assert not ctx.bundle.exists()  # a read-only cache stays read-only, both roots
    assert (legacy / ctx.key / cc._BUNDLE_NAME).exists()


def test_key_absent_from_legacy_never_writes_into_it(monkeypatch, tmp_path, fake_megacache):
    # The old fallback swapped the WHOLE root, so even a key the legacy cache never
    # held wrote its bundle and its inductor output into the home directory.
    legacy, studio_home = _seed_legacy_bundle(monkeypatch, tmp_path, fake_megacache)
    before = {p.name for p in legacy.iterdir()}

    other = dict(_BEGIN_KW, family = "qwen-image")
    ctx = cc.begin(transformer = _transformer(("QwenImageTransformerBlock",)), **other)

    assert ctx.hit is False
    assert cc.save(ctx) is True
    assert str(ctx.bundle).startswith(str(studio_home))
    assert {p.name for p in legacy.iterdir()} == before


def test_portable_mode_never_falls_back_to_the_home_directory(monkeypatch, tmp_path):
    # begin() points TORCHINDUCTOR_CACHE_DIR inside this root, so falling back
    # here would write GBs into the host machine's home directory.
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
