# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A load that cannot download (cache-only, HF offline, no network) reads a cached earlier quant of the sd.cpp text
encoder when the BF16 default is not cached, instead of failing on the missing BF16 file. Fake HF cache under
tmp_path; no network, no weights."""

import pytest

from core.inference import diffusion_families as df
from core.inference.sd_cpp_backend import SdCppDiffusionBackend

_SHA = "a" * 40


def _fam(name):
    return next(f for f in df._FAMILIES if f.name == name)


def _cache_file(
    root,
    repo,
    filename,
    data = b"gguf",
):
    repo_dir = root / f"models--{repo.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / _SHA
    snapshot.mkdir(parents = True, exist_ok = True)
    (snapshot / filename).parent.mkdir(parents = True, exist_ok = True)
    (snapshot / filename).write_bytes(data)
    (repo_dir / "refs").mkdir(parents = True, exist_ok = True)
    (repo_dir / "refs" / "main").write_text(_SHA, encoding = "utf-8")
    return snapshot / filename


@pytest.fixture
def cache(monkeypatch, tmp_path):
    """An isolated live cache root, with huggingface_hub's own root pointed at an empty dir so the shared HF cache on
    this host cannot answer a lookup. The fetch resolves from the fake cache or raises the hub's local-miss error."""
    import huggingface_hub.constants as hub_constants
    from huggingface_hub import try_to_load_from_cache
    from huggingface_hub.errors import LocalEntryNotFoundError

    live = tmp_path / "live"
    live.mkdir()
    monkeypatch.setattr(
        "utils.hf_cache_settings.active_hf_hub_cache", lambda: str(live), raising = False
    )
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.delenv(df.SD_CPP_TEXT_ENCODER_QUANT_ENV, raising = False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    fetched = []

    def fake_download(repo, fn, tok, **kwargs):
        fetched.append((repo, fn, kwargs.get("local_files_only")))
        hit = try_to_load_from_cache(repo, fn, cache_dir = str(live))
        if isinstance(hit, str):
            return hit
        # What huggingface_hub raises both under local_files_only and when the Hub is unreachable.
        raise LocalEntryNotFoundError(f"{repo}/{fn} not cached")

    monkeypatch.setattr("utils.hf_xet_fallback.hf_hub_download_with_xet_fallback", fake_download)
    return live, fetched


def _klein_assets(
    b,
    gguf,
    inner_dim = None,
    **kw,
):
    fam = _fam("flux.2-klein")
    return b._asset_specs("unsloth/FLUX.2-klein-9B-GGUF", gguf, fam, inner_dim, **kw)


def _seed_klein_9b(live):
    _cache_file(live, "unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q4_K_M.gguf")
    _cache_file(live, "black-forest-labs/FLUX.2-dev", "vae/diffusion_pytorch_model.safetensors")
    return _cache_file(live, "unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf")


def test_cache_only_load_reads_the_cached_earlier_quant(cache):
    """Klein 9B loaded before the BF16 default fetched Qwen3-8B Q4_K_M. A background (cache-only) load must still
    resolve it rather than fail on the uncached BF16 file."""
    live, _fetched = cache
    old = _seed_klein_9b(live)
    b = SdCppDiffusionBackend()
    # The specs as every caller builds them today: the fetch alone must recover.
    paths = b._fetch_assets(
        _klein_assets(b, "flux-2-klein-9b-Q4_K_M.gguf"), None, local_files_only = True
    )
    assert paths["llm"] == str(old)
    # And the cache-only load's own specs already name the cached file.
    specs = _klein_assets(b, "flux-2-klein-9b-Q4_K_M.gguf", local_only = True)
    assert ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", "llm") in specs
    assert b._fetch_assets(specs, None, local_files_only = True)["llm"] == str(old)


def test_fetch_falls_back_when_the_default_cannot_be_downloaded(cache):
    """The specs still name BF16 (an online load wants it), but the Hub is unreachable: the fetch reads the cached
    quant, cache-only, instead of raising the hub's local-miss error."""
    live, fetched = cache
    old = _seed_klein_9b(live)
    b = SdCppDiffusionBackend()
    specs = _klein_assets(b, "flux-2-klein-9b-Q4_K_M.gguf")
    assert ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-BF16.gguf", "llm") in specs
    paths = b._fetch_assets(specs, None, local_files_only = False)
    assert paths["llm"] == str(old)
    assert ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf", True) in fetched


def test_hf_offline_plan_shows_the_cached_quant(cache, monkeypatch):
    """Under HF_HUB_OFFLINE the download plan (what the UI prices and stages) names the cached file the load will
    read, not a BF16 download that cannot happen."""
    live, _fetched = cache
    _cache_file(live, "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    b = SdCppDiffusionBackend()
    fam = _fam("z-image")
    specs = b._asset_specs("unsloth/Z-Image-Turbo-GGUF", "z-image-turbo-Q4_K_M.gguf", fam)
    assert ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q8_0.gguf", "llm") in specs
    assert ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf", "llm") not in specs


def test_highest_cached_precision_wins_and_bf16_is_kept_when_cached(cache):
    live, _fetched = cache
    _cache_file(live, "unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf")
    _cache_file(live, "unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf")
    b = SdCppDiffusionBackend()
    specs = _klein_assets(b, "flux-2-klein-9b-Q4_K_M.gguf", local_only = True)
    assert ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf", "llm") in specs
    _cache_file(live, "unsloth/Qwen3-8B-GGUF", "Qwen3-8B-BF16.gguf")
    specs = _klein_assets(b, "flux-2-klein-9b-Q4_K_M.gguf", local_only = True)
    assert ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-BF16.gguf", "llm") in specs


def test_online_specs_and_explicit_override_are_unchanged(cache, monkeypatch):
    """An online load still wants BF16 (fresh loads get the accuracy default), and an explicit
    UNSLOTH_SD_CPP_TEXT_ENCODER_QUANT is honoured, not swapped, even offline."""
    live, _fetched = cache
    _seed_klein_9b(live)
    b = SdCppDiffusionBackend()
    specs = _klein_assets(b, "flux-2-klein-9b-Q4_K_M.gguf")
    assert ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-BF16.gguf", "llm") in specs
    monkeypatch.setenv(df.SD_CPP_TEXT_ENCODER_QUANT_ENV, "q8_0")
    specs = _klein_assets(b, "flux-2-klein-9b-Q4_K_M.gguf", local_only = True)
    assert ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q8_0.gguf", "llm") in specs
    with pytest.raises(RuntimeError, match = "Qwen3-8B-Q8_0.gguf"):
        b._fetch_assets(specs, None, local_files_only = True)


def test_nothing_cached_still_fails_with_the_named_default(cache):
    live, _fetched = cache
    _cache_file(live, "unsloth/FLUX.2-klein-9B-GGUF", "flux-2-klein-9b-Q4_K_M.gguf")
    _cache_file(live, "black-forest-labs/FLUX.2-dev", "vae/diffusion_pytorch_model.safetensors")
    b = SdCppDiffusionBackend()
    specs = _klein_assets(b, "flux-2-klein-9b-Q4_K_M.gguf", local_only = True)
    with pytest.raises(RuntimeError, match = "Qwen3-8B-BF16.gguf"):
        b._fetch_assets(specs, None, local_files_only = True)


def test_only_the_requantable_llm_encoders_have_alternatives(cache):
    assert df.sd_cpp_text_encoder_cached_alternatives(
        "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf", "llm"
    ) == ("Qwen3-4B-Q8_0.gguf", "Qwen3-4B-Q4_K_M.gguf")
    assert (
        df.sd_cpp_text_encoder_cached_alternatives(
            "unsloth/Qwen3-VL-4B-Instruct-GGUF", "Qwen3-VL-4B-Instruct-Q8_0.gguf", "llm"
        )
        == ()
    )
    assert (
        df.sd_cpp_text_encoder_cached_alternatives(
            "unsloth/Qwen3-4B-GGUF", "Qwen3-4B-BF16.gguf", "llm_vision"
        )
        == ()
    )
