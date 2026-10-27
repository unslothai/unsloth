# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hosted bf16 VAE (diffusion_hosted_vae). CPU, tiny, no network."""

from __future__ import annotations

import json
import logging
import types

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

from core.inference import diffusion_hosted_vae as hv
from core.inference.diffusion_families import detect_family

BASE = "Qwen/Qwen-Image-2.1"
REPO = "unsloth/Qwen-Image-2.1-FP8"
FILE = "vae/qwen_image_2.1_vae_bf16.safetensors"


@pytest.fixture
def fam():
    family = detect_family(BASE)
    assert family is not None and family.name == "qwen-image-2.1"
    return family


@pytest.fixture(autouse = True)
def _switch_on(monkeypatch):
    monkeypatch.delenv(hv.HOSTED_VAE_ENV, raising = False)


def test_the_registry_names_the_hosted_file_for_the_base_and_its_mirror(fam):
    assert hv.hosted_vae_source(fam, BASE) == (REPO, FILE)
    assert hv.hosted_vae_source(fam, "unsloth/Qwen-Image-2.1") == (REPO, FILE)
    # Turbo's own vae/ is already bf16; forks and unknown bases never take it.
    assert hv.hosted_vae_source(fam, "Qwen/Qwen-Image-2.1-Turbo") is None
    assert hv.hosted_vae_source(fam, "someone/Qwen-Image-2.1-fork") is None
    assert hv.hosted_vae_source(fam, None) is None


def test_only_a_bf16_pipeline_load_wants_it(fam, monkeypatch):
    assert hv.hosted_vae_wanted(fam, BASE, kind = "pipeline", dtype = torch.bfloat16) == (REPO, FILE)
    for dtype in (torch.float16, torch.float32):
        assert hv.hosted_vae_wanted(fam, BASE, kind = "pipeline", dtype = dtype) is None
    for kind in ("gguf", "single_file", None):
        assert hv.hosted_vae_wanted(fam, BASE, kind = kind, dtype = torch.bfloat16) is None
    monkeypatch.setenv(hv.HOSTED_VAE_ENV, "0")
    assert hv.hosted_vae_wanted(fam, BASE, kind = "pipeline", dtype = torch.bfloat16) is None


def test_other_families_have_no_hosted_vae():
    for repo in ("Qwen/Qwen-Image", "black-forest-labs/FLUX.1-dev", "Tongyi-MAI/Z-Image-Turbo"):
        family = detect_family(repo)
        assert family is not None
        assert hv.hosted_vae_source(family, repo) is None


def _paths_api(
    monkeypatch,
    size = 675_480_808,
    fail = False,
):
    import huggingface_hub
    class _Api:
        def __init__(self, token = None):
            pass

        def get_paths_info(self, repo_id, paths):
            if fail:
                raise RuntimeError("404 Client Error")
            return [types.SimpleNamespace(path = p, size = size) for p in paths]

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)


def test_the_plan_drops_vae_only_when_the_hub_lists_the_file(fam, monkeypatch):
    _paths_api(monkeypatch)
    got = hv.hosted_vae_plan_files(fam, BASE, kind = "pipeline", dtype = torch.bfloat16)
    assert got == {"vae": (REPO, [(FILE, 675_480_808)])}
    assert hv.hosted_vae_plan_files(fam, BASE, kind = "pipeline", dtype = torch.float16) == {}
    assert (
        hv.hosted_vae_plan_files(
            fam, BASE, kind = "pipeline", dtype = torch.bfloat16, local_files_only = True
        )
        == {}
    )
    _paths_api(monkeypatch, fail = True)
    assert hv.hosted_vae_plan_files(fam, BASE, kind = "pipeline", dtype = torch.bfloat16) == {}


def test_the_skip_drops_only_vae_weights():
    from core.inference.diffusion_te_prequant import is_prequant_covered_weight

    assert is_prequant_covered_weight("vae/diffusion_pytorch_model.safetensors", ("vae",))
    assert not is_prequant_covered_weight("vae/config.json", ("vae",))
    assert not is_prequant_covered_weight("transformer/x.safetensors", ("vae",))


def _vae_class():
    cls = getattr(diffusers, "AutoencoderKLQwenImage21", None)
    if cls is not None:
        return cls, {
            "base_dim": 32,
            "decoder_base_dim": 32,
            "z_dim": 4,
            "dim_mult": [1, 1],
            "num_res_blocks": 1,
            "temperal_downsample": [False],
            "latents_mean": [0.0] * 4,
            "latents_std": [1.0] * 4,
        }
    return diffusers.AutoencoderKLWan, {
        "base_dim": 8,
        "z_dim": 4,
        "dim_mult": [1, 2],
        "num_res_blocks": 1,
        "temperal_downsample": [False],
        "latents_mean": [0.0] * 4,
        "latents_std": [1.0] * 4,
    }


@pytest.fixture
def tiny(tmp_path):
    """A base folder with an fp32 vae/ (as the Hub stores it) and the bf16 single file cast from it."""
    from safetensors.torch import save_file

    cls, cfg = _vae_class()
    torch.manual_seed(0)
    dense = cls(**cfg)
    base = tmp_path / "base"
    dense.save_pretrained(base / "vae")
    hosted = tmp_path / "hosted.safetensors"
    save_file(
        {k: v.to(torch.bfloat16).contiguous() for k, v in dense.state_dict().items()}, str(hosted)
    )
    return types.SimpleNamespace(cls = cls, base = str(base), hosted = str(hosted), tmp = tmp_path)


def _hub(monkeypatch, path):
    import huggingface_hub

    seen: list = []

    def fake(repo_id, filename, **kwargs):
        seen.append((repo_id, filename))
        if path is None:
            raise FileNotFoundError(f"{repo_id}/{filename} not found")
        return path

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake)
    return seen


def _kwargs(fam, tiny, **over):
    kw = dict(
        kind = "pipeline",
        dtype = torch.bfloat16,
        config_source = tiny.base,
        logger = logging.getLogger("t"),
    )
    kw.update(over)
    return kw


def test_the_hosted_file_builds_the_module_from_pretrained_builds(fam, tiny, monkeypatch, caplog):
    seen = _hub(monkeypatch, tiny.hosted)
    with caplog.at_level(logging.INFO):
        got = hv.hosted_vae_pipe_kwargs(fam, BASE, **_kwargs(fam, tiny))
    assert seen == [(REPO, FILE)]
    assert "diffusion.hosted_vae: reading" in caplog.text
    vae = got["vae"]
    ref = tiny.cls.from_pretrained(tiny.base, subfolder = "vae", torch_dtype = torch.bfloat16)
    assert type(vae) is type(ref)
    a, b = vae.state_dict(), ref.state_dict()
    assert set(a) == set(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k
    assert not vae.training
    z = torch.randn(1, 4, 1, 4, 4, dtype = torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(vae.decode(z).sample, ref.decode(z).sample)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_other_dtypes_never_open_it(fam, tiny, monkeypatch, dtype):
    seen = _hub(monkeypatch, tiny.hosted)
    assert (
        hv.hosted_vae_pipe_kwargs(fam, BASE, **_kwargs(fam, tiny, dtype = getattr(torch, dtype)))
        == {}
    )
    assert seen == []


def test_kill_switch(fam, tiny, monkeypatch):
    seen = _hub(monkeypatch, tiny.hosted)
    monkeypatch.setenv(hv.HOSTED_VAE_ENV, "0")
    assert hv.hosted_vae_pipe_kwargs(fam, BASE, **_kwargs(fam, tiny)) == {}
    assert seen == []


def test_a_mirrored_file_is_read_in_place_not_downloaded(fam, tiny, monkeypatch):
    from safetensors.torch import load_file, save_file

    from core.inference.diffusion_te_prequant import TE_PREQUANT_MIRROR_ENV

    mirror = tiny.tmp / "mirror"
    dest = mirror / REPO / FILE
    dest.parent.mkdir(parents = True)
    save_file(load_file(tiny.hosted), str(dest))
    monkeypatch.setenv(TE_PREQUANT_MIRROR_ENV, str(mirror))
    seen = _hub(monkeypatch, None)
    got = hv.hosted_vae_pipe_kwargs(fam, BASE, **_kwargs(fam, tiny))
    assert seen == []
    assert "vae" in got


def test_a_missing_file_leaves_the_staged_vae(fam, tiny, monkeypatch, caplog):
    _hub(monkeypatch, None)
    with caplog.at_level(logging.WARNING):
        assert hv.hosted_vae_pipe_kwargs(fam, BASE, **_kwargs(fam, tiny)) == {}
    assert "not used" in caplog.text


def test_a_mismatched_file_is_refused(fam, tiny, monkeypatch, caplog):
    from safetensors.torch import load_file, save_file

    state = load_file(tiny.hosted)
    key = sorted(state)[0]
    state[key] = torch.zeros(3, dtype = torch.bfloat16)
    bad = tiny.tmp / "bad.safetensors"
    save_file(state, str(bad))
    _hub(monkeypatch, str(bad))
    with caplog.at_level(logging.WARNING):
        assert hv.hosted_vae_pipe_kwargs(fam, BASE, **_kwargs(fam, tiny)) == {}
    assert "not used" in caplog.text


def test_an_fp32_file_is_not_a_bf16_source(fam, tiny, monkeypatch):
    from safetensors.torch import load_file, save_file

    fp32 = tiny.tmp / "fp32.safetensors"
    save_file({k: v.float() for k, v in load_file(tiny.hosted).items()}, str(fp32))
    _hub(monkeypatch, str(fp32))
    assert hv.hosted_vae_pipe_kwargs(fam, BASE, **_kwargs(fam, tiny)) == {}


def test_when_the_plan_dropped_vae_a_miss_loads_the_base_vae_itself(fam, tiny, monkeypatch):
    _hub(monkeypatch, None)
    asked: list = []

    def from_pretrained(source, **kwargs):
        asked.append((source, kwargs["subfolder"], kwargs["torch_dtype"]))
        return "dense-vae"

    monkeypatch.setattr(tiny.cls, "from_pretrained", from_pretrained)
    got = hv.hosted_vae_pipe_kwargs(fam, BASE, **_kwargs(fam, tiny, dense_staged = False))
    assert got == {"vae": "dense-vae"}
    assert asked == [(BASE, "vae", torch.bfloat16)]
    # Same when the load landed at another dtype than the plan assumed.
    asked.clear()
    got = hv.hosted_vae_pipe_kwargs(
        fam, BASE, **_kwargs(fam, tiny, dense_staged = False, dtype = torch.float32)
    )
    assert got == {"vae": "dense-vae"} and asked == [(BASE, "vae", torch.float32)]


def test_the_backend_plans_through_the_load_target(fam, monkeypatch):
    from core.inference import diffusion as diff

    _paths_api(monkeypatch)
    backend = diff.DiffusionBackend.__new__(diff.DiffusionBackend)
    monkeypatch.setattr(
        backend,
        "_target_for_ordinal",
        lambda f, o: types.SimpleNamespace(dtype = torch.bfloat16),
        raising = False,
    )
    got = backend._hosted_vae_plan_files(fam, "pipeline", BASE, None, None, False)
    assert got == {"vae": (REPO, [(FILE, 675_480_808)])}
    monkeypatch.setattr(
        backend,
        "_target_for_ordinal",
        lambda f, o: types.SimpleNamespace(dtype = torch.float16),
        raising = False,
    )
    assert backend._hosted_vae_plan_files(fam, "pipeline", BASE, None, None, False) == {}
    assert json.dumps(got)
