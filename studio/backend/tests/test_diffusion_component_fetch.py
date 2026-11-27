# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real diffusers loads; only the Hub fetch is faked (safetensors hits ENOSPC, the ``.bin`` fallback 404s)."""

import errno
import json
import types

import pytest

diffusers = pytest.importorskip("diffusers")
pytest.importorskip("diffusers.modular_pipelines")

from diffusers import AutoencoderKL  # noqa: E402
from diffusers.modular_pipelines import ComponentSpec, ModularPipeline  # noqa: E402
from huggingface_hub.errors import EntryNotFoundError  # noqa: E402

from core.inference.diffusion_component_fetch import (  # noqa: E402
    ComponentFetchError,
    fetch_failure_kind,
    load_components_strict,
)

REPO = "unsloth/MiniMax-H3"
_TINY_VAE = {
    "_class_name": "AutoencoderKL",
    "in_channels": 3,
    "out_channels": 3,
    "down_block_types": ["DownEncoderBlock2D"],
    "up_block_types": ["UpDecoderBlock2D"],
    "block_out_channels": [8],
    "layers_per_block": 1,
    "latent_channels": 4,
    "norm_num_groups": 4,
    "sample_size": 8,
}


class _Pipe:
    def __init__(self, specs):
        self._component_specs = specs
        self._pretrained_model_name_or_path = REPO
        for name in specs:
            setattr(self, name, None)

    def register_components(self, **components):
        for name, value in components.items():
            setattr(self, name, value)

    load_components = ModularPipeline.load_components


def _vae_spec():
    return ComponentSpec(
        name = "vae", type_hint = AutoencoderKL, pretrained_model_name_or_path = REPO, subfolder = "vae"
    )


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(_TINY_VAE), encoding = "utf-8")
    state = types.SimpleNamespace(weights = None, calls = [])

    def fake_config_download(repo_id, filename, **kwargs):
        return str(config)

    def fake_weights_download(repo_id, filename, **kwargs):
        state.calls.append(filename)
        if filename.endswith(".index.json"):
            raise EntryNotFoundError(
                f"404 Client Error. Entry Not Found for url: {repo_id}/{filename}"
            )
        return state.weights(filename)

    monkeypatch.setattr("diffusers.configuration_utils.hf_hub_download", fake_config_download)
    monkeypatch.setattr("diffusers.utils.hub_utils.hf_hub_download", fake_weights_download)
    return state


def _disk_full_then_404(filename):
    if filename.endswith(".safetensors"):
        raise OSError(errno.ENOSPC, "No space left on device", "/hub/blobs/abc.incomplete")
    raise EntryNotFoundError(f"404 Client Error. Entry Not Found for url: {REPO}/vae/{filename}")


def _not_found(filename):
    raise EntryNotFoundError(f"404 Client Error. Entry Not Found for url: {REPO}/vae/{filename}")


def test_diffusers_alone_leaves_the_vae_none_on_a_full_disk(fake_hub):
    fake_hub.weights = _disk_full_then_404
    pipe = _Pipe({"vae": _vae_spec()})
    pipe.load_components(names = ["vae"])
    assert pipe.vae is None
    assert any(c.endswith(".bin") for c in fake_hub.calls), fake_hub.calls


def test_a_full_disk_fails_the_load_naming_the_vae_and_the_disk(fake_hub):
    fake_hub.weights = _disk_full_then_404
    pipe = _Pipe({"vae": _vae_spec()})
    with pytest.raises(ComponentFetchError) as info:
        load_components_strict(pipe, names = ["vae"])
    err = info.value
    assert err.kind == "disk_full"
    assert err.component == "vae"
    assert "Not enough disk space" in str(err)
    assert f"vae component of {REPO} (vae/)" in str(err)
    assert ".bin" not in str(err)
    assert any(isinstance(link, OSError) and link.errno == errno.ENOSPC for link in _chain(err))


def test_a_vae_missing_on_the_hub_fails_the_load_as_not_found(fake_hub):
    fake_hub.weights = _not_found
    pipe = _Pipe({"vae": _vae_spec()})
    with pytest.raises(ComponentFetchError) as info:
        load_components_strict(pipe, names = ["vae"])
    assert info.value.kind == "not_found"
    assert "not found on Hugging Face" in str(info.value)


def test_a_transient_safetensors_failure_is_retried_safetensors_only(fake_hub, tmp_path):
    from safetensors.torch import save_file

    weights = tmp_path / "diffusion_pytorch_model.safetensors"
    save_file(AutoencoderKL.from_config(_TINY_VAE).state_dict(), str(weights))
    attempts = {"n": 0}

    def flaky(filename):
        if filename.endswith(".safetensors"):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ConnectionError("Connection reset by peer")
            return str(weights)
        return _not_found(filename)

    fake_hub.weights = flaky
    pipe = _Pipe({"vae": _vae_spec()})
    load_components_strict(pipe, names = ["vae"])
    assert isinstance(pipe.vae, AutoencoderKL)
    assert attempts["n"] == 2


def test_an_optional_component_may_stay_none(fake_hub):
    fake_hub.weights = _disk_full_then_404
    pipe = _Pipe({"vae": _vae_spec()})
    load_components_strict(pipe, names = ["vae"], optional = ("vae",))
    assert pipe.vae is None


def test_the_spec_is_left_unpatched_after_the_call(fake_hub):
    fake_hub.weights = _disk_full_then_404
    spec = _vae_spec()
    pipe = _Pipe({"vae": spec})
    with pytest.raises(ComponentFetchError):
        load_components_strict(pipe, names = ["vae"])
    assert "load" not in vars(spec)


@pytest.mark.parametrize(
    "exc, kind",
    [
        (OSError(errno.ENOSPC, "No space left on device"), "disk_full"),
        (RuntimeError("Data processing error: No space left on device (os error 28)"), "disk_full"),
        (EntryNotFoundError("404 Client Error. Entry Not Found"), "not_found"),
        (ConnectionError("reset"), "network"),
        (TimeoutError("timed out"), "network"),
        (ValueError("bad config"), "error"),
    ],
)
def test_fetch_failure_kind(exc, kind):
    assert fetch_failure_kind(exc) == kind


def test_disk_full_wins_over_a_wrapping_error():
    try:
        try:
            raise OSError(errno.ENOSPC, "No space left on device")
        except OSError as inner:
            raise OSError("Can't load the model for 'unsloth/MiniMax-H3'") from inner
    except OSError as outer:
        assert fetch_failure_kind(outer) == "disk_full"


def _chain(exc):
    seen = []
    while exc is not None and exc not in seen:
        seen.append(exc)
        exc = exc.__cause__ or exc.__context__
    return seen
