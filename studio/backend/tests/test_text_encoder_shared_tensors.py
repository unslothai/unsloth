# SPDX-License-Identifier: AGPL-3.0-only

import pytest

torch = pytest.importorskip("torch")

from safetensors.torch import save_file

from core.inference import diffusion_te_release as release


class Projection(torch.nn.Module):
    def __init__(self, weight, parameter):
        super().__init__()
        if parameter:
            self.weight = weight
        else:
            self.register_buffer("weight", weight)

    def forward(self, value):
        return torch.nn.functional.linear(value, self.weight)


class Encoder(torch.nn.Module):
    def __init__(self, parameter):
        super().__init__()
        weight = torch.eye(512, dtype = torch.float32, device = "cpu")
        if parameter:
            weight = torch.nn.Parameter(weight)
        self.first = Projection(weight, parameter)
        self.second = Projection(weight, parameter)

    def forward(self, value):
        return self.first(value) + self.second(value)


@pytest.fixture(params = [False, True], ids = ["snapshot", "source"])
def releaser_factory(request, tmp_path, monkeypatch):
    monkeypatch.setattr(release, "_snapshot_root", lambda: tmp_path / "snapshots")
    monkeypatch.setenv(release.SOURCE_REUSE_ENV, str(int(request.param)))
    active = []

    def make(encoder):
        source = tmp_path / "source.safetensors"
        tensors = dict(encoder.named_parameters()) | dict(encoder.named_buffers())
        save_file({name: tensor.detach().clone() for name, tensor in tensors.items()}, str(source))
        setattr(encoder, release.WEIGHT_FILES_ATTR, (str(source),))
        releaser = release.TextEncoderReleaser([("text_encoder", encoder)])
        active.append(releaser)
        return releaser

    yield make

    for releaser in active:
        releaser.close()


@pytest.mark.parametrize("parameter", [False, True], ids = ["buffer", "parameter"])
def test_shared_tensor_is_released_and_restored_without_breaking_aliases(
    parameter, releaser_factory
):
    encoder = Encoder(parameter)
    original = encoder.first.weight
    size = original.numel() * original.element_size()
    value = torch.ones(1, 512, dtype = torch.float32, device = "cpu")
    expected = encoder(value).detach().clone()
    releaser = releaser_factory(encoder)

    assert releaser.release() == size
    assert bool(releaser._from_source) == release.source_reuse_enabled()
    assert encoder.first.weight is original
    assert encoder.second.weight is original
    assert original.numel() == 0
    assert torch.equal(encoder(value), expected)
    assert encoder.first.weight is original
    assert encoder.second.weight is original


@pytest.mark.parametrize("parameter", [False, True], ids = ["buffer", "parameter"])
def test_any_shared_tensor_owner_reloads_before_forward(parameter, releaser_factory):
    encoder = Encoder(parameter)
    value = torch.ones(1, 512, dtype = torch.float32, device = "cpu")
    expected = encoder.second(value).detach().clone()
    releaser = releaser_factory(encoder)
    releaser.release()

    assert torch.equal(encoder.second(value), expected)
    assert not releaser.released
    assert encoder.first.weight is encoder.second.weight


def test_nonpersistent_alias_owner_also_reloads(releaser_factory):
    encoder = Encoder(False)
    encoder.second._non_persistent_buffers_set.add("weight")
    value = torch.ones(1, 512, dtype = torch.float32, device = "cpu")
    expected = encoder.second(value).clone()
    releaser = releaser_factory(encoder)

    assert releaser.release() > 0
    assert encoder.second.weight.numel() == 0
    assert torch.equal(encoder.second(value), expected)
    assert not releaser.released
    assert encoder.first.weight is encoder.second.weight
