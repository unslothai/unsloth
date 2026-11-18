# SPDX-License-Identifier: AGPL-3.0-only

import errno
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from safetensors import torch as safetensors

from core.inference import diffusion_te_release as release


@pytest.fixture
def encoder_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "snapshots"
    monkeypatch.setattr(release, "_snapshot_root", lambda: root)
    monkeypatch.setattr(release, "_SHARD_BYTES", 1)
    monkeypatch.setenv(release.SOURCE_REUSE_ENV, "0")
    encoder = torch.nn.Sequential(
        torch.nn.Linear(512, 512, bias = False, dtype = torch.float32, device = "cpu"),
        torch.nn.Linear(512, 512, bias = False, dtype = torch.float32, device = "cpu"),
    )
    releaser = release.TextEncoderReleaser([("text_encoder", encoder)])
    try:
        yield encoder, releaser, root
    finally:
        releaser.close()


@pytest.mark.parametrize("fail_on", [1, 2], ids = ["first-shard", "later-shard"])
def test_failed_snapshot_is_removed_before_the_releaser_is_closed(
    encoder_snapshot, monkeypatch, fail_on
):
    encoder, releaser, root = encoder_snapshot
    original = {name: tensor.detach().clone() for name, tensor in encoder.state_dict().items()}
    pointers = [parameter.data_ptr() for parameter in encoder.parameters()]
    value = torch.ones(1, 512, dtype = torch.float32, device = "cpu")
    expected = encoder(value).detach().clone()
    save_file = safetensors.save_file
    written = []
    peer = root / f"{os.getpid()}-other-model"
    peer.mkdir(parents = True)
    (peer / "weights.bin").write_bytes(b"other model")

    def fail_write(tensors, filename):
        written.append(filename)
        if len(written) == fail_on:
            Path(filename).write_bytes(b"incomplete snapshot")
            raise OSError(errno.ENOSPC, "No space left on device")
        save_file(tensors, filename)

    monkeypatch.setattr(safetensors, "save_file", fail_write)

    assert releaser.release() == 0
    assert len(written) == fail_on
    assert not releaser.released
    assert releaser.releases == 0
    assert [parameter.data_ptr() for parameter in encoder.parameters()] == pointers
    for name, tensor in encoder.state_dict().items():
        assert torch.equal(tensor, original[name])
    assert torch.equal(encoder(value), expected)
    assert list(root.iterdir()) == [peer]
    assert (peer / "weights.bin").read_bytes() == b"other model"


def test_successful_snapshot_is_kept_for_reload_until_close(encoder_snapshot):
    encoder, releaser, root = encoder_snapshot
    size = sum(parameter.numel() * parameter.element_size() for parameter in encoder.parameters())
    original = {name: tensor.detach().clone() for name, tensor in encoder.state_dict().items()}
    value = torch.ones(1, 512, dtype = torch.float32, device = "cpu")
    expected = encoder(value).detach().clone()

    assert releaser.release() == size
    files = sorted(root.rglob("*.safetensors"))
    assert len(files) == 2
    assert sum(path.stat().st_size for path in files) >= size
    torch.testing.assert_close(encoder(value), expected)
    for name, tensor in encoder.state_dict().items():
        assert torch.equal(tensor, original[name])
    assert sorted(root.rglob("*.safetensors")) == files
    assert releaser.release() == size
    torch.testing.assert_close(encoder(value), expected)
    for name, tensor in encoder.state_dict().items():
        assert torch.equal(tensor, original[name])
    assert sorted(root.rglob("*.safetensors")) == files

    releaser.close()
    assert not list(root.iterdir())
