# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The text-encoder release maps tensors to the encoder's own safetensors names. Those names come from the
checkpoint's publisher: a name made of many dots must not make the suffix index quadratic in memory."""

from __future__ import annotations

import tracemalloc

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from core.inference import diffusion_te_release as te_release


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(512, 1024)
        self.proj = torch.nn.Linear(1024, 1024)


def _source(
    encoder,
    path,
    extra = None,
):
    tensors = {n: t.detach().clone().contiguous() for n, t in encoder.named_parameters()}
    tensors.update(extra or {})
    safetensors_torch.save_file(tensors, str(path))
    return path


def _matched(encoder, path):
    setattr(encoder, te_release.WEIGHT_FILES_ATTR, (str(path),))
    releaser = te_release.TextEncoderReleaser([("text_encoder", encoder)])
    try:
        tracemalloc.start()
        releaser._match_sources()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return sorted(releaser._from_source), peak
    finally:
        releaser.close()


def test_dotted_tensor_name_does_not_blow_up_the_suffix_index(tmp_path, monkeypatch):
    monkeypatch.delenv(te_release.SOURCE_REUSE_ENV, raising = False)
    torch.manual_seed(0)
    encoder = _Encoder()
    # 20k dots: every suffix kept would be ~200 MB of strings; a real tensor name is under 200 chars.
    hostile = {"." * 20_000 + "weight": torch.zeros(0)}
    keys, peak = _matched(encoder, _source(encoder, tmp_path / "te.safetensors", hostile))
    assert keys == ["text_encoder.embed.weight", "text_encoder.proj.weight"]
    assert peak < 32 * 1024 * 1024, f"suffix index peaked at {peak / 1e6:.0f} MB"


def test_nested_checkpoint_names_still_match(tmp_path, monkeypatch):
    """Checkpoint names one level deeper (``model.``) or shallower than the module keep matching."""
    monkeypatch.delenv(te_release.SOURCE_REUSE_ENV, raising = False)
    torch.manual_seed(0)
    encoder = _Encoder()
    tensors = {"model.text." + n: t.detach().clone() for n, t in encoder.named_parameters()}
    safetensors_torch.save_file(tensors, str(tmp_path / "deep.safetensors"))
    keys, _ = _matched(encoder, tmp_path / "deep.safetensors")
    assert keys == ["text_encoder.embed.weight", "text_encoder.proj.weight"]

    wrapper = torch.nn.Module()
    wrapper.inner = _Encoder()
    torch.manual_seed(0)
    wrapper.inner.load_state_dict(_Encoder().state_dict())
    _source(wrapper.inner, tmp_path / "flat.safetensors")
    keys, _ = _matched(wrapper, tmp_path / "flat.safetensors")
    assert keys == ["text_encoder.inner.embed.weight", "text_encoder.inner.proj.weight"]
