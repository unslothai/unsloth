# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-encoder input embeddings kept on the checkpoint's file mapping: host gather, bit-identical, never copied."""

import sys

import pytest
import torch

from core.inference import diffusion_te_embed_host as eh
from core.inference import video_ltx25_te_worker as worker

linux_only = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason = "file-mapping check reads /proc"
)

VOCAB, WIDTH = 64, 16


class _Encoder(torch.nn.Module):
    def __init__(self, table):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(VOCAB, WIDTH)
        self.embed_tokens.weight = torch.nn.Parameter(table, requires_grad = False)
        self.proj = torch.nn.Linear(WIDTH, WIDTH)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, ids):
        return self.proj(self.embed_tokens(ids))


def _mapped_table(tmp_path):
    torch.manual_seed(0)
    data = torch.randn(VOCAB, WIDTH, dtype = torch.float32)
    path = tmp_path / "table.bin"
    data.numpy().tofile(path)
    # MAP_PRIVATE view of the file, as transformers leaves a safetensors tensor of the stored dtype
    return torch.from_file(str(path), shared = False, size = VOCAB * WIDTH, dtype = torch.float32).view(
        VOCAB, WIDTH
    ), data


def _encoder(tmp_path):
    mapped, data = _mapped_table(tmp_path)
    torch.manual_seed(1)
    model = _Encoder(mapped)
    ref = _Encoder(data.clone())
    ref.proj.load_state_dict(model.proj.state_dict())
    return model, ref


@linux_only
def test_file_mapped_detects_views_only(tmp_path):
    mapped, data = _mapped_table(tmp_path)
    assert eh.file_mapped(mapped)
    assert eh.file_mapped(mapped[8:])
    assert not eh.file_mapped(data)
    assert not eh.file_mapped(mapped.clone())


@linux_only
def test_keep_on_host_is_bit_identical_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.delenv(eh.EMBED_HOST_ENV, raising = False)
    model, ref = _encoder(tmp_path)
    ids = torch.tensor([[1, 5, 9, 63, 0]])
    assert eh.keep_on_host(model) == VOCAB * WIDTH * 4
    assert eh.is_host_embedding(model.embed_tokens)
    assert eh.keep_on_host(model) == VOCAB * WIDTH * 4
    assert len(model.embed_tokens._forward_hooks) == 1
    with torch.inference_mode():
        assert torch.equal(model(ids), ref(ids))
    assert eh.release(model)
    assert not eh.is_host_embedding(model.embed_tokens)
    assert not model.embed_tokens._forward_hooks and not model.embed_tokens._forward_pre_hooks


@linux_only
@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_kill_switch(tmp_path, monkeypatch, value):
    monkeypatch.setenv(eh.EMBED_HOST_ENV, value)
    model, _ = _encoder(tmp_path)
    assert eh.keep_on_host(model) == 0
    assert not eh.is_host_embedding(model.embed_tokens)


def test_private_table_is_left_alone(monkeypatch):
    monkeypatch.delenv(eh.EMBED_HOST_ENV, raising = False)
    model = _Encoder(torch.randn(VOCAB, WIDTH))
    assert eh.keep_on_host(model) == 0
    assert not eh.is_host_embedding(model.embed_tokens)


def test_non_linux_never_engages(tmp_path, monkeypatch):
    monkeypatch.setattr(eh.sys, "platform", "win32")
    mapped, _ = _mapped_table(tmp_path)
    assert not eh.file_mapped(mapped)


@linux_only
def test_onload_and_owned_copy_skip_the_host_table(tmp_path, monkeypatch):
    monkeypatch.delenv(eh.EMBED_HOST_ENV, raising = False)
    model, ref = _encoder(tmp_path)
    table_ptr = model.embed_tokens.weight.data_ptr()
    proj_ptr = model.proj.weight.data_ptr()
    assert eh.keep_on_host(model)
    monkeypatch.setattr(worker, "_host_fits", lambda *_a, **_k: True)
    assert worker._own_host_copy(model)
    assert model.embed_tokens.weight.data_ptr() == table_ptr
    assert eh.file_mapped(model.embed_tokens.weight.data)
    assert model.proj.weight.data_ptr() != proj_ptr
    onload = worker._Onload(torch.device("cpu"))
    onload.add([model])
    assert model.embed_tokens.weight.data_ptr() == table_ptr
    moved_proj = model.proj.weight.data_ptr()
    ids = torch.tensor([[3, 2, 1]])
    with torch.inference_mode():
        assert torch.equal(model(ids), ref(ids))
    onload.restore()
    assert model.proj.weight.data_ptr() != moved_proj
    assert model.embed_tokens.weight.data_ptr() == table_ptr


@linux_only
def test_rows_follow_the_encoder_and_ids_follow_the_table(tmp_path, monkeypatch):
    """The hooks read placement at call time: ids to the table's device, rows to the rest of the encoder's."""
    monkeypatch.delenv(eh.EMBED_HOST_ENV, raising = False)
    model, _ = _encoder(tmp_path)
    # the rest of the encoder already on another device (meta stands in for the card)
    model.proj.weight = torch.nn.Parameter(
        torch.empty(WIDTH, WIDTH, device = "meta"), requires_grad = False
    )
    assert eh.keep_on_host(model)
    seen = {}
    model.embed_tokens.register_forward_hook(
        lambda _m, args, _o: seen.__setitem__("ids", args[0].device)
    )
    with torch.inference_mode():
        out = model.embed_tokens(torch.tensor([[1, 2]]))
    assert seen["ids"].type == "cpu"
    assert out.device.type == "meta"
