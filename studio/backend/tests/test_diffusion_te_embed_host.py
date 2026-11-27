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
def test_rows_go_to_the_named_device_and_ids_follow_the_table(tmp_path, monkeypatch):
    """Rows go to the device ``set_device`` named, never to a neighbouring parameter's; ids go to the table."""
    monkeypatch.delenv(eh.EMBED_HOST_ENV, raising = False)
    model, _ = _encoder(tmp_path)
    assert not eh.set_device(model, "meta")  # not engaged yet
    assert eh.keep_on_host(model)
    seen = {}
    model.embed_tokens.register_forward_hook(
        lambda _m, args, _o: seen.__setitem__("ids", args[0].device)
    )
    with torch.inference_mode():
        assert (
            model.embed_tokens(torch.tensor([[1, 2]])).device.type == "cpu"
        )  # no device named: rows stay
        assert eh.set_device(model, "meta")  # meta stands in for the card
        out = model.embed_tokens(torch.tensor([[1, 2]]))
    assert seen["ids"].type == "cpu"
    assert out.device.type == "meta"
    assert eh.release(model) and eh.TARGET_ATTR not in model.embed_tokens.__dict__


class _StrictLayer(torch.nn.Module):
    """A decoder layer that, like a CUDA kernel, refuses an activation on another device than its weight."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(WIDTH, WIDTH)

    def forward(self, h):
        assert (
            h.device == self.proj.weight.device
        ), f"activation on {h.device}, weight on {self.proj.weight.device}"
        return torch.tanh(self.proj(h))


class _Gemma4Like(torch.nn.Module):
    """The worker's Gemma 4 layout in miniature: model.language_model.{embed_tokens, layers, norm}."""

    def __init__(self, table):
        super().__init__()
        language = torch.nn.Module()
        language.embed_tokens = torch.nn.Embedding(VOCAB, WIDTH)
        language.embed_tokens.weight = torch.nn.Parameter(table, requires_grad = False)
        language.layers = torch.nn.ModuleList(_StrictLayer() for _ in range(3))
        language.norm = torch.nn.LayerNorm(WIDTH)
        inner = torch.nn.Module()
        inner.language_model = language
        self.model = inner
        self.dtype = torch.float32

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def forward(
        self,
        input_ids,
        attention_mask,
        output_hidden_states,
        use_cache = None,
    ):
        from types import SimpleNamespace

        language = self.model.language_model
        h = language.embed_tokens(input_ids)
        states = [h]
        for layer in language.layers:
            h = layer(h) * attention_mask[..., None]
            states.append(h)
        states[-1] = language.norm(h)
        return SimpleNamespace(hidden_states = tuple(states))


def _spy_gather(model, seen):
    """Record, at gather time, the device the rows are sent to and where layer 0 sits; then point the rows at the
    host so the encode can finish on a machine without the card (the device under test is the recorded one)."""

    key = getattr(eh, "TARGET_ATTR", None)

    def spy(module, _args):
        seen["target"] = module.__dict__.get(key) if key else None
        seen["layer0"] = model.model.language_model.layers[0].proj.weight.device
        if seen["target"] is not None:
            module.__dict__[key] = torch.device("cpu")

    return model.get_input_embeddings().register_forward_pre_hook(spy)


@linux_only
@pytest.mark.parametrize("streamed", [True, False], ids = ["layer_streamed", "whole_onload"])
def test_run_encode_names_the_encode_device_for_the_rows(tmp_path, monkeypatch, streamed):
    """Through ``_run_encode`` on a card: under layer streaming layer 0 is still on the host when the table gathers,
    so the rows' device must be the encode's, not a neighbouring parameter's."""
    from types import SimpleNamespace

    monkeypatch.delenv(eh.EMBED_HOST_ENV, raising = False)
    mapped, data = _mapped_table(tmp_path)
    torch.manual_seed(2)
    model = _Gemma4Like(mapped).eval()
    torch.manual_seed(2)
    ref = _Gemma4Like(data.clone()).eval()
    ref.load_state_dict(model.state_dict())
    monkeypatch.setattr(worker, "_discrete_card", lambda _d: True)
    monkeypatch.setattr(worker, "_host_fits", lambda *_a, **_k: True)
    monkeypatch.setattr(worker, "_lookahead_ops", lambda _d: None)
    # the "card" copy: a host clone, so the run needs no GPU; placement is checked through the spy
    monkeypatch.setattr(worker, "_to_device", lambda t, _d: t.clone())
    seen = {}
    _spy_gather(model, seen)
    config = {"device": "cuda:0", "resident": False}
    stream = worker._stream_parts(model) if streamed else None
    assert not streamed or stream is not None
    ids, mask = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[0, 1, 1, 1]])
    inputs = SimpleNamespace(input_ids = ids, attention_mask = mask)
    monkeypatch.setattr(torch.Tensor, "to", _host_only_to(torch.Tensor.to))
    embeds, _mask = worker._run_encode(model, config, inputs, "cuda:0", stream)
    assert config["_embed_host_bytes"] == VOCAB * WIDTH * 4
    assert seen["target"] == torch.device("cuda:0")
    assert seen["layer0"].type == "cpu"
    table = model.get_input_embeddings().weight
    assert table.device.type == "cpu" and eh.file_mapped(table.data)
    with torch.inference_mode():
        assert torch.equal(embeds, worker._pack_hidden_states(ref, ids, mask))


def _host_only_to(real_to):
    """``Tensor.to`` with any CUDA target served on the host (this test machine has no card)."""

    def to(self, *args, **kwargs):
        args = tuple(
            "cpu"
            if (isinstance(a, str) and a.startswith("cuda"))
            or (isinstance(a, torch.device) and a.type == "cuda")
            else a
            for a in args
        )
        if "device" in kwargs and str(kwargs["device"]).startswith("cuda"):
            kwargs["device"] = "cpu"
        return real_to(self, *args, **kwargs)

    return to


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA card")
@pytest.mark.parametrize("streamed", [True, False], ids = ["layer_streamed", "whole_onload"])
def test_run_encode_on_a_real_card(tmp_path, monkeypatch, streamed):
    from types import SimpleNamespace

    if not sys.platform.startswith("linux"):
        pytest.skip("file-mapping check reads /proc")
    monkeypatch.delenv(eh.EMBED_HOST_ENV, raising = False)
    mapped, data = _mapped_table(tmp_path)
    torch.manual_seed(2)
    model = _Gemma4Like(mapped).eval()
    ref = _Gemma4Like(data.clone()).eval()
    ref.load_state_dict(model.state_dict())
    monkeypatch.setattr(worker, "_discrete_card", lambda _d: True)
    monkeypatch.setattr(worker, "_host_fits", lambda *_a, **_k: True)
    ids, mask = torch.tensor([[1, 2, 3, 4]]), torch.tensor([[0, 1, 1, 1]])

    def encode(m, switch):
        monkeypatch.setenv(eh.EMBED_HOST_ENV, switch)
        config = {"device": "cuda:0", "resident": False}
        stream = worker._stream_parts(m) if streamed else None
        inputs = SimpleNamespace(input_ids = ids, attention_mask = mask)
        out, _ = worker._run_encode(m, config, inputs, "cuda:0", stream)
        return out, config

    # the same encode with the switch off, on the same card, is the reference (CPU math would differ in the last bit)
    expected, off = encode(ref, "0")
    embeds, on = encode(model, "1")
    assert not off["_embed_host_bytes"] and on["_embed_host_bytes"] == VOCAB * WIDTH * 4
    assert embeds.device.type == "cpu" and torch.equal(embeds, expected)
