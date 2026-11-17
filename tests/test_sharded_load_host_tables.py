# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Sharded loading reads whole 16-bit tables (embedding, lm_head) in row slices instead of one whole-table
host copy, and under tensor parallel reads a large frozen untied input embedding straight into host memory
(Qwen3.8-27B 4-bit on 2x T4: the 2.37 GiB table never touches the GPU). CPU only; meta stands in for the GPU."""

import os
import sys
import types

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import tpn  # noqa: E402

transformers = pytest.importorskip("transformers")
pytest.importorskip("peft")
safetensors_torch = pytest.importorskip("safetensors.torch")

VOCAB = 512
HIDDEN = 32
ROWS = 1003  # not a multiple of the slice rows
WIDTH = 64
SLICE = 8192  # bytes: 64 bf16 rows of 64


def _qwen3(
    tied = False,
    seed = 0,
    meta = False,
):
    config = transformers.Qwen3Config(
        vocab_size = VOCAB,
        hidden_size = HIDDEN,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        tie_word_embeddings = tied,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(seed)
    if meta:
        with torch.device("meta"):
            return transformers.Qwen3ForCausalLM(config)
    return transformers.Qwen3ForCausalLM(config).float()


def _peft(model):
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(1)
    lora = LoraConfig(r = 4, lora_alpha = 8, target_modules = "all-linear", init_lora_weights = False)
    return get_peft_model(model, lora).eval()


class _Tables(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Embedding(ROWS, WIDTH)
        self.norm = nn.Parameter(torch.ones(WIDTH))


@pytest.fixture()
def ckpt(tmp_path):
    g = torch.Generator().manual_seed(0)
    tensors = {
        "table.weight": torch.randn(ROWS, WIDTH, generator = g).to(torch.bfloat16),
        "norm": torch.randn(WIDTH, generator = g).to(torch.bfloat16),
    }
    safetensors_torch.save_file(tensors, str(tmp_path / "model.safetensors"))
    return str(tmp_path), tensors


def _meta_tables():
    with torch.device("meta"):
        return _Tables().to(torch.float16)


@pytest.fixture()
def reads(monkeypatch):
    """{key: [bytes of every single read]} through _SliceReader's safetensors handles."""
    seen = {}
    original = tpn._SliceReader._open

    class _View:
        def __init__(self, view, key):
            self.view, self.key = view, key

        def get_shape(self):
            return self.view.get_shape()

        def get_dtype(self):
            return self.view.get_dtype()

        def __getitem__(self, index):
            t = self.view[index]
            seen.setdefault(self.key, []).append(t.numel() * t.element_size())
            return t

    class _Handle:
        def __init__(self, handle):
            self.handle = handle

        def keys(self):
            return self.handle.keys()

        def get_slice(self, key):
            return _View(self.handle.get_slice(key), key)

        def get_tensor(self, key):
            t = self.handle.get_tensor(key)
            seen.setdefault(key, []).append(t.numel() * t.element_size())
            return t

    def _open(self, key):
        return _Handle(original(self, key))

    monkeypatch.setattr(tpn._SliceReader, "_open", _open)
    monkeypatch.setattr(tpn, "_READ_SLICE_BYTES", SLICE, raising = False)
    return seen


def test_whole_tables_are_read_in_slices(ckpt, reads):
    path, tensors = ckpt
    model = _meta_tables()
    tpn._fill_from_checkpoint(model, path, {}, torch.device("cpu"), lambda old, reader, key: None)
    assert max(reads["table.weight"]) <= SLICE, reads[
        "table.weight"
    ]  # never one whole-table host copy
    assert sum(reads["table.weight"]) == ROWS * WIDTH * 2  # every row read once
    for key, value in tensors.items():
        got = model.get_parameter(key)
        assert got.dtype == torch.float16 and got.device.type == "cpu"
        assert torch.equal(got, value.to(torch.float16)), key


def test_host_tables_land_in_host_memory(ckpt, reads):
    """`host` ids go to the CPU, the rest to the load device (meta stands in for the GPU)."""
    path, tensors = ckpt
    model = _meta_tables()
    tpn._fill_from_checkpoint(
        model,
        path,
        {},
        torch.device("meta"),
        lambda old, reader, key: None,
        host = {id(model.table.weight)},
    )
    assert model.table.weight.device.type == "cpu"
    assert torch.equal(model.table.weight, tensors["table.weight"].to(torch.float16))
    assert model.norm.device.type == "meta"
    assert max(reads["table.weight"]) <= SLICE


@pytest.mark.parametrize(
    "start, stop", [(0, ROWS), (5, 700), (999, ROWS), (64, 65), (0, 64), (3, 3)]
)
@pytest.mark.parametrize("slice_bytes", [SLICE, 1 << 30])
def test_rows_to_matches_rows(ckpt, monkeypatch, start, stop, slice_bytes):
    path, _ = ckpt
    monkeypatch.setattr(tpn, "_READ_SLICE_BYTES", slice_bytes, raising = False)
    reader = tpn._SliceReader(tpn._checkpoint_index(path, {}))
    want = reader.rows("table.weight", start, stop).to(torch.float16)
    got = reader.rows_to("table.weight", start, stop, torch.float16, torch.device("cpu"))
    assert got.dtype == torch.float16 and tuple(got.shape) == tuple(want.shape)
    assert torch.equal(got, want)


def _fake_cuda(monkeypatch, total):
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda *a, **k: types.SimpleNamespace(total_memory = total),
    )


def test_host_load_selection(monkeypatch):
    """Judged on the meta model before reading: untied, single-owner, at least 15% of the GPU (forced by
    UNSLOTH_TP_EMBED_ON_CPU=1, never with =0), CUDA mesh and device only."""
    monkeypatch.delenv("UNSLOTH_TP_EMBED_ON_CPU", raising = False)
    cuda_mesh = types.SimpleNamespace(device_type = "cuda", size = lambda: 1)
    cuda = torch.device("cuda", 0)
    model = _qwen3(meta = True)
    embed = model.get_input_embeddings()
    nbytes = embed.weight.numel() * embed.weight.element_size()
    _fake_cuda(monkeypatch, int(nbytes / 0.15 * 0.9))  # table is ~16.7% of the card
    assert tpn._host_load_embedding(model, cuda_mesh, cuda) is embed
    assert tpn._host_load_embedding(_qwen3(tied = True, meta = True), cuda_mesh, cuda) is None
    cpu_mesh = types.SimpleNamespace(device_type = "cpu", size = lambda: 1)
    assert tpn._host_load_embedding(model, cpu_mesh, torch.device("cpu")) is None
    assert tpn._host_load_embedding(model, cuda_mesh, torch.device("cpu")) is None
    _fake_cuda(monkeypatch, int(nbytes / 0.15 * 2))  # ~7.5%: below the size rule
    assert tpn._host_load_embedding(model, cuda_mesh, cuda) is None
    monkeypatch.setenv("UNSLOTH_TP_EMBED_ON_CPU", "1")
    assert tpn._host_load_embedding(model, cuda_mesh, cuda) is embed
    monkeypatch.setenv("UNSLOTH_TP_EMBED_ON_CPU", "0")
    _fake_cuda(monkeypatch, nbytes)
    assert tpn._host_load_embedding(model, cuda_mesh, cuda) is None


def _host_loaded_peft(pinned = True):
    """PEFT tiny Qwen3: everything on meta (the GPU stand-in) but the host-read input embedding."""
    model = _peft(_qwen3())
    model.to("meta")
    base = model.get_base_model()
    embed = model.get_input_embeddings()
    embed.to_empty(device = "cpu")
    with torch.no_grad():
        embed.weight.normal_()
    embed.weight.requires_grad_(False)
    tpn._mark_host_loaded(base, embed)
    if pinned:  # what vision._pin_device_to_decoder does (unimportable in this stub package)
        base.__dict__[tpn._HOST_PINNED] = True
        base._unsloth_embedding_offloaded = True
    return model, base, embed


def test_compute_device_skips_the_host_read_table():
    model, _, embed = _host_loaded_peft()
    assert next(model.parameters()) is embed.weight
    assert tpn._compute_device(model).type == "meta"


def test_unserved_host_read_table_returns_to_the_gpu():
    model, base, embed = _host_loaded_peft()
    assert tpn._return_host_loaded(model, None) == 1
    assert embed.weight.device.type == "meta"
    assert not embed.__dict__.get(tpn._HOST_LOADED, False)
    assert not base._unsloth_embedding_offloaded
    assert tpn._HOST_MODEL not in base.__dict__ and tpn._HOST_PINNED not in base.__dict__
    assert tpn._return_host_loaded(model, None) == 0  # idempotent


def test_served_host_read_table_stays_on_the_host():
    model, base, embed = _host_loaded_peft()
    assert tpn._return_host_loaded(model, embed) == 0
    assert embed.weight.device.type == "cpu"
    assert base._unsloth_embedding_offloaded


def test_rebuilt_weight_still_returns_to_the_gpu():
    """A pass that rebuilds the weight (or the module) drops the marks; the model-level mark still finds it."""
    model, _, embed = _host_loaded_peft()
    embed.__dict__.pop(tpn._HOST_LOADED, None)
    embed.weight = nn.Parameter(embed.weight.detach().clone(), requires_grad = False)
    assert not tpn._is_host_loaded(embed)
    assert tpn._compute_device(model).type == "meta"
    assert tpn._return_host_loaded(model, None) == 1
    assert embed.weight.device.type == "meta"


def test_candidate_accepts_the_host_read_table(monkeypatch):
    monkeypatch.delenv("UNSLOTH_TP_EMBED_ON_CPU", raising = False)
    model, _, embed = _host_loaded_peft()
    nbytes = embed.weight.numel() * embed.weight.element_size()
    monkeypatch.setattr(tpn, "_host_table_home", lambda *a, **k: torch.device("cuda", 0))
    _fake_cuda(monkeypatch, int(nbytes / 0.15 * 0.9))
    assert tpn._host_embedding_candidate(model, None, 0.15) is embed
    embed.weight.requires_grad_(True)  # trained (modules_to_save): stays out
    assert tpn._host_embedding_candidate(model, None, 0.15) is None
    embed.weight.requires_grad_(False)
    _fake_cuda(monkeypatch, int(nbytes / 0.15 * 2))
    assert tpn._host_embedding_candidate(model, None, 0.15) is None


def test_sharded_from_pretrained_forwards_materialize_kwargs(monkeypatch):
    """tp_env asks for the host read through state["materialize_kwargs"]; the loader passes them on."""
    from unsloth.models import bnb_sharded_load

    monkeypatch.setattr(
        bnb_sharded_load,
        "checkpoint_quantization",
        lambda cls, name, kwargs: (types.SimpleNamespace(), None),
    )
    model = nn.Linear(2, 2)
    model.config = types.SimpleNamespace()

    def original(cls, name, *args, **kwargs):
        assert kwargs["device_map"] == "meta"
        return model

    seen = {}

    def materialize(*args, **kwargs):
        seen.update(kwargs)

    state = {
        "mesh": types.SimpleNamespace(ndim = 1, device_type = "cpu"),
        "plan": None,
        "materialize": materialize,
        "materialize_kwargs": {"host_embed": True},
        "active": False,
        "models": [],
    }
    assert tpn._from_pretrained_sharded(original, object, "x", (), {}, state) is model
    assert seen == {"host_embed": True}
    assert state["models"] == [model]


def test_tp_env_requests_the_host_read_except_full_finetuning():
    source = open(os.path.join(os.path.dirname(tpn.__file__), "tp_env.py"), encoding = "utf-8").read()
    assert '"materialize_kwargs": {"host_embed": host_embed}' in source
    assert 'host_embed = not values.get("full_finetuning")' in source


def _host_read_bf16_model():
    model = _qwen3().to(torch.bfloat16)
    embed = model.get_input_embeddings()
    tpn._mark_host_loaded(model, embed)
    return model, embed


@pytest.mark.parametrize("forced", [True, False])
def test_host_read_table_is_cast_in_place(monkeypatch, forced):
    """The dtype pass's host-side module.to() held a second copy of the table: it is cast in place first,
    in slices, to the dtype that pass gives it (same values as .to())."""
    monkeypatch.setattr(tpn, "_READ_SLICE_BYTES", 4096, raising = False)
    model, embed = _host_read_bf16_model()
    want = embed.weight.detach().to(torch.float16)
    ptr = embed.weight.data_ptr()
    assert tpn.cast_host_read_tables(model, forced, torch.float16) == 1
    assert embed.weight.dtype == torch.float16
    assert embed.weight.data_ptr() == ptr  # no new storage
    assert torch.equal(embed.weight.detach(), want)
    embed.to(torch.float16)  # the pass's module.to() is then a no-op
    assert embed.weight.data_ptr() == ptr
    assert tpn.cast_host_read_tables(model, forced, torch.float16) == 0  # idempotent


def test_host_read_table_cast_respects_pins_and_bf16_runs():
    model, embed = _host_read_bf16_model()
    assert tpn.cast_host_read_tables(model, False, torch.bfloat16) == 0  # bf16 run: nothing to cast
    embed._pre_set_compute_dtype = torch.float32
    assert tpn.cast_host_read_tables(model, False, torch.float16) == 0  # a module pin decides
    assert embed.weight.dtype == torch.bfloat16
