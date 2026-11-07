# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastModel generate() decodes through CUDA graphs by default (generate_scope + plan()), on the host
through the stand-in backend of test_generic_decode_graphs: kill switches, the short-call rule, the
out-of-memory eager redo, the per-call plan the LoRA fold reads, the graph + StaticCache kept between
inference calls (released by training), and the BlockMask graph-argument contract."""

import os
import sys
import types

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)

import unsloth.models._decode_graphs as G
import unsloth.models._generic_decode_graphs as GG
from unsloth.models import vision

from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_generic_decode_graphs import LLAMA, _Host  # noqa: E402

ENV = "UNSLOTH_DECODE_CUDA_GRAPHS"


class _DefaultHost(_Host):
    @staticmethod
    def available():
        return True

    @staticmethod
    def free_bytes(device):
        return 1 << 40


@pytest.fixture
def host(monkeypatch):
    backend = type("Host", (_DefaultHost,), {"graphs": []})
    monkeypatch.setattr(GG, "_BACKEND", backend)
    # The runtime default (NVIDIA CUDA, not Windows), not a programmatic override.
    monkeypatch.setattr(G, "ENABLED", None)
    monkeypatch.setattr(G, "_DEFAULT_OK", True)
    monkeypatch.setattr(G, "_STUDIO_DEFAULT_OK", True)
    monkeypatch.delenv(ENV, raising = False)
    monkeypatch.delenv("UNSLOTH_DECODE_GENERIC_GRAPHS", raising = False)
    monkeypatch.delenv("UNSLOTH_DECODE_GENERIC_GRAPHS_KEEP", raising = False)
    return backend


@pytest.fixture
def llama():
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(LLAMA, dtype = torch.float32).eval()
    model._old_generate = model.generate
    model.generate = types.MethodType(vision.unsloth_base_fast_generate, model)
    return model


def _ids(
    model,
    bsz = 1,
    length = 6,
):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 1000, (bsz, length), generator = g)
    return ids, torch.ones_like(ids)


def _gen(
    model,
    new = 10,
    bsz = 1,
    **kwargs,
):
    ids, mask = _ids(model, bsz)
    return model.generate(
        input_ids = ids,
        attention_mask = mask,
        max_new_tokens = new,
        min_new_tokens = new,
        do_sample = False,
        pad_token_id = 0,
        **kwargs,
    )


def _stats(model):
    return GG.stats(model) or {"captures": 0, "replays": 0, "graphs": 0}


def test_plain_generate_uses_graphs_by_default(host, llama, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    eager = _gen(llama)
    assert _stats(llama)["replays"] == 0
    monkeypatch.delenv(ENV)
    graphed = _gen(llama)
    s = _stats(llama)
    assert s["captures"] == 1 and s["replays"] >= 8, s
    assert torch.equal(eager, graphed)
    assert "forward" not in llama.__dict__


@pytest.mark.parametrize("name", [ENV, "UNSLOTH_DECODE_GENERIC_GRAPHS"])
def test_kill_switches_keep_generate_eager(host, llama, monkeypatch, name):
    monkeypatch.setenv(name, "0")
    _gen(llama)
    assert _stats(llama)["replays"] == 0 and _stats(llama)["captures"] == 0


def test_short_calls_decode_eagerly(host, llama):
    before = GG.PLANS["short call"]
    _gen(llama, new = 3)
    assert GG.PLANS["short call"] == before + 1
    assert _stats(llama)["captures"] == 0


def test_call_plan_follows_the_call(host, llama, monkeypatch):
    seen = []
    real = llama._old_generate

    def recording(*args, **kwargs):
        seen.append(G.call_plan())
        return real(*args, **kwargs)

    llama._old_generate = recording
    assert G.call_plan() is None
    _gen(llama)
    _gen(llama, new = 2)  # one decode step: never a graph, kept or not
    monkeypatch.setenv(ENV, "0")
    _gen(llama)
    assert seen == ["graph", "eager", "eager"]
    assert G.call_plan() is None


def test_out_of_memory_redoes_the_call_eagerly_without_the_wrapper(host, llama, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    eager = _gen(llama)
    monkeypatch.delenv(ENV)
    calls = []
    real = GG.decode_step

    def oom_once(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
        return real(*args, **kwargs)

    monkeypatch.setattr(GG, "decode_step", oom_once)
    # The redo runs after the handler: its traceback would keep the failed call's cache alive.
    handling = []
    real_redo = GG.redo_eagerly

    def redo(model):
        handling.append(sys.exc_info()[0])
        return real_redo(model)

    monkeypatch.setattr(GG, "redo_eagerly", redo)
    redos = GG.EAGER_REDOS
    with pytest.warns(UserWarning, match = "out of GPU memory"):
        out = _gen(llama)
    assert torch.equal(out, eager)
    assert GG.EAGER_REDOS == redos + 1
    assert handling == [None]
    # The redo ran without _GraphDecodeSteps: decode_step saw only the failed attempt's steps.
    assert len(calls) == 2
    assert "forward" not in llama.__dict__
    assert GG.kept_cache(llama) is None


def test_out_of_memory_with_a_streamer_reraises(host, llama, monkeypatch):
    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")

    monkeypatch.setattr(GG, "decode_step", oom)

    class Streamer:
        def put(self, value):
            pass

        def end(self):
            pass

    with pytest.raises(torch.cuda.OutOfMemoryError):
        _gen(llama, streamer = Streamer())
    assert "forward" not in llama.__dict__


def test_kept_graph_replays_the_next_call_and_training_releases_it(host, llama, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    eager = _gen(llama)
    monkeypatch.delenv(ENV)
    first = _gen(llama)
    s1 = _stats(llama)
    assert s1["captures"] == 1 and s1["graphs"] == 1, s1
    kept = GG.kept_cache(llama)
    assert kept is not None
    second = _gen(llama)
    s2 = _stats(llama)
    # Same cache tensors, same graph: no capture, every decode step replays (from the first one).
    assert s2["captures"] == 1, s2
    assert s2["replays"] - s1["replays"] == 9, (s1, s2)
    assert GG.kept_cache(llama) is kept
    assert torch.equal(first, eager) and torch.equal(second, eager)
    # A short call replays the kept graph too.
    _gen(llama, new = 4)
    assert _stats(llama)["captures"] == 1
    # Training forward without for_training() frees the graph and the cache.
    llama.train()
    ids, _ = _ids(llama)
    llama(input_ids = ids)
    assert GG.kept_cache(llama) is None and _stats(llama)["graphs"] == 0
    assert "_cache" not in llama.__dict__
    llama.eval()


def test_other_batch_or_returned_cache_drops_the_kept_graph(host, llama):
    _gen(llama)
    assert GG.kept_cache(llama) is not None
    _gen(llama, bsz = 2)
    s = _stats(llama)
    assert s["captures"] == 2, s
    # A returned cache belongs to the caller: never kept, never reset by a later call.
    out = _gen(llama, return_dict_in_generate = True)
    assert out.past_key_values is not None
    assert GG.kept_cache(llama) is None


def test_weights_changed_revokes_the_kept_graph(host, llama):
    _gen(llama)
    assert _stats(llama)["captures"] == 1
    G.weights_changed()
    _gen(llama)
    assert _stats(llama)["captures"] == 2


def test_release_for_training_drops_the_kept_cache(host, llama):
    _gen(llama)
    assert GG.kept_cache(llama) is not None
    G.release_for_training(llama)
    assert GG.kept_cache(llama) is None and "_cache" not in llama.__dict__


def _block_mask(offset, contract = True):
    from torch.nn.attention.flex_attention import BlockMask, _DEFAULT_SPARSE_BLOCK_SIZE as BLOCK

    def rebuild(tensors):
        (off,) = tensors

        def mask_mod(b, h, q_idx, kv_idx):
            return q_idx + off >= kv_idx

        return mask_mod

    n = 2
    mask = BlockMask.from_kv_blocks(
        torch.ones(1, 1, 1, dtype = torch.int32),
        torch.arange(n, dtype = torch.int32).view(1, 1, 1, n),
        BLOCK_SIZE = BLOCK,
        mask_mod = rebuild((offset,)),
        seq_lengths = (1, n * BLOCK),
    )
    if contract:
        mask._unsloth_graph_closure = ((offset,), rebuild)
    return mask


def test_block_mask_contract_round_trip(host):
    offset = torch.tensor(3)
    mask = _block_mask(offset)
    leaves = []
    spec = GG._spec({"full_attention": mask}, leaves, None)
    again = []
    assert GG._spec({"full_attention": _block_mask(torch.tensor(7))}, again, None) == spec
    assert any(leaf is offset for leaf in leaves)

    def at(t):
        return next(i for i, leaf in enumerate(leaves) if leaf is t)

    buffers = [leaf.clone() for leaf in leaves]
    built = GG._build({"full_attention": mask}, iter(buffers), None)["full_attention"]
    assert built is not mask and built.kv_num_blocks is buffers[at(mask.kv_num_blocks)]
    static = built._unsloth_graph_closure[0][0]
    assert static is buffers[at(offset)]
    q, k = torch.zeros((), dtype = torch.long), torch.tensor(5)
    assert not bool(built.mask_mod(0, 0, q, k))
    static.fill_(5)  # a refill reaches the rebuilt mask_mod
    assert bool(built.mask_mod(0, 0, q, k))


def test_block_mask_without_the_contract_is_refused(host):
    with pytest.raises(GG._Refuse, match = "argument BlockMask"):
        GG._spec(_block_mask(torch.tensor(3), contract = False), [], None)


def test_placeholder_masking_models_embed_outside_the_graph():
    """Muse-Glimmer / Muse-Spark / Kimi-K2.5 zero placeholder ids with ids[mask] = 0 (a boolean index_put_, a
    host sync): their decode step gets the same embeddings computed before the graph, from the same ops."""

    class Model(torch.nn.Module):
        def __init__(self, model_type):
            super().__init__()
            self.config = types.SimpleNamespace(
                model_type = model_type, image_token_id = 5, video_token_id = 6
            )
            self.embed = torch.nn.Embedding(10, 4)

        def get_input_embeddings(self):
            return self.embed

    ids = torch.tensor([[5], [3], [6]])
    model = Model("muse_glimmer")
    out = GG._placeholder_embeds(
        model, {"input_ids": ids, "inputs_embeds": None, "use_cache": True}
    )
    llm = ids.clone()
    llm[(ids == 5) | (ids == 6)] = 0  # what the model's forward does
    assert out["input_ids"] is None and out["use_cache"] is True
    assert torch.equal(out["inputs_embeds"], model.embed(llm))
    # Other models, media steps and an explicit inputs_embeds are left alone.
    kwargs = {"input_ids": ids}
    assert GG._placeholder_embeds(Model("gemma4"), kwargs) is kwargs
    media = {"input_ids": ids, "pixel_values": torch.ones(1)}
    assert GG._placeholder_embeds(model, media) is media


def _hooked(model, hook):
    """What accelerate's add_hook_to_module leaves on a module (without moving its parameters)."""
    import functools

    def new_forward(module, *args, **kwargs):
        args, kwargs = module._hf_hook.pre_forward(module, *args, **kwargs)
        return module._hf_hook.post_forward(module, module._old_forward(*args, **kwargs))

    old = model.forward
    model._old_forward = old
    model._hf_hook = hook
    model.forward = functools.update_wrapper(functools.partial(new_forward, model), old)
    return model


def test_a_single_device_accelerate_hook_keeps_graphs_eligible(monkeypatch):
    """A bitsandbytes model placed on cuda:1 gets accelerate's AlignDevicesHook (force_hooks, vision.py
    _attach_bnb_multidevice_hooks): an instance forward plus _hf_hook, which wanted() used to refuse, so
    no decode step of a model on a second card was ever graphed."""
    from accelerate.hooks import AlignDevicesHook, ModelHook

    class OnCard1(torch.nn.Linear):
        @property
        def device(self):
            return torch.device("cuda", 1)

    monkeypatch.setattr(G, "ENABLED", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.delenv("UNSLOTH_DECODE_GENERIC_GRAPHS", raising = False)
    assert GG.wanted(
        _hooked(OnCard1(2, 2), AlignDevicesHook(execution_device = 1, io_same_device = True))
    )
    assert GG.wanted(_hooked(OnCard1(2, 2), AlignDevicesHook(execution_device = "cuda:1")))
    # Hooks that move weights or run elsewhere stay eager.
    assert not GG.wanted(_hooked(OnCard1(2, 2), AlignDevicesHook(execution_device = 0)))
    assert not GG.wanted(_hooked(OnCard1(2, 2), AlignDevicesHook(execution_device = 1, offload = True)))
    assert not GG.wanted(_hooked(OnCard1(2, 2), ModelHook()))


def test_generate_through_a_single_device_hook_uses_graphs(host, llama, monkeypatch):
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

    monkeypatch.setenv(ENV, "0")
    eager = _gen(llama)
    monkeypatch.delenv(ENV)
    add_hook_to_module(llama, AlignDevicesHook(execution_device = "cpu", io_same_device = True))
    hooked = llama.__dict__["forward"]
    graphed = _gen(llama)
    s = _stats(llama)
    assert s["captures"] == 1 and s["replays"] >= 8, s
    assert torch.equal(eager, graphed)
    # The hook's forward is put back after the call.
    assert llama.__dict__["forward"] is hooked


class _Muse(torch.nn.Module):
    """A Muse-Glimmer-like decoder: the embedding is trainable and may answer on another device."""

    def __init__(self, answer_on = None):
        super().__init__()
        self.config = types.SimpleNamespace(
            model_type = "muse_glimmer", image_token_id = 5, video_token_id = 6
        )
        self.embed = torch.nn.Embedding(10, 4)
        self.answer_on = answer_on

    def get_input_embeddings(self):
        if self.answer_on is None:
            return self.embed
        return lambda ids: torch.zeros(*ids.shape, 4, device = self.answer_on)


def test_placeholder_embeds_never_track_grad_and_pass_the_argument_check(host):
    model = _Muse()
    assert model.embed.weight.requires_grad
    ids = torch.tensor([[5], [3]])
    with torch.enable_grad():
        out = GG._placeholder_embeds(model, {"input_ids": ids})
    embeds = out["inputs_embeds"]
    assert not embeds.requires_grad
    leaves = []
    GG._spec(out, leaves, None)  # not "tensor argument requires grad"
    assert any(leaf is embeds for leaf in leaves)


def test_placeholder_embeds_land_on_the_decode_step_device():
    # The lookup answers on the host while the ids live elsewhere (meta stands in for the card).
    model = _Muse(answer_on = "cpu")
    ids = torch.tensor([[5], [3]], device = "meta")
    out = GG._placeholder_embeds(model, {"input_ids": ids})
    assert out["inputs_embeds"].device == ids.device


def test_refusal_reasons_name_the_argument(host, llama, monkeypatch):
    real = GG._mrope_positions

    def with_grad_tensor(model, kwargs):
        return {**real(model, kwargs), "extra": torch.zeros(1, requires_grad = True)}

    import functools

    monkeypatch.setattr(GG, "_mrope_positions", with_grad_tensor)
    cls_forward = type(llama).forward

    @functools.wraps(cls_forward)
    def forward(
        self,
        *a,
        extra = None,
        **k,
    ):
        return cls_forward(self, *a, **k)

    monkeypatch.setattr(type(llama), "forward", forward)
    _gen(llama)
    assert _stats(llama)["refusals"] == {"tensor argument requires grad (extra)": 9}, _stats(llama)


class _HostEmbedding(torch.nn.Module):
    """Unsloth's offloaded input embedding as the decode step sees it: the weight lives off the compute device
    (meta stands in for host RAM) behind hooks that return the rows on the decoder's device. Records whether
    each lookup ran inside a graph capture or replay."""

    def __init__(self, real, inside):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty_like(real.weight, device = "meta"), requires_grad = False
        )
        self.table = real.weight.detach().clone()  # a plain attribute: the lookup's data
        self.inside = inside
        self.calls = []
        self._unsloth_offload_hooks_installed = True

    def forward(self, ids):
        self.calls.append(self.inside[0])
        return torch.nn.functional.embedding(ids, self.table)


def _offload(
    host,
    llama,
    monkeypatch,
    extra = False,
):
    inside = [False]
    record = host.record.__func__

    def tracked(fn):
        def run():
            inside[0] = True
            try:
                return fn()
            finally:
                inside[0] = False

        return run

    monkeypatch.setattr(
        host,
        "record",
        classmethod(lambda cls, pool, fn, cache: record(cls, pool, tracked(fn), cache)),
    )
    embedding = _HostEmbedding(llama.get_input_embeddings(), inside)
    llama.set_input_embeddings(embedding)
    # As `_pin_device_to_decoder`: model.device is the decoder's, not the offloaded table's.
    monkeypatch.setattr(type(llama), "device", property(lambda self: torch.device("cpu")))
    if extra:
        llama.model.norm.register_parameter(
            "stray", torch.nn.Parameter(torch.zeros(1, device = "meta"), requires_grad = False)
        )
    return embedding


def test_offloaded_input_embedding_is_looked_up_outside_the_graph(host, llama, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    eager = _gen(llama)
    monkeypatch.delenv(ENV)
    embedding = _offload(host, llama, monkeypatch)
    graphed = _gen(llama)
    s = _stats(llama)
    assert s["captures"] == 1 and s["replays"] >= 8, s
    assert torch.equal(eager, graphed)
    # Every decode step looked its rows up eagerly; none inside a capture or replay.
    assert embedding.calls and not any(embedding.calls)


def test_offloaded_input_embedding_swap_recaptures(host, llama, monkeypatch):
    embedding = _offload(host, llama, monkeypatch)
    first = GG._fingerprint(llama)
    assert first is not None
    embedding.weight = torch.nn.Parameter(torch.empty_like(embedding.weight), requires_grad = False)
    assert GG._fingerprint(llama) not in (None, first)


def test_another_offloaded_parameter_is_still_refused(host, llama, monkeypatch):
    _offload(host, llama, monkeypatch, extra = True)
    assert GG._fingerprint(llama) is None
    _gen(llama)
    s = _stats(llama)
    assert s["replays"] == 0 and "not one CUDA device / adapters" in s["refusals"], s


def test_an_unhooked_off_device_embedding_is_still_refused(host, llama, monkeypatch):
    embedding = _offload(host, llama, monkeypatch)
    embedding._unsloth_offload_hooks_installed = False
    assert GG._offloaded_input_embedding(llama) is None
    assert GG._fingerprint(llama) is None
