# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 (gloo CPU world of 2): `generate` gathers one decoder block at a time.

Gathering every unit at once OOMed a 12B 4-bit model after training on 2x 16 GB GPUs. With
UNSLOTH_FSDP2_GENERATE_GATHER=block only the root stays gathered; each block is gathered via its
forward or the decode loop's enter / leave. Greedy tokens must equal one process and gathered
bytes must stay about one block above the root."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

REPO_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
PROMPT = [[11, 23, 9, 77, 102, 5]]
NEW_TOKENS = 6
LAYERS = 6


def _gathered_bytes(model):
    from torch.distributed.tensor import DTensor
    return sum(
        p.numel() * p.element_size() for p in model.parameters() if not isinstance(p, DTensor)
    )


class _Peak:
    """Largest number of gathered (non-DTensor) parameter bytes seen at any leaf module call."""

    def __init__(self, model):
        self.model = model
        self.peak = 0
        self.handles = [
            m.register_forward_pre_hook(self._sample)
            for m in model.modules()
            if not list(m.children())
        ]

    def _sample(self, *_):
        self.peak = max(self.peak, _gathered_bytes(self.model))

    def close(self):
        for h in self.handles:
            h.remove()


# Decode reads block weights directly, never calling the block, with block-swap enter/leave.
class _Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up = torch.nn.Linear(d, 4 * d)
        self.down = torch.nn.Linear(4 * d, d)

    def forward(self, x):
        return x + self.down(torch.tanh(self.up(x)))


class _LoopNet(torch.nn.Module):
    def __init__(
        self,
        vocab = 32,
        d = 16,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, d)
        self.layers = torch.nn.ModuleList([_Block(d) for _ in range(LAYERS)])
        self.head = torch.nn.Linear(d, vocab, bias = False)

    def forward(self, ids):  # prefill: the blocks are called (FSDP2's hooks gather them)
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)

    def _decode_step(self, ids):  # decode: weights read directly
        x = self.embed(ids)
        swap = getattr(self.layers, "_unsloth_block_swap", None)
        for idx, layer in enumerate(self.layers):
            if swap is not None:
                swap.enter(idx)
            peak = getattr(self, "_peak", None)
            if peak is not None:
                peak._sample()
            h = torch.tanh(x @ layer.up.weight.t() + layer.up.bias)
            x = x + h @ layer.down.weight.t() + layer.down.bias
            if swap is not None:
                swap.leave(idx)
        return self.head(x)

    def generate(self, ids, new_tokens):
        out = ids
        logits = self(ids)[:, -1]
        for _ in range(new_tokens):
            nxt = logits.argmax(-1, keepdim = True)
            out = torch.cat([out, nxt], dim = 1)
            logits = self._decode_step(nxt)[:, -1]
        return out[:, ids.shape[1] :]


def _hf_model():
    import transformers

    config = transformers.AutoConfig.from_pretrained(REPO_ID)
    config.num_hidden_layers = LAYERS
    config.vocab_size = 128
    config.hidden_size = 32
    config.intermediate_size = 128
    config.tie_word_embeddings = False
    config.initializer_range = 0.5  # varied greedy tokens
    if hasattr(config, "layer_types") and config.layer_types is not None:
        config.layer_types = ["full_attention"] * LAYERS
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    model.generation_config.pad_token_id = config.eos_token_id
    model.eval()
    return model


def _hf_greedy(model):
    ids = torch.tensor(PROMPT)
    out = model.generate(
        input_ids = ids,
        attention_mask = torch.ones_like(ids),
        max_new_tokens = NEW_TOKENS,
        min_new_tokens = NEW_TOKENS,
        do_sample = False,
    )
    return out[0, ids.shape[1] :].tolist()


def _loop_greedy(model):
    return model.generate(torch.tensor(PROMPT) % 32, NEW_TOKENS)[0].tolist()


# VLM-shaped: ranks may generate from different modalities (image vs text), so the vision tower is
# gathered up front, never streamed; a streamed block would all-gather on rank 0 only.
class _Tower(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block(d) for _ in range(2)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class _Language(torch.nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, d)
        self.layers = torch.nn.ModuleList([_Block(d) for _ in range(LAYERS)])


class _VlmInner(torch.nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.visual = _Tower(d)
        self.language_model = _Language(vocab, d)


class _VlmNet(torch.nn.Module):
    def __init__(
        self,
        vocab = 32,
        d = 16,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.model = _VlmInner(vocab, d)
        self.head = torch.nn.Linear(d, vocab, bias = False)

    def _step(self, x):
        lm = self.model.language_model
        swap = getattr(lm.layers, "_unsloth_block_swap", None)
        for idx, layer in enumerate(lm.layers):
            if swap is not None:
                swap.enter(idx)
            h = torch.tanh(x @ layer.up.weight.t() + layer.up.bias)
            x = x + h @ layer.down.weight.t() + layer.down.bias
            if swap is not None:
                swap.leave(idx)
        return self.head(x)

    def generate(
        self,
        ids,
        new_tokens,
        image = None,
    ):
        from torch.distributed.tensor import DTensor

        self._tower_gathered = all(
            not isinstance(p, DTensor) for p in self.model.visual.parameters()
        )
        self._tower_streamed = (
            self.model.visual.blocks.__dict__.get("_unsloth_block_swap", None) is not None
        )
        x = self.model.language_model.embed(ids)
        if image is not None:
            x = x + 4 * self.model.visual(image).mean(dim = 1, keepdim = True)
        out = ids
        logits = self._step(x)[:, -1]
        for _ in range(new_tokens):
            nxt = logits.argmax(-1, keepdim = True)
            out = torch.cat([out, nxt], dim = 1)
            logits = self._step(self.model.language_model.embed(nxt))[:, -1]
        return out[:, ids.shape[1] :]


def _vlm_inputs(rank):
    ids = torch.tensor(PROMPT) % 32
    image = torch.linspace(-3, 3, 3 * 16).reshape(1, 3, 16) if rank == 0 else None
    return ids, image


def _vlm_reference(rank):
    with torch.no_grad():
        ids, image = _vlm_inputs(rank)
        return _VlmNet().generate(ids, NEW_TOKENS, image = image)[0].tolist()


def _vlm_case(native, rank):
    model = _VlmNet()
    tower = list(model.model.visual.blocks)
    layers = list(model.model.language_model.layers)
    native.shard_model(
        model,
        native.FSDP2Config(
            device = "cpu", units = tower + layers, fit_layout = False, param_dtype = torch.float32
        ),
    )
    ids, image = _vlm_inputs(rank)
    res = {}
    try:
        res["tokens"] = model.generate(ids, NEW_TOKENS, image = image)[0].tolist()
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    res["tower_gathered"] = getattr(model, "_tower_gathered", None)
    res["tower_streamed"] = getattr(model, "_tower_streamed", None)
    return res


def _reference():
    with torch.no_grad():
        return {"hf": _hf_greedy(_hf_model()), "loop": _loop_greedy(_LoopNet())}


def _bytes(module):
    return sum(p.numel() * p.element_size() for p in module.parameters())


def _case(native, name):
    from torch.distributed.tensor import DTensor

    if name == "hf":
        model = _hf_model()
        layers = model.model.layers
    else:
        model = _LoopNet()
        layers = model.layers
    total = _bytes(model)
    block = max(_bytes(layer) for layer in layers)
    root = total - sum(_bytes(layer) for layer in layers)
    native.shard_model(
        model,
        native.FSDP2Config(
            device = "cpu", units = list(layers), fit_layout = False, param_dtype = torch.float32
        ),
    )
    peak = _Peak(model)
    model._peak = peak
    res = {"total": total, "root": root, "block": block}
    try:
        res["tokens"] = _hf_greedy(model) if name == "hf" else _loop_greedy(model)
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        peak.close()
        del model._peak
    res["peak"] = peak.peak
    res["sharded_after"] = all(isinstance(p, DTensor) for p in model.parameters())
    res["swap_cleared"] = layers.__dict__.get("_unsloth_block_swap", None) is None
    return res


def _worker(
    rank,
    world,
    port,
    out,
    mode,
    vlm = False,
):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    if mode is None:
        os.environ.pop("UNSLOTH_FSDP2_GENERATE_GATHER", None)
    else:
        os.environ["UNSLOTH_FSDP2_GENERATE_GATHER"] = mode
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    try:
        native = _load_native()
        if vlm:
            out.put((rank, {"vlm": _vlm_case(native, rank)}))
        else:
            out.put((rank, {name: _case(native, name) for name in ("hf", "loop")}))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _run(
    mode,
    timeout = 240,
    vlm = False,
):
    import torch.multiprocessing as mp

    out = mp.get_context("spawn").Queue()
    procs = mp.start_processes(
        _worker,
        args = (2, _free_port(), out, mode, vlm),
        nprocs = 2,
        join = False,
        start_method = "spawn",
    )
    deadline = time.time() + timeout
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail(f"the ranks did not finish within {timeout} s (hung collective)")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("GENERATE_STREAM", mode, rank, r)
        assert "crash" not in r, r["crash"]
    return results


def test_generate_gathers_one_block_at_a_time():
    pytest.importorskip("transformers")
    want = _reference()
    for rank, r in _run("block").items():
        for name in ("hf", "loop"):
            got = r[name]
            assert "error" not in got, (rank, name, got)
            assert got["tokens"] == want[name], (rank, name, got, want[name])
            assert got["sharded_after"] and got["swap_cleared"], (rank, name, got)
            # The root plus about one block (FSDP2 may prefetch the next), never the whole model.
            assert got["peak"] <= got["root"] + 2 * got["block"], (rank, name, got)
            assert got["peak"] < got["total"], (rank, name, got)


def test_generate_default_on_host_keeps_tokens():
    pytest.importorskip("transformers")
    want = _reference()
    for rank, r in _run(None).items():
        for name in ("hf", "loop"):
            got = r[name]
            assert "error" not in got, (rank, name, got)
            assert got["tokens"] == want[name], (rank, name, got, want[name])
            assert got["sharded_after"] and got["swap_cleared"], (rank, name, got)


def test_generate_keeps_vision_tower_gathered_when_ranks_differ_in_modality():
    want = {rank: _vlm_reference(rank) for rank in (0, 1)}
    assert want[0] != want[1]  # the image changes rank 0's tokens
    for rank, r in _run("block", timeout = 150, vlm = True).items():
        got = r["vlm"]
        assert "error" not in got, (rank, got)
        assert got["tower_gathered"] and not got["tower_streamed"], (rank, got)
        assert got["tokens"] == want[rank], (rank, got, want[rank])
