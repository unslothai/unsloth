# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native FSDP2 (gloo CPU world of 2): the Gemma / Gemma2 / Cohere decode scratch under block streaming.

These decode loops build their norm-weight scratch from layers[0].input_layernorm.weight before
the loop's block-swap enter(0). With generate streaming the blocks, that weight is still a sharded
DTensor there, so torch.empty_like made a DTensor scratch, and the Gemma RMS norm then mixed it with
the gathered weights and plain activations. The scratch statements are taken from the real decode
functions and run against a sharded tiny model; the scratch must be a plain tensor of the hidden
size and greedy tokens must equal one process."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import ast  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port  # noqa: E402

# (file, decode function) whose scratch is built before the loop's enter(0).
LOOPS = (
    ("gemma.py", "GemmaModel_fast_forward_inference"),
    ("gemma2.py", "Gemma2Model_fast_forward_inference"),
    ("cohere.py", "CohereModel_fast_forward_inference"),
)
LAYERS = 4
NEW_TOKENS = 5
PROMPT = [[3, 17, 9, 25, 4]]


def _function(path, name):
    tree = ast.parse(open(path, encoding = "utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path}")


def _scratch_code(filename, name):
    """The `norm_weight = ...` and `out_weights = ...` statements of the decode function."""
    path = os.path.join(REPO, "unsloth", "models", filename)
    func = _function(path, name)
    body = [
        n
        for n in func.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id in ("norm_weight", "out_weights") for t in n.targets
        )
    ]
    assert [t.id for n in body for t in n.targets] == ["norm_weight", "out_weights"], name
    return compile(ast.Module(body = body, type_ignores = []), path, "exec")


def _gemma_rms_norm():
    path = os.path.join(REPO, "unsloth", "models", "llama.py")
    func = _function(path, "fast_rms_layernorm_inference_gemma")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body = [func], type_ignores = []), path, "exec"), namespace)
    return namespace["fast_rms_layernorm_inference_gemma"]


class _Scratch(list):
    """_LazyDeviceScratch on the host: index 0 is the CPU."""

    def __init__(self, device_count, make):
        super().__init__([None] * device_count)
        self._make = make

    def __getitem__(self, index):
        if list.__getitem__(self, index) is None:
            list.__setitem__(self, index, self._make(torch.device("cpu")))
        return list.__getitem__(self, index)


class _Norm(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.linspace(-0.5, 0.5, d))
        self.variance_epsilon = 1e-6


class _Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.input_layernorm = _Norm(d)
        self.up = torch.nn.Linear(d, 2 * d)
        self.down = torch.nn.Linear(2 * d, d)

    def forward(self, x):
        return x + self.down(torch.tanh(self.up(x)))


class _Inner(torch.nn.Module):
    def __init__(self, vocab, d):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(vocab, d)
        self.layers = torch.nn.ModuleList([_Block(d) for _ in range(LAYERS)])


class _GemmaNet(torch.nn.Module):
    def __init__(
        self,
        scratch_code,
        vocab = 32,
        d = 16,
    ):
        super().__init__()
        torch.manual_seed(0)
        self.model = _Inner(vocab, d)
        self.head = torch.nn.Linear(d, vocab, bias = False)
        self._scratch_code = scratch_code
        self._norm = _gemma_rms_norm()

    def _decode_step(self, ids):
        scope = {"self": self, "torch": torch, "_LazyDeviceScratch": _Scratch, "DEVICE_COUNT": 1}
        exec(self._scratch_code, scope)  # the real decode function's scratch, before enter(0)
        out_weights = scope["out_weights"]
        x = self.model.embed_tokens(ids)
        swap = getattr(self.model.layers, "_unsloth_block_swap", None)
        for idx, layer in enumerate(self.model.layers):
            if swap is not None:
                swap.enter(idx)
            scratch = out_weights[0]
            self._scratch_plain = getattr(self, "_scratch_plain", True) and (
                type(scratch) is torch.Tensor and tuple(scratch.shape) == (x.shape[-1],)
            )
            h = self._norm(layer.input_layernorm, x, scratch)
            h = torch.tanh(h @ layer.up.weight.t() + layer.up.bias)
            x = x + h @ layer.down.weight.t() + layer.down.bias
            if swap is not None:
                swap.leave(idx)
        return self.head(x)

    def generate(self, ids, new_tokens):
        out = ids
        logits = self._decode_step(ids)[:, -1]
        for _ in range(new_tokens):
            nxt = logits.argmax(-1, keepdim = True)
            out = torch.cat([out, nxt], dim = 1)
            logits = self._decode_step(nxt)[:, -1]
        return out[:, ids.shape[1] :]


def _reference():
    with torch.no_grad():
        return {
            name: _GemmaNet(_scratch_code(f, name))
            .generate(torch.tensor(PROMPT), NEW_TOKENS)[0]
            .tolist()
            for f, name in LOOPS
        }


def _case(native, filename, name):
    model = _GemmaNet(_scratch_code(filename, name))
    native.shard_model(
        model,
        native.FSDP2Config(
            device = "cpu",
            units = list(model.model.layers),
            fit_layout = False,
            param_dtype = torch.float32,
        ),
    )
    res = {}
    try:
        res["tokens"] = model.generate(torch.tensor(PROMPT), NEW_TOKENS)[0].tolist()
    except Exception as exc:
        res["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    res["scratch_plain"] = model.__dict__.get("_scratch_plain")
    return res


def _worker(rank, world, port, out):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
        UNSLOTH_FSDP2_GENERATE_GATHER = "block",
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 60)
    )
    try:
        native = _load_native()
        out.put((rank, {name: _case(native, f, name) for f, name in LOOPS}))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def test_decode_norm_scratch_is_plain_under_block_streaming():
    import torch.multiprocessing as mp

    want = _reference()
    out = mp.get_context("spawn").Queue()
    procs = mp.start_processes(
        _worker, args = (2, _free_port(), out), nprocs = 2, join = False, start_method = "spawn"
    )
    deadline = time.time() + 180
    try:
        while not procs.join(timeout = 5):
            if time.time() > deadline:
                for p in procs.processes:
                    p.kill()
                pytest.fail("the ranks did not finish within 180 s (hung collective)")
    except mp.ProcessExitedException as exc:
        pytest.fail(f"a rank died ({exc})")
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        assert "crash" not in r, r["crash"]
        for _, name in LOOPS:
            got = r[name]
            assert "error" not in got, (rank, name, got)
            assert got["scratch_plain"], (rank, name, got)
            assert got["tokens"] == want[name], (rank, name, got, want[name])
