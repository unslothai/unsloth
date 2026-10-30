# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Batched multi-LoRA training: kernels against an fp64 reference, and per-adapter isolation."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
import torch
import unsloth  # noqa: F401

from real_accelerator import has_real_cuda
from unsloth.kernels import multi_lora as ml

REPO = pathlib.Path(__file__).resolve().parents[1]


def _case(
    U = 3,
    M = 37,
    K = 48,
    N = 40,
    R = 8,
    seed = 0,
    dtype = torch.float64,
    device = "cpu",
):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(M, K, generator = g, dtype = torch.float64)
    W = torch.randn(N, K, generator = g, dtype = torch.float64) / 7
    A = torch.randn(U, R, K, generator = g, dtype = torch.float64)
    B = torch.randn(U, N, R, generator = g, dtype = torch.float64)
    ids = torch.randint(-1, U, (M,), generator = g)
    ids[ids == U - 1] = 0  # the last adapter gets no tokens
    scales = torch.linspace(0.5, 2.0, U, dtype = torch.float64)
    cast = lambda t: t.to(device = device, dtype = dtype)
    return cast(X), cast(W), cast(A), cast(B), scales.to(device), ids.to(device)


def _dense(X, W, A, B, scales, ids):
    rows = []
    for t in range(X.shape[0]):
        y = X[t] @ W.t()
        u = int(ids[t])
        if u >= 0:
            y = y + scales[u] * (X[t] @ A[u].t()) @ B[u].t()
        rows.append(y)
    return torch.stack(rows)


def test_batch_groups_tokens_by_adapter_in_original_order():
    ids = torch.tensor([2, 0, -1, 0, 2, 1, -1, 0])
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, 4)
    assert batch.seg_start.tolist() == [0, 3, 4, 6, 6]
    assert batch.sorted_idx.tolist()[:6] == [1, 3, 7, 5, 0, 4]
    assert sorted(batch.sorted_idx.tolist()[6:]) == [2, 6]
    per_row = ml.MultiLoraBatch.from_adapter_ids(torch.tensor([1, 0]), 2, seq_len = 3)
    assert per_row.seg_start.tolist() == [0, 3, 6]
    assert per_row.sorted_idx.tolist() == [3, 4, 5, 0, 1, 2]
    with pytest.raises(ValueError):
        ml.MultiLoraBatch.from_adapter_ids(torch.tensor([0, 4]), 4)


def test_reference_delta_matches_the_per_token_definition():
    X, W, A, B, scales, ids = _case()
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, A.shape[0])
    Y = X @ W.t() + ml.multi_lora_delta(X, A, B, scales, batch)
    torch.testing.assert_close(Y, _dense(X, W, A, B, scales, ids), rtol = 1e-12, atol = 1e-12)


def _tiny_model(U, ranks = 4):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 24), torch.nn.Tanh(), torch.nn.Linear(24, 8))
    model = model.double()
    ml.inject_multi_lora(model, ["0", "2"], U, r = ranks, lora_alpha = 8)
    for _, layer in ml._layers(model):
        layer.double()
        for B in layer.lora_B:
            torch.nn.init.normal_(B)
    return model


def test_each_adapter_gets_only_its_own_gradient():
    model = _tiny_model(3)
    x = torch.randn(10, 16, dtype = torch.float64)
    ids = torch.tensor([0, 1, 0, 2, 1, 0, -1, 2, 1, 0])
    ml.set_multi_lora_batch(model, ids)
    out = model(x)
    # Loss on adapter 1's rows only: adapters 0 and 2 must see exactly zero.
    out[ids == 1].pow(2).sum().backward()
    for _, layer in ml._layers(model):
        for params in (layer.lora_A, layer.lora_B):
            assert params[0].grad.abs().max() == 0 and params[2].grad.abs().max() == 0
            assert params[1].grad.abs().max() > 0
    assert all(not p.requires_grad for n, p in model.named_parameters() if "lora_" not in n)


def test_batched_training_matches_each_adapter_trained_alone():
    U, steps = 2, 4
    x = torch.randn(8, 16, dtype = torch.float64)
    target = torch.randn(8, 8, dtype = torch.float64)
    ids = torch.tensor([0, 1, 1, 0, 0, 1, 0, 1])

    def train(model, rows, row_ids):
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr = 1e-2)
        ml.set_multi_lora_batch(model, row_ids)
        for _ in range(steps):
            opt.zero_grad()
            err = (model(x[rows]) - target[rows]).pow(2).mean(1)
            loss = sum(err[row_ids == u].mean() for u in range(U) if (row_ids == u).any())
            loss.backward()
            ml.drop_absent_adapter_grads_(model)
            ml.clip_grad_norm_per_adapter_(model, 0.5)
            opt.step()
        return model

    batched = train(_tiny_model(U), slice(None), ids)
    for u in range(U):
        solo = train(_tiny_model(U), ids == u, torch.full((int((ids == u).sum()),), u))
        for (_, a), (_, b) in zip(ml._layers(batched), ml._layers(solo)):
            torch.testing.assert_close(a.lora_A[u], b.lora_A[u], rtol = 1e-10, atol = 1e-12)
            torch.testing.assert_close(a.lora_B[u], b.lora_B[u], rtol = 1e-10, atol = 1e-12)


def test_ranks_differ_per_adapter_and_export_at_their_own_rank():
    model = _tiny_model(2, ranks = [2, 4])
    layer = ml._layers(model)[0][1]
    A, B = layer.stacked()
    assert A.shape == (2, 4, 16) and B.shape == (2, 24, 4)
    assert A[0, 2:].abs().max() == 0 and B[0, :, 2:].abs().max() == 0
    state = ml.adapter_state_dict(model, 0)
    assert state["base_model.model.0.lora_A.weight"].shape == (2, 16)
    assert state["base_model.model.0.lora_B.weight"].shape == (24, 2)
    assert layer.scales.tolist() == [4.0, 2.0]


def test_an_adapter_absent_from_the_batch_is_not_stepped():
    model = _tiny_model(3)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2, weight_decay = 0.1)
    before = [p.detach().clone() for p in params]
    ml.set_multi_lora_batch(model, torch.tensor([0, 2, 0, 2]))
    model(torch.randn(4, 16, dtype = torch.float64)).sum().backward()
    assert ml.drop_absent_adapter_grads_(model) == [1]
    ml.clip_grad_norm_per_adapter_(model, 1.0)
    opt.step()
    for (name, p), old in zip(
        [(n, p) for n, p in model.named_parameters() if p.requires_grad], before
    ):
        moved = not torch.equal(p, old)
        assert moved == (".1" not in name.rsplit("lora_", 1)[1]), name
        assert (p in opt.state and len(opt.state[p]) > 0) == moved, name


def test_a_target_that_matches_nothing_leaves_the_model_trainable():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    with pytest.raises(ValueError):
        ml.inject_multi_lora(model, ["q_proj"], 2)
    assert all(p.requires_grad for p in model.parameters())


def test_a_single_string_target_is_one_name():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
    assert ml.inject_multi_lora(model, "1", 2) == ["1"]


def test_per_adapter_loss_is_each_adapters_own_mean():
    logits = torch.randn(3, 5, 7)
    labels = torch.randint(0, 7, (3, 5))
    labels[1, 3:] = -100
    total, losses = ml.per_adapter_loss(logits, labels, torch.tensor([1, 0, 1]), 2)
    lone = lambda r: torch.nn.functional.cross_entropy(
        logits[r, :-1].reshape(-1, 7), labels[r, 1:].reshape(-1), ignore_index = -100
    )
    torch.testing.assert_close(losses, torch.stack([lone([1]), lone([0, 2])]))
    torch.testing.assert_close(total, losses.sum())


def test_targets_match_qualified_names_like_peft():
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Sequential(torch.nn.Linear(4, 4)), torch.nn.Linear(4, 4))
    assert ml.inject_multi_lora(model, ["0.0"], 2) == ["0.0"]
    assert isinstance(model[0][0], ml.MultiLoraLinear) and not isinstance(
        model[1], ml.MultiLoraLinear
    )


def test_layers_refuse_a_batch_that_does_not_match_the_tokens():
    model = _tiny_model(2)
    with pytest.raises(ValueError):
        model(torch.randn(4, 16, dtype = torch.float64))
    ml.set_multi_lora_batch(model, torch.tensor([0, 1, 0]))
    with pytest.raises(ValueError):
        model(torch.randn(4, 16, dtype = torch.float64))


_INTERPRET = textwrap.dedent(
    """
    import importlib.util, os, sys, types
    import torch
    import triton
    import triton.language as tl
    root = sys.argv[1]

    @triton.jit
    def _probe(X, n):
        for i in range(0, n, 1):
            tl.store(X + i, i)

    try:  # some interpreter builds cannot take a runtime loop bound with this NumPy
        _probe[(1,)](torch.zeros(2), 2)
    except Exception as exc:  # noqa: BLE001
        print("INTERPRETER UNSUPPORTED", repr(exc)[:200])
        sys.exit(0)
    pkg = types.ModuleType("mlk"); pkg.__path__ = []; sys.modules["mlk"] = pkg
    utils = types.ModuleType("mlk.utils")
    utils._base_quant_state = lambda layer, W: None
    utils.fast_dequantize = lambda W, q = None, dtype = None, **k: W
    utils.matmul_lora = lambda X, W, Wq, A, B, s, out = None: X @ W.t()
    utils.torch_amp_custom_fwd = utils.torch_amp_custom_bwd = lambda f: f
    sys.modules["mlk.utils"] = utils
    spec = importlib.util.spec_from_file_location("mlk.multi_lora", os.path.join(root, "unsloth/kernels/multi_lora.py"))
    ml = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ml; spec.loader.exec_module(ml)
    torch.manual_seed(0)
    U, M, K, N, R = 3, 37, 48, 40, 8
    X = torch.randn(M, K, requires_grad = True)
    W = torch.randn(N, K) / 7
    A = torch.randn(U, R, K, requires_grad = True)
    B = torch.randn(U, N, R, requires_grad = True)
    scales = torch.tensor([0.5, 2.0, 1.0])
    ids = torch.randint(-1, U, (M,)); ids[ids == 2] = 0
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, U)
    if len(sys.argv) > 2:  # negative control: a wrong segment table must be caught
        batch.seg_start[1] += 1
    Y = ml.MultiLoRA_W.apply(X, W, None, A, B, scales, batch.sorted_idx, batch.seg_start)
    g = torch.randn_like(Y)
    got = (Y,) + torch.autograd.grad(Y, (X, A, B), g)
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, U)
    X64, A64, B64 = [t.detach().double().requires_grad_() for t in (X, A, B)]
    Y64 = X64 @ W.double().t() + ml.multi_lora_delta(X64, A64, B64, scales.double(), batch)
    want = (Y64,) + torch.autograd.grad(Y64, (X64, A64, B64), g.double())
    for name, a, b in zip(("Y", "dX", "dA", "dB"), got, want):
        err = (a.double() - b).abs().max().item()
        assert err <= 1e-4 * max(b.abs().max().item(), 1), (name, err)
    assert got[2][2].abs().max() == 0 and got[3][2].abs().max() == 0
    print("KERNELS OK")
    """
)


@pytest.mark.parametrize("negative_control", [False, True])
def test_triton_kernels_match_fp64_in_the_interpreter(tmp_path, negative_control):
    pytest.importorskip("triton")
    script = tmp_path / "interpret.py"
    script.write_text(_INTERPRET, encoding = "utf-8")
    env = {**os.environ, "TRITON_INTERPRET": "1"}
    argv = [sys.executable, str(script), str(REPO)] + (["break"] if negative_control else [])
    proc = subprocess.run(argv, env = env, capture_output = True, text = True, timeout = 600)
    if "INTERPRETER UNSUPPORTED" in proc.stdout:
        pytest.skip(proc.stdout.strip())
    if negative_control:
        assert proc.returncode != 0 and "AssertionError" in proc.stderr, proc.stdout + proc.stderr
    else:
        assert proc.returncode == 0 and "KERNELS OK" in proc.stdout, proc.stdout + proc.stderr


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(3, 37, 48, 40, 8), (4, 1000, 512, 768, 16), (2, 257, 4096, 1024, 64)]
)
def test_triton_kernels_match_fp64_on_gpu(dtype, shape):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("no bf16")
    U, M, K, N, R = shape
    X, W, A, B, scales, ids = _case(U, M, K, N, R, dtype = dtype, device = "cuda")
    X.requires_grad_()
    A.requires_grad_()
    B.requires_grad_()
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, U)
    Y = ml.multi_lora_linear(X, W, None, None, A, B, scales.float(), batch)
    g = torch.randn_like(Y)
    got = (Y,) + torch.autograd.grad(Y, (X, A, B), g)
    X64, A64, B64 = [t.detach().double().requires_grad_() for t in (X, A, B)]
    Y64 = X64 @ W.double().t() + ml.multi_lora_delta(X64, A64, B64, scales, batch)
    want = (Y64,) + torch.autograd.grad(Y64, (X64, A64, B64), g.double())
    tol = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 6e-2}[dtype]
    for name, a, b in zip(("Y", "dX", "dA", "dB"), got, want):
        rel = (a.double() - b).norm() / b.norm().clamp_min(1e-12)
        assert rel <= tol, (name, rel.item())
    assert got[2][U - 1].abs().max() == 0 and got[3][U - 1].abs().max() == 0


def test_presence_spans_gradient_accumulation_and_resets_after_a_step():
    model = _tiny_model(3)
    for ids in (torch.tensor([0, 0]), torch.tensor([2, 2])):
        ml.set_multi_lora_batch(model, ids)
        model(torch.randn(2, 16, dtype = torch.float64)).sum().backward()
    # Adapter 0 trained in the first microbatch: its accumulated gradient survives.
    assert ml.drop_absent_adapter_grads_(model) == [1]
    layer = ml._layers(model)[0][1]
    assert layer.lora_B[0].grad is not None and layer.lora_B[1].grad is None
    assert layer.seen is None


def test_presence_and_loss_means_are_global_across_ranks(monkeypatch):
    groups = []

    def all_reduce(
        t,
        op = None,
        group = None,
    ):
        groups.append(group)
        if t.dtype == torch.int32:  # presence: the other rank trained adapter 1
            t.add_(torch.tensor([0, 1], dtype = torch.int32))
        else:  # valid tokens per adapter: 10 more for adapter 1 on the other rank
            t.add_(torch.tensor([0.0, 10.0], dtype = t.dtype))

    monkeypatch.setattr(ml, "_distributed", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group = None: 2)
    model = _tiny_model(2)
    ml.set_multi_lora_batch(model, torch.tensor([0]))
    model(torch.randn(1, 16, dtype = torch.float64)).sum().backward()
    assert ml.drop_absent_adapter_grads_(model, process_group = "dp") == []

    logits = torch.randn(2, 6, 7)
    labels = torch.randint(0, 7, (2, 6))
    total, local = ml.per_adapter_loss(logits, labels, torch.tensor([0, 1]), 2, process_group = "dp")
    # This rank: 5 tokens per adapter. Adapter 1 also has 10 on the other rank, so it weighs 5/15 here.
    torch.testing.assert_close(total, (local[0] * 5 / 5 + local[1] * 5 / 15) * 2)
    assert groups == ["dp", "dp"]


def test_an_adapter_whose_labels_are_all_ignored_does_not_train():
    model = _tiny_model(2)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2, weight_decay = 0.1)
    ml.set_multi_lora_batch(model, torch.tensor([0, 1]))
    hidden = model(torch.randn(2, 16, dtype = torch.float64))
    logits = hidden.view(2, 4, 2)
    labels = torch.tensor([[0, 1, 0, 1], [-100, -100, -100, -100]])
    total, _ = ml.per_adapter_loss(logits, labels, torch.tensor([0, 1]), 2, model = model)
    total.backward()
    assert ml.drop_absent_adapter_grads_(model) == [1]
    layer = ml._layers(model)[0][1]
    before = layer.lora_B[1].detach().clone()
    opt.step()
    assert torch.equal(layer.lora_B[1], before) and layer.lora_B[1] not in opt.state


def test_gradient_accumulation_weights_microbatches_by_tokens():
    torch.manual_seed(1)
    x = torch.randn(4, 16, dtype = torch.float64)
    ids = torch.tensor([0, 1, 0, 1])
    labels = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, -100, -100], [1, 1, 0, 0]])

    def grads(chunks, num_items):
        model = _tiny_model(2)
        for rows in chunks:
            ml.set_multi_lora_batch(model, ids[rows])
            logits = model(x[rows]).view(len(rows), 4, 2)
            total, _ = ml.per_adapter_loss(
                logits, labels[rows], ids[rows], 2, num_items_in_batch = num_items
            )
            total.backward()
        return [p.grad.clone() for p in model.parameters() if p.requires_grad]

    window = sum(ml.adapter_token_counts(labels[r], ids[r], 2) for r in ([0, 1], [2, 3]))
    assert window.tolist() == [4.0, 6.0]
    whole = grads([[0, 1, 2, 3]], None)
    for a, b in zip(grads([[0, 1], [2, 3]], window), whole):
        torch.testing.assert_close(a, b)
    # Per-microbatch means weight adapter 0's 1-token microbatch like its 3-token one.
    assert any(not torch.allclose(a, b) for a, b in zip(grads([[0, 1], [2, 3]], None), whole))


def test_layers_with_different_adapter_counts_are_refused():
    model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
    ml.inject_multi_lora(model, ["0"], 2, r = 4)
    ml.inject_multi_lora(model, ["1"], 4, r = 4)
    with pytest.raises(ValueError, match = "same number of adapters"):
        ml.set_multi_lora_batch(model, torch.tensor([3, 0, 1]))


def test_a_step_where_no_adapter_trained_clips_to_zero_norms():
    model = _tiny_model(2)
    ml.set_multi_lora_batch(model, torch.tensor([-1, -1]))
    assert ml.drop_absent_adapter_grads_(model) == [0, 1]
    assert ml.clip_grad_norm_per_adapter_(model, 1.0).tolist() == [0.0, 0.0]


def test_fsdp2_sharded_grads_norm_and_clip_on_local_shards(tmp_path):
    script = tmp_path / "dtensor.py"
    script.write_text(
        textwrap.dedent(
            """
            import importlib.util, os, sys, types, torch, torch.distributed as dist
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.tensor import Shard, distribute_tensor

            pkg = types.ModuleType("mlk"); pkg.__path__ = []; sys.modules["mlk"] = pkg
            utils = types.ModuleType("mlk.utils")
            utils._base_quant_state = lambda layer, W: None
            utils.fast_dequantize = lambda W, q = None, dtype = None, **k: W
            utils.matmul_lora = lambda X, W, Wq, A, B, s, out = None: X @ W.t()
            utils.torch_amp_custom_fwd = utils.torch_amp_custom_bwd = lambda f: f
            sys.modules["mlk.utils"] = utils
            path = os.path.join(sys.argv[2], "unsloth/kernels/multi_lora.py")
            spec = importlib.util.spec_from_file_location("mlk.multi_lora", path)
            ml = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ml; spec.loader.exec_module(ml)

            os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = sys.argv[1])
            dist.init_process_group("gloo", rank = 0, world_size = 1)
            mesh = init_device_mesh("cpu", (1,))
            torch.manual_seed(0)
            model = torch.nn.Sequential(torch.nn.Linear(8, 8))
            ml.inject_multi_lora(model, ["0"], 2, r = 4)
            layer = model[0]
            grads = [torch.randn_like(p) for p in layer.parameters() if p.requires_grad]
            want = []
            for u in range(2):
                sq = sum(g.pow(2).sum() for g in (grads[u], grads[2 + u]))
                want.append(float(sq.sqrt()))
            for p, g in zip([p for p in layer.parameters() if p.requires_grad], grads):
                p.data = distribute_tensor(p.data, mesh, [Shard(0)])
                p.grad = distribute_tensor(g, mesh, [Shard(0)])
            got = ml.clip_grad_norm_per_adapter_(model, 0.5)
            after = ml.per_adapter_grad_norm(model)
            assert torch.allclose(got, torch.tensor(want), rtol = 1e-5), (got, want)
            assert torch.allclose(after, torch.tensor([0.5, 0.5]), rtol = 1e-4), after
            dist.destroy_process_group()
            print("DTENSOR OK")
            """
        ),
        encoding = "utf-8",
    )
    port = str(29500 + os.getpid() % 1000)
    proc = subprocess.run(
        [sys.executable, str(script), port, str(REPO)],
        capture_output = True,
        text = True,
        timeout = 600,
        env = os.environ,
    )
    assert proc.returncode == 0 and "DTENSOR OK" in proc.stdout, (
        proc.stdout[-2000:] + proc.stderr[-4000:]
    )


def test_ddp_two_ranks_train_each_adapter_on_its_global_mean(tmp_path):
    # gloo DDP, 2 ranks with different adapters: per-adapter grads must equal the global-batch mean.
    script = tmp_path / "ddp.py"
    script.write_text(
        textwrap.dedent(
            """
            import importlib.util, os, sys, types, torch, torch.distributed as dist
            import torch.multiprocessing as mp

            def load(root):
                pkg = types.ModuleType("mlk"); pkg.__path__ = []; sys.modules["mlk"] = pkg
                utils = types.ModuleType("mlk.utils")
                utils._base_quant_state = lambda layer, W: None
                utils.fast_dequantize = lambda W, q = None, dtype = None, **k: W
                utils.matmul_lora = lambda X, W, Wq, A, B, s, out = None: X @ W.t()
                utils.torch_amp_custom_fwd = utils.torch_amp_custom_bwd = lambda f: f
                sys.modules["mlk.utils"] = utils
                path = os.path.join(root, "unsloth/kernels/multi_lora.py")
                spec = importlib.util.spec_from_file_location("mlk.multi_lora", path)
                ml = importlib.util.module_from_spec(spec); sys.modules[spec.name] = ml
                spec.loader.exec_module(ml)
                return ml

            def model(ml):
                torch.manual_seed(0)
                m = torch.nn.Sequential(torch.nn.Linear(6, 5)).double()
                ml.inject_multi_lora(m, ["0"], 3, r = 2)
                for B in m[0].lora_B:
                    torch.nn.init.normal_(B)
                return m

            def data():
                g = torch.Generator().manual_seed(1)
                x = torch.randn(4, 3, 6, generator = g, dtype = torch.float64)
                labels = torch.randint(0, 5, (4, 3), generator = g)
                labels[3, 1:] = -100  # adapter 1's second row: one valid label fewer
                return x, labels, torch.tensor([0, 0, 1, 1])

            def grads(m):
                return [p.grad.clone() if p.grad is not None else None for p in m.parameters() if p.requires_grad]

            def worker(rank, root, port, out):
                os.environ.update(MASTER_ADDR = "127.0.0.1", MASTER_PORT = port)
                dist.init_process_group("gloo", rank = rank, world_size = 2)
                ml = load(root)
                m = torch.nn.parallel.DistributedDataParallel(model(ml))
                x, labels, ids = data()
                rows = slice(2 * rank, 2 * rank + 2)
                ml.set_multi_lora_batch(m.module, ids[rows], seq_len = 3)
                total, _ = ml.per_adapter_loss(m(x[rows]), labels[rows], ids[rows], 3, model = m.module)
                total.backward()
                absent = ml.drop_absent_adapter_grads_(m.module)
                torch.save({"absent": absent, "grads": grads(m.module)}, f"{out}/rank{rank}.pt")
                dist.destroy_process_group()

            if __name__ == "__main__":
                root, port, out = sys.argv[1:4]
                mp.spawn(worker, args = (root, port, out), nprocs = 2)
                ml = load(root)
                m = model(ml)
                x, labels, ids = data()
                ml.set_multi_lora_batch(m, ids, seq_len = 3)
                total, _ = ml.per_adapter_loss(m(x), labels, ids, 3, model = m)
                total.backward()
                ml.drop_absent_adapter_grads_(m)
                want = grads(m)
                r0, r1 = (torch.load(f"{out}/rank{r}.pt") for r in (0, 1))
                assert r0["absent"] == r1["absent"] == [2], (r0["absent"], r1["absent"])
                for a, b, w in zip(r0["grads"], r1["grads"], want):
                    if w is None:
                        assert a is None and b is None
                        continue
                    torch.testing.assert_close(a, b)
                    torch.testing.assert_close(a, w)
                print("DDP OK")
            """
        ),
        encoding = "utf-8",
    )
    port = str(30500 + os.getpid() % 1000)
    proc = subprocess.run(
        [sys.executable, str(script), str(REPO), port, str(tmp_path)],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert proc.returncode == 0 and "DDP OK" in proc.stdout, (
        proc.stdout[-2000:] + proc.stderr[-4000:]
    )


@pytest.mark.skipif(not has_real_cuda(), reason = "needs a CUDA GPU")
@pytest.mark.parametrize("triton_kernels", [True, False])
def test_compiles_fullgraph_without_recompiling_per_batch(triton_kernels):
    from torch._dynamo.utils import counters

    def model():
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 32))
        m = m.cuda().to(torch.bfloat16)
        ml.inject_multi_lora(m, ["0", "2"], 3, r = [4, 8, 8])
        for _, layer in ml._layers(m):
            for B in layer.lora_B:
                torch.nn.init.normal_(B, std = 0.05)
        return m

    def step(m, fn, x, ids):
        ml.set_multi_lora_batch(m, ids)
        m.zero_grad(set_to_none = True)
        y = fn(x)
        y.float().pow(2).mean().backward()
        return [y.detach()] + [p.grad.clone() for p in m.parameters() if p.requires_grad]

    ml.set_multi_lora_triton(triton_kernels)
    try:
        torch._dynamo.reset()
        counters.clear()
        eager, compiled = model(), model()
        fn = torch.compile(compiled, fullgraph = True)
        g = torch.Generator().manual_seed(0)
        for _ in range(4):  # same size, different routing: one graph
            x = torch.randn(48, 64, generator = g).cuda().to(torch.bfloat16)
            ids = torch.randint(-1, 3, (48,), generator = g).cuda()
            for a, b in zip(step(compiled, fn, x, ids), step(eager, eager, x, ids)):
                rel = (a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)
                assert rel < 3e-2, rel
        assert sum(counters["graph_break"].values()) == 0
        assert counters["stats"]["unique_graphs"] == 1
    finally:
        ml.set_multi_lora_triton(True)
        torch._dynamo.reset()


def _tiny_qwen3(attn):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size = 64,
        hidden_size = 32,
        intermediate_size = 64,
        num_hidden_layers = 2,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 8,
        max_position_embeddings = 128,
        attn_implementation = attn,
    )
    config.use_cache = (
        False  # training setup: transformers only detects packed rows without a KV cache
    )
    model = Qwen3ForCausalLM(config).double()
    targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    ml.inject_multi_lora(model, targets, 8, r = 4)
    for _, layer in ml._layers(model):
        layer.double()
        for B in layer.lora_B:
            torch.nn.init.normal_(B, std = 0.1)
    return model


@pytest.mark.parametrize("attn", ["sdpa", "eager"])
def test_packed_rows_of_eight_adapters_match_each_sequence_alone(attn):
    # bsz 2, 8 users' sequences packed 4 per row, position_ids restarting per sequence (varlen packing).
    model = _tiny_qwen3(attn)
    g = torch.Generator().manual_seed(0)
    lengths = [5, 9, 3, 7, 6, 4, 8, 5]
    seqs = [torch.randint(0, 64, (n,), generator = g) for n in lengths]

    def adapter_grads(u):
        return [p.grad.clone() for _, l in ml._layers(model) for p in (l.lora_A[u], l.lora_B[u])]

    alone = {}
    for u, seq in enumerate(seqs):
        model.zero_grad(set_to_none = True)
        ml.set_multi_lora_batch(model, torch.tensor([u]), seq_len = len(seq))
        out = model(input_ids = seq[None]).logits
        total, losses = ml.per_adapter_loss(out, seq[None], torch.tensor([u]), 8)
        total.backward()
        alone[u] = (losses[u], adapter_grads(u))

    rows = [list(range(0, 4)), list(range(4, 8))]
    T = max(sum(lengths[u] for u in r) for r in rows)
    input_ids = torch.zeros(2, T, dtype = torch.long)
    labels = torch.full((2, T), -100)
    ids = torch.full((2, T), -1)
    position_ids = torch.zeros(2, T, dtype = torch.long)
    for b, r in enumerate(rows):
        at = 0
        for u in r:
            n = lengths[u]
            input_ids[b, at : at + n] = seqs[u]
            labels[b, at + 1 : at + n] = seqs[u][1:]  # first token of a sequence is never a label
            ids[b, at : at + n] = u
            position_ids[b, at : at + n] = torch.arange(n)
            at += n
        position_ids[b, at:] = torch.arange(T - at)  # padding tail: its own sequence, no labels

    def packed(position_ids):
        model.zero_grad(set_to_none = True)
        ml.set_multi_lora_batch(model, ids)
        out = model(input_ids = input_ids, position_ids = position_ids).logits
        total, losses = ml.per_adapter_loss(out, labels, ids, 8)
        total.backward()
        return losses, {u: adapter_grads(u) for u in range(8)}

    losses, grads = packed(position_ids)
    for u in range(8):  # Qwen3's RMSNorm runs in float32: fp32-level tolerance
        torch.testing.assert_close(losses[u], alone[u][0], rtol = 1e-5, atol = 1e-6)
        for a, b in zip(grads[u], alone[u][1]):
            torch.testing.assert_close(a, b, rtol = 1e-5, atol = 1e-6)
    # Negative control: without restarting positions the sequences attend to each other.
    contaminated, _ = packed(torch.arange(T).expand(2, T))
    assert not torch.allclose(contaminated, losses, rtol = 1e-3, atol = 1e-4)


def _absent_adapter_case(dtype):
    # Every token is adapter 0's; adapter 1 is absent from the batch.
    X, W, A, B, scales, _ = _case(U = 2, M = 9, K = 16, N = 12, R = 4, dtype = torch.float64)
    ids = torch.zeros(X.shape[0], dtype = torch.long)
    batch = ml.MultiLoraBatch.from_adapter_ids(ids, 2)
    want = (X @ W.t() + scales[0] * (X @ A[0].t()) @ B[0].t()).to(dtype)
    cast = lambda t: t.to(dtype)
    return cast(X), cast(W), cast(A), cast(B), scales, batch, want


@pytest.mark.parametrize("poison", ["nan_A", "nan_B", "inf_B", "fp16_overflow"])
def test_an_absent_adapter_cannot_corrupt_present_tokens_in_the_reference_path(poison):
    dtype = torch.float16 if poison == "fp16_overflow" else torch.float64
    X, W, A, B, scales, batch, want = _absent_adapter_case(dtype)
    A, B = A.clone(), B.clone()
    if poison == "nan_A":
        A[1, 0, 0] = float("nan")
    elif poison == "nan_B":
        B[1, 0, 0] = float("nan")
    elif poison == "inf_B":
        B[1, 0, 0] = float("inf")
    else:  # finite fp16 weights whose X @ A.T overflows to inf
        A[1] = 6.0e4
    X = X.requires_grad_()
    A = A.requires_grad_()
    B = B.requires_grad_()
    Y = X @ W.t() + ml.multi_lora_delta(X, A, B, scales, batch)
    assert torch.isfinite(Y).all()
    tol = dict(rtol = 1e-2, atol = 1e-2) if dtype == torch.float16 else {}
    torch.testing.assert_close(Y, want, **tol)
    Y.float().pow(2).sum().backward()
    assert torch.isfinite(A.grad[0]).all() and torch.isfinite(B.grad[0]).all()
    assert A.grad[0].abs().max() > 0
    assert torch.isfinite(X.grad).all()
