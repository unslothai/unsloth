# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Vocab-parallel lm_head under apply_tensor_parallel (TPConfig(vocab_parallel_head = True)) on a gloo CPU
world (2 and 4 ranks) vs the same model in one process, fp32: whole-logit forward, vocab-parallel cross
entropy, LoRA / frozen / trainable heads, refusals, generate and adapter save."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import sys  # noqa: E402
import tempfile  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_tp_native_oracle import LLAMA, QWEN3, _batch, build, tpn  # noqa: E402

# tp = 4 needs 4 query / 4 key-value heads and an intermediate size divisible by 4 (tiny Qwen3: 4 / 2).
_QWEN3_TP4 = {"num_attention_heads": 8, "num_key_value_heads": 4, "intermediate_size": 64}


def _build(head, overrides):
    """head: "frozen" (LoRA elsewhere), "lora" (LoRA on lm_head too) or "full" (trainable lm_head)."""
    if head == "lora":
        return build(QWEN3, overrides, target_modules = ["q_proj", "v_proj", "down_proj", "lm_head"])
    model = build(QWEN3, overrides, target_modules = ["q_proj", "v_proj", "down_proj"])
    if head == "full":
        tpn._unwrap(model).lm_head.weight.requires_grad_(True)
    return model


def _head(model):
    return tpn._unwrap(model).get_output_embeddings()


def _grads(model):
    return {
        n: tpn._full(p.grad, p).detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }


def _hf_steps(model, ids, clip):
    """Two optimizer steps through the transformers forward (lm_head(x) -> whole logits -> loss)."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr = 0.5)
    losses, grads, logits = [], None, None
    for step in range(2):
        out = model(input_ids = ids, labels = ids, use_cache = False)
        out.loss.backward()
        if step == 0:
            grads, logits = _grads(model), out.logits.detach().clone()
        clip(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none = True)
        losses.append(float(out.loss))
    return logits, grads, losses


def _ce_ref(head, hidden, labels, n_items, softcap, scale):
    logits = head(hidden).float()
    if scale:
        logits = logits * scale
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    shifted = torch.full_like(labels, -100)
    shifted[..., :-1] = labels[..., 1:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        shifted.reshape(-1),
        ignore_index = -100,
        reduction = "sum",
    )
    return loss / (n_items if n_items is not None else (shifted != -100).sum())


def _ce_grads(model, hidden):
    out = {"hidden": hidden.grad.detach().clone()}
    for n, p in _head(model).named_parameters():
        if p.grad is not None:
            out[n] = tpn._full(p.grad, p).detach().clone()
    return out


def _case_parity(rank, world, tmp, head, overrides):
    from safetensors.torch import load_file
    from torch.distributed.device_mesh import init_device_mesh

    ref, model = _build(head, overrides), _build(head, overrides)
    ref_head_bytes = sum(p.numel() * p.element_size() for p in _head(ref).parameters())
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    tpn.apply_tensor_parallel(model, mesh["tp"], config = tpn.TPConfig(vocab_parallel_head = True))
    vp_head = _head(model)
    assert tpn.is_vocab_parallel_head(vp_head)
    local_head_bytes = sum(tpn._local(p).numel() * p.element_size() for p in vp_head.parameters())
    generated = model.generate(
        input_ids = torch.zeros(1, 2, dtype = torch.long), max_new_tokens = 2, pad_token_id = 0
    )
    ids = _batch(ref)
    calls = dict(tpn.VOCAB_PARALLEL_CALLS)
    ref_out = _hf_steps(ref, ids, torch.nn.utils.clip_grad_norm_)
    tp_out = _hf_steps(model, ids, tpn.tp_clip_grad_norm_)
    gathers = tpn.VOCAB_PARALLEL_CALLS["gather"] - calls["gather"]

    # The loss path: vocab_parallel_cross_entropy on the decoder output, chunked, with a softcap and scale.
    results = {}
    for name, m in (("ref", ref), ("tp", model)):
        m.zero_grad(set_to_none = True)
        hidden = tpn._unwrap(m).model(input_ids = ids, use_cache = False).last_hidden_state
        hidden = hidden.detach().requires_grad_(True)
        n_items = torch.tensor(9)
        if name == "ref":
            loss = _ce_ref(_head(m), hidden, ids, n_items, 3.0, 0.7)
        else:
            loss = tpn.vocab_parallel_cross_entropy(
                _head(m),
                hidden,
                ids,
                n_items = n_items,
                logit_softcapping = 3.0,
                logit_scale_multiply = 0.7,
                chunk_tokens = 3,
            )
        loss.backward()
        results[name] = (float(loss), _ce_grads(m, hidden))
    ce_calls = tpn.VOCAB_PARALLEL_CALLS["cross_entropy"] - calls["cross_entropy"]
    tp_dir, ref_dir = os.path.join(tmp, "tp"), os.path.join(tmp, f"ref{rank}")
    saved = None
    if head == "lora":
        ref.save_pretrained(ref_dir)
        tpn.save_tp_adapter(model, tp_dir)
        if rank == 0:
            saved = (
                load_file(os.path.join(ref_dir, "adapter_model.safetensors")),
                load_file(os.path.join(tp_dir, "adapter_model.safetensors")),
            )
    return {
        "ref": ref_out,
        "tp": tp_out,
        "ce": results,
        "gathers": gathers,
        "ce_calls": ce_calls,
        "bytes": (ref_head_bytes, local_head_bytes),
        "generated": tuple(generated.shape),
        "saved": saved,
    }


def _case_refused(rank, world, tmp, what):
    from torch.distributed.device_mesh import init_device_mesh

    if what == "tied":
        model = build(LLAMA, {"tie_word_embeddings": True})
        inner = tpn._unwrap(model)
        inner.lm_head.weight = inner.get_input_embeddings().weight
    else:  # modules_to_save lm_head
        model = build(QWEN3, {}, target_modules = ["q_proj"], modules_to_save = ["lm_head"])
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names = ("tp",))
    try:
        tpn.apply_tensor_parallel(model, mesh["tp"], config = tpn.TPConfig(vocab_parallel_head = True))
        message = None
    except ValueError as e:
        message = str(e)
    untouched = not tpn.is_tensor_parallel(model) and not any(
        tpn._is_dtensor(p) for p in model.parameters()
    )
    return {"message": message, "untouched": untouched}


CASES = {"parity": _case_parity, "refused": _case_refused}


def _worker(rank, world, port, case, tmp, kwargs):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    import datetime

    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    finished = False
    try:
        res = CASES[case](rank, world, tmp, **kwargs)
        torch.save(res, os.path.join(tmp, f"result_{rank}.pt"))
        finished = True
    finally:
        dist.destroy_process_group()
    if finished:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


def _spawn(case, world, **kwargs):
    from tests._rendezvous_port import free_port

    ctx = multiprocessing.get_context("spawn")
    port = free_port()
    with tempfile.TemporaryDirectory() as tmp:
        procs = [
            ctx.Process(target = _worker, args = (r, world, port, case, tmp, kwargs))
            for r in range(world)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(600)
        codes = [p.exitcode for p in procs]
        assert codes == [0] * world, f"rank exit codes {codes}"
        return [
            torch.load(os.path.join(tmp, f"result_{r}.pt"), weights_only = False)
            for r in range(world)
        ]


@pytest.mark.parametrize(
    "world, head",
    [(2, "frozen"), (2, "lora"), (2, "full"), (4, "frozen"), (4, "lora")],
)
def test_vocab_parallel_head_matches_one_process(world, head):
    results = _spawn("parity", world, head = head, overrides = _QWEN3_TP4 if world == 4 else {})
    for rank, res in enumerate(results):
        ref_logits, ref_grads, ref_losses = res["ref"]
        logits, grads, losses = res["tp"]
        torch.testing.assert_close(logits, ref_logits, rtol = 1e-5, atol = 1e-5)
        assert set(grads) == set(ref_grads), sorted(set(grads) ^ set(ref_grads))[:6]
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol = 1e-4, atol = 1e-6, msg = name)
        assert losses == pytest.approx(ref_losses, rel = 1e-5, abs = 1e-6), (rank, losses, ref_losses)
        ref_loss, ref_ce_grads = res["ce"]["ref"]
        loss, ce_grads = res["ce"]["tp"]
        assert loss == pytest.approx(ref_loss, rel = 1e-5), (rank, loss, ref_loss)
        assert set(ce_grads) == set(ref_ce_grads), sorted(set(ce_grads) ^ set(ref_ce_grads))
        for name in ref_ce_grads:
            torch.testing.assert_close(
                ce_grads[name], ref_ce_grads[name], rtol = 1e-4, atol = 1e-6, msg = name
            )
        # The new paths ran: 2 forwards through the gathering head, 1 vocab-parallel cross entropy.
        assert res["gathers"] == 2 and res["ce_calls"] == 1, (res["gathers"], res["ce_calls"])
        full_bytes, local_bytes = res["bytes"]
        lora_a = 0 if head != "lora" else 4 * 8 * 4  # r x hidden, replicated
        assert local_bytes - lora_a == (full_bytes - lora_a) // world, (local_bytes, full_bytes)
        assert res["generated"] == (1, 4)
    if head == "lora":
        saved_ref, saved_tp = results[0]["saved"]
        assert set(saved_tp) == set(saved_ref)
        assert any("lm_head" in k for k in saved_ref)
        for key in saved_ref:
            torch.testing.assert_close(saved_tp[key], saved_ref[key], rtol = 1e-4, atol = 1e-5, msg = key)


@pytest.mark.parametrize("what", ["tied", "modules_to_save"])
def test_vocab_parallel_head_refusals_leave_the_model_untouched(what):
    for res in _spawn("refused", 2, what = what):
        expected = "tied input / output" if what == "tied" else "modules_to_save"
        assert expected in (res["message"] or ""), res["message"]
        assert res["untouched"]


def test_vocab_parallel_head_is_off_by_default():
    assert tpn.TPConfig().vocab_parallel_head is False


def test_vocab_parallel_head_refuses_a_forward_reading_the_whole_weight():
    # A fused loss given lm_head.weight directly (unsloth_zoo compiled forwards) would see one vocab shard.
    model = build(QWEN3, {}, lora = False)
    inner = tpn._unwrap(model)

    class Direct(type(inner)):
        def forward(
            self,
            input_ids = None,
            **kwargs,
        ):
            hidden = self.model(input_ids = input_ids).last_hidden_state
            return hidden @ self.lm_head.weight.t()

    inner.__class__ = Direct
    head, embed = inner.get_output_embeddings(), inner.get_input_embeddings()
    reason = tpn._vocab_head_refusal(model, head, embed, 2, tpn._tp_chain(model))
    assert "reads lm_head.weight directly" in (reason or ""), reason
    inner.__class__ = Direct.__mro__[1]
    assert tpn._vocab_head_refusal(model, head, embed, 2, tpn._tp_chain(model)) is None
    # Unsloth's own causal LM forward dispatches to vocab_parallel_cross_entropy, so it is not refused.
    with open(os.path.join(os.path.dirname(tpn.__file__), "llama.py")) as f:
        source = f.read()
    assert "lm_head.weight" in source and "_unsloth_vocab_parallel" in source
    assert "vocab_parallel_cross_entropy(" in source


def test_vocab_parallel_loss_only_while_the_head_is_sharded(monkeypatch):
    # tp_for_inference swaps the whole weight in but keeps the flag: the loss must take the one-GPU path.
    head = torch.nn.Linear(4, 8, bias = False)
    head._unsloth_vocab_parallel = True
    assert tpn.vocab_parallel_active(head) is False
    monkeypatch.setattr(tpn, "_is_dtensor", lambda weight: True)
    assert tpn.vocab_parallel_active(head) is True
    head._unsloth_vocab_parallel = False
    assert tpn.vocab_parallel_active(head) is False


def test_llama_loss_checks_the_head_is_still_sharded():
    import ast
    import pathlib

    source = pathlib.Path(tpn.__file__).with_name("llama.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    guarded = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and "vocab_parallel_cross_entropy" in ast.unparse(node)
        and "_vocab_parallel_active" in ast.unparse(node.test)
    ]
    assert guarded, "the vocab-parallel loss branch must require a still-sharded head"
