# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DPO prefix share on a real Unsloth model (CUDA, TRL >= 1.15 fused LM head).

* Per-sequence chosen / rejected logps and LoRA grads from the shared-prefix forward match the
  unshared padded forward in fp32 (<= 1e-4 nats per sequence) over many layouts.
* DPOTrainer runs the decoder on P + 2C tokens per pair, not 2 (P + C) (fails without the feature),
  and its losses match UNSLOTH_DPO_PREFIX_SHARE=0 step for step.

Model: UNSLOTH_DPS_TEST_MODEL (default trl-internal-testing/tiny-Qwen3ForCausalLM)."""

import gc
import os
import random
import types

import pytest

try:
    from real_accelerator import has_real_cuda
except Exception:  # run as a plain script

    def has_real_cuda():
        import torch
        return torch.cuda.is_available()


pytestmark = pytest.mark.skipif(not has_real_cuda(), reason = "needs CUDA")

MODEL = os.environ.get("UNSLOTH_DPS_TEST_MODEL", "trl-internal-testing/tiny-Qwen3ForCausalLM")
DTYPE = os.environ.get("UNSLOTH_DPS_TEST_DTYPE", "float32")
TOL = float(os.environ.get("UNSLOTH_DPS_TEST_TOL", "1e-4"))
GRAD_TOL = float(os.environ.get("UNSLOTH_DPS_TEST_GRAD_TOL", "1e-3"))


@pytest.fixture(scope = "module")
def loaded():
    os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")
    import unsloth  # noqa: F401
    from unsloth import FastLanguageModel
    import torch

    dtype = getattr(torch, DTYPE)
    model, tok = FastLanguageModel.from_pretrained(
        MODEL, max_seq_length = 2048, load_in_4bit = False, dtype = dtype
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 8,
        lora_alpha = 16,
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        use_gradient_checkpointing = "unsloth",
        random_state = 0,
    )
    g = torch.Generator(device = "cpu").manual_seed(0)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:  # a non-trivial adapter, so the policy differs from the reference
                p.copy_((torch.randn(p.shape, generator = g) * 0.02).to(p))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    yield model, tok
    del model
    gc.collect()


def _fused(model):
    from trl.trainer.utils import add_fused_lm_head
    base = model.get_base_model()
    if not getattr(base, "_dps_test_fused", False):
        add_fused_lm_head(base)
        base._dps_test_fused = True


def _batch(
    rng,
    B,
    vocab,
    pad_id,
    spec = None,
):
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    examples = []
    for i in range(B):
        if spec is not None:
            P, Cc, Cr, share = spec[i]
        else:
            P, Cc, Cr, share = rng.randint(1, 300), rng.randint(1, 200), rng.randint(1, 200), 0
        prompt = [rng.randrange(10, vocab) for _ in range(P)]
        head = [rng.randrange(10, vocab) for _ in range(share)]
        chosen = head + [rng.randrange(10, vocab) for _ in range(Cc)]
        rejected = (
            list(chosen) if Cr == "same" else head + [rng.randrange(10, vocab) for _ in range(Cr)]
        )
        examples.append({"prompt_ids": prompt, "chosen_ids": chosen, "rejected_ids": rejected})
    return DataCollatorForPreference(pad_token_id = pad_id)(examples)


def _logps(
    model,
    batch,
    shared,
    train = None,
):
    import torch
    from unsloth.models import dpo_prefix_share as dps

    if train is not None:
        model.train(train)

    trainer = types.SimpleNamespace(model = model, ref_model = None, args = types.SimpleNamespace())
    ids = batch["input_ids"].cuda()
    am = batch["attention_mask"].cuda()
    cm = batch["completion_mask"].cuda()
    labels = ids.masked_fill(cm == 0, -100)
    model.zero_grad(set_to_none = True)
    before = dps.ENGAGED_COUNT
    ctx = dps.preference_prefix_share(trainer, model) if shared else _null()
    with ctx:
        out = model(
            input_ids = ids, attention_mask = am, labels = labels, fused_lm_head = True, use_cache = False
        )
    seq = out.log_probs.sum(dim = 1)
    w = torch.linspace(0.5, 1.5, seq.numel(), device = seq.device, dtype = seq.dtype)
    (seq * w).sum().backward()
    grads = {
        n: p.grad.detach().float().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    return seq.detach().float(), grads, dps.ENGAGED_COUNT - before


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _layouts():
    rng = random.Random(0)
    cases = [(f"rand{i}", None, rng.choice([1, 2, 4])) for i in range(20)]
    cases += [
        ("one_token_prompt", [(1, 50, 60, 0)], 1),
        ("one_token_completions", [(200, 1, 1, 0)], 1),
        ("identical_completions", [(64, 40, "same", 0), (30, 10, 12, 0)], 2),
        ("completions_share_head", [(100, 20, 30, 9)], 1),
        ("t_exactly_128", [(64, 32, 32, 0)], 1),  # flat T = 128
        ("t_129", [(65, 32, 32, 0)], 1),
        ("rows_cross_128", [(120, 9, 7, 0), (127, 1, 2, 0)], 2),
        ("long", [(900, 300, 280, 0), (1000, 120, 300, 0)], 2),
        ("uneven", [(5, 300, 1, 0), (400, 2, 250, 0), (37, 77, 78, 0), (300, 300, 300, 0)], 4),
    ]
    return cases


_CASES = _layouts()


@pytest.mark.parametrize("train", [True, False], ids = ["train", "eval"])
@pytest.mark.parametrize("name,spec,B", _CASES, ids = [f"{c[0]}_{i}" for i, c in enumerate(_CASES)])
def test_logps_and_grads_match_unshared(loaded, name, spec, B, train, monkeypatch):
    import torch

    monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE_TOKR", "1.0")  # exercise every layout, gate aside
    monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE_MIN_SAVED", "0")

    model, tok = loaded
    _fused(model)
    rng = random.Random(sum(map(ord, name)) * 7919)
    batch = _batch(rng, B, min(len(tok), 30000), tok.pad_token_id, spec)
    ref, g_ref, n0 = _logps(model, batch, shared = False, train = train)
    got, g_got, n1 = _logps(model, batch, shared = True, train = train)
    # A/A noise floor of the unshared path: the same rows with 64 more pad columns.
    padded = {
        k: torch.nn.functional.pad(v, (0, 64), value = tok.pad_token_id if k == "input_ids" else 0)
        for k, v in batch.items()
    }
    aa, g_aa, _ = _logps(model, padded, shared = False, train = train)
    aa_err = float((ref - aa).abs().max())
    aa_gerr = max(
        float((g_ref[k] - g_aa[k]).abs().max() / g_ref[k].abs().max().clamp_min(1e-12))
        for k in g_ref
    )
    assert n0 == 0 and n1 == 1, (n0, n1)
    err = float((ref - got).abs().max())
    # fp32 noise: a sum of |logp| ~ 1e3 is not resolved below its ulp (1.2e-4 at 1024 - 2048), and
    # the unshared forward itself moves by aa_err when only its padding changes.
    ulp = float(torch.finfo(torch.float32).eps) * float(ref.abs().max())
    tol = max(TOL, 4 * ulp, 2 * aa_err)
    gerr = max(
        float((g_ref[k] - g_got[k]).abs().max() / g_ref[k].abs().max().clamp_min(1e-12))
        for k in g_ref
    )
    print(
        f"DPS_EXACT {name} train={train} B={B} max_abs_logp_err={err:.3e} max_rel_grad_err={gerr:.3e} "
        f"ulps={err / max(ulp, 1e-30):.2f} aa_logp_err={aa_err:.3e} aa_grad_err={aa_gerr:.3e} logps={ref.tolist()[:4]}"
    )
    assert err <= tol, (name, err, tol)
    assert gerr <= GRAD_TOL, (name, gerr)


def test_dpo_trainer_runs_shared_and_matches(loaded, tmp_path, monkeypatch):
    """Flip test: without the feature the decoder sees 2B x L tokens per forward."""
    monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE_MIN_SAVED", "0")
    import torch
    from datasets import Dataset
    from trl import DPOConfig, DPOTrainer
    from unsloth.models import dpo_prefix_share as dps

    model, tok = loaded
    rng = random.Random(5)
    vocab = min(len(tok), 30000)
    P, C, B = 256, 64, 2

    def text(n):
        return tok.decode([rng.randrange(1000, vocab) for _ in range(n)])

    rows = [{"prompt": text(P), "chosen": text(C), "rejected": text(C)} for _ in range(B * 3)]
    ds = Dataset.from_list(rows)
    init = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def run(
        mode,
        perturb = 0.0,
        pad = None,
    ):
        os.environ["UNSLOTH_DPO_PREFIX_SHARE"] = mode
        if "ref" in getattr(model, "peft_config", {}):
            model.delete_adapter("ref")  # DPOTrainer adds a frozen copy of the adapter each time
        g = torch.Generator(device = "cpu").manual_seed(1)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in init:
                    noise = torch.randn(p.shape, generator = g).to(p) * perturb
                    p.copy_(init[n] * (1 + noise))
        seen = []
        layer0 = model.get_base_model().model.layers[0]
        h = layer0.register_forward_pre_hook(
            lambda m, a, k: seen.append(int((a[0] if a else k["hidden_states"]).shape[:2].numel())),
            with_kwargs = True,
        )
        cfg = DPOConfig(
            output_dir = str(tmp_path / mode),
            per_device_train_batch_size = B,
            max_steps = 3,
            # SGD: Adam's first steps are ~lr * sign(g), which turns fp32 noise in near-zero grad
            # coordinates into full-size updates and makes A/A runs drift apart by step 3.
            learning_rate = 1e-3,
            optim = "sgd",
            logging_steps = 1,
            report_to = "none",
            max_length = None,
            beta = 0.1,
            seed = 0,
            save_strategy = "no",
            bf16 = False,
            fp16 = False,
            **({"pad_to_multiple_of": pad} if pad else {}),
        )
        trainer = DPOTrainer(
            model = model, ref_model = None, args = cfg, train_dataset = ds, processing_class = tok
        )
        batch = trainer.data_collator([trainer.train_dataset[i] for i in range(B)])
        engaged0 = dps.ENGAGED_COUNT
        trainer.train()
        h.remove()
        losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        return losses, seen, batch, dps.ENGAGED_COUNT - engaged0

    try:
        base_losses, base_seen, batch, e0 = run("0")
        # A/A yardstick: the unshared run with other padding (other kernel tiling: fp32-level noise of
        # the same size as the shared forward's, which SGD on DPO at pi = pi_ref then amplifies).
        aa_losses, _, _, _ = run("0", pad = 256)
        pg_losses, pg_seen, _, e1 = run("1")
    finally:
        os.environ.pop("UNSLOTH_DPO_PREFIX_SHARE", None)
    real = int(batch["attention_mask"].sum())
    pg_diff = max(abs(a - b) for a, b in zip(base_losses, pg_losses))
    aa_diff = max(abs(a - b) for a, b in zip(base_losses, aa_losses))
    print(
        f"DPS_E2E real={real} base_seen={base_seen[:3]} pg_seen={pg_seen[:3]} engaged={e1} "
        f"pg_diff={pg_diff:.3e} aa_diff={aa_diff:.3e} base={base_losses} aa={aa_losses} pg={pg_losses}"
    )
    assert e0 == 0 and e1 >= 6  # policy + reference forward per step
    # Every decoder call ran on the shared stream (one prompt per pair), except the one verify forward.
    assert min(base_seen) >= 2 * B * P, base_seen
    assert sum(n > real - B * (P // 2) for n in pg_seen) <= 1, (pg_seen, real)
    assert base_losses[0] == pg_losses[0]
    assert pg_diff <= 3 * aa_diff + 1e-6 * max(1.0, abs(base_losses[0])), (pg_diff, aa_diff)


def test_leaky_mask_is_caught(loaded, monkeypatch):
    """Fault injection: completions that see their sibling must move the logps and trip the verify."""
    import torch
    from unsloth.models import dpo_prefix_share as dps
    from unsloth.utils import prefix_grouper_kernel as pgk

    model, tok = loaded
    _fused(model)

    def leaky(group_of_kv, is_prefix, suffix_of_kv):
        def mask_mod(b, h, q_idx, kv_idx):
            return (group_of_kv[q_idx] == group_of_kv[kv_idx]) & (kv_idx <= q_idx)

        return mask_mod

    batch = _batch(
        random.Random(11),
        2,
        min(len(tok), 30000),
        tok.pad_token_id,
        [(120, 60, 70, 0), (40, 90, 30, 0)],
    )
    monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE_MIN_SAVED", "0")
    ref, _, _ = _logps(model, batch, shared = False)
    backbone = model.get_base_model().model
    pgk.clear_block_mask_cache()
    monkeypatch.setattr(pgk, "_make_mask_mod", leaky)
    monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE_VERIFY", "0")
    try:
        got, _, n = _logps(model, batch, shared = True)
        err = float((ref - got).abs().max())
        print(f"DPS_FAULT leaky_mask engaged={n} max_abs_logp_err={err:.3e}")
        assert n == 1 and err > 1e-2, err
        # With the verify on, a fresh backbone flags the leak and falls back to the unshared result.
        monkeypatch.setattr(backbone, dps._VERIFIED, False, raising = False)
        monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE_VERIFY", "1")
        pgk.clear_block_mask_cache()
        got2, _, n2 = _logps(model, batch, shared = True)
        print(f"DPS_FAULT verify={dps.LAST_VERIFY} unsafe={getattr(backbone, dps._UNSAFE, False)}")
        assert getattr(backbone, dps._UNSAFE, False) and n2 == 0
        assert float((ref - got2).abs().max()) <= 1e-3
    finally:
        pgk.clear_block_mask_cache()
        for attr in (dps._UNSAFE,):
            if hasattr(backbone, attr):
                delattr(backbone, attr)
