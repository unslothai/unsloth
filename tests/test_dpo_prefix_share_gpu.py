# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DPO prefix share on a real Unsloth model (CUDA).

* Per-sequence chosen / rejected logps and LoRA grads from the shared-prefix forward match the
  unshared padded forward in fp32 (<= 1e-4 nats per sequence) over many layouts, in train and eval
  mode, through TRL's fused LM head (TRL >= 1.15) or the plain LM head (older TRL).
* With training dropout on (LoRA dropout p > 0), the forward stays unshared; eval mode still shares.
* DPOTrainer (and ORPO / CPO / SimPO where Unsloth patches them) runs the decoder on P + 2C tokens
  per pair, not 2 (P + C) (fails without the feature), and its losses match
  UNSLOTH_DPO_PREFIX_SHARE=0 step for step.

Model: UNSLOTH_DPS_TEST_MODEL (default trl-internal-testing/tiny-Qwen3ForCausalLM). fp32 runs pin dense
attention to SDPA (see _pin_fp32_attention); UNSLOTH_DPS_TEST_REAL_ATTENTION=1 keeps the real backend."""

import gc
import importlib
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
    pinned = _pin_fp32_attention()
    try:
        yield model, tok
    finally:
        _unpin(pinned)
        del model
        gc.collect()


def _pin_fp32_attention():
    """The fp32 comparisons need an fp32 reference. A training forward has no padding mask, so Unsloth's
    attention takes FlashAttention or xformers when installed; neither has an fp32 kernel on every card
    (xformers on sm_100+ runs only its fp16 / bf16 flash op), and the fp32 Q / K / V are downcast to
    bf16 there. The unshared train-mode reference is then bf16-accurate (LoRA grads 6% off an fp64
    oracle on tiny Qwen3 on a B200), while eval mode keeps the padding mask and SDPA. Pin dense
    attention to SDPA for fp32 runs; bf16 runs (and UNSLOTH_DPS_TEST_REAL_ATTENTION=1) keep the real
    backend."""
    from unsloth.utils import attention_dispatch as ad

    if DTYPE != "float32" or os.environ.get("UNSLOTH_DPS_TEST_REAL_ATTENTION") == "1":
        return {}
    pinned = {}
    for flag in ("HAS_XFORMERS", "HAS_FLASH_ATTENTION"):
        if getattr(ad, flag, False):
            pinned[flag] = getattr(ad, flag)
            setattr(ad, flag, False)
    print(
        f"DPS_PIN fp32 reference: dense attention on SDPA (was {sorted(pinned) or 'SDPA already'})"
    )
    return pinned


def _unpin(pinned):
    from unsloth.utils import attention_dispatch as ad
    for flag, value in pinned.items():
        setattr(ad, flag, value)


def _fused_head_available():
    try:
        from trl.trainer.utils import add_fused_lm_head  # noqa: F401  TRL >= 1.15
    except Exception:
        return False
    return True


def _fused(model):
    """Add TRL's fused LM head when this TRL has one; _logps falls back to the plain head."""
    if not _fused_head_available():
        return False
    from trl.trainer.utils import add_fused_lm_head

    base = model.get_base_model()
    if not getattr(base, "_dps_test_fused", False):
        add_fused_lm_head(base)
        base._dps_test_fused = True
    return True


def _batch(
    rng,
    B,
    vocab,
    pad_id,
    spec = None,
):
    """[chosen_0 .. chosen_{B-1}, rejected_0 .. rejected_{B-1}] right-padded rows, the layout of TRL's
    preference collators, built here so it does not depend on the TRL version."""
    import torch

    rows = [None] * (2 * B)
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
        rows[i] = (prompt, chosen)
        rows[B + i] = (prompt, rejected)
    L = max(len(p) + len(c) for p, c in rows)
    ids = torch.full((2 * B, L), pad_id, dtype = torch.long)
    am = torch.zeros((2 * B, L), dtype = torch.long)
    cm = torch.zeros((2 * B, L), dtype = torch.long)
    for r, (p, c) in enumerate(rows):
        n = len(p) + len(c)
        ids[r, :n] = torch.tensor(p + c)
        am[r, :n] = 1
        cm[r, len(p) : n] = 1
    return {"input_ids": ids, "attention_mask": am, "completion_mask": cm}


def _plain_head_logps(model, ids, am, labels):
    """Per-token logps through the model's own LM head, one row at a time and recomputed in backward,
    so the [rows, L, vocab] logits are never all alive."""
    import torch
    import torch.utils.checkpoint as checkpoint

    causal_lm = model.get_base_model()
    hidden = causal_lm.model(input_ids = ids, attention_mask = am, use_cache = False)[0]
    head = causal_lm.get_output_embeddings()

    def row_logps(h, target):
        logp = head(h[:-1]).float().log_softmax(-1)
        keep = target[1:] != -100
        picked = logp.gather(-1, target[1:].clamp_min(0).unsqueeze(-1)).squeeze(-1)
        return torch.nn.functional.pad(picked * keep, (0, 1))

    rows = [
        checkpoint.checkpoint(row_logps, hidden[r], labels[r], use_reentrant = False)
        for r in range(ids.shape[0])
    ]
    return torch.stack(rows)


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
        if getattr(model.get_base_model(), "_dps_test_fused", False):
            log_probs = model(
                input_ids = ids,
                attention_mask = am,
                labels = labels,
                fused_lm_head = True,
                use_cache = False,
            ).log_probs
        else:
            log_probs = _plain_head_logps(model, ids, am, labels)
    seq = log_probs.sum(dim = 1)
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


_PREFERENCE_METHODS = ("_compute_loss", "compute_ref_log_probs", "concatenated_forward")


def _preference_trainer(kind):
    """(trainer class, config class, wrapped method names) of ``kind`` as this TRL exposes it."""
    import trl

    name = {"dpo": "DPO", "orpo": "ORPO", "cpo": "CPO", "simpo": "CPO"}[kind]
    try:
        trainer_cls = getattr(trl, f"{name}Trainer")
        config_cls = getattr(trl, f"{name}Config")
    except Exception:
        # TRL >= 1.0 moved ORPO / CPO to trl.experimental.
        try:
            module = importlib.import_module(f"trl.experimental.{name.lower()}")
            trainer_cls = getattr(module, f"{name}Trainer")
            config_cls = getattr(module, f"{name}Config")
        except Exception as exc:
            pytest.skip(f"TRL {trl.__version__} has no {name}Trainer ({exc!r})")
    wrapped = [
        m
        for m in _PREFERENCE_METHODS
        if getattr(getattr(trainer_cls, m, None), "_unsloth_dpo_prefix_share", False)
    ]
    if not wrapped:
        pytest.skip(f"Unsloth does not patch {name}Trainer on TRL {trl.__version__}")
    return trainer_cls, config_cls, wrapped


@pytest.mark.parametrize("kind", ["dpo", "orpo", "cpo", "simpo"])
def test_preference_trainer_runs_shared_and_matches(loaded, tmp_path, monkeypatch, kind):
    """Flip test: without the feature the decoder sees 2B x L tokens per forward."""
    monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE_MIN_SAVED", "0")
    monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE_TOKR", "1.0")
    import torch
    from datasets import Dataset
    from unsloth.models import dpo_prefix_share as dps

    trainer_cls, config_cls, wrapped = _preference_trainer(kind)
    legacy = "_compute_loss" not in wrapped  # TRL < 1.0: concatenated_forward
    model, tok = loaded
    rng = random.Random(5)
    vocab = min(len(tok), 30000)
    P, C, B = 256, 64, 2

    def text(n):
        return tok.decode([rng.randrange(1000, vocab) for _ in range(n)])

    rows = [{"prompt": text(P), "chosen": text(C), "rejected": text(C)} for _ in range(B * 3)]
    ds = Dataset.from_list(rows)
    init = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def pad64(features, collate):
        # A/A yardstick: 64 more right-pad columns on the completion side (other kernel tiling, same math).
        out = collate(features)
        for key, value in list(out.items()):
            if not (torch.is_tensor(value) and value.dim() == 2):
                continue
            if key in ("input_ids", "attention_mask", "completion_mask") or key.startswith(
                ("chosen_", "rejected_")
            ):
                fill = (
                    tok.pad_token_id
                    if key.endswith("input_ids")
                    else (-100 if key.endswith("labels") else 0)
                )
                out[key] = torch.nn.functional.pad(value, (0, 64), value = fill)
        return out

    def run(mode, aa = False):
        os.environ["UNSLOTH_DPO_PREFIX_SHARE"] = mode
        if "ref" in getattr(model, "peft_config", {}):
            model.delete_adapter(
                "ref"
            )  # TRL >= 1.0 DPOTrainer adds a frozen copy of the adapter each time
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in init:
                    p.copy_(init[n])
        seen = []
        layer0 = model.get_base_model().model.layers[0]
        h = layer0.register_forward_pre_hook(
            lambda m, a, k: seen.append(int((a[0] if a else k["hidden_states"]).shape[:2].numel())),
            with_kwargs = True,
        )
        extra = (
            {"max_length": None} if not legacy else {"max_length": 1024, "max_prompt_length": 512}
        )
        if kind == "simpo":
            extra.update(loss_type = "simpo", cpo_alpha = 0.0, simpo_gamma = 0.5)
        cfg = config_cls(
            output_dir = str(tmp_path / f"{mode}{'_aa' if aa else ''}"),
            per_device_train_batch_size = B,
            gradient_accumulation_steps = 1,
            max_steps = 3,
            # SGD: Adam's first steps are ~lr * sign(g), which turns fp32 noise in near-zero grad
            # coordinates into full-size updates and makes A/A runs drift apart by step 3.
            learning_rate = 1e-3,
            optim = "sgd",
            logging_steps = 1,
            report_to = "none",
            beta = 0.1,
            seed = 0,
            save_strategy = "no",
            bf16 = False,
            fp16 = False,
            **extra,
        )
        kwargs = {"ref_model": None} if kind == "dpo" else {}
        trainer = trainer_cls(
            model = model, args = cfg, train_dataset = ds, processing_class = tok, **kwargs
        )
        if aa:
            collate = trainer.data_collator
            trainer.data_collator = lambda features: pad64(features, collate)
        engaged0 = dps.ENGAGED_COUNT
        try:
            trainer.train()
        finally:
            h.remove()
            # accelerate leaves the model's forward wrapped in the trainer's autocast (Unsloth runs an fp32
            # model's preference training under bf16 autocast); later fp32 tests need the plain forward.
            trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper = False)
        losses = [x["loss"] for x in trainer.state.log_history if "loss" in x]
        return losses, seen, dps.ENGAGED_COUNT - engaged0

    try:
        base_losses, base_seen, e0 = run("0")
        aa_losses, _, _ = run("0", aa = True)
        pg_losses, pg_seen, e1 = run("1")
    finally:
        os.environ.pop("UNSLOTH_DPO_PREFIX_SHARE", None)
        if "ref" in getattr(model, "peft_config", {}):
            model.delete_adapter("ref")
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in init:
                    p.copy_(init[n])
    pg_diff = max(abs(a - b) for a, b in zip(base_losses, pg_losses))
    aa_diff = max(abs(a - b) for a, b in zip(base_losses, aa_losses))
    # transformers 4.x logs the loss rounded to 4 decimals.
    rounded = all(abs(x * 1e4 - round(x * 1e4)) < 1e-6 for x in base_losses + pg_losses)
    print(
        f"DPS_E2E {kind} legacy={legacy} base_seen={base_seen[:4]} pg_seen={pg_seen[:4]} engaged={e1} "
        f"pg_diff={pg_diff:.3e} aa_diff={aa_diff:.3e} base={base_losses} aa={aa_losses} pg={pg_losses}"
    )
    assert len(base_losses) == len(pg_losses) == 3
    assert e0 == 0 and e1 >= 3, (e0, e1)
    # Every decoder call but the one verify forward ran on the shared stream (one prompt per pair).
    assert min(base_seen) >= 2 * B * (P // 2), base_seen
    assert sum(n > min(base_seen) - B * (P // 2) for n in pg_seen) <= 1, (pg_seen, base_seen)
    if kind == "dpo" and not rounded:
        assert base_losses[0] == pg_losses[0]  # policy == reference at step 0: log 2 both ways
    tol = 3 * aa_diff + 1e-6 * max(1.0, abs(base_losses[0])) + (1.01e-4 if rounded else 0.0)
    assert pg_diff <= tol, (pg_diff, aa_diff)


def test_training_dropout_keeps_the_forward_unshared(loaded, monkeypatch):
    """Train mode with LoRA dropout (p > 0): the forward must stay unshared. A shared prompt would give
    the chosen and rejected rows one dropout mask, and the log-sigmoid of their difference makes that
    correlation change the expected DPO gradient. Eval mode (dropout inactive) still shares."""
    import torch
    from unsloth.models import dpo_prefix_share as dps

    monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE_TOKR", "1.0")
    monkeypatch.setenv("UNSLOTH_DPO_PREFIX_SHARE_MIN_SAVED", "0")
    model, tok = loaded
    _fused(model)
    batch = _batch(
        random.Random(21),
        2,
        min(len(tok), 30000),
        tok.pad_token_id,
        [(200, 40, 50, 0), (120, 60, 30, 0)],
    )
    swapped = []
    for module in model.modules():
        drop = getattr(module, "lora_dropout", None)
        if isinstance(drop, torch.nn.ModuleDict) and "default" in drop:
            swapped.append((drop, drop["default"]))
            drop["default"] = torch.nn.Dropout(0.3)
    assert swapped
    model.__dict__.pop("_unsloth_dps_dropouts", None)
    for m in model.modules():
        m.__dict__.pop("_unsloth_dps_dropouts", None)
    try:
        declined = dps.DECLINED_COUNT
        _, _, n_train = _logps(model, batch, shared = True, train = True)
        assert n_train == 0, n_train
        assert dps.DECLINED_COUNT > declined
        _, _, n_eval = _logps(model, batch, shared = True, train = False)
        assert n_eval == 1, n_eval
    finally:
        for drop, original in swapped:
            drop["default"] = original
        for m in model.modules():
            m.__dict__.pop("_unsloth_dps_dropouts", None)
        model.train()


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


_NON_REENTRANT_FIRST_CALL = r"""
import importlib.util, sys, types, torch
root, sub = sys.argv[1], "_dps_nr_pkg"
for name, path in ((sub, root), (sub + ".utils", root + "/utils")):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
spec = importlib.util.spec_from_file_location(sub + ".utils.pgk", root + "/utils/prefix_grouper_kernel.py")
pgk = importlib.util.module_from_spec(spec); sys.modules[spec.name] = pgk; spec.loader.exec_module(pgk)
seg, _ = pgk.build_seg_info_multigroup([(150, [40, 50]), (90, [30, 20])], torch.device("cuda"))
g = torch.Generator(device = "cuda").manual_seed(0)
q, k, v = (torch.randn(1, seg.T, 4, 64, device = "cuda", generator = g, dtype = torch.bfloat16).requires_grad_(True) for _ in range(3))
def attn(q, k, v):
    return pgk.flex_shared_prefix_attention(q, k, v, seg).float().square().sum()
# Cold first call inside a non-reentrant checkpoint (DDP / use_reentrant=False layers).
loss = torch.utils.checkpoint.checkpoint(attn, q, k, v, use_reentrant = False)
grads = torch.autograd.grad(loss, (q, k, v))
ref = torch.autograd.grad(attn(q, k, v), (q, k, v))
err = max(float((a - b).abs().max()) for a, b in zip(grads, ref))
print("NR_FIRST_CALL failure=%s err=%.3e" % (pgk.compile_check_failed(), err))
assert pgk.compile_check_failed() is None and err == 0.0
"""


def test_first_call_check_inside_non_reentrant_checkpoint(tmp_path):
    # Fresh process: the first-call compile check must be cold, and run inside the checkpoint.
    import subprocess
    import sys
    from pathlib import Path

    import torch

    if not pgk_supported():
        pytest.skip("PrefixGrouper needs torch >= 2.7")
    script = tmp_path / "nr_first_call.py"
    script.write_text(_NON_REENTRANT_FIRST_CALL)
    root = str(Path(__file__).resolve().parents[1] / "unsloth")
    out = subprocess.run(
        [sys.executable, str(script), root], capture_output = True, text = True, timeout = 900
    )
    print(out.stdout[-2000:], out.stderr[-4000:])
    assert out.returncode == 0, out.stderr[-4000:]
    assert "NR_FIRST_CALL failure=None err=0.000e+00" in out.stdout


def pgk_supported():
    import re

    import torch

    match = re.match(r"(\d+)\.(\d+)", torch.__version__)
    return match is None or (int(match.group(1)), int(match.group(2))) >= (2, 7)
