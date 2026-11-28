# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Gradient accumulation, gradients and updates under every parallel mode against one big batch in plain torch.

The loss Unsloth trains (unsloth.ai/blog/gradient) is the sum of the token losses over every micro-batch of the
accumulation window and every data parallel replica, divided by the number of real target tokens in that
window. Here the micro-batches and the ranks hold unequal token counts (masked labels and right padding), and
each parallel run through SFTTrainer (accumulation 2, 2 optimizer steps) is compared with an oracle written in
plain torch on the same rows: one sequence at a time, summed cross entropy over the window's real targets, the
same clipping and AdamW. Logged losses, the first step's gradients and the adapters after both steps must match.
A one-process trainer run is checked against the same oracle, so a bug shared by every layout (a micro-batch
mean, a dropped count) fails too, not only one that makes the layouts disagree.

bfloat16 full finetuning: Adam updates below half a bfloat16 spacing only land through stochastic rounding, so
the fraction of weights an FSDP2 / TP run changed must match one process with the rounding on (and stand clear
of one process with it off), on torch AdamW and on bitsandbytes 8-bit AdamW, with the rounding counted on the
shards.

Fast by construction: one 2-rank gloo world (and one 4-rank world for the composed layouts) per module runs every
cell in turn, beside one process that runs the oracle and the one-process references; tiny 2-layer Llama.

QLoRA-shaped cells train float32 adapters over a frozen bfloat16 base (bitsandbytes 4-bit needs CUDA).
accelerate's FSDP2 is not a cell: a CPU launch runs accelerate as MULTI_CPU, which builds no FSDP plugin, so Unsloth
hands UNSLOTH_FSDP2=1 to native FSDP2 there (fsdp2_launch.reclaim_native_fsdp2_for_cpu_trainer).

``UNSLOTH_TEST_GA_FAULT=<name>`` (tests/_shared/ga_faults.py) injects a known bug in every process; each fails at
least one test here (lora = test_lora_matches_the_oracle, composed = test_composed_lora_matches_the_oracle,
one = test_one_process_matches_the_oracle, full = test_bf16_full_finetuning_changes_as_many_weights_as_one_process):

    drop_num_items          one, lora (every cell), composed (both)
    micro_mean              one, lora (every cell), composed (both)
    count_padding           one, lora (every cell), composed (both)
    per_rank_norm           lora[ddp, fsdp2_native, cp2, ep2, ddp / fsdp2_native frozen bf16 base], composed[fsdp2_tp2]
    skip_count_allreduce    lora[ddp, fsdp2_native, tp2, cp2, ddp_packed, ep2, every frozen bf16 base],
                            composed[fsdp2_tp2]
    cp_double_count         lora[cp2]
    tp_double_count         lora[tp2, pp2_*, tp2 frozen bf16 base], composed (both)
    pp_world_count          lora[pp2_gpipe, pp2_1f1b], composed[tp2_pp2]
    sr_skip_dtensor         full[fsdp2-*, tp2-*]
    sr_skip_bnb_dtensor     full[fsdp2-bnb8bit, tp2-bnb8bit]
    ep_double_expert_grads  lora[ep2] (transformers >= 5)

per_rank_norm leaves tp / pp alone (one data parallel replica: its count is the window's).
"""

import multiprocessing
import os

if os.environ.get("UNSLOTH_TEST_RANKS_ON_GPU") != "1" and (
    multiprocessing.parent_process() is not None
    or getattr(multiprocessing.current_process(), "_inheriting", False)
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import datetime  # noqa: E402
import importlib.metadata  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import test_parallel_cp_compose_gloo as cc  # noqa: E402
from test_tp_native_oracle import LLAMA  # noqa: E402

# One worker runs the module: its tests share the worlds spawned once below, and the first test of each world
# carries that spawn (about a minute on a loaded runner) inside its own timeout.
pytestmark = [pytest.mark.xdist_group("parallel_ga_parity_gloo"), pytest.mark.timeout(900)]

SEQ = cc.SEQ
GA = 2
STEPS = 2
ROWS_PER_STEP = 8  # every layout trains on the same 8 rows per optimizer step
PADDING = "right"  # rows of 16, 11, 7, 13, 9, 16, 12, 5 tokens: unequal counts per micro-batch and per rank
FULL_LR = (
    2e-5  # below half a bfloat16 spacing for most weights: round to nearest loses those updates
)

_TRANSFORMERS_MAJOR = int(importlib.metadata.version("transformers").split(".")[0])


def _pipelining_available():
    """Unsloth's pipeline parallel needs torch.distributed.pipelining with step(return_outputs) (torch >= 2.10)."""
    try:
        import inspect

        from torch.distributed.pipelining import ScheduleGPipe
    except ImportError:
        return False
    return "return_outputs" in inspect.signature(ScheduleGPipe.step).parameters


_PIPELINING = _pipelining_available()
needs_pipelining = pytest.mark.skipif(
    not _PIPELINING,
    reason = "pipeline parallel needs torch >= 2.10 (pipelining step(return_outputs))",
)

# name -> (environment, _case_train kwargs). bs is the per-rank micro-batch: data parallel ranks split the 8 rows,
# tp / pp / cp peers share them.
_LORA = dict(padding = PADDING, ga = GA, steps = STEPS)
WORLD2 = {
    "ddp": ({}, dict(bs = 2, **_LORA)),
    "fsdp2_native": ({"UNSLOTH_FSDP2": "1"}, dict(bs = 2, **_LORA)),
    "tp2": ({"UNSLOTH_TP_SIZE": "2"}, dict(bs = 4, **_LORA)),
    "pp2_gpipe": (
        {"UNSLOTH_PP_SIZE": "2", "UNSLOTH_PP_SCHEDULE": "gpipe"},
        dict(bs = 4, **_LORA),
    ),
    "pp2_1f1b": ({"UNSLOTH_PP_SIZE": "2", "UNSLOTH_PP_SCHEDULE": "1f1b"}, dict(bs = 4, **_LORA)),
    "cp2": ({}, dict(bs = 4, cp = 2, **_LORA)),
    # packed rows (several documents per row, positions restarting): only real targets count
    "ddp_packed": ({}, dict(bs = 1, ga = GA, steps = STEPS, packed = "padfree")),
}
# QLoRA-shaped: a frozen bfloat16 base under float32 adapters (bitsandbytes 4-bit needs CUDA)
for _name in ("ddp", "fsdp2_native", "tp2"):
    _env, _kw = WORLD2[_name]
    WORLD2[f"{_name}-frozen_bf16_base"] = (_env, dict(_kw, dtype = "bfloat16"))
if _TRANSFORMERS_MAJOR >= 5:  # expert parallel shards fused expert stacks (transformers >= 5)
    WORLD2["ep2"] = ({"UNSLOTH_EP_SIZE": "2"}, dict(bs = 2, arch = "qwen3_moe", **_LORA))

_FULL = dict(mode = "full", dtype = "bfloat16", lr = FULL_LR, weights_changed = True, ga = GA, steps = STEPS)
FULL_CELLS = {
    "ddp-torch": ({}, dict(bs = 2, optim = "adamw_torch", **_FULL)),
    "fsdp2-torch": ({"UNSLOTH_FSDP2": "1"}, dict(bs = 2, optim = "adamw_torch", **_FULL)),
    "tp2-torch": ({"UNSLOTH_TP_SIZE": "2"}, dict(bs = 4, optim = "adamw_torch", **_FULL)),
    "fsdp2-bnb8bit": ({"UNSLOTH_FSDP2": "1"}, dict(bs = 2, optim = "adamw_8bit", **_FULL)),
    "tp2-bnb8bit": ({"UNSLOTH_TP_SIZE": "2"}, dict(bs = 4, optim = "adamw_8bit", **_FULL)),
}
WORLD2.update({"full-" + k: v for k, v in FULL_CELLS.items()})

WORLD4 = {
    "fsdp2_tp2": ({"UNSLOTH_PARALLEL": "dp=2,tp=2,fsdp=1"}, dict(bs = 2, **_LORA)),
    "tp2_pp2": ({"UNSLOTH_PARALLEL": "tp=2,pp=2"}, dict(bs = 4, **_LORA)),
}


def _uses_pp(name):
    return name.startswith("pp") or name == "tp2_pp2"


def _params(cells):
    return [pytest.param(k, marks = needs_pipelining) if _uses_pp(k) else k for k in cells]


_LORA_NAMES = [k for k in WORLD2 if not k.startswith("full-")]
_COMPOSED_NAMES = list(WORLD4)
if not _PIPELINING:  # the worlds leave out what the tests skip
    WORLD2 = {k: v for k, v in WORLD2.items() if not _uses_pp(k)}
    WORLD4 = {k: v for k, v in WORLD4.items() if not _uses_pp(k)}

# One process: the trainer references and the oracle.
REFERENCE = {
    "lora": ({}, dict(bs = 4, **_LORA)),
    "packed": ({}, dict(bs = 1, ga = 2 * GA, steps = STEPS, packed = "padfree")),
    "frozen_bf16_base": ({}, dict(bs = 4, dtype = "bfloat16", **_LORA)),
}
if _TRANSFORMERS_MAJOR >= 5:
    REFERENCE["moe"] = ({}, dict(bs = 4, arch = "qwen3_moe", **_LORA))
for _optim in ("adamw_torch", "adamw_8bit"):
    REFERENCE[f"full-{_optim}"] = ({}, dict(bs = 4, optim = _optim, **_FULL))
    REFERENCE[f"full-{_optim}-nearest"] = (
        {"UNSLOTH_BF16_STOCHASTIC_ROUNDING": "0"},
        dict(bs = 4, optim = _optim, **_FULL),
    )

_KEEP = (
    "error",
    "trace",
    "phase",
    "losses",
    "grad_norms",
    "grads",
    "state",
    "layout",
    "is_tp",
    "is_ep",
    "stage",
    "native_fsdp2",
    "epoch",
    "rows_read",
    "seen_lengths",
    "ulysses_calls",
    "cp_group",
    "banner",
    "changed",
    "sr_covered",
    "sr_counters",
    "optimizer",
    "param_dtypes",
    "secs",
    "fault",
    "pp_schedules",
)


def _checkpoints(tmp):
    """Tiny 2-layer Llama (hidden 16, 4 heads, vocab 1024), and the tiny Qwen3-MoE of the cp suite."""
    from transformers import AutoTokenizer

    path = cc._save_checkpoint(
        "llama", os.path.join(tmp, "llama"), {"num_hidden_layers": 2, "vocab_size": 1024}
    )
    AutoTokenizer.from_pretrained(LLAMA).save_pretrained(path)
    out = {"llama": path}
    if _TRANSFORMERS_MAJOR >= 5:
        moe_tmp = os.path.join(tmp, "moe")
        os.makedirs(moe_tmp)
        out["qwen3_moe"] = cc._checkpoint(moe_tmp, "qwen3_moe")
    return out


def _reset_between_cells(env):
    """Each cell reads its layout from the environment as a fresh launch would."""
    for key in (*cc._ENV_KEYS, "UNSLOTH_BF16_STOCHASTIC_ROUNDING"):
        os.environ.pop(key, None)
    os.environ.update(env)
    from accelerate.state import AcceleratorState

    from unsloth import parallel_env
    from unsloth.models import tp_env

    AcceleratorState._reset_state(True)
    parallel_env.apply_parallel_env()
    tp_env._reset_env_layout()


def _row_data(n_rows):
    """The first n_rows dataset rows _case_train builds (seed 1)."""
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 1000, (n_rows, SEQ), generator = g)


def _sequences(rows, packed):
    """The step's rows as the token sequences one process would see, one at a time: a padded row cut to its
    length, a packed row split into its documents. Each is (input_ids, labels)."""
    ids = _row_data(max(rows) + 1)
    out = []
    for r in rows:
        x = ids[r : r + 1]
        batch = {"input_ids": x, "labels": x.clone()}
        batch["labels"][batch["labels"] % 5 == 0] = -100
        if packed:
            lengths = cc._PACKED_DOCS[packed][r % len(cc._PACKED_DOCS[packed])]
            start = 0
            for n in lengths:
                y = batch["labels"][:, start : start + n].clone()
                y[:, 0] = -100  # the packed collator masks the target across a document start
                out.append((x[:, start : start + n], y))
                start += n
        else:
            n = cc._PAD_LENGTHS[PADDING][r % len(cc._PAD_LENGTHS[PADDING])]
            out.append((x[:, :n], batch["labels"][:, :n]))
    return out


def _oracle(
    ckpt,
    packed = None,
    dtype = torch.float32,
):
    """Plain torch: per step, sum of every real target's cross entropy over the window / their count, clip at 1,
    AdamW as _case_train configures it. Returns losses, grad norms, the first step's grads and the final adapters."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype = dtype, attn_implementation = "sdpa"
    )
    model.config.use_cache = False
    torch.manual_seed(1)
    model = get_peft_model(
        model,
        LoraConfig(
            r = 4,
            lora_alpha = 8,
            target_modules = "all-linear",
            init_lora_weights = False,
            lora_dropout = 0.0,
        ),
    )
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr = 1e-2, eps = cc._ADAM_EPS, weight_decay = 0.0)
    rows_per_step = 2 * GA if packed else ROWS_PER_STEP  # packed: one row per micro-batch
    losses, norms, grads = [], [], None
    for step in range(STEPS):
        rows = list(range(step * rows_per_step, (step + 1) * rows_per_step))
        seqs = _sequences(rows, packed)
        total = sum(int((y[:, 1:] != -100).sum()) for _, y in seqs)
        loss_sum = 0.0
        for x, y in seqs:
            logits = model(input_ids = x).logits.float()
            loss = torch.nn.functional.cross_entropy(
                logits[0, :-1], y[0, 1:], ignore_index = -100, reduction = "sum"
            )
            (loss / total).backward()
            loss_sum += float(loss)
        losses.append(loss_sum / total)
        norms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0)))
        if grads is None:
            grads = {
                n.replace("module.", ""): p.grad.detach().clone()
                for n, p in model.named_parameters()
                if p.requires_grad
            }
        opt.step()
        opt.zero_grad(set_to_none = True)
    state = {k: v.detach().float().clone() for k, v in model.state_dict().items() if "lora_" in k}
    return {"losses": losses, "grad_norms": norms, "grads": grads, "state": state}


def _run_cells(rank, world, tmp, ckpts, cells):
    out = {}
    for name, (env, kwargs) in cells.items():
        _reset_between_cells(env)
        kwargs = dict(kwargs)
        arch = kwargs.pop("arch", "llama")
        start = time.time()
        try:
            res = cc._case_train(rank, world, tmp, ckpts[arch], arch = arch, **kwargs)
        except Exception as exc:
            res = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
        res["secs"] = time.time() - start
        out[name] = {k: res[k] for k in _KEEP if k in res}
    return out


def _worker(rank, world, port, tmp, ckpts, which):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    torch.set_num_threads(1)
    import torch.distributed as dist

    out = {}
    try:
        from tests._shared import ga_faults

        out["fault"] = ga_faults.install_on_import()
        error = cc._import_checkout_unsloth()
        if error is not None:
            out = {"skip": error}
        else:
            cc._import_checkout_unsloth = lambda: None  # imported once per process, not per cell
            if world > 1:
                # A cell that leaves a peer waiting fails in a minute, not at accelerate's 30.
                dist.init_process_group(
                    "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 90)
                )
            if which == "reference":
                out["cells"] = _run_cells(rank, world, tmp, ckpts, REFERENCE)
                out["oracle"] = _oracle(ckpts["llama"])
                out["oracle_packed"] = _oracle(ckpts["llama"], packed = "padfree")
                out["oracle_frozen_bf16_base"] = _oracle(ckpts["llama"], dtype = torch.bfloat16)
                if "qwen3_moe" in ckpts:
                    out["oracle_moe"] = _oracle(ckpts["qwen3_moe"])
            else:
                out["cells"] = _run_cells(rank, world, tmp, ckpts, WORLD2 if world == 2 else WORLD4)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()
    torch.save(out, os.path.join(tmp, f"{which}_{world}_{rank}.pt"))
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def _spawn_all(worlds):
    """One process per rank of every world in `worlds` (the reference is a world of one), all at once."""
    ctx = multiprocessing.get_context("spawn")
    results = {}
    with tempfile.TemporaryDirectory(prefix = "ga_parity_") as tmp:
        ckpts = _checkpoints(tmp)
        jobs = []
        for which, world in worlds:
            port = cc._free_port()
            for rank in range(world):
                process = ctx.Process(target = _worker, args = (rank, world, port, tmp, ckpts, which))
                jobs.append((which, world, rank, process))
        for *_, p in jobs:
            p.start()
        # Inside the module's timeout: a hung rank fails the module, not the job.
        deadline = time.time() + 600
        for *_, p in jobs:
            p.join(max(1, deadline - time.time()))
        for *_, p in jobs:
            if p.is_alive():
                p.kill()
                p.join(10)
        for which, world, rank, p in jobs:
            path = os.path.join(tmp, f"{which}_{world}_{rank}.pt")
            res = (
                torch.load(path, weights_only = False)
                if os.path.exists(path)
                else {"error": f"{which} world {world} rank {rank} exit code {p.exitcode}"}
            )
            results.setdefault((which, world), []).append(res)
    for ranks in results.values():
        if "skip" in ranks[0]:
            pytest.skip(f"import unsloth failed on this host: {ranks[0]['skip']}")
    return results


@pytest.fixture(scope = "module")
def worlds2():
    """The one-process reference and the 2-rank world, spawned side by side once per module."""
    return _spawn_all([("reference", 1), ("parallel", 2)])


@pytest.fixture(scope = "module")
def ref(worlds2):
    (out,) = worlds2[("reference", 1)]
    assert "error" not in out, (out.get("error"), out.get("trace"))
    return out


def _ok(ranks, world):
    for rank, res in enumerate(ranks):
        assert "error" not in res, (world, rank, res.get("error"), res.get("trace"))
    return ranks


@pytest.fixture(scope = "module")
def ranks2(worlds2):
    return _ok(worlds2[("parallel", 2)], 2)


@pytest.fixture(scope = "module")
def ranks4(ref):
    return _ok(_spawn_all([("parallel", 4)])[("parallel", 4)], 4)


def _cell(res, name, what):
    cell = res["cells"][name]
    assert "error" not in cell, (what, cell.get("error"), cell.get("trace"))
    return cell


def _matches_oracle(
    got,
    oracle,
    what,
    rel = 1e-4,
    atol = 1e-6,
):
    # HF logs the loss rounded to 4 decimals.
    assert got["losses"] == pytest.approx(oracle["losses"], rel = rel, abs = 1e-4), (
        what,
        got["losses"],
        oracle["losses"],
    )
    assert got["grad_norms"] == pytest.approx(oracle["grad_norms"], rel = rel), (
        what,
        got["grad_norms"],
        oracle["grad_norms"],
    )
    assert got["grads"], (what, "no gradient captured")
    for key, value in got["grads"].items():
        torch.testing.assert_close(
            value, oracle["grads"][key], rtol = rel * 10, atol = atol, msg = f"{what} grad {key}"
        )
    assert got["state"], (what, "no adapter captured")
    for key, value in got["state"].items():
        torch.testing.assert_close(
            value, oracle["state"][key], rtol = rel * 10, atol = atol, msg = f"{what} adapter {key}"
        )


def _matches_bf16_oracle(got, oracle, what):
    """A bfloat16 base: one sequence at a time and batched / sharded matmuls round differently, so each first-step
    gradient is checked against its tensor's largest entry. They agree to about 7% of it here; a wrong
    normaliser moves them by 45% or more."""
    assert got["losses"] == pytest.approx(oracle["losses"], rel = 1e-3), (
        what,
        got["losses"],
        oracle["losses"],
    )
    assert got["grads"], (what, "no gradient captured")
    for key, value in got["grads"].items():
        want = oracle["grads"][key]
        err = float((value - want).abs().max() / want.abs().max().clamp_min(1e-12))
        assert err < 0.2, (what, key, err)


def _engaged(name, cell, rank):
    """The layout under test is the one that ran (no silent fallback to DDP / one process)."""
    if name.startswith("tp2") or name == "fsdp2_tp2":
        assert cell["is_tp"], (name, cell.get("layout"))
    if name.startswith("pp2") or name == "tp2_pp2":
        assert cell["stage"] == rank % 2 if name.startswith("pp2") else cell["stage"] is not None, (
            name,
            cell.get("stage"),
        )
    if name.startswith("pp2_"):
        want = {"gpipe": ["ScheduleGPipe"], "1f1b": ["Schedule1F1B"]}[name.split("_")[1]]
        assert cell["pp_schedules"] == want, (name, cell.get("pp_schedules"))
    if name.startswith("fsdp2"):
        assert cell["native_fsdp2"], (name, cell.get("banner"))
    if name == "cp2":
        assert cell["cp_group"] == [0, 1] and cell["ulysses_calls"] > 0, (
            name,
            cell.get("cp_group"),
        )
    if name == "ep2":
        assert cell["is_ep"], name


def test_one_process_matches_the_oracle(ref):
    """The reference itself: one process accumulating 2 micro-batches of unequal length = one big batch."""
    _matches_oracle(_cell(ref, "lora", "one process"), ref["oracle"], "one process")
    _matches_oracle(_cell(ref, "packed", "one process packed"), ref["oracle_packed"], "packed")
    _matches_bf16_oracle(
        _cell(ref, "frozen_bf16_base", "one process bf16 base"),
        ref["oracle_frozen_bf16_base"],
        "one process bf16 base",
    )


def _oracle_for(ref, name):
    if "packed" in name:
        return ref["oracle_packed"]
    if name.startswith("ep"):
        return ref["oracle_moe"]
    if name.endswith("frozen_bf16_base"):
        return ref["oracle_frozen_bf16_base"]
    return ref["oracle"]


@pytest.mark.parametrize("name", _params(_LORA_NAMES))
def test_lora_matches_the_oracle(name, ref, ranks2):
    oracle = _oracle_for(ref, name)
    for rank, res in enumerate(ranks2):
        cell = _cell(res, name, (name, rank))
        _engaged(name, cell, rank)
        assert cell["epoch"] <= 1.0 + 1e-6, (name, "the dataloader wrapped", cell["epoch"])
        if name.endswith("frozen_bf16_base"):
            _matches_bf16_oracle(cell, oracle, (name, rank))
        else:
            _matches_oracle(cell, oracle, (name, rank))


@pytest.mark.parametrize("name", _params(_COMPOSED_NAMES))
def test_composed_lora_matches_the_oracle(name, ref, ranks4):
    for rank, res in enumerate(ranks4):
        cell = _cell(res, name, (name, rank))
        _engaged(name, cell, rank)
        _matches_oracle(cell, ref["oracle"], (name, rank))


def _fraction(cell):
    changed = sum(c for c, _ in cell["changed"].values())
    total = sum(n for _, n in cell["changed"].values())
    assert total > 0, cell["changed"]
    return changed / total


@pytest.mark.parametrize("name", list(FULL_CELLS))
def test_bf16_full_finetuning_changes_as_many_weights_as_one_process(name, ref, ranks2):
    """FSDP2 / TP shards step like one process under stochastic rounding: the fraction of bfloat16 weights two
    small-lr steps changed is one process's with the rounding on, not round to nearest's."""
    optim = "adamw_8bit" if name.endswith("bnb8bit") else "adamw_torch"
    single = _cell(ref, f"full-{optim}", "one process")
    nearest = _cell(ref, f"full-{optim}-nearest", "one process, rounding off")
    want, floor = _fraction(single), _fraction(nearest)
    assert single["sr_covered"] and single["param_dtypes"] == ["torch.bfloat16"], single
    assert not nearest["sr_covered"], nearest
    # The rounding is what moves these weights: round to nearest changes half as many (22% vs 11% here).
    assert want > 1.5 * floor, (want, floor)
    sharded = not name.startswith("ddp")
    for rank, res in enumerate(ranks2):
        cell = _cell(res, "full-" + name, (name, rank))
        _engaged(name.split("-")[0], cell, rank)
        got = _fraction(cell)
        assert cell["param_dtypes"] == ["torch.bfloat16"], (name, cell["param_dtypes"])
        assert cell["sr_covered"], (
            name,
            rank,
            "stochastic rounding covers no weight",
            cell["optimizer"],
        )
        if sharded:
            counters = cell["sr_counters"]
            assert (
                counters.get("_unsloth_sr_dtensor_rounds", 0)
                + counters.get("_unsloth_sr_bnb_fused_steps", 0)
                > 0
            ), (name, rank, counters)
        # Different shards draw different noise than one process: the same expected fraction, not bitwise.
        assert abs(got - want) < 0.25 * (want - floor), (name, rank, got, want, floor)
