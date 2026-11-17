# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PARALLEL=auto: DDP when the model and its training state fit on every rank, else native FSDP2.

* parsing: `auto` is its own value, an explicit layout parses and translates exactly as before, and what
  already fixes the layout (UNSLOTH_FSDP2, tp / pp / cp, an accelerate FSDP config) is refused;
* sizing: the safetensors headers of a tiny checkpoint, 16-bit vs 4-bit on load;
* on a gloo CPU world of 2: fits -> ddp, does not fit -> fsdp2, ranks disagree -> fsdp2 on every rank;
* end to end: a Trainer under auto trains exactly like the explicit layout it picked (UNSLOTH_FSDP2=0 / 1)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
STEPS = 2
_ENV = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_PARALLEL_AUTO_RESOLVED",
    "UNSLOTH_PARALLEL_AUTO_GB",
    "UNSLOTH_FSDP2",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "ACCELERATE_USE_FSDP",
    "FSDP_VERSION",
    "WORLD_SIZE",
)


def _modules():
    """unsloth.parallel_env / unsloth.parallel_auto from this tree without importing the GPU package."""
    if "unsloth" not in sys.modules:
        pkg = types.ModuleType("unsloth")
        pkg.__path__ = [os.path.join(REPO, "unsloth")]
        sys.modules["unsloth"] = pkg
    env = importlib.import_module("unsloth.parallel_env")
    auto = importlib.import_module("unsloth.parallel_auto")
    assert os.path.dirname(auto.__file__) == os.path.join(REPO, "unsloth"), auto.__file__
    return env, auto


@pytest.fixture
def clean(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    env, auto = _modules()
    auto._STATE.update(choice = None, reason = None)
    yield monkeypatch, env, auto
    auto._STATE.update(choice = None, reason = None)


def _snapshot():
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(TINY, allow_patterns = ["*.json", "*.safetensors"])
    except Exception as exc:  # offline host without the cached tiny repo
        pytest.skip(f"{TINY} unavailable: {exc}")


# --------------------------------------------------------------------------------------------------
# Parsing (no process group)
# --------------------------------------------------------------------------------------------------


def test_auto_parses_and_explicit_layouts_are_unchanged(clean):
    mp, env, auto = clean
    assert env.parse_parallel_spec("auto") == {"auto": True}
    assert env.parse_parallel_spec("  AUTO ") == {"auto": True}
    # Explicit layouts: the same spec and the same legacy variables as before auto existed.
    assert env.parse_parallel_spec("fsdp=1") == {"fsdp": True}
    assert env.parse_parallel_spec("fsdp=0") == {"fsdp": False}
    assert env.parse_parallel_spec("dp_shard=2") == {"dp": 2, "fsdp": True}
    assert env.translate(env.parse_parallel_spec("fsdp=1"), {"WORLD_SIZE": "2"}) == {
        "UNSLOTH_FSDP2": "1"
    }
    with pytest.raises(ValueError, match = "is not key=value"):
        env.parse_parallel_spec("auto,tp=2")
    # auto at import writes nothing: the decision waits for the model.
    environ = {"UNSLOTH_PARALLEL": "auto", "WORLD_SIZE": "2"}
    assert env.apply_parallel_env(environ) == {"auto": True}
    assert environ == {"UNSLOTH_PARALLEL": "auto", "WORLD_SIZE": "2"}
    assert auto.auto_requested(environ) and not auto.auto_requested({"UNSLOTH_PARALLEL": "fsdp=1"})


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"UNSLOTH_FSDP2": "1"}, "UNSLOTH_FSDP2=1 already does"),
        ({"UNSLOTH_FSDP2": "0"}, "UNSLOTH_FSDP2=0 already does"),
        ({"UNSLOTH_TP_SIZE": "2"}, "UNSLOTH_TP_SIZE=2 asks for another axis"),
        ({"UNSLOTH_CP_SIZE": "2"}, "UNSLOTH_CP_SIZE=2 asks for another axis"),
        ({"ACCELERATE_USE_FSDP": "true"}, "accelerate launch"),
        ({"UNSLOTH_PARALLEL_AUTO_GB": "lots"}, "number of GiB"),
    ],
)
def test_auto_refuses_a_layout_already_fixed(clean, extra, match):
    _, env, _ = clean
    with pytest.raises(ValueError, match = match):
        env.apply_parallel_env({"UNSLOTH_PARALLEL": "auto", "WORLD_SIZE": "2", **extra})


def test_child_process_inherits_the_decision(clean):
    _, env, auto = clean
    environ = {
        "UNSLOTH_PARALLEL": "auto",
        "UNSLOTH_FSDP2": "0",
        "UNSLOTH_PARALLEL_AUTO_RESOLVED": "ddp",
        "WORLD_SIZE": "2",
    }
    env.apply_parallel_env(environ)
    assert auto.resolved_choice() == "ddp"
    auto._STATE.update(choice = None)
    with pytest.raises(ValueError):  # a marker that disagrees with UNSLOTH_FSDP2 is not inherited
        env.apply_parallel_env({**environ, "UNSLOTH_PARALLEL_AUTO_RESOLVED": "fsdp2"})


def test_decision_and_state_arithmetic(clean):
    _, _, auto = clean
    gib = 1024**3
    # 5 GiB weights + 1 GiB state + 20% of 14 GiB = 8.8 GiB: fits 14 GiB, not 8 GiB.
    assert auto.decide(5 * gib, 1 * gib, 14 * gib)[0] is True
    assert auto.decide(5 * gib, 1 * gib, 8 * gib, total = 14 * gib)[0] is False
    assert auto.decide(0, 0, 0)[0] is False  # nothing sized: never DDP
    assert auto.training_state_bytes(10, 4, "adamw_torch") == 10 * (8 + 8)
    assert auto.training_state_bytes(10, 2, "adamw_8bit") == 10 * (4 + 2)


def test_checkpoint_headers_size_the_load(clean):
    _, _, auto = clean
    tensors = auto._checkpoint_tensors(_snapshot())
    assert tensors and all(len(t) == 3 for t in tensors)
    raw16 = sum(
        __import__("math").prod(shape) * min(auto._DTYPE_BYTES[dt], 2) for _, dt, shape in tensors
    )
    w16, lora16 = auto.checkpoint_estimate(tensors)
    w4, lora4 = auto.checkpoint_estimate(tensors, load_in_4bit = True)
    wfull, train_full = auto.checkpoint_estimate(tensors, full_finetuning = True, load_in_4bit = True)
    assert w16 == raw16
    assert w4 < w16 and lora4 == lora16 > 0
    assert wfull == w16  # full finetuning does not quantize
    assert train_full == sum(__import__("math").prod(s) for _, _, s in tensors)


def test_trainable_embedding_is_found(clean):
    _, _, auto = clean
    model = torch.nn.Sequential(torch.nn.Embedding(8, 4), torch.nn.Linear(4, 4))
    model[0].weight.requires_grad_(False)
    assert auto._trainable_embedding(model) is None
    model[0].weight.requires_grad_(True)
    assert auto._trainable_embedding(model) == "0.weight"


# --------------------------------------------------------------------------------------------------
# gloo world of 2
# --------------------------------------------------------------------------------------------------


def _init(rank, world, port):
    import datetime

    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    for key in ("UNSLOTH_FSDP2", "UNSLOTH_PARALLEL_AUTO_RESOLVED"):
        os.environ.pop(key, None)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )


def _decision_worker(rank, world, port, out, snapshot):
    """resolve_for_load on a from_pretrained-shaped loader, one case after another in one process group."""
    _init(rank, world, port)
    import torch.distributed as dist

    try:
        _load_native()
        # FSDP2 installs the Trainer wrapper through models._utils, which needs the GPU package.
        utils = types.ModuleType("unsloth.models._utils")
        utils.patch_gradient_accumulation_fix = lambda cls: None
        sys.modules["unsloth.models._utils"] = utils
        _, auto = _modules()

        def from_pretrained(
            model_name = "unused",
            load_in_4bit = True,
            full_finetuning = False,
            **kwargs,
        ):
            return model_name

        res = {}
        # GiB per rank: ample on both, ~1 KiB on both, ample on rank 0 only.
        for case, budgets in (
            ("fits", ("64", "64")),
            ("tight", ("1e-6", "1e-6")),
            ("split", ("64", "1e-6")),
        ):
            auto._STATE.update(choice = None, reason = None)
            os.environ.pop("UNSLOTH_FSDP2", None)
            os.environ.update(UNSLOTH_PARALLEL = "auto", UNSLOTH_PARALLEL_AUTO_GB = budgets[rank])
            if case == "fits":
                # Through the decorator every Unsloth from_pretrained carries (DDP: a plain load follows).
                load = importlib.import_module("unsloth.models.fsdp2_load")
                res["loaded"] = load.sharded_load_env(from_pretrained)(snapshot, load_in_4bit = False)
                choice = auto.resolved_choice()
            else:
                choice = auto.resolve_for_load(
                    from_pretrained, (snapshot,), {"load_in_4bit": False}
                )
            again = auto.resolve_for_load(from_pretrained, (snapshot,), {})  # once per process
            res[case] = dict(
                choice = choice,
                again = again,
                fsdp2_env = os.environ.get("UNSLOTH_FSDP2"),
                marker = os.environ.get("UNSLOTH_PARALLEL_AUTO_RESOLVED"),
                reason = auto._STATE["reason"],
            )
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


def _spawn(fn, *args):
    import torch.multiprocessing as mp

    from test_fsdp2_native_trainer_ckpt_gloo import _free_port

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        fn, args = (2, _free_port(), out, *args), nprocs = 2, join = True, start_method = "spawn"
    )
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for r in results.values():
        assert "crash" not in r, r["crash"]
    return results


def test_ranks_agree_fits_ddp_tight_fsdp2_split_fsdp2():
    snapshot = _snapshot()
    results = _spawn(_decision_worker, snapshot)
    for rank, r in results.items():
        print("AUTO_DECISION", rank, r)
        assert r["loaded"] == snapshot, r  # the decorated load ran after the decision
        assert r["fits"]["choice"] == r["fits"]["again"] == "ddp", r
        assert r["fits"]["fsdp2_env"] == "0" and r["fits"]["marker"] == "ddp", r
        assert r["tight"]["choice"] == "fsdp2" and r["tight"]["fsdp2_env"] == "1", r
        assert r["split"]["choice"] == r["split"]["again"] == "fsdp2", r
        assert r["split"]["fsdp2_env"] == "1" and "disagree" in r["split"]["reason"], r
        assert "rank 1 needs" in r["split"]["reason"], r


def _rows():
    g = torch.Generator().manual_seed(0)
    rows = []
    for _ in range(8):
        ids = torch.randint(3, 100, (16,), generator = g)
        rows.append(
            {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
        )
    return rows


def _lora_model():
    import transformers
    from peft import LoraConfig, get_peft_model

    config = transformers.AutoConfig.from_pretrained(TINY)
    config.pad_token_id = None
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    model = get_peft_model(
        model, LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"])
    )
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.02, generator = g)
    return model


def _e2e_worker(rank, world, port, out, tmp, budget_gb, expected):
    _init(rank, world, port)
    import torch.distributed as dist
    try:
        native = _load_native()
        from test_fsdp2_native_trainer_args_gloo import _install_wrapped_trainer_init
        from test_fsdp2_native_trainer_ckpt_gloo import _stub_unsloth_save

        _stub_unsloth_save()
        _install_wrapped_trainer_init()  # the real _unsloth_trainer_init source
        _, auto = _modules()
        from torch.distributed.tensor import DTensor
        from torch.nn.parallel import DistributedDataParallel
        from transformers import Trainer, TrainingArguments

        res = {}
        for mode in ("auto", "explicit"):
            auto._STATE.update(choice = None, reason = None)
            for key in ("UNSLOTH_FSDP2", "UNSLOTH_PARALLEL", "UNSLOTH_PARALLEL_AUTO_GB"):
                os.environ.pop(key, None)
            if mode == "auto":
                os.environ.update(UNSLOTH_PARALLEL = "auto", UNSLOTH_PARALLEL_AUTO_GB = budget_gb)
            else:
                os.environ["UNSLOTH_FSDP2"] = "1" if expected == "fsdp2" else "0"
            args = TrainingArguments(
                output_dir = os.path.join(tmp, mode),
                report_to = [],
                use_cpu = True,
                per_device_train_batch_size = 2,
                max_steps = STEPS,
                learning_rate = 1e-2,
                lr_scheduler_type = "constant",
                optim = "adamw_torch",
                weight_decay = 0.0,
                max_grad_norm = 1.0,
                save_strategy = "no",
                logging_steps = 1,
                disable_tqdm = True,
                dataloader_drop_last = True,
                seed = 0,
            )
            trainer = Trainer(model = _lora_model(), args = args, train_dataset = _rows())
            trainer.train()
            params = {
                n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
                for n, p in trainer.model.named_parameters()
                if "lora_" in n
            }
            res[mode] = dict(
                fsdp2_env = os.environ.get("UNSLOTH_FSDP2"),
                sharded = bool(native.is_native_fsdp2(trainer.model)),
                ddp = isinstance(trainer.model_wrapped, DistributedDataParallel),
                losses = [h["loss"] for h in trainer.state.log_history if "loss" in h],
                params = params,
                choice = auto._STATE["choice"],
            )
        a, b = res["auto"], res["explicit"]
        out.put(
            (
                rank,
                dict(
                    auto = {k: v for k, v in a.items() if k != "params"},
                    explicit = {k: v for k, v in b.items() if k != "params"},
                    n_params = len(a["params"]),
                    max_diff = max(
                        (a["params"][n] - b["params"][n]).abs().max().item() for n in a["params"]
                    ),
                ),
            )
        )
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("budget_gb, expected", [("64", "ddp"), ("1e-6", "fsdp2")])
def test_auto_trains_like_the_explicit_layout_it_picked(tmp_path, budget_gb, expected):
    pytest.importorskip("peft")
    _snapshot()
    results = _spawn(_e2e_worker, str(tmp_path), budget_gb, expected)
    for rank, r in results.items():
        print("AUTO_E2E", rank, r)
        a, b = r["auto"], r["explicit"]
        assert a["choice"] == expected, r
        assert a["fsdp2_env"] == b["fsdp2_env"] == ("1" if expected == "fsdp2" else "0"), r
        assert a["sharded"] == b["sharded"] == (expected == "fsdp2"), r
        assert a["ddp"] == b["ddp"] == (expected == "ddp"), r
        assert len(a["losses"]) == STEPS and a["losses"] == b["losses"], r
        assert r["n_params"] > 0 and r["max_diff"] == 0.0, r
