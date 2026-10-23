# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_FSDP2=1 for a model no FastLanguageModel / FastModel load built.

Native FSDP2 shards inside Unsloth's Trainer.__init__ wrapper, which only those loads (and
DecisionTrainer) installed. A FastSentenceTransformer encoder (BERT, ModernBERT) never loads through
them, so under `UNSLOTH_FSDP2=1 torchrun` its SentenceTransformerTrainer trained with plain DDP:
nothing sharded and no message. `import unsloth` now installs the wrapper under the flag
(fsdp2_launch.install_native_fsdp2_trainer_init).

* the installer wraps Trainer under the flag, and only there;
* the GPU import path calls it outside any try / except;
* on a gloo CPU world of 2, a SentenceTransformer-shaped model (a plain encoder plus mean pooling and
  a Trainer subclass with its own compute_loss, as sentence-transformers builds it) is sharded by the
  installed wrapper and trains like one process over the same 4 rows."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import ast  # noqa: E402
import copy  # noqa: E402
import importlib.util  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402
from test_fsdp2_native_trainer_ckpt_gloo import _free_port, _stub_unsloth_save  # noqa: E402
from test_fsdp2_native_trainer_args_gloo import _install_wrapped_trainer_init  # noqa: E402

ENCODER = "hf-internal-testing/tiny-random-BertModel"
STEPS = 2


def _load_launch(calls):
    """fsdp2_launch.py with `.models._utils` stubbed (the real one needs the GPU package)."""
    pkg = sys.modules.get("unsloth") or types.ModuleType("unsloth")
    if not hasattr(pkg, "__path__"):
        pkg.__path__ = [os.path.join(REPO, "unsloth")]
    models = sys.modules.get("unsloth.models") or types.ModuleType("unsloth.models")
    if not hasattr(models, "__path__"):
        models.__path__ = [os.path.join(REPO, "unsloth", "models")]
    utils = types.ModuleType("unsloth.models._utils")

    def patch_gradient_accumulation_fix(trainer_cls):
        calls.append(trainer_cls)
        _install_wrapped_trainer_init()

    utils.patch_gradient_accumulation_fix = patch_gradient_accumulation_fix
    sys.modules["unsloth"] = pkg
    sys.modules["unsloth.models"] = models
    sys.modules["unsloth.models._utils"] = utils
    spec = importlib.util.spec_from_file_location(
        "unsloth.fsdp2_launch", os.path.join(REPO, "unsloth", "fsdp2_launch.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["unsloth.fsdp2_launch"] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "env,applies",
    [
        ({"UNSLOTH_FSDP2": "1", "WORLD_SIZE": "2"}, True),
        ({"UNSLOTH_FSDP2": "1", "WORLD_SIZE": "1"}, False),
        ({"WORLD_SIZE": "2"}, False),
        ({"UNSLOTH_FSDP2": "1", "WORLD_SIZE": "2", "UNSLOTH_FSDP2_BACKEND": "accelerate"}, False),
    ],
)
def test_installer_wraps_trainer_only_under_native_fsdp2(monkeypatch, env, applies):
    from transformers import Trainer

    for key in ("UNSLOTH_FSDP2", "WORLD_SIZE", "UNSLOTH_FSDP2_BACKEND", "ACCELERATE_USE_FSDP"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    for name in ("unsloth", "unsloth.models", "unsloth.models._utils", "unsloth.fsdp2_launch"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    monkeypatch.setattr(Trainer, "__init__", Trainer.__init__)
    calls = []
    launch = _load_launch(calls)
    assert launch.install_native_fsdp2_trainer_init() is applies
    assert calls == ([Trainer] if applies else [])


def test_gpu_import_path_installs_it():
    tree = ast.parse(open(os.path.join(REPO, "unsloth", "__init__.py"), encoding = "utf-8").read())
    found = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_install_native_trainer_init"
    ]
    assert found, "unsloth/__init__.py no longer installs the native FSDP2 Trainer wrapper"
    # Not swallowed: a wrapper that failed to install would silently fall back to DDP.
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            inside = {id(n) for stmt in node.body for n in ast.walk(stmt)}
            assert not any(id(call) in inside for call in found)


class _MeanPooled(torch.nn.Module):
    """A SentenceTransformer's shape: the HF encoder plus mean pooling, no PreTrainedModel root."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1)


def _pair_loss(model, batch):
    a = model(batch["a_ids"], batch["a_mask"])
    b = model(batch["b_ids"], batch["b_mask"])
    cos = torch.nn.functional.cosine_similarity(a, b)
    return torch.nn.functional.mse_loss(cos, batch["score"])


def _rows():
    g = torch.Generator().manual_seed(0)
    return [
        {
            "a_ids": torch.randint(5, 90, (12,), generator = g),
            "a_mask": torch.ones(12, dtype = torch.long),
            "b_ids": torch.randint(5, 90, (12,), generator = g),
            "b_mask": torch.ones(12, dtype = torch.long),
            "score": torch.rand((), generator = g),
        }
        for _ in range(4)
    ]


def _model():
    import transformers

    config = transformers.AutoConfig.from_pretrained(ENCODER)
    config.hidden_dropout_prob = config.attention_probs_dropout_prob = 0.0  # deterministic steps
    torch.manual_seed(0)
    return _MeanPooled(transformers.AutoModel.from_config(config, torch_dtype = torch.float32))


def _reference(model, rows):
    """One process: the same AdamW steps on all 4 rows (constant LR, clip 1.0)."""
    model = copy.deepcopy(model)
    batch = {k: torch.stack([r[k] for r in rows]) for k in rows[0]}
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-3, weight_decay = 0.0)
    for _ in range(STEPS):
        opt.zero_grad()
        _pair_loss(model, batch).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _worker(rank, world, port, out, tmp):
    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
        UNSLOTH_FSDP2 = "1",
    )
    import datetime

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )
    try:
        native = _load_native()
        _stub_unsloth_save()
        from transformers import Trainer, TrainingArguments

        res = {"wrapped_before": Trainer.__init__.__name__ == "_unsloth_trainer_init"}
        launch = _load_launch([])
        res["installed"] = launch.install_native_fsdp2_trainer_init()

        class PairTrainer(Trainer):
            # sentence-transformers' own compute_loss: the model returns embeddings, not a loss.
            def compute_loss(
                self,
                model,
                inputs,
                return_outputs = False,
                num_items_in_batch = None,
            ):
                return _pair_loss(model, inputs)

        def collate(rows):
            return {k: torch.stack([r[k] for r in rows]) for k in rows[0]}

        rows = _rows()
        model = _model()
        want = _reference(model, rows)
        args = TrainingArguments(
            output_dir = os.path.join(tmp, "out"),
            report_to = [],
            use_cpu = True,
            per_device_train_batch_size = 2,
            max_steps = STEPS,
            learning_rate = 1e-3,
            lr_scheduler_type = "constant",
            optim = "adamw_torch",
            weight_decay = 0.0,
            max_grad_norm = 1.0,
            save_strategy = "no",
            logging_steps = 1,
            disable_tqdm = True,
            remove_unused_columns = False,
            dataloader_drop_last = True,
            seed = 0,
        )
        trainer = PairTrainer(model = model, args = args, train_dataset = rows, data_collator = collate)
        res["sharded"] = native.is_native_fsdp2(trainer.model)
        res["units"] = sum(
            isinstance(m, torch.distributed.fsdp.FSDPModule) for m in trainer.model.modules()
        )
        trainer.train()
        from torch.distributed.tensor import DTensor

        got = {
            n: (p.full_tensor() if isinstance(p, DTensor) else p).detach()
            for n, p in trainer.model.named_parameters()
        }
        res["n_params"] = len(got)
        res["missing"] = sorted(set(want) - set(got))[:5]
        res["max_diff"] = max((got[n] - want[n]).abs().max().item() for n in want if n in got)
        res["moved"] = max((want[n] - p).abs().max().item() for n, p in _model().named_parameters())
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


def test_sentence_transformer_shaped_model_shards_and_matches_one_process(tmp_path):
    pytest.importorskip("transformers")
    import torch.multiprocessing as mp

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        _worker,
        args = (2, _free_port(), out, str(tmp_path)),
        nprocs = 2,
        join = True,
        start_method = "spawn",
    )
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for rank, r in results.items():
        print("ANY_LOADER_RESULT", rank, r)
        assert "crash" not in r, r["crash"]
        assert r["wrapped_before"] is False, r  # nothing else installed the wrapper here
        assert r["installed"] is True and r["sharded"] is True, r
        assert r["units"] >= 2, r  # encoder blocks + root
        assert not r["missing"], r
        assert (
            r["moved"] > 1e-4
        ), r  # the steps changed the weights, so the comparison means something
        assert r["max_diff"] < 2e-5, r  # fp32, reduction order only
