# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DPO concatenated_forward scored from hidden states (TRL < 0.29) must match TRL's own logits path: loss, every
metric and every gradient. CPU only: the rl_replacements helpers are lifted via ast and the zoo kernel is loaded by
path, so neither package import (which wants a GPU) runs."""

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
trl = pytest.importorskip("trl")

REPO = Path(__file__).resolve().parents[1]
SOURCE_PATH = REPO / "unsloth" / "models" / "rl_replacements.py"
TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"

HELPERS = (
    "_unsloth_get_model_config",
    "_unsloth_text_configs",
    "_unsloth_get_final_logit_softcapping",
    "_unsloth_resolve_logit_scales",
    "_unsloth_grpo_returns_hidden_states",
    "_unsloth_grpo_hidden_states_signal",
    "_unsloth_gkd_dense_head",
    "_unsloth_dpo_chunked_head",
    "_unsloth_dpo_model_forward",
    "_unsloth_dpo_returns_hidden_states",
    "_unsloth_dpo_hidden_logps",
    "_unsloth_dpo_nll",
    "dpo_trainer_chunked_logprobs",
    "_unsloth_kto_hidden_logps",
    "kto_trainer_chunked_logprobs",
)
CONSTANTS = (
    "_DPO_VISION_KEYS",
    "_DPO_CF_FORWARD",
    "_DPO_CF_LOGPS",
    "_DPO_CF_START",
    "_DPO_CF_CHOSEN",
    "_DPO_CF_CE",
    "_KTO_FORWARD",
    "_KTO_LOGPS",
)


def _zoo_kernel():
    zoo = os.environ.get("UNSLOTH_ZOO_PATH")
    candidates = [Path(zoo)] if zoo else []
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is not None and spec.submodule_search_locations:
        candidates += [Path(p).parent for p in spec.submodule_search_locations]
    for root in candidates:
        path = root / "unsloth_zoo" / "fused_losses" / "chunked_logprob.py"
        if path.exists():
            break
    else:
        pytest.skip("unsloth_zoo has no fused_losses/chunked_logprob.py")
    os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
    stub = types.ModuleType("unsloth_zoo.temporary_patches.common")
    stub.torch_compile_options = {}
    saved = {
        k: sys.modules.get(k)
        for k in ("unsloth_zoo", "unsloth_zoo.temporary_patches", stub.__name__)
    }
    sys.modules.setdefault("unsloth_zoo", types.ModuleType("unsloth_zoo"))
    sys.modules.setdefault(
        "unsloth_zoo.temporary_patches", types.ModuleType("unsloth_zoo.temporary_patches")
    )
    sys.modules[stub.__name__] = stub
    try:
        module_spec = importlib.util.spec_from_file_location("_zoo_chunked_logprob", path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return module


def _helpers():
    tree = ast.parse(SOURCE_PATH.read_text(encoding = "utf-8"))
    nodes = [
        n
        for n in tree.body
        if (isinstance(n, ast.FunctionDef) and n.name in HELPERS)
        or (
            isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in CONSTANTS for t in n.targets)
        )
    ]
    ns = {
        "os": os,
        "re": re,
        "torch": torch,
        "inspect": inspect,
        "textwrap": textwrap,
        "detect_logit_transforms": None,
    }
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(SOURCE_PATH), "exec"), ns)
    missing = [n for n in HELPERS + CONSTANTS if n not in ns]
    assert not missing, missing
    kernel = _zoo_kernel()
    ns["_unsloth_masked_logprobs"] = kernel.unsloth_masked_logprobs
    ns["_unsloth_chunked_logprobs"] = kernel.unsloth_chunked_logprobs
    return ns


_STOCK = {}


def _stock_module(mod):
    """A fresh copy of TRL's module: importing unsloth (conftest does, for its import fixes) replaces the trainer
    classes in TRL's modules with Unsloth's generated ones, so re-execute TRL's own file under a sibling name."""
    if mod.__name__ not in _STOCK:
        import importlib.util

        spec = importlib.util.spec_from_file_location(mod.__name__ + "_stock", mod.__file__)
        fresh = importlib.util.module_from_spec(spec)
        # Registered, so datasets' dill pickles its map functions by reference, not by value with every global.
        sys.modules[spec.name] = fresh
        spec.loader.exec_module(fresh)
        _STOCK[mod.__name__] = fresh
    return _STOCK[mod.__name__]


def _stock(mod, name):
    return getattr(_stock_module(mod), name)


@pytest.fixture(scope = "module")
def env():
    import trl.import_utils as iu

    iu._mergekit_available = False  # a broken mergekit dist-info here would fail the trainer import
    import trl.trainer.dpo_trainer as dpo_mod

    if not hasattr(_stock(dpo_mod, "DPOTrainer"), "concatenated_forward"):
        pytest.skip("TRL >= 0.29 has no concatenated_forward")
    ns = _helpers()
    source = inspect.getsource(_stock(dpo_mod, "DPOTrainer").concatenated_forward)
    # Class-indented, as unsloth/models/rl.py hands it to RL_FUNCTIONS.
    patched = ns["dpo_trainer_chunked_logprobs"]("concatenated_forward", source)
    assert patched != source, "rewrite did not apply to the installed TRL"
    g = dict(vars(_stock_module(dpo_mod)))
    g.update({k: v for k, v in ns.items() if not k.startswith("__")})
    exec(compile(textwrap.dedent(patched), "<patched concatenated_forward>", "exec"), g)
    return dpo_mod, g["concatenated_forward"]


def _honour_hidden_states(model):
    """What Unsloth's forward does under UNSLOTH_RETURN_HIDDEN_STATES=1: the final hidden states as ``.logits``."""
    original = model.forward

    def forward(*args, **kwargs):
        if os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") != "1":
            return original(*args, **kwargs)
        kwargs["output_hidden_states"] = True
        out = original(*args, **kwargs)
        out.logits = out.hidden_states[-1]
        return out

    model.forward = forward
    return model


def _dataset():
    from datasets import Dataset

    g = torch.Generator().manual_seed(0)
    words = "the of and to in is that for it as with was on be by this are from at or an".split()

    def text(n):
        return " ".join(words[i] for i in torch.randint(len(words), (n,), generator = g).tolist())

    return Dataset.from_list(
        [
            {"prompt": text(5 + i), "chosen": text(9 + 2 * i), "rejected": text(4 + 3 * i)}
            for i in range(4)
        ]
    )


def _run(
    dpo_mod,
    forward,
    cfg_kwargs,
    chunked,
    ref = "model",
    honour = True,
    peft_ref = False,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype = torch.float32)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}  # transformers 5 dropped it; stock TRL < 0.26 still writes to it
    tok = AutoTokenizer.from_pretrained(TINY)
    ref_model = copy.deepcopy(model) if ref == "model" else None
    with torch.no_grad():  # a ref that differs from the policy, so the margins are not all zero
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    if honour:
        _honour_hidden_states(model)
        if ref_model is not None:
            _honour_hidden_states(ref_model)
    cls = _stock(dpo_mod, "DPOTrainer")
    if chunked:
        cls = type(
            "PatchedDPOTrainer", (_stock(dpo_mod, "DPOTrainer"),), {"concatenated_forward": forward}
        )
    cfg = dpo_mod.DPOConfig(
        output_dir = str(REPO / ".pytest_dpo_chunked"),
        per_device_train_batch_size = 4,
        report_to = "none",
        use_cpu = True,
        max_length = 64,
        dataset_num_proc = None,
        **cfg_kwargs,
    )
    # In process: an imported unsloth patches the config to auto-size workers, and a worker pool has to pickle
    # the map function, whose globals here hold torch's config module.
    cfg.dataset_num_proc = None
    trainer = cls(
        model = model, ref_model = ref_model, args = cfg, train_dataset = _dataset(), processing_class = tok
    )
    batch = next(iter(trainer.get_train_dataloader()))
    loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch, train_eval = "train")
    loss.backward()
    grads = {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}
    return trainer, loss.detach(), metrics, grads


CASES = {
    "sigmoid": {"loss_type": "sigmoid"},
    "ipo": {"loss_type": "ipo"},
    "hinge_rpo": {"loss_type": "hinge", "rpo_alpha": 0.5},
    "sigmoid_sft": {"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, 0.7]},
    "robust_ld": {"loss_type": "robust", "ld_alpha": 0.5},
    "precompute": {"loss_type": "sigmoid", "precompute_ref_log_probs": True},
}


@pytest.mark.parametrize("case", list(CASES))
def test_matches_trl(env, case):
    dpo_mod, forward = env
    kwargs = dict(CASES[case])
    if "ld_alpha" in kwargs and "ld_alpha" not in dpo_mod.DPOConfig.__dataclass_fields__:
        pytest.skip("no ld_alpha in this TRL")
    if (
        isinstance(kwargs.get("loss_type"), list)
        and "loss_weights" not in dpo_mod.DPOConfig.__dataclass_fields__
    ):
        pytest.skip("no multi-loss in this TRL")
    _, base_loss, base_metrics, base_grads = _run(dpo_mod, forward, kwargs, chunked = False)
    trainer, loss, metrics, grads = _run(dpo_mod, forward, kwargs, chunked = True)
    assert trainer._unsloth_dpo_chunked_calls >= 1
    assert torch.allclose(loss, base_loss, atol = 1e-5, rtol = 1e-5), (case, loss, base_loss)
    assert metrics.keys() == base_metrics.keys()
    for k in base_metrics:
        assert abs(metrics[k] - base_metrics[k]) <= 1e-5 + 1e-5 * abs(base_metrics[k]), (
            case,
            k,
            metrics[k],
            base_metrics[k],
        )
    assert grads.keys() == base_grads.keys()
    worst = max(
        (grads[n] - base_grads[n]).abs().max().item() / (base_grads[n].abs().max().item() + 1e-12)
        for n in grads
    )
    assert worst < 1e-4, (case, worst)
    print(
        f"{case}: loss {loss.item():.6f} vs {base_loss.item():.6f}, worst rel grad diff {worst:.2e}, calls {trainer._unsloth_dpo_chunked_calls}"
    )


def test_forward_without_flag_keeps_trl_math(env):
    dpo_mod, forward = env
    _, base_loss, _, _ = _run(
        dpo_mod, forward, {"loss_type": "sigmoid"}, chunked = False, honour = False
    )
    trainer, loss, _, _ = _run(
        dpo_mod, forward, {"loss_type": "sigmoid"}, chunked = True, honour = False
    )
    assert getattr(trainer, "_unsloth_dpo_chunked_calls", 0) == 0
    assert (
        trainer._unsloth_dpo_chunked_fallbacks.get("forward returned logits, not hidden states", 0)
        >= 1
    )
    assert torch.allclose(loss, base_loss, atol = 1e-6)
    assert "UNSLOTH_RETURN_HIDDEN_STATES" not in os.environ


def test_use_weighting_falls_back(env):
    dpo_mod, forward = env
    _, base_loss, _, _ = _run(dpo_mod, forward, {"use_weighting": True}, chunked = False)
    trainer, loss, _, _ = _run(dpo_mod, forward, {"use_weighting": True}, chunked = True)
    assert getattr(trainer, "_unsloth_dpo_chunked_calls", 0) == 0
    assert "use_weighting needs full-vocabulary log-probs" in trainer._unsloth_dpo_chunked_fallbacks
    assert torch.allclose(loss, base_loss, atol = 1e-6)


TRL_TAGS = [
    "v0.18.2",
    "v0.19.1",
    "v0.20.0",
    "v0.21.0",
    "v0.22.2",
    "v0.23.1",
    "v0.24.0",
    "v0.25.1",
    "v0.26.2",
    "v0.27.2",
    "v0.28.0",
]


@pytest.mark.parametrize("tag", TRL_TAGS)
def test_rewrite_applies_across_trl(tag):
    repo = os.environ.get("TRL_HISTORY")
    if not repo or not Path(repo, ".git").exists():
        pytest.skip("set TRL_HISTORY to a huggingface/trl clone")
    text = subprocess.run(
        ["git", "-C", repo, "show", f"{tag}:trl/trainer/dpo_trainer.py"],
        capture_output = True,
        text = True,
        check = True,
    ).stdout
    cls = next(
        n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "DPOTrainer"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "concatenated_forward"
    )
    source = (
        "\n".join(text.splitlines()[fn.lineno - 1 - len(fn.decorator_list) : fn.end_lineno]) + "\n"
    )
    ns = _helpers()
    patched = ns["dpo_trainer_chunked_logprobs"]("concatenated_forward", source)
    assert patched != source
    compile(textwrap.dedent(patched), tag, "exec")
    assert patched.count("_unsloth_dpo_hidden") >= 5
    assert "_unsloth_dpo_nll(_unsloth_dpo_chosen_logps, chosen_labels)" in patched
    assert ns["dpo_trainer_chunked_logprobs"]("concatenated_forward", patched) == patched


def _trl_source(tag, paths, cls_name, fn_name):
    repo = os.environ.get("TRL_HISTORY")
    if not repo or not Path(repo, ".git").exists():
        pytest.skip("set TRL_HISTORY to a huggingface/trl clone")
    for path in paths:
        r = subprocess.run(
            ["git", "-C", repo, "show", f"{tag}:{path}"], capture_output = True, text = True
        )
        if r.returncode == 0 and f"class {cls_name}" in r.stdout:
            text = r.stdout
            break
    else:
        return None
    cls = next(
        n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == cls_name
    )
    fn = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == fn_name), None)
    if fn is None:
        return None
    return (
        "\n".join(text.splitlines()[fn.lineno - 1 - len(fn.decorator_list) : fn.end_lineno]) + "\n"
    )


KTO_PATHS = ("trl/experimental/kto/kto_trainer.py", "trl/trainer/kto_trainer.py")


@pytest.mark.parametrize("tag", TRL_TAGS + ["v0.29.1", "v1.0.0", "v1.5.0"])
def test_kto_rewrite_applies_across_trl(tag):
    source = _trl_source(tag, KTO_PATHS, "KTOTrainer", "forward")
    ns = _helpers()
    patched = ns["kto_trainer_chunked_logprobs"]("forward", source)
    assert patched != source
    compile(textwrap.dedent(patched), tag, "exec")
    assert "_unsloth_kto_hidden_logps(" in patched and "self.get_batch_logps(" in patched
    assert ns["kto_trainer_chunked_logprobs"]("forward", patched) == patched


@pytest.mark.parametrize("tag", ["v1.10.0", "v1.14.2", "v1.15.0"])
def test_kto_newer_trl_untouched(tag):
    source = _trl_source(tag, KTO_PATHS, "KTOTrainer", "forward")
    if source is None:
        return
    assert _helpers()["kto_trainer_chunked_logprobs"]("forward", source) == source


def _kto_run(
    cfg_kwargs,
    chunked,
    honour = True,
):
    import trl.import_utils as iu

    iu._mergekit_available = False
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        import trl.experimental.kto.kto_trainer as kto_mod
    except ImportError:
        import trl.trainer.kto_trainer as kto_mod
    ns = _helpers()
    stock_forward = getattr(_stock(kto_mod, "KTOTrainer"), "forward", None)
    if stock_forward is None:
        pytest.skip("installed KTOTrainer has no forward (TRL >= 1.10)")
    source = inspect.getsource(stock_forward)
    patched = ns["kto_trainer_chunked_logprobs"]("forward", source)
    if patched == source:
        pytest.skip(
            "installed KTOTrainer.forward has no rewritable get_batch_logps call (TRL >= 1.10)"
        )
    g = dict(vars(_stock_module(kto_mod)))
    g.update({k: v for k, v in ns.items() if not k.startswith("__")})
    exec(compile(textwrap.dedent(patched), "<patched kto forward>", "exec"), g)

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(TINY, torch_dtype = torch.float32)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}  # transformers 5 dropped it; stock TRL < 0.26 still writes to it
    tok = AutoTokenizer.from_pretrained(TINY)
    ref_model = copy.deepcopy(model)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn_like(p))
    if honour:
        _honour_hidden_states(model)
        _honour_hidden_states(ref_model)
    cls = _stock(kto_mod, "KTOTrainer")
    if chunked:
        cls = type("PatchedKTOTrainer", (_stock(kto_mod, "KTOTrainer"),), {"forward": g["forward"]})
    rows = _dataset()
    ds = Dataset.from_list(
        [
            {
                "prompt": r["prompt"],
                "completion": r["chosen" if i % 2 else "rejected"],
                "label": bool(i % 2),
            }
            for i, r in enumerate(rows)
        ]
    )
    cfg = kto_mod.KTOConfig(
        output_dir = str(REPO / ".pytest_kto_chunked"),
        per_device_train_batch_size = 4,
        report_to = "none",
        use_cpu = True,
        max_length = 64,
        max_prompt_length = 32,
        dataset_num_proc = None,
        **cfg_kwargs,
    )
    # In process: an imported unsloth patches the config to auto-size workers, and a worker pool has to pickle
    # the map function, whose globals here hold torch's config module.
    cfg.dataset_num_proc = None
    trainer = cls(
        model = model, ref_model = ref_model, args = cfg, train_dataset = ds, processing_class = tok
    )
    batch = next(iter(trainer.get_train_dataloader()))
    loss, metrics = trainer.get_batch_loss_metrics(trainer.model, batch)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}
    return trainer, loss.detach(), metrics, grads


@pytest.mark.parametrize("cfg", [{}, {"loss_type": "apo_zero_unpaired"}])
def test_kto_matches_trl(cfg):
    _, base_loss, base_metrics, base_grads = _kto_run(cfg, chunked = False)
    trainer, loss, metrics, grads = _kto_run(cfg, chunked = True)
    assert trainer._unsloth_dpo_chunked_calls >= 1
    assert torch.allclose(loss, base_loss, atol = 1e-5, rtol = 1e-5), (loss, base_loss)
    assert metrics.keys() == base_metrics.keys()
    for k in base_metrics:
        assert abs(metrics[k] - base_metrics[k]) <= 1e-4 + 1e-5 * abs(base_metrics[k]), (
            k,
            metrics[k],
            base_metrics[k],
        )
    worst = max(
        (grads[n] - base_grads[n]).abs().max().item() / (base_grads[n].abs().max().item() + 1e-12)
        for n in grads
    )
    assert worst < 1e-4, worst
    print(
        f"kto {cfg}: loss {loss.item():.6f} vs {base_loss.item():.6f}, worst rel grad diff {worst:.2e}"
    )
