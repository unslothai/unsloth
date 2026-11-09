# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DPO default reference: an in-memory policy is snapshotted, not reloaded from its checkpoint path."""

import ast
import pathlib
import re
import textwrap
import types

import pytest
import torch


def _hooks():
    path = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    wanted = {
        "_insert_after_docstring",
        "_unsloth_dpo_reference_loadable",
        "_unsloth_dpo_copy_policy_into_reference",
        "dpo_trainer_reference_snapshot",
    }
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {"re": re, "textwrap": textwrap, "torch": torch}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(path), "exec"), namespace)
    return namespace


_NS = _hooks()
hook = _NS["dpo_trainer_reference_snapshot"]

# Shape of TRL >= 0.29 DPOTrainer.__init__ around the default reference.
TRL_INIT = '''
    def __init__(self, model, ref_model = None, args = None):
        """Docstring."""
        if isinstance(model, str):
            model = create_model_from_path(model)
        self.model = model
        if ref_model is None:
            if is_peft_model(self.model):
                self.ref_model = None
            else:
                ref_model_init_kwargs = dict(args or {})  # copy
                ref_model_init_kwargs.setdefault("trust_remote_code", False)
                ref_model_path = get_config_model_id(self.model.config)
                self.ref_model = create_model_from_path(ref_model_path, **ref_model_init_kwargs)
        else:
            self.ref_model = ref_model
        warnings.warn("later TRL warning")  # e.g. use_liger_kernel, via TRL's module-level import
'''


class _Policy(torch.nn.Module):
    config = "ckpt"

    def __init__(
        self,
        out = 2,
        tied = False,
        offload = False,
    ):
        super().__init__()
        torch.manual_seed(0)  # the checkpoint's weights
        self.lin = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, out)
        if tied:
            self.head.weight = self.lin.weight
        if offload == "meta":  # one meta tensor: two would also differ in tying (both data_ptr 0)
            self.head.weight = torch.nn.Parameter(self.head.weight.to("meta"))
        elif offload == "quantized":  # e.g. TorchAO Int8Tensor from model_init_kwargs
            self.head.weight = torch.nn.Parameter(self.head.weight.detach().as_subclass(_Shard))


class _Shard(torch.Tensor):  # stands in for DTensor (FSDP2 / tensor parallel)
    pass


class _ShardedPolicy(_Policy):
    def __init__(self):
        super().__init__()
        self.lin.weight = torch.nn.Parameter(self.lin.weight.detach().as_subclass(_Shard))


def _run(
    source,
    model,
    monkeypatch,
    zero3 = False,
    out = 2,
    ref_model = None,
    tied = False,
    offload = False,
    dtype = torch.float32,
):
    import transformers.integrations.deepspeed as ds

    monkeypatch.setattr(ds, "is_deepspeed_zero3_enabled", lambda: zero3)
    namespace = {
        "_unsloth_dpo_reference_loadable": _NS["_unsloth_dpo_reference_loadable"],
        "_unsloth_dpo_copy_policy_into_reference": _NS["_unsloth_dpo_copy_policy_into_reference"],
        "create_model_from_path": lambda path, **kwargs: _Policy(out, tied, offload).to(dtype),
        "is_peft_model": lambda m: False,
        "get_config_model_id": lambda config: config,
        "torch": torch,
        "warnings": __import__("warnings"),
    }
    exec(textwrap.dedent(source), namespace)
    trainer = types.SimpleNamespace()
    with pytest.warns(UserWarning, match = "later TRL warning"):
        namespace["__init__"](trainer, model, ref_model)
    return trainer.ref_model


def _trained(cls = _Policy):
    policy = cls()
    with torch.no_grad():
        policy.lin.weight.add_(1.0)
    return policy


def test_instance_reference_is_in_memory_policy(monkeypatch):
    policy = _trained()
    # Unsloth patches instance forwards with closures over the policy's modules (block-compiled experts).
    policy.lin.forward = torch.compiler.disable(policy.lin.forward)
    ref = _run(hook("__init__", TRL_INIT), policy, monkeypatch)
    assert ref is not policy and torch.equal(ref.lin.weight, policy.lin.weight)
    x = torch.ones(1, 2)
    before = ref.lin(x)
    with torch.no_grad():
        policy.lin.weight.add_(1.0)
    assert torch.equal(ref.lin(x), before)
    ref = _run(TRL_INIT, _trained(), monkeypatch)
    assert torch.equal(ref.lin.weight, _Policy().lin.weight)


@pytest.mark.parametrize("model, zero3", [("hub/id", False), (None, True)])
def test_path_and_zero3_keep_checkpoint(monkeypatch, model, zero3):
    model = _trained() if model is None else model
    ref = _run(hook("__init__", TRL_INIT), model, monkeypatch, zero3 = zero3)
    assert torch.equal(ref.lin.weight, _Policy().lin.weight)


def test_reference_keeps_policy_dtypes(monkeypatch):
    # BF16 full finetune keeps FP32 norms; the reference loads in the policy's (BF16) dtype.
    policy = _trained().to(torch.bfloat16)
    policy.lin.float()
    with torch.no_grad():
        policy.lin.weight.add_(1e-3)  # below BF16 resolution at ~1
    ref = _run(hook("__init__", TRL_INIT), policy, monkeypatch, dtype = torch.bfloat16)
    for name, value in policy.state_dict().items():
        assert ref.state_dict()[name].dtype == value.dtype and torch.equal(
            ref.state_dict()[name], value
        ), name


def test_explicit_reference_kept(monkeypatch):
    explicit = _Policy()
    assert _run(hook("__init__", TRL_INIT), _trained(), monkeypatch, ref_model = explicit) is explicit


@pytest.mark.parametrize(
    "policy, out, tied, offload",
    [
        (_Policy, 3, False, False),
        (_Policy, 2, True, False),
        (_ShardedPolicy, 2, False, False),
        (_Policy, 2, False, "meta"),
        (_Policy, 2, False, "quantized"),
    ],
    ids = [
        "resized",
        "untied_policy_tied_checkpoint",
        "sharded",
        "offloaded_reference",
        "quantized_reference",
    ],
)
def test_mismatched_reference_keeps_whole_checkpoint(monkeypatch, policy, out, tied, offload):
    with pytest.warns(UserWarning, match = "do not match its checkpoint"):
        ref = _run(
            hook("__init__", TRL_INIT),
            _trained(policy),
            monkeypatch,
            out = out,
            tied = tied,
            offload = offload,
        )
    checkpoint = _Policy(out, tied, offload)
    for name, value in checkpoint.state_dict().items():
        if not value.is_meta:
            assert torch.equal(ref.state_dict()[name], value), name


def test_rewrite_scope():
    patched = hook("__init__", TRL_INIT)
    assert patched != TRL_INIT and hook("__init__", patched) == patched
    assert ".state_dict()" not in patched
    assert hook("compute_loss", TRL_INIT) == TRL_INIT
    legacy = TRL_INIT.replace(
        "create_model_from_path(ref_model_path, **ref_model_init_kwargs)",
        "create_reference_model(model)",
    )
    assert hook("__init__", legacy) == legacy


def test_installed_trl_rewrites_and_compiles():
    trl = pytest.importorskip("trl")
    import trl.trainer.dpo_trainer as installed

    # From the file: conftest imports unsloth, which swaps in the compiled UnslothDPOTrainer.
    text = pathlib.Path(installed.__file__).read_text(encoding = "utf-8")
    cls = next(
        n for n in ast.parse(text).body if isinstance(n, ast.ClassDef) and n.name == "DPOTrainer"
    )
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    source = ast.get_source_segment(text, init, padded = True)
    if "create_model_from_path(ref_model_path" not in source:
        pytest.skip(f"TRL {trl.__version__} already copies the policy")
    patched = hook("__init__", source)
    assert "_unsloth_dpo_copy_policy_into_reference(self.ref_model" in patched
    # rl.py's _strip_vllm_weight_loads deletes `.state_dict()` from every generated trainer.
    assert ".state_dict()" not in patched
    compile(textwrap.dedent(patched), "<patched>", "exec")
