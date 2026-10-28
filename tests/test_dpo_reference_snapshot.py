# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""DPO default reference: an in-memory policy is snapshotted, not reloaded from its checkpoint path."""

import ast
import pathlib
import re
import sys
import textwrap
import types

import pytest


def _hooks():
    path = pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "models" / "rl_replacements.py"
    tree = ast.parse(path.read_text(encoding = "utf-8"))
    wanted = {"_insert_after_docstring", "dpo_trainer_reference_snapshot"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    namespace = {"re": re, "textwrap": textwrap}
    exec(compile(ast.Module(body = nodes, type_ignores = []), str(path), "exec"), namespace)
    return namespace["dpo_trainer_reference_snapshot"]


hook = _hooks()

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
'''


class _Model:
    config = "ckpt"


class _FSDPModule:
    pass


class _ShardedModel(_Model, _FSDPModule):
    pass


def _run(
    source,
    model,
    monkeypatch,
    zero3 = False,
):
    calls = []
    import torch.distributed.fsdp as fsdp
    import transformers.integrations.deepspeed as ds

    monkeypatch.setattr(fsdp, "FSDPModule", _FSDPModule)
    monkeypatch.setattr(ds, "is_deepspeed_zero3_enabled", lambda: zero3)
    utils = types.ModuleType("trl.experimental.utils")
    utils.create_reference_model = lambda m: calls.append(("snapshot", m)) or "snapshot"
    monkeypatch.setitem(sys.modules, "trl.experimental.utils", utils)

    def create_model_from_path(path, **kwargs):
        calls.append(("load", path))
        return _Model() if path != "ckpt" else "reloaded"

    namespace = {
        "create_model_from_path": create_model_from_path,
        "is_peft_model": lambda m: False,
        "get_config_model_id": lambda config: config,
    }
    exec(textwrap.dedent(source), namespace)
    trainer = types.SimpleNamespace()
    namespace["__init__"](trainer, model)
    return trainer.ref_model, calls


def test_instance_reference_is_snapshot(monkeypatch):
    model = _Model()
    ref, calls = _run(hook("__init__", TRL_INIT), model, monkeypatch)
    assert ref == "snapshot" and calls == [("snapshot", model)]
    # Base TRL reloads the checkpoint, dropping in-memory training.
    ref, _ = _run(TRL_INIT, model, monkeypatch)
    assert ref == "reloaded"


@pytest.mark.parametrize(
    "model, zero3", [("hub/id", False), (_Model(), True), (_ShardedModel(), False)]
)
def test_path_and_zero3_keep_loader(monkeypatch, model, zero3):
    ref, calls = _run(hook("__init__", TRL_INIT), model, monkeypatch, zero3 = zero3)
    assert ref == "reloaded" and calls[-1] == ("load", "ckpt")


def test_rewrite_scope():
    patched = hook("__init__", TRL_INIT)
    assert patched != TRL_INIT and hook("__init__", patched) == patched
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
    assert "_unsloth_dpo_model_is_instance and not is_deepspeed_zero3_enabled()" in patched
    compile(textwrap.dedent(patched), "<patched>", "exec")
