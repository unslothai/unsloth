# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Reentrant checkpointing drops the MoE router aux loss gradient. The RL config pins use_reentrant = True; with
the aux loss on, the trainer undoes a pin the user did not ask for, offline."""

from __future__ import annotations

import re
import textwrap
import types

from tests.version_compat._rl_anchors import RL_PY, _rl_py_constant


def _config_post():
    match = re.search(r"RLConfig_post = \(\n(.*?)\n    \)\n", RL_PY.read_text("utf-8"), re.S)
    assert match is not None, "RLConfig_post assignment not found"
    snippet = textwrap.dedent(eval("(\n" + match.group(1) + "\n)", {"__builtins__": {}}))
    ns = {}
    exec(
        compile("def _post(self):\n" + textwrap.indent(snippet, "    "), "<RLConfig_post>", "exec"),
        ns,
    )
    return ns["_post"]


def _trainer(
    gc_kwargs,
    aux = True,
    model_type = "qwen3_moe",
    gradient_checkpointing = True,
    zero_stage = None,
):
    args = types.SimpleNamespace(
        gradient_checkpointing = gradient_checkpointing, gradient_checkpointing_kwargs = gc_kwargs
    )
    if zero_stage is not None:
        args.deepspeed_plugin = types.SimpleNamespace(zero_stage = zero_stage)
    _config_post()(args)
    self = types.SimpleNamespace(
        aux_loss_enabled = aux,
        args = args,
        model = types.SimpleNamespace(config = types.SimpleNamespace(model_type = model_type)),
    )
    exec(_rl_py_constant("_ROUTER_AUX_LOSS_NON_REENTRANT_GC"), {"self": self})
    return args


def test_aux_loss_undoes_the_default_pin():
    assert _trainer(None).gradient_checkpointing_kwargs == {"use_reentrant": False}


def test_aux_loss_restores_an_explicit_false():
    assert _trainer({"use_reentrant": False}).gradient_checkpointing_kwargs == {
        "use_reentrant": False
    }


def test_other_checkpoint_kwargs_survive():
    args = _trainer({"determinism_check": "none"})
    assert args.gradient_checkpointing_kwargs == {
        "determinism_check": "none",
        "use_reentrant": False,
    }


def test_an_explicit_true_is_kept():
    assert _trainer({"use_reentrant": True}).gradient_checkpointing_kwargs == {
        "use_reentrant": True
    }


def test_without_the_aux_loss_the_pin_stays():
    assert _trainer(None, aux = False).gradient_checkpointing_kwargs == {"use_reentrant": True}


def test_gemma_keeps_reentrant():
    # vision.py keeps Gemma 3n / 4 on reentrant checkpointing (AOT backward strides).
    for model_type in ("gemma3n", "gemma4", "gemma4_text"):
        args = _trainer(None, model_type = model_type)
        assert args.gradient_checkpointing_kwargs == {"use_reentrant": True}, model_type


def _fake_deepspeed(monkeypatch, ds_version):
    import importlib.metadata as md

    real = md.version

    def fake(name):
        if name == "deepspeed":
            if ds_version is None:
                raise md.PackageNotFoundError(name)
            return ds_version
        return real(name)

    monkeypatch.setattr(md, "version", fake)


def test_zero3_on_old_deepspeed_keeps_trls_reentrant(monkeypatch):
    # TRL forces use_reentrant = True for PEFT + ZeRO-3 on DeepSpeed < 0.19.3 (deepspeedai/DeepSpeed#8148).
    _fake_deepspeed(monkeypatch, "0.19.2")
    assert _trainer(None, zero_stage = 3).gradient_checkpointing_kwargs == {"use_reentrant": True}


def test_zero3_with_unknown_deepspeed_keeps_reentrant(monkeypatch):
    _fake_deepspeed(monkeypatch, None)
    assert _trainer(None, zero_stage = 3).gradient_checkpointing_kwargs == {"use_reentrant": True}


def test_zero3_on_fixed_deepspeed_switches(monkeypatch):
    _fake_deepspeed(monkeypatch, "0.19.3")
    assert _trainer(None, zero_stage = 3).gradient_checkpointing_kwargs == {"use_reentrant": False}


def test_zero2_switches(monkeypatch):
    _fake_deepspeed(monkeypatch, "0.18.0")
    assert _trainer(None, zero_stage = 2).gradient_checkpointing_kwargs == {"use_reentrant": False}


def test_no_checkpointing_is_untouched():
    assert _trainer(None, gradient_checkpointing = False).gradient_checkpointing_kwargs is None


def test_rl_py_appends_it_after_trl_init():
    src = RL_PY.read_text("utf-8")
    assert "RLTrainer_post += _ROUTER_AUX_LOSS_NON_REENTRANT_GC" in src
    assert src.index("RLTrainer_post += _DENSE_ROUTER_AUX_LOSS_OFF") < src.index(
        "RLTrainer_post += _ROUTER_AUX_LOSS_NON_REENTRANT_GC"
    )
