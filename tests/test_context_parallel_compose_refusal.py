# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Context parallelism beside UNSLOTH_TP_SIZE (or a model split by hand) is refused before the trainer __init__,
without a process group (tests/test_parallel_cp_compose_gloo.py trains the layouts on gloo ranks)."""

import functools
import types

import pytest
import torch

import unsloth  # noqa: F401
from unsloth import context_parallel as cp

_ENV = ("UNSLOTH_TP_SIZE", "UNSLOTH_FSDP2", "UNSLOTH_CP_SIZE", "WORLD_SIZE")


@pytest.fixture
def clean_env(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    return monkeypatch


def _fake(monkeypatch, inits, kind):
    class Trainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            inits.append(model)
            self.model = model
            self.args = args

    if kind == "sft":
        import trl
        monkeypatch.setattr(trl, "SFTTrainer", Trainer)
        cp.patch_sft_trainer()
    else:
        cp.patch_rl_trainer(Trainer, kind)
    return Trainer


def _config(size, **kwargs):
    return types.SimpleNamespace(
        context_parallel_size = size, context_parallel_backend = "ulysses", **kwargs
    )


@pytest.mark.parametrize("kind", ["sft", "dpo_trainer", "grpo_trainer"])
@pytest.mark.parametrize("via", ["config", "kwarg", "positional"])
def test_cp_with_tp_env_is_refused_before_the_trainer_init(clean_env, kind, via):
    clean_env.setenv("UNSLOTH_TP_SIZE", "2")
    if kind != "sft" and via == "kwarg":
        pytest.skip("the RL trainers read context_parallel_size from their config only")
    inits = []
    Trainer = _fake(clean_env, inits, kind)
    model = torch.nn.Linear(2, 2)
    if via == "config":
        call = lambda: Trainer(model = model, args = _config(2))
    elif via == "positional":
        call = lambda: Trainer(model, _config(2))
    else:
        call = lambda: Trainer(model, _config(1), context_parallel_size = 2)
    with pytest.raises(NotImplementedError, match = "does not compose with") as info:
        call()
    assert "tensor parallel (UNSLOTH_TP_SIZE=2)" in str(info.value), str(info.value)
    assert inits == [], "refused after the trainer __init__ ran (the model may already be split)"


@pytest.mark.parametrize("split", ["is_tensor_parallel", "is_pipeline_parallel"])
def test_cp_on_a_model_split_by_hand_is_refused(clean_env, split):
    from unsloth.models import pp_native, tp_native

    module = tp_native if split == "is_tensor_parallel" else pp_native
    clean_env.setattr(module, split, lambda model: True)
    inits = []
    Trainer = _fake(clean_env, inits, "sft")
    with pytest.raises(NotImplementedError, match = "apply_tensor_parallel|apply_pipeline_parallel"):
        Trainer(model = torch.nn.Linear(2, 2), args = _config(2))
    assert inits == []


@pytest.mark.parametrize(
    "env, size",
    [
        ({}, 2),
        ({"UNSLOTH_FSDP2": "1"}, 2),
        ({"UNSLOTH_TP_SIZE": "1"}, 2),
        ({"UNSLOTH_TP_SIZE": "2"}, 1),
    ],
    ids = ["cp_alone", "cp_fsdp2", "tp1_cp2", "tp2_cp1"],
)
def test_cp_alone_with_fsdp2_and_tp_without_cp_are_not_refused(clean_env, monkeypatch, env, size):
    for key, value in env.items():
        clean_env.setenv(key, value)
    # Past the refusal, the CP __init__ needs a process group; stop at the original __init__.
    monkeypatch.setattr(cp, "_check_cp_runtime", lambda *a, **k: None)
    inits = []
    Trainer = _fake(clean_env, inits, "sft")
    try:
        Trainer(model = torch.nn.Linear(2, 2), args = _config(size))
    except NotImplementedError as e:
        assert "does not compose with" not in str(e), str(e)
    except Exception:
        pass
    assert len(inits) == 1


@pytest.mark.parametrize("raw", ["+2", " 2 ", "02", "4"])
def test_cp_with_any_tp_env_spelling_tp_env_accepts_is_refused(clean_env, raw):
    """tp_env reads UNSLOTH_TP_SIZE with int(): every value it turns into tp > 1 is refused beside CP."""
    from unsloth.models import tp_env

    clean_env.setenv("UNSLOTH_TP_SIZE", raw)
    assert tp_env._int_env("UNSLOTH_TP_SIZE") > 1
    inits = []
    Trainer = _fake(clean_env, inits, "sft")
    with pytest.raises(NotImplementedError, match = "does not compose with"):
        Trainer(model = torch.nn.Linear(2, 2), args = _config(2))
    assert inits == []


@pytest.mark.parametrize("raw", ["\u00b2", "2.0", "abc"])
def test_cp_with_a_tp_env_value_tp_env_rejects_is_left_to_tp_env(clean_env, monkeypatch, raw):
    """A value tp_env refuses is not parsed into a crash here: tp_env names it itself."""
    clean_env.setenv("UNSLOTH_TP_SIZE", raw)
    monkeypatch.setattr(cp, "_check_cp_runtime", lambda *a, **k: None)
    inits = []
    Trainer = _fake(clean_env, inits, "sft")
    try:
        Trainer(model = torch.nn.Linear(2, 2), args = _config(2))
    except Exception:
        pass
    assert len(inits) == 1


def test_env_cp_survives_plain_training_arguments(clean_env):
    # Failing before: the env cp size was set as an attribute of plain TrainingArguments, which Unsloth's trainer
    # wrapper rebuilds from to_dict() (dropping it): the trainer silently ran without context parallelism.
    from transformers import TrainingArguments

    clean_env.setenv("UNSLOTH_CP_SIZE", "2")
    args = TrainingArguments(output_dir = "unused", report_to = "none")
    kwargs = {}
    assert cp._inject_env_cp(args, kwargs, 1, "SFTTrainer") == 2
    assert kwargs.get("context_parallel_size") == 2
    assert "context_parallel_size" not in args.to_dict()


def test_env_cp_sets_trl_style_configs_in_place(clean_env):
    clean_env.setenv("UNSLOTH_CP_SIZE", "2")
    config = _config(1)
    kwargs = {}
    assert cp._inject_env_cp(config, kwargs, 1, "SFTTrainer") == 2
    assert config.context_parallel_size == 2 and "context_parallel_size" not in kwargs


def test_env_cp_with_positional_plain_training_arguments_becomes_a_keyword(clean_env, monkeypatch):
    # Failing before: SFTTrainer(model, TrainingArguments(...)) got context_parallel_size as a kwarg while the
    # config stayed positional; Unsloth's trainer wrapper folds cp kwargs only into a keyword `args`, so the
    # original __init__ raised "unexpected keyword argument".
    import trl
    from transformers import TrainingArguments

    clean_env.setenv("UNSLOTH_CP_SIZE", "2")
    monkeypatch.setattr(cp, "_check_cp_runtime", lambda *a, **k: None)
    monkeypatch.setattr(cp, "_resolve_auto_backend", lambda *a, **k: None)
    monkeypatch.setattr(cp, "_refuse_model_parallel_cp", lambda *a, **k: None)
    calls = []

    def signature(
        self,
        model = None,
        args = None,
        data_collator = None,
        **kwargs,
    ):
        pass

    # As Unsloth's trainer wrapper: (*args, **kwargs) behind functools.wraps of the real signature.
    @functools.wraps(signature)
    def init(self, *args, **kwargs):
        calls.append((args, kwargs))

    class Trainer:
        __init__ = init

    monkeypatch.setattr(trl, "SFTTrainer", Trainer)
    cp.patch_sft_trainer()
    model, training_args = (
        torch.nn.Linear(2, 2),
        TrainingArguments(output_dir = "unused", report_to = "none"),
    )
    try:
        trl.SFTTrainer(model, training_args)
    except Exception:
        pass  # past the original __init__ the CP setup needs a process group
    assert calls, "the original __init__ never ran"
    got_args, got_kwargs = calls[0]
    assert got_args == () and got_kwargs["model"] is model and got_kwargs["args"] is training_args
    assert got_kwargs.get("context_parallel_size") == 2


def test_a_backend_kwarg_overrides_the_config_in_the_pre_init_check(clean_env, monkeypatch):
    # Failing before: the capability check read the config's backend (ring) although the constructor kwarg
    # (ulysses) is what the trainer uses, refusing a valid ulysses + packing call.
    import unsloth.parallel_capabilities as caps

    seen = []
    monkeypatch.setattr(
        caps, "check_layout", lambda *a, cp_backend = None, **k: seen.append(cp_backend)
    )
    cp._refuse_model_parallel_cp(2, torch.nn.Linear(2, 2), None, _config(2), backend = "ulysses")
    assert seen == ["ulysses"]


def test_env_cp_dpo_with_a_positional_reference_model_gets_a_config(clean_env, monkeypatch):
    # Failing before: DPOTrainer(model, ref_model, ...) skipped the default config (two positionals), so the env
    # cp values reached the constructor as kwargs, which TRL rejects.
    clean_env.setenv("UNSLOTH_CP_SIZE", "2")
    monkeypatch.setattr(cp, "_check_cp_runtime", lambda *a, **k: None)
    monkeypatch.setattr(cp, "_resolve_auto_backend", lambda *a, **k: None)
    monkeypatch.setattr(cp, "_refuse_model_parallel_cp", lambda *a, **k: None)
    import dataclasses
    import sys

    @dataclasses.dataclass
    class DPOConfig:
        context_parallel_size: int = 1
        context_parallel_backend: str = "auto"

    module = types.ModuleType("fake_dpo_module")
    module.DPOConfig = DPOConfig
    monkeypatch.setitem(sys.modules, "fake_dpo_module", module)
    calls = []

    def signature(
        self,
        model = None,
        ref_model = None,
        args = None,
        train_dataset = None,
        **kwargs,
    ):
        pass

    @functools.wraps(signature)
    def init(self, *args, **kwargs):
        calls.append((args, kwargs))

    Trainer = type("DPOTrainer", (), {"__init__": init, "__module__": "fake_dpo_module"})
    cp.patch_rl_trainer(Trainer, "dpo_trainer")
    model, ref = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    try:
        Trainer(model, ref, train_dataset = [])
    except Exception:
        pass
    assert calls
    got_args, got_kwargs = calls[0]
    assert got_kwargs.get("ref_model") is ref and "context_parallel_size" not in got_kwargs
    assert (
        isinstance(got_kwargs.get("args"), DPOConfig)
        and got_kwargs["args"].context_parallel_size == 2
    )


def test_auto_backend_reads_a_model_id_config_with_the_loader_options(clean_env, monkeypatch):
    # Failing before: only trust_remote_code was forwarded, so a checkpoint under a subfolder was not found and
    # 'auto' committed to ulysses with unknown heads.
    import unsloth.trainer as trainer_mod

    seen = []
    monkeypatch.setattr(
        trainer_mod,
        "_resolve_string_model_config",
        lambda name, config: seen.append((name, config)) or None,
    )
    config = _config(2, model_init_kwargs = {"subfolder": "weights"})
    config.context_parallel_backend = "auto"
    try:
        cp._resolve_auto_backend(config, {}, 2, "org/model")
    except Exception:
        pass
    assert seen and seen[0][0] == "org/model" and seen[0][1] is config


def test_config_field_kwargs_override_the_config_in_the_pre_init_check(clean_env, monkeypatch):
    # Failing before: SFTTrainer(..., args = config, router_aux_loss_coef = 0.0) was checked against the config's
    # own value, refusing a layout the trainer would run (the wrapper moves the kwarg into the config).
    import trl

    seen = []
    monkeypatch.setattr(cp, "_check_cp_runtime", lambda *a, **k: None)
    monkeypatch.setattr(cp, "_resolve_auto_backend", lambda *a, **k: None)
    monkeypatch.setattr(
        cp,
        "_refuse_model_parallel_cp",
        lambda size, model, cls, config, **k: seen.append(config.router_aux_loss_coef),
    )

    class Trainer:
        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            pass

    monkeypatch.setattr(trl, "SFTTrainer", Trainer)
    cp.patch_sft_trainer()
    config = _config(2, router_aux_loss_coef = None)
    try:
        trl.SFTTrainer(model = torch.nn.Linear(2, 2), args = config, router_aux_loss_coef = 0.0)
    except Exception:
        pass
    assert seen == [0.0]
