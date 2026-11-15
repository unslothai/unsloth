# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""SFT packs text datasets by default when the user left `packing` unset (UnslothSFTConfig's None),
and keeps today's padding-free batches for an explicit packing / padding_free, a custom collator,
a stream, UNSLOTH_DEFAULT_PACKING=0, or a trainer that refuses packing (CPU, stub trainer)."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
datasets = pytest.importorskip("datasets")
import unsloth.trainer as trainer_module  # noqa: E402


class _Model(torch.nn.Module):
    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return input_ids


def _patched(monkeypatch, refuse_packing = False):
    calls = {"pack": 0, "padding_free": 0, "inits": []}

    class _StubSFTTrainer:
        def __init__(
            self,
            model = None,
            args = None,
            data_collator = None,
            train_dataset = None,
            **kwargs,
        ):
            calls["inits"].append(bool(args.packing))
            if refuse_packing and args.packing:
                raise ValueError("this trainer cannot pack")
            self.model, self.args, self.data_collator = model, args, data_collator

    monkeypatch.setattr(
        trainer_module,
        "enable_sample_packing",
        lambda m, t: calls.__setitem__("pack", calls["pack"] + 1),
    )
    monkeypatch.setattr(
        trainer_module,
        "enable_padding_free_metadata",
        lambda m, t: calls.__setitem__("padding_free", calls["padding_free"] + 1),
    )
    monkeypatch.setattr(trainer_module, "install_token_budget", lambda t: None)
    monkeypatch.delenv("UNSLOTH_DEFAULT_PACKING", raising = False)
    monkeypatch.delenv("UNSLOTH_RETURN_LOGITS", raising = False)
    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)
    return module.SFTTrainer, calls


def _config(**kw):
    base = dict(
        packing = None,
        padding_free = None,
        eval_packing = None,
        max_length = 512,
        remove_unused_columns = True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _text():
    return datasets.Dataset.from_dict({"text": ["a b c", "d e"]})


def test_unset_packing_packs_a_text_dataset(monkeypatch, capsys):
    SFT, calls = _patched(monkeypatch)
    config = _config()
    SFT(model = _Model(), args = config, train_dataset = _text())
    assert config.packing is True and config.padding_free is True
    assert config.eval_packing is False  # eval batches stay as before
    assert calls["pack"] == 1 and calls["padding_free"] == 0
    assert "UNSLOTH_DEFAULT_PACKING=0" in capsys.readouterr().out


@pytest.mark.parametrize(
    "case",
    [
        "explicit_false",
        "explicit_padding_free",
        "env_off",
        "collator",
        "stream",
        "no_dataset",
        "no_text",
    ],
)
def test_default_packing_stays_off(monkeypatch, case):
    SFT, calls = _patched(monkeypatch)
    config, kwargs = _config(), dict(train_dataset = _text())
    if case == "explicit_false":
        config.packing = False
    elif case == "explicit_padding_free":
        config.padding_free = False
    elif case == "env_off":
        monkeypatch.setenv("UNSLOTH_DEFAULT_PACKING", "0")
    elif case == "collator":
        kwargs["data_collator"] = lambda rows: rows
    elif case == "stream":
        kwargs["train_dataset"] = _text().to_iterable_dataset()
    elif case == "no_dataset":
        kwargs["train_dataset"] = None
    elif case == "no_text":
        kwargs["train_dataset"] = datasets.Dataset.from_dict({"image": ["x"]})
    SFT(model = _Model(), args = config, **kwargs)
    assert config.packing is False and calls["pack"] == 0


def test_a_trainer_refusing_default_packing_falls_back_to_padding_free(monkeypatch):
    SFT, calls = _patched(monkeypatch, refuse_packing = True)
    config = _config()
    SFT(model = _Model(), args = config, train_dataset = _text())
    assert calls["inits"] == [True, False]
    assert config.packing is False and config.padding_free is True and config.eval_packing is None
    assert calls["pack"] == 0 and calls["padding_free"] == 1


def test_unsloth_sft_config_leaves_packing_unset():
    import inspect

    from unsloth.models import rl

    source = inspect.getsource(rl._patch_trl_rl_trainers_impl)
    assert '"packing": None' in source


def test_a_reused_config_decides_again(monkeypatch):
    SFT, calls = _patched(monkeypatch)
    config = _config()
    SFT(model = _Model(), args = config, train_dataset = _text())
    assert config.packing is True
    SFT(model = _Model(), args = config, train_dataset = _text(), data_collator = lambda rows: rows)
    assert config.packing is False and calls["pack"] == 1
