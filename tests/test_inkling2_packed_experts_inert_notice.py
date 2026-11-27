# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""UNSLOTH_INKLING2_PACKED_EXPERTS=1 only engages on the released NVFP4 / MXFP8 checkpoint. With load_in_4bit = True
the loader resolves Inkling 2 to its bitsandbytes 4bit copy (NF4 experts), where the flag cannot do anything: the
loader says so once instead of loading silently. The released checkpoint, the flag off, other models and full
finetuning print nothing."""

import types

import pytest

import unsloth  # noqa: F401
from unsloth.models import inkling2_packed
from unsloth.models.loader_utils import check_and_disable_bitsandbytes_loading

ENV = "UNSLOTH_INKLING2_PACKED_EXPERTS"
NOTICE = f"{ENV}=1 has no effect"


def _config(model_type = "inkling2", quant = "bitsandbytes"):
    text = types.SimpleNamespace(
        model_type = model_type + "_text" if model_type == "inkling2" else model_type
    )
    config = types.SimpleNamespace(model_type = model_type, text_config = text)
    if quant is not None:
        config.quantization_config = {"quant_method": quant}
    return config


@pytest.fixture(autouse = True)
def _fresh(monkeypatch):
    monkeypatch.delenv("UNSLOTH_ENABLE_FULL_FINETUNING", raising = False)
    if hasattr(inkling2_packed, "_INERT_WARNED"):
        monkeypatch.setattr(inkling2_packed, "_INERT_WARNED", False)


def _load(config, name = "unsloth/Inkling-2-Mini-unsloth-bnb-4bit"):
    return check_and_disable_bitsandbytes_loading(config, load_in_4bit = True, model_name = name)


def test_flag_on_bnb_4bit_copy_says_so_once(monkeypatch, capsys):
    monkeypatch.setenv(ENV, "1")
    assert _load(_config())[0] is True  # the 4bit load itself is unchanged
    out = capsys.readouterr().out
    assert out.count(NOTICE) == 1, out
    assert "unsloth/Inkling-2-Mini-unsloth-bnb-4bit" in out and "load_in_4bit = False" in out
    _load(_config())
    assert NOTICE not in capsys.readouterr().out  # one time only


@pytest.mark.parametrize(
    "env, config, full_ft",
    [
        ("0", _config(), False),  # flag off
        ("1", _config(model_type = "llama"), False),  # not Inkling 2
        ("1", _config(), True),  # full finetuning keeps the experts dequantized anyway
    ],
)
def test_no_notice_otherwise(monkeypatch, capsys, env, config, full_ft):
    monkeypatch.setenv(ENV, env)
    if full_ft:
        monkeypatch.setenv("UNSLOTH_ENABLE_FULL_FINETUNING", "1")
    _load(config, name = "some/model")
    assert NOTICE not in capsys.readouterr().out


def test_released_checkpoint_engages_without_notice(monkeypatch, capsys):
    """The released checkpoint (ModelOpt marker) takes the packed path: no 'no effect' notice there."""
    monkeypatch.setenv(ENV, "1")
    installed = []
    monkeypatch.setattr(
        inkling2_packed,
        "install_inkling2_packed_expert_loading",
        lambda: installed.append(1) or True,
    )
    _load(_config(quant = "modelopt"), name = "unsloth/Inkling-2-Mini")
    out = capsys.readouterr().out
    assert installed == [1]
    assert NOTICE not in out and "keeps them packed as released" in out
