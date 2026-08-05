# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Padding-free + `max_length` handshake across TRL versions.

TRL >= 1.0.0 refuses to build an SFTTrainer when padding-free is on, packing is
off and `args.max_length` is set:

    ValueError: When `padding_free=True` without packing, `max_length` is not
    enforced. Either enable packing ..., or set `max_length=None`.

Unsloth auto-enables padding-free whenever the user leaves `padding_free` at its
`None` default, and rl.py's `max_length_check` codegen always wrote
`args.max_length`, so every default SFT run tripped that guard. rl.py now hands
those TRLs the `None` they ask for and keeps truncating through
`max_seq_length`, which is what Unsloth's own dataset prep reads.

The assertions branch on whether the installed TRL actually carries the guard,
so the same file pins the new behaviour on TRL >= 1.0.0 and the untouched old
behaviour on TRL < 1.0.0.
"""

from __future__ import annotations

import os

os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("ACCELERATE_MIXED_PRECISION", "no")

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest


if importlib.util.find_spec("torch") is None:
    pytest.skip("torch not installed", allow_module_level = True)
if importlib.util.find_spec("trl") is None or importlib.util.find_spec("unsloth") is None:
    pytest.skip("trl or unsloth not installed", allow_module_level = True)

# Apply the CUDA spoof before any unsloth-touching import, so a CPU-only runner
# can still import unsloth and generate the patched trainer.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _zoo_aggressive_cuda_spoof as _spoof  # noqa: E402

_spoof.apply()

import torch  # noqa: E402


def _eager_compile(model = None, *args, **kwargs):
    if callable(model):
        return model
    return lambda fn: fn


torch.compile = _eager_compile
if hasattr(torch, "accelerator"):
    torch.accelerator.is_available = lambda *a, **k: False

_MODEL = "hf-internal-testing/tiny-random-LlamaForCausalLM"
# Model-level length the trainer resolves from when the user names none.
_MODEL_MAX_SEQ_LENGTH = 128
# Deliberately smaller than _MODEL_MAX_SEQ_LENGTH, so honouring it is visible.
_USER_MAX_LENGTH = 64


@pytest.fixture(scope = "module")
def trl_has_guard():
    global torch  # the `import torch._dynamo` below would otherwise shadow it
    import unsloth  # noqa: F401
    torch.compile = _eager_compile
    try:
        import torch._dynamo
        torch._dynamo.config.disable = True
    except Exception:
        pass
    from trl.trainer import sft_trainer

    return "`max_length` is not enforced" in inspect.getsource(sft_trainer)


def _load_plain():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(_MODEL)
        model = AutoModelForCausalLM.from_pretrained(_MODEL, dtype = torch.float32)
    except OSError as e:
        pytest.skip(f"could not fetch {_MODEL} (network/hub): {str(e)[:150]}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.max_seq_length = _MODEL_MAX_SEQ_LENGTH
    return model.to("cpu"), tok


def _build(tmp_path, **config_kwargs):
    """Construct the Unsloth-patched SFTTrainer over a long, truncatable dataset."""
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    assert SFTTrainer.__name__ == "UnslothSFTTrainer", "SFT patch did not apply"
    model, tok = _load_plain()
    ds = Dataset.from_list([{"text": "The quick brown fox. " * 200}] * 4)
    cfg = SFTConfig(
        output_dir = str(tmp_path),
        per_device_train_batch_size = 2,
        max_steps = 1,
        report_to = "none",
        save_strategy = "no",
        use_cpu = True,
        dataset_text_field = "text",
        fp16 = False,
        bf16 = False,
        optim = "adamw_torch",
        **config_kwargs,
    )
    return SFTTrainer(model = model, processing_class = tok, args = cfg, train_dataset = ds)


def _longest(trainer):
    return max(len(x) for x in trainer.train_dataset["input_ids"])


def test_default_sft_construction_does_not_trip_the_guard(tmp_path, trl_has_guard):
    """The reported break: a plain SFTTrainer(), no padding_free / max_length given."""
    trainer = _build(tmp_path)
    args = trainer.args

    assert args.padding_free is True, "padding-free should still auto-enable"
    assert args.packing is False
    assert args.max_seq_length == _MODEL_MAX_SEQ_LENGTH
    if trl_has_guard:
        # What TRL >= 1.0.0 asks for; truncation moves to Unsloth's dataset prep.
        assert args.max_length is None
    else:
        assert args.max_length == _MODEL_MAX_SEQ_LENGTH
    assert _longest(trainer) == _MODEL_MAX_SEQ_LENGTH


def test_explicit_max_length_is_still_truncated_to(tmp_path, trl_has_guard):
    """An explicit `max_length` is honoured, not silently dropped.

    Under padding-free the length is enforced during Unsloth's dataset prep
    (via `max_seq_length`) rather than by TRL, so the user keeps both their
    truncation length and the padding-free speedup.
    """
    trainer = _build(tmp_path, max_length = _USER_MAX_LENGTH)
    args = trainer.args

    assert args.padding_free is True
    if trl_has_guard:
        assert args.max_length is None
        assert args.max_seq_length == _USER_MAX_LENGTH
    else:
        assert args.max_length == _MODEL_MAX_SEQ_LENGTH
    assert _longest(trainer) == (
        _USER_MAX_LENGTH if trl_has_guard else _MODEL_MAX_SEQ_LENGTH
    )


def test_padding_free_off_keeps_max_length(tmp_path):
    """Nothing is cleared when padding-free is not in play."""
    trainer = _build(tmp_path, padding_free = False)

    assert trainer.args.padding_free is False
    assert trainer.args.max_length == _MODEL_MAX_SEQ_LENGTH


def test_packing_keeps_max_length(tmp_path):
    """TRL's guard only fires without packing, so packing runs keep max_length."""
    trainer = _build(tmp_path, packing = True)

    assert trainer.args.packing is True
    assert trainer.args.max_length == _MODEL_MAX_SEQ_LENGTH


def test_generator_only_emits_the_none_for_a_trl_that_guards():
    """The codegen edit is gated on the guard text, so old TRLs are untouched."""
    from unsloth.models import rl

    source = inspect.getsource(rl)
    assert '"`max_length` is not enforced" in old_RLTrainer_source' in source


@pytest.mark.parametrize(
    "message, expected",
    [
        (
            "When `padding_free=True` without packing, `max_length` is not enforced.",
            True,
        ),
        ("Some other max_length problem", False),
        ("padding_free is unsupported here", False),
    ],
)
def test_padding_free_error_matcher(message, expected):
    from unsloth.trainer import _should_skip_auto_padding_free_error

    assert _should_skip_auto_padding_free_error(ValueError(message)) is expected
