# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Padding-free is auto-enabled, and it hands the model `packed_seq_lengths`.

A forward that declares neither that argument nor `**kwargs` raises TypeError
on the first training step, long after the trainer was built:

    Phi4ForCausalLMV.forward() got an unexpected keyword argument
    'packed_seq_lengths'

so the gate asks the signature instead of the model name. It only ever turns
padding-free OFF where the signature positively shows it cannot work: anything
unknown or uninspectable answers True, because refusing on an unreadable
signature would silently disable padding-free for models that support it.
"""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

try:
    from unsloth.trainer import _forward_accepts_packing_kwargs  # noqa: E402
except ImportError:
    # On Apple Silicon with MLX, `unsloth/__init__.py` replaces `unsloth.trainer` with a
    # synthetic module carrying only the MLX trainer names, so `unsloth/trainer.py` never
    # loads and every private helper in it is unreachable. Nothing to test there either:
    # padding-free belongs to the torch trainer, and MLX training does not go through it.
    # Skip rather than let the whole module fail collection, which aborts more than itself.
    import unsloth
    if getattr(unsloth, "DEVICE_TYPE", None) != "mlx":
        raise
    pytest.skip("unsloth.trainer is the MLX shim here", allow_module_level = True)


class _NoKwargs(nn.Module):
    """microsoft/Phi-4-reasoning-vision-15B's shape: no **kwargs, no packing."""

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
        labels = None,
    ):
        return None


class _TakesKwargs(nn.Module):
    def forward(
        self,
        input_ids = None,
        **kwargs,
    ):
        return None


class _NamesPacking(nn.Module):
    def forward(
        self,
        input_ids = None,
        packed_seq_lengths = None,
    ):
        return None


class _FakePeft(nn.Module):
    """PEFT forwards **kwargs straight through, so the wrapper always says yes
    while the checkpoint underneath is the one that raises."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def get_base_model(self):
        return self.inner

    def forward(self, *args, **kwargs):
        return self.inner(*args, **kwargs)


class _InnerDecoderTakesKwargs(nn.Module):
    """`PreTrainedModel.base_model` is a property returning the inner decoder,
    whose forward usually does take **kwargs. Following it answers for the
    wrong module, so the unwrap is PEFT's `get_base_model` only."""

    def __init__(self):
        super().__init__()
        self.model = _TakesKwargs()

    @property
    def base_model(self):
        return self.model

    def forward(
        self,
        input_ids = None,
        attention_mask = None,
    ):
        return None


class _SelfReturningUnwrap(nn.Module):
    def get_base_model(self):
        return self

    def forward(self, input_ids = None):
        return None


class _RaisingUnwrap(nn.Module):
    def get_base_model(self):
        raise RuntimeError("adapter not ready")

    def forward(self, input_ids = None):
        return None


@pytest.mark.parametrize(
    "model, expected, why",
    [
        (_NoKwargs(), False, "the real defect"),
        (_TakesKwargs(), True, "**kwargs absorbs it"),
        (_NamesPacking(), True, "names it explicitly"),
        (_FakePeft(_NoKwargs()), False, "unwrap past the adapter"),
        (_FakePeft(_TakesKwargs()), True, "unwrapped model accepts it"),
        (_InnerDecoderTakesKwargs(), False, "base_model must NOT be followed"),
        (_SelfReturningUnwrap(), False, "self-returning unwrap terminates"),
        (_RaisingUnwrap(), False, "a raising unwrap falls back to the wrapper"),
        (None, True, "unknown fails open"),
        ("meta-llama/Llama-3.1-8B", True, "a name, not a model: fails open"),
        (object(), True, "no forward at all: fails open"),
    ],
)
def test_gate(model, expected, why):
    assert _forward_accepts_packing_kwargs(model) is expected, why


def test_the_defect_is_real():
    """Negative control: the rejected shape really does raise."""
    with pytest.raises(TypeError, match = "packed_seq_lengths"):
        _NoKwargs()(input_ids = torch.zeros(1, 4).long(), packed_seq_lengths = [4])


def test_the_accepted_shape_really_accepts_it():
    """And the shape the gate allows really does tolerate the argument."""
    _TakesKwargs()(input_ids = torch.zeros(1, 4).long(), packed_seq_lengths = [4])


def test_the_blocker_names_itself_in_the_warning():
    """The reason chain must not blame an unset environment variable.

    When this gate is the sole blocker and the user asked for packing=True,
    the chain used to fall through to `reason = "UNSLOTH_RETURN_LOGITS=1"`,
    telling the user to investigate a flag they never set.
    """
    import inspect as _inspect

    from unsloth import trainer as trainer_module

    source = _inspect.getsource(trainer_module._patch_sft_trainer_auto_packing)
    assert "forward_rejects_packing" in source
    # the new branch must come BEFORE the catch-all env-var branch
    assert source.index("elif forward_rejects_packing") < source.index(
        'reason = "UNSLOTH_RETURN_LOGITS=1"'
    )
    # and the predicate is evaluated once, not twice
    assert source.count("_forward_accepts_packing_kwargs(model)") == 1


@pytest.mark.parametrize("packing", [False, True])
def test_a_string_model_is_rechecked_once_trl_has_built_it(monkeypatch, packing):
    """The gate fails open on a string, so the real check has to happen after init.

    Both `enable_padding_free_metadata` and `enable_sample_packing` put
    `packed_seq_lengths` into the batch, so the recheck has to turn both off;
    clearing padding-free alone leaves the packing branch raising the same
    TypeError. Driven through the real wrapper with a stub `SFTTrainer.__init__`,
    so nothing is downloaded.
    """
    from types import SimpleNamespace

    import unsloth.trainer as trainer_module

    built = _NoKwargs()

    class _StubSFTTrainer:
        # The collator and the dataset are built from the config that was live at the time,
        # which is the point of the test: only a rebuilt trainer gets padded ones.
        inits = []

        def __init__(
            self,
            model = None,
            args = None,
            **kwargs,
        ):
            # What TRL does with a string: materialize it, then expose it as self.model.
            self.model = built if isinstance(model, str) else model
            self.args = args
            self.collator_padding_free = bool(getattr(args, "padding_free", False))
            self.dataset_packed = bool(getattr(args, "packing", False))
            _StubSFTTrainer.inits.append((self.collator_padding_free, self.dataset_packed))

    injected = []
    for _name in ("enable_padding_free_metadata", "enable_sample_packing"):
        monkeypatch.setattr(trainer_module, _name, lambda model, trainer: injected.append(model))
    # No hub access: the config is irrelevant to the signature question.
    monkeypatch.setattr(trainer_module, "_resolve_string_model_config", lambda *a, **k: None)

    module = SimpleNamespace(SFTTrainer = _StubSFTTrainer)
    trainer_module._patch_sft_trainer_auto_packing(module)

    config = SimpleNamespace(packing = packing, padding_free = True, max_length = 512)
    instance = module.SFTTrainer(model = "microsoft/Phi-4-reasoning-vision-15B", args = config)

    assert injected == [], "nothing may wrap the collator for this forward"
    for target in (config, instance.args):
        assert target.padding_free is False
        assert target.packing is False
    # Flipping the flags after the fact is not enough and is worse than doing nothing: the
    # collator and dataset TRL already built would keep flattening while nothing names the
    # sequence boundaries, so training would cross them silently instead of raising. Only a
    # second __init__ replaces both.
    assert len(_StubSFTTrainer.inits) == 2, "the trainer must be rebuilt, not just re-flagged"
    assert _StubSFTTrainer.inits[-1] == (False, False)
    assert instance.collator_padding_free is False
    assert instance.dataset_packed is False
