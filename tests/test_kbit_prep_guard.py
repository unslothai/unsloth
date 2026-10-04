"""TRL below 0.24.0 re-prepares a model Unsloth already prepared.

``prepare_peft_model`` calls PEFT's ``prepare_model_for_kbit_training``, which
upcasts every non-``Params4bit`` parameter to float32. The one that matters is
the dense, frozen ``lm_head``: 4.74 GiB on Qwen3.8, 5.01 GiB on Muse Glimmer,
enough to OOM a T4 that is already holding the weights. TRL added
``and not isinstance(model, PeftModel)`` in 0.24.0; these tests pin that we
apply the same clause below that version, and that we touch nothing at or above
it.

The fixture is TRL 0.22.2's real function body, not a paraphrase, so a change in
how the branch is spelled shows up as a failure here rather than as a silent
no-op in the field.
"""

import linecache
import sys
import types
import pytest

from unsloth.models.rl import (
    _guard_kbit_prep_against_peft_models,
    _UNSLOTH_KBIT_PREP_GUARD_FLAG,
)

# Verbatim from trl 0.22.2 trl/models/utils.py, trimmed to the branch under test.
TRL_0_22_2_SOURCE = '''
def prepare_peft_model(model, peft_config, args):
    """Prepares a model for PEFT training."""
    if isinstance(model, PeftModel) and peft_config is not None:
        model = model.merge_and_unload()

    is_qlora = getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False)

    is_sharded_qlora = False
    if getattr(model, "is_loaded_in_4bit", False):
        for _, param in model.named_parameters():
            if param.__class__.__name__ == "Params4bit":
                is_sharded_qlora = param.data.device.type in {"cpu", "meta"}
                break

    if is_qlora and not is_sharded_qlora:
        model = prepare_model_for_kbit_training(model)
        CALLS.append("kbit_prep")
    elif args.gradient_checkpointing:
        CALLS.append("enable_gc")
    return model
'''


def _seed_linecache(filename, source):
    lines = [l + "\n" for l in source.splitlines()]
    linecache.cache[filename] = (len(source), None, lines, filename)


class FakePeftModel:
    """Stands in for a model Unsloth has already applied LoRA to."""

    is_loaded_in_4bit = True

    def named_parameters(self):
        return iter(())


class PlainModel:
    is_loaded_in_4bit = True

    def named_parameters(self):
        return iter(())


class Args:
    gradient_checkpointing = False


@pytest.fixture
def trl_modules(monkeypatch):
    """A trl.models.utils plus the trainer modules that re-export the name."""
    calls = []

    utils = types.ModuleType("trl.models.utils")
    utils.__file__ = "<trl-0.22.2-fixture>"
    utils.PeftModel = FakePeftModel
    utils.prepare_model_for_kbit_training = lambda m, **kw: m
    utils.CALLS = calls
    # inspect.getsource reads through linecache, and an installed TRL has a real
    # file behind it. Seed the cache so the fixture is readable the same way,
    # otherwise the guard bails for a reason that never occurs in the field and
    # every assertion below passes vacuously.
    _seed_linecache(utils.__file__, TRL_0_22_2_SOURCE)
    exec(compile(TRL_0_22_2_SOURCE, utils.__file__, "exec"), vars(utils))

    trl = types.ModuleType("trl")
    trl.__version__ = "0.22.2"
    models = types.ModuleType("trl.models")
    models.utils = utils
    models.prepare_peft_model = utils.prepare_peft_model
    trl.models = models

    # Every trainer module that does `from ..models import prepare_peft_model`.
    trainers = {}
    for name in (
        "sft_trainer",
        "grpo_trainer",
        "rloo_trainer",
        "prm_trainer",
        "online_dpo_trainer",
        "reward_trainer",
    ):
        m = types.ModuleType(f"trl.trainer.{name}")
        m.prepare_peft_model = utils.prepare_peft_model
        trainers[name] = m
        monkeypatch.setitem(sys.modules, f"trl.trainer.{name}", m)

    monkeypatch.setitem(sys.modules, "trl", trl)
    monkeypatch.setitem(sys.modules, "trl.models", models)
    monkeypatch.setitem(sys.modules, "trl.models.utils", utils)
    return types.SimpleNamespace(
        trl = trl, utils = utils, models = models, trainers = trainers, calls = calls
    )


def test_an_already_peft_model_skips_the_upcast(trl_modules):
    assert _guard_kbit_prep_against_peft_models() is True
    trl_modules.utils.prepare_peft_model(FakePeftModel(), None, Args())
    assert "kbit_prep" not in trl_modules.calls


def test_a_plain_quantized_model_still_gets_prepared(trl_modules):
    """The guard must be narrow: a model that has NOT been through PEFT still
    needs the preparation TRL does for it."""
    assert _guard_kbit_prep_against_peft_models() is True
    trl_modules.utils.prepare_peft_model(PlainModel(), None, Args())
    assert "kbit_prep" in trl_modules.calls


def test_every_module_that_re_exports_the_name_is_rebound(trl_modules):
    """sft_trainer is the one GKD inherits, but rebinding only the definition
    would leave six other trainers on the original."""
    original = trl_modules.utils.prepare_peft_model
    assert _guard_kbit_prep_against_peft_models() is True
    for name, module in trl_modules.trainers.items():
        assert module.prepare_peft_model is not original, name
        assert getattr(module.prepare_peft_model, _UNSLOTH_KBIT_PREP_GUARD_FLAG, False), name


def test_it_is_idempotent(trl_modules):
    assert _guard_kbit_prep_against_peft_models() is True
    assert _guard_kbit_prep_against_peft_models() is False


def test_trl_at_or_above_0_24_is_left_alone(trl_modules):
    """Upstream already carries the clause; patching it again would mean
    maintaining a copy of a function we no longer need to correct."""
    trl_modules.trl.__version__ = "0.24.0"
    original = trl_modules.utils.prepare_peft_model
    assert _guard_kbit_prep_against_peft_models() is False
    assert trl_modules.utils.prepare_peft_model is original


def test_an_unrecognised_branch_is_left_alone(trl_modules, monkeypatch):
    """If TRL respells the branch, bail rather than edit blindly."""
    src = TRL_0_22_2_SOURCE.replace(
        "if is_qlora and not is_sharded_qlora:",
        "if is_qlora and (not is_sharded_qlora):",
    )
    _seed_linecache("<respelled>", src)
    exec(compile(src, "<respelled>", "exec"), vars(trl_modules.utils))
    original = trl_modules.utils.prepare_peft_model
    assert _guard_kbit_prep_against_peft_models() is False
    assert trl_modules.utils.prepare_peft_model is original


def test_a_source_checkout_with_no_metadata_is_still_guarded(trl_modules):
    """TRL run from a source tree sets ``__version__ = "unknown"``, which does
    not parse. Bailing there would leave exactly the pre-0.24 installs this
    exists for on the upcast."""
    trl_modules.trl.__version__ = "unknown"
    assert _guard_kbit_prep_against_peft_models() is True
    trl_modules.utils.prepare_peft_model(FakePeftModel(), None, Args())
    assert "kbit_prep" not in trl_modules.calls


def test_an_unparseable_version_does_not_patch_an_already_guarded_trl(trl_modules):
    """Falling through on an unparseable version is only safe because the
    source check is self-guarding: 0.24.0 and above spell the branch with the
    clause already in it, so there is nothing for us to match."""
    trl_modules.trl.__version__ = "unknown"
    src = TRL_0_22_2_SOURCE.replace(
        "if is_qlora and not is_sharded_qlora:",
        "if is_qlora and not is_sharded_qlora and not isinstance(model, PeftModel):",
    )
    _seed_linecache("<trl-0.24-fixture>", src)
    exec(compile(src, "<trl-0.24-fixture>", "exec"), vars(trl_modules.utils))
    original = trl_modules.utils.prepare_peft_model
    assert _guard_kbit_prep_against_peft_models() is False
    assert trl_modules.utils.prepare_peft_model is original
