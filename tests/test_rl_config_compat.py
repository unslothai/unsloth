"""Config arguments the installed TRL retired must not crash trainer construction.

`unsloth/models/rl.py` emits `Unsloth<X>Config.__init__` with a `**kwargs`
catch-all that used to be splatted raw into `super().__init__()`. TRL removed
`GRPOConfig.max_prompt_length` in 0.28.0, so every pinned notebook that sets it
died with

    TypeError: GRPOConfig.__init__() got an unexpected keyword argument
    'max_prompt_length'

the moment TRL was upgraded. Fourteen `DPOConfig` fields went the same way in
0.29.0. `filter_config_init_kwargs` is what absorbs that.

The module is loaded by file spec: `import unsloth.models.rl_config_compat`
would execute `unsloth/__init__.py` first and drag in torch, numpy and
unsloth_zoo, which these tests neither need nor want.
"""

import dataclasses
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "unsloth" / "models" / "rl_config_compat.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_unsloth_rl_config_compat_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MODULE = _load_module()
filter_config_init_kwargs = _MODULE.filter_config_init_kwargs
TRL_CONFIG_RENAMES = _MODULE.TRL_CONFIG_RENAMES


@dataclasses.dataclass
class ModernGRPOConfig:
    """Stands in for TRL >= 0.28 `GRPOConfig`: no `max_prompt_length`, and the
    post-rename spellings of the three arguments TRL renamed."""

    output_dir: str = "out"
    max_completion_length: int = 256
    use_liger_kernel: bool = False
    vllm_structured_outputs_regex: str = None
    log_unique_prompts: bool = False


@dataclasses.dataclass
class LegacyGRPOConfig:
    """Stands in for TRL <= 0.27: the retired spellings are still real fields."""

    output_dir: str = "out"
    max_prompt_length: int = None
    use_liger_loss: bool = False
    vllm_guided_decoding_regex: str = None
    wandb_log_unique_prompts: bool = False


def _collect(config_class, arguments):
    """Filter `arguments`, returning the survivors and the messages emitted."""
    messages = []
    kept = filter_config_init_kwargs(config_class, arguments, notify=messages.append)
    return kept, messages


def test_a_retired_argument_is_dropped_rather_than_raising():
    """The bug: this exact call is what a pinned GRPO notebook makes."""
    kept, messages = _collect(
        ModernGRPOConfig, {"output_dir": "out", "max_prompt_length": 256}
    )
    assert kept == {"output_dir": "out"}
    # Constructing with the survivors is the thing that used to raise.
    assert ModernGRPOConfig(**kept).output_dir == "out"
    assert any("max_prompt_length" in m for m in messages)


def test_the_drop_is_announced_with_trls_own_advice():
    """A silent drop would change training semantics behind the user's back."""
    _, messages = _collect(ModernGRPOConfig, {"max_prompt_length": 256})
    assert len(messages) == 1
    assert "IGNORED" in messages[0]
    assert "filter overlong prompts" in messages[0]


def test_every_documented_rename_is_carried_across():
    """TRL renamed these; dropping them would silently disable real features."""
    kept, _ = _collect(
        ModernGRPOConfig,
        {
            "use_liger_loss": True,
            "vllm_guided_decoding_regex": "abc",
            "wandb_log_unique_prompts": True,
        },
    )
    assert kept == {
        "use_liger_kernel": True,
        "vllm_structured_outputs_regex": "abc",
        "log_unique_prompts": True,
    }
    assert ModernGRPOConfig(**kept).use_liger_kernel is True


def test_a_rename_overwrites_the_mirrored_default_not_a_real_value():
    """The generated __init__ always passes the new name, carrying the class
    default when untouched. The rename must win over that default..."""
    kept, _ = _collect(
        ModernGRPOConfig, {"use_liger_kernel": False, "use_liger_loss": True}
    )
    assert kept["use_liger_kernel"] is True


def test_an_explicitly_set_new_name_beats_the_old_one():
    """...but must not clobber a value the caller actually chose."""
    kept, messages = _collect(
        ModernGRPOConfig,
        {"vllm_structured_outputs_regex": "mine", "vllm_guided_decoding_regex": "old"},
    )
    assert kept["vllm_structured_outputs_regex"] == "mine"
    assert any("ignored" in m for m in messages)


def test_the_two_pass_result_does_not_depend_on_ordering():
    """`**kwargs` lands last today, but nothing in the contract promises it."""
    forward = {"use_liger_kernel": False, "use_liger_loss": True}
    backward = {"use_liger_loss": True, "use_liger_kernel": False}
    assert _collect(ModernGRPOConfig, forward)[0] == _collect(
        ModernGRPOConfig, backward
    )[0]


def test_an_older_trl_that_still_has_the_field_is_left_alone():
    """Forwards compatible is not enough; the pinned stacks must not change."""
    arguments = {
        "max_prompt_length": 256,
        "use_liger_loss": True,
        "vllm_guided_decoding_regex": "abc",
        "wandb_log_unique_prompts": True,
    }
    kept, messages = _collect(LegacyGRPOConfig, arguments)
    assert kept == arguments
    assert messages == []


def test_a_field_retired_on_one_config_survives_on_another():
    """`max_completion_length` is gone from DPOConfig but current on GRPOConfig,
    so the advice table must never fire on acceptance alone."""
    kept, messages = _collect(ModernGRPOConfig, {"max_completion_length": 64})
    assert kept == {"max_completion_length": 64}
    assert messages == []


def test_an_unknown_argument_is_reported_by_name():
    """A typo stops being fatal, so the message has to carry the whole signal."""
    kept, messages = _collect(ModernGRPOConfig, {"learnign_rate": 3e-4})
    assert kept == {}
    assert len(messages) == 1
    assert "learnign_rate" in messages[0]
    assert "IGNORED" in messages[0]


def test_a_config_taking_its_own_kwargs_is_never_filtered():
    """Nothing can be judged unacceptable if the base forwards it onwards."""

    class Permissive:
        def __init__(self, output_dir="out", **kwargs):
            pass

    arguments = {"output_dir": "out", "anything_at_all": 1}
    kept, messages = _collect(Permissive, arguments)
    assert kept == arguments
    assert messages == []


def test_an_unreadable_signature_forwards_everything_unchanged():
    """Guessing would be worse than the status quo, so it stands down."""

    arguments = {"whatever": 1}
    kept, messages = _collect(object(), arguments)
    assert kept == arguments
    assert messages == []


def test_empty_kwargs_short_circuit():
    """The common path allocates nothing and says nothing."""
    messages = []
    assert filter_config_init_kwargs(ModernGRPOConfig, {}, notify=messages.append) == {}
    assert messages == []


def test_rename_targets_are_real_fields_of_the_modern_config():
    """A typo in the rename table would silently degrade to a plain drop."""
    modern = {f.name for f in dataclasses.fields(ModernGRPOConfig)}
    for old, new in TRL_CONFIG_RENAMES.items():
        assert old not in modern, old
        assert new in modern, new


def test_a_default_factory_field_is_compared_not_crashed_on():
    """`include_for_metrics` is a real GRPOConfig field with a list factory;
    reading its default must not raise while resolving a rename."""

    @dataclasses.dataclass
    class WithFactory:
        include_for_metrics: list = dataclasses.field(default_factory=list)
        use_liger_kernel: bool = False

    kept, _ = _collect(
        WithFactory, {"include_for_metrics": [], "use_liger_loss": True}
    )
    assert kept["use_liger_kernel"] is True
    assert kept["include_for_metrics"] == []


# The tests above exercise the filter. These two guard the wiring: without
# them, reverting the rl.py template edit would leave every test above green
# while the generated config went back to splatting kwargs raw into
# super().__init__(). rl.py is read as text because importing it pulls in
# torch, trl and unsloth_zoo.

RL_SOURCE = (REPO_ROOT / "unsloth" / "models" / "rl.py").read_text()


def test_the_generated_config_routes_super_through_the_filter():
    assert "_unsloth_config_arguments = dict({RLConfig_call_args}{RLConfig_kwargs})" in RL_SOURCE
    assert (
        "super().__init__(**_unsloth_filter_config_init_kwargs("
        "{RLConfig_name}, _unsloth_config_arguments))"
    ) in RL_SOURCE
    # The raw splat is what the fix removes; it must not come back.
    assert "super().__init__({RLConfig_call_args}{RLConfig_kwargs})" not in RL_SOURCE


def test_the_generated_file_imports_the_filter_with_a_safe_fallback():
    assert (
        "from unsloth.models.rl_config_compat import filter_config_init_kwargs"
        " as _unsloth_filter_config_init_kwargs"
    ) in RL_SOURCE
    # An import failure must degrade to the historical passthrough, never to a
    # NameError inside a generated trainer.
    assert (
        "def _unsloth_filter_config_init_kwargs(config_class, kwargs): return kwargs"
        in RL_SOURCE
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
