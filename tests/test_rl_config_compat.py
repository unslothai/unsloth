"""Config arguments the installed TRL retired must not crash trainer construction.

The `**kwargs` catch-all in the generated `Unsloth<X>Config.__init__` used to be
splatted raw into `super().__init__()`, so a pinned notebook setting
`GRPOConfig.max_prompt_length` (removed in TRL 0.28.0) died with a `TypeError`
on upgrade. `filter_config_init_kwargs` is what absorbs that.

The module is loaded by file spec because `import unsloth.models.rl_config_compat`
would run `unsloth/__init__.py` first and drag in torch, numpy and unsloth_zoo.
"""

import ast
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
TRANSFORMERS_CONFIG_RENAMES = _MODULE.TRANSFORMERS_CONFIG_RENAMES
TRANSFORMERS_REMOVED_FIELD_ADVICE = _MODULE.TRANSFORMERS_REMOVED_FIELD_ADVICE


@dataclasses.dataclass
class ModernGRPOConfig:
    """TRL >= 0.28 `GRPOConfig`: no `max_prompt_length`, post-rename spellings."""

    output_dir: str = "out"
    max_completion_length: int = 256
    use_liger_kernel: bool = False
    vllm_structured_outputs_regex: str = None
    log_unique_prompts: bool = False


@dataclasses.dataclass
class LegacyGRPOConfig:
    """TRL <= 0.27: the retired spellings are still real fields."""

    output_dir: str = "out"
    max_prompt_length: int = None
    use_liger_loss: bool = False
    vllm_guided_decoding_regex: str = None
    wandb_log_unique_prompts: bool = False


def _collect(config_class, arguments):
    """Filter `arguments`, returning the survivors and the messages emitted."""
    messages = []
    kept = filter_config_init_kwargs(config_class, arguments, notify = messages.append)
    return kept, messages


def test_a_retired_argument_is_dropped_rather_than_raising():
    """The bug: this exact call is what a pinned GRPO notebook makes."""
    kept, messages = _collect(ModernGRPOConfig, {"output_dir": "out", "max_prompt_length": 256})
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
    """The generated __init__ always passes the new name, so the rename must win
    over the class default it carries when untouched..."""
    kept, _ = _collect(ModernGRPOConfig, {"use_liger_kernel": False, "use_liger_loss": True})
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
    assert _collect(ModernGRPOConfig, forward)[0] == _collect(ModernGRPOConfig, backward)[0]


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
    """`max_completion_length` is gone from DPOConfig but current on GRPOConfig."""
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
        def __init__(
            self,
            output_dir = "out",
            **kwargs,
        ):
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
    assert filter_config_init_kwargs(ModernGRPOConfig, {}, notify = messages.append) == {}
    assert messages == []


def test_rename_targets_are_real_fields_of_the_modern_config():
    """A typo in the rename table would silently degrade to a plain drop."""
    modern = {f.name for f in dataclasses.fields(ModernGRPOConfig)}
    for old, new in TRL_CONFIG_RENAMES.items():
        assert old not in modern, old
        assert new in modern, new


def test_the_transformers_tables_do_not_overlap_or_contradict_the_trl_ones():
    """A name in both tables would resolve to whichever was consulted first."""
    renames = set(TRANSFORMERS_CONFIG_RENAMES) | set(TRL_CONFIG_RENAMES)
    advice = set(TRANSFORMERS_REMOVED_FIELD_ADVICE) | set(_MODULE.TRL_REMOVED_FIELD_ADVICE)
    assert not (renames & advice), sorted(renames & advice)
    assert not (set(TRANSFORMERS_CONFIG_RENAMES) & set(TRL_CONFIG_RENAMES))


def test_a_field_the_installed_version_still_declares_is_never_migrated():
    """The invariant that makes a table entry safe to write ahead of its removal.

    The 28 arguments did not all go in 5.0.0: `group_by_length` survived to 5.1.0,
    `warmup_ratio` and `logging_dir` to 5.14.1. An entry is consulted only after
    the config rejects the name, so on a version that still has it the entry must
    be inert. Asserted per entry, and per installed version, rather than assuming
    one cutoff.
    """
    transformers = pytest.importorskip("transformers")
    fields = {f.name for f in dataclasses.fields(transformers.TrainingArguments)}

    for key in list(TRANSFORMERS_CONFIG_RENAMES) + list(TRANSFORMERS_REMOVED_FIELD_ADVICE):
        if key in fields:
            verdict, _ = _MODULE.classify_config_kwarg(transformers.TrainingArguments, key)
            assert verdict == "accepted", f"{key} is still a field but classified {verdict}"


def test_a_transformers_rename_target_exists_once_the_old_name_is_gone():
    """A rename is only reachable after the old name goes, so that is when its
    target has to be real. A typo there degrades the migration to a plain drop."""
    transformers = pytest.importorskip("transformers")
    fields = {f.name for f in dataclasses.fields(transformers.TrainingArguments)}

    checked = 0
    for old, new in TRANSFORMERS_CONFIG_RENAMES.items():
        if old in fields:
            continue
        assert new in fields, f"{old} renames to {new}, which does not exist"
        checked += 1
    if not checked:
        pytest.skip("this transformers still declares every renamed argument")


def test_a_transformers_5_removal_is_carried_across_on_a_real_config():
    """`warmup_ratio` is the one every notebook sets."""
    transformers = pytest.importorskip("transformers")
    fields = {f.name for f in dataclasses.fields(transformers.TrainingArguments)}
    if "warmup_ratio" in fields:
        pytest.skip("this transformers still declares warmup_ratio")

    @dataclasses.dataclass
    class ModernSFTConfig:
        output_dir: str = "out"
        warmup_steps: float = 0.0

    messages = []
    kept = filter_config_init_kwargs(
        ModernSFTConfig,
        {"output_dir": "out", "warmup_ratio": 0.1},
        notify = messages.append,
    )
    assert kept == {"output_dir": "out", "warmup_steps": 0.1}
    assert any("warmup_steps" in m for m in messages)


def test_a_rename_survives_a_default_unsloth_overrode_on_the_generated_config():
    """The bug this guards: `rl.py` mirrors the base parameter under its OWN
    default (`warmup_steps = 0.1`, `per_device_train_batch_size = 4`), so
    comparing against TRL's declared default reads Unsloth's injected value as
    caller intent and silently trains at the injected number instead.
    """

    @dataclasses.dataclass
    class ModernSFTConfig:
        output_dir: str = "out"
        warmup_steps: float = 0.0  # what TRL declares

    class UnslothSFTConfig(ModernSFTConfig):
        def __init__(
            self,
            output_dir = "out",
            warmup_steps = 0.1,
            **kwargs,
        ):
            pass  # what rl.py generates: same field, different default

    messages = []
    kept = filter_config_init_kwargs(
        ModernSFTConfig,
        {"output_dir": "out", "warmup_steps": 0.1, "warmup_ratio": 0.03},
        notify = messages.append,
        mirrored_from = UnslothSFTConfig,
    )
    assert kept["warmup_steps"] == 0.03, "the caller's warmup_ratio was thrown away"
    assert not any("ignored" in m for m in messages)


def test_a_value_the_caller_really_set_still_beats_the_rename():
    """The other half: `mirrored_from` must not turn every collision into a win."""

    @dataclasses.dataclass
    class ModernSFTConfig:
        warmup_steps: float = 0.0

    class UnslothSFTConfig(ModernSFTConfig):
        def __init__(
            self,
            warmup_steps = 0.1,
            **kwargs,
        ):
            pass

    kept, messages = [], []
    kept = filter_config_init_kwargs(
        ModernSFTConfig,
        {"warmup_steps": 0.25, "warmup_ratio": 0.03},
        notify = messages.append,
        mirrored_from = UnslothSFTConfig,
    )
    assert kept["warmup_steps"] == 0.25
    assert any("ignored" in m for m in messages)


def test_the_renames_rl_py_overrides_the_default_of_are_the_known_ones():
    """The systemic check behind the `mirrored_from` fix.

    A rename whose target `rl.py` also assigns a default is only correct because
    the config path passes `mirrored_from`; without it the injected default reads
    as caller intent and the rename is dropped. Three are in that position today.
    A fourth appearing means someone added an `rl.py` default or a rename without
    checking the interaction, so it should fail here rather than in training.
    """
    overridden = _rl_py_overridden_defaults()
    # The entries the audit found, so a matcher that silently stops working fails.
    assert {"warmup_steps", "per_device_train_batch_size", "include_num_input_tokens_seen"} <= (
        overridden
    ), sorted(overridden)

    needing_mirror = {
        old for old, new in TRANSFORMERS_CONFIG_RENAMES.items() if new in overridden
    } | {old for old, new in TRL_CONFIG_RENAMES.items() if new in overridden}
    assert needing_mirror == {
        "warmup_ratio",
        "per_gpu_train_batch_size",
        "per_gpu_eval_batch_size",
    }, sorted(needing_mirror)


def test_setting_the_new_name_to_its_own_default_is_reported_as_ambiguous():
    """The one case a value comparison cannot decide, so it is stated not hidden.

    `UnslothSFTConfig(warmup_steps = 0.1, warmup_ratio = 0.03)` passes the new
    name at exactly the default `rl.py` injects. Nothing in a mirrored parameter
    records whether it was supplied, so the legacy value wins and the message has
    to say so. Sentinel defaults would resolve it, at the cost of the signature
    that `HfArgumentParser` and users read. The trainer path is unaffected: it
    knows which names actually arrived.
    """

    @dataclasses.dataclass
    class ModernSFTConfig:
        warmup_steps: float = 0.0

    class UnslothSFTConfig(ModernSFTConfig):
        def __init__(
            self,
            warmup_steps = 0.1,
            **kwargs,
        ):
            pass

    messages = []
    kept = filter_config_init_kwargs(
        ModernSFTConfig,
        {"warmup_steps": 0.1, "warmup_ratio": 0.03},
        notify = messages.append,
        mirrored_from = UnslothSFTConfig,
    )
    assert kept["warmup_steps"] == 0.03
    assert "cannot be distinguished" in messages[0]
    assert "drop `warmup_ratio`" in messages[0]

    # No `mirrored_from` means no mirrored parameter, so no ambiguity to report.
    messages = []
    filter_config_init_kwargs(
        ModernSFTConfig,
        {"warmup_ratio": 0.03},
        notify = messages.append,
    )
    assert "cannot be distinguished" not in messages[0]


def test_a_legacy_optional_forwarded_at_none_does_not_erase_the_target():
    """`per_gpu_train_batch_size` really did default to `None` in transformers 4.x
    (checked against 4.57.6), so a wrapper mirroring that signature forwards a
    `None` nobody asked for. Writing it onto `per_device_train_batch_size` leaves
    the trainer doing arithmetic on `None`."""

    @dataclasses.dataclass
    class ModernSFTConfig:
        per_device_train_batch_size: int = 8

    kept, messages = _collect(ModernSFTConfig, {"per_gpu_train_batch_size": None})
    assert kept == {}, "an unset legacy alias must not be migrated"
    assert messages == []

    # A value that was actually chosen still migrates.
    kept, _ = _collect(ModernSFTConfig, {"per_gpu_train_batch_size": 16})
    assert kept == {"per_device_train_batch_size": 16}


def test_none_still_migrates_when_the_target_itself_defaults_to_none():
    """The guard is about losing information, not about `None` being special."""

    @dataclasses.dataclass
    class ModernSFTConfig:
        hub_token: str = None

    kept, _ = _collect(ModernSFTConfig, {"push_to_hub_token": None})
    assert kept == {"hub_token": None}


def test_an_alias_whose_target_is_read_during_post_init_is_not_a_rename():
    """`use_cpu` is consumed by `__post_init__`, which resolves `device` (a
    cached_property) and `_n_gpu` from it. Measured on transformers 5.16.1: after
    `setattr(args, "use_cpu", True)` the device stays `cuda:0`, so routing
    `no_cuda` through the trainer path would report a change that never happened.
    """
    assert "no_cuda" in TRANSFORMERS_REMOVED_FIELD_ADVICE
    assert "no_cuda" not in TRANSFORMERS_CONFIG_RENAMES


def test_a_rename_target_normalised_in_post_init_is_not_a_rename():
    """`setattr` on an existing config skips `__post_init__`, so a field that
    normalises its own value cannot be migrated by assignment."""
    assert "include_tokens_per_second" in TRANSFORMERS_REMOVED_FIELD_ADVICE
    assert "include_tokens_per_second" not in TRANSFORMERS_CONFIG_RENAMES


def test_a_default_factory_field_is_compared_not_crashed_on():
    """Reading a `default_factory` default must not raise while resolving a rename."""

    @dataclasses.dataclass
    class WithFactory:
        include_for_metrics: list = dataclasses.field(default_factory = list)
        use_liger_kernel: bool = False

    kept, _ = _collect(WithFactory, {"include_for_metrics": [], "use_liger_loss": True})
    assert kept["use_liger_kernel"] is True
    assert kept["include_for_metrics"] == []


# These two guard the wiring: reverting the rl.py template edit would leave every test above green.
# rl.py is read as text because importing it pulls in torch, trl and unsloth_zoo.
RL_SOURCE = (REPO_ROOT / "unsloth" / "models" / "rl.py").read_text(encoding = "utf-8")


def _rl_py_overridden_defaults():
    """Config parameters whose default `rl.py` rewrites in the generated `__init__`.

    Both spellings it uses: the `replacements = {...}` literals and the later
    `replacements["warmup_steps"] = 0.1` version-conditional assignments.
    """
    names = set()
    for node in ast.walk(ast.parse(RL_SOURCE)):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "replacements"
                and isinstance(node.value, ast.Dict)
            ):
                names.update(
                    k.value
                    for k in node.value.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                )
            elif (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "replacements"
                and isinstance(target.slice, ast.Constant)
            ):
                names.add(target.slice.value)
    return names


def test_the_generated_config_routes_super_through_the_filter():
    assert "_unsloth_config_arguments = dict({RLConfig_call_args}{RLConfig_kwargs})" in RL_SOURCE
    assert (
        "super().__init__(**_unsloth_filter_config_init_kwargs("
        "{RLConfig_name}, _unsloth_config_arguments, mirrored_from = __class__))"
    ) in RL_SOURCE
    # The raw splat is what the fix removes; it must not come back.
    assert "super().__init__({RLConfig_call_args}{RLConfig_kwargs})" not in RL_SOURCE


def test_the_generated_file_imports_the_filter_with_a_safe_fallback():
    assert (
        "from unsloth.models.rl_config_compat import filter_config_init_kwargs"
        " as _unsloth_filter_config_init_kwargs"
    ) in RL_SOURCE
    # An import failure must degrade to the historical passthrough, never to a NameError inside a generated trainer.
    assert (
        "def _unsloth_filter_config_init_kwargs(config_class, kwargs, **kw): return kwargs"
        in RL_SOURCE
    )
    # ...and an older Unsloth, whose filter has no `mirrored_from`, must not see it.
    assert '"mirrored_from" not in inspect.signature(' in RL_SOURCE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
