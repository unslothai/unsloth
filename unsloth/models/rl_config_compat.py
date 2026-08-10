# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Forward only the config arguments the installed TRL still accepts.

`unsloth/models/rl.py` generates, for every `trl.trainer.*_trainer` module, a
`Unsloth<X>Config(<X>Config)` whose `__init__` mirrors the installed
`<X>Config.__init__` signature and ends in a `**kwargs` catch-all that is
splatted straight into `super().__init__()`.

The mirrored part tracks TRL correctly, because it is built from
`inspect.signature` at generation time. The catch-all does not. When TRL
retires a config field, a caller still passing it lands in `**kwargs` and
detonates inside the base dataclass:

    TypeError: GRPOConfig.__init__() got an unexpected keyword argument
    'max_prompt_length'

TRL removed `max_prompt_length` from `GRPOConfig` in 0.28.0, so every pinned
notebook and script that sets it stops working the moment TRL is upgraded,
even though nothing about the user's code is wrong. Fourteen `DPOConfig`
fields went the same way in 0.29.0.

This module is the filter for that catch-all. It is deliberately dependency
free (stdlib only) so it can be imported without pulling in torch, and so the
generated cache file can import it behind a try/except.

Three outcomes per unrecognised argument:

* TRL documented a straight rename (`use_liger_loss` -> `use_liger_kernel`)
  and the new name exists on the installed config: the value is moved across,
  so the user's intent survives intact.
* TRL removed the field outright (`max_prompt_length`): the value is dropped
  and the user is told, by name, along with TRL's own advice on what to do
  instead. Dropping silently would change training semantics behind the
  user's back; raising would simply reinstate the breakage this exists to fix.
* Anything else unrecognised: dropped with the same warning. This is the one
  real cost of the approach, and it is taken knowingly (see below).

Known limitation, when both the old and the new name of a renamed argument are
passed at once: the generated `__init__` mirrors every parameter with a real
default, so by the time the filter runs it cannot tell "the caller passed the
new name" from "the new name is sitting at its default". A caller who sets the
new name to exactly its default value *and* sets the old name to something else
gets the old name's value. That only bites on contradictory input, and the
message says which name won.

On typos: before this change, `GRPOConfig(learnign_rate = 3e-4)` raised a
`TypeError`. Now it prints a warning naming `learnign_rate` and carries on.
The mistake is still surfaced, and prominently, but it is no longer fatal.
That trade is accepted because the failure mode it replaces -- a hard crash
on TRL upgrade for code that was correct when it was written -- is both far
more common and impossible for the user to fix without editing library code.
"""

import dataclasses
import inspect

__all__ = [
    "filter_config_init_kwargs",
    "TRL_CONFIG_RENAMES",
    "TRL_REMOVED_FIELD_ADVICE",
]


# Straight renames, taken from TRL's own deprecation warnings. Only entries
# where the old value can be handed to the new name unchanged belong here.
# `DPOConfig.max_completion_length` and `DPOConfig.rpo_alpha` deliberately do
# not: TRL suggests `max_length` and a `loss_type`/`loss_weights` combination
# respectively, and neither is a like-for-like substitution.
TRL_CONFIG_RENAMES = {
    # Deprecated in TRL 0.26.0 (DPO) / 0.27.0 (GRPO), removed in 0.28.0.
    "use_liger_loss": "use_liger_kernel",
    # Deprecated in TRL 0.27.0, removed in 0.28.0.
    "vllm_guided_decoding_regex": "vllm_structured_outputs_regex",
    # Deprecated in TRL 0.26.0, removed in 0.27.0.
    "wandb_log_unique_prompts": "log_unique_prompts",
}


# What TRL tells users to do instead, condensed. Keyed on the retired argument
# name. Only consulted for arguments the installed config genuinely rejects,
# so an entry here is harmless for a TRL version that still accepts the field
# (`max_completion_length` is retired on `DPOConfig` but current on
# `GRPOConfig`, and a GRPO caller never reaches this table).
TRL_REMOVED_FIELD_ADVICE = {
    # GRPOConfig, removed in TRL 0.28.0.
    "max_prompt_length": (
        "filter overlong prompts out of your dataset before training instead"
    ),
    # DPOConfig, removed in TRL 0.29.0.
    "max_completion_length": "use `max_length` to cap total sample length instead",
    "base_model_attribute_name": "the base model is now retrieved via `get_decoder`",
    "force_use_ref_model": "no longer needed, pass `ref_model` and it is used automatically",
    "generate_during_eval": "use a `TrainerCallback` instead",
    "label_pad_token_id": "this value is no longer configurable",
    "model_adapter_name": "only the default adapter is supported now",
    "ref_adapter_name": "the trainer handles the reference adapter itself now",
    "ref_model_init_kwargs": (
        "build the reference model yourself and pass it as `ref_model`"
    ),
    "reference_free": "use `CPOTrainer` for a reference-free objective",
    "rpo_alpha": (
        "add 'sft' to `loss_type` and set its weight in `loss_weights` instead"
    ),
    "tools": "pass tools through the dataset instead",
    "use_logits_to_keep": "the DPO trainer no longer uses this setting",
    "padding_value": "this value is no longer configurable",
}


# Introspection is cheap but configs are rebuilt inside sweeps, so memoise on
# the class object. Config classes are module level and outlive the process.
_ACCEPTED_CACHE = {}


def _accepted_parameters(config_class):
    """Names `config_class.__init__` accepts, and whether it takes `**kwargs`.

    A config whose `__init__` has its own `**kwargs` cannot be judged -- it may
    forward anywhere -- so it is reported as accepting everything and the
    caller forwards unfiltered, exactly as before this module existed.
    """
    try:
        cached = _ACCEPTED_CACHE.get(config_class)
    except TypeError:  # unhashable, so uncacheable; fall through and recompute
        cached = None
    if cached is not None:
        return cached

    names = set()
    takes_var_keyword = False
    try:
        parameters = inspect.signature(config_class.__init__).parameters
    except (TypeError, ValueError):
        parameters = None

    if parameters is not None:
        for name, parameter in parameters.items():
            if name == "self":
                continue
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                takes_var_keyword = True
            elif parameter.kind is not inspect.Parameter.VAR_POSITIONAL:
                names.add(name)
    elif dataclasses.is_dataclass(config_class):
        # A C-implemented or otherwise unreadable `__init__` on a dataclass:
        # the field list is the next best description of what it accepts.
        names = {f.name for f in dataclasses.fields(config_class) if f.init}
    else:
        # Nothing to go on. Forward everything rather than guess.
        takes_var_keyword = True

    result = (frozenset(names), takes_var_keyword)
    try:
        _ACCEPTED_CACHE[config_class] = result
    except TypeError:  # unhashable class, vanishingly rare but not fatal
        pass
    return result


_MISSING = object()


def _field_default(config_class, name):
    """The declared default for `name`, or `_MISSING` if it cannot be read.

    Used to tell "the caller set the new name too" from "the new name is
    sitting at its default because nobody touched it". Every generated config
    passes the full mirrored parameter list, so a rename target is essentially
    always present in the dict and presence alone proves nothing.
    """
    try:
        if dataclasses.is_dataclass(config_class):
            for field in dataclasses.fields(config_class):
                if field.name != name:
                    continue
                if field.default is not dataclasses.MISSING:
                    return field.default
                if field.default_factory is not dataclasses.MISSING:
                    return field.default_factory()
                return _MISSING
        parameter = inspect.signature(config_class.__init__).parameters.get(name)
        if parameter is not None and parameter.default is not inspect.Parameter.empty:
            return parameter.default
    except Exception:
        pass
    return _MISSING


def _is_untouched(config_class, name, value):
    """True if `value` is indistinguishable from `name`'s declared default."""
    default = _field_default(config_class, name)
    if default is _MISSING:
        return False
    try:
        return bool(default == value)
    except Exception:
        return default is value


def _default_notifier(message):
    print(message)


def filter_config_init_kwargs(config_class, kwargs, notify=None):
    """Return `kwargs` reduced to what `config_class.__init__` will accept.

    Renames are applied where TRL documented one; everything else the config
    rejects is dropped, and each decision is reported through `notify`
    (`print` by default, which is what the surrounding generated code uses and
    what shows up reliably in a notebook).
    """
    if not kwargs:
        return kwargs

    accepted, takes_var_keyword = _accepted_parameters(config_class)
    if takes_var_keyword:
        return kwargs

    if notify is None:
        notify = _default_notifier
    config_name = getattr(config_class, "__name__", str(config_class))

    # Two passes, so the outcome does not depend on dict ordering: everything
    # the config accepts is taken first, then the leftovers are renamed onto
    # it or dropped.
    forwarded = {key: value for key, value in kwargs.items() if key in accepted}

    for key, value in kwargs.items():
        if key in accepted:
            continue

        renamed = TRL_CONFIG_RENAMES.get(key)
        if renamed is not None and renamed in accepted:
            # The new name is a mirrored parameter, so it is already in the
            # dict carrying either the caller's value or the class default.
            # Only overwrite in the latter case: an explicitly set new name
            # is the more current expression of intent and wins.
            existing = kwargs.get(renamed, _MISSING)
            if existing is _MISSING or _is_untouched(config_class, renamed, existing):
                forwarded[renamed] = value
                notify(
                    f"Unsloth: TRL renamed `{key}` to `{renamed}`. Forwarding your "
                    f"value to `{renamed}` - update your code when convenient."
                )
            else:
                notify(
                    f"Unsloth: `{key}` was renamed to `{renamed}` by TRL and this "
                    f"{config_name} accepts only the new name. You set both, so "
                    f"`{key}` is ignored and your `{renamed}` is kept."
                )
            continue

        advice = TRL_REMOVED_FIELD_ADVICE.get(key)
        if advice:
            notify(
                f"Unsloth: `{key}` is not supported by the installed TRL's "
                f"{config_name} and will be IGNORED - {advice}."
            )
        else:
            notify(
                f"Unsloth: `{key}` is not a valid {config_name} argument for the "
                f"installed TRL and will be IGNORED. Check the spelling, or your "
                f"TRL version if this argument used to work."
            )

    return forwarded
