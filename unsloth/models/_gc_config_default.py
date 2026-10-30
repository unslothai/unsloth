# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Honour ``use_gradient_checkpointing = False`` when the trainer config only carries TRL's default.

TRL configs default ``gradient_checkpointing = True``, so ``Trainer.train()`` would re-enable it (~25-30% slower).
Unsloth configs record whether it was passed to ``__init__`` (or assigned on the config afterwards, e.g.
``args.gradient_checkpointing = True``) as ``_unsloth_gradient_checkpointing_explicit``, not compared to a default (which varies by TRL release); full rebuilds (``dataclasses.replace``,
``Config(**args.to_dict())``) count as explicit, keeping the old behaviour.

Rules, at trainer construction:
- model ``False`` + config at TRL's default -> config set to ``False``.
- config explicitly ``True`` -> kept.
- config not an Unsloth class -> kept, with a hint.
- model ``"unsloth"`` / ``True`` / unset -> untouched.
"""

import functools
import inspect

__all__ = [
    "EXPLICIT_FLAG",
    "track_explicit_gradient_checkpointing",
    "honour_gradient_checkpointing_off",
]

EXPLICIT_FLAG = "_unsloth_gradient_checkpointing_explicit"
_WRAPPED_FLAG = "_unsloth_tracks_explicit_gradient_checkpointing"
_NOTED = set()


def _note(key, message):
    if key in _NOTED:
        return
    _NOTED.add(key)
    print(message)


def track_explicit_gradient_checkpointing(config_class):
    """Wrap ``config_class.__init__`` to record whether ``gradient_checkpointing`` was passed. Idempotent."""
    init = config_class.__dict__.get("__init__")
    if init is None or getattr(init, _WRAPPED_FLAG, False):
        return False
    try:
        names = list(inspect.signature(init).parameters)
    except (TypeError, ValueError):
        return False
    if "gradient_checkpointing" not in names:
        return False
    position = names.index("gradient_checkpointing") - 1

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        explicit = "gradient_checkpointing" in kwargs or len(args) > position
        init(self, *args, **kwargs)
        # Not a dataclass field: to_dict(), to_json_string() and asdict() never see it.
        self.__dict__[EXPLICIT_FLAG] = explicit

    setattr(__init__, _WRAPPED_FLAG, True)
    config_class.__init__ = __init__

    # An assignment after construction (args.gradient_checkpointing = True) is a user choice too.
    # Inside __init__ the flag is not set yet, so the dataclass's own field assignments do not count.
    original_setattr = config_class.__setattr__

    def __setattr__(self, name, value):
        original_setattr(self, name, value)
        if name == "gradient_checkpointing" and EXPLICIT_FLAG in self.__dict__:
            self.__dict__[EXPLICIT_FLAG] = True

    config_class.__setattr__ = __setattr__
    return True


def honour_gradient_checkpointing_off(args, model):
    if args is None or model is None or not getattr(args, "gradient_checkpointing", False):
        return False
    try:
        recorded = model._unsloth_gradient_checkpointing
    except Exception:
        return False
    if recorded is not False:
        return False
    explicit = getattr(args, "__dict__", {}).get(EXPLICIT_FLAG, None)
    if explicit is None:
        _note(
            "unknown",
            "Unsloth: the model was set up with use_gradient_checkpointing = False, but the trainer config "
            f"({type(args).__name__}) has gradient_checkpointing = True, so gradient checkpointing stays on. "
            "Pass gradient_checkpointing = False in the config to train without it.",
        )
        return False
    if explicit:
        _note(
            "explicit",
            "Unsloth: the model was set up with use_gradient_checkpointing = False, but the trainer config sets "
            "gradient_checkpointing = True explicitly, so gradient checkpointing stays on.",
        )
        return False
    args.gradient_checkpointing = False
    _note(
        "off",
        "Unsloth: the model was set up with use_gradient_checkpointing = False, so gradient checkpointing stays "
        f"off (TRL's {type(args).__name__} defaults it to True). If you run out of memory, use "
        'use_gradient_checkpointing = "unsloth" (or pass gradient_checkpointing = True in the config).',
    )
    return True
