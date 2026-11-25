# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The KV cache keyword across transformers releases.

transformers 5.24 renamed ``past_key_values`` (SSM mixers: ``cache_params``) to ``cache`` in every
model forward, ``prepare_inputs_for_generation``, ``generate``'s ``model_kwargs`` and the model outputs;
the old names stay deprecated aliases until 5.40. Unsloth code that reads the cache by the old name
from ``**kwargs`` sees None on 5.24 and silently runs without it. Plain dict helpers, no torch import.
"""

__all__ = ["CACHE_KWARG_NAMES", "get_cache_kwarg", "output_cache", "pop_cache_kwarg"]

# Newest first: a caller on 5.24 passes `cache`, older code and users still pass past_key_values.
CACHE_KWARG_NAMES = ("cache", "past_key_values")


def get_cache_kwarg(kwargs, default = None):
    """The cache in ``kwargs`` under ``cache`` or ``past_key_values`` (first non-None), else ``default``."""
    for name in CACHE_KWARG_NAMES:
        value = kwargs.get(name)
        if value is not None:
            return value
    return default


def pop_cache_kwarg(kwargs, default = None):
    """Like ``get_cache_kwarg`` but removes every cache alias from ``kwargs``."""
    value = get_cache_kwarg(kwargs, default)
    for name in CACHE_KWARG_NAMES:
        kwargs.pop(name, None)
    return value


def output_cache(output):
    """The cache a model / generate output carries: field ``cache`` from 5.24, ``past_key_values`` before.

    Reads ``cache`` first: on 5.24 the ``past_key_values`` property logs a deprecation warning.
    """
    if output is None:
        return None
    value = getattr(output, "cache", None)
    if value is None:
        value = getattr(output, "past_key_values", None)
    return value
