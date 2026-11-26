# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""KV cache keyword across transformers releases: 5.24 renamed past_key_values to cache (alias until 5.40)."""

__all__ = ["CACHE_KWARG_NAMES", "get_cache_kwarg", "output_cache", "pop_cache_kwarg"]

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
    """Output cache; reads ``cache`` first since 5.24's ``past_key_values`` property warns."""
    if output is None:
        return None
    value = getattr(output, "cache", None)
    if value is None:
        value = getattr(output, "past_key_values", None)
    return value
