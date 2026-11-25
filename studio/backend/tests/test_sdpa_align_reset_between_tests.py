# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The SDPA alignment wrapper one test installs does not leak into the next test on the same xdist worker.

The DiT trainer's compile step (``_maybe_compile_transformer``) installs it and only the full run's ``finally``
takes it out, so a test that called the compile step alone left inductor's SDPA constraint wrapped and
``|unsloth_sdpa_align=unguarded`` in torch's ``cache_key_tag``; test_diffusion_cache_tag_contexts.py then failed
on the next test of that worker. The two tests below run in order on one worker: the first leaves the wrapper
installed, the second fails unless conftest took it out in between."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
import core.training.diffusion_sdpa_align as sa  # noqa: E402

pytestmark = pytest.mark.xdist_group("sdpa_align_reset_between_tests")

_SEEN: dict = {}


def test_a_test_leaves_the_wrapper_installed(monkeypatch):
    monkeypatch.delenv(sa.SDPA_ALIGN_GUARD_ENV, raising = False)
    if not sa.internals_supported():
        pytest.skip("this torch lacks the inductor internals the wrapper needs")
    assert not sa.installed(), "a wrapper was already installed when this test started"
    assert sa.install_sdpa_align_without_guard() > 0
    _SEEN["tag"] = sa._cache_config().cache_key_tag
    assert sa.CACHE_TAG in _SEEN["tag"]


def test_the_next_test_starts_without_it():
    if not _SEEN:
        pytest.skip(
            "test_a_test_leaves_the_wrapper_installed did not install the wrapper on this worker"
        )
    assert not sa.installed()
    assert sa.CACHE_TAG not in (sa._cache_config().cache_key_tag or "")
