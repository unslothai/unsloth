# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto keeps every layer on the GPU down to llama.cpp's own fit floor before handing placement to --fit on.

A Qwen3.8-27B-shaped load a few hundred MiB short at an 8192 context used to go to --fit on at 8192 with 4 slots
and MTP off, which moves layers to the CPU (2-7x slower decode on a dense target). At 4096 the same card holds
every layer and the MTP head on one slot.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import pytest  # noqa: E402

import core.inference.llama_cpp as llama_cpp  # noqa: E402
from test_slot_reduction_context_refit import _plan  # noqa: E402


class TestAutoTriesTheFitFloorBeforeOffload:
    @pytest.mark.parametrize("weights_mib", [10_400, 10_500])
    def test_mtp_load_short_at_8192_stays_on_gpu_with_its_drafter(self, tmp_path, weights_mib):
        got = _plan(tmp_path, weights_mib = weights_mib, n_parallel = 4, spec = "auto")
        assert (got["fit"], got["slots"], got["spec"]) == ("off", 1, "draft-mtp")
        assert llama_cpp._LLAMA_FIT_MIN_CTX <= got["ctx"] < llama_cpp._FIT_MIN_CTX

    @pytest.mark.parametrize("weights_mib", [10_900, 11_000, 11_100])
    def test_spec_off_load_short_at_8192_stays_on_gpu(self, tmp_path, weights_mib):
        got = _plan(tmp_path, weights_mib = weights_mib, n_parallel = 4, spec = "off")
        assert (got["fit"], got["slots"]) == ("off", 1)
        assert llama_cpp._LLAMA_FIT_MIN_CTX <= got["ctx"] < llama_cpp._FIT_MIN_CTX

    def test_a_load_that_holds_8192_is_unchanged(self, tmp_path):
        got = _plan(tmp_path, weights_mib = 10_300, n_parallel = 4, spec = "auto")
        assert (got["fit"], got["slots"], got["spec"], got["ctx"]) == ("off", 1, "draft-mtp", 8_448)

    @pytest.mark.parametrize("spec", ["auto", "off"])
    def test_a_load_short_even_at_4096_still_offloads(self, tmp_path, spec):
        got = _plan(tmp_path, weights_mib = 11_400, n_parallel = 4, spec = spec)
        assert (got["fit"], got["slots"], got["ctx"]) == ("on", 4, llama_cpp._AUTO_OFFLOAD_CTX)
