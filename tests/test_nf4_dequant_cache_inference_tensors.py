# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""NF4 dequant cache with inference tensors (built under torch.inference_mode, no version counter).

One inference tensor among the scales used to drop the version of every scale from the cache key, so an in place
write to an ordinary scale served a stale dense weight; an inference packed weight raised on admission. Runs on
CPU with real bitsandbytes NF4 states and a faked memory reading (fixtures from test_nf4_dequant_cache_scales)."""

import pytest
import torch

from tests.test_nf4_dequant_cache_scales import _fresh, _state, bnb_functional, dc  # noqa: F401


def test_imports_this_tree():
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(dc.__file__).startswith(here), dc.__file__


@pytest.mark.parametrize("inference", ["offset", "absmax"])
def test_other_scale_written_in_place_with_an_inference_scale(inference):
    W, qs = _state(True)
    with torch.inference_mode():
        if inference == "offset":
            qs.offset = qs.offset.clone()
        else:
            qs.absmax = qs.absmax.clone()
    first = dc.dequantize(W, qs)
    assert dc.STATS.get("admits", 0) == 1
    qs.state2.absmax.mul_(2)  # an ordinary (versioned) tensor, written outside inference mode
    got = dc.dequantize(W, qs)
    fresh = bnb_functional.dequantize_4bit(W, qs)
    assert not torch.equal(first, fresh)
    assert torch.equal(got, fresh)


def test_inference_packed_weight_is_decoded_not_cached():
    W, qs = _state(False)
    with torch.inference_mode():
        W = W.clone()
    got = dc.dequantize(W, qs)
    assert torch.equal(got, bnb_functional.dequantize_4bit(W, qs))
    assert dc.STATS.get("admits", 0) == 0
