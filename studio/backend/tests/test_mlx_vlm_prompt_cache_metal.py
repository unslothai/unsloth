# SPDX-License-Identifier: AGPL-3.0-only
import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason = "Requires Metal")

from mlx_vlm.models.cache import ArraysCache, KVCache, RotatingKVCache
from core.inference.mlx_inference import copy_cache_entries


def _bits(value):
    return np.array(value.view(mx.uint8)).tobytes()


@pytest.mark.parametrize("factory", [KVCache, lambda: RotatingKVCache(max_size = 8, keep = 2)])
def test_snapshot_isolated_from_cache_growth_and_rotating_overwrites(factory):
    cache = factory()
    data = mx.arange(16, dtype = mx.float32).reshape(1, 2, 1, 8)
    for i in range(8):
        cache.update_and_fetch(data + i, data + 100 + i)
    mx.eval(cache.state)
    saved = copy_cache_entries([cache])[0]
    original = [_bits(value) for value in saved.state]
    for i in range(12):
        cache.update_and_fetch(data + i, data + 200 + i)
        mx.eval(cache.state)
    assert [_bits(value) for value in saved.state] == original
    advanced = [_bits(value) for value in cache.state]
    for i in range(10):
        saved.update_and_fetch(data + 1000 + i, data + 2000 + i)
        mx.eval(saved.state)
    assert [_bits(value) for value in cache.state] == advanced


def test_recurrent_snapshot_shares_storage_but_isolates_both_state_slots():
    cache = ArraysCache(2)
    cache[0] = mx.arange(1 << 20, dtype = mx.float32).reshape(1024, 1024)
    cache[1] = cache[0] * 2
    mx.eval(cache.state)
    expected = [_bits(value) for value in cache.state]
    before = mx.get_active_memory()
    saved = copy_cache_entries([cache])[0]
    assert mx.get_active_memory() - before < cache[0].nbytes // 10
    for i in (1, 0):
        cache[i][1:, 1::2] += 7
        mx.eval(cache[i])
    assert [_bits(value) for value in saved.state] == expected
    advanced = [_bits(value) for value in cache.state]
    for i in (0, 1):
        saved[i] = saved[i] * 1.25
        saved[i][2:, ::2] -= 3
        mx.eval(saved[i])
    assert [_bits(value) for value in cache.state] == advanced


def test_snapshot_preserves_signed_zero_and_nan_payloads():
    cache = ArraysCache(2)
    bits = mx.array([0x80000000, 0, 0x7FC00123, 0xFFC00456], dtype = mx.uint32)
    cache[0] = bits.view(mx.float32)
    cache[1] = mx.roll(bits, 1).view(mx.float32)
    saved = copy_cache_entries([cache])[0]
    assert [_bits(value) for value in saved.state] == [_bits(value) for value in cache.state]
