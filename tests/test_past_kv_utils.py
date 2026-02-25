"""
Unit tests for past_key_values utilities.
Self-contained — does NOT import unsloth, so runs without a GPU.

Run with:
    python -m pytest tests/test_past_kv_utils.py -v
"""
import unittest
import torch
from transformers.cache_utils import DynamicCache, Cache


# ── Inline copies of the functions under test ──────────────────────────
# These match the implementations in unsloth/models/llama.py exactly.
# Kept inline so the test suite can run on any machine (no GPU needed).

def _ensure_cache_is_dynamic(past_key_values):
    """Convert list/tuple of (K, V) pairs to DynamicCache for transformers v5 compat."""
    if past_key_values is None:
        return None
    if isinstance(past_key_values, Cache):
        return past_key_values
    if isinstance(past_key_values, (tuple, list)) and len(past_key_values) > 0:
        cache = DynamicCache()
        for layer_idx, layer_kv in enumerate(past_key_values):
            cache.update(layer_kv[0], layer_kv[1], layer_idx)
        return cache
    return past_key_values


def _slice_position_ids(position_ids, input_ids):
    """Slice position_ids to match input_ids length if needed."""
    if position_ids is None:
        return None
    if position_ids.dim() == 2:
        if position_ids.shape[1] > input_ids.shape[1]:
            position_ids = position_ids[:, -input_ids.shape[1]:]
    elif position_ids.dim() == 1:
        if position_ids.shape[0] > input_ids.shape[1]:
            position_ids = position_ids[-input_ids.shape[1]:]
    return position_ids


# ── Tests ──────────────────────────────────────────────────────────────

class TestEnsureCacheIsDynamic(unittest.TestCase):
    """Tests for _ensure_cache_is_dynamic conversion utility."""

    def test_none_passthrough(self):
        self.assertIsNone(_ensure_cache_is_dynamic(None))

    def test_dynamic_cache_passthrough(self):
        cache = DynamicCache()
        k = torch.randn(1, 4, 8, 16)
        v = torch.randn(1, 4, 8, 16)
        cache.update(k, v, 0)
        result = _ensure_cache_is_dynamic(cache)
        self.assertIs(result, cache)

    def test_tuple_conversion(self):
        """Tuple of (K, V) pairs should be converted to DynamicCache."""
        n_layers = 3
        layers = []
        for _ in range(n_layers):
            k = torch.randn(1, 4, 8, 16)
            v = torch.randn(1, 4, 8, 16)
            layers.append((k, v))
        past_kv = tuple(layers)

        result = _ensure_cache_is_dynamic(past_kv)
        self.assertIsInstance(result, DynamicCache)
        for i in range(n_layers):
            cached_k, cached_v = result[i]
            self.assertTrue(torch.equal(cached_k, layers[i][0]))
            self.assertTrue(torch.equal(cached_v, layers[i][1]))

    def test_list_conversion(self):
        """List of (K, V) pairs should be converted to DynamicCache."""
        layers = [(torch.randn(1, 4, 8, 16), torch.randn(1, 4, 8, 16))]
        result = _ensure_cache_is_dynamic(layers)
        self.assertIsInstance(result, DynamicCache)
        cached_k, cached_v = result[0]
        self.assertTrue(torch.equal(cached_k, layers[0][0]))

    def test_empty_tuple_passthrough(self):
        result = _ensure_cache_is_dynamic(())
        self.assertEqual(result, ())

    def test_empty_list_passthrough(self):
        result = _ensure_cache_is_dynamic([])
        self.assertEqual(result, [])

    def test_seq_length_preserved(self):
        """Verify DynamicCache reports correct sequence length after conversion."""
        seq_len = 42
        layers = [(torch.randn(1, 4, seq_len, 16), torch.randn(1, 4, seq_len, 16))]
        result = _ensure_cache_is_dynamic(tuple(layers))
        self.assertEqual(result.get_seq_length(), seq_len)


class TestSlicePositionIds(unittest.TestCase):
    """Tests for _slice_position_ids utility."""

    def test_none_passthrough(self):
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        self.assertIsNone(_slice_position_ids(None, input_ids))

    def test_2d_no_slice_needed(self):
        input_ids = torch.zeros(1, 10, dtype=torch.long)
        position_ids = torch.arange(10).unsqueeze(0)
        result = _slice_position_ids(position_ids, input_ids)
        self.assertTrue(torch.equal(result, position_ids))

    def test_2d_slice_needed(self):
        """position_ids longer than input_ids — should take last N."""
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        position_ids = torch.arange(10).unsqueeze(0)  # shape (1, 10)
        result = _slice_position_ids(position_ids, input_ids)
        self.assertEqual(result.shape, (1, 3))
        expected = torch.tensor([[7, 8, 9]])
        self.assertTrue(torch.equal(result, expected))

    def test_1d_no_slice_needed(self):
        input_ids = torch.zeros(1, 5, dtype=torch.long)
        position_ids = torch.arange(5)
        result = _slice_position_ids(position_ids, input_ids)
        self.assertTrue(torch.equal(result, position_ids))

    def test_1d_slice_needed(self):
        input_ids = torch.zeros(1, 3, dtype=torch.long)
        position_ids = torch.arange(10)  # shape (10,)
        result = _slice_position_ids(position_ids, input_ids)
        self.assertEqual(result.shape, (3,))
        expected = torch.tensor([7, 8, 9])
        self.assertTrue(torch.equal(result, expected))

    def test_shorter_position_ids_passthrough(self):
        """position_ids shorter than input_ids — should pass through unchanged."""
        input_ids = torch.zeros(1, 10, dtype=torch.long)
        position_ids = torch.arange(5).unsqueeze(0)
        result = _slice_position_ids(position_ids, input_ids)
        self.assertTrue(torch.equal(result, position_ids))

    def test_exact_match(self):
        """Exact same length — no slicing."""
        input_ids = torch.zeros(2, 7, dtype=torch.long)
        position_ids = torch.arange(7).unsqueeze(0).expand(2, -1)
        result = _slice_position_ids(position_ids, input_ids)
        self.assertEqual(result.shape, (2, 7))


if __name__ == "__main__":
    unittest.main()
