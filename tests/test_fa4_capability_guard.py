"""Unit test for the FA4 capability guard in FlexInference.__init__.

Runs on any GPU (and on CPU) because we monkey-patch
`torch.cuda.get_device_capability` and stub out the page-table / model
patching that the constructor does after the guard.
"""

import os
import sys
import types
import warnings
import unittest
from unittest import mock

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BENCH_DIR = os.path.join(REPO_ROOT, "scripts", "benchmarks")
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

# qwen3_flex_inference imports heavy siblings (flex_paged_attention).
# Stub the PageTable and patch_qwen3_model the constructor calls after the
# guard so we don't need a real model / CUDA device.
import qwen3_flex_inference as qfi  # noqa: E402


class _FakePageTable:
    def __init__(self, *a, **kw):
        pass

    def create_causal_blockmask(self, *a, **kw):
        return None


class _FakeTokenizer:
    eos_token_id = 0


def _make_fake_model(device_str = "cpu"):
    m = types.SimpleNamespace()
    m.device = torch.device(device_str)
    return m


def _build(fa4_prefill, cc_major, cc_minor = 0):
    """Construct a FlexInference with the guard exercised.

    Returns the instance. Patches torch.cuda.get_device_capability,
    torch.zeros (to avoid CUDA allocation), PageTable, and
    patch_qwen3_model so __init__ can run to completion without a real
    model.
    """
    fake_model = _make_fake_model("cpu")
    fake_tok = _FakeTokenizer()

    _real_zeros = torch.zeros

    def _fake_zeros(*a, **kw):
        kw.pop("device", None)
        return _real_zeros(*a, **kw)

    with (
        mock.patch.object(
            torch.cuda, "get_device_capability", return_value = (cc_major, cc_minor)
        ),
        mock.patch.object(qfi, "PageTable", _FakePageTable),
        mock.patch.object(qfi, "patch_qwen3_model", lambda *a, **kw: None),
        mock.patch.object(torch, "zeros", _fake_zeros),
    ):
        return qfi.FlexInference(
            model = fake_model,
            tokenizer = fake_tok,
            max_batch_size = 2,
            max_seq_length = 128,
            n_pages = 4,
            page_size = 128,
            max_new_tokens = 16,
            fa4_prefill = fa4_prefill,
        )


def _fa4_warnings(caught):
    return [
        w
        for w in caught
        if issubclass(w.category, RuntimeWarning) and "fa4_prefill" in str(w.message)
    ]


class TestFA4CapabilityGuard(unittest.TestCase):
    # --- explicit opt-in: --fa4_prefill=True ---
    def test_explicit_on_sub_hopper_disables_and_warns(self):
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            fi = _build(fa4_prefill = True, cc_major = 8)
        self.assertTrue(
            _fa4_warnings(caught),
            f"expected RuntimeWarning about fa4_prefill, got {caught!r}",
        )
        self.assertIs(fi.fa4_prefill, False)
        self.assertEqual(fi.prefill_q_block, 128)
        self.assertNotIn("BACKEND", fi.prefill_kernel_options)

    def _assert_fa4_enabled(self, cc_major, fa4_prefill):
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            fi = _build(fa4_prefill = fa4_prefill, cc_major = cc_major)
        self.assertEqual(
            _fa4_warnings(caught),
            [],
            f"unexpected fa4 RuntimeWarning on sm_{cc_major}0 "
            f"with fa4_prefill={fa4_prefill}: {caught!r}",
        )
        self.assertIs(fi.fa4_prefill, True)
        self.assertEqual(fi.prefill_q_block, 256)
        self.assertEqual(fi.prefill_kernel_options.get("BACKEND"), "FLASH")

    def test_explicit_on_hopper_enables(self):
        self._assert_fa4_enabled(cc_major = 9, fa4_prefill = True)

    def test_explicit_on_blackwell_sm100_enables(self):
        self._assert_fa4_enabled(cc_major = 10, fa4_prefill = True)

    def test_explicit_on_blackwell_sm120_enables(self):
        self._assert_fa4_enabled(cc_major = 12, fa4_prefill = True)

    # --- auto-detect: fa4_prefill is None ---
    def test_auto_on_sub_hopper_disables_silently(self):
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            fi = _build(fa4_prefill = None, cc_major = 8)
        self.assertEqual(
            _fa4_warnings(caught),
            [],
            f"auto-detect must not warn on unsupported GPU: {caught!r}",
        )
        self.assertIs(fi.fa4_prefill, False)
        self.assertEqual(fi.prefill_q_block, 128)
        self.assertNotIn("BACKEND", fi.prefill_kernel_options)

    def test_auto_on_hopper_enables(self):
        self._assert_fa4_enabled(cc_major = 9, fa4_prefill = None)

    def test_auto_on_blackwell_sm100_enables(self):
        self._assert_fa4_enabled(cc_major = 10, fa4_prefill = None)

    def test_auto_on_blackwell_sm120_enables(self):
        self._assert_fa4_enabled(cc_major = 12, fa4_prefill = None)

    # --- explicit opt-out: --no-fa4_prefill / fa4_prefill=False ---
    def test_explicit_off_on_blackwell_stays_off(self):
        with warnings.catch_warnings(record = True) as caught:
            warnings.simplefilter("always")
            fi = _build(fa4_prefill = False, cc_major = 10)
        self.assertEqual(_fa4_warnings(caught), [])
        self.assertIs(fi.fa4_prefill, False)
        self.assertEqual(fi.prefill_q_block, 128)
        self.assertNotIn("BACKEND", fi.prefill_kernel_options)


if __name__ == "__main__":
    unittest.main()
