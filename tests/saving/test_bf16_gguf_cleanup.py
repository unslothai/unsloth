"""
Tests for the bf16 GGUF file preservation fix during mixed quantization export.

Verifies that when bf16 (or f16) is both the intermediate conversion format
and a user-requested output quantization, the base GGUF file is NOT deleted
during the cleanup step.

Related issue: https://github.com/unslothai/unsloth/issues/4932
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers – simulate the cleanup logic extracted from save_to_gguf()
# ---------------------------------------------------------------------------

def _simulate_cleanup(
    first_conversion: str,
    quantization_method: list,
    base_gguf: str,
    all_saved_locations: list,
    quants_created: bool,
):
    """
    Reproduce the cleanup block from save.py (lines ~1350-1362, post-fix)
    so we can unit-test it without loading a real model.
    """
    if quants_created:
        # --- THIS IS THE FIX under test ---
        if first_conversion not in frozenset(quantization_method):
            all_saved_locations.remove(base_gguf)
            Path(base_gguf).unlink(missing_ok=True)

        all_saved_locations.reverse()

    return all_saved_locations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def gguf_dir():
    """Create a temporary directory with fake GGUF files."""
    tmpdir = tempfile.mkdtemp(prefix="unsloth_test_gguf_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def _create_fake_gguf(directory: str, name: str) -> str:
    """Create a tiny placeholder file and return its path."""
    path = os.path.join(directory, name)
    Path(path).write_text("fake-gguf-data")
    return path


# ---------------------------------------------------------------------------
# Tests – mixed export WITH bf16 in the user's list
# ---------------------------------------------------------------------------

class TestBf16PreservedInMixedExport:
    """
    When the user passes quantization_method=["q4_k_m", "q6_k", "q8_0", "bf16"]
    and first_conversion="bf16", the bf16 GGUF must survive cleanup.
    """

    def test_bf16_file_not_deleted(self, gguf_dir):
        """The bf16 base file should still exist on disk after cleanup."""
        base = _create_fake_gguf(gguf_dir, "model.BF16.gguf")
        q4   = _create_fake_gguf(gguf_dir, "model.Q4_K_M.gguf")
        q6   = _create_fake_gguf(gguf_dir, "model.Q6_K.gguf")
        q8   = _create_fake_gguf(gguf_dir, "model.Q8_0.gguf")

        all_locations = [base, q4, q6, q8]

        result = _simulate_cleanup(
            first_conversion="bf16",
            quantization_method=["q4_k_m", "q6_k", "q8_0", "bf16"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=True,
        )

        assert os.path.exists(base), "BF16 GGUF was deleted despite being requested"
        assert base in result, "BF16 GGUF missing from returned locations"

    def test_all_quant_files_present(self, gguf_dir):
        """All four output files (bf16 + 3 quants) must be in the result."""
        base = _create_fake_gguf(gguf_dir, "model.BF16.gguf")
        q4   = _create_fake_gguf(gguf_dir, "model.Q4_K_M.gguf")
        q6   = _create_fake_gguf(gguf_dir, "model.Q6_K.gguf")
        q8   = _create_fake_gguf(gguf_dir, "model.Q8_0.gguf")

        all_locations = [base, q4, q6, q8]

        result = _simulate_cleanup(
            first_conversion="bf16",
            quantization_method=["q4_k_m", "q6_k", "q8_0", "bf16"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=True,
        )

        assert len(result) == 4
        for f in [base, q4, q6, q8]:
            assert f in result
            assert os.path.exists(f)

    def test_f16_preserved_similarly(self, gguf_dir):
        """Same logic applies when first_conversion='f16' and user requests f16."""
        base = _create_fake_gguf(gguf_dir, "model.F16.gguf")
        q4   = _create_fake_gguf(gguf_dir, "model.Q4_K_M.gguf")

        all_locations = [base, q4]

        result = _simulate_cleanup(
            first_conversion="f16",
            quantization_method=["q4_k_m", "f16"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=True,
        )

        assert os.path.exists(base), "F16 GGUF was deleted despite being requested"
        assert base in result

    def test_f32_preserved_similarly(self, gguf_dir):
        """Same logic applies when first_conversion='f32' and user requests f32."""
        base = _create_fake_gguf(gguf_dir, "model.F32.gguf")
        q8   = _create_fake_gguf(gguf_dir, "model.Q8_0.gguf")

        all_locations = [base, q8]

        result = _simulate_cleanup(
            first_conversion="f32",
            quantization_method=["q8_0", "f32"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=True,
        )

        assert os.path.exists(base), "F32 GGUF was deleted despite being requested"
        assert base in result


# ---------------------------------------------------------------------------
# Tests – mixed export WITHOUT bf16 in the user's list (base should be deleted)
# ---------------------------------------------------------------------------

class TestBaseDeletedWhenNotRequested:
    """
    When the user does NOT include the first_conversion format in their list
    (e.g., quantization_method=["q4_k_m", "q8_0"] with first_conversion="bf16"),
    the intermediate bf16 file should be cleaned up as before.
    """

    def test_bf16_base_deleted(self, gguf_dir):
        """bf16 base file should be deleted when not in quantization_method."""
        base = _create_fake_gguf(gguf_dir, "model.BF16.gguf")
        q4   = _create_fake_gguf(gguf_dir, "model.Q4_K_M.gguf")
        q8   = _create_fake_gguf(gguf_dir, "model.Q8_0.gguf")

        all_locations = [base, q4, q8]

        result = _simulate_cleanup(
            first_conversion="bf16",
            quantization_method=["q4_k_m", "q8_0"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=True,
        )

        assert not os.path.exists(base), "BF16 base should have been deleted"
        assert base not in result
        assert len(result) == 2

    def test_only_quant_files_remain(self, gguf_dir):
        """After cleanup, only the quantized files should remain."""
        base = _create_fake_gguf(gguf_dir, "model.BF16.gguf")
        q4   = _create_fake_gguf(gguf_dir, "model.Q4_K_M.gguf")
        q6   = _create_fake_gguf(gguf_dir, "model.Q6_K.gguf")

        all_locations = [base, q4, q6]

        result = _simulate_cleanup(
            first_conversion="bf16",
            quantization_method=["q4_k_m", "q6_k"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=True,
        )

        for f in [q4, q6]:
            assert os.path.exists(f)
            assert f in result
        assert not os.path.exists(base)

    def test_result_is_reversed(self, gguf_dir):
        """The returned list should be reversed (for [text_model, mmproj] ordering)."""
        base = _create_fake_gguf(gguf_dir, "model.BF16.gguf")
        q4   = _create_fake_gguf(gguf_dir, "model.Q4_K_M.gguf")
        q6   = _create_fake_gguf(gguf_dir, "model.Q6_K.gguf")

        all_locations = [base, q4, q6]

        result = _simulate_cleanup(
            first_conversion="bf16",
            quantization_method=["q4_k_m", "q6_k"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=True,
        )

        # After removing base and reversing [q4, q6] -> [q6, q4]
        assert result == [q6, q4]


# ---------------------------------------------------------------------------
# Tests – single bf16 export (no additional quants)
# ---------------------------------------------------------------------------

class TestSingleBf16Export:
    """
    When quantization_method=["bf16"] only, no additional quants are created
    so quants_created=False and cleanup should be a no-op.
    """

    def test_single_bf16_preserved(self, gguf_dir):
        """bf16-only export: file must survive (quants_created=False)."""
        base = _create_fake_gguf(gguf_dir, "model.BF16.gguf")
        all_locations = [base]

        result = _simulate_cleanup(
            first_conversion="bf16",
            quantization_method=["bf16"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=False,
        )

        assert os.path.exists(base), "Single bf16 export should not be deleted"
        assert base in result
        assert len(result) == 1

    def test_single_f16_preserved(self, gguf_dir):
        """f16-only export: file must survive."""
        base = _create_fake_gguf(gguf_dir, "model.F16.gguf")
        all_locations = [base]

        result = _simulate_cleanup(
            first_conversion="f16",
            quantization_method=["f16"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=False,
        )

        assert os.path.exists(base)
        assert base in result


# ---------------------------------------------------------------------------
# Tests – edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for the cleanup logic."""

    def test_no_quants_created_preserves_base(self, gguf_dir):
        """When quants_created=False, no deletion happens regardless."""
        base = _create_fake_gguf(gguf_dir, "model.BF16.gguf")
        all_locations = [base]

        result = _simulate_cleanup(
            first_conversion="bf16",
            quantization_method=["q4_k_m"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=False,
        )

        assert os.path.exists(base)
        assert base in result

    def test_base_already_missing_on_disk(self, gguf_dir):
        """missing_ok=True: should not crash if base file is already gone."""
        base = os.path.join(gguf_dir, "model.BF16.gguf")  # never created
        q4 = _create_fake_gguf(gguf_dir, "model.Q4_K_M.gguf")
        all_locations = [base, q4]

        # Should not raise even though file doesn't exist
        result = _simulate_cleanup(
            first_conversion="bf16",
            quantization_method=["q4_k_m"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=True,
        )

        assert base not in result
        assert q4 in result

    def test_want_full_precision_flag_when_bf16_requested(self):
        """
        Verify the want_full_precision logic (line ~1369) correctly detects
        that first_conversion is in quantization_method.
        """
        first_conversion = "bf16"
        quantization_method = ["q4_k_m", "bf16"]
        want_full_precision = first_conversion in frozenset(quantization_method)
        assert want_full_precision is True

    def test_want_full_precision_flag_when_bf16_not_requested(self):
        """want_full_precision should be False when bf16 is not requested."""
        first_conversion = "bf16"
        quantization_method = ["q4_k_m", "q8_0"]
        want_full_precision = first_conversion in frozenset(quantization_method)
        assert want_full_precision is False

    def test_mixed_with_bf16_only_extra_quant(self, gguf_dir):
        """bf16 + one additional quant: bf16 preserved, quant added."""
        base = _create_fake_gguf(gguf_dir, "model.BF16.gguf")
        q4   = _create_fake_gguf(gguf_dir, "model.Q4_K_M.gguf")

        all_locations = [base, q4]

        result = _simulate_cleanup(
            first_conversion="bf16",
            quantization_method=["q4_k_m", "bf16"],
            base_gguf=base,
            all_saved_locations=all_locations,
            quants_created=True,
        )

        assert os.path.exists(base)
        assert len(result) == 2
        assert base in result
        assert q4 in result
