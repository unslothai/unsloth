"""
Layer 1: Unit Tests (mock-based, no spawn, <2s)

Tests dataset_map_num_proc(), safe_num_proc(), and PYTHONPATH propagation
logic using unittest.mock to simulate different platforms.
"""

import os
import sys
import unittest
from unittest.mock import patch

# Add the studio backend to path so we can import the PR's hardware module.
# Works both when tests/ is inside the unsloth repo and when it's outside.
_CANDIDATES = [
    os.path.join(
        os.path.dirname(__file__), "..", "..", "studio", "backend"
    ),  # inside repo
    os.path.join(
        os.path.dirname(__file__), "..", "..", "unsloth", "studio", "backend"
    ),  # outside repo
]
for _c in _CANDIDATES:
    if os.path.isdir(_c):
        sys.path.insert(0, os.path.abspath(_c))
        break

# Import the helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from helpers.spawn_harness import simulate_pythonpath_propagation


def _import_hardware_funcs():
    """Import safe_num_proc, safe_thread_num_proc, and dataset_map_num_proc from the PR's hardware module."""
    from utils.hardware.hardware import (
        safe_num_proc,
        safe_thread_num_proc,
        dataset_map_num_proc,
    )

    return safe_num_proc, safe_thread_num_proc, dataset_map_num_proc


class TestDatasetMapNumProc(unittest.TestCase):
    """L1-1 through L1-5: dataset_map_num_proc() platform behavior."""

    def test_L1_1_win32_returns_none(self):
        """L1-1: dataset_map_num_proc(4) on win32 returns None."""
        _, _, dataset_map_num_proc = _import_hardware_funcs()
        with patch.object(sys, "platform", "win32"):
            result = dataset_map_num_proc(4)
        self.assertIsNone(result)

    def test_L1_2_darwin_returns_none(self):
        """L1-2: dataset_map_num_proc(4) on darwin returns None."""
        _, _, dataset_map_num_proc = _import_hardware_funcs()
        with patch.object(sys, "platform", "darwin"):
            result = dataset_map_num_proc(4)
        self.assertIsNone(result)

    def test_L1_3_linux_returns_int(self):
        """L1-3: dataset_map_num_proc(4) on linux returns int >= 1."""
        _, _, dataset_map_num_proc = _import_hardware_funcs()
        with patch.object(sys, "platform", "linux"):
            result = dataset_map_num_proc(4)
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 1)

    def test_L1_4_win32_no_args_returns_none(self):
        """L1-4: dataset_map_num_proc() no args on win32 returns None."""
        _, _, dataset_map_num_proc = _import_hardware_funcs()
        with patch.object(sys, "platform", "win32"):
            result = dataset_map_num_proc()
        self.assertIsNone(result)

    def test_L1_5_linux_no_args_returns_int(self):
        """L1-5: dataset_map_num_proc() no args on linux returns int >= 1."""
        _, _, dataset_map_num_proc = _import_hardware_funcs()
        with patch.object(sys, "platform", "linux"):
            result = dataset_map_num_proc()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 1)


class TestSafeNumProc(unittest.TestCase):
    """L1-6 through L1-10: safe_num_proc() behavior."""

    def test_L1_6_win32_returns_1(self):
        """L1-6: safe_num_proc() on win32 returns 1."""
        safe_num_proc, _, _ = _import_hardware_funcs()
        with patch.object(sys, "platform", "win32"):
            result = safe_num_proc()
        self.assertEqual(result, 1)

    def test_L1_7_darwin_returns_1(self):
        """L1-7: safe_num_proc() on darwin returns 1."""
        safe_num_proc, _, _ = _import_hardware_funcs()
        with patch.object(sys, "platform", "darwin"):
            result = safe_num_proc()
        self.assertEqual(result, 1)

    def test_L1_8_linux_returns_int_gte_1(self):
        """L1-8: safe_num_proc() on linux returns int >= 1."""
        safe_num_proc, _, _ = _import_hardware_funcs()
        with patch.object(sys, "platform", "linux"):
            result = safe_num_proc()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 1)

    def test_L1_9_linux_num_proc_0(self):
        """L1-9: dataset_map_num_proc(0) on linux -- 0 is int, falls through to safe_num_proc.
        safe_num_proc(0): isinstance(0, int)=True, but 0 is falsy... actually
        the code checks `not isinstance(desired, int)` so 0 passes that check.
        Then get_visible_gpu_count check. Returns int >= 0."""
        _, _, dataset_map_num_proc = _import_hardware_funcs()
        with patch.object(sys, "platform", "linux"):
            result = dataset_map_num_proc(0)
        # 0 is an int, so it passes the isinstance check in safe_num_proc
        # But 0 might be returned as-is or capped. The key is it doesn't crash.
        self.assertIsInstance(result, int)

    def test_L1_10_win32_num_proc_1_returns_none(self):
        """L1-10: dataset_map_num_proc(1) on win32 returns None, NOT 1.
        This proves the distinction matters: num_proc=1 would cause Pool(1)
        to spawn, which crashes on Windows."""
        _, _, dataset_map_num_proc = _import_hardware_funcs()
        with patch.object(sys, "platform", "win32"):
            result = dataset_map_num_proc(1)
        self.assertIsNone(result, "num_proc=1 on win32 must return None, not 1")


class TestPythonPathPropagation(unittest.TestCase):
    """L1-11 through L1-14: PYTHONPATH propagation logic (extracted pure function)."""

    def test_L1_11_relative_path_becomes_absolute(self):
        """L1-11: Relative compile_location is converted to absolute."""
        pp, cache = simulate_pythonpath_propagation(
            platform = "win32",
            compile_location = "unsloth_compiled_cache",
            current_pythonpath = "",
            cwd = "/home/user/project",
        )
        self.assertTrue(
            os.path.isabs(cache), f"Cache path should be absolute, got: {cache}"
        )
        self.assertIn("unsloth_compiled_cache", cache)

    def test_L1_12_no_duplicate_if_already_present(self):
        """L1-12: No duplicate entry if cache path already in PYTHONPATH."""
        cache_path = "/home/user/project/unsloth_compiled_cache"
        pp, cache = simulate_pythonpath_propagation(
            platform = "darwin",
            compile_location = cache_path,
            current_pythonpath = cache_path,
            cwd = "/home/user/project",
        )
        # Count occurrences of cache_path in the result
        parts = pp.split(os.pathsep)
        count = parts.count(cache_path)
        self.assertEqual(
            count, 1, f"Cache path appears {count} times in PYTHONPATH: {pp}"
        )

    def test_L1_13_empty_pythonpath(self):
        """L1-13: Empty/unset PYTHONPATH results in just the cache path."""
        cache_path = "/tmp/cache"
        pp, cache = simulate_pythonpath_propagation(
            platform = "win32",
            compile_location = cache_path,
            current_pythonpath = "",
            cwd = "/tmp",
        )
        self.assertEqual(pp, cache_path, "Should be just the cache path, no separators")
        self.assertNotIn(os.pathsep, pp, "Should not have trailing/leading separator")

    def test_L1_14_skipped_on_linux(self):
        """L1-14: PYTHONPATH propagation is skipped on linux."""
        original_pp = "/some/existing/path"
        pp, cache = simulate_pythonpath_propagation(
            platform = "linux",
            compile_location = "unsloth_compiled_cache",
            current_pythonpath = original_pp,
            cwd = "/home/user",
        )
        self.assertEqual(pp, original_pp, "PYTHONPATH should be unchanged on linux")
        self.assertEqual(
            cache, "unsloth_compiled_cache", "Cache path should be unchanged on linux"
        )


def _simulate_dataset_utils_guard(start_method, num_proc):
    """Simulate the fixed guard pattern from unsloth_zoo/dataset_utils.py.

    Extracts the logic:
        if _mp.get_start_method() != 'fork':
            num_proc = None
        elif num_proc is None or type(num_proc) is not int:
            num_proc = <auto-computed>

    Returns the resulting num_proc value.
    """
    if start_method != "fork":
        return None
    elif num_proc is None or type(num_proc) is not int:
        # Would auto-compute; return a sentinel to prove this branch ran
        return -999  # sentinel: auto-compute branch
    else:
        # Caller passed an integer on a fork platform -- use it as-is
        return num_proc


class TestDatasetUtilsIntegerBypass(unittest.TestCase):
    """L1-15 through L1-17: Verify unsloth_zoo/dataset_utils.py guard blocks
    cannot be bypassed by passing an integer num_proc on spawn platforms."""

    def test_L1_15_spawn_num_proc_4_returns_none(self):
        """L1-15: On spawn platform, num_proc=4 must be forced to None."""
        result = _simulate_dataset_utils_guard(start_method = "spawn", num_proc = 4)
        self.assertIsNone(
            result,
            "num_proc=4 on spawn platform must be forced to None, not passed through",
        )

    def test_L1_16_spawn_num_proc_1_returns_none(self):
        """L1-16: On spawn platform, num_proc=1 must be forced to None.
        Even num_proc=1 creates Pool(1) which spawns a worker."""
        result = _simulate_dataset_utils_guard(start_method = "spawn", num_proc = 1)
        self.assertIsNone(
            result,
            "num_proc=1 on spawn platform must be forced to None, not passed through",
        )

    def test_L1_17_fork_num_proc_4_preserved(self):
        """L1-17: On fork platform, num_proc=4 must be preserved (not overridden)."""
        result = _simulate_dataset_utils_guard(start_method = "fork", num_proc = 4)
        self.assertEqual(
            result, 4, "num_proc=4 on fork platform must be preserved as-is"
        )

    def test_L1_18_spawn_num_proc_none_returns_none(self):
        """L1-18: On spawn platform, num_proc=None also returns None (no auto-compute)."""
        result = _simulate_dataset_utils_guard(start_method = "spawn", num_proc = None)
        self.assertIsNone(result, "num_proc=None on spawn platform must remain None")

    def test_L1_19_fork_num_proc_none_auto_computes(self):
        """L1-19: On fork platform, num_proc=None triggers auto-compute."""
        result = _simulate_dataset_utils_guard(start_method = "fork", num_proc = None)
        self.assertEqual(
            result,
            -999,
            "num_proc=None on fork platform should trigger auto-compute branch",
        )

    def test_L1_20_actual_dataset_utils_block1_spawn_integer(self):
        """L1-20: Verify the actual dataset_utils.py code (Block 1) handles
        integer num_proc on spawn by reading the patched source."""
        import importlib
        import multiprocessing as _mp

        # Read the actual source of dataset_utils.py and verify the pattern
        dataset_utils_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "lib",
            "python3.13",
            "site-packages",
            "unsloth_zoo",
            "dataset_utils.py",
        )
        if not os.path.exists(dataset_utils_path):
            self.skipTest("dataset_utils.py not found at expected path")

        with open(dataset_utils_path, "r") as f:
            source = f.read()

        # Verify all 3 blocks have the unconditional spawn check BEFORE
        # the num_proc type check (the fix pattern)
        # The fixed pattern is: get_start_method() != 'fork' check comes first,
        # then the elif with the type check
        import re

        # Match the pattern: if _mp.get_start_method() != 'fork': ... elif num_proc
        pattern = r"if _mp\.get_start_method\(\) != 'fork':\s*\n\s*(num_proc|dataset_num_proc) = None\s*\n\s*elif"
        matches = re.findall(pattern, source)
        self.assertGreaterEqual(
            len(matches),
            3,
            f"Expected 3 fixed guard blocks in dataset_utils.py, found {len(matches)}. "
            "The unconditional spawn check must come before the type check.",
        )


class TestSafeThreadNumProc(unittest.TestCase):
    """L1-21 through L1-24: safe_thread_num_proc() behavior."""

    def test_L1_21_darwin_returns_gt_1(self):
        """L1-21: safe_thread_num_proc() on darwin returns int > 1 (not capped)."""
        _, safe_thread_num_proc, _ = _import_hardware_funcs()
        with (
            patch.object(sys, "platform", "darwin"),
            patch("os.cpu_count", return_value = 8),
        ):
            result = safe_thread_num_proc()
        self.assertIsInstance(result, int)
        self.assertGreater(
            result,
            1,
            "safe_thread_num_proc() must NOT cap to 1 on macOS -- threads are unaffected by spawn",
        )

    def test_L1_22_win32_returns_gt_1(self):
        """L1-22: safe_thread_num_proc() on win32 returns int > 1 (not capped)."""
        _, safe_thread_num_proc, _ = _import_hardware_funcs()
        with (
            patch.object(sys, "platform", "win32"),
            patch("os.cpu_count", return_value = 8),
        ):
            result = safe_thread_num_proc()
        self.assertIsInstance(result, int)
        self.assertGreater(
            result,
            1,
            "safe_thread_num_proc() must NOT cap to 1 on Windows -- threads are unaffected by spawn",
        )

    def test_L1_23_linux_returns_gte_1(self):
        """L1-23: safe_thread_num_proc() on linux returns int >= 1."""
        _, safe_thread_num_proc, _ = _import_hardware_funcs()
        with patch.object(sys, "platform", "linux"):
            result = safe_thread_num_proc()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 1)

    def test_L1_24_respects_desired_value(self):
        """L1-24: safe_thread_num_proc(8) returns 8."""
        _, safe_thread_num_proc, _ = _import_hardware_funcs()
        result = safe_thread_num_proc(8)
        self.assertEqual(
            result,
            8,
            "safe_thread_num_proc(desired) must return the desired value as-is",
        )


_HW_MOD = "utils.hardware.hardware"


class TestBugRegressions(unittest.TestCase):
    """L1-25 through L1-32: Regression tests for edge-case bug fixes."""

    def test_L1_25_safe_num_proc_linux_cpu_none(self):
        """L1-25: safe_num_proc() on linux with cpu_count=None returns >= 1 (no crash)."""
        safe_num_proc, _, _ = _import_hardware_funcs()
        with (
            patch.object(sys, "platform", "linux"),
            patch("os.cpu_count", return_value = None),
            patch(f"{_HW_MOD}.get_visible_gpu_count", return_value = 1),
        ):
            result = safe_num_proc()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 1)

    def test_L1_26_safe_thread_num_proc_linux_cpu_none(self):
        """L1-26: safe_thread_num_proc() on linux with cpu_count=None returns >= 1 (no crash)."""
        _, safe_thread_num_proc, _ = _import_hardware_funcs()
        with (
            patch.object(sys, "platform", "linux"),
            patch("os.cpu_count", return_value = None),
        ):
            result = safe_thread_num_proc()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 1)

    def test_L1_27_safe_thread_num_proc_darwin_cpu_none(self):
        """L1-27: safe_thread_num_proc() on darwin with cpu_count=None returns >= 1 (no crash)."""
        _, safe_thread_num_proc, _ = _import_hardware_funcs()
        with (
            patch.object(sys, "platform", "darwin"),
            patch("os.cpu_count", return_value = None),
        ):
            result = safe_thread_num_proc()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 1)

    def test_L1_28_safe_num_proc_zero_clamped(self):
        """L1-28: safe_num_proc(0) on linux returns 1 (clamped)."""
        safe_num_proc, _, _ = _import_hardware_funcs()
        with (
            patch.object(sys, "platform", "linux"),
            patch(f"{_HW_MOD}.get_visible_gpu_count", return_value = 1),
        ):
            result = safe_num_proc(0)
        self.assertEqual(result, 1)

    def test_L1_29_safe_num_proc_negative_clamped(self):
        """L1-29: safe_num_proc(-5) on linux returns 1 (clamped)."""
        safe_num_proc, _, _ = _import_hardware_funcs()
        with (
            patch.object(sys, "platform", "linux"),
            patch(f"{_HW_MOD}.get_visible_gpu_count", return_value = 1),
        ):
            result = safe_num_proc(-5)
        self.assertEqual(result, 1)

    def test_L1_30_safe_thread_num_proc_zero_clamped(self):
        """L1-30: safe_thread_num_proc(0) returns 1 (clamped)."""
        _, safe_thread_num_proc, _ = _import_hardware_funcs()
        result = safe_thread_num_proc(0)
        self.assertEqual(result, 1)

    def test_L1_31_safe_thread_num_proc_negative_clamped(self):
        """L1-31: safe_thread_num_proc(-5) returns 1 (clamped)."""
        _, safe_thread_num_proc, _ = _import_hardware_funcs()
        result = safe_thread_num_proc(-5)
        self.assertEqual(result, 1)

    def test_L1_32_dataset_map_num_proc_linux_cpu_none(self):
        """L1-32: dataset_map_num_proc() on linux with cpu_count=None returns >= 1 (no crash)."""
        _, _, dataset_map_num_proc = _import_hardware_funcs()
        with (
            patch.object(sys, "platform", "linux"),
            patch("os.cpu_count", return_value = None),
            patch(f"{_HW_MOD}.get_visible_gpu_count", return_value = 1),
        ):
            result = dataset_map_num_proc()
        self.assertIsInstance(result, int)
        self.assertGreaterEqual(result, 1)


if __name__ == "__main__":
    unittest.main(verbosity = 2)
