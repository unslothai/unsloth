"""
Sandbox Edge-Case Tests for PR #4507 Hardware Functions

Runs the FIXED function bodies inside an isolated uv venv via subprocess
to prove they work in a clean environment with no dependencies.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# The fixed function bodies inlined as a Python string.
# Includes a stub get_visible_gpu_count that always returns 1 (single GPU).
_FIXED_FUNCTIONS = r'''
import os
import sys

def get_visible_gpu_count():
    """Stub: always returns 1 (single GPU)."""
    return 1

def safe_num_proc(desired=None):
    if sys.platform in ("win32", "darwin"):
        return 1
    if desired is None or not isinstance(desired, int):
        desired = max(1, (os.cpu_count() or 1) // 3)
    visible = get_visible_gpu_count()
    if visible > 1:
        capped = max(1, min(4, desired))
        return capped
    return max(1, desired)

def safe_thread_num_proc(desired=None):
    if desired is None or not isinstance(desired, int):
        desired = max(1, (os.cpu_count() or 1) // 3)
    return max(1, desired)

def dataset_map_num_proc(desired=None):
    if sys.platform in ("win32", "darwin"):
        return None
    return safe_num_proc(desired)
'''

_VENV_DIR = "/tmp/pr4507_sandbox_venv"


def _create_venv():
    """Create an isolated uv venv."""
    if os.path.exists(_VENV_DIR):
        shutil.rmtree(_VENV_DIR)
    result = subprocess.run(
        ["uv", "venv", _VENV_DIR, "--python", sys.executable],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv venv creation failed: {result.stderr}")


def _venv_python():
    return os.path.join(_VENV_DIR, "bin", "python")


def _run_in_sandbox(script: str) -> subprocess.CompletedProcess:
    """Run a Python script inside the sandbox venv."""
    full_script = _FIXED_FUNCTIONS + "\n" + script
    return subprocess.run(
        [_venv_python(), "-c", full_script],
        capture_output=True, text=True, timeout=30,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


class TestSandboxEdgeCases(unittest.TestCase):
    """Run edge-case scenarios in an isolated uv venv."""

    @classmethod
    def setUpClass(cls):
        _create_venv()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(_VENV_DIR):
            shutil.rmtree(_VENV_DIR)

    def _assert_sandbox(self, script: str, check_fn, msg: str = ""):
        """Run script in sandbox and apply check_fn to stdout."""
        result = _run_in_sandbox(script)
        self.assertEqual(result.returncode, 0,
            f"Sandbox script failed (rc={result.returncode}):\nstderr: {result.stderr}\nstdout: {result.stdout}")
        check_fn(result.stdout.strip())

    # --- cpu_count=None scenarios ---

    def test_SB_1_safe_num_proc_linux_cpu_none(self):
        """SB-1: safe_num_proc() on linux with cpu_count=None."""
        script = """
import os
os.cpu_count = lambda: None
sys.platform = "linux"
r = safe_num_proc()
assert isinstance(r, int) and r >= 1, f"Got {r}"
print(r)
"""
        self._assert_sandbox(script,
            lambda out: self.assertGreaterEqual(int(out), 1),
            "cpu_count=None on linux must not crash")

    def test_SB_2_safe_thread_num_proc_linux_cpu_none(self):
        """SB-2: safe_thread_num_proc() on linux with cpu_count=None."""
        script = """
import os
os.cpu_count = lambda: None
r = safe_thread_num_proc()
assert isinstance(r, int) and r >= 1, f"Got {r}"
print(r)
"""
        self._assert_sandbox(script,
            lambda out: self.assertGreaterEqual(int(out), 1))

    def test_SB_3_dataset_map_num_proc_linux_cpu_none(self):
        """SB-3: dataset_map_num_proc() on linux with cpu_count=None."""
        script = """
import os
os.cpu_count = lambda: None
sys.platform = "linux"
r = dataset_map_num_proc()
assert isinstance(r, int) and r >= 1, f"Got {r}"
print(r)
"""
        self._assert_sandbox(script,
            lambda out: self.assertGreaterEqual(int(out), 1))

    def test_SB_4_safe_num_proc_darwin_cpu_none(self):
        """SB-4: safe_num_proc() on darwin with cpu_count=None returns 1."""
        script = """
import os
os.cpu_count = lambda: None
sys.platform = "darwin"
r = safe_num_proc()
assert r == 1, f"Got {r}"
print(r)
"""
        self._assert_sandbox(script,
            lambda out: self.assertEqual(out, "1"))

    def test_SB_5_dataset_map_num_proc_win32_cpu_none(self):
        """SB-5: dataset_map_num_proc() on win32 with cpu_count=None returns None."""
        script = """
import os
os.cpu_count = lambda: None
sys.platform = "win32"
r = dataset_map_num_proc()
assert r is None, f"Got {r}"
print("None")
"""
        self._assert_sandbox(script,
            lambda out: self.assertEqual(out, "None"))

    # --- desired=0 and desired=-1 scenarios ---

    def test_SB_6_safe_num_proc_zero(self):
        """SB-6: safe_num_proc(0) on linux returns 1."""
        script = """
sys.platform = "linux"
r = safe_num_proc(0)
assert r == 1, f"Got {r}"
print(r)
"""
        self._assert_sandbox(script,
            lambda out: self.assertEqual(out, "1"))

    def test_SB_7_safe_num_proc_negative(self):
        """SB-7: safe_num_proc(-1) on linux returns 1."""
        script = """
sys.platform = "linux"
r = safe_num_proc(-1)
assert r == 1, f"Got {r}"
print(r)
"""
        self._assert_sandbox(script,
            lambda out: self.assertEqual(out, "1"))

    def test_SB_8_safe_thread_num_proc_zero(self):
        """SB-8: safe_thread_num_proc(0) returns 1."""
        script = """
r = safe_thread_num_proc(0)
assert r == 1, f"Got {r}"
print(r)
"""
        self._assert_sandbox(script,
            lambda out: self.assertEqual(out, "1"))

    def test_SB_9_safe_thread_num_proc_negative(self):
        """SB-9: safe_thread_num_proc(-1) returns 1."""
        script = """
r = safe_thread_num_proc(-1)
assert r == 1, f"Got {r}"
print(r)
"""
        self._assert_sandbox(script,
            lambda out: self.assertEqual(out, "1"))

    def test_SB_10_safe_thread_num_proc_darwin_cpu_none(self):
        """SB-10: safe_thread_num_proc() on darwin with cpu_count=None returns >= 1."""
        script = """
import os
os.cpu_count = lambda: None
sys.platform = "darwin"
r = safe_thread_num_proc()
assert isinstance(r, int) and r >= 1, f"Got {r}"
print(r)
"""
        self._assert_sandbox(script,
            lambda out: self.assertGreaterEqual(int(out), 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
