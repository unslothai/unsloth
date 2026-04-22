# Tests for CPU-only and no-torch import paths (refs #5008).
# Each test spawns a fresh subprocess because `import unsloth` mutates global
# state that cannot be undone within a single process.

import os
import subprocess
import sys
import unittest
import warnings


def _run(code: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    # Ensure a clean slate: strip any inherited UNSLOTH_NO_TORCH
    env.pop("UNSLOTH_NO_TORCH", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, "-W", "all", "-c", code],
        capture_output = True,
        text = True,
        env = env,
    )


class TestCpuImport(unittest.TestCase):
    """import unsloth with CPU torch and no GPU should succeed with a warning."""

    def test_import_unsloth_cpu(self):
        result = _run("import unsloth")
        self.assertEqual(result.returncode, 0, msg = result.stderr)
        combined = result.stdout + result.stderr
        self.assertIn(
            "No GPU detected",
            combined,
            msg = "Expected CPU-mode warning not found in output",
        )

    def test_import_fast_language_model_cpu(self):
        result = _run("from unsloth import FastLanguageModel")
        self.assertEqual(result.returncode, 0, msg = result.stderr)
        combined = result.stdout + result.stderr
        self.assertIn(
            "No GPU detected",
            combined,
            msg = "Expected CPU-mode warning not found in output",
        )


class TestNoTorchMode(unittest.TestCase):
    """UNSLOTH_NO_TORCH=1 should suppress GPU logic and emit the right warning."""

    def test_import_unsloth_no_torch_flag(self):
        result = _run("import unsloth", env_extra = {"UNSLOTH_NO_TORCH": "1"})
        self.assertEqual(result.returncode, 0, msg = result.stderr)
        combined = result.stdout + result.stderr
        self.assertIn(
            "UNSLOTH_NO_TORCH is set",
            combined,
            msg = "Expected no-torch warning not found in output",
        )
        self.assertNotIn(
            "No GPU detected",
            combined,
            msg = "CPU warning should not appear when UNSLOTH_NO_TORCH is set",
        )

    def test_no_torch_does_not_raise(self):
        """No exception should be raised in no-torch mode."""
        result = _run(
            "import unsloth; assert not hasattr(unsloth, 'torch') or unsloth.torch is None",
            env_extra = {"UNSLOTH_NO_TORCH": "1"},
        )
        self.assertEqual(result.returncode, 0, msg = result.stderr)


class TestFromPretrainedCpuGuard(unittest.TestCase):
    """from_pretrained must fail fast with RuntimeError on CPU (refs #5008, Roland's review)."""

    def test_fast_language_model_from_pretrained_raises_on_cpu(self):
        result = _run(
            "from unsloth import FastLanguageModel; "
            "FastLanguageModel.from_pretrained('unsloth/Llama-3.2-1B-Instruct')"
        )
        self.assertNotEqual(
            result.returncode, 0, msg = "Expected RuntimeError but got exit 0"
        )
        combined = result.stdout + result.stderr
        self.assertIn(
            "requires a GPU",
            combined,
            msg = "Expected clear GPU-required error message not found",
        )

    def test_fast_model_from_pretrained_raises_on_cpu(self):
        result = _run(
            "from unsloth import FastModel; "
            "FastModel.from_pretrained('unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit')"
        )
        self.assertNotEqual(
            result.returncode, 0, msg = "Expected RuntimeError but got exit 0"
        )
        combined = result.stdout + result.stderr
        self.assertIn(
            "requires a GPU",
            combined,
            msg = "Expected clear GPU-required error message not found",
        )


if __name__ == "__main__":
    unittest.main()
