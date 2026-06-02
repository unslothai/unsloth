"""Tests that HfFileSystem().glob() is skipped when is_model or is_peft is False.

The glob calls in FastLanguageModel.from_pretrained and FastModel.from_pretrained
exist solely to detect repos with both config.json and adapter_config.json. When
either AutoConfig or PeftConfig fails to load, the glob cannot find both files,
so calling it is redundant and risks hanging on slow networks.
"""

import os
import unittest
from unittest.mock import MagicMock, patch


class TestGlobSkippedWhenNotBothConfigs(unittest.TestCase):
    """Verify HfFileSystem.glob is not called when is_model or is_peft is False."""

    def _run_both_exist_block(
        self, is_model, is_peft, supports_llama32, model_name, is_local_dir = False
    ):
        """Simulate the both_exist detection block from loader.py.

        This mirrors the exact logic at lines 500-517 / 1276-1292 of loader.py.
        Returns (both_exist, glob_called).
        """
        from unittest.mock import MagicMock

        both_exist = (is_model and is_peft) and not supports_llama32
        glob_mock = MagicMock(
            return_value = [
                f"{model_name}/config.json",
                f"{model_name}/adapter_config.json",
            ]
        )

        # This mirrors the guarded block in loader.py
        if supports_llama32 and is_model and is_peft:
            if is_local_dir:
                # Local path branch — would use os.path.exists in real code
                both_exist = True  # simulate both files present locally
            else:
                files = glob_mock(f"{model_name}/*.json")
                files = list(os.path.split(x)[-1] for x in files)
                if (
                    sum(x == "adapter_config.json" or x == "config.json" for x in files)
                    >= 2
                ):
                    both_exist = True

        return both_exist, glob_mock.called

    # --- Cases where glob should NOT be called ---

    def test_glob_skipped_when_is_model_false(self):
        both_exist, glob_called = self._run_both_exist_block(
            is_model = False,
            is_peft = True,
            supports_llama32 = True,
            model_name = "org/some-adapter",
        )
        self.assertFalse(glob_called, "glob should not be called when is_model=False")
        self.assertFalse(both_exist)

    def test_glob_skipped_when_is_peft_false(self):
        both_exist, glob_called = self._run_both_exist_block(
            is_model = True,
            is_peft = False,
            supports_llama32 = True,
            model_name = "org/some-model",
        )
        self.assertFalse(glob_called, "glob should not be called when is_peft=False")
        self.assertFalse(both_exist)

    def test_glob_skipped_when_both_false(self):
        both_exist, glob_called = self._run_both_exist_block(
            is_model = False,
            is_peft = False,
            supports_llama32 = True,
            model_name = "org/bad-repo",
        )
        self.assertFalse(glob_called, "glob should not be called when both are False")
        self.assertFalse(both_exist)

    def test_glob_skipped_when_supports_llama32_false(self):
        both_exist, glob_called = self._run_both_exist_block(
            is_model = True,
            is_peft = True,
            supports_llama32 = False,
            model_name = "org/some-model",
        )
        self.assertFalse(
            glob_called, "glob should not be called when SUPPORTS_LLAMA32=False"
        )
        # both_exist is set by the old-style check: (is_model and is_peft) and not SUPPORTS_LLAMA32
        self.assertTrue(both_exist)

    # --- Cases where glob SHOULD be called ---

    def test_glob_called_when_both_true_and_supports_llama32(self):
        both_exist, glob_called = self._run_both_exist_block(
            is_model = True,
            is_peft = True,
            supports_llama32 = True,
            model_name = "org/mixed-repo",
        )
        self.assertTrue(
            glob_called, "glob should be called when is_model and is_peft are both True"
        )
        self.assertTrue(both_exist)

    def test_local_dir_skips_glob(self):
        both_exist, glob_called = self._run_both_exist_block(
            is_model = True,
            is_peft = True,
            supports_llama32 = True,
            model_name = "/local/path/to/model",
            is_local_dir = True,
        )
        self.assertFalse(glob_called, "glob should not be called for local directories")
        self.assertTrue(both_exist)


class TestLoaderSourceHasGuard(unittest.TestCase):
    """Verify the actual loader.py source code has the is_model/is_peft guard."""

    def test_loader_source_has_guard(self):
        """Check that both SUPPORTS_LLAMA32 checks in loader.py include is_model and is_peft."""
        loader_path = os.path.join(
            os.path.dirname(__file__), os.pardir, "unsloth", "models", "loader.py"
        )
        with open(loader_path) as f:
            source = f.read()

        # Find all lines with the SUPPORTS_LLAMA32 check near glob usage
        lines = source.splitlines()
        guard_lines = [
            line.strip()
            for line in lines
            if "SUPPORTS_LLAMA32" in line and "if " in line and "is_model" in line
        ]
        # There should be exactly 2 guarded checks (one per from_pretrained method)
        self.assertEqual(
            len(guard_lines),
            2,
            f"Expected 2 guarded SUPPORTS_LLAMA32 checks with is_model/is_peft, found {len(guard_lines)}: {guard_lines}",
        )
        for line in guard_lines:
            self.assertIn("is_model", line)
            self.assertIn("is_peft", line)


if __name__ == "__main__":
    unittest.main()
