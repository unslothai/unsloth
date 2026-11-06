"""Cold-import policy tests: no accelerator, Torch import or model download."""

import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "unsloth" / "_ascend_compat.py"
spec = importlib.util.spec_from_file_location("ascend_bootstrap_test_target", PATH)
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


class AscendEagerBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                "UNSLOTH_ASCEND_EAGER_BOOTSTRAP": "1",
                "TORCH_COMPILE_DISABLE": "1",
                "UNSLOTH_COMPILE_DISABLE": "1",
            },
            clear = True,
        )
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in ("torch", "torch_npu", "triton"):
            sys.modules.pop(name, None)
        self.patches = [
            patch.object(bootstrap.platform, "system", return_value = "Linux"),
            patch.object(bootstrap.platform, "machine", return_value = "aarch64"),
            patch.object(bootstrap.util, "find_spec", return_value = object()),
            patch.object(
                bootstrap,
                "version",
                side_effect = lambda n: {"torch": "2.6.0+cpu", "triton": "3.5.1"}[n],
            ),
        ]
        self.mocks = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)
        self.importer = patch.object(bootstrap, "import_module")
        self.import_mock = self.importer.start()
        self.addCleanup(self.importer.stop)

    def test_default_does_not_probe_or_import_optional_backend(self):
        os.environ.pop("UNSLOTH_ASCEND_EAGER_BOOTSTRAP")
        self.assertFalse(bootstrap.bootstrap_ascend_eager())
        self.mocks[2].assert_not_called()
        self.import_mock.assert_not_called()
        self.assertNotIn("TORCH_DEVICE_BACKEND_AUTOLOAD", os.environ)

    def test_either_compile_enabled_keeps_normal_import_policy(self):
        for key in ("TORCH_COMPILE_DISABLE", "UNSLOTH_COMPILE_DISABLE"):
            with self.subTest(key = key), patch.dict(os.environ, {key: "0"}):
                self.assertFalse(bootstrap.bootstrap_ascend_eager())
        self.import_mock.assert_not_called()
        self.assertNotIn("TORCH_DEVICE_BACKEND_AUTOLOAD", os.environ)

    def test_non_ascend_platforms_are_not_changed(self):
        for system, machine in (("Linux", "x86_64"), ("Darwin", "arm64"), ("Windows", "ARM64")):
            with self.subTest(system = system, machine = machine):
                self.mocks[0].return_value = system
                self.mocks[1].return_value = machine
                self.assertFalse(bootstrap.bootstrap_ascend_eager())
        self.import_mock.assert_not_called()

    def test_missing_vendor_package_is_not_imported(self):
        self.mocks[2].return_value = None
        self.assertFalse(bootstrap.bootstrap_ascend_eager())
        self.import_mock.assert_not_called()

    def test_does_not_change_initialized_torch_or_npu(self):
        for name in ("torch", "torch_npu"):
            with self.subTest(name = name), patch.dict(sys.modules, {name: ModuleType(name)}):
                self.assertFalse(bootstrap.bootstrap_ascend_eager())
        self.import_mock.assert_not_called()

    def test_other_version_pairs_are_not_changed(self):
        for torch, triton in (("2.6.0", "3.2.0"), ("2.8.0", "3.5.1")):
            with self.subTest(torch = torch, triton = triton):
                self.mocks[3].side_effect = lambda n: {"torch": torch, "triton": triton}[n]
                self.assertFalse(bootstrap.bootstrap_ascend_eager())
        self.import_mock.assert_not_called()

    def test_missing_version_metadata_keeps_normal_import_policy(self):
        self.mocks[3].side_effect = bootstrap.PackageNotFoundError("triton")
        self.assertFalse(bootstrap.bootstrap_ascend_eager())
        self.import_mock.assert_not_called()

    def test_imports_vendor_backend_with_no_triton_then_removes_sentinel(self):
        observed = []

        def import_backend(name):
            self.assertIn("triton", sys.modules)
            self.assertIsNone(sys.modules["triton"])
            self.assertEqual(os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"], "0")
            observed.append(name)

        self.import_mock.side_effect = import_backend
        self.assertTrue(bootstrap.bootstrap_ascend_eager())
        self.assertEqual(observed, ["torch", "torch_npu"])
        self.assertNotIn("triton", sys.modules)

    def test_restores_existing_triton_object(self):
        previous = ModuleType("triton")
        sys.modules["triton"] = previous
        self.assertTrue(bootstrap.bootstrap_ascend_eager())
        self.assertIs(sys.modules["triton"], previous)

    def test_original_vendor_failure_propagates_and_triton_is_restored(self):
        previous = ModuleType("triton")
        sys.modules["triton"] = previous
        failure = RuntimeError("vendor driver unavailable")
        self.import_mock.side_effect = [None, failure]
        with self.assertRaises(RuntimeError) as raised:
            bootstrap.bootstrap_ascend_eager()
        self.assertIs(raised.exception, failure)
        self.assertIs(sys.modules["triton"], previous)

    def test_failure_without_prior_triton_does_not_leave_sentinel(self):
        self.import_mock.side_effect = RuntimeError("torch unavailable")
        with self.assertRaises(RuntimeError):
            bootstrap.bootstrap_ascend_eager()
        self.assertNotIn("triton", sys.modules)

    def test_explicit_vendor_autoload_setting_is_preserved(self):
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "1"
        self.assertTrue(bootstrap.bootstrap_ascend_eager())
        self.assertEqual(os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"], "1")


if __name__ == "__main__":
    unittest.main()
