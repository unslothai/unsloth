#!/usr/bin/env python3
"""
Tests for the CUDA wheel-index mapping logic used in install.ps1 and setup.ps1.

Proves:
  1. The PR #4515 original mapping had bugs (cu124 frozen fallback, no CUDA 11.x guard).
  2. The fixed install.ps1 and setup.ps1 mappings resolve those bugs.
  3. Both fixed variants agree on all CUDA >= 12 inputs.
  4. nvidia-smi output parsing handles real and malformed inputs.
  5. The Python reimplementations stay in sync with the on-disk PS1 files.

Usage:
    python test_cuda_wheel_mapping.py             # Fast unit tests (~1s)
    python test_cuda_wheel_mapping.py --network    # + URL validation (~5s)
    python test_cuda_wheel_mapping.py --integration # + sandbox installs (~5min)
    python test_cuda_wheel_mapping.py --all        # Everything
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

# ---------------------------------------------------------------------------
# nvidia-smi output parser (same regex as PS1)
# ---------------------------------------------------------------------------

_CUDA_RE = re.compile(r"CUDA Version:\s+(\d+)\.(\d+)")


def parse_cuda_version(output: str):
    """Parse ``nvidia-smi`` output -> (major, minor) or None."""
    m = _CUDA_RE.search(output)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


# ---------------------------------------------------------------------------
# Three mapping variants (pure Python mirrors of the PS1 functions)
# ---------------------------------------------------------------------------


def map_cuda_pr_original(major, minor, *, has_smi=True, parse_ok=True):
    """PR #4515 original setup.ps1: cu124 fallbacks, no CUDA 11.x guard."""
    if not has_smi:
        return "cu124"
    if not parse_ok:
        return "cu124"
    if major >= 13:
        return "cu130"
    if major == 12 and minor >= 8:
        return "cu128"
    if major == 12 and minor >= 6:
        return "cu126"
    return "cu124"  # BUG: 12.0-12.5 AND 11.x both land here


def map_cuda_fixed_install(major, minor, *, has_smi=True, parse_ok=True):
    """Fixed install.ps1: cu126 fallback, CUDA 11.x -> cpu, no smi -> cpu."""
    if not has_smi:
        return "cpu"
    if not parse_ok:
        return "cu126"
    if major >= 13:
        return "cu130"
    if major == 12 and minor >= 8:
        return "cu128"
    if major == 12 and minor >= 6:
        return "cu126"
    if major >= 12:
        return "cu126"
    return "cpu"


def map_cuda_fixed_setup(major, minor, *, has_smi=True, parse_ok=True):
    """Fixed setup.ps1: cu126 fallback, CUDA 11.x -> cpu, no smi -> cu126."""
    if not has_smi:
        return "cu126"
    if not parse_ok:
        return "cu126"
    if major >= 13:
        return "cu130"
    if major == 12 and minor >= 8:
        return "cu128"
    if major == 12 and minor >= 6:
        return "cu126"
    if major >= 12:
        return "cu126"
    return "cpu"


# ---------------------------------------------------------------------------
# nvidia-smi output fixtures
# ---------------------------------------------------------------------------

_NVIDIA_SMI_TEMPLATE = textwrap.dedent("""\
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI {smi_ver}                 Driver Version: {drv_ver}         CUDA Version: {cuda_ver}  |
    |-------------------------------+------------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf  Pwr:Usage/Cap|          Memory-Usage | GPU-Util  Compute M. |
    |===============================+========================+======================|
    |   0  NVIDIA GeForce ...   Off | 00000000:01:00.0  On |                  N/A |
    | 30%   45C    P8    10W / 250W |    512MiB / 24576MiB |      0%      Default |
    +-------------------------------+------------------------+----------------------+
""")


def _smi_output(cuda_ver, smi_ver="570.00", drv_ver="570.00"):
    return _NVIDIA_SMI_TEMPLATE.format(
        cuda_ver=cuda_ver, smi_ver=smi_ver, drv_ver=drv_ver
    )


# Keyed by a friendly label -> (nvidia-smi output, expected (major, minor))
NVIDIA_SMI_FIXTURES = {
    "CUDA 10.2": (_smi_output("10.2"), (10, 2)),
    "CUDA 11.0": (_smi_output("11.0"), (11, 0)),
    "CUDA 11.8": (_smi_output("11.8"), (11, 8)),
    "CUDA 12.0": (_smi_output("12.0"), (12, 0)),
    "CUDA 12.4": (_smi_output("12.4"), (12, 4)),
    "CUDA 12.5": (_smi_output("12.5"), (12, 5)),
    "CUDA 12.6": (_smi_output("12.6"), (12, 6)),
    "CUDA 12.7": (_smi_output("12.7"), (12, 7)),
    "CUDA 12.8": (_smi_output("12.8"), (12, 8)),
    "CUDA 12.9": (_smi_output("12.9"), (12, 9)),
    "CUDA 13.0": (_smi_output("13.0"), (13, 0)),
    "CUDA 14.0": (_smi_output("14.0"), (14, 0)),
}

# Also grab real nvidia-smi if available
_real_smi = shutil.which("nvidia-smi")
if _real_smi:
    try:
        _real_output = subprocess.check_output(
            [_real_smi], stderr=subprocess.STDOUT, timeout=5
        ).decode("utf-8", errors="replace")
        _parsed = parse_cuda_version(_real_output)
        if _parsed:
            NVIDIA_SMI_FIXTURES["real_machine"] = (_real_output, _parsed)
    except Exception:
        pass

MALFORMED_FIXTURES = {
    "no_cuda_line": ("GPU 0: Tesla V100-SXM2-16GB\nDriver Version: 450.80.02\n", None),
    "wrong_format": ("CUDA Version: abc.def\n", None),
    "major_only": ("CUDA Version: 12\n", None),
    "extra_spaces": ("CUDA Version:   12 .  8\n", None),
    "empty_string": ("", None),
    "cuda_no_version": ("CUDA blah blah\n", None),
}


# ======================================================================
# Test classes
# ======================================================================


class TestBugDemonstration(unittest.TestCase):
    """Side-by-side proof: PR original is broken, fixed version is correct."""

    # (label, major, minor, pr_original_tag, fixed_install_tag, fixed_setup_tag)
    BUG_CASES = [
        ("CUDA 12.4", 12, 4, "cu124", "cu126", "cu126"),
        ("CUDA 12.0", 12, 0, "cu124", "cu126", "cu126"),
        ("CUDA 11.8", 11, 8, "cu124", "cpu", "cpu"),
        ("CUDA 11.0", 11, 0, "cu124", "cpu", "cpu"),
    ]

    def test_pr_bug_vs_fix_with_smi(self):
        for label, major, minor, pr_tag, inst_tag, setup_tag in self.BUG_CASES:
            with self.subTest(label=label):
                self.assertEqual(
                    map_cuda_pr_original(major, minor),
                    pr_tag,
                    f"{label}: PR original should return {pr_tag}",
                )
                self.assertEqual(
                    map_cuda_fixed_install(major, minor),
                    inst_tag,
                    f"{label}: fixed install should return {inst_tag}",
                )
                self.assertEqual(
                    map_cuda_fixed_setup(major, minor),
                    setup_tag,
                    f"{label}: fixed setup should return {setup_tag}",
                )

    def test_no_smi_bug(self):
        self.assertEqual(
            map_cuda_pr_original(0, 0, has_smi=False),
            "cu124",
            "PR original: no smi -> cu124 (frozen, BAD)",
        )
        self.assertEqual(
            map_cuda_fixed_install(0, 0, has_smi=False),
            "cpu",
            "Fixed install: no smi -> cpu (GOOD)",
        )
        self.assertEqual(
            map_cuda_fixed_setup(0, 0, has_smi=False),
            "cu126",
            "Fixed setup: no smi -> cu126 (optimistic, GOOD)",
        )


class TestMappingPROriginal(unittest.TestCase):
    """Characterizes the PR original mapping -- including its bugs."""

    CASES = [
        # (major, minor, expected, note)
        (10, 2, "cu124", "ancient CUDA gets frozen cu124 (BUG)"),
        (11, 0, "cu124", "CUDA 11.0 gets frozen cu124 (BUG)"),
        (11, 8, "cu124", "CUDA 11.8 gets frozen cu124 (BUG)"),
        (12, 0, "cu124", "CUDA 12.0 gets frozen cu124 (BUG)"),
        (12, 1, "cu124", "CUDA 12.1 gets frozen cu124 (BUG)"),
        (12, 2, "cu124", "CUDA 12.2 gets frozen cu124 (BUG)"),
        (12, 3, "cu124", "CUDA 12.3 gets frozen cu124 (BUG)"),
        (12, 4, "cu124", "CUDA 12.4 gets frozen cu124 (BUG)"),
        (12, 5, "cu124", "CUDA 12.5 gets frozen cu124 (BUG)"),
        (12, 6, "cu126", "correctly mapped"),
        (12, 7, "cu126", "correctly mapped"),
        (12, 8, "cu128", "correctly mapped"),
        (12, 9, "cu128", "correctly mapped"),
        (13, 0, "cu130", "correctly mapped"),
        (13, 1, "cu130", "correctly mapped"),
        (14, 0, "cu130", "correctly mapped"),
        (20, 0, "cu130", "far future"),
    ]

    def test_version_mapping(self):
        for major, minor, expected, note in self.CASES:
            with self.subTest(cuda=f"{major}.{minor}", note=note):
                self.assertEqual(map_cuda_pr_original(major, minor), expected)

    def test_no_smi(self):
        self.assertEqual(map_cuda_pr_original(0, 0, has_smi=False), "cu124")

    def test_parse_failure(self):
        self.assertEqual(map_cuda_pr_original(0, 0, parse_ok=False), "cu124")


class TestMappingFixedInstall(unittest.TestCase):
    """Verifies the corrected install.ps1 mapping."""

    CASES = [
        (10, 2, "cpu"),
        (11, 0, "cpu"),
        (11, 8, "cpu"),
        (12, 0, "cu126"),
        (12, 1, "cu126"),
        (12, 2, "cu126"),
        (12, 3, "cu126"),
        (12, 4, "cu126"),
        (12, 5, "cu126"),
        (12, 6, "cu126"),
        (12, 7, "cu126"),
        (12, 8, "cu128"),
        (12, 9, "cu128"),
        (13, 0, "cu130"),
        (13, 1, "cu130"),
        (14, 0, "cu130"),
        (20, 0, "cu130"),
    ]

    def test_version_mapping(self):
        for major, minor, expected in self.CASES:
            with self.subTest(cuda=f"{major}.{minor}"):
                self.assertEqual(map_cuda_fixed_install(major, minor), expected)

    def test_no_smi(self):
        self.assertEqual(map_cuda_fixed_install(0, 0, has_smi=False), "cpu")

    def test_parse_failure(self):
        self.assertEqual(map_cuda_fixed_install(0, 0, parse_ok=False), "cu126")


class TestMappingFixedSetup(unittest.TestCase):
    """Verifies the corrected setup.ps1 mapping."""

    CASES = [
        (10, 2, "cpu"),
        (11, 0, "cpu"),
        (11, 8, "cpu"),
        (12, 0, "cu126"),
        (12, 1, "cu126"),
        (12, 2, "cu126"),
        (12, 3, "cu126"),
        (12, 4, "cu126"),
        (12, 5, "cu126"),
        (12, 6, "cu126"),
        (12, 7, "cu126"),
        (12, 8, "cu128"),
        (12, 9, "cu128"),
        (13, 0, "cu130"),
        (13, 1, "cu130"),
        (14, 0, "cu130"),
        (20, 0, "cu130"),
    ]

    def test_version_mapping(self):
        for major, minor, expected in self.CASES:
            with self.subTest(cuda=f"{major}.{minor}"):
                self.assertEqual(map_cuda_fixed_setup(major, minor), expected)

    def test_no_smi(self):
        self.assertEqual(
            map_cuda_fixed_setup(0, 0, has_smi=False),
            "cu126",
            "setup.ps1 uses optimistic cu126 when no nvidia-smi (Studio always has a GPU)",
        )

    def test_parse_failure(self):
        self.assertEqual(map_cuda_fixed_setup(0, 0, parse_ok=False), "cu126")


class TestFixedScriptsConsistency(unittest.TestCase):
    """Both fixed variants must agree on every CUDA >= 12 input."""

    def test_cuda12_and_above_agree(self):
        for major in range(12, 21):
            for minor in range(0, 10):
                with self.subTest(cuda=f"{major}.{minor}"):
                    self.assertEqual(
                        map_cuda_fixed_install(major, minor),
                        map_cuda_fixed_setup(major, minor),
                    )

    def test_cuda11_and_below_agree(self):
        for major, minor in [(11, 8), (11, 0), (10, 2), (9, 0)]:
            with self.subTest(cuda=f"{major}.{minor}"):
                self.assertEqual(map_cuda_fixed_install(major, minor), "cpu")
                self.assertEqual(map_cuda_fixed_setup(major, minor), "cpu")

    def test_intentional_divergence_no_smi(self):
        """install.ps1 -> cpu, setup.ps1 -> cu126 when nvidia-smi is absent."""
        self.assertEqual(map_cuda_fixed_install(0, 0, has_smi=False), "cpu")
        self.assertEqual(map_cuda_fixed_setup(0, 0, has_smi=False), "cu126")
        self.assertNotEqual(
            map_cuda_fixed_install(0, 0, has_smi=False),
            map_cuda_fixed_setup(0, 0, has_smi=False),
        )


class TestNvidiaSmiParsing(unittest.TestCase):
    """Parser handles real outputs, synthetic outputs, and malformed inputs."""

    def test_synthetic_fixtures(self):
        for label, (output, expected) in NVIDIA_SMI_FIXTURES.items():
            with self.subTest(fixture=label):
                result = parse_cuda_version(output)
                self.assertIsNotNone(result, f"Should parse {label}")
                self.assertEqual(result, expected)

    def test_malformed_fixtures(self):
        for label, (output, expected) in MALFORMED_FIXTURES.items():
            with self.subTest(fixture=label):
                self.assertIsNone(
                    parse_cuda_version(output),
                    f"Should reject malformed input: {label}",
                )

    def test_end_to_end_parse_and_map(self):
        """Parse fixture -> map -> assert correct tag (fixed install)."""
        expected_tags = {
            "CUDA 10.2": "cpu",
            "CUDA 11.0": "cpu",
            "CUDA 11.8": "cpu",
            "CUDA 12.0": "cu126",
            "CUDA 12.4": "cu126",
            "CUDA 12.5": "cu126",
            "CUDA 12.6": "cu126",
            "CUDA 12.7": "cu126",
            "CUDA 12.8": "cu128",
            "CUDA 12.9": "cu128",
            "CUDA 13.0": "cu130",
            "CUDA 14.0": "cu130",
        }
        for label, expected_tag in expected_tags.items():
            with self.subTest(fixture=label):
                output, _ = NVIDIA_SMI_FIXTURES[label]
                parsed = parse_cuda_version(output)
                self.assertIsNotNone(parsed)
                tag = map_cuda_fixed_install(*parsed)
                self.assertEqual(tag, expected_tag)


class TestPS1FileSync(unittest.TestCase):
    """Verify Python reimplementations match the on-disk PS1 source."""

    # Paths relative to the repo root
    _REPO_ROOT = os.path.normpath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    _INSTALL_PS1 = os.path.join(_REPO_ROOT, "unsloth", "install.ps1")
    _SETUP_PS1 = os.path.join(_REPO_ROOT, "unsloth", "studio", "setup.ps1")
    _PR_SETUP_PS1 = os.path.join(
        _REPO_ROOT, "temp", "unsloth_pr4515", "studio", "setup.ps1"
    )

    # Regexes to extract key values from PS1 source
    _RE_CUDA_REGEX = re.compile(r"""['"]CUDA Version:\\s\+\(\\d\+\)\\\.\(\\d\+\)['"]""")
    _RE_THRESHOLD = re.compile(
        r"""\$major\s+-(?:ge|eq)\s+(\d+)(?:\s+-and\s+\$minor\s+-ge\s+(\d+))?"""
    )
    _RE_RETURN_TAG = re.compile(r"""return\s+["'](?:\$baseUrl/)?(\w+)["']""")
    _RE_NO_SMI_RETURN = re.compile(
        r"""if\s+\(-not\s+\$(?:NvidiaSmiExe|smiExe)\)\s*\{\s*return\s+["'](?:\$baseUrl/)?(\w+)["']"""
    )

    def _read_ps1(self, path):
        if not os.path.isfile(path):
            self.skipTest(f"PS1 not found: {path}")
        with open(path, "r", encoding="utf-8-sig") as f:
            return f.read()

    def _extract_function(self, source, func_name):
        """Extract a PS1 function body by name."""
        pattern = re.compile(
            rf"function\s+{re.escape(func_name)}\s*\{{", re.IGNORECASE
        )
        m = pattern.search(source)
        if not m:
            return None
        start = m.end()
        depth = 1
        i = start
        while i < len(source) and depth > 0:
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
            i += 1
        return source[m.start() : i]

    def _extract_thresholds(self, func_body):
        """Return sorted list of (major, minor_or_None) from the if-chains."""
        results = []
        for m in self._RE_THRESHOLD.finditer(func_body):
            major = int(m.group(1))
            minor = int(m.group(2)) if m.group(2) else None
            results.append((major, minor))
        return results

    def _extract_no_smi_fallback(self, func_body):
        m = self._RE_NO_SMI_RETURN.search(func_body)
        return m.group(1) if m else None

    def test_install_ps1_thresholds(self):
        source = self._read_ps1(self._INSTALL_PS1)
        func = self._extract_function(source, "Get-TorchIndexUrl")
        self.assertIsNotNone(func, "Get-TorchIndexUrl not found in install.ps1")

        thresholds = self._extract_thresholds(func)
        # Expected: (13, None), (12, 8), (12, 6), (12, None)
        self.assertIn((13, None), thresholds, "Missing >= 13 threshold")
        self.assertIn((12, 8), thresholds, "Missing 12.8 threshold")
        self.assertIn((12, 6), thresholds, "Missing 12.6 threshold")
        self.assertIn((12, None), thresholds, "Missing >= 12 catch-all threshold")

    def test_install_ps1_no_smi_fallback(self):
        source = self._read_ps1(self._INSTALL_PS1)
        func = self._extract_function(source, "Get-TorchIndexUrl")
        self.assertIsNotNone(func)
        fallback = self._extract_no_smi_fallback(func)
        self.assertEqual(fallback, "cpu", "install.ps1 no-smi should fallback to cpu")

    def test_setup_ps1_thresholds(self):
        source = self._read_ps1(self._SETUP_PS1)
        func = self._extract_function(source, "Get-PytorchCudaTag")
        self.assertIsNotNone(func, "Get-PytorchCudaTag not found in setup.ps1")

        thresholds = self._extract_thresholds(func)
        self.assertIn((13, None), thresholds)
        self.assertIn((12, 8), thresholds)
        self.assertIn((12, 6), thresholds)
        self.assertIn((12, None), thresholds)

    def test_setup_ps1_no_smi_fallback(self):
        source = self._read_ps1(self._SETUP_PS1)
        func = self._extract_function(source, "Get-PytorchCudaTag")
        self.assertIsNotNone(func)
        fallback = self._extract_no_smi_fallback(func)
        self.assertEqual(
            fallback, "cu126", "setup.ps1 no-smi should fallback to cu126"
        )

    def test_pr_original_setup_ps1_has_cu124_bug(self):
        source = self._read_ps1(self._PR_SETUP_PS1)
        func = self._extract_function(source, "Get-PytorchCudaTag")
        if func is None:
            self.skipTest("PR original setup.ps1 not found")

        fallback = self._extract_no_smi_fallback(func)
        self.assertEqual(
            fallback, "cu124", "PR original should have cu124 no-smi fallback (bug)"
        )
        # Confirm no >= 12 catch-all exists
        thresholds = self._extract_thresholds(func)
        self.assertNotIn(
            (12, None),
            thresholds,
            "PR original should NOT have >= 12 catch-all (the bug)",
        )

    def test_cuda_regex_consistent(self):
        """All PS1 files use the same CUDA Version regex."""
        for path in [self._INSTALL_PS1, self._SETUP_PS1]:
            if not os.path.isfile(path):
                continue
            source = self._read_ps1(path)
            self.assertIn(
                r"CUDA Version:\s+(\d+)\.(\d+)",
                source,
                f"CUDA regex mismatch in {os.path.basename(path)}",
            )


@unittest.skipUnless(
    "--network" in sys.argv or "--all" in sys.argv,
    "Requires --network or --all flag",
)
class TestUrlValidation(unittest.TestCase):
    """Verify PyTorch wheel index URLs are reachable."""

    BASE = "https://download.pytorch.org/whl"
    TAGS = ["cpu", "cu124", "cu126", "cu128", "cu130"]

    def test_index_urls_resolve(self):
        import urllib.request

        for tag in self.TAGS:
            url = f"{self.BASE}/{tag}/torch/"
            with self.subTest(tag=tag):
                req = urllib.request.Request(url, method="HEAD")
                try:
                    resp = urllib.request.urlopen(req, timeout=15)
                    self.assertIn(
                        resp.status,
                        (200, 301, 302),
                        f"{tag} index returned {resp.status}",
                    )
                except urllib.error.HTTPError as e:
                    self.fail(f"{tag} index returned HTTP {e.code}")

    def test_cu124_is_frozen(self):
        """cu124 should not have torch >= 2.7 (it stopped at 2.6.0)."""
        import urllib.request

        url = f"{self.BASE}/cu124/torch/"
        req = urllib.request.Request(url)
        try:
            resp = urllib.request.urlopen(req, timeout=15)
            page = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            self.skipTest(f"Could not fetch cu124 index: {e}")

        # Look for torch-2.7 or torch-2.8 etc. in the listing
        has_27_plus = bool(re.search(r"torch-2\.[7-9]|torch-2\.1\d|torch-3\.", page))
        self.assertFalse(
            has_27_plus,
            "cu124 index should NOT contain torch >= 2.7 (it is frozen at 2.6.0)",
        )


@unittest.skipUnless(
    "--integration" in sys.argv or "--all" in sys.argv,
    "Requires --integration or --all flag",
)
class TestSandboxIntegration(unittest.TestCase):
    """Create real venvs and install torch to verify wheel tags."""

    _WORKSPACE = os.path.normpath(
        os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
    )
    _SANDBOX_DIR = os.path.join(_WORKSPACE, "temp", "test_cuda_sandbox")

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls._SANDBOX_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.isdir(cls._SANDBOX_DIR):
            shutil.rmtree(cls._SANDBOX_DIR, ignore_errors=True)

    def _python_exe(self, venv_dir):
        if sys.platform == "win32":
            return os.path.join(venv_dir, "Scripts", "python.exe")
        return os.path.join(venv_dir, "bin", "python")

    def _create_venv_and_install(self, tag):
        venv_dir = os.path.join(self._SANDBOX_DIR, f"venv_{tag}")
        if os.path.isdir(venv_dir):
            shutil.rmtree(venv_dir)

        url = f"https://download.pytorch.org/whl/{tag}"

        # Create venv
        subprocess.check_call(
            ["uv", "venv", venv_dir, "--python", sys.executable],
            timeout=60,
        )

        python = self._python_exe(venv_dir)

        # Install torch only (minimal)
        subprocess.check_call(
            ["uv", "pip", "install", "--python", python, "torch", "--index-url", url],
            timeout=600,
        )

        # Get torch version
        version = subprocess.check_output(
            [python, "-c", "import torch; print(torch.__version__)"],
            timeout=30,
        ).decode().strip()

        return version

    def test_cpu_install(self):
        version = self._create_venv_and_install("cpu")
        self.assertNotIn("+cu", version, f"CPU torch should not have +cu: {version}")

    def test_cu128_install(self):
        version = self._create_venv_and_install("cu128")
        self.assertIn("+cu128", version, f"cu128 torch should have +cu128: {version}")


# ---------------------------------------------------------------------------
# CLI: strip custom flags before unittest sees them
# ---------------------------------------------------------------------------

def main():
    custom_flags = {"--network", "--integration", "--all"}
    # Keep custom flags accessible via sys.argv for skip decorators,
    # but remove them before passing to unittest
    unittest_argv = [a for a in sys.argv if a not in custom_flags]
    unittest.main(module=__name__, argv=unittest_argv, verbosity=2)


if __name__ == "__main__":
    main()
