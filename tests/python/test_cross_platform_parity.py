"""Cross-platform parity tests between install.sh and install.ps1."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_PS1 = REPO_ROOT / "install.ps1"
SETUP_PS1 = REPO_ROOT / "studio" / "setup.ps1"
STACK_PY = REPO_ROOT / "studio" / "install_python_stack.py"


class TestNoTorchBackendAutoInInstallSh:
    """install.sh primary paths must not use --torch-backend=auto (only the fallback else-branch may)."""

    def test_no_torch_backend_auto_outside_fallback(self):
        lines = INSTALL_SH.read_text(encoding = "utf-8").splitlines()
        # Fallback block: from "GPU detection failed" to the next "fi".
        fallback_start = None
        fallback_end = None
        for i, line in enumerate(lines):
            if fallback_start is None and "GPU detection failed" in line:
                fallback_start = i
            elif fallback_start is not None and fallback_end is None and line.strip() == "fi":
                fallback_end = i
                break
        fallback_range = (
            range(fallback_start or 0, (fallback_end or 0) + 1) if fallback_start else range(0)
        )

        matches = [
            (i + 1, line)
            for i, line in enumerate(lines)
            if "--torch-backend=auto" in line
            and not line.lstrip().startswith("#")
            and i not in fallback_range
        ]
        assert matches == [], (
            f"install.sh contains --torch-backend=auto outside the fallback block at lines: "
            f"{[m[0] for m in matches]}"
        )

    def test_fallback_uses_torch_backend_auto(self):
        """The fallback branch should use --torch-backend=auto as recovery."""
        text = INSTALL_SH.read_text(encoding = "utf-8")
        assert (
            "GPU detection failed" in text
        ), "install.sh should have a fallback branch for when GPU detection fails"


class TestInstallShHasGpuDetection:
    """install.sh must contain the get_torch_index_url function."""

    def test_function_exists(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        assert (
            "get_torch_index_url()" in text
        ), "install.sh is missing the get_torch_index_url() function"

    def test_torch_index_url_assigned(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        assert (
            "TORCH_INDEX_URL=$(get_torch_index_url)" in text
        ), "install.sh should assign TORCH_INDEX_URL from get_torch_index_url()"


class TestCudaMappingParity:
    """CUDA version thresholds must match between install.sh and install.ps1."""

    @staticmethod
    def _extract_cuda_thresholds_sh(text: str) -> list[str]:
        """Extract cu* suffixes from the major/minor comparison chain in install.sh."""
        # Only match lines in the if/elif chain that compare _major/_minor
        in_func = False
        results = []
        for line in text.splitlines():
            if "get_torch_index_url()" in line:
                in_func = True
                continue
            if in_func and line.startswith("}"):
                break
            if in_func and ("_major" in line or "_minor" in line):
                m = re.search(r"/(cu\d+|cpu)", line)
                if m:
                    results.append(m.group(1))
        return results

    @staticmethod
    def _extract_cuda_thresholds_ps1(text: str) -> list[str]:
        """Extract cu* suffixes from the major/minor comparison chain in install.ps1."""
        in_func = False
        depth = 0
        results = []
        for line in text.splitlines():
            if "function Get-TorchIndexUrl" in line:
                in_func = True
                depth = 1
                continue
            if in_func:
                depth += line.count("{") - line.count("}")
                if depth <= 0:
                    break
                # Only match the if-chain lines that compare $major/$minor
                if "$major" in line or "$minor" in line:
                    m = re.search(r"/(cu\d+|cpu)", line)
                    if m:
                        results.append(m.group(1))
        return results

    def test_same_cuda_suffixes(self):
        """Both scripts should produce the same ordered list of CUDA index suffixes."""
        sh_text = INSTALL_SH.read_text(encoding = "utf-8")
        ps1_text = INSTALL_PS1.read_text(encoding = "utf-8")

        sh_thresholds = self._extract_cuda_thresholds_sh(sh_text)
        ps1_thresholds = self._extract_cuda_thresholds_ps1(ps1_text)

        assert len(sh_thresholds) > 0, "Could not extract thresholds from install.sh"
        assert len(ps1_thresholds) > 0, "Could not extract thresholds from install.ps1"
        assert sh_thresholds == ps1_thresholds, (
            f"CUDA mapping mismatch:\n"
            f"  install.sh:  {sh_thresholds}\n"
            f"  install.ps1: {ps1_thresholds}"
        )


class TestPyTorchMirrorEnvVar:
    """Both install scripts must support the UNSLOTH_PYTORCH_MIRROR env var."""

    def test_install_sh_has_mirror_var(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        assert (
            "UNSLOTH_PYTORCH_MIRROR" in text
        ), "install.sh should reference UNSLOTH_PYTORCH_MIRROR"

    def test_install_ps1_has_mirror_var(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            "UNSLOTH_PYTORCH_MIRROR" in text
        ), "install.ps1 should reference UNSLOTH_PYTORCH_MIRROR"


class TestUvBytecodeCompileTimeout:
    """Installers should relax uv bytecode compilation timeout by default."""

    @staticmethod
    def _version_tuple(version: str) -> tuple[int, ...]:
        return tuple(int(part) for part in version.split("."))

    def test_install_sh_uses_uv_version_with_timeout_env(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        match = re.search(r'^UV_MIN_VERSION="([^"]+)"$', text, re.MULTILINE)
        assert match, "install.sh should declare UV_MIN_VERSION"
        assert self._version_tuple(match.group(1)) >= self._version_tuple("0.7.22")

    def test_install_ps1_uses_uv_version_with_timeout_env(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        match = re.search(r'^\s*\$UvMinVersion = "([^"]+)"$', text, re.MULTILINE)
        assert match, "install.ps1 should declare $UvMinVersion"
        assert self._version_tuple(match.group(1)) >= self._version_tuple("0.7.22")
        assert "function Test-UvVersionOk" in text
        assert "if (-not (Test-UvVersionOk))" in text

    def test_install_sh_preserves_timeout_override(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        assert (
            ': "${UV_COMPILE_BYTECODE_TIMEOUT:=180}"' in text
        ), "install.sh should default UV_COMPILE_BYTECODE_TIMEOUT without overwriting callers"
        assert (
            "export UV_COMPILE_BYTECODE_TIMEOUT" in text
        ), "install.sh should export UV_COMPILE_BYTECODE_TIMEOUT for uv subprocesses"

    def test_install_ps1_preserves_timeout_override(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            "if (-not $env:UV_COMPILE_BYTECODE_TIMEOUT)" in text
        ), "install.ps1 should preserve caller UV_COMPILE_BYTECODE_TIMEOUT overrides"
        assert (
            '$env:UV_COMPILE_BYTECODE_TIMEOUT = "180"' in text
        ), "install.ps1 should default UV_COMPILE_BYTECODE_TIMEOUT"


class TestTorchIndexOverrideParity:
    """Every installer must honor UNSLOTH_TORCH_INDEX_URL / _FAMILY so a pinned wheel
    index wins over GPU probing on all platforms (no asymmetric, per-OS coverage)."""

    @pytest.mark.parametrize(
        "path",
        [INSTALL_SH, INSTALL_PS1, SETUP_PS1, STACK_PY],
        ids = ["install.sh", "install.ps1", "setup.ps1", "install_python_stack.py"],
    )
    def test_installer_reads_override_env(self, path):
        text = path.read_text(encoding = "utf-8")
        for var in ("UNSLOTH_TORCH_INDEX_URL", "UNSLOTH_TORCH_INDEX_FAMILY"):
            assert var in text, f"{path.name} does not honor {var}"

    @pytest.mark.parametrize(
        "path",
        [INSTALL_PS1, SETUP_PS1],
        ids = ["install.ps1", "setup.ps1"],
    )
    def test_amd_reroute_guarded_when_pinned(self, path):
        # The AMD ROCm reroute must be skipped when the index is explicitly pinned,
        # so an explicit cpu / cu* / rocm pin on an AMD host is not overwritten.
        text = path.read_text(encoding = "utf-8")
        assert (
            "TorchIndexPinned" in text
        ), f"{path.name} should gate the AMD ROCm reroute on a pinned-index flag"

    def test_cuda_pin_overrides_cvd_hide_gate(self):
        # A pinned cu* index skips ALL host-GPU probing, so the CUDA repair must clear the
        # CUDA_VISIBLE_DEVICES hide gate too (else the GPU-less CI case bails).
        text = STACK_PY.read_text(encoding = "utf-8")
        m = re.search(r"def _ensure_cuda_torch\(\).*?(?=\ndef )", text, re.DOTALL)
        assert m, "could not locate _ensure_cuda_torch"
        body = m.group(0)
        # The CVD hide-gate return must be guarded by the CUDA-pin flag.
        assert "_cuda_pinned" in body, (
            "_ensure_cuda_torch should compute a CUDA-pin flag so the pin can "
            "override the CVD hide gate"
        )
        assert re.search(
            r"if not _cuda_pinned and _cvd is not None", body
        ), "the CVD hide gate must be bypassed when a CUDA index is pinned"

    def test_cpu_repair_pins_supported_torch_range(self):
        # The explicit-CPU repair must use the bounded CPU/CUDA spec, not a bare trio (the
        # /cpu index serves torch 2.11+, so a bare install could resolve out of range).
        text = STACK_PY.read_text(encoding = "utf-8")
        m = re.search(r"def _ensure_cpu_torch\(\).*?(?=\ndef )", text, re.DOTALL)
        assert m, "could not locate _ensure_cpu_torch"
        body = m.group(0)
        assert "_CPU_TORCH_PKG_SPEC" in body, (
            "_ensure_cpu_torch should install the bounded _CPU_TORCH_PKG_SPEC, "
            "not a bare torch/torchvision/torchaudio trio"
        )

    def test_verbatim_update_pins_supported_torch_range(self):
        # The unknown-family verbatim UPDATE must keep the SAME torch<2.11.0 ceiling a fresh
        # install applies, not the wider cu* ceiling (<2.12.0).
        text = STACK_PY.read_text(encoding = "utf-8")
        m = re.search(r"def _ensure_verbatim_torch_index\(\).*?(?=\ndef )", text, re.DOTALL)
        assert m, "could not locate _ensure_verbatim_torch_index"
        assert "_CUSTOM_INDEX_TORCH_PKG_SPEC" in m.group(0), (
            "_ensure_verbatim_torch_index should install the bounded "
            "_CUSTOM_INDEX_TORCH_PKG_SPEC trio (torch<2.11.0, matching the fresh "
            "unknown-leaf bound), not _CUDA_TORCH_PKG_SPEC (<2.12.0) or a bare trio"
        )
        # The custom-index spec must cap torch below 2.11 to mirror install.sh's
        # default TORCH_CONSTRAINT for an unknown leaf.
        spec = re.search(r"_CUSTOM_INDEX_TORCH_PKG_SPEC[^(]*\(\s*(.*?)\)", text, re.DOTALL)
        assert spec and '"torch>=2.4,<2.11.0"' in spec.group(1), (
            "_CUSTOM_INDEX_TORCH_PKG_SPEC must pin torch>=2.4,<2.11.0 (the fresh "
            "unknown-leaf ceiling)"
        )

    def test_setup_ps1_stale_check_gates_rocm_on_supported_arch(self):
        # The stale-venv check must expect ROCm torch only for arches the install path maps
        # to a repo.amd.com index; expecting "rocm" for an unmapped arch (which installs CPU
        # torch) marks a correct CPU venv stale every update.
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "_rocmWheelArches" in text, (
            "setup.ps1 stale check should restrict the ROCm expected-tag to the "
            "supported gfx wheel arches"
        )


class TestGfx211AllowlistParity:
    """The gfx per-arch 2.11-floor leaves (gfx120X-all / gfx1151 / gfx1150) must be the
    SAME set in every installer and its stale/mismatch check. When they diverged, a
    pinned gfx110X-all / gfx90a / gfx908 wheel (<2.11) was force-reinstalled every update."""

    EXPECTED = {"gfx120x-all", "gfx1151", "gfx1150"}

    def test_install_sh_allowlist(self):
        text = INSTALL_SH.read_text(encoding = "utf-8").lower()
        # install.sh: the TORCH_CONSTRAINT case (rocm7.2|gfx120x-all|gfx1151|gfx1150).
        m = re.search(r"rocm7\.2\|gfx120x-all\|gfx1151\|gfx1150", text)
        assert m, "install.sh gfx-2.11 allowlist case not found / changed"

    def test_install_ps1_allowlist(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8").lower()
        m = re.search(r"@\('gfx120x-all',\s*'gfx1151',\s*'gfx1150'\)", text)
        assert m, "install.ps1 $_pinGfx211 allowlist not found / changed"

    def test_setup_ps1_defines_single_allowlist_helper(self):
        # setup.ps1 must define the allowlist once (Test-RocmGfx211Leaf) and reuse it, so
        # the stale check and install spec can't disagree.
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert (
            "function Test-RocmGfx211Leaf" in text
        ), "setup.ps1 should define a single Test-RocmGfx211Leaf allowlist helper"
        assert re.search(
            r"@\('gfx120x-all',\s*'gfx1151',\s*'gfx1150'\)", text.lower()
        ), "Test-RocmGfx211Leaf should hold the gfx-2.11 allowlist"
        assert "$_pinGfx211 = Test-RocmGfx211Leaf" in text, (
            "setup.ps1 install-spec path should reuse Test-RocmGfx211Leaf, not "
            "re-hardcode the allowlist (they must not diverge)"
        )

    def test_stack_py_allowlist(self):
        text = STACK_PY.read_text(encoding = "utf-8").lower()
        assert (
            '"gfx120x-all", "gfx1151", "gfx1150"' in text
        ), "install_python_stack.py _ROCM_GFX_TORCH211_LEAVES not found / changed"


class TestCudaLeafDigitParity:
    """A wheel-family leaf is CUDA only when it is "cu" + digits (cu118/cu128/...).
    A bare cu* glob wrongly catches mirror leaves like /custom or /current; when
    that happened the venv was marked stale and rebuilt on every run. Every
    installer must require a digit after "cu" in its family/CUDA classification."""

    def test_stack_py_requires_cu_digit(self):
        text = STACK_PY.read_text(encoding = "utf-8")
        # EXACT cu+digits: a custom leaf like cu128-private must route to the
        # verbatim/unknown path, not be compared against the installed +cu128 tag.
        assert re.search(
            r'r"cu\[0-9\]\+"', text
        ), "install_python_stack.py _is_cuda_family_leaf must fullmatch cu[0-9]+"

    def test_setup_ps1_requires_cu_digit(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        # EXACT cu+digits: cu128-private must not classify as CUDA (it would become
        # the expected tag and rebuild the venv on every update).
        assert re.search(
            r"'\^cu\[0-9\]\+\$'", text
        ), "setup.ps1 Test-CudaFamilyLeaf must match ^cu[0-9]+$, not a cu* prefix"
        # The stale-venv branch must go through the digit-guarded helper.
        assert (
            "Test-CudaFamilyLeaf $_pinLeaf" in text
        ), "setup.ps1 stale check should classify CUDA via Test-CudaFamilyLeaf"

    def test_install_ps1_requires_cu_digit_in_gpu_branch(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert re.search(
            r"'\^cu\[0-9\]'", text
        ), "install.ps1 Get-TauriGpuBranch must require a digit after cu"

    def test_install_sh_requires_cu_digit_in_gpu_branch(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        # The _tauri_gpu_branch cuda case must be cu[0-9]*, not a bare cu*.
        assert re.search(
            r"cu\[0-9\]\*\)\s*echo \"cuda\"", text
        ), "install.sh _tauri_gpu_branch cuda case must be cu[0-9]*, not cu*"

    def test_install_sh_backend_export_requires_cu_digit(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        # The backend export must brand CUDA only on cu[0-9]*; a bare catch-all *) -> cuda
        # would mis-brand /current, /custom pins and skip ROCm repair on AMD hosts.
        assert re.search(
            r'cu\[0-9\]\*\)\s*export UNSLOTH_TORCH_BACKEND="cuda"', text
        ), "install.sh backend export must brand cuda only on cu[0-9]*"
        # An unknown leaf must NOT commit a cuda backend (it unsets instead).
        assert re.search(
            r"\*\)\s*unset UNSLOTH_TORCH_BACKEND", text
        ), "install.sh backend export must unset (not force cuda) on an unknown leaf"

    def test_install_sh_lowercases_backend_leaf(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        # The leaf feeding both the backend case and the 2.11 floor case must be
        # lowercased so the canonical gfx120X-all (capital X) matches.
        assert re.search(
            r"_torch_index_leaf=\$\(printf '%s' \"\$_torch_index_leaf\" \| tr '\[:upper:\]' '\[:lower:\]'\)",
            text,
        ), "install.sh must lowercase _torch_index_leaf before the gfx/rocm/cu case matches"


class TestTorchIndexMarkerParity:
    """All four installers must agree on the torch-index marker (PR #6692):
    the same filename, the same write points, and the same read helpers."""

    MARKER = ".unsloth-torch-index"

    def test_all_installers_use_same_marker_filename(self):
        # The exact marker filename must appear in every installer so bash / py /
        # ps write and read the same per-venv path.
        for path, label in (
            (INSTALL_SH, "install.sh"),
            (INSTALL_PS1, "install.ps1"),
            (SETUP_PS1, "setup.ps1"),
            (STACK_PY, "install_python_stack.py"),
        ):
            text = path.read_text(encoding = "utf-8")
            assert self.MARKER in text, f"{label} must reference the marker '{self.MARKER}'"

    def test_all_installers_write_the_marker(self):
        # install.sh + PowerShell use a _write_torch_index_marker / Write-TorchIndexMarker
        # helper; the Python stack calls _write_torch_index_marker at its install sites.
        assert "_write_torch_index_marker" in INSTALL_SH.read_text(encoding = "utf-8")
        assert "Write-TorchIndexMarker" in INSTALL_PS1.read_text(encoding = "utf-8")
        assert "Write-TorchIndexMarker" in SETUP_PS1.read_text(encoding = "utf-8")
        assert "_write_torch_index_marker" in STACK_PY.read_text(encoding = "utf-8")

    def test_setup_ps1_and_python_read_the_marker(self):
        # The repair/update side (setup.ps1 stale check, python _ensure_rocm_torch)
        # must READ the marker to make the exact pin-change decision.
        assert "Read-TorchIndexMarker" in SETUP_PS1.read_text(encoding = "utf-8")
        assert "Test-MarkerPinMismatch" in SETUP_PS1.read_text(encoding = "utf-8")
        stack = STACK_PY.read_text(encoding = "utf-8")
        assert "_read_torch_index_marker" in stack
        assert "_marker_pin_mismatch" in stack

    def test_all_installers_normalize_index_url(self):
        # The exact-compare normalization (trim, strip trailing slash, lowercase
        # leaf) must exist in every language so the compare is identical.
        assert "_normalize_index_url" in INSTALL_SH.read_text(encoding = "utf-8")
        assert "Get-NormalizedIndexUrl" in SETUP_PS1.read_text(encoding = "utf-8")
        assert "_normalize_index_url" in STACK_PY.read_text(encoding = "utf-8")

    def test_marker_written_atomically(self):
        # bash uses a temp file + mv; PowerShell a temp file + Move-Item; Python
        # tempfile + os.replace. Confirm the atomic-write intent in each.
        assert re.search(r"mv -f \"\$_wm_tmp\"", INSTALL_SH.read_text(encoding = "utf-8"))
        for ps in (INSTALL_PS1, SETUP_PS1):
            assert "Move-Item" in ps.read_text(encoding = "utf-8")
        assert "os.replace" in STACK_PY.read_text(encoding = "utf-8")

    def test_all_writers_strip_credentials_before_persisting(self):
        """An authenticated pin (https://user:token@mirror/...) must not persist
        its credentials in the per-venv marker (mode 0644 under a default POSIX
        umask). Every writer strips userinfo before writing; every normalizer
        strips it before comparing, so an OLD cred-bearing marker still compares
        equal to the same pin (backward compatibility, no reinstall loop)."""
        sh = INSTALL_SH.read_text(encoding = "utf-8")
        assert "_strip_index_url_credentials()" in sh
        assert (
            '_wm_url=$(_strip_index_url_credentials "$_wm_url")' in sh
        ), "install.sh _write_torch_index_marker must strip credentials"
        assert (
            '_n_url=$(_strip_index_url_credentials "$_n_url")' in sh
        ), "install.sh _normalize_index_url must strip credentials"
        for path in (INSTALL_PS1, SETUP_PS1):
            text = path.read_text(encoding = "utf-8")
            assert "function Remove-IndexUrlCredentials" in text, path.name
            assert (
                "Remove-IndexUrlCredentials $IndexUrl.Trim()" in text
            ), f"{path.name} Write-TorchIndexMarker must strip credentials"
        setup = SETUP_PS1.read_text(encoding = "utf-8")
        assert (
            "(Remove-IndexUrlCredentials $Url.Trim()).TrimEnd('/')" in setup
        ), "setup.ps1 Get-NormalizedIndexUrl must strip credentials"
        stack = STACK_PY.read_text(encoding = "utf-8")
        assert "def _strip_index_url_credentials" in stack
        assert (
            'payload = _strip_index_url_credentials(index_url.strip()) + "\\n"' in stack
        ), "install_python_stack.py _write_torch_index_marker must strip credentials"
        assert (
            "url = _strip_index_url_credentials(url)" in stack
        ), "install_python_stack.py _normalize_index_url must strip credentials"

    def test_all_credential_strippers_drop_query_and_fragment(self):
        """The credential stripper feeding the marker writer and the logged repair
        messages must drop a query/fragment token, not just user:pass@ userinfo: a
        private feed can carry its auth token as .../simple?token=SECRET, which would
        otherwise persist in the world-readable marker and print in substep output.
        All four implementations drop it before building the sanitized URL."""
        sh = INSTALL_SH.read_text(encoding = "utf-8")
        assert (
            '_sic_rest="${_sic_rest%%\\?*}"' in sh and '_sic_rest="${_sic_rest%%#*}"' in sh
        ), "install.sh _strip_index_url_credentials must drop query/fragment"
        for path in (INSTALL_PS1, SETUP_PS1):
            text = path.read_text(encoding = "utf-8")
            assert (
                "$rest.IndexOfAny([char[]]('?', '#'))" in text
            ), f"{path.name} Remove-IndexUrlCredentials must drop query/fragment"
        stack = STACK_PY.read_text(encoding = "utf-8")
        assert (
            'rest = rest.split("?", 1)[0].split("#", 1)[0]' in stack
        ), "install_python_stack.py _strip_index_url_credentials must drop query/fragment"

    def test_all_installers_drop_query_before_leaf_classification(self):
        """A token-authenticated index URL (.../cu128?token=x) must classify as the
        cu128 family in every installer: the raw last-path-segment leaf keeps the
        query attached, which passes the cu prefix check but can never equal the
        installed cu128 tag, forcing a multi-GB reinstall on every update."""
        sh = INSTALL_SH.read_text(encoding = "utf-8")
        assert "_torch_index_url_leaf()" in sh
        # The top-level backend-branding leaf strips query + fragment first.
        assert (
            '_torch_index_leaf="${TORCH_INDEX_URL%%\\?*}"' in sh
        ), "install.sh backend-branding leaf must drop the query string first"
        setup = SETUP_PS1.read_text(encoding = "utf-8")
        assert (
            "($Url -split '[?#]', 2)[0]" in setup
        ), "setup.ps1 Get-TorchIndexLeaf must drop query/fragment first"
        ps1 = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            "-split '[?#]', 2" in ps1
        ), "install.ps1 leaf classification must drop query/fragment first"
        stack = STACK_PY.read_text(encoding = "utf-8")
        assert "def _torch_index_leaf" in stack
        assert (
            'url.split("?", 1)[0].split("#", 1)[0]' in stack
        ), "install_python_stack.py _torch_index_leaf must drop query/fragment first"


class TestKnown211SetParity:
    """The KNOWN-2.11 rocm/gfx set must be identical across all four installers:
    exactly {rocm7.2} plus the gfx allowlist {gfx120x-all, gfx1151, gfx1150}.
    rocm7.3 / torch 2.12 do not exist, so no side may floor them speculatively."""

    def test_install_sh_known_211_leaf_is_rocm72_and_gfx_allowlist(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        # The 2.11 floor case matches exactly rocm7.2 + the three gfx leaves.
        assert re.search(
            r"rocm7\.2\|gfx120x-all\|gfx1151\|gfx1150\)", text
        ), "install.sh 2.11 floor must be exactly rocm7.2|gfx120x-all|gfx1151|gfx1150"
        # No speculative rocm7.3 anywhere.
        assert "rocm7.3" not in text, "install.sh must not reference a non-existent rocm7.3"

    def test_python_known_211_versions_is_only_rocm72(self):
        text = STACK_PY.read_text(encoding = "utf-8")
        assert "_ROCM_KNOWN_TORCH211_VERSIONS" in text
        # The frozenset literal is exactly {(7, 2)}.
        m = re.search(r"_ROCM_KNOWN_TORCH211_VERSIONS[^=]*=\s*frozenset\(\{([^}]*)\}\)", text)
        assert m is not None, "install_python_stack.py must define _ROCM_KNOWN_TORCH211_VERSIONS"
        assert "(7, 2)" in m.group(1)
        assert "7, 3" not in m.group(1) and "7, 1" not in m.group(1)

    def test_setup_ps1_known_211_helper_is_only_rocm72(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "Test-RocmKnown211Version" in text
        # The predicate is Major -eq 7 -and Minor -eq 2 (only rocm7.2).
        assert re.search(
            r"Test-RocmKnown211Version[\s\S]{0,400}\$Major -eq 7 -and \$Minor -eq 2", text
        ), "setup.ps1 Test-RocmKnown211Version must accept only rocm7.2"

    def test_install_ps1_pin_floor_is_only_rocm72(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        # The pinned-ROCm install-spec floor must be Major -eq 7 -and Minor -eq 2,
        # not the speculative >= 2 that would floor a non-existent rocm7.3.
        assert re.search(
            r"\$_pinRocm211 = \(\[int\]\$Matches\[1\] -eq 7 -and \[int\]\$Matches\[2\] -eq 2\)",
            text,
        ), "install.ps1 pinned-ROCm floor must be rocm7.2 only (no speculative >= 2)"

    def test_ps1_pin_floor_gate_is_anchored(self):
        """The floor-selection gate that reads $_pinRocm211 from the raw leaf must anchor
        the rocm match ($), or a suffixed custom leaf (rocm7.2-private) matches the rocm7.2
        prefix, takes the 2.11-floor branch, and is force-routed through the ROCm path
        before the exact-match elseif can send it to the verbatim install (Codex P2)."""
        for path, label in ((INSTALL_PS1, "install.ps1"), (SETUP_PS1, "setup.ps1")):
            text = path.read_text(encoding = "utf-8")
            assert "-match '^rocm(\\d+)\\.(\\d+)$'" in text, (
                f"{label} floor gate must anchor the rocm match (^rocm(\\d+)\\.(\\d+)$) so a "
                "suffixed custom leaf is not floored/routed as rocm7.2"
            )
            assert (
                "-match '^rocm(\\d+)\\.(\\d+)'\n" not in text
            ), f"{label} floor gate must not use the unanchored ^rocm(\\d+)\\.(\\d+) prefix"

    def test_install_ps1_bounds_unknown_leaf_pinned_torch(self):
        """install.ps1's custom (non-cu-family) pinned-torch install must bound BOTH
        companions, not just torch: a private mirror serving newer torchvision/torchaudio
        must not pull a companion built for a newer torch ABI while the marker records the
        pin as applied. A cu<digits> family index keeps bare companions (Codex P2)."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            '$_pinVisionSpec = "torchvision>=0.19,<0.26.0"' in text
        ), "install.ps1 custom-pin install must bound torchvision (>=0.19,<0.26.0)"
        assert (
            '$_pinAudioSpec = "torchaudio>=2.4,<2.11.0"' in text
        ), "install.ps1 custom-pin install must bound torchaudio (>=2.4,<2.11.0)"
        # Gated on the leaf NOT being a cu<digits> family (a cu index bounds itself).
        assert (
            "$_pinCuLeaf -notmatch '^cu[0-9]+$'" in text
        ), "install.ps1 must bound companions only for a non-cu-family (custom) pin"
        # The bounded companions must actually be passed to the install command.
        assert re.search(
            r'"torch>=2\.4,<2\.11\.0" \$_pinVisionSpec \$_pinAudioSpec --default-index \$TorchIndexUrl',
            text,
        ), "install.ps1 custom-pin install must pass the bounded companion specs to uv"

    def test_gfx_allowlist_matches_across_installers(self):
        # The gfx 2.11 allowlist {gfx120x-all, gfx1151, gfx1150} must appear in each.
        gfx = ("gfx120x-all", "gfx1151", "gfx1150")
        for path, label in (
            (INSTALL_SH, "install.sh"),
            (INSTALL_PS1, "install.ps1"),
            (SETUP_PS1, "setup.ps1"),
            (STACK_PY, "install_python_stack.py"),
        ):
            low = path.read_text(encoding = "utf-8").lower()
            for g in gfx:
                assert g in low, f"{label} missing gfx 2.11 allowlist member {g}"


class TestPinnedRocmLeafDigitParity:
    """A pinned index is a pip ROCm --default-index family only when its leaf is an
    EXACT rocm+digits (rocm7 / rocm7.2) or gfx*. A ^rocm[0-9] PREFIX (or a bare rocm*
    glob) wrongly catches a custom mirror / find-links leaf (rocm-current /
    rocm-rel-7.2.1) AND a suffixed private-mirror leaf (rocm7.2-private / rocm7-current),
    routing it through the ROCm install path (which silently falls back to CPU on
    failure) or skipping the custom-index companion bounds, instead of the verbatim
    --default-index install. All installers must match the family EXACTLY: Python and
    install.sh via a shared _is_pip_rocm_family_leaf, setup.ps1 via Test-PipRocmFamilyLeaf,
    install.ps1 via an anchored ^rocm[0-9]+(\\.[0-9]+)?$ reroute."""

    def test_install_ps1_pinned_reroute_requires_rocm_digit(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        # The pinned gfx*/rocm reroute must match rocm EXACTLY (anchored), so a suffixed
        # rocm7.2-private / rocm-current falls through to the verbatim --default-index path.
        assert "-match '^rocm[0-9]+(\\.[0-9]+)?$'" in text, (
            "install.ps1 pinned-index reroute must anchor the rocm match "
            "(^rocm[0-9]+(\\.[0-9]+)?$), not a bare -like 'rocm*' or an unanchored ^rocm\\d"
        )
        # Neither the broad glob nor the unanchored prefix may drive that reroute.
        assert (
            "-like 'rocm*'" not in text
        ), "install.ps1 must not route a pinned index on a bare -like 'rocm*' glob"
        assert (
            "-match '^rocm\\d'" not in text
        ), "install.ps1 must not route a pinned index on an unanchored -match '^rocm\\d'"

    def test_setup_ps1_pinned_reroute_requires_rocm_digit(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        # setup.ps1 routes every family decision through Test-PipRocmFamilyLeaf, which
        # anchors the rocm match so a suffixed custom leaf stays on the verbatim path.
        assert (
            "function Test-PipRocmFamilyLeaf" in text
        ), "setup.ps1 must define Test-PipRocmFamilyLeaf (the exact rocm/gfx family gate)"
        assert "'^rocm[0-9]+(\\.[0-9]+)?$'" in text, (
            "setup.ps1 Test-PipRocmFamilyLeaf must anchor the rocm match "
            "(^rocm[0-9]+(\\.[0-9]+)?$) so rocm7.2-private / rocm-current stay verbatim"
        )
        pinned_block = text[text.find("$_pinGfx211 = Test-RocmGfx211Leaf") :][:2000]
        assert (
            "-like 'rocm*'" not in pinned_block
        ), "setup.ps1 pinned reroute must not route on a bare -like 'rocm*' glob"

    def test_install_sh_repairable_requires_rocm_digit(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        # _torch_index_repairable routes rocm/gfx through the exact-match helper.
        assert (
            "_is_pip_rocm_family_leaf" in text
        ), "install.sh must define/use _is_pip_rocm_family_leaf for the exact rocm gate"
        assert re.search(
            r'case "\$1" in\n\s*gfx\*\) return 0', text
        ), "install.sh _is_pip_rocm_family_leaf must treat gfx* as a family"

    def test_stack_py_pip_rocm_family_requires_digit(self):
        text = STACK_PY.read_text(encoding = "utf-8")
        assert re.search(
            r'fullmatch\(r"rocm\\d\+\(\?:\\\.\\d\+\)\?", leaf\)', text
        ), "install_python_stack.py _is_pip_rocm_family_leaf must fullmatch rocm\\d+(?:\\.\\d+)?"
        # The unanchored prefix must be gone from the family/flavor gates.
        assert (
            're.match(r"^rocm\\d"' not in text
        ), "install_python_stack.py must not gate a family on an unanchored re.match(^rocm\\d)"

    def test_normalize_family_leaf_digit_gates_rocm(self):
        """_normalize_family_leaf lowercases only true family leaves. rocm7.2 / cu128
        are families (lowercased), but rocm-Current / rocm-rel-7.2.1 / cu128-private AND
        a suffixed rocm7.2-private keep their case so a custom-index leaf is not falsely
        matched equal (URL paths can be case-sensitive). All three installers match the
        rocm family EXACTLY (not a ^rocm[0-9] prefix) and match the cu leaf EXACTLY."""
        sh = INSTALL_SH.read_text(encoding = "utf-8")
        assert '_is_pip_rocm_family_leaf "$_l_low"' in sh, (
            "install.sh _normalize_family_leaf must gate rocm via the exact-match "
            "_is_pip_rocm_family_leaf helper (not a rocm[0-9]* glob)"
        )
        # cu is exact: the leaf is a family only when stripping "cu" leaves all digits.
        assert '"${_l_low#cu}"' in sh, (
            "install.sh _normalize_family_leaf must match cu EXACTLY (strip cu, require "
            "an all-digit remainder) so cu128-private stays a custom leaf"
        )
        setup = SETUP_PS1.read_text(encoding = "utf-8")
        assert "(Test-PipRocmFamilyLeaf $low)" in setup, (
            "setup.ps1 Get-NormalizedFamilyLeaf must gate rocm via the exact-match "
            "Test-PipRocmFamilyLeaf helper (not a ^(rocm[0-9]|gfx) prefix)"
        )
        assert (
            "-match '^cu[0-9]+$'" in setup
        ), "setup.ps1 Get-NormalizedFamilyLeaf must match cu EXACTLY (^cu[0-9]+$)"
        stack = STACK_PY.read_text(encoding = "utf-8")
        assert re.search(
            r'fullmatch\(r"rocm\[0-9\]\+\(\?:\\\.\[0-9\]\+\)\?", low\)', stack
        ), "install_python_stack.py _normalize_family_leaf must fullmatch rocm[0-9]+(?:\\.[0-9]+)?"
        assert re.search(
            r'r"cu\[0-9\]\+"', stack
        ), "install_python_stack.py _normalize_family_leaf must fullmatch cu[0-9]+"

    def test_setup_ps1_marker_compare_is_case_sensitive(self):
        """Test-MarkerPinMismatch must use -cne, not -ne: normalization preserves
        unknown-leaf case, so a case-only custom-index change (/Simple -> /simple)
        is a real mismatch that PowerShell's case-insensitive -ne would miss."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "(Get-NormalizedIndexUrl $PinUrl) -cne (Get-NormalizedIndexUrl $marker)" in text, (
            "setup.ps1 Test-MarkerPinMismatch must compare normalized URLs with -cne "
            "(case-sensitive) so a case-only custom-index change triggers reinstall"
        )

    def test_install_sh_rocm_side_effects_digit_gated(self):
        """The AMD bitsandbytes + 'repair ROCm torch' side effects must fire only on
        an EXACT ROCm family (rocm7.2/gfx*), not a bare */rocm* whole-URL glob nor a
        ^rocm[0-9] prefix that catches a custom CPU/CUDA index like /rocm-current or a
        suffixed /rocm7.2-private and force-repairs it from the wrong --default-index."""
        text = INSTALL_SH.read_text(encoding = "utf-8")
        assert (
            'if _is_pip_rocm_family_leaf "$_torch_index_leaf"; then\n    _torch_index_is_rocm_family=true'
            in text
        ), "install.sh must set _torch_index_is_rocm_family from the exact-match helper"
        assert (
            '[ "$_torch_index_is_rocm_family" = true ]' in text
        ), "install.sh ROCm bnb/repair hooks must gate on _torch_index_is_rocm_family"
        assert (
            "*/rocm*|*/gfx*)\n                _install_bnb_rocm" not in text
        ), "install.sh must not gate _install_bnb_rocm on a bare */rocm* whole-URL glob"


class TestFirstCustomPinAppliedWithoutMarker:
    """An explicitly-set custom (unknown-family) UNSLOTH_TORCH_INDEX_URL must be
    applied on the FIRST `studio update` of a venv that predates the marker feature.
    Such a venv has no .unsloth-torch-index marker, so the marker compare returns
    None/$null and the version-tag heuristics cannot judge an unknown leaf; without
    treating "no marker" as "apply verbatim once", the explicit pin would be silently
    ignored until a marker happened to exist. The verbatim reinstall writes the
    marker, so every later update is a no-op (marker == pin)."""

    def test_stack_py_applies_pin_when_marker_absent(self):
        text = STACK_PY.read_text(encoding = "utf-8")
        body = text[text.find("def _ensure_verbatim_torch_index") :]
        body = body[: body.find("\ndef _record_torch_index_pin_baseline")]
        # The short-circuit must be "already this exact pin" (False), NOT the old
        # "is not True", which also bailed on a None (no-marker) result.
        assert "if _mismatch is False:" in body, (
            "_ensure_verbatim_torch_index must reinstall on an absent marker (None), "
            "returning early only when the marker already records this exact pin (False)"
        )
        assert "if _mismatch is not True:" not in body, (
            "_ensure_verbatim_torch_index must no longer skip the reinstall when the "
            "marker is absent (None)"
        )

    def test_setup_sh_forces_stack_pass_only_when_pin_needs_applying(self):
        """On Linux `studio update` runs setup.sh, which skips install_python_stack.py (the
        only place the marker-driven torch reinstall lives) when unsloth is current. setup.sh
        must force the pass for an explicit pin, but only when not yet applied (via the
        --torch-pin-needs-apply probe), so an already-applied pin keeps the fast path."""
        setup_sh = REPO_ROOT / "studio" / "setup.sh"
        text = setup_sh.read_text(encoding = "utf-8")
        assert (
            '[ -n "${UNSLOTH_TORCH_INDEX_URL:-}${UNSLOTH_TORCH_INDEX_FAMILY:-}" ]' in text
        ), "setup.sh must gate the pin-apply pass on the pin env vars"
        assert "--torch-pin-needs-apply" in text, (
            "setup.sh must probe install_python_stack.py --torch-pin-needs-apply so the "
            "expensive pass runs only when the marker does not already record the pin"
        )
        assert '[ "$_PIN_NEEDS_APPLY" != 1 ]' in text, (
            "setup.sh must keep the fast path only on an explicit exit 1 (already applied) "
            "and fail safe (run the pass) on exit 0 or a probe error"
        )
        # Under `set -euo pipefail` the probe's exit 1 (already-applied) must be absorbed
        # with `|| _PIN_NEEDS_APPLY=$?`, else it aborts the update before the capture.
        assert "|| _PIN_NEEDS_APPLY=$?" in text, (
            "setup.sh must capture the probe exit with `|| _PIN_NEEDS_APPLY=$?` so a "
            "nonzero probe result does not kill the script under set -e"
        )

    def test_stack_py_exposes_torch_pin_needs_apply_probe(self):
        """install_python_stack.py must answer the setup.sh/setup.ps1 probe: exit 0 when a
        pin is set and the marker doesn't match OR the torch flavor drifted; exit 1 otherwise.
        Reusing the Python normalization keeps the shell side from drifting from it."""
        text = STACK_PY.read_text(encoding = "utf-8")
        assert (
            '"--torch-pin-needs-apply" in sys.argv' in text
        ), "install_python_stack.py must handle the --torch-pin-needs-apply query"
        assert (
            "def _torch_pin_needs_apply()" in text
        ), "the probe decision must live in a testable _torch_pin_needs_apply() helper"
        assert "_marker_pin_mismatch(pin) is not False" in text, (
            "the probe must report 'needs apply' when the marker differs (True) or is absent "
            "(None), and only reach the flavor check when it already matches (False)"
        )
        assert "_torch_flavor_matches_pin(pin, flavor) is False" in text, (
            "the probe must also force the pass when a known-family pin's installed flavor "
            "drifted (clobbered), which the marker (last install source) cannot detect"
        )

    def test_setup_ps1_probes_pin_needs_apply_in_fast_path(self):
        """setup.ps1 must apply the same probe in its fast 'up to date' path for parity, so
        a known-family pin on a marker-less venv records its baseline (breaking the loop)."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert (
            "--torch-pin-needs-apply" in text
        ), "setup.ps1 fast path must probe install_python_stack.py --torch-pin-needs-apply"
        assert (
            "$env:UNSLOTH_TORCH_INDEX_URL -or $env:UNSLOTH_TORCH_INDEX_FAMILY" in text
        ), "setup.ps1 must gate the probe on the pin env vars"

    def test_stack_py_records_pin_baseline_after_ensures(self):
        """The update torch-ensure sequence must record a pin baseline so a known-family
        pin on a marker-less venv (no reinstall needed) still writes the marker once --
        otherwise setup.sh/setup.ps1 would re-run the pass on every update forever."""
        text = STACK_PY.read_text(encoding = "utf-8")
        assert (
            "def _record_torch_index_pin_baseline" in text
        ), "install_python_stack.py must define _record_torch_index_pin_baseline"
        # It must run right after _ensure_verbatim_torch_index in every ensure block
        # (step 2b + the step-13 final blocks), whatever the indent level.
        assert (
            len(
                re.findall(
                    r"_ensure_verbatim_torch_index\(\)\n\s+_record_torch_index_pin_baseline\(\)",
                    text,
                )
            )
            >= 2
        ), (
            "_record_torch_index_pin_baseline must be called after the verbatim helper in "
            "every update torch-ensure block"
        )

    def test_stack_py_captures_verbatim_baseline_before_core_step(self):
        """The verbatim custom-pin baseline must be captured BEFORE the core package step
        (# 3), which can re-resolve and clobber a custom-pinned torch. If it ran after,
        the step-2b verbatim pass would record the already-clobbered trio as the baseline
        for a matching marker and the final pass would see no drift (round-11 P2)."""
        text = STACK_PY.read_text(encoding = "utf-8")
        cap_idx = text.find("_capture_verbatim_baseline()\n")
        step3_idx = text.find("# 3. Core packages")
        assert cap_idx != -1, "install_python_stack.py must call _capture_verbatim_baseline()"
        assert step3_idx != -1, "the core package step (# 3) must exist"
        assert cap_idx < step3_idx, (
            "_capture_verbatim_baseline() must run BEFORE the core package step that can "
            "clobber a custom-pinned torch, or the pre-clobber baseline is lost"
        )

    def test_stack_py_final_repair_covers_windows_and_macos_arm(self):
        """The step-13 final torch repair runs the full cuda/rocm/cpu family sequence on
        Linux only, but a custom (unknown-family) pin also needs its verbatim drift
        repair on Windows (a dependency step can clobber the pin step 2b applied) and
        macOS ARM (step 2b is skipped there, so this is where the pin is first applied).
        The Windows/macOS-ARM branch must run the verbatim helper without the Linux-only
        family repair."""
        text = STACK_PY.read_text(encoding = "utf-8")
        assert "elif IS_WINDOWS or IS_MAC_ARM:" in text, (
            "the step-13 final repair must have a Windows/macOS-ARM branch so a custom "
            "pin is repaired/applied there, not only on Linux"
        )
        # A KNOWN-family pin also needs a repair on Windows: a later dependency step can
        # clobber it and _ensure_{cuda,rocm,cpu}_torch self-skip there, so the branch must
        # call _ensure_pinned_known_family_torch.
        _win_start = text.find("elif IS_WINDOWS or IS_MAC_ARM:")
        win_branch = text[_win_start : text.find("# 14.", _win_start)]
        assert "_ensure_pinned_known_family_torch()" in win_branch, (
            "the Windows/macOS-ARM branch must repair a clobbered known-family pin "
            "via _ensure_pinned_known_family_torch (the family helpers no-op on Windows)"
        )
        # An explicit rocm/gfx pin's wheel can be clobbered too. That repair is OWNED by
        # _ensure_pinned_known_family_torch (reinstalls from the PINNED url), NOT
        # _ensure_rocm_torch's Windows path (auto-detects the arch via hipinfo, wrong for a
        # private pin, and returns early on a headless box), so the branch must NOT call it.
        assert "_ensure_rocm_torch()" not in win_branch, (
            "the Windows/macOS-ARM final branch must not auto-detect the ROCm index via "
            "_ensure_rocm_torch; an explicit rocm/gfx pin is repaired from the pinned URL "
            "by _ensure_pinned_known_family_torch"
        )
        # _ensure_pinned_known_family_torch must own Windows rocm/gfx pins: gate ROCm via
        # the pin leaf and reinstall from the explicit pin, never an auto-detected index.
        _pinned_start = text.find("def _ensure_pinned_known_family_torch(")
        pinned_fn = text[_pinned_start : text.find("\ndef ", _pinned_start + 1)]
        assert "_is_win_rocm = IS_WINDOWS and _is_pip_rocm_family_leaf(leaf)" in pinned_fn, (
            "_ensure_pinned_known_family_torch must handle a Windows rocm/gfx pin from the "
            "explicit pin URL (round-7 item 2), gated on IS_WINDOWS + a pip rocm leaf"
        )
        assert '"--index-url",' in pinned_fn and "pin," in pinned_fn, (
            "_ensure_pinned_known_family_torch must reinstall the trio from the explicit "
            "pin URL (--index-url pin), not an auto-detected arch index"
        )

    def test_setup_ps1_forces_reinstall_when_marker_absent(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        # The unknown-family else branch must promote a null marker to an in-place
        # force-reinstall (PinChangedForceReinstall), NOT a wipe ($shouldRebuild).
        assert (
            "if ($null -eq $_markerMismatch) { $script:PinChangedForceReinstall = $true }" in text
        ), (
            "setup.ps1 must set PinChangedForceReinstall for an unknown-family pin on a "
            "marker-less venv so the torch block reinstalls from $PinnedTorchIndexUrl in place"
        )
        # Guard: the unknown-leaf branch must not trigger the venv wipe path.
        else_branch = text[text.find("PEP 503 mirror ending in /simple") :][:900]
        assert "$shouldRebuild = $true" not in else_branch, (
            "setup.ps1 unknown-family branch must repair in place (PinChangedForceReinstall), "
            "not wipe the venv ($shouldRebuild), which would strand a direct studio update"
        )


class TestPinnedIndexClearsUvEnvParity:
    """Every installer must neutralise the uv index env vars for a pinned torch
    install (#6898). uv treats the default index (--index-url / --default-index) as
    lowest priority, so an inherited UV_INDEX / UV_EXTRA_INDEX_URL mirror would win
    under uv's first-index strategy and pull torch from the wrong index -- after
    which the torch-index marker records a wheel index that was never used."""

    UV_VARS = ("UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX", "UV_EXTRA_INDEX_URL")

    def test_install_sh_clears_uv_index_vars(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        assert (
            "env -u UV_DEFAULT_INDEX -u UV_INDEX_URL -u UV_INDEX -u UV_EXTRA_INDEX_URL" in text
        ), "install.sh run_install_cmd must clear the uv index vars for --default-index installs"

    def test_install_ps1_clears_uv_index_vars(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        for var in self.UV_VARS:
            assert var in text, f"install.ps1 must clear {var} for pinned installs"

    def test_setup_ps1_clears_uv_index_vars(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        for var in self.UV_VARS:
            assert var in text, f"setup.ps1 must clear {var} for pinned installs"

    def test_stack_py_clears_uv_index_vars(self):
        text = STACK_PY.read_text(encoding = "utf-8")
        assert "_install_env_for_cmd" in text, (
            "install_python_stack.py must scrub inherited uv index vars for pinned "
            "installs via _install_env_for_cmd (parity with install.sh #6898)"
        )
        for var in self.UV_VARS:
            assert var in text, f"install_python_stack.py must clear {var} for pinned installs"

    def test_all_installers_clear_uv_torch_backend(self):
        """uv's torch backend redirects torch resolution to its own per-backend
        index even against an explicit pin, so every installer's pinned-install
        scrub must clear UV_TORCH_BACKEND too."""
        sh = INSTALL_SH.read_text(encoding = "utf-8")
        assert "-u UV_TORCH_BACKEND" in sh, "install.sh pinned scrub must clear UV_TORCH_BACKEND"
        for path in (INSTALL_PS1, SETUP_PS1):
            text = path.read_text(encoding = "utf-8")
            assert (
                "'UV_TORCH_BACKEND'" in text
            ), f"{path.name} pinned scrub must clear UV_TORCH_BACKEND"
        stack = STACK_PY.read_text(encoding = "utf-8")
        assert (
            '"UV_TORCH_BACKEND",' in stack
        ), "install_python_stack.py strip tuple must include UV_TORCH_BACKEND"

    def test_stack_py_strips_pip_extra_index_for_pip_fallback(self):
        """The pip fallback honours PIP_EXTRA_INDEX_URL (pip adds it IN ADDITION
        to --index-url), so the pinned-command scrub must strip it."""
        stack = STACK_PY.read_text(encoding = "utf-8")
        assert (
            '"PIP_EXTRA_INDEX_URL",' in stack
        ), "install_python_stack.py strip tuple must include PIP_EXTRA_INDEX_URL"

    def test_all_installers_scrub_find_links(self):
        """uv's --find-links (env UV_FIND_LINKS) adds candidate locations that can
        satisfy torch off a pinned index; every pinned-install scrub must clear it."""
        sh = INSTALL_SH.read_text(encoding = "utf-8")
        assert "-u UV_FIND_LINKS" in sh
        for path in (INSTALL_PS1, SETUP_PS1):
            assert "'UV_FIND_LINKS'" in path.read_text(encoding = "utf-8"), path.name
        stack = STACK_PY.read_text(encoding = "utf-8")
        assert '"UV_FIND_LINKS",' in stack and '"PIP_FIND_LINKS",' in stack

    def test_setup_ps1_scrub_covers_pip_fallback(self):
        """setup.ps1's Fast-Install must keep the scrub active through the pip
        fallback (pip honours PIP_EXTRA_INDEX_URL / PIP_FIND_LINKS in addition to
        --index-url); restoring the vars before the fallback reopens the hole."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        fi = text[text.find("function Fast-Install") :][:2500]
        assert "'PIP_EXTRA_INDEX_URL'" in fi and "'PIP_FIND_LINKS'" in fi
        # the pip fallback must sit INSIDE the try whose finally restores the vars
        assert fi.find("python -m pip install") < fi.find(
            "finally"
        ), "pip fallback must run before the scrub is restored"

    def test_all_installers_disable_uv_config_for_pinned_installs(self):
        """A DISCOVERED uv.toml / pyproject [tool.uv] outranks the CLI pin
        (verified with uv 0.10: [pip] torch-backend = "cpu" and a non-default
        [[index]] both resolve torch+cpu against an explicit --index-url /
        --default-index cu126 pin; UV_NO_CONFIG=1 restores the pin). Every
        installer's pinned scrub must set UV_NO_CONFIG=1 and drop UV_CONFIG_FILE."""
        sh = INSTALL_SH.read_text(encoding = "utf-8")
        assert "-u UV_CONFIG_FILE UV_NO_CONFIG=1" in sh, (
            "install.sh run_install_cmd must set UV_NO_CONFIG=1 and drop "
            "UV_CONFIG_FILE for --default-index installs"
        )
        for path in (INSTALL_PS1, SETUP_PS1):
            text = path.read_text(encoding = "utf-8")
            assert "'UV_CONFIG_FILE'" in text, f"{path.name} must drop UV_CONFIG_FILE"
            assert (
                "$env:UV_NO_CONFIG = '1'" in text
            ), f"{path.name} must set UV_NO_CONFIG=1 for pinned installs"
        stack = STACK_PY.read_text(encoding = "utf-8")
        assert (
            '"UV_CONFIG_FILE",' in stack
        ), "install_python_stack.py strip tuple must include UV_CONFIG_FILE"
        assert (
            'env["UV_NO_CONFIG"] = "1"' in stack
        ), "_install_env_for_cmd must set UV_NO_CONFIG=1 for pinned installs"

    def test_pip_fallbacks_disable_pip_config_files(self):
        """The pip FALLBACK (uv missing/failed) honours user/site pip config files
        even with the PIP_* env vars stripped: `pip config set
        global.extra-index-url` still adds indexes to a pinned install. pip loads
        NO configuration files when PIP_CONFIG_FILE is the platform devnull, so
        the two installers that HAVE a pip fallback (install_python_stack.py and
        setup.ps1's Fast-Install) must set it in their pinned scrub. install.sh
        and install.ps1 are uv-only (no python -m pip fallback) and need no
        equivalent."""
        stack = STACK_PY.read_text(encoding = "utf-8")
        assert 'env["PIP_CONFIG_FILE"] = os.devnull' in stack, (
            "_install_env_for_cmd must point PIP_CONFIG_FILE at os.devnull for "
            "pinned installs (pip fallback isolation)"
        )
        setup = SETUP_PS1.read_text(encoding = "utf-8")
        assert "$env:PIP_CONFIG_FILE = 'nul'" in setup, (
            "setup.ps1 Fast-Install pinned scrub must point PIP_CONFIG_FILE at nul "
            "(Windows devnull) so the pip fallback ignores user/site pip config"
        )
        assert (
            "'PIP_CONFIG_FILE'" in setup
        ), "setup.ps1 must save/restore PIP_CONFIG_FILE around the pinned scrub"

    def test_setup_ps1_bounds_unknown_leaf_pinned_torch(self):
        """A first-time/changed unknown-leaf custom pin routes through setup.ps1's
        CUDA branch; install.ps1's fresh pinned install, install.sh, and the Python
        verbatim path bound the WHOLE trio, so the Windows update path must too -- a
        private mirror serving newer torch OR newer companions must not lift the venv
        above the supported range while the marker records the pin as applied."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        # The custom-leaf branch bounds torch AND both companions (parity with
        # _CUSTOM_INDEX_TORCH_PKG_SPEC), gated on a non-cu-family leaf.
        for spec in (
            '$cudaTorchSpec = "torch>=2.4,<2.11.0"',
            '$cudaVisionSpec = "torchvision>=0.19,<0.26.0"',
            '$cudaAudioSpec = "torchaudio>=2.4,<2.11.0"',
        ):
            assert spec in text, f"setup.ps1 must bound the custom-leaf trio: {spec}"
        assert (
            "if ($TorchIndexPinned -and -not (Test-CudaFamilyLeaf $CuTag)) {" in text
        ), "the custom-leaf trio bounds must be gated on a pinned non-cu-family leaf"
        assert (
            "Fast-Install $cudaTorchSpec $cudaVisionSpec $cudaAudioSpec" in text
        ), "setup.ps1's CUDA branch must install via the bounded spec variables"

    def test_setup_ps1_stale_check_requires_rocm_digit(self):
        """The marker stale check must use the same EXACT rocm/gfx gate as the install
        selection (Test-PipRocmFamilyLeaf), or a custom rocm-* / suffixed rocm7.2-private
        leaf is stale-compared as a family and force-reinstalls on every studio update."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        stale = text[text.find("Get-RocmPinStaleTags -PinLeaf") - 2500 :][:2500]
        assert (
            "Test-PipRocmFamilyLeaf" in stale
        ), "setup.ps1 stale check must gate rocm leaves via the exact Test-PipRocmFamilyLeaf"
        assert (
            stale.count("-like 'rocm*'") == 0
        ), "setup.ps1 stale check must not use a bare -like 'rocm*' glob"
        assert (
            "-match '^rocm\\d'" not in stale
        ), "setup.ps1 stale check must not use an unanchored -match '^rocm\\d'"


class TestIndexPathSlashTrimParity:
    """Every installer must trim trailing PATH slashes only on the verbatim
    UNSLOTH_TORCH_INDEX_URL override, preserving a ?query/#fragment token: a whole-URL
    strip corrupts a base64 token ending in "/", a single strip leaves a double-slash leaf
    empty. The helper must be DEFINED and WIRED into the override return in all four."""

    def test_helper_defined_in_all_installers(self):
        assert "def _trim_index_path_slashes(" in STACK_PY.read_text(encoding = "utf-8")
        assert "_trim_index_path_slashes()" in INSTALL_SH.read_text(encoding = "utf-8")
        assert "function Trim-IndexPathSlashes" in INSTALL_PS1.read_text(encoding = "utf-8")
        assert "function Trim-IndexPathSlashes" in SETUP_PS1.read_text(encoding = "utf-8")

    def test_helper_wired_into_override_in_all_installers(self):
        assert "_trim_index_path_slashes(url)" in STACK_PY.read_text(encoding = "utf-8")
        assert '_url=$(_trim_index_path_slashes "$_url")' in INSTALL_SH.read_text(encoding = "utf-8")
        assert "Trim-IndexPathSlashes $env:UNSLOTH_TORCH_INDEX_URL" in INSTALL_PS1.read_text(
            encoding = "utf-8"
        )
        assert "Trim-IndexPathSlashes $env:UNSLOTH_TORCH_INDEX_URL" in SETUP_PS1.read_text(
            encoding = "utf-8"
        )


class TestInstallOutputRedactionParity:
    """uv/pip failure text embeds the failing --index-url verbatim, so a captured install
    log dumped on error can leak a user:token@ or ?token= secret. Every installer must
    DEFINE a redaction helper and WIRE it into the captured-output print path."""

    def test_helper_defined_in_all_installers(self):
        assert "def _redact_install_output(" in STACK_PY.read_text(encoding = "utf-8")
        assert "_redact_install_output()" in INSTALL_SH.read_text(encoding = "utf-8")
        assert "function Redact-InstallOutput" in INSTALL_PS1.read_text(encoding = "utf-8")
        assert "function Redact-InstallOutput" in SETUP_PS1.read_text(encoding = "utf-8")

    def test_helper_wired_into_failure_print(self):
        # install.sh dumps the captured log through the redactor on failure.
        assert '_redact_install_output "$_log"' in INSTALL_SH.read_text(encoding = "utf-8")
        # Both ps1 installers redact the captured $output before Write-Host on non-zero exit.
        assert (
            "Write-Host (Redact-InstallOutput $output) -ForegroundColor Red"
            in INSTALL_PS1.read_text(encoding = "utf-8")
        )
        assert (
            "Write-Host (Redact-InstallOutput $output) -ForegroundColor Red"
            in SETUP_PS1.read_text(encoding = "utf-8")
        )
        # Python redacts the captured stdout before printing.
        assert "_redact_install_output(" in STACK_PY.read_text(encoding = "utf-8")


class TestPipNoIndexScrubParity:
    """The plain-pip fallback honours PIP_*: PIP_NO_INDEX=1 makes it ignore ALL indexes
    (defeating the pinned --index-url) and PIP_INDEX_URL replaces the pin. The two installers
    that HAVE a plain-pip fallback (Python + setup.ps1) must scrub both for a pinned install.
    install.sh / install.ps1 are uv-only (--default-index), which ignores pip config/env."""

    def test_python_scrubs_pip_no_index_and_pip_index_url(self):
        text = STACK_PY.read_text(encoding = "utf-8")
        assert '"PIP_NO_INDEX"' in text
        assert '"PIP_INDEX_URL"' in text

    def test_setup_ps1_scrubs_pip_no_index_and_pip_index_url(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "'PIP_NO_INDEX'" in text
        assert "'PIP_INDEX_URL'" in text
