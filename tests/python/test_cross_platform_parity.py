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


def _fallback_range(lines):
    """The half-open line range of install.sh's "GPU detection failed" fallback.

    The end is the `fi` that closes the branch, which means counting nesting
    rather than taking the first `fi` that appears. Any `if` inside the branch
    contributes one, and so does a `case`: it closes with `esac`, so a version
    that only counted `if`/`fi` would still be off by one from the `fi` of an
    `if` nested inside a case arm.
    """
    start = next(i for i, line in enumerate(lines) if "GPU detection failed" in line)
    depth = 0
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if re.match(r"^(if|case)\b", stripped):
            depth += 1
        elif stripped in ("fi", "esac") or stripped.startswith(("fi ", "esac ")):
            if depth == 0:
                return range(start, i + 1)
            depth -= 1
    raise AssertionError("install.sh's GPU-detection fallback branch is never closed")


class TestNoTorchBackendAutoInInstallSh:
    """install.sh primary paths must not use --torch-backend=auto (only the fallback else-branch may)."""

    def test_no_torch_backend_auto_outside_fallback(self):
        lines = INSTALL_SH.read_text(encoding = "utf-8").splitlines()
        fallback_range = _fallback_range(lines)

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

    def test_the_fallback_range_reaches_the_end_of_the_branch(self):
        """A range that stops early makes the assertion above fire on correct code.

        It did. #8670 put a `case` with a nested `if`/`fi` in this branch to pick
        the desktop install spec, and the previous "first `fi` after the comment"
        scan then ended the block four lines short of the install call it exists
        to permit -- reporting the fallback's own line as a primary path.
        """
        lines = INSTALL_SH.read_text(encoding = "utf-8").splitlines()
        block = _fallback_range(lines)
        body = "\n".join(lines[block.start : block.stop])
        # Both arms of the branch, so neither an early stop nor a runaway passes.
        assert "STUDIO_LOCAL_INSTALL" in body, "the fallback block stops before its own first if"
        assert body.count("--torch-backend=auto") == 2, (
            "the fallback runs --torch-backend=auto once per arm; the detected block "
            f"contains {body.count('--torch-backend=auto')}"
        )
        # And it must not have swallowed the rest of the file.
        assert "_installed_package_version" not in body, "the fallback block ran past its `fi`"

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


class TestPreTuringCapParity:
    """Every wheel-selection site caps cu128/cu130 on a pre-Turing host (issue #7765).

    PyTorch 2.11 builds those families for sm_75 and newer, so a Maxwell/Pascal/Volta
    box needs cu126 -- both for torch itself and for the CUDA 12 runtime that gets it
    a llama.cpp GGUF bundle. Four scripts pick the family; none may be left behind.
    """

    # (file, call spelling, selection function that must invoke it, its end marker). The
    # spelling carries the first argument, so a prose mention cannot satisfy the assertion.
    _SITES = (
        (INSTALL_SH, '_cap_cuda_family_for_pre_turing "', "get_torch_index_url() {", "\n}"),
        (
            INSTALL_PS1,
            "Get-CudaFamilyCappedForPreTuring $",
            "function Get-TorchIndexUrl",
            "\n    }",
        ),
        (SETUP_PS1, "Get-CudaFamilyCappedForPreTuring $", "function Get-PytorchCudaTag", "\n}"),
        (
            STACK_PY,
            "_cap_cuda_family_for_pre_turing(",
            "def _detect_cuda_torch_index_url",
            "\ndef ",
        ),
    )

    def test_cu126_span_agrees_across_the_python_modules(self):
        # Neither module imports the other (the installer runs before dependencies
        # exist), so assert the shared span here rather than let it drift silently.
        span = "_CU126_SM_RANGE = (50, 90)"
        for path in (STACK_PY, REPO_ROOT / "studio" / "install_llama_prebuilt.py"):
            assert span in path.read_text(encoding = "utf-8"), f"{path.name} lost {span}"

    @pytest.mark.parametrize("path,call,start,end", _SITES)
    def test_selection_function_applies_the_cap(self, path, call, start, end):
        text = path.read_text(encoding = "utf-8")
        assert start in text, f"{path.name} no longer defines {start!r}"
        body = text.split(start, 1)[1].split(end, 1)[0]
        assert call in body, f"{path.name}'s selection function never applies {call!r}"


# A ladder rung names its family either as an index-URL suffix ("$base/cu128") or as a
# variable a later step can still cap ("_cuda_tag=cu128"), so accept both spellings.
_CUDA_LEAF_RE = r"""[/=]\s*["']?(cu\d+|cpu)"""


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
                m = re.search(_CUDA_LEAF_RE, line)
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
                    m = re.search(_CUDA_LEAF_RE, line)
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

    def test_setup_ps1_stale_check_gates_rocm_on_supported_arch(self):
        # The stale check must expect ROCm torch only for arches the install path maps to a
        # repo.amd.com index; expecting "rocm" for an unmapped arch marks a good CPU venv stale.
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "_rocmWheelArches" in text, (
            "setup.ps1 stale check should restrict the ROCm expected-tag to the "
            "supported gfx wheel arches"
        )


class TestGfx211AllowlistParity:
    """The gfx per-arch 2.11-floor leaves must be the SAME set in every installer
    and its stale/mismatch check. When they diverged, a pinned gfx110X-all /
    gfx90a / gfx908 wheel (<2.11) was force-reinstalled every update.

    Each test extracts the set each installer actually holds and compares it
    against EXPECTED, rather than matching one hardcoded ordering. Order and
    spacing are free; membership is not. The earlier literal-string form had to
    be edited in four places whenever a leaf was added, which is how adding
    gfx1152 (Krackan Point) turned this class red without any installer
    actually disagreeing with another."""

    EXPECTED = {"gfx120x-all", "gfx1151", "gfx1150", "gfx1152"}

    @staticmethod
    def _leaves(blob: str) -> set[str]:
        """The gfx leaves named in an allowlist literal, quoting-agnostic."""
        return set(re.findall(r"gfx[0-9a-z-]+", blob.lower()))

    def test_install_sh_allowlist(self):
        text = INSTALL_SH.read_text(encoding = "utf-8").lower()
        # install.sh: the TORCH_CONSTRAINT case (rocm7.2|gfx...|gfx...).
        m = re.search(r"^\s*(rocm7\.2\|[a-z0-9|.\-]*)\)", text, re.MULTILINE)
        assert m, "install.sh gfx-2.11 allowlist case not found / changed"
        assert self._leaves(m.group(1)) == self.EXPECTED, (
            f"install.sh gfx-2.11 allowlist is {sorted(self._leaves(m.group(1)))}, "
            f"expected {sorted(self.EXPECTED)}"
        )

    def test_install_ps1_allowlist(self):
        text = INSTALL_PS1.read_text(encoding = "utf-8").lower()
        m = re.search(r"\$_pingfx211\s*=\s*@\(([^)]*)\)", text)
        assert m, "install.ps1 $_pinGfx211 allowlist not found / changed"
        assert self._leaves(m.group(1)) == self.EXPECTED, (
            f"install.ps1 $_pinGfx211 is {sorted(self._leaves(m.group(1)))}, "
            f"expected {sorted(self.EXPECTED)}"
        )

    def test_setup_ps1_defines_single_allowlist_helper(self):
        # setup.ps1 must define the allowlist once (Test-RocmGfx211Leaf) and reuse it, so
        # the stale check and install spec can't disagree.
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert (
            "function Test-RocmGfx211Leaf" in text
        ), "setup.ps1 should define a single Test-RocmGfx211Leaf allowlist helper"
        m = re.search(r"function test-rocmgfx211leaf[\s\S]{0,400}?@\(([^)]*)\)", text.lower())
        assert m, "Test-RocmGfx211Leaf should hold the gfx-2.11 allowlist"
        assert self._leaves(m.group(1)) == self.EXPECTED, (
            f"Test-RocmGfx211Leaf holds {sorted(self._leaves(m.group(1)))}, "
            f"expected {sorted(self.EXPECTED)}"
        )
        assert "$_pinGfx211 = Test-RocmGfx211Leaf" in text, (
            "setup.ps1 install-spec path should reuse Test-RocmGfx211Leaf, not "
            "re-hardcode the allowlist (they must not diverge)"
        )

    def test_stack_py_allowlist(self):
        text = STACK_PY.read_text(encoding = "utf-8").lower()
        m = re.search(r"_rocm_gfx_torch211_leaves[^=]*=\s*frozenset\(\s*\{([^}]*)\}", text)
        assert m, "install_python_stack.py _ROCM_GFX_TORCH211_LEAVES not found / changed"
        assert self._leaves(m.group(1)) == self.EXPECTED, (
            f"_ROCM_GFX_TORCH211_LEAVES is {sorted(self._leaves(m.group(1)))}, "
            f"expected {sorted(self.EXPECTED)}"
        )


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
        # Brand CUDA only on cu[0-9]*; a bare catch-all *) -> cuda would mis-brand
        # /current, /custom pins and skip ROCm repair on AMD hosts.
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


class TestKnown211SetParity:
    """The KNOWN-2.11 rocm/gfx set must be identical across all four installers:
    exactly {rocm7.2} plus TestGfx211AllowlistParity.EXPECTED.
    rocm7.3 / torch 2.12 do not exist, so no side may floor them speculatively."""

    def test_install_sh_known_211_leaf_is_rocm72_and_gfx_allowlist(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        # The 2.11 floor case matches exactly rocm7.2 + the gfx allowlist, in
        # any order: it is the same set as TestGfx211AllowlistParity.EXPECTED,
        # asserted here so the rocm-version half cannot drift on its own.
        m = re.search(r"^\s*(rocm7\.2\|[a-zA-Z0-9|.\-]*)\)", text, re.MULTILINE)
        assert m, "install.sh 2.11 floor case (rocm7.2|gfx...) not found / changed"
        alternatives = set(m.group(1).lower().split("|"))
        assert alternatives == {"rocm7.2"} | TestGfx211AllowlistParity.EXPECTED, (
            f"install.sh 2.11 floor is {sorted(alternatives)}, expected "
            f"{sorted({'rocm7.2'} | TestGfx211AllowlistParity.EXPECTED)}"
        )
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
        """install.ps1's pinned-torch install must bound the whole trio on EVERY
        index with the default torch 2.11 line (<2.12 trio, matching install.sh's
        ceiling-composed default and _CUDA_TORCH_PKG_SPEC): torchaudio 2.11
        dropped its exact torch pin from the wheel metadata, so a bare companion
        beside a capped torch can resolve a mismatched build."""
        text = INSTALL_PS1.read_text(encoding = "utf-8")
        assert (
            '$_pinTorchSpec = "torch>=2.4,<2.12.0"' in text
        ), "install.ps1 default install must use the torch 2.11 line (<2.12.0)"
        assert (
            '$_pinVisionSpec = "torchvision>=0.19,<0.27.0"' in text
        ), "install.ps1 must pair torchvision <0.27.0 with torch <2.12"
        assert (
            '$_pinAudioSpec = "torchaudio>=2.4,<2.12.0"' in text
        ), "install.ps1 must pair torchaudio <2.12.0 with torch <2.12"
        assert (
            "$_pinCuLeaf" not in text
        ), "install.ps1 must bound companions on every index (no cu-family exemption)"
        # No stale 2.10-line DEFAULT remains. Checked against the default trio's own
        # assignments rather than as a blanket "<2.11.0 appears nowhere": Get-XpuTorchSpecs
        # deliberately keeps a curated sub-2.11 cap for the whl/xpu index (mirroring
        # install.sh's xpu case arm), and that is not the default. The XPU CPU fallback is
        # NOT part of that carve-out -- it installs from the plain whl/cpu index, the same
        # one the ROCm-to-CPU fallback uses, so it moved to the 2.11 line with the rest.
        for _stale in (
            '$_pinTorchSpec = "torch>=2.4,<2.11.0"',
            '$_pinVisionSpec = "torchvision>=0.19,<0.26.0"',
            '$_pinAudioSpec = "torchaudio>=2.4,<2.11.0"',
            '$_torchSpecs = @("torch>=2.4,<2.11.0"',
        ):
            assert (
                _stale not in text
            ), f"install.ps1 must not retain the <2.11.0 default torch line: {_stale}"
        # The bounded trio must actually be built and passed to the install command.
        # Specs are splatted, so check both halves: the list is built, and it is passed.
        assert (
            "$_torchSpecs = @($_pinTorchSpec, $_pinVisionSpec, $_pinAudioSpec)" in text
        ), "install.ps1 pinned install must build the bounded trio spec list"
        assert (
            "@_torchSpecs --default-index $TorchIndexUrl" in text
        ), "install.ps1 pinned install must pass the bounded trio specs to uv"

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
        # gfx needs a following digit: gfx-private / gfxfoo are custom verbatim pins.
        assert re.search(
            r'case "\$1" in\n\s*gfx\[0-9\]\*\) return 0', text
        ), "install.sh _is_pip_rocm_family_leaf must treat only gfx<digit>* as a family"
        assert not re.search(
            r'case "\$1" in\n\s*gfx\*\) return 0', text
        ), "install.sh _is_pip_rocm_family_leaf must not family-match a bare gfx* glob"

    def test_stack_py_pip_rocm_family_requires_digit(self):
        text = STACK_PY.read_text(encoding = "utf-8")
        assert re.search(
            r'fullmatch\(r"rocm\\d\+\(\?:\\\.\\d\+\)\?", leaf\)', text
        ), "install_python_stack.py _is_pip_rocm_family_leaf must fullmatch rocm\\d+(?:\\.\\d+)?"
        # The unanchored prefix must be gone from the family/flavor gates.
        assert (
            're.match(r"^rocm\\d"' not in text
        ), "install_python_stack.py must not gate a family on an unanchored re.match(^rocm\\d)"

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


class TestPinnedIndexClearsUvEnvParity:
    """Every installer must neutralise the uv index env vars for a pinned torch
    install (#6898). uv treats the default index (--index-url / --default-index) as
    lowest priority, so an inherited UV_INDEX / UV_EXTRA_INDEX_URL mirror would win
    under uv's first-index strategy and pull torch from the wrong index -- after
    which the pinned wheel index is silently never used."""

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

    def test_windows_installers_probe_uv_before_replacing_an_incumbent(self):
        """A host can have a working older uv while AppLocker, WDAC or endpoint
        protection refuses the one we just downloaded. Both PowerShell installers must
        run the extracted uv.exe where it landed BEFORE anything at the destination is
        touched, and must restore the incumbent if the published copy will not run."""
        for path, probe in (
            (INSTALL_PS1, "Get-UvExecutableVerdict"),
            (SETUP_PS1, "Get-SetupUvExecutableVerdict"),
        ):
            text = path.read_text(encoding = "utf-8")
            assert f"function {probe}" in text, f"{path.name} must define {probe}"
            # WaitForExit takes a timeout: an unbounded wait on a freshly downloaded
            # binary is exactly how an unattended install hangs.
            assert "WaitForExit(20000)" in text, f"{path.name}'s uv probe must bound its wait"
            probe_at = text.index(f"({probe} -Path $stagedUv)")
            # Tri-state, not a boolean. A launch that throws or a wait that times out got no
            # verdict, and treating that as a broken binary turned three clean-machine CI legs
            # into hard install failures: Start-Process -NoNewWindow with redirected streams
            # does not behave in a Windows container or on arm64 as it does on a desktop. Only
            # the binary answering non-zero may block the install.
            body = text.split(f"function {probe}", 1)[1].split("\n    }\n", 1)[0]
            # An EMPTY exit code is no verdict either. WaitForExit(ms) can return before the
            # code is cached, which is how arm64 and the Windows containers reported "exited ."
            # and had a working uv read as broken.
            assert (
                "try { $proc.WaitForExit() } catch {}" in body
            ), f"{path.name} must settle the exit code before reading it"
            assert (
                '$null -eq $code -or "$code" -eq ""' in body
            ), f"{path.name} must treat a missing exit code as inconclusive"
            assert (
                body.count('return "unknown"') == 3
            ), f"{path.name}: a launch failure and a timeout must both be inconclusive"
            assert (
                'return "failed"' in body and 'return "ok"' in body
            ), f"{path.name}'s probe must report a real answer as well"
            assert (
                f'({probe} -Path $stagedUv) -eq "failed"' in text
            ), f"{path.name} must gate only on a failed verdict"
            copy_at = text.index("Copy-Item -LiteralPath $src -Destination $dst -Force")
            assert probe_at < copy_at, (
                f"{path.name} must probe the extracted uv.exe before copying over the "
                "destination"
            )
            # The publish is not a transaction: a locked or ACL-denied destination fails the
            # install rather than being skipped, which is what the caller's fallback is for.
            assert (
                "Copy-Item -LiteralPath $src -Destination $dst -Force -ErrorAction Stop" in text
            ), f"{path.name} must copy each executable under -ErrorAction Stop"

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
        above the supported range under the pin."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        # The custom-leaf branch bounds torch AND both companions (parity with the
        # other installers' custom-pin trio bounds), gated on a non-cu-family leaf.
        for spec in (
            '$cudaTorchSpec = "torch>=2.4,<2.12.0"',
            '$cudaVisionSpec = "torchvision>=0.19,<0.27.0"',
            '$cudaAudioSpec = "torchaudio>=2.4,<2.12.0"',
        ):
            assert spec in text, f"setup.ps1 must bound the custom-leaf trio: {spec}"
        assert (
            "if ($TorchIndexPinned -and -not (Test-CudaFamilyLeaf $CuTag)) {" in text
        ), "the custom-leaf trio bounds must be gated on a pinned non-cu-family leaf"
        # Specs are splatted, so check both halves: the list is built, and it is passed.
        assert (
            "$_cudaTrio = @($cudaTorchSpec, $cudaVisionSpec, $cudaAudioSpec)" in text
        ), "setup.ps1's CUDA branch must build the trio from the bounded spec variables"
        assert (
            "Fast-Install @_cudaTrio @cudaForce" in text
        ), "setup.ps1's CUDA branch must install the trio it built"

    def test_setup_ps1_bounds_pinned_cpu_torch(self):
        """setup.ps1's CPU branch must bound the trio under an explicit pin (parity with
        _CPU_TORCH_PKG_SPEC): the /cpu index serves newer torch, and _ensure_cpu_torch
        keeps any CPU build, so a bare pinned trio could land an unsupported version.
        An unpinned CPU host keeps the bare trio (pre-pin behavior unchanged)."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        for spec in (
            '$cpuTorchSpec  = "torch>=2.4,<2.12.0"',
            '$cpuVisionSpec = "torchvision>=0.19,<0.27.0"',
            '$cpuAudioSpec  = "torchaudio>=2.4,<2.12.0"',
        ):
            assert spec in text, f"setup.ps1 must bound the pinned CPU trio: {spec}"
        assert (
            "if ($TorchIndexPinned) {" in text
        ), "the CPU trio bounds must be gated on an explicit pin"
        assert (
            "$_torchTrio = @($cpuTorchSpec, $cpuVisionSpec, $cpuAudioSpec)" in text
        ), "setup.ps1's CPU branch must build the trio from the spec variables"
        assert (
            "Fast-Install @_torchTrio @cpuForce" in text
        ), "setup.ps1's CPU branch must install the trio it built"
        # The ceilings mirror the Python repair spec exactly.
        stack = STACK_PY.read_text(encoding = "utf-8")
        spec_block = re.search(r"_CUDA_TORCH_PKG_SPEC[^(]*\(\s*(.*?)\)", stack, re.DOTALL)
        assert spec_block and '"torch>=2.4,<2.12.0"' in spec_block.group(1), (
            "_CPU_TORCH_PKG_SPEC (via _CUDA_TORCH_PKG_SPEC) must keep the torch<2.12 "
            "ceiling the setup.ps1 pinned CPU branch mirrors"
        )

    def test_setup_ps1_stale_check_requires_rocm_digit(self):
        """The stale-venv check must use the same EXACT rocm/gfx gate as the install
        selection (Test-PipRocmFamilyLeaf), or a custom rocm-* / suffixed rocm7.2-private
        leaf is stale-compared as a family and force-reinstalls on every studio update."""
        text = SETUP_PS1.read_text(encoding = "utf-8")
        anchor = text.find("$_pinLeaf = Get-TorchIndexLeaf $_pinnedIdx")
        assert anchor >= 0, "setup.ps1 stale check must classify the pinned leaf"
        stale = text[anchor:][:2500]
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
        # Both ps1 installers redact the captured $output before printing it on a
        # non-zero exit. Write-StudioLine is the UTF-8 stdout sink both now use.
        assert (
            "Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Red"
            in INSTALL_PS1.read_text(encoding = "utf-8")
        )
        assert (
            "Write-StudioLine (Redact-InstallOutput $output) -ForegroundColor Red"
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


class TestNoTorchPersistenceParity:
    """No-torch mode must outlive the process that requested it.

    install.sh / install.ps1 export UNSLOTH_NO_TORCH for their own run only.
    `unsloth studio update` exports nothing, so both the PowerShell setup and the
    shared Python stack have to recover the mode from the install manifest, or an
    update reinstalls PyTorch into a GGUF-only venv. On Windows it is worse than
    cosmetic: setup.ps1 reads the missing torch as a stale venv and tries to delete
    the venv it is itself running out of, which fails on a locked python.exe."""

    def test_the_stack_records_the_mode_it_installed(self):
        text = STACK_PY.read_text(encoding = "utf-8")
        assert "no_torch = NO_TORCH" in text
        assert "install_manifest.recorded_no_torch()" in text
        # Written after the manifest is dropped and before the dependency pass, so
        # a pass killed part-way still leaves the mode recorded somewhere.
        assert text.index("install_manifest.set_no_torch_marker(NO_TORCH)") > text.index(
            "if not install_manifest.remove_manifest():"
        )

    def test_both_sides_use_the_same_marker_filename(self):
        manifest = (REPO_ROOT / "studio" / "install_manifest.py").read_text(encoding = "utf-8")
        assert 'NO_TORCH_MARKER = ".unsloth-no-torch"' in manifest
        assert '$NoTorchMarker = ".unsloth-no-torch"' in SETUP_PS1.read_text(encoding = "utf-8")

    def test_setup_ps1_recovers_the_mode_when_no_env_var_is_exported(self):
        text = SETUP_PS1.read_text(encoding = "utf-8")
        assert "function Get-PersistedNoTorch" in text
        assert "function Set-PersistedNoTorch" in text
        # setup.ps1 drops the manifest before running install_python_stack.py, so
        # the resolved answer has to be handed down through the environment.
        assert text.index("Get-PersistedNoTorch -VenvPath $VenvDir") < text.index(
            '$env:UNSLOTH_NO_TORCH = if ($NoTorchMode) { "true" } else { "false" }'
        )

    def test_both_sides_accept_the_same_spellings(self):
        # install.ps1 / install.sh accept 1|true|yes|on; the two consumers must not
        # be narrower, or a value one layer honours another silently ignores.
        assert "'^\\s*(?i:true|1|yes|on)\\s*$'" in SETUP_PS1.read_text(encoding = "utf-8")
        manifest = (REPO_ROOT / "studio" / "install_manifest.py").read_text(encoding = "utf-8")
        assert 'NO_TORCH_TRUTHY: Tuple[str, ...] = ("1", "true", "yes", "on")' in manifest
        assert "install_manifest.NO_TORCH_TRUTHY" in STACK_PY.read_text(encoding = "utf-8")


class TestAmdBnbFloorParity:
    """bitsandbytes <= 0.49.2 NaNs at 4-bit decode shape on every AMD GPU; the ROCm
    4-bit GEMV fix (bnb #1887) first ships on PyPI in 0.50.0. The `amd` extra,
    install.sh and the Unsloth stack resolve bitsandbytes independently, so all three
    must carry the same floor or an unreachable pre-release wheel silently reinstates
    the broken range."""

    FLOOR = "0.50.0"
    PYPROJECT = REPO_ROOT / "pyproject.toml"

    def test_amd_extra_floor(self):
        text = self.PYPROJECT.read_text(encoding = "utf-8")
        amd = re.search(r"^amd = \[(.*?)^\]", text, re.S | re.M)
        assert amd, "pyproject.toml must define an `amd` extra"
        specs = re.findall(r'"(bitsandbytes[^"]*)"', amd.group(1))
        assert specs, "the amd extra must pin bitsandbytes"
        for spec in specs:
            assert spec.startswith(
                f"bitsandbytes>={self.FLOOR}"
            ), f"amd extra bitsandbytes floor must be >={self.FLOOR}, got {spec!r}"

    def test_install_sh_pypi_fallback_floor(self):
        text = INSTALL_SH.read_text(encoding = "utf-8")
        assert (
            f'_BNB_ROCM_PYPI_FALLBACK="bitsandbytes>={self.FLOOR}"' in text
        ), f"install.sh _install_bnb_rocm PyPI fallback must floor at {self.FLOOR}"

    def test_stack_py_pypi_fallback_floor(self):
        text = STACK_PY.read_text(encoding = "utf-8")
        assert (
            f'_BNB_ROCM_PYPI_FALLBACK = "bitsandbytes>={self.FLOOR}"' in text
        ), f"install_python_stack.py PyPI fallback must floor at {self.FLOOR}"

    def test_no_installer_still_allows_the_broken_range(self):
        for path in (INSTALL_SH, INSTALL_PS1, SETUP_PS1, STACK_PY, self.PYPROJECT):
            text = path.read_text(encoding = "utf-8")
            for line in text.splitlines():
                if "bitsandbytes>=0.49" in line and not line.lstrip().startswith(("#", "//")):
                    raise AssertionError(
                        f"{path.name} still floors bitsandbytes in the broken ROCm range: {line.strip()!r}"
                    )

    def test_fallback_is_not_reported_as_broken(self):
        """The fallback now installs the first fixed release, so neither installer
        may still call 4-bit decode broken on ROCm."""
        for path in (INSTALL_SH, STACK_PY):
            text = path.read_text(encoding = "utf-8")
            assert (
                "4-bit decode broken on ROCm" not in text
            ), f"{path.name} still reports the repaired PyPI fallback as broken"
            assert (
                "4-bit decode will be broken on ROCm" not in text
            ), f"{path.name} still reports the repaired PyPI fallback as broken"

    def test_aarch64_is_not_told_it_has_a_rocm_backend(self):
        """bitsandbytes ships no ROCm kernels in its aarch64 wheel at any version, so
        neither installer may hand aarch64 the x86_64 "carries the ROCm 4-bit fix"
        message, and both must warn that 4-bit needs a source build there."""
        sh = INSTALL_SH.read_text(encoding = "utf-8")
        assert "_bnb_rocm_arch_has_binary()" in sh
        assert "_warn_bnb_no_rocm_binary()" in sh
        assert (
            sh.count("_warn_bnb_no_rocm_binary\n") >= 2
        ), "install.sh must warn on aarch64 after both the pre-release and the fallback install"
        py = STACK_PY.read_text(encoding = "utf-8")
        assert "def _bnb_rocm_arch_has_binary(" in py
        assert "_bnb_rocm_arch_has_binary()" in py
        for text, name in ((sh, "install.sh"), (py, "install_python_stack.py")):
            assert (
                "4-bit QLoRA needs a source build" in text
            ), f"{name} must tell aarch64 users 4-bit needs a source build"
