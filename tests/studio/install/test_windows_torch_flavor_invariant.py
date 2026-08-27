# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Structural cover for the Windows torch-flavor invariant on the update path.

The behavioural tests for _ensure_expected_torch_flavor live in test_cuda_repair.py.
This file asserts the parts that are not a Python call: that setup.ps1 hands the flavor
over before it invokes the stack, that setup.ps1 no longer wipes a healthy cu* venv when
nvidia-smi fails to answer, that the repair specs and the mismatch line stay identical to
install.ps1's, that the manifest round-trips the flavor, and that none of it reaches the
Linux/macOS branch of install_python_stack(). Source/AST only -- no Windows required."""

import ast
import importlib.util
import json
import re
import sys
import textwrap
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]

_SETUP_PS1 = PACKAGE_ROOT / "studio" / "setup.ps1"
_INSTALL_PS1 = PACKAGE_ROOT / "install.ps1"
_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"

_SETUP_SRC = _SETUP_PS1.read_text(encoding = "utf-8")
_INSTALL_SRC = _INSTALL_PS1.read_text(encoding = "utf-8")
_STACK_SRC = _STACK_PATH.read_text(encoding = "utf-8")

_MANIFEST_SPEC = importlib.util.spec_from_file_location(
    "studio_install_manifest_flavor", PACKAGE_ROOT / "studio" / "install_manifest.py"
)
assert _MANIFEST_SPEC is not None and _MANIFEST_SPEC.loader is not None
install_manifest = importlib.util.module_from_spec(_MANIFEST_SPEC)
sys.modules[_MANIFEST_SPEC.name] = install_manifest
_MANIFEST_SPEC.loader.exec_module(install_manifest)


def _line_of(source: str, needle: str) -> int:
    """1-based line number of the first line containing `needle`."""
    for number, line in enumerate(source.splitlines(), start = 1):
        if needle in line:
            return number
    raise AssertionError(f"not found in source: {needle!r}")


# ── setup.ps1: the venv wipe ─────────────────────────────────────────────────


class TestSetupPs1NoWipeEscape:
    """A direct `studio update` has no rollback copy -- only install.ps1 makes one -- so a
    wipe there is unrecoverable. Every way the bounded nvidia-smi probe can come back
    empty on a working NVIDIA box collapses the expected tag to "cpu", and a healthy cu124
    venv then reads as stale."""

    def test_the_escape_sits_ahead_of_the_wipe(self):
        escape = _line_of(_SETUP_SRC, "nvidia-smi did not answer, but this venv holds a")
        wipe = _line_of(_SETUP_SRC, "Remove-Item -LiteralPath $VenvDir -Recurse -Force")
        stale = _line_of(_SETUP_SRC, "Stale venv detected ($reason) -- rebuilding...")
        assert escape < stale < wipe, (
            "the no-wipe escape must be evaluated before the stale-venv branch that deletes "
            f"the venv (escape={escape}, stale={stale}, wipe={wipe})"
        )

    def test_the_escape_cancels_the_rebuild(self):
        body = _SETUP_SRC[
            _SETUP_SRC.index("nvidia-smi did not answer, but this venv holds a") :
        ][:1200]
        assert "$shouldRebuild = $false" in body
        # Keeping the wheel is only half the job: the index selection re-runs the same
        # rescan and would route the pass to the CPU index without this.
        assert "$script:PreservedInstallerTorchTag = $installedTorchTag" in body

    def test_the_escape_is_narrow(self):
        # Everything between the `if (` and the opening brace of the escape.
        start = _SETUP_SRC.index("if ($shouldRebuild -and -not $InstallerManagedSetup -and\n")
        condition = _SETUP_SRC[start : _SETUP_SRC.index("{", start)]
        for clause in (
            "-not $InstallerManagedSetup",   # install.ps1 repairs in place instead
            "-not $_pinnedIdx",              # a cpu index PIN is deliberate and still rebuilds
            "Test-CudaFamilyLeaf $installedTorchTag",  # only a cu* wheel is preserved
            "-not $HasNvidiaSmi",            # only when the NVIDIA probe gave no answer
            '$expectedTorchTag -eq "cpu"',   # ... and that is why the expectation collapsed
        ):
            assert clause in condition, f"the escape must be gated on {clause!r}"

    def test_the_installed_tag_is_tested_before_the_variables_it_implies(self):
        # $_pinnedIdx and $expectedTorchTag are assigned only inside the
        # `if (-not $shouldRebuild)` block, and a probe that answered is what makes that
        # block run. Under a caller's Set-StrictMode, ordering the -and chain the other
        # way turns the read into a fatal error on a venv whose torch cannot import.
        start = _SETUP_SRC.index("if ($shouldRebuild -and -not $InstallerManagedSetup -and\n")
        condition = _SETUP_SRC[start : _SETUP_SRC.index("{", start)]
        assert condition.index("$installedTorchTag -and") < condition.index("$_pinnedIdx")
        assert condition.index("$installedTorchTag -and") < condition.index("$expectedTorchTag")

    def test_an_xpu_venv_keeps_its_own_escape(self):
        # Regression guard: the pre-existing XPU escape must not have been folded in.
        assert "Keeping the installed Intel XPU environment" in _SETUP_SRC


# ── setup.ps1: the handover ──────────────────────────────────────────────────


class TestSetupPs1PublishesTheFlavor:
    def test_the_tag_is_exported_before_the_stack_runs(self):
        export = _line_of(_SETUP_SRC, "$env:UNSLOTH_EXPECTED_TORCH_TAG =")
        index = _line_of(_SETUP_SRC, "$env:UNSLOTH_TORCH_INSTALL_INDEX_URL =")
        handoff = _line_of(_SETUP_SRC, 'python "$PSScriptRoot\\install_python_stack.py"')
        assert export < handoff and index < handoff

    def test_the_rocm_index_decides_before_the_leaf(self):
        # The AMD Windows path installs from repo.amd.com while $TorchInstallIndexUrl still
        # points at /cpu, so the leaf alone would publish the wrong flavor.
        block = _SETUP_SRC[_SETUP_SRC.index("$_expectedTag = if ($ROCmIndexUrl)") :][:600]
        assert block.startswith('$_expectedTag = if ($ROCmIndexUrl) { "rocm" }')
        assert "Test-CudaFamilyLeaf $_expectedLeaf" in block
        assert "Test-PipRocmFamilyLeaf $_expectedLeaf" in block
        # An unknown leaf publishes nothing rather than a tag nothing can verify.
        assert "else { $null }" in block

    def test_no_torch_mode_publishes_nothing(self):
        block = _SETUP_SRC[
            _SETUP_SRC.index("# ── Publish the torch flavor this run settled on ──") :
        ][:1800]
        assert "if (-not $NoTorchMode) {" in block


# ── install.ps1 parity ───────────────────────────────────────────────────────


class TestInstallPs1Parity:
    """A venv repaired by `studio update` and one repaired by install.ps1 must land on the
    same wheels, and a support log from either must read the same."""

    def test_the_repair_trio_matches_install_ps1(self):
        # install.ps1's $_fixSpecs, the non-XPU arm of its flavor repair.
        match = re.search(
            r'else\s*\{\s*@\((\s*"torch[^)]*?)\)\s*\}', _INSTALL_SRC, re.S
        )
        assert match is not None, "install.ps1's flavor-repair spec array moved"
        ps_specs = tuple(re.findall(r'"([^"]+)"', match.group(1)))
        py_specs = tuple(
            re.findall(
                r'"([^"]+)"',
                re.search(
                    r"_TORCH_FLAVOR_REPAIR_PKG_SPEC: tuple\[str, str, str\] = \((.*?)\)",
                    _STACK_SRC,
                    re.S,
                ).group(1),
            )
        )
        assert py_specs == ps_specs, (
            "_TORCH_FLAVOR_REPAIR_PKG_SPEC must mirror install.ps1's flavor-repair trio "
            f"(python={py_specs}, install.ps1={ps_specs})"
        )

    def test_the_mismatch_line_matches_install_ps1(self):
        assert (
            "PyTorch flavor mismatch (installed $installedTorchTag, need $expectedTorchTag) "
            "-- reinstalling correct build..."
        ) in _INSTALL_SRC
        assert (
            "PyTorch flavor mismatch (installed {installed}, need {expected}) -- "
        ) in _STACK_SRC
        assert "reinstalling correct build..." in _STACK_SRC

    def test_the_loud_warning_matches_install_ps1(self):
        for line in (
            "PyTorch is CPU-only but a",
            "GPU build was expected for this machine.",
            "Training and GPU inference will run on CPU until this is fixed.",
            "Re-run this installer, or reinstall the GPU build manually for your GPU.",
        ):
            assert line in _INSTALL_SRC, f"install.ps1 no longer prints {line!r}"
            assert line in _STACK_SRC, f"install_python_stack.py no longer prints {line!r}"

    def test_the_flavor_vocabulary_matches_convertto_torchflavortag(self):
        # install.ps1's classifier, arm for arm.
        arms = _INSTALL_SRC[_INSTALL_SRC.index("function ConvertTo-TorchFlavorTag") :][:900]
        assert r"'\+(cu\d+)'" in arms
        assert r"'\+rocm'" in arms
        assert r"'\+xpu'" in arms
        assert r"'\+cpu'" in arms
        py = _STACK_SRC[_STACK_SRC.index("def _torch_flavor_tag(") :][:1600]
        assert r'r"\+(cu\d+)"' in py
        assert '"+rocm" in value' in py
        assert '"+xpu" in value' in py


# ── install_python_stack: step wiring ────────────────────────────────────────


def _install_stack_ast():
    tree = ast.parse(_STACK_SRC)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "install_python_stack":
            return node
    raise AssertionError("install_python_stack() not found")


def _calls_in(node) -> list:
    """Every plain function name called under `node`, in source order.

    Depth first, not ast.walk: walk is breadth first, so a nested call reads as if it came
    after its own siblings and the assertions below would encode the wrong order.
    """
    names = []
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        names.append(node.func.id)
    for child in ast.iter_child_nodes(node):
        names.extend(_calls_in(child))
    return names


def _guards_containing(call_name: str) -> list:
    """Every TOP-LEVEL `if` in install_python_stack() whose body calls `call_name`.

    Plural on purpose: the four existing repair helpers run at two points (the step-2b
    check and the step-13 final pass), so a test that took the first match would silently
    assert about the wrong one. Top level only, or a guard's own nested `if` counts twice.
    """
    found = [
        node
        for node in _install_stack_ast().body
        if isinstance(node, ast.If) and call_name in _calls_in(node)
    ]
    assert found, f"no guard around {call_name}()"
    return found


class TestStepThirteenWiring:
    def test_the_flavor_invariant_is_windows_only(self):
        guards = _guards_containing("_ensure_expected_torch_flavor")
        assert [ast.unparse(guard.test) for guard in guards] == ["IS_WINDOWS and (not NO_TORCH)"]

    def test_a_failed_invariant_returns_non_zero(self):
        (guard,) = _guards_containing("_ensure_expected_torch_flavor")
        returns = [
            node.value.value
            for node in ast.walk(guard)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Constant)
        ]
        assert returns == [1], (
            "a torch flavor that could not be repaired must fail the install; today that "
            "state exits 0 and the app silently runs on CPU"
        )

    def test_the_existing_repair_set_is_untouched(self):
        # Both points, verbatim: step 2b (which Windows does enter -- the four helpers
        # return early there themselves) and the Linux-only step 13.
        guards = _guards_containing("_ensure_cuda_torch")
        assert [ast.unparse(guard.test) for guard in guards] == [
            "not IS_MACOS and (not NO_TORCH)",
            "not IS_WINDOWS and (not IS_MACOS) and (not NO_TORCH)",
        ]
        for guard in guards:
            assert _calls_in(guard) == [
                "_progress",
                "_torch_step_label",
                "_ensure_cuda_torch",
                "_ensure_rocm_torch",
                "_ensure_xpu_torch",
                "_ensure_cpu_torch",
                "_ensure_xpu_triton",
            ]

    def test_the_invariant_is_wired_in_exactly_once(self):
        body = ast.unparse(_install_stack_ast())
        assert body.count("_ensure_expected_torch_flavor(") == 1


def _base_total(**flags) -> int:
    """Re-execute install_python_stack()'s step-total arithmetic under given flags.

    Read out of the function rather than duplicated, so a step added without a matching
    total fails here instead of drawing a progress bar past 100%.
    """
    lines = _STACK_SRC.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("base_total = 12"))
    end = next(
        i for i, line in enumerate(lines) if line.strip().startswith("base_requirements =")
    )
    block = textwrap.dedent("\n".join(lines[start:end]))
    namespace = {
        "IS_WINDOWS": False,
        "IS_MACOS": False,
        "NO_TORCH": False,
        "IS_MAC_ARM": False,
        "skip_base": False,
    }
    namespace.update(flags)
    exec(block, namespace)  # noqa: S102 -- the source under test, not user input
    return namespace["base_total"]


class TestStepTotals:
    def test_windows_gained_one_step(self):
        assert _base_total(IS_WINDOWS = True) == 14
        assert _base_total(IS_WINDOWS = True, NO_TORCH = True) == 12

    @pytest.mark.parametrize(
        "flags,total",
        [
            ({}, 16),                                     # Linux, torch
            ({"NO_TORCH": True}, 13),                     # Linux, GGUF-only
            ({"IS_MACOS": True, "IS_MAC_ARM": True}, 13),  # Apple Silicon
            ({"IS_MACOS": True}, 12),                     # Intel Mac
        ],
    )
    def test_the_other_platforms_are_unchanged(self, flags, total):
        assert _base_total(**flags) == total


# ── manifest round-trip ──────────────────────────────────────────────────────


class TestManifestRecordsTheFlavor:
    def test_round_trip(self, tmp_path):
        assert install_manifest.write_manifest(
            root = tmp_path, req_root = tmp_path, expected_torch_tag = "cu124"
        ) is not None
        assert install_manifest.recorded_torch_flavor(tmp_path) == "cu124"

    def test_the_tag_is_normalised(self, tmp_path):
        install_manifest.write_manifest(
            root = tmp_path, req_root = tmp_path, expected_torch_tag = "  CU128 "
        )
        assert install_manifest.recorded_torch_flavor(tmp_path) == "cu128"

    def test_absent_reads_as_unknown_not_cpu(self, tmp_path):
        # Claiming a flavor nobody selected would let a repair reinstall over a deliberate
        # build, so an install written before this key existed must answer None.
        install_manifest.write_manifest(root = tmp_path, req_root = tmp_path)
        assert install_manifest.recorded_torch_flavor(tmp_path) is None

    def test_no_manifest_reads_as_unknown(self, tmp_path):
        assert install_manifest.recorded_torch_flavor(tmp_path) is None

    def test_a_hand_edited_non_string_reads_as_unknown(self, tmp_path):
        path = install_manifest.manifest_path(tmp_path)
        path.write_text(json.dumps({"schema": 1, "expected_torch_tag": 124}), encoding = "utf-8")
        assert install_manifest.recorded_torch_flavor(tmp_path) is None

    def test_the_key_is_additive(self, tmp_path):
        # MANIFEST_SCHEMA must not move: every existing manifest stays valid, and
        # verify_install rejects a schema it does not know.
        assert install_manifest.MANIFEST_SCHEMA == 1
        install_manifest.write_manifest(
            root = tmp_path, req_root = tmp_path, expected_torch_tag = "cu124", no_torch = False
        )
        payload = json.loads(install_manifest.manifest_path(tmp_path).read_text(encoding = "utf-8"))
        assert payload["schema"] == 1
        assert payload["no_torch"] is False
        assert payload["expected_torch_tag"] == "cu124"

    def test_no_index_url_is_ever_written(self, tmp_path):
        # A pinned index can carry a token in its userinfo, query or fragment, and this
        # file sits in the venv and is read back by every later check.
        install_manifest.write_manifest(
            root = tmp_path, req_root = tmp_path, expected_torch_tag = "cu124"
        )
        raw = install_manifest.manifest_path(tmp_path).read_text(encoding = "utf-8")
        assert "http" not in raw

    def test_the_stack_carries_a_previous_record_forward(self):
        # A platform that never resolves a flavor must not erase the one already recorded.
        assert "expected_torch_tag = torch_flavor_tag or _RECORDED_TORCH_TAG," in _STACK_SRC

    def test_the_record_is_read_before_the_manifest_is_dropped(self):
        # install_python_stack() removes the manifest before its dependency pass, so a
        # read deferred into main() always answers None.
        read = _line_of(_STACK_SRC, "_RECORDED_TORCH_TAG = install_manifest.recorded_torch_flavor()")
        drop = _line_of(_STACK_SRC, "if not install_manifest.remove_manifest():")
        assert read < drop
        assert "def install_python_stack" not in _STACK_SRC[: _STACK_SRC.index("_RECORDED_TORCH_TAG =")]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
