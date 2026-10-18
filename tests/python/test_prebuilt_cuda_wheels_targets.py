# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SASS trimming for the prebuilt CUDA wheels: patch_cuda_targets.py, cuda_archs, targets-gate."""

from __future__ import annotations

import importlib.util
import os
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "prebuilt-cuda-wheels.yml"
SCRIPTS = REPO / ".github" / "scripts"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prebuilt_wheels = _load("prebuilt_wheels", SCRIPTS / "prebuilt_wheels.py")
sys.modules["patch_cuda_targets"] = _load("patch_cuda_targets", SCRIPTS / "patch_cuda_targets.py")

# The gencode block and nvcc list of causal-conv1d's setup.py at the pinned commit (mamba-ssm's is
# the same shape).
SSM_SETUP = """
def build():
    cc_flag = []
    if True:
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_75,code=sm_75")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_87,code=sm_87")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_90,code=sm_90")
        if bare_metal_version >= Version("12.8"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_100,code=sm_100")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_120,code=sm_120")
        if bare_metal_version >= Version("13.0"):
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_103,code=sm_103")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_110,code=sm_110")
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_121,code=sm_121")
    nvcc = [
        "-O3",
        "--use_fast_math",
        "--ptxas-options=-v",
        "-lineinfo",
    ] + cc_flag
    return nvcc
"""

FLASH_SETUP = """
def add_cuda_gencodes(cc_flag, archs, bare_metal_version):
    if "80" in archs:
        cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]
    numeric = [a for a in archs if a.isdigit()]
    if numeric:
        newest = max(numeric, key=int)
        cc_flag += ["-gencode", f"arch=compute_{newest},code=compute_{newest}"]
    return cc_flag
"""


def _patch(tmp_path, source, *args):
    setup = tmp_path / "setup.py"
    setup.write_text(source, encoding = "utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "patch_cuda_targets.py"), *args, str(setup)],
        capture_output = True,
        text = True,
    )
    return result, setup


def _gencodes(source: str) -> set[str]:
    return {
        line.split("code=sm_")[1].split('"')[0]
        for line in source.splitlines()
        if "code=sm_" in line and "cc_flag.append" in line
    }


class TestSsmTrim:
    @pytest.mark.parametrize("package", ["causal-conv1d", "mamba-ssm"])
    def test_keeps_only_the_requested_sass_and_drops_lineinfo(self, tmp_path, package):
        result, setup = _patch(
            tmp_path, SSM_SETUP, "--package", package, "--archs", "75;80;90;100;120"
        )
        assert result.returncode == 0, result.stderr
        patched = setup.read_text(encoding = "utf-8")
        assert _gencodes(patched) == {"75", "80", "90", "100", "120"}
        assert '"-lineinfo"' not in patched
        # The CUDA 13 block lost every target; it must still be valid Python.
        compile(patched, "setup.py", "exec")
        assert "dropped ['87', '103', '110', '121']" in result.stdout

    def test_the_arm_list_only_drops_lineinfo(self, tmp_path):
        result, setup = _patch(
            tmp_path,
            SSM_SETUP,
            "--package",
            "causal-conv1d",
            "--archs",
            prebuilt_wheels.SSM_SASS_ARM,
        )
        assert result.returncode == 0, result.stderr
        patched = setup.read_text(encoding = "utf-8")
        assert patched == SSM_SETUP.replace('        "-lineinfo",\n', "")
        assert "dropped none" in result.stdout

    def test_the_x86_list(self, tmp_path):
        result, setup = _patch(
            tmp_path, SSM_SETUP, "--package", "mamba-ssm", "--archs", prebuilt_wheels.SSM_SASS
        )
        assert result.returncode == 0, result.stderr
        assert _gencodes(setup.read_text(encoding = "utf-8")) == {
            "75",
            "80",
            "90",
            "100",
            "103",
            "120",
        }

    def test_is_idempotent(self, tmp_path):
        result, setup = _patch(
            tmp_path, SSM_SETUP, "--package", "causal-conv1d", "--archs", "75;80;90;100;120"
        )
        assert result.returncode == 0
        once = setup.read_text(encoding = "utf-8")
        again = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "patch_cuda_targets.py"),
                "--package",
                "causal-conv1d",
                "--archs",
                "75;80;90;100;120",
                str(setup),
            ],
            capture_output = True,
            text = True,
        )
        assert again.returncode == 0, again.stderr
        assert setup.read_text(encoding = "utf-8") == once

    def test_an_unknown_upstream_target_fails_loudly(self, tmp_path):
        source = SSM_SETUP.replace("compute_121,code=sm_121", "compute_130,code=sm_130")
        result, setup = _patch(
            tmp_path, source, "--package", "causal-conv1d", "--archs", "75;80;90;100;120"
        )
        assert result.returncode == 1
        assert "does not know" in result.stderr
        assert setup.read_text(encoding = "utf-8") == source

    def test_a_requested_target_upstream_dropped_fails(self, tmp_path):
        source = SSM_SETUP.replace(
            '        cc_flag.append("-gencode")\n        cc_flag.append("arch=compute_75,code=sm_75")\n',
            "",
        )
        result, _ = _patch(
            tmp_path, source, "--package", "causal-conv1d", "--archs", "75;80;90;100;120"
        )
        assert result.returncode == 1
        assert "not in setup.py" in result.stderr

    def test_a_partial_list_is_neither_known_state(self, tmp_path):
        source = SSM_SETUP.replace(
            '            cc_flag.append("-gencode")\n            cc_flag.append("arch=compute_121,code=sm_121")\n',
            "",
        )
        result, _ = _patch(
            tmp_path, source, "--package", "causal-conv1d", "--archs", "75;80;90;100;120"
        )
        assert result.returncode == 1
        assert "re-read it" in result.stderr

    def test_rejects_an_arch_outside_the_known_set(self, tmp_path):
        result, _ = _patch(tmp_path, SSM_SETUP, "--package", "causal-conv1d", "--archs", "75;86")
        assert result.returncode == 1


class TestFlashTrim:
    def test_drops_the_ptx_and_keeps_the_sass(self, tmp_path):
        result, setup = _patch(tmp_path, FLASH_SETUP, "--package", "flash-attn")
        assert result.returncode == 0, result.stderr
        patched = setup.read_text(encoding = "utf-8")
        assert "code=compute_" not in patched
        assert 'cc_flag += ["-gencode", "arch=compute_80,code=sm_80"]' in patched
        compile(patched, "setup.py", "exec")
        again, _ = _patch(tmp_path, patched, "--package", "flash-attn")
        assert again.returncode == 0
        assert "already" in again.stdout

    def test_a_moved_ptx_line_fails(self, tmp_path):
        source = FLASH_SETUP.replace("code=compute_{newest}", "code=[compute_{newest}]")
        result, _ = _patch(tmp_path, source, "--package", "flash-attn")
        assert result.returncode == 1


class TestCudaArchs:
    def _cells(self):
        cells = prebuilt_wheels.build_matrix(
            torches = "2.13.0,2.14.0", platforms = ",".join(prebuilt_wheels.PLATFORMS)
        )
        cells += prebuilt_wheels.build_matrix(
            torches = "2.10.0,2.11.0,2.12.0", pythons = "3.12", platforms = "win_amd64"
        )
        return cells

    def test_every_cell_names_its_targets(self):
        cells = self._cells()
        assert {cell["platform"] for cell in cells} == set(prebuilt_wheels.PLATFORMS)
        integrated_arm = {"87", "110", "121"}
        for cell in cells:
            archs = cell["cuda_archs"].split(";")
            if cell["package"] == "flash-attn":
                assert archs == ["80", "90", "100", "120"]
            elif cell["platform"] in ("linux_aarch64", "win_arm64"):
                # The full upstream list: only -lineinfo goes.
                assert set(archs) == set(
                    importlib.import_module("patch_cuda_targets").SSM_KNOWN_ARCHS
                ), cell["label"]
            else:
                # Every GPU an x86 host can have keeps its own SASS, B300's sm_103 included.
                assert archs == ["75", "80", "90", "100", "103", "120"], cell["label"]
                assert not integrated_arm & set(archs)

    def test_flash_attn_targets_come_from_its_own_env(self):
        env = prebuilt_wheels.SPECS["flash-attn"]["env"]["FLASH_ATTN_CUDA_ARCHS"]
        assert prebuilt_wheels.cuda_archs("flash-attn", "linux_x86_64") == env

    def test_the_ccache_tag_changed_with_the_targets(self):
        """ccache would miss on the changed command lines anyway; a new tag keeps the cache key
        from restoring objects built for the old target list at all."""
        assert prebuilt_wheels.CCACHE_TAG != "fatbin-size"


LISTING = """\
ELF file    1: causal_conv1d_cuda.1.sm_75.cubin
ELF file    2: causal_conv1d_cuda.2.sm_80.cubin
ELF file    3: causal_conv1d_cuda.3.sm_90.cubin
ELF file    4: causal_conv1d_cuda.4.sm_100.cubin
ELF file    5: causal_conv1d_cuda.5.sm_120.cubin
"""


class TestTargetsGate:
    def test_parses_cuobjdump(self):
        sass, ptx = prebuilt_wheels.device_targets(
            LISTING + "PTX file    1: flash.1.compute_120.ptx\nELF file 6: x.sm_100a.cubin\n"
        )
        assert sass == {"75", "80", "90", "100", "120"}
        assert ptx == {"120"}

    def test_check_targets(self):
        want = {"75", "80", "90", "100", "120"}
        assert prebuilt_wheels.check_targets("x", want, set(), want) == []
        problems = prebuilt_wheels.check_targets("x", {"75", "80", "87"}, {"120"}, want)
        assert any("no SASS for sm_90, sm_100, sm_120" in p for p in problems)
        assert any("sm_87 was not asked for" in p for p in problems)
        assert any("PTX for compute_120" in p for p in problems)

    @pytest.fixture
    def fake_cuobjdump(self, tmp_path, monkeypatch):
        bindir = tmp_path / "bin"
        bindir.mkdir()
        listing = tmp_path / "listing.txt"
        if os.name == "nt":
            tool = bindir / "cuobjdump.bat"
            tool.write_text(f'@type "{listing}"\n', encoding = "utf-8")
        else:
            tool = bindir / "cuobjdump"
            tool.write_text(f'#!/bin/sh\ncat "{listing}"\n', encoding = "utf-8")
            tool.chmod(tool.stat().st_mode | stat.S_IEXEC)
        monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
        return listing

    def _wheel(self, tmp_path):
        wheel = tmp_path / "causal_conv1d-1.7.0-cp313-cp313-linux_x86_64.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr("causal_conv1d_cuda.cpython-313-x86_64-linux-gnu.so", b"\0")
            archive.writestr("causal_conv1d/__init__.py", b"")
        return wheel

    @pytest.mark.parametrize(
        "archs, extra, status",
        [
            ("75;80;90;100;120", "", 0),
            ("75;80;90;100;110;120", "", 1),
            ("75;80;90;100;120", "ELF file    6: causal_conv1d_cuda.6.sm_121.cubin\n", 1),
            ("75;80;90;100;120", "PTX file    1: causal_conv1d_cuda.1.compute_120.ptx\n", 1),
        ],
    )
    def test_gate(self, tmp_path, fake_cuobjdump, archs, extra, status):
        fake_cuobjdump.write_text(LISTING + extra, encoding = "utf-8")
        wheel = self._wheel(tmp_path)
        assert prebuilt_wheels.main(["targets-gate", "--archs", archs, str(wheel)]) == status

    def test_a_listing_with_no_device_code_fails(self, tmp_path, fake_cuobjdump):
        fake_cuobjdump.write_text(
            "cuobjdump info    : File does not contain device code\n", encoding = "utf-8"
        )
        wheel = self._wheel(tmp_path)
        assert prebuilt_wheels.main(["targets-gate", "--archs", "75;80", str(wheel)]) == 1


@pytest.fixture(scope = "module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


class TestWorkflowWiring:
    @pytest.mark.parametrize("job", ["warm", "build"])
    def test_trim_runs_after_the_source_patches_and_before_compiling(self, workflow, job):
        steps = workflow["jobs"][job]["steps"]
        names = [s.get("name") for s in steps]
        trim = names.index("Trim the CUDA targets")
        assert names.index("Patch mamba-ssm for the MSVC host compiler (Windows)") < trim
        compile_step = (
            "Build the wheel"
            if job == "build"
            else ("Compile slice ${{ matrix.shard }} of ${{ matrix.shards }}")
        )
        assert trim < names.index(compile_step)
        step = steps[trim]
        # Unconditional: every package and platform builds trimmed, or the gate below fails.
        assert "if" not in step
        assert step["env"]["CUDA_ARCHS"] == "${{ matrix.cuda_archs }}"
        assert "patch_cuda_targets.py" in step["run"]

    def test_the_targets_gate_runs_before_the_wheel_leaves_the_build(self, workflow):
        steps = workflow["jobs"]["build"]["steps"]
        names = [s.get("name") for s in steps]
        gate = names.index("Check the device code targets")
        assert names.index("Check the extension size") < gate
        assert gate < names.index("Smoke test the wheel in a fresh venv")
        assert "if" not in steps[gate]
        assert "targets-gate" in steps[gate]["run"]
        assert steps[gate]["env"]["CUDA_ARCHS"] == "${{ matrix.cuda_archs }}"
