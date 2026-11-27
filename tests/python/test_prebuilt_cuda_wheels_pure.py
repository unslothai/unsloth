# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The extension-free mamba-ssm wheel: pure_matrix, pure-gate, notes and the workflow jobs."""

from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "prebuilt-cuda-wheels.yml"
SCRIPTS = REPO / ".github" / "scripts"

_spec = importlib.util.spec_from_file_location("prebuilt_wheels", SCRIPTS / "prebuilt_wheels.py")
pw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pw)

NAME = "mamba_ssm-2.3.2.post1-py3-none-any.whl"


def _wheel(
    tmp_path,
    name = NAME,
    extra = (),
    tag = "py3-none-any",
    purelib = "true",
):
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mamba_ssm/__init__.py", '__version__ = "2.3.2.post1"\n')
        archive.writestr("mamba_ssm/ops/triton/ssd_combined.py", "")
        archive.writestr(
            "mamba_ssm-2.3.2.post1.dist-info/WHEEL",
            f"Wheel-Version: 1.0\nRoot-Is-Purelib: {purelib}\nTag: {tag}\n",
        )
        for member in extra:
            archive.writestr(member, b"\0")
    return path


def test_name_follows_the_cuda_wheel_pin():
    assert pw.pure_wheel_name() == NAME
    assert pw.SPECS["mamba-ssm"]["version"] == "2.3.2.post1"


@pytest.mark.parametrize(
    "packages, count",
    [
        ("", 1),
        ("mamba-ssm", 1),
        ("flash-attn,mamba-ssm", 1),
        ("flash-attn", 0),
        ("causal-conv1d", 0),
    ],
)
def test_one_cell_only_when_the_run_builds_mamba_ssm(packages, count):
    cells = pw.pure_matrix(packages)
    assert len(cells) == count
    if cells:
        cell = cells[0]
        assert cell["wheel_name"] == NAME
        # Same source as the CUDA wheel.
        assert (cell["repo"], cell["ref"]) == (
            pw.SPECS["mamba-ssm"]["repo"],
            pw.SPECS["mamba-ssm"]["ref"],
        )
        assert "ssd_combined import mamba_chunk_scan_combined" in cell["smoke_imports"]


def test_matrix_writes_pure_outputs(tmp_path, monkeypatch):
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising = False)
    for packages, count in (("mamba-ssm", 1), ("flash-attn", 0)):
        out.write_text("")
        monkeypatch.setenv("UW_PACKAGES", packages)
        monkeypatch.setenv("UW_PLATFORMS", "linux_x86_64")
        assert pw.main(["matrix", "--github"]) == 0
        lines = dict(line.split("=", 1) for line in out.read_text().splitlines())
        assert lines["pure_count"] == str(count)
        assert len(json.loads(lines["pure_matrix"])["include"]) == count
        # The CUDA cells are unchanged by it.
        assert "py3-none-any" not in lines["matrix"]


def test_gate_accepts_the_pure_wheel(tmp_path):
    assert pw.main(["pure-gate", str(_wheel(tmp_path))]) == 0


@pytest.mark.parametrize(
    "kwargs, problem",
    [
        ({"extra": ["selective_scan_cuda.cpython-313-x86_64-linux-gnu.so"]}, "compiled code"),
        ({"extra": ["selective_scan_cuda.cp313-win_amd64.pyd"]}, "compiled code"),
        ({"name": "mamba_ssm-2.3.2.post1-cp313-cp313-linux_x86_64.whl"}, "expected"),
        ({"tag": "cp313-cp313-linux_x86_64"}, "not pure"),
        ({"purelib": "false"}, "not pure"),
    ],
)
def test_gate_refuses_anything_else(tmp_path, kwargs, problem):
    wheel = _wheel(tmp_path, **kwargs)
    assert any(problem in p for p in pw.check_pure_wheel(str(wheel)))
    assert pw.main(["pure-gate", str(wheel)]) == 1


def test_notes_mention_it_and_sums_list_it():
    cuda = "mamba_ssm-2.3.2.post1+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
    entries = [("a" * 64, cuda), ("b" * 64, NAME)]
    notes = pw.render_notes(entries, tag = "t", repo = "o/r")
    assert "without its CUDA extension" in notes and NAME in notes
    assert "mamba-ssm 2.3.2.post1" in notes
    assert NAME in pw.render_sums(entries)
    only = pw.render_notes([("b" * 64, NAME)], tag = "t", repo = "o/r")
    assert NAME in only and "No wheels" not in only
    assert "without its CUDA extension" not in pw.render_notes(
        [("a" * 64, cuda)], tag = "t", repo = "o/r"
    )


@pytest.fixture(scope = "module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def test_pure_jobs_are_planned_and_gated(workflow):
    jobs = workflow["jobs"]
    outputs = jobs["plan"]["outputs"]
    assert "pure_matrix" in outputs and "pure_count" in outputs
    pure = jobs["pure"]
    assert "pure_count != '0'" in pure["if"]
    assert pure["runs-on"] == "ubuntu-latest"
    steps = {step.get("name", ""): step for step in pure["steps"]}
    build = steps["Build the wheel"]
    assert build["env"]["MAMBA_KEEP_CUDA_BUILD"] == "FALSE"
    assert build["env"]["MAMBA_FORCE_BUILD"] == "FALSE"
    assert "pure-gate" in steps["Check the wheel is pure"]["run"]
    assert "--no-deps" in steps["Import it (Linux)"]["run"]
    upload = next(step for step in pure["steps"] if "upload-artifact" in step.get("uses", ""))
    # The sign job downloads wheel-*; the Windows import job downloads the same name.
    assert upload["with"]["name"] == "wheel-py3-none-any-mamba_ssm"
    windows = jobs["pure-windows"]
    assert windows["runs-on"] == "windows-latest"
    assert any("triton_windows" in str(step.get("env", {})) for step in windows["steps"])
    download = next(
        step for step in windows["steps"] if "download-artifact" in step.get("uses", "")
    )
    assert download["with"]["name"] == upload["with"]["name"]


def test_sign_waits_for_the_pure_wheel_only_when_planned(workflow):
    sign = workflow["jobs"]["sign"]
    assert {"pure", "pure-windows"} <= set(sign["needs"])
    condition = sign["if"]
    assert "pure_count == '0'" in condition
    assert "needs.pure.result == 'success'" in condition
    assert "needs.pure-windows.result == 'success'" in condition
    arrived = next(
        step for step in sign["steps"] if step.get("name") == "Every planned wheel arrived"
    )
    assert "PURE_COUNT" in arrived["env"] and "CUDA_COUNT" in arrived["env"]
    assert "CUDA_COUNT + ${PURE_COUNT:-0}" in arrived["run"]
