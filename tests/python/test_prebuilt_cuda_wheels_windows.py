# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The win_amd64 half of .github/workflows/prebuilt-cuda-wheels.yml.

Nobody publishes flash-attn, causal-conv1d or mamba-ssm for Windows: not one Windows asset exists
across every Dao-AILab and state-spaces release. These pin the parts of the Windows legs that can
be checked without a Windows runner:

* the Linux cells are byte-for-byte what they were, only gaining keys;
* Windows names follow each upstream setup.py's own naming (`cxx11abiTRUE`, as torch reports, and `win_amd64`);
* the shard selector understands Windows ninja targets, and can never run a targetless ninja;
* the deadline wrapper ends the whole process tree;
* SHA256SUMS is rebuilt over every wheel on the release, so a one-platform publish keeps the other;
* the workflow's Windows steps exist in warm and build alike, are gated on the OS, and keep the
  supply-chain checks (Authenticode on the CUDA installer, a pinned sha256 on ccache).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
import types
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


prebuilt_wheels = _load("prebuilt_wheels_win", SCRIPTS / "prebuilt_wheels.py")
shard = _load("prebuilt_wheels_shard_win", SCRIPTS / "prebuilt_wheels_shard.py")
deadline = _load("prebuilt_wheels_timeout_win", SCRIPTS / "prebuilt_wheels_timeout.py")


def triggers(doc: dict) -> dict:
    return doc.get(True) if True in doc else doc.get("on")


@pytest.fixture(scope = "module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


def _steps(workflow, job):
    return workflow["jobs"][job]["steps"]


def _step(workflow, job, name):
    return next(s for s in _steps(workflow, job) if s.get("name") == name)


# ── The matrix ────────────────────────────────────────────────────────────────


# What every Linux cell held before Windows existed, for the default dispatch. A change here is a
# change to the Linux wheels that are already published, so it has to be deliberate.
LINUX_BEFORE = {
    "flash-attn": {
        "build_env": "FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE "
        "FLASH_ATTN_CUDA_ARCHS=80;90;100;120",
        "abi": "TRUE",
        "shards": 8,
        "label": "flash-attn / torch 2.13 / cp313",
        "wheel_name": "flash_attn-2.8.4+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
    },
    "causal-conv1d": {
        "build_env": "CAUSAL_CONV1D_FORCE_BUILD=TRUE",
        "abi": "TRUE",
        "shards": 0,
        "label": "causal-conv1d / torch 2.13 / cp313",
        "wheel_name": "causal_conv1d-1.7.0+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
    },
    "mamba-ssm": {
        "build_env": "MAMBA_FORCE_BUILD=TRUE MAMBA_KEEP_CUDA_BUILD=TRUE",
        "abi": "TRUE",
        "shards": 0,
        "label": "mamba-ssm / torch 2.13 / cp313",
        "wheel_name": "mamba_ssm-2.3.2.post1+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl",
    },
}


class TestMatrix:
    def test_linux_cells_are_unchanged(self):
        cells = prebuilt_wheels.build_matrix(torches = "2.13.0", platforms = "linux_x86_64")
        assert [c["package"] for c in cells] == list(prebuilt_wheels.SPECS)
        for cell in cells:
            for key, value in LINUX_BEFORE[cell["package"]].items():
                assert cell[key] == value, (cell["package"], key)
            assert cell["runner"] == "ubuntu-22.04"
            assert cell["pytorch_nvcc"] == "ccache /usr/local/cuda-13.0/bin/nvcc"
            assert cell["smoke_exclude"] == cell["smoke_extra"] == ""

    def test_linux_comes_first_so_its_cells_keep_their_order(self):
        cells = prebuilt_wheels.build_matrix()
        platforms = [c["platform"] for c in cells]
        assert platforms == sorted(platforms, key = list(prebuilt_wheels.PLATFORMS).index)

    def test_windows_cells(self):
        cells = prebuilt_wheels.build_matrix(platforms = "win_amd64")
        assert len(cells) == 6
        for cell in cells:
            assert cell["os"] == "windows"
            assert cell["runner"] == "windows-2022"
            assert cell["abi"] == "TRUE"
            assert cell["wheel_name"].endswith("cxx11abiTRUE-cp313-cp313-win_amd64.whl")
            assert cell["label"].endswith(" / win_amd64")
            assert cell["pytorch_nvcc"] == "C:/ccache/ccache.exe C:/cuda/bin/nvcc.exe"

    def test_windows_flash_attn_names_the_abi_torch_reports(self):
        """The Windows cu130 torch reports _GLIBCXX_USE_CXX11_ABI=True (measured on windows-2022),
        and FORCE_CXX11_ABI follows the platform's `abi`, so the name and the assert agree."""
        (cell,) = prebuilt_wheels.build_matrix(
            packages = "flash-attn", torches = "2.14.0", platforms = "win_amd64"
        )
        assert "FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE" in cell["build_env"].split()
        assert "FLASH_ATTN_CUDA_ARCHS=80;90;100;120" in cell["build_env"].split()
        assert cell["shards"] == 8

    def test_only_windows_mamba_swaps_its_smoke_dependencies(self):
        for cell in prebuilt_wheels.build_matrix():
            if cell["os"] == "windows" and cell["package"] == "mamba-ssm":
                pin = prebuilt_wheels.TRITON_WINDOWS[cell["torch_mm"]]
                assert cell["smoke_extra"] == f"triton-windows=={pin}"
                assert set(cell["smoke_exclude"].split()) == {
                    "triton",
                    "tilelang",
                    "apache-tvm-ffi",
                    "quack-kernels",
                }
            else:
                assert cell["smoke_extra"] == cell["smoke_exclude"] == ""

    def test_every_torch_minor_has_a_triton_windows_pin(self):
        """torch 2.13 pins triton 3.7.1 and 2.14 pins triton 3.8.x; triton-windows mirrors those."""
        minors = {prebuilt_wheels.torch_minor(v) for v in prebuilt_wheels.TORCH_VERSIONS}
        assert set(prebuilt_wheels.TRITON_WINDOWS) == minors
        assert prebuilt_wheels.TRITON_WINDOWS["2.13"].startswith("3.7.1.post")
        assert prebuilt_wheels.TRITON_WINDOWS["2.14"].startswith("3.8.0.post")

    def test_pytorch_nvcc_paths_have_no_spaces(self):
        """torch writes PYTORCH_NVCC into build.ninja verbatim; a space would split a path."""
        for target in prebuilt_wheels.PLATFORMS.values():
            words = target["pytorch_nvcc"].split(" ")
            assert len(words) == 2 and all(words), target

    @pytest.mark.parametrize("platforms", ["win_arm64", "macosx", "win_amd64; whoami", "$(id)"])
    def test_unknown_platforms_are_refused(self, platforms):
        with pytest.raises(SystemExit):
            prebuilt_wheels.build_matrix(platforms = platforms)

    def test_the_gpu_runner_only_gets_linux_cells(self):
        cells = prebuilt_wheels.build_matrix()
        gpu = prebuilt_wheels.gpu_matrix(cells)
        assert gpu and all(c["os"] == "linux" for c in gpu)
        assert len(gpu) == len([c for c in cells if c["platform"] == "linux_x86_64"])
        assert prebuilt_wheels.gpu_matrix(prebuilt_wheels.build_matrix(platforms = "win_amd64")) == []

    def test_a_windows_only_plan_writes_an_empty_gpu_matrix(self, tmp_path):
        output = tmp_path / "output"
        subprocess.run(
            [sys.executable, str(SCRIPTS / "prebuilt_wheels.py"), "matrix", "--github"],
            env = {"GITHUB_OUTPUT": str(output), "UW_PLATFORMS": "win_amd64"},
            check = True,
            capture_output = True,
        )
        values = dict(line.split("=", 1) for line in output.read_text().splitlines())
        assert values["count"] == "6"
        assert values["gpu_count"] == "0"
        assert values["warm_count"] == "16"
        warm = json.loads(values["warm_matrix"])["include"]
        assert {w["platform"] for w in warm} == {"win_amd64"}


# ── Names ─────────────────────────────────────────────────────────────────────


class TestNames:
    def test_windows_names_are_what_upstream_setup_py_would_write(self):
        """flash-attn, causal-conv1d and mamba's get_wheel_url() all format
        `+cu{cuda}torch{mm}cxx11abi{torch._C._GLIBCXX_USE_CXX11_ABI}` and `win_amd64`."""
        assert (
            prebuilt_wheels.wheel_name("flash-attn", "2.13.0", "3.13", "win_amd64")
            == "flash_attn-2.8.4+cu13torch2.13cxx11abiTRUE-cp313-cp313-win_amd64.whl"
        )
        assert (
            prebuilt_wheels.wheel_name("causal-conv1d", "2.14.0", "3.13", "win_amd64")
            == "causal_conv1d-1.7.0+cu13torch2.14cxx11abiTRUE-cp313-cp313-win_amd64.whl"
        )
        assert (
            prebuilt_wheels.wheel_name("mamba-ssm", "2.14.0", "3.12", "win_amd64")
            == "mamba_ssm-2.3.2.post1+cu13torch2.14cxx11abiTRUE-cp312-cp312-win_amd64.whl"
        )

    def test_the_default_platform_is_still_linux(self):
        assert prebuilt_wheels.wheel_name("flash-attn", "2.13.0", "3.13") == (
            prebuilt_wheels.wheel_name("flash-attn", "2.13.0", "3.13", "linux_x86_64")
        )

    def test_wheel_name_cli_takes_a_platform(self):
        out = subprocess.run(
            [
                sys.executable,
                str(SCRIPTS / "prebuilt_wheels.py"),
                "wheel-name",
                "--package",
                "causal-conv1d",
                "--torch",
                "2.13.0",
                "--python",
                "3.13",
                "--platform",
                "win_amd64",
            ],
            check = True,
            capture_output = True,
            text = True,
        ).stdout.strip()
        assert out == "causal_conv1d-1.7.0+cu13torch2.13cxx11abiTRUE-cp313-cp313-win_amd64.whl"

    def test_round_trip_on_both_platforms(self):
        for platform in prebuilt_wheels.PLATFORMS:
            for package in prebuilt_wheels.SPECS:
                name = prebuilt_wheels.wheel_name(package, "2.14.0", "3.13", platform)
                parsed = prebuilt_wheels.parse_wheel_name(name)
                assert parsed["package"] == package
                assert parsed["platform"] == platform
                assert parsed["abi"] == prebuilt_wheels.PLATFORMS[platform]["abi"]
                assert parsed["version"] == prebuilt_wheels.SPECS[package]["version"]

    @pytest.mark.parametrize(
        "name",
        [
            "flash_attn-2.8.4+cu13torch2.13cxx11abiFALSE-cp313-cp313-win_amd64.whl",
            "flash_attn-2.8.4+cu13torch2.13cxx11abiFALSE-cp313-cp313-linux_x86_64.whl",
            "flash_attn-2.8.4+cu13torch2.14-cp313-cp313-win_arm64.whl",
        ],
    )
    def test_an_abi_the_platform_does_not_have_is_not_ours(self, name):
        assert prebuilt_wheels.parse_wheel_name(name) is None

    def test_the_version_comes_from_the_file(self):
        parsed = prebuilt_wheels.parse_wheel_name(
            "flash_attn-9.9.9+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
        )
        assert parsed["version"] == "9.9.9"


# ── Release notes and SHA256SUMS ──────────────────────────────────────────────


def _entries(platforms, torches = ("2.13.0", "2.14.0")):
    return [
        (f"{index:064x}", prebuilt_wheels.wheel_name(package, torch, "3.13", platform))
        for index, (platform, torch, package) in enumerate(
            (platform, torch, package)
            for platform in platforms
            for torch in torches
            for package in prebuilt_wheels.SPECS
        )
    ]


class TestNotes:
    def test_both_platforms(self):
        notes = prebuilt_wheels.render_notes(
            _entries(["linux_x86_64", "win_amd64"]), tag = "t", repo = "o/r"
        )
        assert notes == (
            "Prebuilt CUDA 13 wheels for flash-attn 2.8.4, causal-conv1d 1.7.0 and "
            "mamba-ssm 2.3.2.post1: Linux x86_64 for PyTorch 2.13 and 2.14 on Python 3.13; "
            "Windows x86_64 for PyTorch 2.13 and 2.14 on Python 3.13.\n"
        )

    def test_platforms_with_different_coverage_are_not_merged(self):
        entries = _entries(["linux_x86_64"]) + _entries(["win_amd64"], torches = ("2.14.0",))
        notes = prebuilt_wheels.render_notes(entries, tag = "t", repo = "o/r")
        assert "Linux x86_64 for PyTorch 2.13 and 2.14" in notes
        assert "Windows x86_64 for PyTorch 2.14 on Python 3.13" in notes

    def test_windows_only(self):
        notes = prebuilt_wheels.render_notes(_entries(["win_amd64"]), tag = "t", repo = "o/r")
        assert notes.startswith("Prebuilt Windows x86_64 CUDA 13 wheels for flash-attn 2.8.4")

    def test_linux_only_reads_as_before(self):
        notes = prebuilt_wheels.render_notes(_entries(["linux_x86_64"]), tag = "t", repo = "o/r")
        assert notes == (
            "Prebuilt Linux x86_64 CUDA 13 wheels for flash-attn 2.8.4, causal-conv1d 1.7.0 and "
            "mamba-ssm 2.3.2.post1, built for PyTorch 2.13 and 2.14 on Python 3.13.\n"
        )


class TestSums:
    def test_every_wheel_sorted_by_name(self):
        entries = _entries(["win_amd64", "linux_x86_64"])
        entries.append(("sha256:" + "f" * 64, "SHA256SUMS"))
        entries.append(("a" * 64, "x.whl.sigstore.json"))
        text = prebuilt_wheels.render_sums(entries)
        lines = text.splitlines()
        assert len(lines) == 12
        names = [line.split("  ", 1)[1] for line in lines]
        assert names == sorted(names)
        assert all(name.endswith(".whl") for name in names)
        assert {n.rsplit("-", 1)[1] for n in names} == {"linux_x86_64.whl", "win_amd64.whl"}

    def test_the_sha256_prefix_is_dropped(self):
        text = prebuilt_wheels.render_sums([("sha256:" + "AB" * 32, "a.whl")])
        assert text == "ab" * 32 + "  a.whl\n"

    @pytest.mark.parametrize("digest", ["unknown", "sha256:unknown", "-", "", "z" * 64, "a" * 63])
    def test_a_missing_digest_is_refused(self, digest):
        with pytest.raises(SystemExit):
            prebuilt_wheels.render_sums([(digest, "a.whl")])

    def test_nothing_to_list_is_refused(self):
        with pytest.raises(SystemExit):
            prebuilt_wheels.render_sums([("a" * 64, "SHA256SUMS")])

    def test_cli(self):
        stdin = f"{'b' * 64}  z.whl\n{'a' * 64}  a.whl\n"
        out = subprocess.run(
            [sys.executable, str(SCRIPTS / "prebuilt_wheels.py"), "sums"],
            input = stdin,
            check = True,
            capture_output = True,
            text = True,
        ).stdout
        assert out == f"{'a' * 64}  a.whl\n{'b' * 64}  z.whl\n"


# ── Shards ────────────────────────────────────────────────────────────────────

WINDOWS_LISTING = "\n".join(
    [
        rf"D:\a\unsloth\unsloth\src\build\temp.win-amd64-cpython-313\Release\csrc\flash_attn\src\k_{i:03d}.obj: cuda_compile"
        for i in range(20)
    ]
    + [
        r"D:\a\unsloth\unsloth\src\build\temp.win-amd64-cpython-313\Release\csrc\flash_attn\flash_api.obj: compile",
        r"D:\a\unsloth\unsloth\src\build\lib.win-amd64-cpython-313\flash_attn_2_cuda.cp313-win_amd64.pyd: link",
        "all: phony",
    ]
)


class TestShards:
    def test_windows_targets_are_whole_paths(self):
        objects = shard.list_objects(WINDOWS_LISTING)
        assert len(objects) == 21
        assert all(o.startswith("D:\\a\\") and o.endswith(".obj") for o in objects)

    def test_windows_slices_cover_everything_once(self):
        slices = [shard.slice_objects(WINDOWS_LISTING, k, 8) for k in range(8)]
        assert sorted(o for s in slices for o in s) == shard.list_objects(WINDOWS_LISTING)

    def test_linux_targets_are_unchanged(self):
        listing = "/src/build/temp/a.o: cuda_compile\n/src/build/temp/b.o: compile\nall: phony"
        assert shard.list_objects(listing) == ["/src/build/temp/a.o", "/src/build/temp/b.o"]

    def test_no_objects_is_an_error_not_a_full_build(self):
        with pytest.raises(SystemExit) as raised:
            shard.slice_objects("all: phony\nfoo.pyd: link", 0, 8)
        assert raised.value.code != 0

    def test_an_empty_slice_never_runs_ninja(self, tmp_path, monkeypatch):
        """Two objects across eight shards leaves six empty slices. `ninja -v -j 1` with no
        target would build the default target, the whole extension, in every one of them."""
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return types.SimpleNamespace(stdout = "a.obj: cuda_compile\nb.obj: cuda_compile\n")

        cpp_extension = types.ModuleType("torch.utils.cpp_extension")
        cpp_extension._run_ninja_build = None
        torch = types.ModuleType("torch")
        utils = types.ModuleType("torch.utils")
        torch.utils = utils
        utils.cpp_extension = cpp_extension
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "torch.utils", utils)
        monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)
        monkeypatch.setattr(shard.subprocess, "run", fake_run)
        (tmp_path / "setup.py").write_text(
            "import torch.utils.cpp_extension as c\nc._run_ninja_build('.', False, '')\n"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prebuilt_wheels_shard.py", "5", "8"])
        with pytest.raises(SystemExit) as raised:
            shard.main()
        assert raised.value.code == 0
        assert calls == [["ninja", "-t", "targets", "all"]]

    def test_a_non_empty_slice_names_its_targets(self, tmp_path, monkeypatch):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return types.SimpleNamespace(stdout = "a.obj: cuda_compile\nb.obj: cuda_compile\n")

        cpp_extension = types.ModuleType("torch.utils.cpp_extension")
        torch = types.ModuleType("torch")
        utils = types.ModuleType("torch.utils")
        torch.utils = utils
        utils.cpp_extension = cpp_extension
        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "torch.utils", utils)
        monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)
        monkeypatch.setattr(shard.subprocess, "run", fake_run)
        monkeypatch.setenv("MAX_JOBS", "1")
        (tmp_path / "setup.py").write_text(
            "import torch.utils.cpp_extension as c\nc._run_ninja_build('.', False, '')\n"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["prebuilt_wheels_shard.py", "1", "8"])
        with pytest.raises(SystemExit):
            shard.main()
        assert calls[-1] == ["ninja", "-v", "-j", "1", "b.obj"]


# ── The deadline wrapper ──────────────────────────────────────────────────────


class TestDeadline:
    @pytest.mark.parametrize(
        "text, seconds", [("300m", 18000), ("90s", 90), ("2h", 7200), ("45", 45)]
    )
    def test_durations(self, text, seconds):
        assert deadline.parse_duration(text) == seconds

    @pytest.mark.parametrize("text", ["", "m", "-5m", "0", "five"])
    def test_bad_durations(self, text):
        with pytest.raises(SystemExit):
            deadline.parse_duration(text)

    def test_the_exit_status_passes_through(self):
        assert deadline.run(30, [sys.executable, "-c", "raise SystemExit(3)"]) == 0 + 3

    @pytest.mark.skipif(sys.platform == "win32", reason = "POSIX process groups")
    def test_the_whole_tree_ends_at_the_deadline(self, tmp_path):
        marker = tmp_path / "grandchild.pid"
        script = (
            "import subprocess, sys, time\n"
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
            f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
            "time.sleep(120)\n"
        )
        started = time.monotonic()
        assert deadline.run(2, [sys.executable, "-c", script]) == 124
        assert time.monotonic() - started < 60
        grandchild = int(marker.read_text())
        for _ in range(50):
            try:
                import os
                os.kill(grandchild, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            pytest.fail("the grandchild outlived the deadline")

    def test_usage(self):
        assert deadline.main(["x", "5m"]) == 2
        assert deadline.main(["x", "5m", "echo"]) == 2


# ── The workflow ──────────────────────────────────────────────────────────────


WINDOWS_SETUP = (
    "Enable long paths (Windows)",
    "Install the CUDA ${{ matrix.cuda_tag }} toolkit (Windows)",
    "Activate the MSVC x64 environment (Windows)",
    "Install ccache (Windows)",
    "Report memory, page file and disk (Windows)",
)
LINUX_ONLY = (
    "Harden runner (audit)",
    "Free up disk space",
    "Install the CUDA ${{ matrix.cuda_tag }} toolkit",
    "Install ccache",
    "Set up swap space",
)


class TestWorkflow:
    def test_platforms_input_defaults_to_both(self, workflow):
        inputs = triggers(workflow)["workflow_dispatch"]["inputs"]
        assert inputs["platforms"]["default"] == "linux_x86_64,win_amd64"
        plan = _step(workflow, "plan", "Expand the dispatch inputs into a matrix")
        assert plan["env"]["UW_PLATFORMS"] == "${{ inputs.platforms }}"

    @pytest.mark.parametrize("job", ["warm", "build"])
    def test_windows_setup_exists_in_both_jobs_and_is_gated(self, workflow, job):
        names = [s.get("name") for s in _steps(workflow, job)]
        for name in WINDOWS_SETUP:
            step = _step(workflow, job, name)
            assert step["if"] == "runner.os == 'Windows'", (job, name)
            assert step["shell"] == "pwsh", (job, name)
        for name in LINUX_ONLY:
            assert _step(workflow, job, name)["if"] == "runner.os == 'Linux'", (job, name)
        # Long paths before any checkout, or the cutlass submodule cannot be checked out.
        first_checkout = next(i for i, n in enumerate(names) if n and n.startswith("Checkout"))
        assert names.index("Enable long paths (Windows)") < first_checkout
        assert workflow["jobs"][job]["defaults"]["run"]["shell"] == "bash"

    def test_warm_and_build_run_the_same_windows_setup(self, workflow):
        for name in WINDOWS_SETUP:
            assert _step(workflow, "warm", name) == _step(workflow, "build", name), name

    def test_the_cuda_installer_is_nvidias_and_signature_checked(self, workflow):
        run = _step(workflow, "build", "Install the CUDA ${{ matrix.cuda_tag }} toolkit (Windows)")[
            "run"
        ]
        assert "https://developer.download.nvidia.com/compute/cuda/" in run
        assert "Get-AuthenticodeSignature" in run and "O=NVIDIA Corporation" in run
        assert "'-s'" in run and "'-n'" in run
        assert "New-Item -ItemType Junction -Path 'C:\\cuda'" in run
        for package in ("nvcc", "cudart", "cublas_dev", "cusparse_dev", "cusolver_dev"):
            assert f"'{package}'" in run, package
        # No display driver and no Visual Studio integration on a build machine.
        assert "visual_studio_integration" not in run and "Display.Driver" not in run

    def test_ccache_is_pinned_by_sha256(self, workflow):
        run = _step(workflow, "build", "Install ccache (Windows)")["run"]
        assert "ccache-4.14.1-windows-x86_64" in run
        assert "6219f3865ca59aec41ee4b678df171d5d35855ecb2b6dbbbd20690b3a68af7b4" in run
        # The default keys on nvcc.exe's mtime, which an installer need not keep across runners.
        assert "CCACHE_COMPILERCHECK=content" in run

    def test_msvc_environment_is_exported_with_distutils_use_sdk(self, workflow):
        run = _step(workflow, "build", "Activate the MSVC x64 environment (Windows)")["run"]
        assert "vswhere.exe" in run and "vcvars64.bat" in run
        assert "DISTUTILS_USE_SDK=1" in run
        assert "GITHUB_ENV" in run and "GITHUB_PATH" in run

    def test_artifacts_and_caches_are_per_platform(self, workflow):
        body = json.dumps(workflow["jobs"])
        for prefix in ("ccache-", "wheel-", "prebuilt-ccache-"):
            for value in (
                v
                for v in body.split('"')
                if v.startswith(prefix) and "${{ matrix" in v and "*" not in v
            ):
                assert value.startswith(prefix + "${{ matrix.platform }}"), value

    def test_the_windows_build_ends_the_whole_tree_at_the_deadline(self, workflow):
        for job, name in (
            ("warm", "Compile slice ${{ matrix.shard }} of ${{ matrix.shards }}"),
            ("build", "Build the wheel"),
        ):
            run = _step(workflow, job, name)["run"]
            assert "prebuilt_wheels_timeout.py" in run
            assert 'timeout --signal=INT --kill-after=120 "$BUILD_TIMEOUT"' in run

    def test_the_rename_expects_the_cells_platform(self, workflow):
        step = _step(workflow, "build", "Name the wheel the way upstream names it")
        assert step["env"]["PLATFORM"] == "${{ matrix.platform }}"
        assert "-${PLATFORM}.whl" in step["run"]

    def test_the_torch_abi_check_follows_the_cell(self, workflow):
        for job in ("warm", "build"):
            step = next(
                s for s in _steps(workflow, job) if s.get("name", "").startswith("Install torch")
            )
            assert step["env"]["MATRIX_ABI"] == "${{ matrix.abi }}"
            assert 'os.environ["MATRIX_ABI"]' in step["run"]

    def test_the_windows_smoke_hides_the_toolkit(self, workflow):
        """A user has no CUDA toolkit; the import must resolve against torch's own DLLs."""
        smoke = _step(workflow, "build", "Smoke test the wheel in a fresh venv")
        run = smoke["run"]
        assert "Scripts/python.exe" in run
        assert "unset CUDA_HOME CUDA_PATH" in run
        assert smoke["env"]["SMOKE_EXCLUDE"] == "${{ matrix.smoke_exclude }}"
        assert smoke["env"]["SMOKE_EXTRA"] == "${{ matrix.smoke_extra }}"
        assert 'os.environ.get("SMOKE_EXCLUDE"' in run and 'os.environ.get("SMOKE_EXTRA"' in run

    def test_the_gpu_job_only_gets_linux_cells(self, workflow):
        job = workflow["jobs"]["gpu-smoke"]
        assert job["strategy"]["matrix"] == "${{ fromJSON(needs.plan.outputs.gpu_matrix) }}"
        assert "needs.plan.outputs.gpu_count != '0'" in job["if"]
        assert workflow["jobs"]["plan"]["outputs"]["gpu_count"]

    def test_publish_rebuilds_sha256sums_over_the_whole_release(self, workflow):
        steps = workflow["jobs"]["publish"]["steps"]
        create = next(s for s in steps if s.get("name") == "Create or update the release")
        # This run's SHA256SUMS lists this run only; it must not become the release's.
        assert "wheels/SHA256SUMS" not in create["run"]
        regen = next(s for s in steps if "SHA256SUMS" in s.get("name", ""))
        run = regen["run"]
        assert "prebuilt_wheels.py sums" in run
        assert "--clobber SHA256SUMS" in run
        assert "releases/assets/$id" in run
        assert "sha256:unknown" not in run
        # Built before the notes, from the same listing.
        assert run.index("prebuilt_wheels.py sums") < run.index("prebuilt_wheels.py notes")
