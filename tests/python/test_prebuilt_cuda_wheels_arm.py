# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The ARM half of .github/workflows/prebuilt-cuda-wheels.yml.

* linux_aarch64 builds all three packages for torch 2.13 / 2.14 (upstream's aarch64 wheels stop
  at 2.10); win_arm64 builds causal-conv1d and mamba-ssm for NVIDIA's torch 2.14.0+cu134 only,
  since NVIDIA publishes flash-attn for it and no other CUDA torch exists there;
* ARM is opt-in: the default dispatch plans exactly the x86_64 cells it planned before;
* the x86_64 GPU runner never gets an aarch64 wheel;
* the CUDA 13.4 redist installer checks a pinned manifest and every archive against it;
* the binary check catches an x64 extension under an arm64 name and a glibc above the floor.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import struct
import zipfile
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "prebuilt-cuda-wheels.yml"
SCRIPTS = REPO / ".github" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pw = _load("prebuilt_wheels")
redist = _load("prebuilt_wheels_cuda_redist")
binaries = _load("prebuilt_wheels_check_binaries")


@pytest.fixture(scope = "module")
def workflow():
    return yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))


class TestMatrix:
    def test_the_default_dispatch_plans_no_arm_cell(self):
        cells = pw.build_matrix()
        assert {cell["platform"] for cell in cells} == {"linux_x86_64", "win_amd64"}
        assert len(cells) == 12

    def test_linux_aarch64_builds_every_package_for_both_torches(self):
        cells = pw.build_matrix(platforms = "linux_aarch64")
        assert sorted(cell["wheel_name"] for cell in cells) == sorted(
            [
                "flash_attn-2.8.4+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_aarch64.whl",
                "flash_attn-2.8.4+cu13torch2.14cxx11abiTRUE-cp313-cp313-linux_aarch64.whl",
                "causal_conv1d-1.7.0+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_aarch64.whl",
                "causal_conv1d-1.7.0+cu13torch2.14cxx11abiTRUE-cp313-cp313-linux_aarch64.whl",
                "mamba_ssm-2.3.2.post1+cu13torch2.13cxx11abiTRUE-cp313-cp313-linux_aarch64.whl",
                "mamba_ssm-2.3.2.post1+cu13torch2.14cxx11abiTRUE-cp313-cp313-linux_aarch64.whl",
            ]
        )
        for cell in cells:
            assert cell["runner"] == "ubuntu-22.04-arm"
            assert cell["torch_spec"] == f"torch=={cell['torch']}"
            assert cell["torch_index_args"] == "--index-url https://download.pytorch.org/whl/cu130"
            assert cell["py_platform"] == "linux-aarch64"
            assert cell["smoke_exclude"] == "" and cell["smoke_extra"] == ""
        assert len(pw.warm_matrix(cells)) == 16

    def test_win_arm64_builds_two_packages_for_nvidias_torch_only(self):
        cells = pw.build_matrix(platforms = "win_arm64")
        assert sorted(cell["wheel_name"] for cell in cells) == [
            "causal_conv1d-1.7.0+cu13torch2.14cxx11abiTRUE-cp313-cp313-win_arm64.whl",
            "mamba_ssm-2.3.2.post1+cu13torch2.14cxx11abiTRUE-cp313-cp313-win_arm64.whl",
        ]
        for cell in cells:
            assert cell["runner"] == "windows-11-arm"
            assert cell["torch_spec"] == "torch==2.14.0+cu134"
            assert (
                cell["torch_index_args"] == "--extra-index-url https://pypi.nvidia.com/nvtorch_oot"
            )
            assert cell["py_platform"] == "win-arm64"
            assert cell["shards"] == 0
        mamba = next(cell for cell in cells if cell["package"] == "mamba-ssm")
        # triton-windows 3.8.0.post29 publishes win_arm64 wheels, the release torch 2.14 pairs with.
        assert mamba["smoke_extra"] == "triton-windows==3.8.0.post29"
        assert "triton" in mamba["smoke_exclude"].split()

    def test_win_arm64_adds_interpreters_on_request(self):
        cells = pw.build_matrix(platforms = "win_arm64", pythons = "3.11,3.12,3.13")
        assert sorted({cell["python_tag"] for cell in cells}) == ["cp311", "cp312", "cp313"]
        assert len(cells) == 6

    @pytest.mark.parametrize(
        "request_",
        [
            {"platforms": "win_arm64", "torches": "2.13.0"},
            {"platforms": "win_arm64", "packages": "flash-attn"},
        ],
    )
    def test_a_request_with_no_buildable_cell_fails(self, request_):
        with pytest.raises(SystemExit, match = "no cells"):
            pw.build_matrix(**request_)

    def test_a_mixed_request_keeps_only_the_buildable_cells(self):
        cells = pw.build_matrix(platforms = "linux_x86_64,win_arm64", packages = "flash-attn")
        assert {cell["platform"] for cell in cells} == {"linux_x86_64"}

    def test_x86_64_cells_are_unchanged_by_arm(self):
        before = [cell for cell in pw.build_matrix() if cell["platform"] == "linux_x86_64"]
        with_arm = [
            cell
            for cell in pw.build_matrix(platforms = "linux_x86_64,linux_aarch64,win_arm64")
            if cell["platform"] == "linux_x86_64"
        ]
        assert before == with_arm
        for cell in before:
            assert cell["torch_spec"] == f"torch=={cell['torch']}"
            assert cell["torch_index_args"] == "--index-url https://download.pytorch.org/whl/cu130"

    def test_the_gpu_runner_only_gets_x86_64_linux_wheels(self):
        cells = pw.build_matrix(platforms = "linux_x86_64,linux_aarch64,win_amd64,win_arm64")
        assert {cell["platform"] for cell in pw.gpu_matrix(cells)} == {"linux_x86_64"}

    def test_arm_wheel_names_parse_and_render(self):
        names = [
            cell["wheel_name"] for cell in pw.build_matrix(platforms = "linux_aarch64,win_arm64")
        ]
        for name in names:
            row = pw.parse_wheel_name(name)
            assert row is not None and row["platform"] in ("linux_aarch64", "win_arm64")
        notes = pw.render_notes([("0" * 64, name) for name in names], tag = "t", repo = "o/r")
        assert (
            "Linux aarch64: flash-attn 2.8.4, causal-conv1d 1.7.0 and mamba-ssm 2.3.2.post1"
            in notes
        )
        assert (
            "Windows ARM64: causal-conv1d 1.7.0 and mamba-ssm 2.3.2.post1, built for PyTorch 2.14"
            in notes
        )


def _manifest_and_archives(tmp_path, components):
    manifest = {"release_label": "13.4.1", "release_date": "2026-09-09"}
    for name in components:
        top = f"{name}-windows-arm64-1.0-archive"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr(f"{top}/", "")
            zf.writestr(f"{top}/LICENSE", "license")
            zf.writestr(f"{top}/bin/{name}.exe", name)
            zf.writestr(f"{top}/include/{name}.h", name)
        data = buffer.getvalue()
        relative = f"{name}/windows-arm64/{top}.zip"
        (tmp_path / "srv" / Path(relative).parent).mkdir(parents = True, exist_ok = True)
        (tmp_path / "srv" / relative).write_bytes(data)
        manifest[name] = {
            "windows-arm64": {"relative_path": relative, "sha256": hashlib.sha256(data).hexdigest()}
        }
    raw = json.dumps(manifest).encode()
    (tmp_path / "srv" / "manifest.json").write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def local_redist(tmp_path, monkeypatch):
    def fetch(url, dest):
        name = url.rsplit("/", 1)[1] if url.endswith(".json") else url[len(redist.REDIST_BASE) :]
        dest.write_bytes((tmp_path / "srv" / name).read_bytes())

    monkeypatch.setattr(redist, "_fetch", fetch)
    return tmp_path


class TestCudaRedist:
    def test_installs_every_component_into_one_tree(self, local_redist):
        sha = _manifest_and_archives(local_redist, ["cuda_nvcc", "cccl"])
        dest = local_redist / "cuda"
        assert (
            redist.main(
                ["https://x/manifest.json", sha, "windows-arm64", str(dest), "cuda_nvcc", "cccl"]
            )
            == 0
        )
        assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()) == [
            "bin/cccl.exe",
            "bin/cuda_nvcc.exe",
            "include/cccl.h",
            "include/cuda_nvcc.h",
        ]

    def test_a_manifest_that_is_not_the_pinned_one_is_refused(self, local_redist):
        _manifest_and_archives(local_redist, ["cuda_nvcc"])
        with pytest.raises(SystemExit, match = "manifest sha256"):
            redist.main(
                [
                    "https://x/manifest.json",
                    "0" * 64,
                    "windows-arm64",
                    str(local_redist / "c"),
                    "cuda_nvcc",
                ]
            )

    def test_an_archive_that_does_not_match_the_manifest_is_refused(self, local_redist):
        sha = _manifest_and_archives(local_redist, ["cuda_nvcc"])
        archive = next((local_redist / "srv" / "cuda_nvcc").rglob("*.zip"))
        archive.write_bytes(archive.read_bytes() + b"tampered")
        with pytest.raises(SystemExit, match = "manifest says"):
            redist.main(
                [
                    "https://x/manifest.json",
                    sha,
                    "windows-arm64",
                    str(local_redist / "c"),
                    "cuda_nvcc",
                ]
            )

    def test_a_component_missing_for_the_platform_is_refused(self):
        with pytest.raises(SystemExit, match = "not in the manifest"):
            redist.plan({"cuda_nvcc": {"linux-sbsa": {}}}, "windows-arm64", ["cuda_nvcc"])

    def test_an_archive_escaping_dest_is_refused(self, tmp_path):
        archive = tmp_path / "evil.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr("top/../../escape.txt", "x")
        with pytest.raises(SystemExit, match = "escapes"):
            redist.merge(archive, tmp_path / "dest")

    def test_the_workflow_pins_the_manifest_it_was_written_against(self, workflow):
        for job in ("warm", "build"):
            step = next(
                s
                for s in workflow["jobs"][job]["steps"]
                if s.get("name") == "Install the CUDA 13.4 toolkit (Windows ARM64)"
            )
            assert "redistrib_13.4.1.json" in step["run"]
            assert "291178934b2139727407c76f697bb0dfbac4014753a8094d7c290b0f4124a302" in step["run"]
            for component in (
                "cuda_nvcc",
                "libnvvm",
                "cuda_crt",
                "cuda_cudart",
                "cccl",
                "libcublas",
            ):
                assert component in step["run"].split(), component


def _elf(machine: int, glibc: str = "2.34") -> bytes:
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    struct.pack_into("<H", header, 18, machine)
    return bytes(header) + f"\x00GLIBC_{glibc}\x00GLIBC_2.17\x00".encode()


def _pe(machine: int) -> bytes:
    data = bytearray(0x100)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 0x80)
    data[0x80:0x84] = b"PE\x00\x00"
    struct.pack_into("<H", data, 0x84, machine)
    return bytes(data)


def _wheel(tmp_path, files: dict[str, bytes]) -> str:
    path = tmp_path / "x.whl"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return str(path)


class TestBinaryCheck:
    def test_matching_arm_extensions_pass(self, tmp_path):
        assert (
            binaries.check(
                _wheel(tmp_path, {"m/a.cpython-313-aarch64-linux-gnu.so": _elf(183)}),
                "linux_aarch64",
            )
            == []
        )
        assert (
            binaries.check(_wheel(tmp_path, {"m/a.cp313-win_arm64.pyd": _pe(0xAA64)}), "win_arm64")
            == []
        )

    def test_an_x64_extension_under_an_arm_name_fails(self, tmp_path):
        assert binaries.check(_wheel(tmp_path, {"m/a.so": _elf(62)}), "linux_aarch64")
        assert binaries.check(_wheel(tmp_path, {"m/a.pyd": _pe(0x8664)}), "win_arm64")

    def test_a_glibc_above_the_floor_fails(self, tmp_path):
        problems = binaries.check(_wheel(tmp_path, {"m/a.so": _elf(183, "2.38")}), "linux_aarch64")
        assert problems and "GLIBC 2.38" in problems[0]

    def test_a_wheel_without_an_extension_fails(self, tmp_path):
        assert binaries.check(_wheel(tmp_path, {"m/__init__.py": b""}), "win_arm64")


class TestWorkflow:
    def test_linux_cuda_comes_from_the_sbsa_repo_on_aarch64(self, workflow):
        for job in ("warm", "build"):
            step = next(
                s
                for s in workflow["jobs"][job]["steps"]
                if s.get("name") == "Install the CUDA ${{ matrix.cuda_tag }} toolkit"
            )
            assert (
                step["env"]["CUDA_REPO_ARCH"]
                == "${{ matrix.platform == 'linux_aarch64' && 'sbsa' || 'x86_64' }}"
            )
            assert "ubuntu2204/$CUDA_REPO_ARCH/" in step["run"]

    def test_both_windows_ccache_builds_are_pinned(self, workflow):
        for job in ("warm", "build"):
            run = next(
                s
                for s in workflow["jobs"][job]["steps"]
                if s.get("name") == "Install ccache (Windows)"
            )["run"]
            assert "6219f3865ca59aec41ee4b678df171d5d35855ecb2b6dbbbd20690b3a68af7b4" in run
            assert "28121851c8c2c8b76214cfc5e119d769fbc39e93d27862e4e3a556905ebd23b2" in run

    def test_torch_comes_from_the_cell_and_the_interpreter_is_checked(self, workflow):
        for job in ("warm", "build"):
            step = next(
                s
                for s in workflow["jobs"][job]["steps"]
                if s.get("name") == "Install ${{ matrix.torch_spec }}"
            )
            assert '"$TORCH_SPEC" $TORCH_INDEX_ARGS' in step["run"]
            assert 'os.environ["PY_PLATFORM"]' in step["run"]
            assert step["env"]["PY_PLATFORM"] == "${{ matrix.py_platform }}"

    def test_the_smoke_test_checks_the_binaries_first(self, workflow):
        run = next(
            s
            for s in workflow["jobs"]["build"]["steps"]
            if s.get("name") == "Smoke test the wheel in a fresh venv"
        )["run"]
        assert run.index("prebuilt_wheels_check_binaries.py") < run.index('python -m venv "$smoke"')
        assert '"$TORCH_SPEC" $TORCH_INDEX_ARGS' in run
