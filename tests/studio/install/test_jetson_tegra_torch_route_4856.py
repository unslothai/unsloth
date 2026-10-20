# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Jetson (Tegra) before CUDA 13 never gets the generic aarch64 CUDA torch (#4856).

download.pytorch.org's aarch64 CUDA wheels are SBSA server builds that see no GPU on JetPack 5/6,
so both installers take the CPU index a GPU-less host gets, keep a JetPack torch already in the
venv, and leave Thor (CUDA 13, SBSA), explicit pins and every non-Tegra host on today's route.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SH = PACKAGE_ROOT / "install.sh"

_STACK_PATH = PACKAGE_ROOT / "studio" / "install_python_stack.py"
_STACK_SPEC = importlib.util.spec_from_file_location(
    "studio_install_python_stack_tegra", _STACK_PATH
)
assert _STACK_SPEC is not None and _STACK_SPEC.loader is not None
stack_mod = importlib.util.module_from_spec(_STACK_SPEC)
sys.modules[_STACK_SPEC.name] = stack_mod
_STACK_SPEC.loader.exec_module(stack_mod)

BASE = "https://download.pytorch.org/whl"

# The helpers tests/sh/test_get_torch_index_url.sh extracts for get_torch_index_url.
_SH_FUNCS = (
    "_run_bounded _cvd_hides_nvidia _has_amd_rocm_gpu _has_usable_nvidia_gpu "
    "_rocm_torch_explicitly_requested _ensure_rocm_probe_env _probe_amd_gfx_arch "
    "_amd_gpu_present_via_pci _infer_amd_gfx_arch_from_gpu_name _infer_linux_amd_gfx_arch "
    "_amd_arch_index_family_for_gfx _amd_probe_arches _amd_agreed_index_family "
    "_amd_sole_index_arch _hsa_override_gfx_arch _trim_index_path_slashes "
    "_nvidia_library_inventory _nvidia_driver_cuda_version _nvidia_cu126_verdict "
    "_cap_cuda_family_for_pre_turing _rocm_tag_from_amd_smi _rocm_tag_from_version_file "
    "_rocm_tag_from_hipconfig _rocm_tag_from_dpkg _rocm_tag_from_rpm _highest_rocm_tag "
    "_detect_rocm_version_tag _rocm_bnb_compatible_generic_tag get_torch_index_url"
).split()


def _extract(src: str, name: str) -> str:
    start = src.find(f"\n{name}() {{")
    assert start >= 0, f"install.sh no longer defines {name}()"
    end = src.find("\n}\n", start)
    return src[start + 1 : end + 2]


def _host(
    tmp_path: Path,
    *,
    tegra: str = "",
    cuda: "str | None" = "12.6",
    gpu: bool = True,
    arch: str = "aarch64",
    gpu_name: str = "Orin (nvgpu)",
) -> Path:
    """A fake host root: tegra "" | "release" | "dt" | "dt-other", nvidia-smi on PATH when gpu."""
    root = tmp_path / "host"
    (root / "bin").mkdir(parents = True)
    (root / "etc").mkdir()
    (root / "dt").mkdir()
    (root / "rocm").mkdir()
    if tegra == "release":
        (root / "etc" / "nv_tegra_release").write_text("# R36 (release), REVISION: 4.3\n")
    elif tegra == "dt":
        (root / "dt" / "compatible").write_bytes(b"nvidia,p3768-0000+p3767-0005\0nvidia,tegra234\0")
    elif tegra == "dt-other":
        (root / "dt" / "compatible").write_bytes(b"nvidia,grace-hopper\0")
    uname = root / "bin" / "uname"
    uname.write_text(f'#!/bin/sh\ncase "$1" in -m) echo {arch} ;; *) echo Linux ;; esac\n')
    uname.chmod(0o755)
    if gpu:
        banner = f"CUDA Version: {cuda}" if cuda else "Driver Version: 540.4.0"
        smi = root / "bin" / "nvidia-smi"
        smi.write_text(
            "#!/bin/sh\n"
            'case "$1" in\n'
            f'  -L) echo "GPU 0: {gpu_name} (UUID: GPU-fake)" ;;\n'
            "  --query-gpu=compute_cap) echo 8.7 ;;\n"
            f'  *) echo "| NVIDIA-SMI 540.4.0  {banner} |" ;;\n'
            "esac\n"
        )
        smi.chmod(0o755)
    return root


def _run_install_sh(
    root: Path,
    env: "dict[str, str] | None" = None,
    arch: str = "aarch64",
):
    src = INSTALL_SH.read_text(encoding = "utf-8")
    body = "\n".join(_extract(src, name) for name in _SH_FUNCS)
    # Point every absolute path into the fake root so the real host cannot leak in.
    for real, fake in (
        ("/usr/bin/nvidia-smi", root / "absent-nvidia-smi"),
        ("/opt/rocm", root / "rocm"),
        ("/proc/driver/nvidia/version", root / "absent-nvidia-version"),
        ("/proc/driver/nvidia/gpus", root / "absent-nvidia-gpus"),
        ("/etc/nv_tegra_release", root / "etc" / "nv_tegra_release"),
        ("/proc/device-tree/compatible", root / "dt" / "compatible"),
    ):
        body = body.replace(real, str(fake))
    script = root / "probe.sh"
    script.write_text(body + f"\n_ARCH={arch}\nget_torch_index_url\n")
    tools = root / "tools"
    tools.mkdir(exist_ok = True)
    for cmd in ("bash", "sh", "grep", "sed", "head", "cat", "awk", "printf", "tr", "ls", "timeout"):
        real = shutil.which(cmd)
        if real and not (tools / cmd).exists():
            (tools / cmd).symlink_to(real)
    run_env = {
        "PATH": f"{root / 'bin'}:{tools}",
        "HOME": str(root),
        "UNSLOTH_NVIDIA_LIBRARY_PROBE": "0",
        "UNSLOTH_DISABLE_XPU_AUTO": "1",
    }
    run_env.update(env or {})
    out = subprocess.run(
        [shutil.which("bash") or "bash", str(script)],
        capture_output = True,
        text = True,
        timeout = 60,
        env = run_env,
    )
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "", out.stderr


_SH_CELLS = [
    ("jp6_release_cuda126", dict(tegra = "release", cuda = "12.6"), {}, "cpu", True),
    ("jp6_release_cuda122", dict(tegra = "release", cuda = "12.2"), {}, "cpu", True),
    ("jp5_devicetree_cuda114", dict(tegra = "dt", cuda = "11.4"), {}, "cpu", True),
    ("jp6_devicetree_cuda128", dict(tegra = "dt", cuda = "12.8"), {}, "cpu", True),
    ("jetson_unknown_cuda", dict(tegra = "release", cuda = None), {}, "cpu", True),
    ("thor_cuda130_keeps_ladder", dict(tegra = "release", cuda = "13.0"), {}, "cu130", False),
    ("thor_cuda132_keeps_ladder", dict(tegra = "dt", cuda = "13.2"), {}, "cu130", False),
    (
        "jetson_family_pin_wins",
        dict(tegra = "release"),
        {"UNSLOTH_TORCH_INDEX_FAMILY": "cu126"},
        "cu126",
        False,
    ),
    (
        "jetson_url_pin_wins",
        dict(tegra = "release"),
        {"UNSLOTH_TORCH_INDEX_URL": "https://pypi.jetson-ai-lab.io/jp6/cu126"},
        "https://pypi.jetson-ai-lab.io/jp6/cu126",
        False,
    ),
    ("jetson_no_gpu_seen", dict(tegra = "release", gpu = False), {}, "cpu", False),
    ("gh200_not_tegra", dict(tegra = "dt-other", cuda = "12.8"), {}, "cu128", False),
    ("arm64_sbsa_no_tegra_files", dict(cuda = "12.6"), {}, "cu126", False),
    ("arm64_unknown_cuda_not_tegra", dict(cuda = None), {}, "cu126", False),
    (
        "x86_cuda126",
        dict(cuda = "12.6", arch = "x86_64", gpu_name = "NVIDIA RTX 4090"),
        {},
        "cu126",
        False,
    ),
    ("x86_no_gpu", dict(gpu = False, arch = "x86_64"), {}, "cpu", False),
]


# install.sh runs on Linux / macOS only; Git Bash on Windows lacks the tools it shells out to.
_POSIX = pytest.mark.skipif(sys.platform == "win32", reason = "install.sh is POSIX-only")


@_POSIX
@pytest.mark.parametrize(
    "host_kw, env, expected, warns", [c[1:] for c in _SH_CELLS], ids = [c[0] for c in _SH_CELLS]
)
def test_install_sh_jetson_torch_index(tmp_path, host_kw, env, expected, warns):
    arch = host_kw.get("arch", "aarch64")
    root = _host(tmp_path, **host_kw)
    url, err = _run_install_sh(root, env, arch = arch)
    assert url == (expected if expected.startswith("https://") else f"{BASE}/{expected}"), err
    assert ("NVIDIA Jetson (Tegra) detected" in err) is warns, err
    if warns:
        assert "install-pytorch-jetson-platform" in err


def _run_cpu_note(
    tmp_path: Path,
    tegra: str,
    pinned: bool = False,
    cuda: str = "",
) -> str:
    """install.sh's "PyTorch wheel index note" block for a CPU index, on a fake root."""
    root = _host(tmp_path, tegra = tegra, gpu = False)
    src = INSTALL_SH.read_text(encoding = "utf-8")
    start = src.index("# ── PyTorch wheel index note ──")
    block = src[start : src.index("\nesac\n", start) + 6]
    for real, fake in (
        ("/etc/nv_tegra_release", root / "etc" / "nv_tegra_release"),
        ("/proc/device-tree/compatible", root / "dt" / "compatible"),
    ):
        block = block.replace(real, str(fake))
    script = root / "note.sh"
    script.write_text(
        'substep() { echo "$1"; }\n_has_amd_rocm_gpu() { return 1; }\n'
        f'_nvidia_driver_cuda_version() {{ [ -n "{cuda}" ] && echo "{cuda}"; }}\n'
        f"TORCH_INDEX_URL={BASE}/cpu SKIP_TORCH=false OS=linux C_WARN=\n"
        f"_torch_index_pinned={'true' if pinned else 'false'}\n" + block
    )
    out = subprocess.run(
        [shutil.which("bash") or "bash", str(script)], capture_output = True, text = True, timeout = 60
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


@_POSIX
@pytest.mark.parametrize("tegra", ["release", "dt"])
def test_install_sh_cpu_note_keeps_the_jetson_diagnosis(tmp_path, tegra):
    out = _run_cpu_note(tmp_path, tegra)
    assert "NVIDIA Jetson detected" in out
    # The JetPack remedy is here too: a Jetson nvidia-smi cannot see returns before get_torch_index_url's warning.
    assert "install-pytorch-jetson-platform" in out
    assert "No GPU detected" not in out and "AMD ROCm users" not in out
    assert "--no-torch" in out


@_POSIX
def test_install_sh_cpu_note_leaves_thor_off_the_jetson_diagnosis(tmp_path):
    # Thor (CUDA 13) keeps the generic SBSA route; reaching the CPU index there is some other failure.
    out = _run_cpu_note(tmp_path, "release", cuda = "13.0")
    assert "NVIDIA Jetson detected" not in out and "No GPU detected" in out
    assert "NVIDIA Jetson detected" in _run_cpu_note(tmp_path / "jp6", "release", cuda = "12.6")


@_POSIX
@pytest.mark.parametrize("tegra, pinned", [("", False), ("dt-other", False), ("release", True)])
def test_install_sh_cpu_note_unchanged_off_the_jetson_route(tmp_path, tegra, pinned):
    out = _run_cpu_note(tmp_path, tegra, pinned)
    assert "NVIDIA Jetson detected" not in out
    if pinned:
        assert "index pinned" in out
    else:
        assert "No GPU detected" in out and "AMD ROCm users" in out


# ── install_python_stack.py: the family the update / repair path would install ──


def _detect_family(
    monkeypatch,
    *,
    tegra: bool,
    cuda: "str | None",
    inventory = None,
    known_only = False,
):
    monkeypatch.setattr(stack_mod, "_is_tegra", lambda: tegra, raising = False)
    monkeypatch.setattr(stack_mod, "_TEGRA_WARNED", False, raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_FAMILY", raising = False)
    smi = ["/fake/nvidia-smi"] if cuda else []

    def _run(cmd, *a, **k):
        res = MagicMock()
        res.returncode = 0
        res.stdout = f"| NVIDIA-SMI 540.4.0  CUDA Version: {cuda} |\n"
        return res

    with (
        patch.object(stack_mod, "_nvidia_smi_usable_candidates", return_value = smi),
        patch.object(stack_mod, "_nvidia_smi_lists_a_gpu", return_value = True),
        patch.object(stack_mod, "_nvidia_library_inventory", return_value = inventory),
        patch.object(stack_mod, "_cap_cuda_family_for_pre_turing", side_effect = lambda t, *a: t),
        patch.object(stack_mod.subprocess, "run", side_effect = _run),
    ):
        return stack_mod._detect_cuda_torch_index_family(known_only = known_only)


_INV_126 = SimpleNamespace(cuda_driver_version = (12, 6), devices = [{"compute_cap": "8.7"}])
_INV_130 = SimpleNamespace(cuda_driver_version = (13, 0), devices = [{"compute_cap": "11.0"}])

_PY_CELLS = [
    ("jp6_smi_126", True, "12.6", None, False, "cpu"),
    ("jp5_smi_114", True, "11.4", None, False, "cpu"),
    ("jp6_library_126", True, None, _INV_126, False, "cpu"),
    ("jetson_unknown_default", True, None, None, False, "cpu"),
    ("jetson_unknown_known_only", True, None, None, True, None),
    ("thor_smi_130", True, "13.0", None, False, "cu130"),
    ("thor_library_130", True, None, _INV_130, False, "cu130"),
    ("non_tegra_smi_126", False, "12.6", None, False, "cu126"),
    ("non_tegra_smi_128", False, "12.8", None, False, "cu128"),
    ("non_tegra_library_126", False, None, _INV_126, False, "cu126"),
    ("non_tegra_unknown_default", False, None, None, False, "cu126"),
    ("non_tegra_ancient", False, "10.2", None, False, "cpu"),
]


@pytest.mark.parametrize(
    "tegra, cuda, inventory, known_only, expected",
    [c[1:] for c in _PY_CELLS],
    ids = [c[0] for c in _PY_CELLS],
)
def test_stack_jetson_cuda_family(monkeypatch, tegra, cuda, inventory, known_only, expected):
    got = _detect_family(
        monkeypatch, tegra = tegra, cuda = cuda, inventory = inventory, known_only = known_only
    )
    assert got == expected


def test_stack_jetson_pins_win(monkeypatch):
    monkeypatch.setattr(stack_mod, "_is_tegra", lambda: True, raising = False)
    monkeypatch.delenv("UNSLOTH_TORCH_INDEX_URL", raising = False)
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_FAMILY", "cu126")
    assert stack_mod._detect_cuda_torch_index_family() == "cu126"
    monkeypatch.setenv("UNSLOTH_TORCH_INDEX_URL", "https://pypi.jetson-ai-lab.io/jp6/cu126/")
    assert stack_mod._detect_cuda_torch_index_family() == "cu126"


def _repair(
    monkeypatch,
    *,
    tegra: bool,
    version: str,
    cuda: str,
    smi_cuda: str = "12.6",
):
    """_ensure_cuda_torch on an aarch64 NVIDIA host whose installed torch is `version`."""
    monkeypatch.setattr(stack_mod, "_is_tegra", lambda: tegra, raising = False)
    for var in (
        "UNSLOTH_TORCH_INDEX_URL",
        "UNSLOTH_TORCH_INDEX_FAMILY",
        "UNSLOTH_ROCM_TORCH_INSTALLED",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(var, raising = False)

    def _run(cmd, *a, **k):
        res = MagicMock()
        res.returncode = 0
        res.stdout = f"CUDA Version: {smi_cuda}\n"
        return res

    with (
        patch.object(stack_mod, "_TORCH_BACKEND", ""),
        patch.object(stack_mod, "IS_MACOS", False),
        patch.object(stack_mod, "IS_WINDOWS", False),
        patch.object(stack_mod, "NO_TORCH", False),
        patch.object(stack_mod, "_RECORDED_TORCH_TAG", None),
        patch.object(stack_mod.platform, "machine", return_value = "aarch64"),
        patch.object(stack_mod, "_rocm_torch_explicitly_requested", return_value = False),
        patch.object(stack_mod, "_has_usable_nvidia_gpu", return_value = True),
        patch.object(stack_mod, "_nvidia_smi_usable_candidates", return_value = ["/fake/nvidia-smi"]),
        patch.object(stack_mod, "_nvidia_smi_lists_a_gpu", return_value = True),
        patch.object(stack_mod, "_cap_cuda_family_for_pre_turing", side_effect = lambda t, *a: t),
        patch.object(
            stack_mod, "_probe_torch_runtime", return_value = (True, True, version, "", cuda)
        ),
        patch.object(stack_mod.subprocess, "run", side_effect = _run),
        patch.object(stack_mod, "pip_install") as pip,
    ):
        stack_mod._ensure_cuda_torch()
    return pip


def test_stack_jetson_cpu_torch_not_replaced_by_sbsa_wheel(monkeypatch):
    """A CPU torch on JetPack 6 was 'repaired' to the cu126 SBSA wheel, which sees no GPU."""
    pip = _repair(monkeypatch, tegra = True, version = "2.11.0+cpu", cuda = "")
    pip.assert_not_called()


def test_stack_jetson_jetpack_torch_kept(monkeypatch):
    pip = _repair(monkeypatch, tegra = True, version = "2.5.0a0+872d972e41.nv24.08", cuda = "12.6")
    pip.assert_not_called()


def test_stack_non_tegra_arm64_cpu_torch_still_repaired(monkeypatch):
    """GH200 / DGX Spark: a stray CPU torch on an NVIDIA aarch64 host is still repaired."""
    pip = _repair(monkeypatch, tegra = False, version = "2.11.0+cpu", cuda = "", smi_cuda = "12.8")
    assert pip.call_count == 1
    args = [str(a) for a in pip.call_args.args]
    assert args[args.index("--index-url") + 1].endswith("/cu128")


def test_stack_thor_cpu_torch_still_repaired(monkeypatch):
    pip = _repair(monkeypatch, tegra = True, version = "2.11.0+cpu", cuda = "", smi_cuda = "13.0")
    assert pip.call_count == 1


@pytest.mark.parametrize(
    "release, compatible, expected",
    [
        (True, None, True),
        (False, b"nvidia,p3768-0000+p3767-0005\0nvidia,tegra234\0", True),
        (False, b"nvidia,grace-hopper\0", False),
        (False, None, False),
    ],
    ids = ["release_file", "device_tree_tegra", "device_tree_other", "neither"],
)
def test_stack_is_tegra_reads_release_and_device_tree(
    monkeypatch, tmp_path, release, compatible, expected
):
    rel = tmp_path / "nv_tegra_release"
    dt = tmp_path / "compatible"
    if release:
        rel.write_text("# R35 (release), REVISION: 4.1\n")
    if compatible is not None:
        dt.write_bytes(compatible)
    monkeypatch.setattr(stack_mod, "_TEGRA_RELEASE", str(rel), raising = False)
    monkeypatch.setattr(stack_mod, "_DEVICE_TREE_COMPATIBLE", str(dt), raising = False)
    stack_mod._is_tegra.cache_clear()
    try:
        assert stack_mod._is_tegra() is expected
    finally:
        stack_mod._is_tegra.cache_clear()


def test_tegra_detection_matches_the_studio_runtime_helper():
    """One convention: the installer's paths are the ones studio/backend/utils/tegra.py reads."""
    helper = (PACKAGE_ROOT / "studio" / "backend" / "utils" / "tegra.py").read_text(
        encoding = "utf-8"
    )
    sh = INSTALL_SH.read_text(encoding = "utf-8")
    for path in ("/etc/nv_tegra_release", "/proc/device-tree/compatible"):
        assert path in helper and path in sh
    assert "nvidia,tegra" in helper and "nvidia,tegra" in sh
    assert getattr(stack_mod, "_TEGRA_RELEASE", "") == "/etc/nv_tegra_release"
    assert getattr(stack_mod, "_DEVICE_TREE_COMPATIBLE", "") == "/proc/device-tree/compatible"
