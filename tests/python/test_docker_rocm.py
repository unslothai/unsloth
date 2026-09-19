# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The AMD ROCm image's host-side plumbing: `docker/run.sh --rocm`,
`docker/build.sh --rocm`, and the container entrypoint's refusal paths.

None of these need an AMD GPU. run.sh and build.sh are driven with a recording
`docker` stub and a staged /dev tree (the UNSLOTH_DEV_ROOT idiom); the
entrypoint is driven the same way with stub `rocm-smi` and `python` binaries,
so every message a user can hit before torch loads is checked here.
"""

import os
import shutil
import stat
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
_DOCKER = os.path.join(_REPO, "docker")
_RUN_SH = os.path.join(_DOCKER, "run.sh")
_BUILD_SH = os.path.join(_DOCKER, "build.sh")
_ENTRYPOINT = os.path.join(_DOCKER, "entrypoint-rocm.sh")
_DOCKERFILE = os.path.join(_DOCKER, "Dockerfile.rocm")
_SMOKE = os.path.join(_DOCKER, "smoke_test_rocm.py")
_WORKFLOW = os.path.join(_REPO, ".github", "workflows", "docker-publish-rocm.yml")
_HUB_PAGE = os.path.join(_DOCKER, "DOCKERHUB-ROCM.md")
_README = os.path.join(_REPO, "README.md")

_posix_shell = pytest.mark.skipif(
    os.name != "posix" or shutil.which("bash") is None,
    reason = "POSIX shell required",
)


def _stub(path, body):
    with open(path, "w", encoding = "utf-8") as f:
        f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


# ── run.sh --rocm ────────────────────────────────────────────────────────────


def _run_sh(
    tmp_path,
    args,
    *,
    kfd = True,
    dri = True,
    dxg = False,
    librocdxg = False,
    wsl_lib = True,
    nvidia = False,
    groups = "both",
    extra_env = None,
):
    # a fresh sandbox per call: a test may drive run.sh twice
    tmp_path = tmp_path / f"run{len(os.listdir(tmp_path))}"
    tmp_path.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "argv"
    _stub(
        str(bindir / "docker"),
        'if [ "$1" = "info" ]; then echo " Runtimes: io.containerd.runc.v2 runc"; exit 0; fi\n'
        'printf "%s\\n" "$@" > ' + str(argv_log) + "\nexit 0\n",
    )
    if nvidia:
        _stub(str(bindir / "nvidia-smi"), 'echo "GPU 0: NVIDIA H100 (UUID: GPU-abc)"\n')
    else:
        _stub(str(bindir / "nvidia-smi"), "exit 1\n")
    known = {"both": ("44", "992"), "none": (None, None)}
    vid, ren = known[groups]
    _stub(
        str(bindir / "getent"),
        'case "$2" in\n'
        + (f'  video)  echo "video:x:{vid}:"; exit 0 ;;\n' if vid else "  video)  exit 2 ;;\n")
        + (f'  render) echo "render:x:{ren}:"; exit 0 ;;\n' if ren else "  render) exit 2 ;;\n")
        + "esac\nexit 2\n",
    )
    dev_root = tmp_path / "root"
    (dev_root / "dev").mkdir(parents = True)
    if nvidia:
        (dev_root / "dev" / "nvidiactl").write_text("")
    if kfd:
        (dev_root / "dev" / "kfd").write_text("")
        if dri:
            (dev_root / "dev" / "dri").mkdir()
    if dxg:
        (dev_root / "dev" / "dxg").write_text("")
    if librocdxg:
        lib = dev_root / "opt" / "rocm" / "lib"
        lib.mkdir(parents = True)
        (lib / "librocdxg.so.1.2.1").write_text("")
    if wsl_lib:
        (dev_root / "usr" / "lib" / "wsl" / "lib").mkdir(parents = True)
    env = dict(os.environ)
    env["PATH"] = str(bindir) + ":/usr/bin:/bin"
    env["UNSLOTH_DEV_ROOT"] = str(dev_root)
    env["HOME"] = str(tmp_path / "home")
    env["UNSLOTH_WORKDIR"] = str(tmp_path)
    for leak in (
        "HF_TOKEN",
        "WANDB_API_KEY",
        "UNSLOTH_GPUS",
        "UNSLOTH_ALLOW_CPU",
        "UNSLOTH_STUDIO_VOLUME",
        "UNSLOTH_IMAGE",
        "UNSLOTH_ROCM",
        "HSA_OVERRIDE_GFX_VERSION",
        "UNSLOTH_ROCM_GFX_ARCH",
    ):
        env.pop(leak, None)
    env.update(extra_env or {})
    proc = subprocess.run(
        [shutil.which("bash") or "/bin/bash", _RUN_SH, *args],
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert proc.returncode == 0, f"run.sh failed: {proc.stderr}"
    argv = argv_log.read_text().splitlines()
    return argv, proc.stderr


def _image_and_cmd(argv):
    """The positional tail of `docker run`: image, then the container command."""
    # every option run.sh emits takes a value or is a known flag
    flags_with_value = {
        "--device",
        "--group-add",
        "--ulimit",
        "-v",
        "-e",
        "-p",
        "--gpus",
        "--stop-timeout",
    }
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("run", "--rm", "-it", "--ipc=host"):
            i += 1
        elif a in flags_with_value:
            i += 2
        elif a.startswith("-"):
            i += 1
        else:
            return argv[i], argv[i + 1 :]
    raise AssertionError(f"no image in {argv}")


@_posix_shell
class TestRunShRocm:
    def test_rocm_passes_the_device_nodes_and_numeric_gids_not_gpus(self, tmp_path):
        argv, _ = _run_sh(tmp_path, ["--rocm", "true"])
        assert "--gpus" not in argv
        assert "/dev/kfd" in argv and "/dev/dri" in argv
        gids = [argv[i + 1] for i, a in enumerate(argv) if a == "--group-add"]
        assert gids == ["44", "992"], argv
        image, cmd = _image_and_cmd(argv)
        assert image == "unsloth/unsloth-rocm:latest"
        assert cmd == ["true"]

    def test_a_missing_dri_node_is_left_out_so_docker_still_starts(self, tmp_path):
        """docker rejects a --device path that does not exist on the host."""
        argv, err = _run_sh(tmp_path, ["--rocm", "true"], dri = False)
        assert "/dev/kfd" in argv and "/dev/dri" not in argv, argv
        assert "/dev/dri is not" in err, err

    def test_the_wrapper_option_is_only_taken_from_the_front(self, tmp_path):
        """A container command's own --rocm belongs to that command."""
        argv, _ = _run_sh(tmp_path, ["--rocm", "python", "train.py", "--rocm"])
        _, cmd = _image_and_cmd(argv)
        assert cmd == ["python", "train.py", "--rocm"], argv

    def test_a_later_rocm_is_not_the_wrapper_option(self, tmp_path):
        argv, _ = _run_sh(tmp_path, ["python", "train.py", "--rocm"], nvidia = True)
        assert "--gpus" in argv, argv
        image, cmd = _image_and_cmd(argv)
        assert image == "unsloth/unsloth:latest"
        assert cmd == ["python", "train.py", "--rocm"]

    def test_the_env_form_selects_rocm_too(self, tmp_path):
        argv, _ = _run_sh(tmp_path, ["true"], extra_env = {"UNSLOTH_ROCM": "1"})
        image, _ = _image_and_cmd(argv)
        assert image == "unsloth/unsloth-rocm:latest"
        assert "/dev/kfd" in argv

    def test_no_kfd_and_no_dxg_warns_and_starts_without_devices(self, tmp_path):
        """A host with neither node has no GPU to pass: the container must still start
        rather than docker failing on a missing device."""
        argv, stderr = _run_sh(tmp_path, ["--rocm", "true"], kfd = False)
        assert "--device" not in argv and "--gpus" not in argv, argv
        assert "/dev/kfd is not present" in stderr, stderr
        assert "/dev/dxg" in stderr, stderr

    def test_wsl_passes_dxg_instead_of_kfd(self, tmp_path):
        """WSL2 has no /dev/kfd: the card is reached over the DXG bridge, so the flags
        are the device plus the runtime's opt-in, and librocdxg off the host (its cmake
        build needs Windows SDK headers, so no Linux image build can carry it)."""
        argv, stderr = _run_sh(
            tmp_path,
            ["--rocm", "true"],
            kfd = False,
            dxg = True,
            librocdxg = True,
        )
        assert "/dev/dxg" in argv, argv
        assert "/dev/kfd" not in argv, argv
        assert "HSA_ENABLE_DXG_DETECTION=1" in argv, argv
        assert any("librocdxg.so" in a for a in argv), argv
        # librocdxg dlopens libdxcore from here; without the mount hsa_init fails
        # (measured on an R9700: "Failed to load libdxcore.so")
        assert "/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro" in argv, argv
        assert "LD_LIBRARY_PATH=/usr/lib/wsl/lib" in argv, argv
        assert "--gpus" not in argv, argv
        assert "DXG" in stderr or "dxg" in stderr, stderr

    def test_a_missing_wsl_lib_dir_is_named_not_silently_dropped(self, tmp_path):
        argv, stderr = _run_sh(
            tmp_path,
            ["--rocm", "true"],
            kfd = False,
            dxg = True,
            librocdxg = True,
            wsl_lib = False,
        )
        assert "LD_LIBRARY_PATH=/usr/lib/wsl/lib" not in argv, argv
        assert "/usr/lib/wsl/lib is missing" in stderr, stderr

    def test_dxg_without_librocdxg_warns_and_points_at_the_helper(self, tmp_path):
        """/dev/dxg alone is not enough: without the bridge library the runtime cannot
        reach the card, and the fix is the WSL ROCm helper, not a docker flag."""
        argv, stderr = _run_sh(tmp_path, ["--rocm", "true"], kfd = False, dxg = True)
        assert "librocdxg" in stderr, stderr
        assert "install_rocm_wsl_strixhalo.sh" in stderr, stderr

    def test_a_mixed_host_is_not_offered_the_nvidia_toolkit(self, tmp_path):
        """An NVIDIA + AMD box under --rocm runs the ROCm image through the AMD nodes;
        the toolkit prompt is for the --gpus path only."""
        argv, stderr = _run_sh(
            tmp_path, ["--rocm", "true"], nvidia = True, extra_env = {"UNSLOTH_INSTALL_TOOLKIT": "0"}
        )
        assert "--gpus" not in argv
        assert "/dev/kfd" in argv
        assert "Container Toolkit" not in stderr and "no NVIDIA GPU" not in stderr, stderr

    def test_missing_groups_degrade_to_the_devices_alone(self, tmp_path):
        argv, _ = _run_sh(tmp_path, ["--rocm", "true"], groups = "none")
        assert "/dev/kfd" in argv and "--group-add" not in argv, argv

    def test_the_gfx_overrides_are_forwarded_only_when_set(self, tmp_path):
        argv, _ = _run_sh(tmp_path, ["--rocm", "true"])
        assert "HSA_OVERRIDE_GFX_VERSION" not in argv and "UNSLOTH_ROCM_GFX_ARCH" not in argv
        argv, _ = _run_sh(
            tmp_path,
            ["--rocm", "true"],
            extra_env = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0", "UNSLOTH_ROCM_GFX_ARCH": "gfx1151"},
        )
        env_flags = [argv[i + 1] for i, a in enumerate(argv) if a == "-e"]
        # the dash-only form: docker reads the value from the environment, so it
        # never lands in argv
        assert "HSA_OVERRIDE_GFX_VERSION" in env_flags and "UNSLOTH_ROCM_GFX_ARCH" in env_flags

    def test_the_studio_volume_and_caches_are_mounted_like_the_cuda_path(self, tmp_path):
        argv, _ = _run_sh(tmp_path, ["--rocm", "true"])
        mounts = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
        assert "unsloth-studio:/opt/unsloth-studio" in mounts, mounts
        assert any(m.endswith(":/workspace/.cache/huggingface") for m in mounts), mounts

    def test_a_custom_image_is_kept(self, tmp_path):
        argv, _ = _run_sh(tmp_path, ["--rocm", "true"], extra_env = {"UNSLOTH_IMAGE": "me/rocm:dev"})
        image, _ = _image_and_cmd(argv)
        assert image == "me/rocm:dev"

    def test_no_mapfile(self):
        """run.sh runs on the host, and macOS ships bash 3.2, which has no mapfile."""
        code = "\n".join(
            ln
            for ln in open(_RUN_SH, encoding = "utf-8").read().splitlines()
            if not ln.lstrip().startswith("#")
        )
        assert "mapfile" not in code and "readarray" not in code

    def test_the_plain_nvidia_path_is_unchanged(self, tmp_path):
        argv, stderr = _run_sh(tmp_path, ["true"], nvidia = True)
        assert argv[argv.index("--gpus") + 1] == "all"
        assert "/dev/kfd" not in argv and "--group-add" not in argv
        assert "rocm" not in stderr.lower(), stderr


# ── build.sh --rocm ──────────────────────────────────────────────────────────

_SHA_U = "a" * 40
_SHA_Z = "b" * 40


def _build_sh(
    tmp_path,
    args,
    extra_env = None,
    expect_rc = 0,
):
    tmp_path = tmp_path / f"build{len(os.listdir(tmp_path))}"
    tmp_path.mkdir()
    bindir = tmp_path / "bin"
    bindir.mkdir()
    args_file = tmp_path / "docker-args.txt"
    _stub(str(bindir / "docker"), f'printf "%s\\n" "$@" > {args_file}\n')
    _stub(
        str(bindir / "git"),
        'if [ "$1" = "ls-remote" ]; then\n'
        '  case "$2" in\n'
        f'    *unsloth-zoo*) echo -e "{_SHA_Z}\\tHEAD" ;;\n'
        '    *notebooks*) echo "ls-remote notebooks should not run for --rocm" >&2; exit 3 ;;\n'
        f'    *) echo -e "{_SHA_U}\\tHEAD" ;;\n'
        "  esac\n  exit 0\nfi\nexit 0\n",
    )
    _stub(str(bindir / "curl"), 'echo "curl should not run for --rocm" >&2; exit 1\n')
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env['PATH']}"
    for leak in ("ROCM_GFX", "ROCM_VERSION", "TORCH_INDEX_URL", "TAG", "IMAGE_NAME"):
        env.pop(leak, None)
    env.update(extra_env or {})
    proc = subprocess.run(
        ["bash", _BUILD_SH, *args], env = env, capture_output = True, text = True, cwd = str(tmp_path)
    )
    assert proc.returncode == expect_rc, proc.stdout + proc.stderr
    argv = args_file.read_text(encoding = "utf-8").splitlines() if args_file.exists() else []
    return proc, argv


def _build_arg(argv, name):
    for i, item in enumerate(argv):
        if item == "--build-arg" and argv[i + 1].startswith(f"{name}="):
            return argv[i + 1].split("=", 1)[1]
    raise AssertionError(f"--build-arg {name} was never passed: {argv}")


@_posix_shell
class TestBuildShRocm:
    def test_defaults_are_rocm72_frozen_refs_and_no_cuda_lookups(self, tmp_path):
        proc, argv = _build_sh(tmp_path, ["--rocm"])
        assert "Dockerfile.rocm" in argv
        assert _build_arg(argv, "ROCM_VERSION") == "7.2.4"
        assert _build_arg(argv, "TORCH_INDEX_URL") == "https://download.pytorch.org/whl/rocm7.2"
        assert _build_arg(argv, "ROCM_GFX") == ""
        assert _build_arg(argv, "PYTHON_VERSION") == "3.12"
        assert _build_arg(argv, "UNSLOTH_REF") == _SHA_U
        assert _build_arg(argv, "UNSLOTH_ZOO_REF") == _SHA_Z
        assert argv[argv.index("-t") + 1] == "unsloth-rocm:latest"
        assert "should not run" not in proc.stderr
        assert "run.sh --rocm" in proc.stdout

    def test_gfx_selects_the_per_arch_wheels(self, tmp_path):
        _, argv = _build_sh(tmp_path, ["--rocm", "--gfx", "gfx1151"])
        assert _build_arg(argv, "ROCM_GFX") == "gfx1151"
        _, argv = _build_sh(tmp_path, ["--rocm", "--gfx=gfx1201"])
        assert _build_arg(argv, "ROCM_GFX") == "gfx1201"
        _, argv = _build_sh(tmp_path, ["--rocm"], extra_env = {"ROCM_GFX": "gfx1150"})
        assert _build_arg(argv, "ROCM_GFX") == "gfx1150"

    def test_the_index_follows_the_rocm_version_unless_named(self, tmp_path):
        """ROCM_VERSION=6.3.4 alone must not pair a 6.3 base with the 7.2 wheels: the
        build would pass (torch.version.hip is set either way) and not run."""
        _, argv = _build_sh(tmp_path, ["--rocm"], extra_env = {"ROCM_VERSION": "6.3.4"})
        assert _build_arg(argv, "ROCM_VERSION") == "6.3.4"
        assert _build_arg(argv, "TORCH_INDEX_URL") == "https://download.pytorch.org/whl/rocm6.3"
        _, argv = _build_sh(
            tmp_path,
            ["--rocm"],
            extra_env = {"ROCM_VERSION": "6.3.4", "TORCH_INDEX_URL": "https://example/whl/custom"},
        )
        assert _build_arg(argv, "TORCH_INDEX_URL") == "https://example/whl/custom"
        body = open(_WORKFLOW, encoding = "utf-8").read()
        assert "https://download.pytorch.org/whl/rocm${ROCM%.*}" in body
        local = open(os.path.join(_DOCKER, "test_locally-rocm.sh"), encoding = "utf-8").read()
        assert "rocm${ROCM_VERSION%.*}" in local

    def test_the_local_end_to_end_script_builds_through_build_sh(self):
        """A bare docker build there passed mutable main refs, so a rerun after main
        moved could reuse the install layer and validate stale code."""
        local = open(os.path.join(_DOCKER, "test_locally-rocm.sh"), encoding = "utf-8").read()
        code = "\n".join(ln for ln in local.splitlines() if not ln.lstrip().startswith("#"))
        assert (
            "docker buildx build" not in code and "docker build" not in code
        ), "builds outside build.sh"
        assert 'bash "$BUILD_SH" --rocm' in code

    def test_gfx_without_rocm_is_refused(self, tmp_path):
        proc, argv = _build_sh(tmp_path, ["--gfx", "gfx1151"], expect_rc = 2)
        assert argv == [], "docker build ran anyway"
        assert "--rocm" in proc.stderr

    def test_an_unknown_option_is_refused(self, tmp_path):
        proc, argv = _build_sh(tmp_path, ["--rocm", "--bogus"], expect_rc = 2)
        assert argv == []
        assert "unknown option" in proc.stderr

    def test_the_dockerfile_defaults_match(self):
        body = open(_DOCKERFILE, encoding = "utf-8").read()
        assert "ARG ROCM_VERSION=7.2.4" in body
        assert "ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.2" in body
        assert "FROM rocm/dev-ubuntu-24.04:${ROCM_VERSION}" in body
        # a knowingly broken fallback range is worse than a failed build
        assert "bitsandbytes>=0.49" not in body
        assert "bitsandbytes>=0.50.0" in body

    def test_the_workflow_defaults_match_and_its_main_group_is_per_commit(self):
        body = open(_WORKFLOW, encoding = "utf-8").read()
        assert "DEFAULT_ROCM_VERSION: '7.2.4'" in body
        assert "DEFAULT_TORCH_INDEX_URL: 'https://download.pytorch.org/whl/rocm7.2'" in body
        assert "6.2" not in body.replace("ubuntu-22.04", "")
        # per RUN on main: a sha would still pair a scheduled run with a dispatch on
        # an unchanged main, and the group keeps only one pending run
        assert "github.ref == 'refs/heads/main' && github.run_id" in body
        assert "-r{0}', github.run_id" in body, "override sha tags need the run id suffix"
        assert "git ls-remote https://github.com/unslothai/unsloth-zoo" in body
        assert "needs.prepare.outputs.stable == 'true'" in body
        assert "org.opencontainers.image.licenses=Apache-2.0 AND AGPL-3.0-only" in body


class TestTheUserFacingDocsCoverWsl:
    """docker/DOCKERHUB-ROCM.md is synced to the Docker Hub page and README.md is the
    first thing a Windows user reads. Both said the image needs native Linux, and the
    Hub quick start passed --device /dev/kfd unconditionally, which the daemon rejects
    on WSL before the entrypoint runs."""

    def test_the_hub_page_gives_the_same_wsl_flags_as_run_sh(self):
        text = open(_HUB_PAGE, encoding = "utf-8").read()
        for needle in (
            "--device /dev/dxg",
            "HSA_ENABLE_DXG_DETECTION=1",
            "librocdxg.so.1:/usr/lib/x86_64-linux-gnu/librocdxg.so:ro",
            "-v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro",
            "LD_LIBRARY_PATH=/usr/lib/wsl/lib",
            "ROCM_GFX=<your gfx> bash docker/build.sh --rocm",
            # run.sh defaults to the published image, which is refused on DXG
            "UNSLOTH_IMAGE=unsloth-rocm:latest bash run.sh --rocm",
            # Dockerfile.rocm maps no RDNA3 arch to a per-arch index, so the page must not
            # promise one
            "RDNA3 cards (`gfx1100` to `gfx1103`) have no bridge path yet",
        ):
            assert needle in text, needle

    def test_the_readme_no_longer_says_native_linux_only(self):
        text = open(_README, encoding = "utf-8").read()
        assert "needs native Linux" not in text
        assert "/dev/dxg" in text and "docker/run.sh --rocm" in text
        assert "UNSLOTH_IMAGE=unsloth-rocm:latest" in text
        assert "RDNA3 cards have no bridge path yet" in text


# ── entrypoint-rocm.sh ───────────────────────────────────────────────────────


def _entrypoint(
    tmp_path,
    *,
    kfd = True,
    readable = True,
    dxg = False,
    smi_sees_gpu = True,
    python_body = None,
    env_extra = None,
    build_info_gfx = "",
    command = "echo ran",
):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    dev_root = tmp_path / "root"
    (dev_root / "dev").mkdir(parents = True)
    if kfd:
        (dev_root / "dev" / "kfd").write_text("")
        if not readable:
            os.chmod(dev_root / "dev" / "kfd", 0)
    if dxg:
        (dev_root / "dev" / "dxg").write_text("")
    _stub(
        str(bindir / "rocm-smi"),
        'echo "GPU[0] : GPU ID: 0x1586"\n' if smi_sees_gpu else "echo 'No AMD GPUs specified'\n",
    )
    # the two torch heredocs; stand in for torch on this host
    _stub(str(bindir / "python"), python_body or "cat > /dev/null\nexit 0\n")
    build_info = tmp_path / "build-info"
    build_info.write_text(f"TORCH_INDEX_URL=x\nROCM_GFX={build_info_gfx}\nROCM_VERSION=7.2.4\n")
    dump = tmp_path / "ran"
    env = {
        "PATH": str(bindir) + ":/usr/bin:/bin",
        "HOME": str(tmp_path),
        "UNSLOTH_DEV_ROOT": str(dev_root),
        "UNSLOTH_ROCM_BUILD_INFO": str(build_info),
    }
    env.update(env_extra or {})
    proc = subprocess.run(
        [shutil.which("bash") or "/bin/bash", _ENTRYPOINT, "bash", "-c", f"{command} > {dump}"],
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    return proc.returncode, dump.exists(), proc.stderr


@_posix_shell
def _fake_rocm_torch(tmp_path, libnames, available = True):
    """A ROCm torch on a supported arch whose lib/ holds exactly `libnames`."""
    fake = tmp_path / "fake"
    (fake / "torch" / "cuda").mkdir(parents = True)
    (fake / "torch" / "lib").mkdir()
    for name in libnames:
        (fake / "torch" / "lib" / name).write_text("")
    (fake / "torch" / "__init__.py").write_text(
        "__version__ = '2.11.0+rocm7.2'\n"
        "class version:\n    hip = '7.2.53211'\n"
        "from . import cuda\n"
    )
    (fake / "torch" / "cuda" / "__init__.py").write_text(
        "class _P:\n    gcnArchName = 'gfx1201'\n"
        f"def is_available(): return {available}\n"
        "def device_count(): return 1\n"
        "def get_device_name(i): return 'AMD Radeon AI PRO R9700'\n"
        "def get_device_properties(i): return _P()\n"
        "def is_bf16_supported(): return True\n"
    )
    return f'PYTHONPATH="{fake}" exec python3 "$@"\n'


class TestRocmEntrypoint:
    def test_no_kfd_refuses_and_names_docker_desktop(self, tmp_path):
        rc, ran, err = _entrypoint(tmp_path, kfd = False)
        assert rc == 1 and not ran
        assert "/dev/kfd not found" in err and "Docker Desktop" in err, err
        # the old advice: a WSL or Docker Desktop host cannot modprobe anything
        assert "modprobe" not in err, err
        assert "run.sh --rocm" in err

    @pytest.mark.skipif(os.geteuid() == 0, reason = "root reads a mode-0 file")
    def test_an_unreadable_kfd_names_the_group_ids(self, tmp_path):
        rc, ran, err = _entrypoint(tmp_path, readable = False)
        assert rc == 1 and not ran
        assert "not readable" in err and "--group-add" in err and "NUMERIC" in err, err

    def test_no_gpu_for_rocm_smi_is_a_note_not_a_refusal(self, tmp_path):
        """Measured on a gfx1151 runner: rocm-smi lists nothing inside the container
        while HIP torch drives the card. torch is the gate; rocm-smi only advises."""
        rc, ran, err = _entrypoint(tmp_path, smi_sees_gpu = False)
        assert rc == 0 and ran, err
        assert "rocm-smi lists no GPU" in err and "/dev/dri" in err, err

    def test_the_skip_flag_runs_the_command_without_probing(self, tmp_path):
        rc, ran, err = _entrypoint(tmp_path, kfd = False, env_extra = {"UNSLOTH_SKIP_GPU_CHECK": "1"})
        assert rc == 0 and ran, err

    def test_a_happy_host_runs_the_command(self, tmp_path):
        rc, ran, err = _entrypoint(tmp_path)
        assert rc == 0 and ran, err

    def test_dxg_is_accepted_when_kfd_is_absent(self, tmp_path):
        """WSL2 never has /dev/kfd. /dev/dxg plus librocdxg is the same GPU evidence
        install.sh gates a WSL host on, so the run must proceed, not refuse."""
        lib = tmp_path / "rocmlib"
        lib.mkdir()
        (lib / "librocdxg.so.1").write_text("")
        rc, ran, err = _entrypoint(
            tmp_path,
            kfd = False,
            dxg = True,
            env_extra = {"UNSLOTH_ROCM_DXG_LIBDIRS": str(lib)},
        )
        assert rc == 0 and ran, err
        assert "DXG bridge" in err, err

    def test_dxg_skips_the_rocm_smi_advice(self, tmp_path):
        """rocm-smi reads the amdgpu sysfs, which the bridge has none of (measured in the
        container: "Driver not initialized"), and its advice is /dev/dri and group ids,
        neither of which exists on WSL."""
        lib = tmp_path / "rocmlib"
        lib.mkdir()
        (lib / "librocdxg.so.1").write_text("")
        rc, ran, err = _entrypoint(
            tmp_path,
            kfd = False,
            dxg = True,
            smi_sees_gpu = False,
            env_extra = {"UNSLOTH_ROCM_DXG_LIBDIRS": str(lib)},
        )
        assert rc == 0 and ran, err
        assert "--group-add" not in err and "/dev/dri" not in err, err

    def test_dxg_without_the_bridge_library_refuses(self, tmp_path):
        """/dev/dxg alone cannot reach the card: the HSA runtime needs librocdxg."""
        rc, ran, err = _entrypoint(
            tmp_path,
            kfd = False,
            dxg = True,
            env_extra = {"UNSLOTH_ROCM_DXG_LIBDIRS": str(tmp_path / "empty")},
        )
        assert rc == 1 and not ran
        assert "librocdxg" in err, err
        # librocdxg dlopens libdxcore from WSL's lib dir: the hand-run recovery must
        # mount it and put it on the search path, or hsa_init fails after this check.
        assert "-v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro" in err, err
        assert "LD_LIBRARY_PATH=/usr/lib/wsl/lib" in err, err
        # run.sh defaults to the published image, which the bridge refuses: the
        # recovery has to name the per-arch build, or it sends the user there
        assert "UNSLOTH_IMAGE=unsloth-rocm:latest bash docker/run.sh --rocm" in err, err
        assert "unsloth/unsloth-rocm:latest" not in err, err

    def _dxg_with_torch(self, tmp_path, libnames, available = True):
        lib = tmp_path / "dxglib"
        lib.mkdir()
        (lib / "librocdxg.so.1").write_text("")
        return _entrypoint(
            tmp_path,
            kfd = False,
            dxg = True,
            python_body = _fake_rocm_torch(tmp_path, libnames, available),
            env_extra = {"UNSLOTH_ROCM_DXG_LIBDIRS": str(lib)},
        )

    def test_dxg_torch_failure_gives_bridge_advice_not_amdgpu_advice(self, tmp_path):
        """WSL has no host amdgpu stack, so the rocm-smi / dkms / rebuild-against-the-host
        advice is wrong there; the bridge has its own two causes (measured: the libdxcore
        mount, and the Windows driver)."""
        rc, ran, err = self._dxg_with_torch(
            tmp_path,
            ["librocprofiler-register.so"],
            available = False,
        )
        assert rc == 1 and not ran, err
        assert "libdxcore" in err and "Windows AMD driver" in err, err
        assert "dkms" not in err and "amdgpu driver has to be" not in err, err

    def test_dxg_refuses_a_torch_bundling_librocprofiler_sdk(self, tmp_path):
        """That library enumerates GPUs from a KFD topology WSL does not have, and aborts."""
        rc, ran, err = self._dxg_with_torch(
            tmp_path,
            ["librocprofiler-register.so", "librocprofiler-sdk.so"],
        )
        assert rc == 1 and not ran, err
        assert "librocprofiler-sdk.so" in err, err
        assert "UNSLOTH_IMAGE=unsloth-rocm:latest bash docker/run.sh --rocm" in err, err

    def test_dxg_accepts_a_torch_carrying_only_librocprofiler_register(self, tmp_path):
        """torch 2.11+rocm7.2 ships -register.so and runs on the bridge (measured on an
        R9700), so matching every "rocprof" name refused a build that works."""
        rc, ran, err = self._dxg_with_torch(tmp_path, ["librocprofiler-register.so"])
        assert rc == 0 and ran, err

    def test_a_failing_torch_check_stops_before_the_command(self, tmp_path):
        rc, ran, _ = _entrypoint(tmp_path, python_body = "cat > /dev/null\nexit 1\n")
        assert rc == 1 and not ran

    def test_the_torch_check_asserts_a_hip_build_first(self):
        """A CUDA or CPU torch must be named as the image's fault, not the host's."""
        body = open(_ENTRYPOINT, encoding = "utf-8").read()
        check3 = body[body.index("Check 3") : body.index("Check 4")]
        assert "hip_ver is None" in check3 and "not a ROCm build" in check3
        assert "6.2" not in check3, "the ROCm version is read from the build, not hardcoded"

    def test_gfx1033_is_refused_not_spoofed(self, tmp_path):
        """Van Gogh (Steam Deck) computes wrong results under ROCm (studio/ROCM_RDNA2_APU.md);
        install.sh routes it to CPU torch. A HIP-only image can only refuse, and must not
        advise HSA_OVERRIDE_GFX_VERSION, which would hide the silicon from this check."""
        fake = tmp_path / "fake"
        (fake / "torch" / "cuda").mkdir(parents = True)
        (fake / "torch" / "__init__.py").write_text(
            "__version__ = '2.12.1+rocm7.2'\n"
            "class version:\n    hip = '7.2.53211'\n"
            "from . import cuda\n"
        )
        (fake / "torch" / "cuda" / "__init__.py").write_text(
            "class _P:\n    gcnArchName = 'gfx1033:xnack-'\n"
            "def is_available(): return True\n"
            "def device_count(): return 1\n"
            "def get_device_name(i): return 'AMD Custom GPU 0405'\n"
            "def get_device_properties(i): return _P()\n"
            "def is_bf16_supported(): return False\n"
        )
        python_body = f'PYTHONPATH="{fake}" exec python3 "$@"\n'
        rc, ran, err = _entrypoint(tmp_path, python_body = python_body)
        assert rc == 1 and not ran, err
        assert "gfx1033" in err and "refuses" in err, err
        assert "HSA_OVERRIDE_GFX_VERSION=10.3.0" not in err, err
        # the same fake torch on a supported arch runs the command
        (fake / "torch" / "cuda" / "__init__.py").write_text(
            (fake / "torch" / "cuda" / "__init__.py")
            .read_text()
            .replace("gfx1033:xnack-", "gfx1100:sramecc+")
        )
        (tmp_path / "ok").mkdir()
        rc, ran, err = _entrypoint(tmp_path / "ok", python_body = python_body)
        assert rc == 0 and ran, err
        assert "RDNA 3" in err, err

    def _fake_torch(self, tmp_path, arch):
        fake = tmp_path / "fake"
        (fake / "torch" / "cuda").mkdir(parents = True)
        (fake / "torch" / "__init__.py").write_text(
            "__version__ = '2.12.1+rocm7.2'\nclass version:\n    hip = '7.2.53211'\nfrom . import cuda\n"
        )
        (fake / "torch" / "cuda" / "__init__.py").write_text(
            f"class _P:\n    gcnArchName = '{arch}'\n"
            "def is_available(): return True\ndef device_count(): return 1\n"
            "def get_device_name(i): return 'AMD GPU'\ndef get_device_properties(i): return _P()\n"
            "def is_bf16_supported(): return False\n"
        )
        return f'PYTHONPATH="{fake}" exec python3 "$@"\n'

    def test_a_spoofed_gfx1033_is_caught_from_the_kernels_topology(self, tmp_path):
        """HSA_OVERRIDE_GFX_VERSION=10.3.0 makes ROCr report gfx1030 for a Steam Deck,
        and run.sh forwards that variable, so gcnArchName alone would wave it through.
        amdkfd's gfx_target_version (100303) is immune to the userland spoof."""
        topo = tmp_path / "topo" / "1"
        topo.mkdir(parents = True)
        (topo / "properties").write_text("vendor_id 4098\ngfx_target_version 100303\n")
        (tmp_path / "topo" / "0").mkdir()
        (tmp_path / "topo" / "0" / "properties").write_text("vendor_id 0\ngfx_target_version 0\n")
        body = self._fake_torch(tmp_path, "gfx1030")
        rc, ran, err = _entrypoint(
            tmp_path,
            python_body = body,
            env_extra = {
                "UNSLOTH_KFD_TOPOLOGY": str(tmp_path / "topo"),
                "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
            },
        )
        assert rc == 1 and not ran, err
        assert "kernel reports a gfx1033" in err and "10.3.0" in err, err
        # the same topology without the spoof: torch already names gfx1033 and the plain refusal fires
        (tmp_path / "b").mkdir()
        rc, ran, err = _entrypoint(
            tmp_path / "b",
            python_body = self._fake_torch(tmp_path / "b", "gfx1033"),
            env_extra = {"UNSLOTH_KFD_TOPOLOGY": str(tmp_path / "topo")},
        )
        assert rc == 1 and not ran and "refuses" in err, err
        # a real gfx1030 with the override set is not refused
        (tmp_path / "c").mkdir()
        (tmp_path / "topo" / "1" / "properties").write_text(
            "vendor_id 4098\ngfx_target_version 100300\n"
        )
        rc, ran, err = _entrypoint(
            tmp_path / "c",
            python_body = self._fake_torch(tmp_path / "c", "gfx1030"),
            env_extra = {
                "UNSLOTH_KFD_TOPOLOGY": str(tmp_path / "topo"),
                "HSA_OVERRIDE_GFX_VERSION": "10.3.0",
            },
        )
        assert rc == 0 and ran, err
        assert "KFD reports: gfx1030" in err, err

    def test_a_per_arch_image_on_a_generic_card_points_at_the_generic_image(self, tmp_path):
        """Dockerfile.rocm refuses ROCM_GFX outside the per-arch families, so the advice
        must not be a rebuild command that fails on the spot."""
        body = self._fake_torch(tmp_path, "gfx1100:sramecc+")
        rc, ran, err = _entrypoint(tmp_path, python_body = body, build_info_gfx = "gfx1151")
        assert rc == 0 and ran, err
        assert "no per-arch index" in err and "ROCM_GFX=gfx1100" not in err, err
        (tmp_path / "d").mkdir()
        rc, ran, err = _entrypoint(
            tmp_path / "d",
            python_body = self._fake_torch(tmp_path / "d", "gfx1201"),
            build_info_gfx = "gfx1151",
        )
        assert rc == 0 and "ROCM_GFX=gfx1201 bash docker/build.sh --rocm" in err, err

    def test_a_per_arch_image_drops_a_stale_gfx_override_and_a_generic_one_keeps_it(self, tmp_path):
        """HSA_OVERRIDE_GFX_VERSION=11.0.0 is the generic-wheel workaround on Strix; a
        gfx1151 image has native kernels the override would hide (install.sh clears it)."""
        body = self._fake_torch(tmp_path, "gfx1151")
        rc, ran, err = _entrypoint(
            tmp_path,
            python_body = body,
            build_info_gfx = "gfx1151",
            env_extra = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
            command = "echo ${HSA_OVERRIDE_GFX_VERSION:-unset}",
        )
        assert rc == 0 and ran, err
        assert (tmp_path / "ran").read_text().strip() == "unset"
        assert "ignoring HSA_OVERRIDE_GFX_VERSION=11.0.0" in err, err
        (tmp_path / "g").mkdir()
        rc, ran, err = _entrypoint(
            tmp_path / "g",
            python_body = self._fake_torch(tmp_path / "g", "gfx1100"),
            env_extra = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
            command = "echo ${HSA_OVERRIDE_GFX_VERSION:-unset}",
        )
        assert rc == 0 and (tmp_path / "g" / "ran").read_text().strip() == "11.0.0", err

    def test_the_skip_flag_still_drops_a_stale_override_on_a_per_arch_image(self, tmp_path):
        """UNSLOTH_SKIP_GPU_CHECK=1 skips the diagnostics, not the override cleanup."""
        rc, ran, err = _entrypoint(
            tmp_path,
            kfd = False,
            build_info_gfx = "gfx1151",
            env_extra = {"UNSLOTH_SKIP_GPU_CHECK": "1", "HSA_OVERRIDE_GFX_VERSION": "11.0.0"},
            command = "echo ${HSA_OVERRIDE_GFX_VERSION:-unset}",
        )
        assert rc == 0 and ran, err
        assert (tmp_path / "ran").read_text().strip() == "unset"
        assert "ignoring HSA_OVERRIDE_GFX_VERSION" in err and "/dev/kfd" not in err, err

    @staticmethod
    def _build_args(tmp_path, **inputs):
        """Run the prepare job's build_args step as the workflow would."""
        import yaml

        wf = yaml.safe_load(open(_WORKFLOW, encoding = "utf-8"))
        step = next(s for s in wf["jobs"]["prepare"]["steps"] if s.get("id") == "build_args")
        out = tmp_path / f"out{len(os.listdir(tmp_path))}"
        out.write_text("")
        env = {"PATH": os.environ["PATH"], "GITHUB_OUTPUT": str(out)}
        env.update({k: str(v) for k, v in wf["env"].items()})
        env.update({"IN_UNSLOTH": "", "IN_ZOO": "", "IN_ROCM": "", "IN_INDEX": "", "IN_GFX": ""})
        env.update(inputs)
        proc = subprocess.run(
            ["bash", "-e", "-c", step["run"]], env = env, capture_output = True, text = True
        )
        got = dict(ln.split("=", 1) for ln in out.read_text().splitlines() if "=" in ln)
        return proc.returncode, got, proc.stdout + proc.stderr

    def test_gfx906_defaults_to_the_last_rocm_that_carries_it(self, tmp_path):
        """The public :gfx906 tag must name a 6.3 build: a gfx906 dispatch on the
        7.2.4 default would be tagged and fail on the first matmul."""
        rc, got, log = self._build_args(tmp_path, IN_GFX = "gfx906")
        assert rc == 0, log
        assert got["rocm_version"] == "6.3.4" and got["torch_index_url"].endswith("/rocm6.3"), got
        assert got["gfx_tag"] == "true" and got["stable"] == "false", got
        rc, got, log = self._build_args(tmp_path, IN_GFX = "gfx906", IN_ROCM = "6.3.4")
        assert rc == 0 and got["gfx_tag"] == "true", (got, log)
        rc, got, log = self._build_args(tmp_path, IN_GFX = "gfx906", IN_ROCM = "7.2.4")
        assert rc != 0 and "needs a ROCm 6.3 base" in log, log
        rc, got, log = self._build_args(
            tmp_path, IN_GFX = "gfx906", IN_ROCM = "6.3.4", IN_UNSLOTH = "feature"
        )
        assert rc == 0 and got["gfx_tag"] == "false", got
        rc, got, log = self._build_args(tmp_path, IN_GFX = "gfx1151")
        assert rc == 0 and got["rocm_version"] == "7.2.4" and got["gfx_tag"] == "true", got
        rc, got, log = self._build_args(tmp_path)
        assert rc == 0 and got["stable"] == "true" and got["gfx_tag"] == "false", got

    def test_the_workflow_accepts_a_gfx906_dispatch(self):
        """Dockerfile.rocm relies on ROCM_GFX=gfx906 to leave out bitsandbytes, so the
        dispatch validation must let it through (with a 6.3 base)."""
        body = open(_WORKFLOW, encoding = "utf-8").read()
        assert '""|gfx906|gfx1150|gfx1151|gfx1152|gfx1200|gfx1201) ;;' in body
        docker = open(_DOCKERFILE, encoding = "utf-8").read()
        assert (
            "--build-arg UNSLOTH_REF=<sha>" in docker
        ), "the bare docker build line must not suggest mutable refs"

    def test_a_gfx906_build_ships_without_bitsandbytes(self):
        """No prebuilt bitsandbytes wheel has gfx906 kernels; install.sh skips it for
        gfx906 and the image must too, or its own 4-bit smoke test cannot run."""
        docker = open(_DOCKERFILE, encoding = "utf-8").read()
        assert "gfx906) ;;" in docker
        assert "grep -q '^ROCM_GFX=gfx906$' /etc/unsloth-rocm-build" in docker
        assert "pip uninstall -y bitsandbytes" in docker
        assert 'WANT_BNB = BUILD_GFX != "gfx906"' in docker
        smoke = open(_SMOKE, encoding = "utf-8").read()
        assert "load_in_4bit = four_bit" in smoke and "ROCM_GFX=gfx906" in smoke
        entry = open(_ENTRYPOINT, encoding = "utf-8").read()
        assert "ROCM_GFX=gfx906 ROCM_VERSION=6.3.4" in entry

    def test_the_gfx_tag_needs_every_other_input_at_its_default(self):
        """A feature-branch ref plus rocm_gfx=gfx1151 must not replace the public
        gfx1151 image: the gfx tag is gated like latest, minus the gfx itself."""
        body = open(_WORKFLOW, encoding = "utf-8").read()
        assert "gfx_tag=${GFX_TAG}" in body
        assert (
            'GFX_TAG=false\n          [ "$DEFAULTS" = "true" ] && [ -n "$GFX" ] && GFX_TAG=true'
            in body
        )
        raw = [
            ln
            for ln in body.splitlines()
            if "type=raw,value=${{ needs.prepare.outputs.rocm_gfx }}" in ln
        ]
        assert len(raw) == 1 and "needs.prepare.outputs.gfx_tag == 'true'" in raw[0], raw

    def test_the_arch_table_carries_no_marketing_names(self):
        """Card-name tables live in install.sh and studio/ under a parity test; a
        seventh copy here would drift. Families only."""
        body = open(_ENTRYPOINT, encoding = "utf-8").read()
        import re

        assert not re.search(r"RX\s*\d{4}", body), "marketing names in the entrypoint's arch table"
        assert "gfx906" in body and "6.3" in body, "gfx906 needs the version-aware note"
