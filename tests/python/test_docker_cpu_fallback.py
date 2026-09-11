# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The :latest cutover: unsloth/unsloth:latest is the Studio image, so a plain
`docker run unsloth/unsloth` has to survive a host with no NVIDIA GPU.

Two independent failure points, one per class below:
  * the DAEMON rejects `--gpus` before the container exists (exit 125), so
    entrypoint.sh never runs -- docker/run.sh has to stop asking for it;
  * entrypoint.sh itself exits 1 without UNSLOTH_ALLOW_CPU, so the Studio image
    has to default it on.
"""

import os
import shutil
import stat
import subprocess

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_DOCKER = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "docker")

_RUN_SH = os.path.join(_DOCKER, "run.sh")
_STUDIO_DF = os.path.join(_DOCKER, "Dockerfile.studio")
_BASE_DF = os.path.join(_DOCKER, "Dockerfile")
_ENTRYPOINT = os.path.join(_DOCKER, "entrypoint.sh")


def _stub(path, body):
    with open(path, "w") as f:
        f.write("#!/usr/bin/env bash\n" + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _invoke_run_sh(
    tmp_path,
    *,
    nvidia,
    amd,
    groups = "both",
):
    """Run docker/run.sh with a recording `docker` stub and a staged /dev tree.

    Returns the argv docker/run.sh would have handed to `docker run`.
    """
    bindir = tmp_path / "bin"
    bindir.mkdir()
    argv_log = tmp_path / "argv"

    # `docker run` must not exec anything real; record argv and stop.
    _stub(
        str(bindir / "docker"),
        'if [ "$1" = "info" ]; then echo " Runtimes: io.containerd.runc.v2 runc"; exit 0; fi\n'
        'printf "%s\\n" "$@" > ' + str(argv_log) + "\nexit 0\n",
    )
    # nvidia-smi is ALWAYS shadowed. /usr/bin has to stay on PATH for cut/grep/getent,
    # and a CI or dev host with a real GPU there would otherwise make the
    # "no NVIDIA" case unreachable and pass this test vacuously.
    if nvidia:
        _stub(str(bindir / "nvidia-smi"), 'echo "GPU 0: NVIDIA H100 (UUID: GPU-abc)"\n')
    else:
        # driver present but zero GPUs: the harder of the two no-GPU shapes, and it
        # covers the missing-binary shape too (same && chain, same outcome)
        _stub(str(bindir / "nvidia-smi"), "exit 1\n")

    # getent is ALWAYS shadowed too, for the same reason as nvidia-smi: this host has
    # both video and render, so relying on the real one made the missing-group cases
    # unreachable and the AMD test green for the wrong reason.
    known = {"both": ("44", "992"), "video_only": ("44", None), "none": (None, None)}
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
    if amd:
        (dev_root / "dev" / "kfd").write_text("")
        (dev_root / "dev" / "dri").mkdir()

    env = dict(os.environ)
    # PATH is replaced, not prepended: a real nvidia-smi on this host would
    # otherwise make the "no NVIDIA" case unreachable.
    env["PATH"] = str(bindir) + ":/usr/bin:/bin"
    env["UNSLOTH_DEV_ROOT"] = str(dev_root)
    env["HOME"] = str(tmp_path / "home")
    env["UNSLOTH_WORKDIR"] = str(tmp_path)
    for leak in ("HF_TOKEN", "WANDB_API_KEY", "UNSLOTH_GPUS", "UNSLOTH_ALLOW_CPU"):
        env.pop(leak, None)

    # absolute: the "absent" case strips /usr/bin from PATH, so `bash` itself would
    # not resolve either
    proc = subprocess.run(
        [shutil.which("bash") or "/bin/bash", _RUN_SH, "true"],
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
    )
    assert proc.returncode == 0, f"run.sh failed: {proc.stderr}"
    return argv_log.read_text().splitlines(), proc.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash required")
class TestRunShDegradesWithoutNvidia:
    def test_gpus_flag_is_dropped_when_the_host_has_no_nvidia_gpu(self, tmp_path):
        """`--gpus all` on an NVIDIA-less host is exit 125 AT THE DAEMON, so the
        container never starts and entrypoint.sh never gets to explain itself."""
        argv, stderr = _invoke_run_sh(tmp_path, nvidia = False, amd = False)
        assert "--gpus" not in argv, f"run.sh still passed --gpus: {argv}"
        assert "no NVIDIA GPU on this host" in stderr

    def test_amd_host_gets_the_render_nodes_with_numeric_gids(self, tmp_path):
        """--group-add by NAME resolves inside the container, where the host's
        video/render groups do not exist, so the gids must be numeric."""
        argv, _ = _invoke_run_sh(tmp_path, nvidia = False, amd = True)
        assert "--gpus" not in argv
        assert "--device" in argv
        assert "/dev/kfd" in argv and "/dev/dri" in argv
        gids = [argv[i + 1] for i, a in enumerate(argv) if a == "--group-add"]
        assert all(g.isdigit() for g in gids), f"non-numeric --group-add: {gids}"

    def test_the_group_lookup_is_guarded_on_getent_existing(self):
        """A host with no getent at all (busybox, some slim images) must skip the
        lookup rather than fail it."""
        body = open(_RUN_SH, encoding = "utf-8").read()
        idx = body.index("getent group")
        assert "command -v getent" in body[:idx]

    @pytest.mark.parametrize("groups", ["none", "video_only"])
    def test_a_missing_group_record_does_not_abort_the_run(self, tmp_path, groups):
        """getent exits nonzero for a name that is not in NSS. Under `set -o pipefail`
        that propagates out of the command substitution and `set -e` kills run.sh
        before docker run, so the AMD fallback could never start on a host without a
        render group. Degrade to whatever gids exist instead."""
        argv, _ = _invoke_run_sh(tmp_path, nvidia = False, amd = True, groups = groups)
        # the devices are the point; the gids are best-effort
        assert "/dev/kfd" in argv and "/dev/dri" in argv
        assert "--gpus" not in argv
        gids = [argv[i + 1] for i, a in enumerate(argv) if a == "--group-add"]
        assert all(g.isdigit() for g in gids), f"non-numeric --group-add: {gids}"
        expected = {"none": 0, "video_only": 1}[groups]
        assert len(gids) == expected, f"expected {expected} gids, got {gids}"

    def test_an_nvidia_host_is_untouched(self, tmp_path):
        """The degrade path must not fire where --gpus actually works."""
        argv, _ = _invoke_run_sh(tmp_path, nvidia = True, amd = False)
        assert "--gpus" in argv
        assert "all" in argv
        assert "/dev/kfd" not in argv


class TestStudioImageAllowsCpu:
    def test_studio_image_opts_in_through_its_own_variable(self):
        """:latest is the Studio image. Without an opt-in every CPU-only, AMD and
        Docker-Desktop user goes from working Studio to an exit 1."""
        body = open(_STUDIO_DF, encoding = "utf-8").read()
        env_lines = [
            ln.strip()
            for ln in body.splitlines()
            if "UNSLOTH_IMAGE_ALLOW_CPU=1" in ln and not ln.strip().startswith("#")
        ]
        assert env_lines, "Dockerfile.studio does not default UNSLOTH_IMAGE_ALLOW_CPU=1"

    def test_studio_image_env_never_carries_allow_cpu(self):
        """An image ENV reaches every process, so UNSLOTH_ALLOW_CPU=1 there broke
        training everywhere on a GPU host. Only install.sh may see it, inline."""
        body = open(_STUDIO_DF, encoding = "utf-8").read()
        env_block = body[body.index("ENV UNSLOTH_STUDIO_HOME") :]
        env_block = env_block[: env_block.index("\n\n")]
        assert "UNSLOTH_ALLOW_CPU" not in env_block.replace("UNSLOTH_IMAGE_ALLOW_CPU", "")
        inline = [
            ln.strip()
            for ln in body.splitlines()
            if "UNSLOTH_ALLOW_CPU=1" in ln and not ln.strip().startswith("#")
        ]
        assert inline == ["UNSLOTH_ALLOW_CPU=1 \\"], inline

    def test_the_base_training_image_keeps_the_strict_check(self):
        """FastLanguageModel genuinely needs a GPU, so :core must NOT default it."""
        body = open(_BASE_DF, encoding = "utf-8").read()
        offenders = [
            ln.strip()
            for ln in body.splitlines()
            if "ALLOW_CPU=1" in ln and not ln.strip().startswith("#")
        ]
        assert not offenders, f"base image weakened the GPU check: {offenders}"


def _run_entrypoint(tmp_path, *, gpu, env_extra):
    """Run docker/entrypoint.sh with stubbed nvidia-smi and python, exec'ing a command
    that dumps its environment. Returns (returncode, child env dict, stderr)."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    if gpu:
        _stub(
            str(bindir / "nvidia-smi"),
            'case "$*" in\n'
            '  *compute_cap*) echo "8.9" ;;\n'
            '  *driver_version*) echo "610.43.02" ;;\n'
            '  *) echo "GPU 0: NVIDIA RTX 6000 Ada Generation (UUID: GPU-abc)" ;;\n'
            "esac\n",
        )
    else:
        _stub(str(bindir / "nvidia-smi"), "exit 1\n")
    # the GPU path runs two torch heredocs; accept them without torch
    _stub(str(bindir / "python"), "cat > /dev/null\nexit 0\n")
    dump = tmp_path / "child_env"
    env = {
        "PATH": str(bindir) + ":/usr/bin:/bin",
        "HOME": str(tmp_path),
        "UNSLOTH_STUDIO_HOME": str(tmp_path / "studio"),
    }
    env.update(env_extra)
    proc = subprocess.run(
        [shutil.which("bash") or "/bin/bash", _ENTRYPOINT, "bash", "-c", f"env > {dump}"],
        env = env,
        capture_output = True,
        text = True,
        timeout = 60,
    )
    child = {}
    if dump.exists():
        for line in dump.read_text().splitlines():
            key, _, value = line.partition("=")
            child[key] = value
    return proc.returncode, child, proc.stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash required")
class TestEntrypointAllowCpu:
    def test_studio_image_on_a_cpu_host_starts_and_exports_allow_cpu(self, tmp_path):
        """Studio's own processes still need UNSLOTH_ALLOW_CPU=1 to import unsloth
        without a GPU, so the entrypoint exports it for its children."""
        rc, child, stderr = _run_entrypoint(
            tmp_path, gpu = False, env_extra = {"UNSLOTH_IMAGE_ALLOW_CPU": "1"}
        )
        assert rc == 0, stderr
        assert child.get("UNSLOTH_ALLOW_CPU") == "1"
        assert "continuing on CPU" in stderr

    def test_studio_image_on_a_gpu_host_hides_allow_cpu(self, tmp_path):
        """The regression: with a GPU visible the children must not see the variable,
        or Studio training and every notebook run on stock TRL."""
        rc, child, stderr = _run_entrypoint(
            tmp_path, gpu = True, env_extra = {"UNSLOTH_IMAGE_ALLOW_CPU": "1"}
        )
        assert rc == 0, stderr
        assert "UNSLOTH_ALLOW_CPU" not in child
        assert "Ignoring UNSLOTH_ALLOW_CPU" not in stderr

    def test_an_explicit_allow_cpu_on_a_gpu_host_is_dropped_with_a_warning(self, tmp_path):
        """docker/run.sh forwards a host-shell UNSLOTH_ALLOW_CPU, and the docs tell
        CPU users to pass it, so a GPU host can receive it explicitly too."""
        rc, child, stderr = _run_entrypoint(
            tmp_path, gpu = True, env_extra = {"UNSLOTH_ALLOW_CPU": "1"}
        )
        assert rc == 0, stderr
        assert "UNSLOTH_ALLOW_CPU" not in child
        assert "Ignoring UNSLOTH_ALLOW_CPU=1" in stderr

    def test_an_explicit_zero_restores_the_strict_check(self, tmp_path):
        rc, _, stderr = _run_entrypoint(
            tmp_path,
            gpu = False,
            env_extra = {"UNSLOTH_IMAGE_ALLOW_CPU": "1", "UNSLOTH_ALLOW_CPU": "0"},
        )
        assert rc == 1
        assert "No GPU visible" in stderr

    def test_core_without_an_opt_in_still_refuses(self, tmp_path):
        rc, _, stderr = _run_entrypoint(tmp_path, gpu = False, env_extra = {})
        assert rc == 1
        assert "No GPU visible" in stderr

    def test_core_with_an_explicit_opt_in_starts_on_cpu(self, tmp_path):
        rc, child, stderr = _run_entrypoint(
            tmp_path, gpu = False, env_extra = {"UNSLOTH_ALLOW_CPU": "1"}
        )
        assert rc == 0, stderr
        assert child.get("UNSLOTH_ALLOW_CPU") == "1"

    @pytest.mark.parametrize("gpu", [True, False])
    def test_skipping_the_gpu_check_applies_the_same_rule(self, tmp_path, gpu):
        """UNSLOTH_SKIP_GPU_CHECK=1 skips the torch checks, not the variable's
        meaning: a GPU host must still lose it and a CPU host must still get it."""
        rc, child, stderr = _run_entrypoint(
            tmp_path,
            gpu = gpu,
            env_extra = {"UNSLOTH_IMAGE_ALLOW_CPU": "1", "UNSLOTH_SKIP_GPU_CHECK": "1"},
        )
        assert rc == 0, stderr
        if gpu:
            assert "UNSLOTH_ALLOW_CPU" not in child
        else:
            assert child.get("UNSLOTH_ALLOW_CPU") == "1"
