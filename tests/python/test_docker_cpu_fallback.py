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
    def test_studio_image_defaults_allow_cpu_on(self):
        """:latest is the Studio image. Without this default every CPU-only, AMD
        and Docker-Desktop user goes from working Studio to an exit 1."""
        body = open(_STUDIO_DF, encoding = "utf-8").read()
        env_lines = [
            ln.strip()
            for ln in body.splitlines()
            if "UNSLOTH_ALLOW_CPU=1" in ln and not ln.strip().startswith("#")
        ]
        assert env_lines, "Dockerfile.studio does not default UNSLOTH_ALLOW_CPU=1"

    def test_the_base_training_image_keeps_the_strict_check(self):
        """FastLanguageModel genuinely needs a GPU, so :core must NOT default it."""
        body = open(_BASE_DF, encoding = "utf-8").read()
        offenders = [
            ln.strip()
            for ln in body.splitlines()
            if "UNSLOTH_ALLOW_CPU=1" in ln and not ln.strip().startswith("#")
        ]
        assert not offenders, f"base image weakened the GPU check: {offenders}"

    def test_entrypoint_reads_allow_cpu_from_the_environment(self):
        """An image-level ENV and a `-e` flag reach the process identically, so the
        entrypoint must read it from the environment with no `-e`-only handling."""
        body = open(_ENTRYPOINT, encoding = "utf-8").read()
        assert '"${UNSLOTH_ALLOW_CPU:-0}" == "1"' in body

    def test_allow_cpu_only_applies_when_no_gpu_is_visible(self):
        """The default must not weaken a GPU host: the CPU branch has to be gated
        on nvidia-smi finding nothing, otherwise it would skip the torch checks."""
        body = open(_ENTRYPOINT, encoding = "utf-8").read()
        idx = body.index('"${UNSLOTH_ALLOW_CPU:-0}" == "1"')
        branch = body[idx : idx + 400]
        assert "nvidia-smi" in branch and "grep -q '^GPU'" in branch
