# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #8473 -- the installer says the GPU is fine and the backend runs CPU-only.

Reporter: AMD host, `unsloth studio update` prints `gpu AMD ROCm (gfx1201)` then
`python dependencies up to date`, and Studio shows VRAM `--` and "No visible GPU".
The installer's GPU line comes from rocminfo / amd-smi / hipinfo; the backend's
verdict is torch.cuda.is_available() in its own process. Nothing reconciled the two.

No CI runner has AMD hardware, so the real block is extracted from setup.sh and run
under bash against a FAKE venv interpreter whose answer, exit code and latency the
test controls, plus a fake `timeout` that records the bound setup.sh asked for while
enforcing a short one.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"

_BLOCK_START = "# ── Does PyTorch see the GPU this installer just announced? ──"
_BLOCK_END = "# ── 7. Prefer prebuilt llama.cpp bundles"

# Colour is the assertion surface for severity: a report demoted from $C_ERR to $C_WARN fails here.
# setup.sh runs under `set -euo pipefail` (line 5), so the harness does too. Without it the
# block is exercised more forgivingly than it ships: an unguarded command substitution whose
# binary is missing exits 127 and takes the installer down, and every test here passed while
# that was true. Reverting the `cut` -> `IFS` read fix now fails 4 checks instead of 2.
_HARNESS_HEAD = """
set -euo pipefail
C_DIM="DIM"; C_RST=""; C_OK="OK"; C_WARN="WARN"; C_ERR="ERR"
step()    { printf 'STEP|%s|%s|%s\\n' "$1" "$2" "${3:-OK}"; }
substep() { printf 'SUB|%s|%s\\n' "$1" "${2:-DIM}"; }
verbose_substep() { printf 'VSUB|%s\\n' "$1"; }
"""

_HARNESS_TAIL = '\necho "BLOCK_DONE"\n'


def _block() -> str:
    text = SETUP_SH.read_text(encoding = "utf-8")
    start = text.index(_BLOCK_START)
    end = text.index(_BLOCK_END, start)
    return text[start:end]


@pytest.fixture(scope = "module")
def block() -> str:
    extracted = _block()
    # An empty extraction would make every check below pass vacuously.
    assert "torch.cuda.is_available()" in extracted
    assert "_setup_amd_detected" in extracted
    return extracted


def _write_exec(path: Path, body: str) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(body, encoding = "utf-8")
    path.chmod(0o755)


def _make_venv(
    tmp_path: Path,
    *,
    stdout: str = "",
    exit_code: int = 0,
    sleep_seconds: float = 0.0,
    torch_on_disk: bool = True,
    torch_local_label: str = "+rocm6.4",
    torch_hip: str = "",
) -> Path:
    """A venv whose `python` prints exactly what the test wants, when it wants."""
    venv = tmp_path / "venv"
    calls = venv / "calls.log"
    _write_exec(
        venv / "bin" / "python",
        "#!/bin/sh\n"
        f'echo "call" >> "{calls}"\n'
        + (f"sleep {sleep_seconds}\n" if sleep_seconds else "")
        + (f"printf '%s' \"$(cat <<'PROBE_EOF'\n{stdout}\nPROBE_EOF\n)\"\n" if stdout else "")
        + f"exit {exit_code}\n",
    )
    if torch_on_disk:
        (venv / "lib" / "python3.11" / "site-packages" / "torch").mkdir(parents = True)
        # The `hip` line is real: torch writes it on every build, quoted on ROCm and None
        # elsewhere. Derived from the label so a fixture cannot claim a shape torch never ships.
        _hip = torch_hip or ("6.4.43483" if "+rocm" in torch_local_label else "")
        _hip = f"'{_hip}'" if _hip else "None"
        (venv / "lib" / "python3.11" / "site-packages" / "torch" / "version.py").write_text(
            f"__version__ = '2.9.0{torch_local_label}'\nhip: Optional[str] = {_hip}\n",
            encoding = "utf-8",
        )
    return venv


def _run_block(
    block_text: str,
    venv: Path,
    tmp_path: Path,
    *,
    amd: bool = False,
    nvidia: bool = False,
    gfx: str = "",
    marketing: str = "",
    env: dict[str, str] | None = None,
    with_timeout: bool = True,
    timeout_bound: int = 5,
    colab: bool = False,
    venv_dir: Path | None = None,
    path_python: str | None = None,
    sabotage: tuple[str, ...] = (),
    uname_machine: str | None = None,
) -> dict:
    """Run the real setup.sh block with stubbed printers and a stubbed `timeout`."""
    stub_bin = tmp_path / "stubbin"
    stub_bin.mkdir(exist_ok = True)
    timeout_log = tmp_path / "timeout_args.log"
    real_timeout = shutil.which("timeout")
    if with_timeout:
        # macOS ships no timeout(1), so the shim below stands in with the same contract:
        # inherited stdout, child killed at the bound, 124 on expiry.
        if real_timeout:
            enforcer = f'exec "{real_timeout}" {timeout_bound} "$@"'
        else:
            shim = tmp_path / "timeout_shim.py"
            # Killed as a group, like timeout(1): otherwise the `sleep` grandchild holds the
            # captured stdout open and setup.sh's command substitution waits it out anyway.
            shim.write_text(
                "import os, signal, subprocess, sys\n"
                "child = subprocess.Popen(sys.argv[2:], start_new_session = True)\n"
                "try:\n"
                "    sys.exit(child.wait(timeout = float(sys.argv[1])))\n"
                "except subprocess.TimeoutExpired:\n"
                "    try:\n"
                "        os.killpg(os.getpgid(child.pid), signal.SIGKILL)\n"
                "    except (ProcessLookupError, PermissionError):\n"
                "        child.kill()\n"
                "    child.wait()\n"
                "    sys.exit(124)\n",
                encoding = "utf-8",
            )
            enforcer = f'exec "{sys.executable}" "{shim}" {timeout_bound} "$@"'
        _write_exec(
            stub_bin / "timeout",
            "#!/bin/sh\n" f'printf "%s\\n" "$*" >> "{timeout_log}"\n' "shift\n" f"{enforcer}\n",
        )
    else:
        # No `timeout` on PATH, and only the block's own utilities reachable. No `cut` and
        # no `uname`: both are coreutils, and the block must answer without either.
        for tool in ("bash", "grep", "tail", "sh", "sleep", "cat"):
            found = shutil.which(tool)
            assert found, f"missing {tool}"
            os.symlink(found, stub_bin / tool)

    # A tool that answers 127 the way a missing one does, but is reachable, so the block runs
    # with the rest of PATH intact and the failure is isolated to the one binary named.
    for _tool in sabotage:
        _write_exec(
            stub_bin / _tool,
            f'#!/bin/sh\necho "{_tool}: command not found" >&2\nexit 127\n',
        )
    # The host architecture the arch gate consults. Only a positive non-x86_64 answer demotes.
    # Pinned rather than inherited, or the suite would answer differently per runner: on an arm64
    # macOS runner the real `uname -m` demotes every AMD arch, the report is never reached, and 35
    # checks fail for a reason that has nothing to do with what they test. The tests that DO test
    # the gate name the arch they want. Not in the no-timeout mode, where PATH is deliberately cut
    # down to the tools the block must run without, `uname` among them.
    if uname_machine is None and with_timeout:
        uname_machine = "x86_64"
    if uname_machine is not None:
        _write_exec(
            stub_bin / "uname",
            f'#!/bin/sh\ncase "$1" in -m) echo {uname_machine} ;; *) echo Linux ;; esac\n',
        )

    # Colab's system interpreter is found on PATH, so that is where the fake one goes.
    if colab:
        shutil.copy2(venv / "bin" / "python", stub_bin / "python")
        (stub_bin / "python").chmod(0o755)
    elif path_python is not None:
        # A DIFFERENT answer than the venv's, so a drifted probe changes the report.
        _write_exec(stub_bin / "python", f"#!/bin/sh\nprintf '%s' '{path_python}'\n")

    script = "\n".join(
        [
            _HARNESS_HEAD,
            f'VENV_DIR="{venv_dir if venv_dir is not None else venv}"',
            f"_COLAB_NO_VENV={'true' if colab else 'false'}",
            f"_setup_nvidia_usable={'true' if nvidia else 'false'}",
            f"_setup_amd_detected={'true' if amd else 'false'}",
            f'_setup_gfx="{gfx}"',
            f'_setup_mkt="{marketing}"',
            block_text,
            _HARNESS_TAIL,
        ]
    )
    run_env = dict(os.environ)
    run_env.pop("UNSLOTH_SKIP_TORCH_GPU_CHECK", None)
    run_env.pop("UNSLOTH_TORCH_INDEX_URL", None)
    run_env.pop("UNSLOTH_TORCH_INDEX_FAMILY", None)
    run_env.pop("UNSLOTH_TORCH_BACKEND", None)
    for _mask in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "UNSLOTH_NO_TORCH",
    ):
        run_env.pop(_mask, None)
    run_env.update(env or {})
    run_env["PATH"] = (
        str(stub_bin) if not with_timeout else f"{stub_bin}{os.pathsep}{run_env.get('PATH', '')}"
    )
    started = time.monotonic()
    completed = subprocess.run(
        ["bash", "-c", script], capture_output = True, text = True, timeout = 120, env = run_env
    )
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "elapsed": time.monotonic() - started,
        "calls": (venv / "calls.log").read_text(encoding = "utf-8")
        if (venv / "calls.log").exists()
        else "",
        "timeout_args": timeout_log.read_text(encoding = "utf-8") if timeout_log.exists() else "",
    }


def _answer(
    available: str,
    count: str = "0",
    version: str = "2.9.0+cpu",
    hip: str = "",
    xpu: str = "0",
) -> str:
    return f"UNSLOTHTORCHGPU={available}|{count}|{version}|{hip}|{xpu}"


pytestmark = pytest.mark.skipif(os.name == "nt", reason = "setup.sh is the POSIX installer")


def test_amd_gpu_invisible_to_torch_is_reported_loudly(block, tmp_path):
    """The whole point of #8473: say the two verdicts disagree, and which is which."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block, venv, tmp_path, amd = True, gfx = "gfx1201", marketing = "Radeon RX 9070 XT"
    )
    out = result["stdout"]
    assert "STEP|gpu check|PyTorch cannot see the AMD GPU reported above|ERR" in out
    # The announcement is repeated verbatim, so the report is falsifiable on its own.
    assert "SUB|detected by the installer: AMD ROCm (gfx1201) -- Radeon RX 9070 XT|ERR" in out
    assert f"SUB|torch.cuda.is_available() is False in {venv}|ERR" in out
    assert "SUB|torch 2.9.0+cpu, device_count 0, torch.version.hip none|ERR" in out
    # Conditional, and scoped to PyTorch: with a Vulkan GGUF bundle the monitor shows real VRAM.
    assert (
        "SUB|PyTorch training and GPU inference are unavailable; chat and GGUF still work.|ERR"
        in out
    )
    # hardware.py leaves CHAT_ONLY true on the fallback and disables Train/Export.
    assert "will run on CPU" not in out
    assert (
        'SUB|If the Live monitor shows VRAM "--" and "No visible GPU", that is this, not a second bug.|ERR'
        in out
    )
    assert "Studio will run CPU-only" not in out
    assert "No visible GPU" in out
    assert "https://github.com/unslothai/unsloth/issues" in out
    # Loud, never fatal: a CPU-only Studio still chats.
    assert "BLOCK_DONE" in out
    assert result["returncode"] == 0


def test_hip_version_is_reported_when_torch_has_one(block, tmp_path):
    """A +rocm wheel that still sees nothing is a different fault from a CPU wheel."""
    venv = _make_venv(tmp_path, stdout = _answer("0", version = "2.9.0+rocm6.4", hip = "6.4.43482"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1151")
    assert (
        "SUB|torch 2.9.0+rocm6.4, device_count 0, torch.version.hip 6.4.43482|ERR"
        in result["stdout"]
    )


def test_a_working_amd_gpu_prints_no_mismatch(block, tmp_path):
    venv = _make_venv(tmp_path, stdout = _answer("1", count = "1", version = "2.9.0+rocm6.4", hip = "6.4"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201")
    out = result["stdout"]
    assert "gpu check" not in out
    assert "cannot see" not in out
    assert "VSUB|torch sees 1 CUDA device(s), xpu false (torch 2.9.0+rocm6.4, hip 6.4)" in out


def test_nvidia_host_is_named_as_nvidia(block, tmp_path):
    """The mismatch is not AMD-specific, and calling an NVIDIA host AMD would be worse than silence."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, nvidia = True)
    out = result["stdout"]
    assert "STEP|gpu check|PyTorch cannot see the NVIDIA GPU reported above|ERR" in out
    assert "AMD" not in out


def test_a_banner_cannot_spoof_the_answer(block, tmp_path):
    """torch imports print to stdout on some hosts; only a line-anchored sentinel is the answer.

    Spoofed on BOTH sides, because the reader takes the last match: a leading banner alone
    would be caught by the tail while an unanchored match survives it.
    """
    venv = _make_venv(
        tmp_path,
        stdout = (
            "warning: overriding UNSLOTHTORCHGPU=1|8|2.9.0+rocm6.4|6.4\n"
            + _answer("0")
            + "\nnote: trailing UNSLOTHTORCHGPU=1|8|2.9.0+rocm6.4|6.4"
        ),
    )
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201")
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


def test_the_two_guards_against_a_spoofed_answer_both_exist(block):
    """Either guard alone rejects a mid-line sentinel, so the behavioural test above cannot tell
    them apart. Asserted per guard."""
    assert "grep '^UNSLOTHTORCHGPU='" in block
    assert '"${_setup_torch_line#UNSLOTHTORCHGPU=}"' in block


def test_a_crashing_probe_warns_and_accuses_nobody(block, tmp_path):
    """A probe that did not answer says nothing about the GPU. It must not read as "no GPU"."""
    venv = _make_venv(tmp_path, exit_code = 1)
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201")
    out = result["stdout"]
    assert "gpu check" not in out
    assert (
        "SUB|[WARN] could not check whether PyTorch sees this GPU (the probe crashed or did not answer within 90s).|WARN"
        in out
    )
    assert result["returncode"] == 0


def test_a_gguf_only_venv_says_nothing(block, tmp_path):
    """No torch installed is not a mismatch, and a warning there is noise on every update."""
    venv = _make_venv(tmp_path, exit_code = 1, torch_on_disk = False)
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201")
    out = result["stdout"]
    assert "gpu check" not in out
    assert "could not check" not in out
    assert result["returncode"] == 0


def test_a_hanging_import_cannot_hang_the_installer(block, tmp_path):
    """`import torch` on a stalled GPU driver is the classic hang, and this probe exists for
    exactly those hosts."""
    venv = _make_venv(tmp_path, sleep_seconds = 60, stdout = _answer("1", count = "1"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", timeout_bound = 3)
    assert result["elapsed"] < 30
    assert "could not check whether PyTorch sees this GPU" in result["stdout"]
    assert result["returncode"] == 0
    # ...and the bound setup.sh asked for is the source's, not the short one enforced here.
    assert result["timeout_args"].split()[0] == "90"
    assert str(venv / "bin" / "python") in result["timeout_args"]


def test_the_probe_runs_where_timeout_is_missing(block, tmp_path):
    """Base macOS and minimal images have no timeout(1), so only the probe's SIGALRM bounds them."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", with_timeout = False)
    assert result["calls"].count("call") == 1
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


def test_both_probe_arms_carry_the_in_process_deadline(block):
    """Per call site: one arm losing signal.alarm leaves that host unbounded, and a file-level
    check passes with the other arm intact."""
    arms = [line for line in block.splitlines() if '-c "$_setup_torch_probe"' in line]
    assert len(arms) == 2, arms
    assert all('"$_setup_torch_py"' in line for line in arms), arms
    # One arm bounded by timeout(1), one for hosts without it; the shared probe string carries the
    # in-process deadline for both.
    assert sum(1 for line in arms if "timeout 90 " in line) == 1, arms
    assert block.count("_setup_torch_probe='import signal; signal.alarm(90); ") == 1


def test_no_accelerator_means_no_interpreter_launch(block, tmp_path):
    """A CPU-only host must not pay for an `import torch` on every update."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path)
    assert result["calls"] == ""
    assert result["stdout"].strip() == "BLOCK_DONE"


def test_the_check_can_be_switched_off(block, tmp_path):
    """An escape hatch for hosts where probing torch is itself the problem."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        env = {"UNSLOTH_SKIP_TORCH_GPU_CHECK": "1"},
    )
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        {"UNSLOTH_TORCH_INDEX_FAMILY": "cpu"},
        {"UNSLOTH_TORCH_INDEX_FAMILY": "CPU"},
        {"UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/cpu/"},
        {"UNSLOTH_TORCH_INDEX_URL": "https://mirror.internal/whl/cpu?token=abc"},
    ],
)
def test_an_explicit_cpu_pin_is_a_request_not_a_fault(block, tmp_path, env):
    """install_python_stack force-reinstalls the CPU wheel for a cpu leaf, so torch answering
    False is the pin working."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        {"UNSLOTH_TORCH_INDEX_FAMILY": " cpu "},
        {"UNSLOTH_TORCH_INDEX_FAMILY": "\tCPU\n"},
        {"UNSLOTH_TORCH_INDEX_URL": " https://download.pytorch.org/whl/cpu "},
        {"UNSLOTH_TORCH_INDEX_URL": " https://mirror.internal/whl/cpu?token=abc "},
        # Pins the ORDER: trimming after the trailing-slash loop leaves the slash on (the value
        # ends in a space, so the loop never fires) and the leaf comes out empty, not "cpu".
        {"UNSLOTH_TORCH_INDEX_URL": " https://download.pytorch.org/whl/cpu/ "},
    ],
)
def test_a_padded_cpu_pin_is_still_a_request_not_a_fault(block, tmp_path, env):
    """get_torch_index_url (install.sh:3272) and Trim-IndexPathSlashes (setup.ps1:863) both trim
    before resolving, so a padded " cpu " really does install the CPU wheel."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        {"UNSLOTH_TORCH_INDEX_FAMILY": "   "},
        {"UNSLOTH_TORCH_INDEX_URL": "\t\n"},
    ],
)
def test_a_whitespace_only_pin_is_not_a_gpu_pin(block, tmp_path, env):
    """A whitespace-only override is unset to install.sh (install.sh:3952). Untrimmed it leaves a
    non-empty leaf, which reads as a GPU pin below and overrides the arch gate."""
    # a whitespace pin is not a GPU pin, so this host is CPU-routed.
    venv = _make_venv(tmp_path, stdout = _answer("0"), torch_local_label = "+cpu")
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1010", env = env)
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        {"UNSLOTH_TORCH_INDEX_URL": "   ", "UNSLOTH_TORCH_INDEX_FAMILY": "cpu"},
        {"UNSLOTH_TORCH_INDEX_URL": "\t\n", "UNSLOTH_TORCH_INDEX_FAMILY": " CPU "},
    ],
)
def test_a_blank_url_falls_back_to_the_family(block, tmp_path, env):
    """get_torch_index_url trims the URL BEFORE choosing, so a blank URL is unset there and the
    family wins. A single ${URL:-${FAMILY}} picks the blank URL instead, because a space is
    non-empty to :-, and the family never gets read."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


def test_a_blank_url_falls_back_to_a_gpu_family(block, tmp_path):
    """Same precedence, other direction: a cu128 family behind a blank URL is a GPU pin."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1010",
        env = {"UNSLOTH_TORCH_INDEX_URL": "  ", "UNSLOTH_TORCH_INDEX_FAMILY": "cu128"},
    )
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        {"UNSLOTH_TORCH_INDEX_FAMILY": "rocm6.4"},
        {"UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128"},
        {"UNSLOTH_TORCH_INDEX_URL": "https://mirror.internal/whl/cpu-private"},
        {"UNSLOTH_TORCH_INDEX_FAMILY": " cu128 "},
    ],
)
def test_a_gpu_pin_is_still_reconciled(block, tmp_path, env):
    """EXACT cpu leaf only: a cu128 or rocm pin asked for a GPU wheel."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


def test_an_installer_resolved_cpu_backend_is_not_a_fault(block, tmp_path):
    """install.sh exports UNSLOTH_TORCH_BACKEND from the index it RESOLVED, so a host it
    deliberately sent to CPU arrives with the CPU wheel it asked for and _setup_amd_detected
    still true."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx906",
        env = {"UNSLOTH_TORCH_BACKEND": "cpu"},
    )
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


@pytest.mark.parametrize("backend", ["rocm", "cuda", ""])
def test_a_gpu_backend_or_no_backend_is_still_reconciled(block, tmp_path, backend):
    """Only the exact "cpu". Unset is the normal standalone `studio update` state, the run this
    check exists for."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        env = {"UNSLOTH_TORCH_BACKEND": backend},
    )
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        {"HIP_VISIBLE_DEVICES": "-1"},
        {"HIP_VISIBLE_DEVICES": ""},
        {"ROCR_VISIBLE_DEVICES": "-1"},
        {"HIP_VISIBLE_DEVICES": " -1 "},
        # clr discards everything to the right of the first invalid entry, and a negative index is
        # invalid, so a leading negative leaves zero agents just as a bare "-1" does.
        {"HIP_VISIBLE_DEVICES": "-1,0"},
        {"ROCR_VISIBLE_DEVICES": "-2"},
    ],
)
def test_a_hidden_amd_gpu_is_not_a_broken_one(block, tmp_path, env):
    """The KFD sysfs fallback reads the kernel topology and ignores the mask, so a user who hid
    every AMD device still gets the GPU announced and a torch that correctly sees nothing."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        {"HIP_VISIBLE_DEVICES": "0"},
        {"ROCR_VISIBLE_DEVICES": "1,0"},
        # The negative is not leading, so device 0 survives it: a selection, not a hide.
        {"HIP_VISIBLE_DEVICES": "0,-1"},
    ],
)
def test_a_mask_that_selects_a_gpu_is_still_reconciled(block, tmp_path, env):
    """Only a hide-ALL mask: muting on a selection would silence every host that pins its GPU."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        # ROCr decides which agents exist at all, so it hides even when clr's own mask selects.
        {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": ""},
        {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "-1"},
        {"HIP_VISIBLE_DEVICES": "0,1", "ROCR_VISIBLE_DEVICES": "-1,0"},
        # ROCR selects, but clr falls through to CUDA because HIP is absent, and that hides.
        {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "-1"},
        {"ROCR_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": ""},
    ],
)
def test_the_two_amd_mask_layers_compose(block, tmp_path, env):
    """ROCr filters agents, then clr masks the survivors, so either layer alone can hide the lot.

    Judging only the first mask that happened to be set read "hidden at one layer, selected at the
    other" as a plain selection and accused a torch that was correctly seeing nothing.
    """
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


def test_both_amd_layers_selecting_is_still_reconciled(block, tmp_path):
    """Neither layer hides, so a blind torch is a real fault and must still be reported."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        env = {"HIP_VISIBLE_DEVICES": "0", "ROCR_VISIBLE_DEVICES": "0"},
    )
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        # _setup_cvd_hides_nvidia already refuses these two before the block is reached; kept so
        # the property is pinned here even if that helper is ever narrowed.
        {"CUDA_VISIBLE_DEVICES": ""},
        {"CUDA_VISIBLE_DEVICES": "-1"},
        # These are the ones it lets through. CUDA discards everything to the right of an invalid
        # index, so a leading negative leaves no visible device: "If the invalid index is first in
        # the list (e.g. -1,0,1), no devices are visible" (CUDA Programming Guide, environment
        # variables). Reporting them would blame the install for the user's own mask.
        {"CUDA_VISIBLE_DEVICES": "-1,0"},
        {"CUDA_VISIBLE_DEVICES": "-2"},
        {"CUDA_VISIBLE_DEVICES": " -1,0 "},
    ],
)
def test_a_hidden_nvidia_gpu_is_not_a_broken_one(block, tmp_path, env):
    """Same rule as the AMD arm, on the NVIDIA side."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = False, nvidia = True, env = env)
    assert "gpu check" not in result["stdout"]


def test_an_nvidia_mask_that_selects_a_gpu_is_still_reconciled(block, tmp_path):
    """The negative is not leading, so device 0 survives it. A torch that cannot see device 0
    there is a real fault, and muting it would lose the report this block exists for."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block, venv, tmp_path, amd = False, nvidia = True, env = {"CUDA_VISIBLE_DEVICES": "0,-1"}
    )
    assert "PyTorch cannot see the NVIDIA GPU reported above" in result["stdout"]


def test_a_bare_cuda_mask_hides_the_amd_card_too(block, tmp_path):
    """ROCm layers HIP/ROCR on CUDA_VISIBLE_DEVICES and falls back to it when neither is set."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block, venv, tmp_path, amd = True, gfx = "gfx1201", env = {"CUDA_VISIBLE_DEVICES": "-1"}
    )
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


def test_a_mixed_host_steered_to_its_amd_card_is_still_reconciled(block, tmp_path):
    """First-set-wins keeps the fallback narrow: hiding NVIDIA while naming an AMD device is a
    selection."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        env = {"HIP_VISIBLE_DEVICES": "0", "CUDA_VISIBLE_DEVICES": "-1"},
    )
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


def test_the_mask_check_needs_no_coreutils(block, tmp_path):
    """Trimmed with bash expansion rather than `tr`: without coreutils on PATH the pipe fails,
    leaving the mask empty, which reads as hide-all and silences every host."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        with_timeout = False,
        env = {"CUDA_VISIBLE_DEVICES": "0"},
    )
    assert result["calls"].count("call") == 1
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


def test_an_empty_hip_mask_shadows_rocr(block, tmp_path):
    """First-set-wins, like the runtime: an empty HIP mask hides everything even when ROCR
    names a device."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        env = {"HIP_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": "0"},
    )
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


@pytest.mark.parametrize("marker", ["env", "manifest", "file"])
def test_a_no_torch_install_never_launches_the_interpreter(block, tmp_path, marker):
    """A GGUF-only install has no torch to reconcile, so without this POSIX half of setup.ps1's
    $NoTorchMode a no-torch venv carrying a user-added CPU torch got the red mismatch."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    env = {}
    if marker == "env":
        env = {"UNSLOTH_NO_TORCH": "1"}
    elif marker == "manifest":
        (venv / "unsloth_install_manifest.json").write_text(
            '{"schema": 1, "no_torch": true}', encoding = "utf-8"
        )
    else:
        (venv / ".unsloth-no-torch").write_text("", encoding = "utf-8")
    result = _run_block(block, venv, tmp_path, nvidia = True, env = env)
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


def test_a_torch_install_is_not_read_as_no_torch(block, tmp_path):
    """The manifest key is false on every normal install, so matching the name alone mutes all."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    (venv / "unsloth_install_manifest.json").write_text(
        '{"schema": 1, "no_torch": false}', encoding = "utf-8"
    )
    result = _run_block(block, venv, tmp_path, nvidia = True)
    assert "PyTorch cannot see the NVIDIA GPU reported above" in result["stdout"]


def test_a_missing_interpreter_is_skipped_by_name(block, tmp_path):
    """setup.sh runs before the venv exists in some repair paths. Asserted on the skip line, not
    on silence: without the guard the probe merely fails to exec, which is silent too."""
    venv = tmp_path / "novenv"
    venv.mkdir()
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201")
    assert "VSUB|torch GPU visibility check skipped: no interpreter at" in result["stdout"]
    assert "gpu check" not in result["stdout"]
    assert result["returncode"] == 0


def test_the_check_runs_after_the_gpu_summary_and_before_the_llama_step():
    """_setup_gfx and _setup_amd_detected are computed by the summary above."""
    text = SETUP_SH.read_text(encoding = "utf-8")
    summary = text.index("# ── GPU detection summary")
    announcement = text.index('step "gpu" "AMD ROCm ($_setup_gfx)"')
    check = text.index(_BLOCK_START)
    llama = text.index(_BLOCK_END)
    assert summary < announcement < check < llama


def test_a_working_xpu_host_is_not_accused_of_running_on_cpu(block, tmp_path):
    """Hybrid Intel/NVIDIA on the XPU wheel: torch.cuda.is_available() is False and the machine
    still runs on its GPU, because _detect_hardware_locked falls through CUDA -> XPU (#8473)."""
    venv = _make_venv(tmp_path, stdout = _answer("0", xpu = "1"))
    result = _run_block(block, venv, tmp_path, nvidia = True)
    out = result["stdout"]
    assert "PyTorch cannot see" not in out
    assert "No visible GPU" not in out
    assert "BLOCK_DONE" in out
    assert result["returncode"] == 0


def test_the_verbose_line_does_not_call_zero_cuda_devices_the_total(block, tmp_path):
    """device_count() is CUDA-only, so it is 0 on the host the XPU answer suppresses."""
    venv = _make_venv(tmp_path, stdout = _answer("0", version = "2.9.0+xpu", xpu = "1"))
    result = _run_block(block, venv, tmp_path, nvidia = True, env = {"UNSLOTH_VERBOSE": "1"})
    out = result["stdout"]
    assert "VSUB|torch sees 0 CUDA device(s), xpu true (torch 2.9.0+xpu, hip none)" in out
    assert "GPU(s)" not in out


def test_a_cpu_only_hybrid_host_is_still_reported(block, tmp_path):
    """The suppression is XPU-specific, not a blanket mute."""
    venv = _make_venv(tmp_path, stdout = _answer("0", xpu = "0"))
    result = _run_block(block, venv, tmp_path, nvidia = True)
    assert "PyTorch cannot see the NVIDIA GPU reported above" in result["stdout"]


def test_a_colab_runtime_probes_the_interpreter_its_deps_went_into(block, tmp_path):
    """Colab has no Unsloth venv: setup.sh installs the deps into the SYSTEM python and sets
    _COLAB_NO_VENV, so a guard requiring $VENV_DIR/bin/python skipped the probe there."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block, venv, tmp_path, nvidia = True, colab = True, venv_dir = tmp_path / "no_such_venv"
    )
    out = result["stdout"]
    assert "skipped" not in out
    assert result["calls"].count("call") == 1
    assert "STEP|gpu check|PyTorch cannot see the NVIDIA GPU reported above|ERR" in out
    # The interpreter it actually asked, not a venv path that does not exist on Colab.
    assert f"SUB|torch.cuda.is_available() is False in {tmp_path / 'stubbin' / 'python'}|ERR" in out
    assert result["returncode"] == 0


def test_a_working_colab_runtime_prints_no_mismatch(block, tmp_path):
    """Same expansion, other direction: Colab is where most working GPUs are."""
    venv = _make_venv(tmp_path, stdout = _answer("1", count = "1", version = "2.9.0+cu128"))
    result = _run_block(
        block, venv, tmp_path, nvidia = True, colab = True, venv_dir = tmp_path / "no_such_venv"
    )
    assert "cannot see" not in result["stdout"]
    assert (
        "VSUB|torch sees 1 CUDA device(s), xpu false (torch 2.9.0+cu128, hip none)"
        in result["stdout"]
    )


def test_a_colab_probe_that_does_not_answer_still_warns(block, tmp_path):
    """The GGUF-only silence keys on the venv's torch on disk, and Colab has no venv layout."""
    venv = _make_venv(tmp_path, exit_code = 1, torch_on_disk = False)
    result = _run_block(
        block, venv, tmp_path, nvidia = True, colab = True, venv_dir = tmp_path / "no_such_venv"
    )
    assert "could not check whether PyTorch sees this GPU" in result["stdout"]
    assert result["returncode"] == 0


def test_the_venv_interpreter_still_wins_where_there_is_a_venv(block, tmp_path):
    """Colab support must not redirect the normal path onto a system python."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        path_python = _answer("1", count = "1"),
    )
    assert result["calls"].count("call") == 1
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]
    assert f"SUB|torch.cuda.is_available() is False in {venv}|ERR" in result["stdout"]


def test_an_xpu_wheel_with_a_dead_runtime_is_still_reported(block, tmp_path):
    """The suppression must turn on what the probe SAW, not the wheel label: a +xpu wheel with a
    broken Intel runtime falls to CPU exactly like the hosts this exists for (#8473)."""
    venv = _make_venv(tmp_path, stdout = _answer("0", version = "2.9.0+xpu", xpu = "0"))
    result = _run_block(block, venv, tmp_path, nvidia = True)
    assert "PyTorch cannot see the NVIDIA GPU reported above" in result["stdout"]
    assert "torch 2.9.0+xpu" in result["stdout"]


# ── The supported-arch gate: hosts Unsloth deliberately puts on CPU torch ────────────────────
#
# _setup_amd_detected is "an AMD GPU is present", not "Unsloth installed GPU wheels for it".
# install.sh routes Vega / RDNA1 / unreadable-arch hosts to CPU torch on purpose, and
# UNSLOTH_TORCH_BACKEND cannot cover them: `studio update` runs setup.sh and never install.sh, so
# it is unset on exactly the repeat path. Mirrors $AmdHasGpuWheels (setup.ps1:2593).


@pytest.mark.parametrize(
    ("gfx", "marketing"),
    [
        ("gfx1010", "Radeon RX 5700 XT"),  # RDNA1, no wheels in any index Unsloth uses
        ("gfx1012", "Radeon RX 5500 XT"),  # RDNA1
        ("gfx803", "Radeon RX 580 Series"),  # GCN4 / Polaris
    ],
)
def test_an_arch_with_no_gpu_wheels_is_not_accused(block, tmp_path, gfx, marketing):
    """CPU torch is the CORRECT outcome on these hosts, and the report repeats on every update.

    The venv must hold a CPU wheel to be that host. A +rocm wheel on an unmapped arch is a
    different host entirely, covered below: install.sh routes any readable arch with a readable
    ROCm version to a generic rocmX.Y index, so gfx1010 really can end up on a ROCm build.
    """
    venv = _make_venv(tmp_path, stdout = _answer("0"), torch_local_label = "+cpu")
    result = _run_block(block, venv, tmp_path, amd = True, gfx = gfx, marketing = marketing)
    assert "gpu check" not in result["stdout"]
    assert "unsloth/issues" not in result["stdout"]
    # Cheap as well as quiet: nothing to reconcile, so the update must not pay for an
    # interpreter launch either.
    assert result["calls"] == ""
    assert result["returncode"] == 0


@pytest.mark.parametrize("gfx", ["gfx1010", "gfx1012", "gfx803"])
def test_an_unmapped_arch_on_a_real_rocm_wheel_is_still_reported(block, tmp_path, gfx):
    """The arch table is not install.sh's routing, so it cannot excuse this host.

    get_torch_index_url consults _amd_arch_index_family_for_gfx only in its unreadable-arch and
    no-ROCm-version fallbacks. With both readable it routes to a generic $_base/rocmX.Y, so an
    arch absent from the per-arch table still receives a ROCm build. Dismissing it as a CPU route
    silenced exactly the mismatch #8473 exists to report.
    """
    venv = _make_venv(tmp_path, stdout = _answer("0"), torch_local_label = "+rocm6.4")
    result = _run_block(block, venv, tmp_path, amd = True, gfx = gfx)
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]
    assert result["calls"] != "", "the probe never ran, so the wheel was never reconciled"


@pytest.mark.parametrize("gfx", ["gfx1010", "gfx803"])
def test_a_rocm_build_with_no_local_label_is_still_reported(block, tmp_path, gfx):
    """AMD's own builds and source builds record ROCm in `hip`, not in the local label.

    "2.5.0a0+git1234567" with a populated hip field is a supported shape elsewhere in the tree
    (tests/studio/install/test_amd_fastpath_probe.py:118). Reading only __version__ classified it
    as a CPU route, so on an arch outside the wheel table the probe never ran.
    """
    venv = _make_venv(
        tmp_path,
        stdout = _answer("0", version = "2.5.0a0+git1234567", hip = "6.2.41134"),
        torch_local_label = "a0+git1234567",
        torch_hip = "6.2.41134",
    )
    result = _run_block(block, venv, tmp_path, amd = True, gfx = gfx)
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]
    assert result["calls"] != "", "the probe never ran, so the wheel was never reconciled"


def test_a_null_hip_field_is_not_read_as_a_rocm_build(block, tmp_path):
    """Every CUDA and CPU wheel carries `hip: Optional[str] = None`, so an unquoted value must
    not count. Without that the wheel read would answer true for every venv and re-open the
    report on the RDNA 1 / Polaris hosts the arch table exists to keep quiet."""
    venv = _make_venv(tmp_path, stdout = _answer("0"), torch_local_label = "+cpu")
    assert "hip: Optional[str] = None" in (
        venv / "lib" / "python3.11" / "site-packages" / "torch" / "version.py"
    ).read_text(encoding = "utf-8")
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1010")
    assert "gpu check" not in result["stdout"]
    assert result["calls"] == ""


def test_an_unreadable_arch_is_not_accused(block, tmp_path):
    """The KFD-sysfs path: /dev/kfd names an AMD vendor_id but no gfx arch is readable.
    get_torch_index_url cannot route GPU wheels without an arch (install.sh:3330 warns and
    returns the cpu index), and this is EVERY such host, not a named arch."""
    venv = _make_venv(tmp_path, stdout = _answer("0"), torch_local_label = "+cpu")
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "", marketing = "")
    assert "gpu check" not in result["stdout"]
    assert result["calls"] == ""
    assert result["returncode"] == 0


@pytest.mark.parametrize("gfx", ["gfx1201", "gfx1100", "gfx1030", "gfx90a", "gfx908"])
def test_an_arch_with_gpu_wheels_is_still_reported(block, tmp_path, gfx):
    """The other direction, and the whole point of #8473: these arches DO get GPU wheels."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = gfx)
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]
    assert f"SUB|detected by the installer: AMD ROCm ({gfx})|ERR" in result["stdout"]


@pytest.mark.parametrize("gfx", ["gfx942", "gfx950", "gfx906", "gfx900"])
def test_the_linux_only_arches_are_reported(block, tmp_path, gfx):
    """These four are why the POSIX list is not a copy of setup.ps1's. gfx906 also gets GPU
    wheels from Unsloth's own rocm6.3 reroute (install.sh:4298); all four are built into the
    generic rocm wheels a ROCm host resolves (upstream PYTORCH_ROCM_ARCH carries
    gfx900/906/908/90a/942 for 2.6-2.11, gfx950 from 2.10). None are in $_rocmWheelArches
    because AMD publishes no Windows wheels for them."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = gfx)
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        {"UNSLOTH_TORCH_INDEX_FAMILY": "rocm6.4"},
        {"UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/rocm6.4"},
        {"UNSLOTH_TORCH_INDEX_FAMILY": "cu128"},
    ],
)
def test_a_gpu_index_pin_reconciles_even_on_an_unwheeled_arch(block, tmp_path, env):
    """An explicit GPU index pin is a request for GPU wheels, which install.sh honours verbatim
    even on an arch with no wheels of its own. Matches $_amdPinIsGpu (setup.ps1:2664)."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1010", env = env)
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


def test_a_cpu_pin_still_wins_over_a_wheeled_arch(block, tmp_path):
    """The new gate widens suppression; it must not narrow the exclusions already there."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        env = {"UNSLOTH_TORCH_INDEX_FAMILY": "cpu"},
    )
    assert "gpu check" not in result["stdout"]
    assert result["calls"] == ""


@pytest.mark.parametrize(
    ("gfx", "reported"),
    [
        ("GFX1201", True),  # UNSLOTH_ROCM_GFX_ARCH is not lowercased on the way in
        ("gfx906:sramecc-:xnack-", True),  # a copied HIP gcnArchName
        ("  gfx1201  ", True),
        ("GFX1010", False),  # normalisation must not become a bypass
        ("gfx803:xnack-", False),
    ],
)
def test_the_arch_is_normalised_before_the_membership_test(block, tmp_path, gfx, reported):
    """$_setup_gfx takes UNSLOTH_ROCM_GFX_ARCH verbatim (setup.sh:1837), so an unnormalised
    comparison hands every accused host a one-variable bypass."""
    # normalisation is the subject; the wheel must match the arch verdict.
    venv = _make_venv(tmp_path, stdout = _answer("0"), torch_local_label = "+cpu")
    result = _run_block(block, venv, tmp_path, amd = True, gfx = gfx)
    assert ("PyTorch cannot see the AMD GPU reported above" in result["stdout"]) is reported


def test_the_nvidia_path_is_untouched_by_the_amd_arch_gate(block, tmp_path):
    """An NVIDIA host never assigns $_setup_gfx at all, and setup.sh runs under `set -u`."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, nvidia = True, gfx = "")
    assert "PyTorch cannot see the NVIDIA GPU reported above" in result["stdout"]
    assert result["returncode"] == 0


# ── Drift guards for the arch list ──────────────────────────────────────────────────────────


def _posix_wheel_arches() -> set[str]:
    """The gfx arches setup.sh's gate accepts, parsed out of the real case block."""
    text = SETUP_SH.read_text(encoding = "utf-8")
    start = text.index("_setup_amd_has_gpu_wheels=false")
    end = text.index("esac", start)
    # gfx90[aA] -> gfx90a: the letter class is part of the real pattern, not a typo.
    body = re.sub(r"\[([a-zA-Z])[a-zA-Z]*\]", lambda m: m.group(1).lower(), text[start:end])
    found = set(re.findall(r"gfx[0-9a-z]+", body))
    assert found, "the arch list is no longer parseable out of setup.sh"
    return found


def _windows_wheel_arches() -> set[str]:
    text = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    start = text.index("$_rocmWheelArches = @(")
    # To the closing paren on its own line: entries carry trailing comments that contain
    # parentheses ("# RDNA 2 (RX 6000)"), so the first ")" is not the end of the array.
    end = text.index("\n)", start)
    found = set(re.findall(r'"(gfx[0-9a-z]+)"', text[start:end]))
    assert found, "$_rocmWheelArches is no longer parseable out of setup.ps1"
    return found


def _install_sh_family_arches() -> set[str]:
    text = (PACKAGE_ROOT / "install.sh").read_text(encoding = "utf-8")
    start = text.index("_amd_arch_index_family_for_gfx() {")
    end = text.index("esac", start)
    found = set(re.findall(r"(gfx[0-9a-z]+)\|?(?=[|)])", text[start:end]))
    assert found, "_amd_arch_index_family_for_gfx is no longer parseable out of install.sh"
    return found


def test_every_arch_unsloth_routes_gpu_wheels_to_is_in_the_posix_gate():
    """The gate's authority is install.sh's own routing, not AMD's support matrix."""
    missing = _install_sh_family_arches() - _posix_wheel_arches()
    assert not missing, (
        f"install.sh routes {sorted(missing)} to AMD per-arch GPU wheels, but setup.sh's "
        f"reconciliation gate would treat those hosts as CPU-by-design and stay silent."
    )


def test_windows_demotes_a_non_x86_host_too():
    """The arch list answers "does this GPU have wheels", not "does this HOST get them".

    setup.sh demotes on `uname -m` because PyTorch and AMD both publish per-arch ROCm wheels for
    one platform each (linux x86_64, win_amd64), so a host outside that reads as wheeled, is
    correctly given CPU torch, and would then be accused on every update of the documented
    routing. Windows needs the same demotion for the same reason.

    It lives on its own flag rather than inside $AmdHasGpuWheels. That flag has seven consumers
    predating this branch and the stale-venv check is one of them: demoting it there expects
    "cpu", meets the +rocm wheel the ROCm override installs with no host-arch gate of its own,
    deletes the venv, and setup exits because the venv is gone.
    """
    text = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    start = text.index("$AmdHasGpuWheels = [bool](")
    end = text.index("\n)", start)
    assign = text[start:end]
    assert "Get-HostMachineArch" not in assign, (
        "$AmdHasGpuWheels demotes an ARM64 host again. Its other consumers include the stale-venv "
        "check, which then expects cpu, wipes a working +rocm venv, and exits."
    )

    gate = next((ln for ln in text.splitlines() if ln.startswith("$_gpuCheckArm64Amd =")), "")
    assert "Get-HostMachineArch" in gate and '-eq "arm64"' in gate, (
        "nothing demotes an ARM64 host any more, so a Windows-on-ARM box whose arch is in "
        "$_rocmWheelArches would be accused of a fault that is the documented routing."
    )


def test_the_arm64_demotion_asks_the_machine_not_a_proxy():
    """It must test the host arch, not the inverse of "do wheels reach this host".

    That proxy is false for two unrelated reasons, ARM64 and an arch with no wheels, and only the
    first is excusable. An x64 host on an unmapped arch with an explicit non-CPU pin is announced
    as AMD, so the inverted form called it an ARM64 candidate; if the pinned index then served a
    wheel with no HIP build, the post-probe excuse silenced a genuine mismatch.
    """
    text = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    code = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))
    gate = next((ln for ln in code.splitlines() if ln.startswith("$_gpuCheckArm64Amd =")), "")
    assert gate, "the ARM64 candidate flag is gone"
    assert "-not $AmdWheelsReachThisHost" not in gate, (
        "the gate infers ARM64 from wheel reachability again, so an x64 unmapped-arch host with "
        "a GPU pin is treated as ARM64 and can be excused into silence."
    )
    assert '(Get-HostMachineArch) -eq "arm64"' in gate

    # And the proxy is gone entirely rather than left sitting unread.
    users = [ln for ln in code.splitlines() if "$AmdWheelsReachThisHost" in ln]
    assert not users, f"$AmdWheelsReachThisHost is dead but still present: {users}"


def test_the_arm64_excuse_is_decided_on_the_wheel_not_the_host():
    """Excusing an ARM64 AMD host outright contradicts the fact that made the demotion necessary.

    The ROCm override carries no host-arch gate, so the emulated x64 Python this installer prefers
    can hold a real win_amd64 +rocm wheel here, and an explicit ROCm pin routes the same way. A
    blanket exclusion silences precisely the report #8473 exists to make. torch.version.hip is a
    BUILD constant, so it names the installed wheel even when the runtime cannot initialise, which
    is the state under test.
    """
    text = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    code = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))
    assert "-not $_gpuCheckArm64Amd -and" not in code, (
        "the gate excuses an ARM64 AMD host before the probe runs, so a +rocm wheel that cannot "
        "see its GPU is never reported there."
    )
    assert (
        "$_gpuCheckArm64Excused = $_gpuCheckArm64Amd -and -not $_gpuVisibility.Hip" in code
    ), "the ARM64 excuse no longer reads the wheel the probe found."
    assert "-not $_gpuCheckArm64Excused" in code

    # It has to be decided after the probe, or it reads a Hip that does not exist yet.
    probe = code.index("$_gpuVisibility = Get-TorchGpuVisibility")
    excuse = code.index("$_gpuCheckArm64Excused = ")
    assert probe < excuse, (
        "the excuse is computed before the probe, so $_gpuVisibility.Hip is always empty and "
        "every ARM64 AMD host is excused regardless of its wheel."
    )


def test_the_posix_cpu_route_excuse_reads_the_wheel_not_only_the_arch_table():
    """The arch table is not install.sh's routing, so it cannot decide this alone.

    get_torch_index_url consults _amd_arch_index_family_for_gfx only in its unreadable-arch and
    no-ROCm-version fallbacks. Once both the gfx arch and a ROCm version are readable it routes to
    a generic $_base/rocmX.Y, so a Linux x86_64 host on gfx1010 or gfx803 with ROCm 6.0+ gets a
    real ROCm build and is announced as AMD, while the table calls it a deliberate CPU route.
    torch.version.hip is a build constant, so it names that wheel even with a dead runtime.
    """
    text = SETUP_SH.read_text(encoding = "utf-8")

    assert "_setup_torch_is_rocm=false" in text
    assert "^__version__ = '[^']*[+]rocm" in text, (
        "the POSIX gate no longer reads the installed wheel, so an unmapped arch routed to a "
        "generic ROCm index is dismissed as a CPU route and never probed."
    )
    # A build with no local label records ROCm in `hip` instead, and only a QUOTED value counts.
    assert "^hip[[:space:]]*(:[^=]*)?=[[:space:]]*'[^']" in text

    entry = text[text.index('if { [ "$_setup_nvidia_usable" = true ]') :]
    entry = entry[: entry.index("then")]
    assert '[ "$_setup_torch_is_rocm" = true ]' in entry, (
        "the entry gate decides on the arch table alone again, so gfx1010 or gfx803 on a real "
        "ROCm wheel stays silent when that wheel cannot see the device."
    )
    # Read off disk, not by importing torch: a genuinely CPU-routed host must not pay for it.
    assert "$VENV_DIR" in text[text.index("_setup_torch_is_rocm=false") : text.index(entry[:60])]

    # The read has to happen before the gate, or the flag is always false.
    assert text.index("_setup_torch_is_rocm=false") < text.index(
        'if { [ "$_setup_nvidia_usable" = true ]'
    ), "the wheel is read after the gate consults it, so it is always false."


def test_a_local_cpu_fallback_is_not_a_mismatch():
    """$ROCmCpuFallback / $XpuCpuFallback mean this run failed to install the GPU wheel and
    force-installed CPU torch on purpose. $InstallerTorchTag carries neither, so without them the
    user reads the install failure and then a red accusation about the result of it.

    Both are declared outside `if (-not $SkipPythonDeps)`: the fast path never enters that block,
    and the fast path is the run this whole check exists for, so a caller's Set-StrictMode would
    make the read fatal on exactly the reported host.
    """
    text = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    assert "-not $_gpuCheckLocalCpuFallback -and" in text
    assert "$_gpuCheckLocalCpuFallback = $XpuCpuFallback -or" in text
    assert "($ROCmCpuFallback -and " in text

    decl = text.index("$ROCmCpuFallback = $false\n$XpuCpuFallback = $false")
    branch = text.index("if (-not $SkipPythonDeps) {")
    assert decl < branch, (
        "the fallback flags are declared inside the dependency-pass branch, so on the fast path "
        "they are never assigned and the gate reads an undefined variable."
    )


def test_the_cpu_fallback_excuse_reads_the_wheel_the_dependency_pass_left():
    """The flag says what setup DECIDED; the dependency pass that runs after it can undo that.

    On Windows _ensure_rocm_torch retries the AMD index precisely because setup fell back to CPU
    (studio/install_python_stack.py:2980), so a run that set $ROCmCpuFallback can still end on a
    ROCm wheel, and a ROCm wheel that cannot see its GPU is the report. Excusing on the flag alone
    silenced #8473 on exactly that wheel.

    $XpuCpuFallback is deliberately NOT reconciled: _ensure_xpu_torch returns on Windows
    (install_python_stack.py:2481), so nothing reinstalls over that fallback, and reading the wheel
    there would contradict the XPU suppression, which is decided on what the probe saw.
    """
    text = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    code = "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("#"))
    assert "($ROCmCpuFallback -and -not (Test-VenvTorchIsRocm -VenvPath $VenvDir))" in code, (
        "the CPU-fallback excuse no longer reads the wheel, so a ROCm build installed by the "
        "repair pass is never reconciled."
    )

    excuse = code[code.index("$_gpuCheckLocalCpuFallback = ") :]
    excuse = excuse[: excuse.index("\nif (")]
    assert "Test-VenvTorchIsXpu" not in excuse
    # Free on the fast path: -and short-circuits, and the flag is never set there.
    assert excuse.index("$ROCmCpuFallback") < excuse.index("Test-Venv")


def test_a_rocm_build_with_no_local_label_is_read_as_rocm_on_windows():
    """Same shape as the POSIX read: no local label, ROCm named in `hip`. Test-VenvTorchIsRocm
    also decides whether a venv survives repair (setup.ps1:4228), so widening it there is the
    safe direction as well: an unlabelled ROCm venv was being classed as unknown."""
    text = (PACKAGE_ROOT / "studio" / "setup.ps1").read_text(encoding = "utf-8")
    fn = text[text.index("function Test-VenvTorchIsRocm") :]
    fn = fn[: fn.index("\n}")]
    assert (
        "^hip\\s*(:[^=]*)?=\\s*'[^']" in fn
    ), "Test-VenvTorchIsRocm reads the local label alone, so an AMD or source build is unknown."
    assert "__version__\\s*=\\s*'[^']*\\+(rocm|gfx)" in fn


def test_the_posix_gate_covers_the_windows_list():
    """setup.sh's list is a superset of setup.ps1's, never a divergent one: the delta is
    Linux-only by construction (gfx906 / gfx942 / gfx950 have no Windows ROCm wheels)."""
    missing = _windows_wheel_arches() - _posix_wheel_arches()
    assert (
        not missing
    ), f"in setup.ps1 $_rocmWheelArches but not in setup.sh's gate: {sorted(missing)}"
    extra = _posix_wheel_arches() - _windows_wheel_arches()
    assert extra == {"gfx900", "gfx906", "gfx942", "gfx950"}, (
        f"unexpected POSIX-only arches {sorted(extra)}: each one needs a documented Linux "
        f"routing path in install.sh, or it silences a host that does get GPU wheels."
    )


def test_the_known_unwheeled_arches_are_absent_from_both_lists():
    """Pins the reported hosts themselves."""
    for arch in ("gfx803", "gfx1010", "gfx1011", "gfx1012"):
        assert arch not in _posix_wheel_arches()
        assert arch not in _windows_wheel_arches()


# ── The block must survive a host without coreutils, not just avoid `tr` ────────────────────────
# The block twice refuses to depend on coreutils (the mask trim and the arch trim both say so),
# and then split the probe answer with four `cut` calls. Under setup.sh's `set -euo pipefail` an
# unguarded `x=$(cut ...)` does not degrade to an empty value the way a failed `tr` pipe does: it
# exits 127 and takes the whole installer down, at the last step of an otherwise successful run.


def test_a_missing_cut_cannot_abort_the_installer(block, tmp_path):
    """127 from a split utility must not be fatal: this is the LAST step of `studio update`, and
    aborting here fails an install that has already succeeded."""
    venv = _make_venv(tmp_path, stdout = _answer("0", version = "2.9.0+rocm6.4"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        sabotage = ("cut",),
    )
    assert result["returncode"] == 0
    assert "BLOCK_DONE" in result["stdout"]
    # ...and it still answers, rather than surviving by falling silent.
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]
    assert "torch 2.9.0+rocm6.4" in result["stdout"]


def test_the_probe_answer_is_split_without_a_subprocess(block):
    """Pins the mechanism, not just the outcome: a future edit that reaches for `cut`, `awk` or
    `sed` again reintroduces both the 127 and the coreutils dependency the comments disclaim."""
    body = "\n".join(line for line in block.splitlines() if not line.strip().startswith("#"))
    fields = body[body.index("_setup_torch_fields=") :]
    for tool in ("cut", "awk", "sed", "tr"):
        assert f"| {tool} " not in fields and f"|{tool} " not in fields, (
            f"the probe answer is split with `{tool}`, which is coreutils and which setup.sh's "
            f"`set -e` turns into a fatal 127 when it is absent"
        )
    assert "IFS='|' read" in fields


# ── The host architecture, not just the GPU architecture ───────────────────────────────────────
# PyTorch publishes ROCm wheels for linux-x86_64 only, so install.sh returns the cpu index for
# every other machine BEFORE it looks at the gfx arch at all.


def test_a_non_x86_amd_host_is_not_accused(block, tmp_path):
    """aarch64 + a wheeled arch is CPU torch by install.sh's own routing, so reporting it accuses
    a host behaving exactly as designed -- the same failure the arch table prevents for RDNA 1."""
    # aarch64 is CPU torch by install.sh's routing.
    venv = _make_venv(tmp_path, stdout = _answer("0"), torch_local_label = "+cpu")
    for machine in ("aarch64", "arm64", "ppc64le", "riscv64"):
        work = tmp_path / machine
        work.mkdir()
        result = _run_block(
            block,
            venv,
            work,
            amd = True,
            gfx = "gfx942",
            uname_machine = machine,
        )
        assert "gpu check" not in result["stdout"], machine
        assert result["returncode"] == 0


def test_an_x86_amd_host_is_still_reconciled(block, tmp_path):
    """The other direction: the gate must not silence the hosts this report exists for."""
    for machine in ("x86_64", "amd64"):
        work = tmp_path / machine
        work.mkdir()
        venv = _make_venv(work, stdout = _answer("0"))
        result = _run_block(
            block,
            venv,
            work,
            amd = True,
            gfx = "gfx942",
            uname_machine = machine,
        )
        assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"], machine


def test_an_unreadable_architecture_does_not_silence_the_report(block, tmp_path):
    """`uname` is coreutils too. An empty answer is no evidence about the host, and silencing the
    whole report on no evidence is the failure the mask and arch gates both refuse."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        sabotage = ("uname",),
    )
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]
    assert result["returncode"] == 0


def test_the_architecture_gate_does_not_touch_nvidia(block, tmp_path):
    """install.sh scopes the x86_64 gate INSIDE its `no NVIDIA` branch, so aarch64 NVIDIA
    (Jetson, GH200) keeps its CUDA wheels and must still be reconciled."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        nvidia = True,
        uname_machine = "aarch64",
    )
    assert "PyTorch cannot see the NVIDIA GPU reported above" in result["stdout"]


def test_the_architecture_gate_does_not_touch_a_pin(block, tmp_path):
    """install.sh honours an explicit index verbatim, returning before the arch gate, so a pinned
    aarch64 host really does get whatever that index publishes and is still worth reconciling."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1010",
        uname_machine = "aarch64",
        env = {"UNSLOTH_TORCH_INDEX_FAMILY": "rocm6.4"},
    )
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


# ── The two escape hatches must read the same on both platforms ────────────────────────────────


@pytest.mark.parametrize(
    "value",
    ["1", "true", "TRUE", "True", "yes", "YES", "Yes", "on", "ON", " 1 ", "  true  "],
)
def test_the_skip_flag_matches_the_windows_spelling(block, tmp_path, value):
    """setup.ps1:5314 accepts `^\\s*(?i:true|1|yes|on)\\s*$`. The flag is introduced by this
    change and read in exactly two places, so a user told to set it must get the same answer on
    both."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        env = {"UNSLOTH_SKIP_TORCH_GPU_CHECK": value},
    )
    assert "gpu check" not in result["stdout"], value
    assert result["calls"].count("call") == 0, f"{value!r} launched the probe anyway"


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  ", "garbage", "truthy"])
def test_the_skip_flag_does_not_swallow_anything_else(block, tmp_path, value):
    """The other direction: a value Windows would reject must not silence the report here."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        amd = True,
        gfx = "gfx1201",
        env = {"UNSLOTH_SKIP_TORCH_GPU_CHECK": value},
    )
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"], value


@pytest.mark.parametrize("value", ["cpu", "CPU", "Cpu", "cPU"])
def test_a_cpu_backend_is_honoured_whatever_its_case(block, tmp_path, value):
    """install_python_stack.py:3432 lowercases UNSLOTH_TORCH_BACKEND and :2400 tells users to set
    it by hand, so a typed `CPU` gets CPU torch there and must not be accused here."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        nvidia = True,
        env = {"UNSLOTH_TORCH_BACKEND": value},
    )
    assert "gpu check" not in result["stdout"], value
    assert result["calls"].count("call") == 0


@pytest.mark.parametrize("value", ["rocm", "cuda", "ROCM", "xpu", " cpu ", ""])
def test_a_non_cpu_backend_is_still_reconciled(block, tmp_path, value):
    """Folded, not trimmed, to match `.lower()` exactly: `" cpu "` is not cpu to
    install_python_stack either, so silencing it here would be the same bug mirrored."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(
        block,
        venv,
        tmp_path,
        nvidia = True,
        env = {"UNSLOTH_TORCH_BACKEND": value},
    )
    assert "PyTorch cannot see the NVIDIA GPU reported above" in result["stdout"], value


def test_the_no_torch_flag_keeps_the_installers_own_spelling(block):
    """UNSLOTH_NO_TORCH is NOT folded like the skip flag above, deliberately: install.sh:103 reads
    that exact literal list, so a looser reading here would skip the check on a host the installer
    decided to give torch to."""
    body = "\n".join(line for line in block.splitlines() if not line.strip().startswith("#"))
    line = next(ln for ln in body.splitlines() if "UNSLOTH_NO_TORCH" in ln)
    assert "1|true|TRUE|yes|YES|on|ON" in line
    install_sh = (PACKAGE_ROOT / "install.sh").read_text(encoding = "utf-8")
    assert 'case "${UNSLOTH_NO_TORCH:-}" in 1|true|TRUE|yes|YES|on|ON)' in install_sh
