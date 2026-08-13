# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #8473 -- the installer says the GPU is fine and the backend runs CPU-only.

Reporter: AMD host, `unsloth studio update` prints `gpu AMD ROCm (gfx1201)` then
`python dependencies up to date`, and Studio then shows VRAM `--`, "No visible GPU"
and a `CPU` detail line. The installer's GPU line comes from rocminfo / amd-smi /
hipinfo plus a marketing-name table; the backend's verdict is
torch.cuda.is_available() in its own process (on ROCm, get_backend_visible_gpu_info
skips the SMI branch, so torch.cuda is the only thing that can populate devices).
Nothing ever reconciled the two, so the user was told twice the GPU was fine.

setup.sh now runs one bounded probe in the venv after the GPU summary and prints
the mismatch. There is no AMD hardware in CI -- every runner is a hosted
ubuntu/windows/macos box -- so the real block is extracted from setup.sh and run
under bash against a FAKE venv interpreter whose answer, exit code and latency the
test controls, plus a fake `timeout` that records the bound setup.sh asked for
while enforcing a short one, so the hang case finishes in seconds.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
SETUP_SH = PACKAGE_ROOT / "studio" / "setup.sh"

_BLOCK_START = "# ── Does PyTorch see the GPU this installer just announced? ──"
_BLOCK_END = "# ── 7. Prefer prebuilt llama.cpp bundles"

# Colours are the assertion surface for severity: the harness substitutes the names
# themselves, so a report demoted from $C_ERR to $C_WARN fails rather than passing on
# text alone.
_HARNESS_HEAD = """
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
    # An empty or truncated extraction would make every check below pass vacuously.
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
        (venv / "lib" / "python3.11" / "site-packages" / "torch" / "version.py").write_text(
            "__version__ = '2.9.0+rocm6.4'\n", encoding = "utf-8"
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
) -> dict:
    """Run the real setup.sh block with stubbed printers and a stubbed `timeout`."""
    stub_bin = tmp_path / "stubbin"
    stub_bin.mkdir(exist_ok = True)
    timeout_log = tmp_path / "timeout_args.log"
    real_timeout = shutil.which("timeout")
    if with_timeout:
        assert real_timeout, "this test host has no timeout(1) to delegate to"
        # Records the bound setup.sh ASKED for, then enforces a short one, so the
        # hang case is observable without waiting out the real 90 seconds.
        _write_exec(
            stub_bin / "timeout",
            "#!/bin/sh\n"
            f'printf "%s\\n" "$*" >> "{timeout_log}"\n'
            "shift\n"
            f'exec "{real_timeout}" {timeout_bound} "$@"\n',
        )
    else:
        # No `timeout` on PATH at all: the fallback arm of the probe must still run.
        # Only the utilities the block itself uses are reachable.
        for tool in ("bash", "grep", "tail", "cut", "sh", "sleep", "cat"):
            found = shutil.which(tool)
            assert found, f"missing {tool}"
            os.symlink(found, stub_bin / tool)

    # Colab's system interpreter is found on PATH, so that is where the fake one goes.
    if colab:
        shutil.copy2(venv / "bin" / "python", stub_bin / "python")
        (stub_bin / "python").chmod(0o755)
    elif path_python is not None:
        # A DIFFERENT answer than the venv's, so a probe that drifted onto the system
        # interpreter changes the report instead of passing on identical output.
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
    for _mask in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "UNSLOTH_NO_TORCH"):
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
    """The whole point of #8473: say the two verdicts disagree, and say which is which."""
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
    # Naming the symptom the user is about to see is what stops it being filed twice, but only
    # as a conditional: llama.cpp is a separate stack, and with the Vulkan bundle the backend
    # fills inference_gpu from get_vulkan_inference_gpu_info() and the monitor shows that card's
    # real VRAM, so promising a CPU-only Studio and a "--" readout would be false there.
    # "PyTorch", because a false torch.cuda.is_available() says nothing about llama.cpp: a
    # CUDA / HIP / Vulkan GGUF bundle still offloads to the same card.
    assert (
        "SUB|PyTorch training and GPU inference are unavailable; chat and GGUF still work.|ERR"
        in out
    )
    # Not "runs on CPU": hardware.py leaves CHAT_ONLY true on the fallback and disables
    # Train/Export, so promising CPU training is the opposite of what happens. Same sentence
    # the XPU-runtime-unavailable arm above already uses.
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

    The spoof is placed on BOTH sides of the real answer on purpose: the reader takes the last
    match, so a leading banner alone is caught by the tail and an unanchored match survives it.
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
    """The line anchor and the sentinel prefix strip are independent, and either alone rejects a
    mid-line sentinel -- so the behavioural test above cannot tell them apart, and removing one
    silently leaves the reader resting on the other. Asserted here per guard instead."""
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
    """`import torch` on a box with a stalled GPU driver is the classic hang, and this probe
    exists precisely for hosts whose driver is misbehaving."""
    venv = _make_venv(tmp_path, sleep_seconds = 60, stdout = _answer("1", count = "1"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", timeout_bound = 3)
    assert result["elapsed"] < 30
    assert "could not check whether PyTorch sees this GPU" in result["stdout"]
    assert result["returncode"] == 0
    # ...and the bound setup.sh actually asked for is the one in the source, not the
    # short one this test enforces.
    assert result["timeout_args"].split()[0] == "90"
    assert str(venv / "bin" / "python") in result["timeout_args"]


def test_the_probe_runs_where_timeout_is_missing(block, tmp_path):
    """Base macOS and minimal Linux images have no timeout(1); the SIGALRM deadline inside the
    probe is what bounds them, and the check must still happen."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", with_timeout = False)
    assert result["calls"].count("call") == 1
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


def test_both_probe_arms_carry_the_in_process_deadline(block):
    """Per call site, not per file: one arm losing signal.alarm leaves that host unbounded, and a
    file-level check passes with the other arm intact."""
    arms = [line for line in block.splitlines() if '-c "$_setup_torch_probe"' in line]
    assert len(arms) == 2, arms
    assert all('"$_setup_torch_py"' in line for line in arms), arms
    # One arm bounded by timeout(1), one for hosts without it -- both share the same probe
    # string, so the in-process deadline covers each.
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
    """install_python_stack's _explicit_cpu_torch_index_url treats a cpu leaf as authoritative
    and force-reinstalls the CPU wheel, so torch answering False is the pin working. Accusing
    that host of a broken GPU and asking it to file an issue is the false alarm this whole
    check must not become. install.sh already carries the same exclusion for its ROCm line."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


@pytest.mark.parametrize(
    "env",
    [
        {"UNSLOTH_TORCH_INDEX_FAMILY": "rocm6.4"},
        {"UNSLOTH_TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128"},
        {"UNSLOTH_TORCH_INDEX_URL": "https://mirror.internal/whl/cpu-private"},
    ],
)
def test_a_gpu_pin_is_still_reconciled(block, tmp_path, env):
    """The exclusion is the EXACT cpu leaf, not any pin: a cu128 or rocm pin asked for a GPU
    wheel, so a torch that sees nothing under one is exactly the mismatch worth reporting."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


def test_an_installer_resolved_cpu_backend_is_not_a_fault(block, tmp_path):
    """install.sh exports UNSLOTH_TORCH_BACKEND from the index it RESOLVED, so a host it
    deliberately sent to CPU (non-x86_64, ROCm older than 6.0, an unreadable ROCm runtime)
    arrives with the CPU wheel it asked for and _setup_amd_detected still true, since this
    file's AMD detection has no uname or ROCm-version gate. install.sh already explains that
    fallback on screen; repeating it in red and asking for an issue is the false alarm."""
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
    """Only the exact "cpu". Unset is the normal standalone `studio update` state -- the run
    this check exists for -- so treating "not a GPU backend" as CPU would disable the feature
    for everyone who updates without re-running the installer."""
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
    ],
)
def test_a_hidden_amd_gpu_is_not_a_broken_one(block, tmp_path, env):
    """The KFD sysfs fallback reads the kernel topology and ignores the mask, so a user who
    deliberately hid every AMD device still gets the GPU announced and a torch that correctly
    sees nothing. _setup_cvd_hides_nvidia already keeps the masked NVIDIA host out; this is
    the missing AMD half, not a new policy."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert result["calls"] == ""
    assert "gpu check" not in result["stdout"]


@pytest.mark.parametrize("env", [{"HIP_VISIBLE_DEVICES": "0"}, {"ROCR_VISIBLE_DEVICES": "1,0"}])
def test_a_mask_that_selects_a_gpu_is_still_reconciled(block, tmp_path, env):
    """Only a hide-ALL mask. Selecting a device is the opposite of hiding one, and muting
    there would silence every host that pins its GPU."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201", env = env)
    assert "PyTorch cannot see the AMD GPU reported above" in result["stdout"]


def test_an_empty_hip_mask_shadows_rocr(block, tmp_path):
    """First-set-wins, like the runtime: an empty HIP mask hides everything even when ROCR
    names a device, so reading ROCR first would report a host that torch cannot see by
    request."""
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
    """A GGUF-only install has no torch to reconcile. setup.ps1 already excludes it through
    $NoTorchMode; without the POSIX half every update paid for an `import torch` that could
    only fail, and a no-torch venv carrying a user-added CPU torch got the red mismatch."""
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
    """The manifest key is false on every normal install, and matching the key name alone
    would mute the whole feature."""
    venv = _make_venv(tmp_path, stdout = _answer("0"))
    (venv / "unsloth_install_manifest.json").write_text(
        '{"schema": 1, "no_torch": false}', encoding = "utf-8"
    )
    result = _run_block(block, venv, tmp_path, nvidia = True)
    assert "PyTorch cannot see the NVIDIA GPU reported above" in result["stdout"]


def test_a_missing_interpreter_is_skipped_by_name(block, tmp_path):
    """setup.sh runs before the venv exists in some repair paths. Asserted on the skip line
    rather than on the absence of a report: without the guard the probe merely fails to exec,
    which is silent too, so silence proves nothing."""
    venv = tmp_path / "novenv"
    venv.mkdir()
    result = _run_block(block, venv, tmp_path, amd = True, gfx = "gfx1201")
    assert "VSUB|torch GPU visibility check skipped: no interpreter at" in result["stdout"]
    assert "gpu check" not in result["stdout"]
    assert result["returncode"] == 0


def test_the_check_runs_after_the_gpu_summary_and_before_the_llama_step():
    """Ordering is load-bearing: _setup_gfx and _setup_amd_detected are computed by the summary,
    and the report has to sit next to the line it contradicts."""
    text = SETUP_SH.read_text(encoding = "utf-8")
    summary = text.index("# ── GPU detection summary")
    announcement = text.index('step "gpu" "AMD ROCm ($_setup_gfx)"')
    check = text.index(_BLOCK_START)
    llama = text.index(_BLOCK_END)
    assert summary < announcement < check < llama


def test_a_working_xpu_host_is_not_accused_of_running_on_cpu(block, tmp_path):
    """Hybrid Intel/NVIDIA on the XPU wheel: torch.cuda.is_available() is False and the
    machine still runs on its GPU, because _detect_hardware_locked falls through from
    CUDA to XPU (hardware.py). Reporting a CPU verdict there is a false alarm about a
    working host, which is worse than saying nothing (#8473)."""
    venv = _make_venv(tmp_path, stdout = _answer("0", xpu = "1"))
    result = _run_block(block, venv, tmp_path, nvidia = True)
    out = result["stdout"]
    assert "PyTorch cannot see" not in out
    assert "No visible GPU" not in out
    assert "BLOCK_DONE" in out
    assert result["returncode"] == 0


def test_the_verbose_line_does_not_call_zero_cuda_devices_the_total(block, tmp_path):
    """device_count() is CUDA-only. On the host the XPU answer suppresses, it is 0 while torch
    is using the Intel GPU, so "torch sees 0 GPU(s)" reads as a failure on a working machine."""
    venv = _make_venv(tmp_path, stdout = _answer("0", version = "2.9.0+xpu", xpu = "1"))
    result = _run_block(block, venv, tmp_path, nvidia = True, env = {"UNSLOTH_VERBOSE": "1"})
    out = result["stdout"]
    assert "VSUB|torch sees 0 CUDA device(s), xpu true (torch 2.9.0+xpu, hip none)" in out
    assert "GPU(s)" not in out


def test_a_cpu_only_hybrid_host_is_still_reported(block, tmp_path):
    """The suppression is XPU-specific, not a blanket mute: with no XPU the same
    invisible-GPU host must still be reported."""
    venv = _make_venv(tmp_path, stdout = _answer("0", xpu = "0"))
    result = _run_block(block, venv, tmp_path, nvidia = True)
    assert "PyTorch cannot see the NVIDIA GPU reported above" in result["stdout"]


def test_a_colab_runtime_probes_the_interpreter_its_deps_went_into(block, tmp_path):
    """Colab has no Unsloth venv: setup.sh installs the backend deps into the SYSTEM python and
    sets _COLAB_NO_VENV. A guard requiring $VENV_DIR/bin/python skipped the probe there, so an
    NVIDIA Colab runtime whose torch cannot see the GPU got no diagnostic at all."""
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
    """The GGUF-only silence keys on the venv's torch on disk, and Colab has no venv layout to
    look at while its runtimes ship torch. Falling back to silence there would hide the crash."""
    venv = _make_venv(tmp_path, exit_code = 1, torch_on_disk = False)
    result = _run_block(
        block, venv, tmp_path, nvidia = True, colab = True, venv_dir = tmp_path / "no_such_venv"
    )
    assert "could not check whether PyTorch sees this GPU" in result["stdout"]
    assert result["returncode"] == 0


def test_the_venv_interpreter_still_wins_where_there_is_a_venv(block, tmp_path):
    """Colab support must not redirect the normal path onto a system python carrying a
    different torch, and must not add a second launch to it."""
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
    """The suppression must turn on what the probe SAW, not on the wheel label. A +xpu
    wheel whose Intel runtime is broken answers False for both CUDA and XPU and falls
    through to CPU exactly like the hosts this check exists for, so a disk-only
    'is this an XPU wheel' test would silence the one case that needs saying (#8473)."""
    venv = _make_venv(tmp_path, stdout = _answer("0", version = "2.9.0+xpu", xpu = "0"))
    result = _run_block(block, venv, tmp_path, nvidia = True)
    assert "PyTorch cannot see the NVIDIA GPU reported above" in result["stdout"]
    assert "torch 2.9.0+xpu" in result["stdout"]
