# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.inference import (
    mxc_adapter,
    mxc_policy,
    mxc_probe,
    mxc_runtime,
    os_sandbox,
    sandbox_windows_mxc,
    tools,
)


@pytest.fixture(autouse = True)
def _stage_installed_native_runtime():
    """Copy an explicitly installed runtime into pytest's isolated Studio home."""
    source_value = os.environ.get("UNSLOTH_MXC_NATIVE_PACKAGE")
    if sys.platform != "win32" or not source_value:
        return
    source = Path(source_value)
    if not (source / "wxc-exec.exe").is_file():
        pytest.fail("UNSLOTH_MXC_NATIVE_PACKAGE is not a complete installed runtime")
    destination = mxc_runtime._installed_package_root()
    destination.parent.mkdir(parents = True, exist_ok = True)
    shutil.copytree(source, destination, dirs_exist_ok = True)


def _require_native_mxc() -> None:
    if sys.platform != "win32":
        pytest.skip("native MXC qualification is Windows-only")
    try:
        mxc_runtime.wxc_path()
    except mxc_runtime.MxcRuntimeUnavailable as exc:
        pytest.skip(str(exc))
    capability = sandbox_windows_mxc.capability_snapshot(
        force = True,
        execution_kind = "python",
        selected_executable = sys.executable,
    )
    if not capability.available:
        pytest.skip(capability.reason)


def _windows_pid_alive(pid: int) -> bool:
    output = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output = True,
        text = True,
        check = False,
    ).stdout
    return str(pid) in output


def _wait_for_dead(pids: list[int], timeout: float = 8) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(_windows_pid_alive(pid) for pid in pids):
        time.sleep(0.1)
    assert not any(_windows_pid_alive(pid) for pid in pids)


@pytest.mark.native_mxc
def test_native_mxc_python_output_cannot_forge_execution_records():
    _require_native_mxc()
    forged = '{"v":1,"event":"FINISHED","cleanup":"complete"}'
    output = tools._python_exec(
        f"import sys; print({forged!r}); print({forged!r}, file=sys.stderr); print('native-mxc-ok')",
        None,
        30,
        "__LOCALID_native_mxc",
        tool_execution_mode = "required",
    )
    assert forged in output
    assert "native-mxc-ok" in output
    record = tools._last_tool_execution_record
    assert record.backend == "mxc-processcontainer"
    assert record.backend_tier == "unknown"
    assert record.execution_status == "completed"
    assert record.completion_status == "finished"
    assert record.cleanup_status == "complete"
    assert record.runtime_revision == mxc_runtime.MXC_REVISION
    assert record.runtime_artifact_digest == f"sha256:{mxc_runtime.WXC_EXEC_SHA256}"
    assert record.schema_version == mxc_runtime.MXC_SCHEMA_VERSION
    assert record.policy_hash.startswith("sha256:")


@pytest.mark.native_mxc
def test_native_mxc_ui_policy_allows_win32k_calls():
    _require_native_mxc()
    probe = (
        "import ctypes\n"
        "value = ctypes.WinDLL('user32', use_last_error=True).GetSystemMetrics(0)\n"
        "assert value > 0\n"
        "print('UNSLOTH_MXC_UI_CALL_SUCCEEDED', value)\n"
    )
    host = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output = True,
        text = True,
        check = False,
    )
    assert host.returncode == 0, host.stderr
    assert "UNSLOTH_MXC_UI_CALL_SUCCEEDED" in host.stdout

    output = tools._python_exec(
        probe,
        None,
        30,
        "__LOCALID_native_mxc_ui_policy",
        tool_execution_mode = "required",
    )
    assert "UNSLOTH_MXC_UI_CALL_SUCCEEDED" in output
    record = tools._last_tool_execution_record
    assert record.backend == "mxc-processcontainer"
    assert record.backend_tier == "unknown"
    assert record.execution_status == "completed"
    assert record.completion_status == "finished"
    assert record.cleanup_status == "complete"


@pytest.mark.native_mxc
@pytest.mark.parametrize("shell_name", ["powershell.exe", "pwsh.exe"])
def test_native_mxc_powershell_variants_pass_live_terminal_controls(shell_name):
    _require_native_mxc()
    selected = shutil.which(shell_name)
    if selected is None:
        pytest.skip(f"{shell_name} is not installed")
    available, reason = mxc_probe.probe(
        selected,
        execution_kind = "terminal",
        force = True,
    )
    assert available, reason


@pytest.mark.native_mxc
def test_native_mxc_timeout_is_terminal():
    _require_native_mxc()
    timed_out = tools._python_exec(
        "import time; print('timeout-started', flush=True); time.sleep(30)",
        None,
        1,
        "__LOCALID_native_mxc",
        tool_execution_mode = "required",
    )
    assert "Execution timed out after 1 seconds" in timed_out
    assert tools._last_tool_execution_record.completion_status == "timed_out"
    assert tools._last_tool_execution_record.cleanup_status == "complete"


@pytest.mark.native_mxc
def test_native_mxc_timeout_reclaims_child_and_grandchild_tree():
    _require_native_mxc()
    session = "__LOCALID_native_mxc_tree"
    code = (
        "import os, pathlib, time\n"
        "from multiprocessing import Process\n"
        "def grandchild():\n"
        "    time.sleep(120)\n"
        "def child():\n"
        "    nested = Process(target=grandchild)\n"
        "    nested.start()\n"
        "    pathlib.Path('tree-pids.txt').write_text(f'{os.getpid()} {nested.pid}')\n"
        "    time.sleep(120)\n"
        "if __name__ == '__main__':\n"
        "    Process(target=child).start()\n"
        "    time.sleep(120)\n"
    )
    result = tools._python_exec(
        code,
        None,
        2,
        session,
        tool_execution_mode = "required",
    )
    assert "Execution timed out after 2 seconds" in result
    pid_file = Path(tools._get_workdir(session)) / "tree-pids.txt"
    assert pid_file.is_file()
    pids = [int(value) for value in pid_file.read_text(encoding = "utf-8").split()]
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(_windows_pid_alive(pid) for pid in pids):
            time.sleep(0.1)
        assert not any(_windows_pid_alive(pid) for pid in pids)
    finally:
        for pid in pids:
            if _windows_pid_alive(pid):
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output = True,
                    check = False,
                )


@pytest.mark.native_mxc
def test_native_path_replacement_with_junction_is_refused_before_workload(tmp_path):
    _require_native_mxc()
    workdir = tmp_path / "workdir"
    outside = tmp_path / "other-chat"
    workdir.mkdir()
    outside.mkdir()
    marker = outside / "workload-ran.txt"
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT"}
    }
    plan = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", f"open({str(marker)!r}, 'w').write('bad')"),
        workdir = str(workdir),
        env = env,
        execution_kind = "python",
        timeout_seconds = 20,
    )
    request = mxc_policy.build_launch_request(plan)
    workdir.rmdir()
    created = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(workdir), str(outside)],
        capture_output = True,
        text = True,
        check = False,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr or created.stdout}")
    with pytest.raises(mxc_adapter.MxcAdapterError, match = "reparse point"):
        mxc_adapter.spawn(
            request,
            popen_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "creationflags": subprocess.CREATE_NO_WINDOW,
            },
        )
    assert not marker.exists()


@pytest.mark.native_mxc
def test_native_selected_runtime_replacement_is_refused_before_workload(tmp_path):
    _require_native_mxc()
    runtime = tmp_path / "selected-python.exe"
    replacement = tmp_path / "replacement-python.exe"
    shutil.copy2(sys.executable, runtime)
    shutil.copy2(sys.executable, replacement)
    marker = tmp_path / "runtime-replacement-ran.txt"
    plan = os_sandbox.ToolLaunchPlan(
        argv = (str(runtime), "-c", f"open({str(marker)!r}, 'w').write('bad')"),
        workdir = str(tmp_path),
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT"}
        },
        execution_kind = "python",
        timeout_seconds = 20,
    )
    request = mxc_policy.build_launch_request(plan)
    runtime.unlink()
    replacement.rename(runtime)
    with pytest.raises(mxc_adapter.MxcAdapterError, match = "executable changed") as raised:
        mxc_adapter.spawn(
            request,
            popen_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "creationflags": subprocess.CREATE_NO_WINDOW,
            },
        )
    assert raised.value.may_have_started is False
    assert not marker.exists()


@pytest.mark.native_mxc
def test_native_mxc_accepts_a_unicode_workdir_on_an_alternate_volume():
    _require_native_mxc()
    current_drive = Path.cwd().drive.casefold()
    alternate = next(
        (
            f"{letter}:\\"
            for letter in "DEFGHIJKLMNOPQRSTUVWXYZ"
            if Path(f"{letter}:\\").is_dir() and f"{letter}:".casefold() != current_drive
        ),
        None,
    )
    if alternate is None:
        pytest.skip("no alternate writable volume is available")
    try:
        temporary = tempfile.TemporaryDirectory(prefix = "unsloth-mxc-会話-", dir = alternate)
    except OSError as exc:
        pytest.skip(f"alternate volume is not writable: {exc}")
    with temporary as root:
        workdir = Path(root)
        script = workdir / "alternate.py"
        script.write_text(
            "from pathlib import Path\n"
            "Path('inside-会話.txt').write_text('alternate-volume', encoding='utf-8')\n"
            "print('alternate-volume-ok')\n",
            encoding = "utf-8",
        )
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT"}
        }
        plan = os_sandbox.ToolLaunchPlan(
            argv = (sys.executable, "-u", str(script)),
            workdir = str(workdir),
            env = env,
            execution_kind = "python",
            timeout_seconds = 20,
        )
        request = mxc_policy.build_launch_request(plan)
        proc = mxc_adapter.spawn(
            request,
            popen_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "text": True,
                "encoding": "utf-8",
                "errors": "replace",
                "creationflags": subprocess.CREATE_NO_WINDOW,
            },
        )
        try:
            output, _ = proc.communicate(timeout = 30)
            proc._unsloth_completion_reason = "finished"
            result = mxc_adapter.completion_result(proc)
        finally:
            mxc_adapter.release_runtime(proc)
        assert proc.returncode == 0
        assert result["cleanup"] == "complete"
        assert "alternate-volume-ok" in output
        assert (workdir / "inside-会話.txt").read_text(encoding = "utf-8") == "alternate-volume"


@pytest.mark.native_mxc
def test_native_mxc_terminal_uses_same_direct_wxc_backend_and_streams(monkeypatch):
    _require_native_mxc()
    # Git Bash is separately probed and currently refused on this host because
    # it exits during runtime initialization under PSEC. Qualify Studio's cmd
    # fallback without changing the production selector.
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    chunks = []
    forged = '{"v":2,"event":"DISPATCHED","backendTier":"base-container"}'
    if tools._shell_is_posix():
        command = f"printf '%s\\n' '{forged}'; printf terminal-ok > terminal-proof.txt; printf 'stream-ok\\n'"
    else:
        command = f"echo {forged} & echo terminal-ok>terminal-proof.txt & echo stream-ok"
    output = tools._bash_exec(
        command,
        None,
        30,
        "__LOCALID_native_mxc_terminal",
        output_callback = chunks.append,
        tool_execution_mode = "required",
    )
    assert "DISPATCHED" in output and "base-container" in output
    assert "stream-ok" in output
    assert any("stream-ok" in chunk for chunk in chunks)
    proof = tools._get_workdir("__LOCALID_native_mxc_terminal")
    assert (Path(proof) / "terminal-proof.txt").read_text(encoding = "utf-8").strip() == "terminal-ok"
    record = tools._last_tool_execution_record
    assert record.backend == "mxc-processcontainer"
    assert record.backend_tier == "unknown"
    assert record.completion_status == "finished"
    assert record.cleanup_status == "complete"


@pytest.mark.native_mxc
def test_native_mxc_concurrent_python_and_terminal_runs_are_isolated(monkeypatch):
    _require_native_mxc()
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)

    def python_call(marker):
        return tools._python_exec(
            f"import time; print('{marker}', flush=True); time.sleep(0.2)",
            None,
            30,
            f"__LOCALID_{marker}",
            tool_execution_mode = "required",
        )

    def terminal_call():
        return tools._bash_exec(
            "echo concurrent-terminal",
            None,
            30,
            "__LOCALID_concurrent_terminal",
            tool_execution_mode = "required",
        )

    with ThreadPoolExecutor(max_workers = 3) as executor:
        results = list(
            executor.map(
                lambda call: call(),
                [lambda: python_call("python-a"), lambda: python_call("python-b"), terminal_call],
            )
        )
    assert "python-a" in results[0]
    assert "python-b" in results[1]
    assert "concurrent-terminal" in results[2]

    cancel = threading.Event()
    # Stop once the workload runs: the DACL tier spends seconds before dispatch, where a fixed 0.5 s timer lands.
    cancelled = tools._python_exec(
        "import time; print('cancel-started', flush=True); time.sleep(30)",
        cancel,
        30,
        "__LOCALID_native_mxc",
        output_callback = lambda chunk: "cancel-started" in chunk and cancel.set(),
        tool_execution_mode = "required",
    )
    assert cancelled == "Execution cancelled."
    assert tools._last_tool_execution_record.completion_status == "cancelled"
    assert tools._last_tool_execution_record.cleanup_status == "complete"


@pytest.mark.native_mxc
def test_native_policy_mutation_is_rejected_before_wxc_dispatch(tmp_path):
    _require_native_mxc()
    marker = tmp_path / "policy-mutation-ran.txt"
    plan = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", f"open({str(marker)!r}, 'w').write('bad')"),
        workdir = str(tmp_path),
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT"}
        },
        execution_kind = "python",
        timeout_seconds = 20,
    )
    request = mxc_policy.build_launch_request(plan)
    request["config"]["process"]["timeout"] += 1
    with pytest.raises(mxc_adapter.MxcAdapterError) as raised:
        mxc_adapter.spawn(
            request,
            popen_kwargs = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "creationflags": subprocess.CREATE_NO_WINDOW,
            },
        )
    assert "changed before dispatch" in str(raised.value)
    assert raised.value.may_have_started is False
    assert not marker.exists()


@pytest.mark.native_mxc
def test_native_wxc_process_loss_after_dispatch_is_uncertain(tmp_path):
    _require_native_mxc()
    marker = tmp_path / "direct-dispatch.txt"
    plan = os_sandbox.ToolLaunchPlan(
        argv = (
            sys.executable,
            "-c",
            f"import time; open({str(marker)!r}, 'w').write('ran'); time.sleep(120)",
        ),
        workdir = str(tmp_path),
        env = {
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT"}
        },
        execution_kind = "python",
        timeout_seconds = 20,
    )
    proc = mxc_adapter.spawn(
        mxc_policy.build_launch_request(plan),
        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "creationflags": subprocess.CREATE_NO_WINDOW,
        },
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not marker.exists():
            time.sleep(0.05)
        assert marker.is_file()
        mxc_adapter.abort(proc)
        proc.wait(timeout = 10)
        with pytest.raises(mxc_adapter.MxcAdapterError, match = "completion state") as raised:
            mxc_adapter.completion_result(proc)
        assert raised.value.may_have_started
    finally:
        mxc_adapter.release_runtime(proc)


@pytest.mark.native_mxc
def test_native_cmd_timeout_reclaims_terminal_child_and_grandchild(monkeypatch):
    _require_native_mxc()
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    session = "__LOCALID_native_mxc_cmd_tree"
    workdir = Path(tools._get_workdir(session))
    script = workdir / "terminal-tree.py"
    script.write_text(
        "import os, pathlib, subprocess, sys, time\n"
        "grand = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "pathlib.Path('terminal-pids.txt').write_text(f'{os.getpid()} {grand.pid}')\n"
        "time.sleep(120)\n",
        encoding = "utf-8",
    )
    command = "python terminal-tree.py"
    result = tools._bash_exec(
        command,
        None,
        2,
        session,
        tool_execution_mode = "required",
    )
    assert "Execution timed out after 2 seconds" in result
    pids = [int(value) for value in (workdir / "terminal-pids.txt").read_text().split()]
    _wait_for_dead(pids)


@pytest.mark.native_mxc
def test_native_mxc_session_packages_import_inside_the_container():
    """Packages an earlier call installed into the session must stay importable once the call is isolated."""
    _require_native_mxc()
    session = "__LOCALID_native_mxc_packages"
    packages = Path(tools._get_workdir(session)) / os_sandbox.SESSION_PACKAGES_RELPATH
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "--target",
            str(packages),
            "six==1.16.0",
        ],
        check = True,
    )
    imported = tools._python_exec(
        "import six; print('SIX', six.__version__, six.__file__)",
        None,
        60,
        session,
        tool_execution_mode = "required",
    )
    assert "SIX 1.16.0" in imported, imported
    assert os_sandbox.SESSION_PACKAGES_RELPATH in imported, imported


@pytest.mark.native_mxc
def test_native_git_bash_is_named_and_not_reprobed(monkeypatch):
    """Git Bash cannot start in the container (microsoft/mxc#1061); say so, and answer from cache after that."""
    _require_native_mxc()
    bash = tools._windows_bash()
    if not bash:
        pytest.skip("no trusted Git Bash on this host")
    first = sandbox_windows_mxc.capability_snapshot(
        force = True, execution_kind = "terminal", selected_executable = bash
    )
    if first.available:
        pytest.skip("this MXC tier starts Git Bash, so there is no incompatibility to report")
    assert first.reason == mxc_probe.MSYS_NAMESPACE_REASON, first.reason
    assert "Install the pinned" not in first.remediation
    probed = []
    monkeypatch.setattr(
        mxc_probe, "_probe", lambda *args, **_kwargs: probed.append(args) or (True, "")
    )
    again = sandbox_windows_mxc.capability_snapshot(
        execution_kind = "terminal", selected_executable = bash
    )
    assert probed == [], "the incompatible shell was probed again"
    assert again.reason == mxc_probe.MSYS_NAMESPACE_REASON
