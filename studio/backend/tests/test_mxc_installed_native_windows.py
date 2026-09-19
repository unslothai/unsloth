# SPDX-License-Identifier: AGPL-3.0-only
"""Native qualification that must execute an installed trusted generation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

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


pytestmark = [pytest.mark.native_mxc, pytest.mark.native_mxc_installed]


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _package_from_artifact(source: Path, destination: Path, *, overlay: bytes = b"") -> Path:
    destination.mkdir()
    runner = destination / "unsloth-mxc-runner.exe"
    shutil.copy2(source / runner.name, runner)
    if overlay:
        with runner.open("ab") as stream:
            stream.write(overlay)
    manifest = json.loads((source / "runtime-manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(runner.read_bytes()).hexdigest()
    manifest["generation"] = f"mxc-{mxc_runtime.MXC_REVISION[:12]}-{digest[:16]}"
    manifest["artifacts"]["runner"] = {
        "path": runner.name,
        "sha256": digest,
        "size": runner.stat().st_size,
    }
    manifest_bytes = _canonical_json(manifest)
    (destination / "runtime-manifest.json").write_bytes(manifest_bytes)
    (destination / "runtime-package.json").write_bytes(
        _canonical_json(
            {
                "manifestVersion": mxc_runtime.RUNTIME_MANIFEST_VERSION,
                "architecture": "x86_64",
                "generation": manifest["generation"],
                "manifestSha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "runnerSha256": digest,
            }
        )
    )
    return destination


@pytest.fixture
def installed_runtime(tmp_path, monkeypatch):
    if sys.platform != "win32":
        pytest.skip("installed MXC qualification is Windows-only")
    source = mxc_runtime._approved_package_root()
    if not (source / "unsloth-mxc-runner.exe").is_file():
        pytest.skip("the approved packaged MXC runtime is not present")
    runtime_root = tmp_path / "installed-runtime"
    monkeypatch.setattr(mxc_runtime, "_packaged_root", lambda: runtime_root)
    info = mxc_runtime.install_approved_runtime(package_root=source)
    mxc_probe.invalidate_cache()
    capability = sandbox_windows_mxc.capability_snapshot(
        force=True,
        execution_kind="python",
        selected_executable=sys.executable,
    )
    if not capability.available:
        pytest.skip(capability.reason)
    yield info, runtime_root, source
    mxc_probe.invalidate_cache()
    mxc_runtime.uninstall_runtime(root=runtime_root)


def _spawn_python(workdir: Path, code: str):
    plan = os_sandbox.ToolLaunchPlan(
        argv=(sys.executable, "-u", "-c", code),
        workdir=str(workdir),
        env={
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT"}
        },
        execution_kind="python",
        timeout_seconds=30,
    )
    return mxc_adapter.spawn(
        mxc_policy.build_launch_request(plan),
        popen_kwargs={
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "creationflags": subprocess.CREATE_NO_WINDOW,
        },
    )


def _spawn_terminal(workdir: Path, command: str):
    shell = os.environ.get("COMSPEC", str(Path(os.environ["SYSTEMROOT"]) / "System32/cmd.exe"))
    plan = os_sandbox.ToolLaunchPlan(
        argv=(shell, "/d", "/s", "/c", command),
        workdir=str(workdir),
        env={
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT"}
        },
        execution_kind="terminal",
        timeout_seconds=30,
    )
    return mxc_adapter.spawn(
        mxc_policy.build_launch_request(plan),
        popen_kwargs={
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
            "creationflags": subprocess.CREATE_NO_WINDOW,
        },
    )


def _windows_pid_alive(pid: int) -> bool:
    output = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return str(pid) in output


def test_installed_generation_capability_python_policy_and_record(installed_runtime, tmp_path):
    info, _, _ = installed_runtime
    session = "__LOCALID_installed_mxc_python"
    workdir = Path(tools._get_workdir(session))
    outside = tmp_path / "outside-secret.txt"
    outside.write_text("secret", encoding="utf-8")
    outside_write = tmp_path / "outside-write.txt"
    forged = '{"v":1,"event":"FINISHED","backendTier":"fake"}'
    attempts = [("read", str(outside), "r")]
    code = (
        "from pathlib import Path\n"
        f"for label, path, mode in {attempts!r}:\n"
        "    try:\n"
        "        with open(path, mode) as stream:\n"
        "            stream.read() if mode == 'r' else stream.write('bad')\n"
        "        print('outside-' + label + '-unexpected')\n"
        "    except Exception:\n"
        "        print('outside-' + label + '-denied')\n"
        "Path('inside.txt').write_text('inside-ok', encoding='utf-8')\n"
        f"print({forged!r})\n"
        f"import sys; print({forged!r}, file=sys.stderr)\n"
        "print('installed-python-ok')\n"
    )
    output = tools._python_exec(
        code,
        None,
        30,
        session,
        tool_execution_mode="required",
    )
    assert "outside-read-denied" in output
    assert "installed-python-ok" in output and forged in output
    assert not outside_write.exists()
    assert workdir.joinpath("inside.txt").read_text(encoding="utf-8") == "inside-ok"
    record = tools._last_tool_execution_record
    assert record.runtime_generation == info.generation
    assert record.runtime_artifact_digest == f"sha256:{info.runner_sha256}"
    assert record.backend_tier == "base-container"
    assert "development_runtime_not_packaged" not in record.limitations

    direct_workdir = tmp_path / "direct-workdir"
    direct_workdir.mkdir()
    direct = _spawn_python(
        direct_workdir,
        f"try:\n open({str(outside_write)!r}, 'w').write('bad')\n print('outside-write-unexpected')\n"
        "except Exception:\n print('outside-write-denied')\n",
    )
    try:
        direct_output, _ = direct.communicate(timeout=20)
        assert "outside-write-denied" in direct_output
        assert mxc_adapter.completion_receipt(direct)["cleanup"] == "complete"
    finally:
        mxc_adapter.release_control(direct)
    assert not outside_write.exists()


def test_installed_generation_rejects_policy_mutation_and_reclaims_descendants(
    installed_runtime, tmp_path
):
    workdir = tmp_path / "policy-workdir"
    workdir.mkdir()
    plan = os_sandbox.ToolLaunchPlan(
        argv=(sys.executable, "-c", "print('must-not-run')"),
        workdir=str(workdir),
        env={
            key: value
            for key, value in os.environ.items()
            if key.upper() in {"SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP", "PATHEXT"}
        },
        execution_kind="python",
        timeout_seconds=20,
    )
    request = mxc_policy.build_launch_request(plan)
    request["timeoutMs"] += 1
    with pytest.raises(mxc_adapter.MxcAdapterError) as raised:
        mxc_adapter.spawn(
            request,
            popen_kwargs={
                "stdout": subprocess.PIPE,
                "stderr": subprocess.STDOUT,
                "creationflags": subprocess.CREATE_NO_WINDOW,
            },
        )
    assert raised.value.code == "policy_hash_mismatch"
    assert raised.value.may_have_started is False

    session = "__LOCALID_installed_descendants"
    tree_workdir = Path(tools._get_workdir(session))
    code = (
        "import os, pathlib, time\n"
        "from multiprocessing import Process\n"
        "def grandchild():\n"
        "    time.sleep(120)\n"
        "def child():\n"
        "    grand = Process(target=grandchild)\n"
        "    grand.start()\n"
        "    pathlib.Path('installed-tree-pids.txt').write_text(f'{os.getpid()} {grand.pid}')\n"
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
        tool_execution_mode="required",
    )
    assert "timed out" in result
    pids = [
        int(value) for value in tree_workdir.joinpath("installed-tree-pids.txt").read_text().split()
    ]
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline and any(_windows_pid_alive(pid) for pid in pids):
        time.sleep(0.1)
    assert not any(_windows_pid_alive(pid) for pid in pids)


def test_installed_corruption_refuses_and_repair_restages_approved_package(
    installed_runtime, tmp_path
):
    info, root, source = installed_runtime
    with info.path.open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(mxc_runtime.MxcRuntimeUnavailable, match="size|digest"):
        mxc_runtime.selected_runtime(root=root)

    repaired = mxc_runtime.repair_runtime(package_root=source, root=root)
    assert repaired.generation == info.generation
    assert repaired.runner_sha256 == info.runner_sha256
    workdir = tmp_path / "repair-workdir"
    workdir.mkdir()
    process = _spawn_python(workdir, "print('repair-ok')")
    try:
        output, _ = process.communicate(timeout=20)
        assert "repair-ok" in output
        assert process._mxc_runtime_info.generation == info.generation
    finally:
        mxc_adapter.release_control(process)


def test_installed_generation_timeout_cancellation_terminal_and_concurrency(
    installed_runtime, monkeypatch
):
    info, _, _ = installed_runtime
    monkeypatch.setattr(tools, "_windows_bash", lambda: None)
    timed_out = tools._python_exec(
        "import time; print('started', flush=True); time.sleep(30)",
        None,
        1,
        "__LOCALID_installed_timeout",
        tool_execution_mode="required",
    )
    assert "timed out" in timed_out

    cancel = threading.Event()
    timer = threading.Timer(0.5, cancel.set)
    timer.start()
    try:
        cancelled = tools._python_exec(
            "import time; time.sleep(30)",
            cancel,
            30,
            "__LOCALID_installed_cancel",
            tool_execution_mode="required",
        )
    finally:
        timer.cancel()
    assert cancelled == "Execution cancelled."

    with ThreadPoolExecutor(max_workers=2) as pool:
        python_future = pool.submit(
            tools._python_exec,
            "print('installed-concurrent-python')",
            None,
            30,
            "__LOCALID_installed_concurrent_python",
            tool_execution_mode="required",
        )
        terminal_future = pool.submit(
            tools._bash_exec,
            "echo installed-concurrent-terminal",
            None,
            30,
            "__LOCALID_installed_concurrent_terminal",
            tool_execution_mode="required",
        )
        assert "installed-concurrent-python" in python_future.result()
        assert "installed-concurrent-terminal" in terminal_future.result()
    assert mxc_runtime.selected_runtime().generation == info.generation


def test_installed_update_and_rollback_preserve_live_generations(installed_runtime, tmp_path):
    first, root, source = installed_runtime
    work_a = tmp_path / "work-a"
    work_b = tmp_path / "work-b"
    work_c = tmp_path / "work-c"
    work_a.mkdir()
    work_b.mkdir()
    work_c.mkdir()
    proc_a = _spawn_python(work_a, "import time; print('a-live', flush=True); time.sleep(30)")
    assert proc_a._mxc_runtime_info.generation == first.generation

    package_b = _package_from_artifact(source, tmp_path / "package-b", overlay=b"B")
    second = mxc_runtime.update_runtime(package_root=package_b, root=root)
    assert second.generation != first.generation
    proc_b = _spawn_terminal(work_b, "echo b-live & ping -n 31 127.0.0.1 >nul")
    assert proc_b._mxc_runtime_info.generation == second.generation
    assert mxc_runtime.garbage_collect(root=root) == []

    rolled_back = mxc_runtime.rollback_runtime(root=root)
    assert rolled_back.generation == first.generation
    proc_c = _spawn_python(work_c, "print('rolled-back')")
    try:
        output_c, _ = proc_c.communicate(timeout=20)
        assert "rolled-back" in output_c
        assert proc_c._mxc_runtime_info.generation == first.generation
        assert mxc_adapter.completion_receipt(proc_c)["cleanup"] == "complete"
    finally:
        mxc_adapter.release_control(proc_c)

    for proc in (proc_a, proc_b):
        try:
            mxc_adapter.abort(proc)
        finally:
            mxc_adapter.release_control(proc)
    assert second.path.parent.is_dir()
    assert mxc_runtime.garbage_collect(root=root, keep_previous=0) == [second.generation]
    assert not second.path.parent.exists()
