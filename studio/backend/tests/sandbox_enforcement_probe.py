# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Measure what the OS sandbox actually enforces, on whichever host runs it.

Not a pytest module on purpose. Every assertion here is about kernel behaviour
rather than about our own code, so it has to drive the real
``prepare_tool_launch`` and then look at the host afterwards. It exits non-zero
when the sandbox could not be built, so a runner that cannot isolate reports
VOID instead of a green run full of skips, which is the failure mode this
replaces.

Run: ``python studio/backend/tests/sandbox_enforcement_probe.py``

Every negative control is paired with a positive control proven on the host
FIRST. A read that fails because the file was never there proves nothing, and a
child that cannot start at all would otherwise look like a perfect sandbox.
"""

import json
import os
import subprocess
import sys
import tempfile

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import os_sandbox  # noqa: E402

VOID_NO_SANDBOX = 3
VOID_NO_POSITIVE_CONTROLS = 2

# Shaped like the real thing, because the point is the class of file a model
# authored script would reach for.
CANARIES = {
    "aws": (os.path.join(".aws", "credentials"), "AWS_CANARY_aKq93"),
    "ssh": (os.path.join(".ssh", "id_rsa"), "SSH_CANARY_bTr71"),
    "gcloud": (os.path.join(".config", "gcloud", "credentials.db"), "GCLOUD_CANARY_cYu42"),
    "netrc": (".netrc", "NETRC_CANARY_dZi85"),
}

if sys.platform == "win32":
    SYSTEM_WRITE_TARGETS = {
        "windows": "C:\\Windows\\System32\\unsloth-escape",
        "program_files": "C:\\Program Files\\unsloth-escape",
    }
else:
    SYSTEM_WRITE_TARGETS = {
        "etc": "/etc/unsloth-escape",
        "usr": "/usr/unsloth-escape",
        "root": "/unsloth-escape",
    }


PAYLOAD = r'''
import json, os, sys
results = {}

def attempt(name, fn):
    try:
        results[name] = {"ok": True, "value": fn()}
    except Exception as exc:
        results[name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

for name, path in json.loads(os.environ["CANARY_PATHS"]).items():
    attempt("read_" + name, lambda p = path: open(p).read()[:40])

for name, path in json.loads(os.environ["SYSTEM_TARGETS"]).items():
    attempt("write_" + name, lambda p = path: open(p, "w").write("x"))

attempt("write_outside", lambda: open(os.environ["OUTSIDE_TARGET"], "w").write("escaped"))

# Positive controls. If these fail the sandbox is not confining, it is breaking.
attempt("write_workdir", lambda: open("canary-write.txt", "w").write("in"))
attempt("read_workdir", lambda: open("canary-write.txt").read())
attempt("import_json", lambda: __import__("json").__name__)
attempt("spawn_child", lambda: __import__("subprocess").run(
    [sys.executable, "-I", "-S", "-c", "print(42)"],
    capture_output = True, timeout = 60).stdout.decode().strip())

if sys.platform != "win32":
    attempt("write_tmp", lambda: open("/tmp/in-sandbox.txt", "w").write("in"))
if sys.platform == "linux":
    attempt("host_pids", lambda: len([p for p in os.listdir("/proc") if p.isdigit()]))

print("ENFORCEMENT_JSON:" + json.dumps(results))
'''


def seed_canaries(home):
    """Plant the secrets and prove the HOST can read every one of them."""
    positives = {}
    for name, (relpath, token) in CANARIES.items():
        path = os.path.join(home, relpath)
        os.makedirs(os.path.dirname(path) or home, exist_ok = True)
        with open(path, "w") as handle:
            handle.write(token)
        if sys.platform != "win32":
            os.chmod(path, 0o600)
        with open(path) as handle:
            positives[name] = handle.read() == token
    return positives


def host_can_write(target):
    """Whether the host itself can write here, so a denial inside means something.

    A target the host cannot write either (an unelevated runner against
    System32) is dropped from the verdict rather than counted as confinement.
    """
    try:
        with open(target, "w") as handle:
            handle.write("host")
        os.remove(target)
        return True
    except OSError:
        return False


def main():
    home = tempfile.mkdtemp(prefix = "us-enforce-home-")
    outside = tempfile.mkdtemp(prefix = "us-enforce-outside-")
    outside_target = os.path.join(outside, "host-writable.txt")
    with open(outside_target, "w") as handle:
        handle.write("original")

    positives = seed_canaries(home)
    if not all(positives.values()):
        print(json.dumps({"verdict": "VOID", "why": "host positive controls failed",
                          "positives": positives}, indent = 2))
        return VOID_NO_POSITIVE_CONTROLS

    # Which system writes are meaningful on THIS host, decided before the run.
    meaningful = {name: path for name, path in SYSTEM_WRITE_TARGETS.items()
                  if host_can_write(path)}

    capability = os_sandbox.capability_snapshot(force = True)
    if not capability.available:
        print(json.dumps({
            "verdict": "VOID",
            "why": "no OS sandbox on this host, so confinement is unmeasurable here",
            "platform": sys.platform,
            "reason": capability.reason,
            "remediation": capability.remediation,
        }, indent = 2))
        return VOID_NO_SANDBOX

    workdir = tempfile.mkdtemp(prefix = "us-enforce-work-")
    plan = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", PAYLOAD),
        workdir = workdir,
        env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": home,
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", ""),
            "CANARY_PATHS": json.dumps(
                {n: os.path.join(home, r) for n, (r, _) in CANARIES.items()}),
            "SYSTEM_TARGETS": json.dumps(SYSTEM_WRITE_TARGETS),
            "OUTSIDE_TARGET": outside_target,
        },
        preexec_fn = None,
        requested_mode = "required",
        timeout_seconds = 180,
        execution_kind = "python",
    )

    prepared = os_sandbox.prepare_tool_launch(plan)
    popen_kwargs = {
        "cwd": prepared.workdir,
        "env": prepared.env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
    }
    if sys.platform != "win32":
        popen_kwargs["preexec_fn"] = prepared.preexec_fn
        popen_kwargs["pass_fds"] = tuple(prepared.pass_fds)
    proc = subprocess.Popen(prepared.argv, **popen_kwargs)
    out, err = proc.communicate(timeout = 300)

    inside = {}
    for line in out.decode(errors = "replace").splitlines():
        if line.startswith("ENFORCEMENT_JSON:"):
            inside = json.loads(line[len("ENFORCEMENT_JSON:"):])

    # Decided on the HOST and BEFORE cleanup: a private tmpfs makes the write
    # succeed inside while nothing ever arrives, and cleanup would erase the
    # evidence either way.
    with open(outside_target) as handle:
        outside_after = handle.read()

    prepared.cleanup()

    checks = {}
    for name in CANARIES:
        checks[f"{name} credential unreadable"] = not inside.get(
            "read_" + name, {}).get("ok", True)
    for name in meaningful:
        checks[f"write to {name} refused"] = not inside.get(
            "write_" + name, {}).get("ok", True)
    checks["host file outside the workdir unmodified"] = outside_after == "original"
    checks["workdir writable (positive)"] = inside.get("write_workdir", {}).get("ok") is True
    checks["workdir readable (positive)"] = inside.get("read_workdir", {}).get("value") == "in"
    checks["stdlib importable (positive)"] = inside.get("import_json", {}).get("ok") is True
    checks["child process spawns (positive)"] = inside.get(
        "spawn_child", {}).get("value") == "42"
    if sys.platform == "linux":
        checks["host processes hidden"] = (inside.get("host_pids", {}).get("value") or 999) < 10

    verdict = "PASS" if all(checks.values()) else "FAIL"
    print(json.dumps({
        "verdict": verdict,
        "platform": sys.platform,
        "backend": prepared.backend,
        "record": prepared.execution_record.as_dict() if prepared.execution_record else None,
        "system_targets_the_host_itself_can_write": sorted(meaningful),
        "checks": checks,
        "exit": proc.returncode,
        "inside": inside,
        "stderr": err.decode(errors = "replace")[:1000],
        "cleanup_diagnostics": prepared.cleanup_diagnostics,
    }, indent = 2))
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
