#!/usr/bin/env python3
"""Cross-platform GPU-offload spoof driver for CI (no GPU required).

Builds an OS-appropriate `llama-server` wrapper around fake_llama_server.py and
drives install_llama_prebuilt.py --smoke-test against it, asserting the exit
code contract the setup scripts depend on:

  FAKE_LLAMA_MODE=cpu , GPU install_kind -> exit 2 (EXIT_FALLBACK)  rejected
  FAKE_LLAMA_MODE=cuda, GPU install_kind -> exit 0 (EXIT_SUCCESS)   accepted
  FAKE_LLAMA_MODE=cpu , CPU install_kind -> exit 0                  not gated

Runs on windows-latest / macos-latest / ubuntu-latest. Exits non-zero on any
mismatch so the CI job fails loudly.
"""

import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
INSTALLER = REPO / "studio" / "install_llama_prebuilt.py"
FAKE = HERE / "fake_llama_server.py"
IS_WIN = sys.platform == "win32"

GPU_KIND = {
    "win32": "windows-cuda",
    "darwin": "macos-arm64",
}.get(sys.platform, "linux-cuda")
CPU_KIND = {
    "win32": "windows-cpu",
    "darwin": "macos-cpu",
}.get(sys.platform, "linux-cpu")


def make_wrapper(workdir: Path) -> Path:
    if IS_WIN:
        wrapper = workdir / "llama-server.bat"
        wrapper.write_text(f'@"{sys.executable}" "{FAKE}" %*\r\n')
        return wrapper
    wrapper = workdir / "llama-server"
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{FAKE}" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper


def run_smoke(wrapper: Path, probe: Path, install_kind: str, mode: str) -> int:
    env = dict(os.environ, FAKE_LLAMA_MODE = mode)
    proc = subprocess.run(
        [
            sys.executable,
            str(INSTALLER),
            "--smoke-test",
            str(wrapper),
            "--probe",
            str(probe),
            "--install-kind",
            install_kind,
        ],
        env = env,
        capture_output = True,
        text = True,
    )
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    return proc.returncode


def main() -> int:
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        wrapper = make_wrapper(work)
        probe = work / "probe.gguf"
        probe.write_bytes(b"GGUF\x00fake")

        cases = [
            ("cpu", GPU_KIND, 2, "CPU-only binary tagged GPU is rejected"),
            ("offloaded_zero", GPU_KIND, 2, "offloaded 0/N tagged GPU is rejected"),
            ("cuda", GPU_KIND, 0, "GPU binary tagged GPU is accepted"),
            ("cuda_buffer", GPU_KIND, 0, "GPU buffer-format binary is accepted"),
            ("cpu", CPU_KIND, 0, "CPU binary tagged CPU is not gated"),
            ("no_signal", GPU_KIND, 0, "no-signal log is not rejected"),
        ]
        for mode, kind, expected, label in cases:
            rc = run_smoke(wrapper, probe, kind, mode)
            ok = rc == expected
            print(
                f"[{'PASS' if ok else 'FAIL'}] {label}: mode={mode} kind={kind} exit={rc} (want {expected})"
            )
            if not ok:
                failures.append(label)

    if failures:
        print(f"\n{len(failures)} spoof case(s) failed: {failures}")
        return 1
    print("\nAll GPU-offload spoof cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
