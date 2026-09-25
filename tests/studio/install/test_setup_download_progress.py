# SPDX-License-Identifier: AGPL-3.0-only

import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[3]
PROGRESS = "Downloading runtime.tar.gz: 25.0% (1.0 MiB/4.0 MiB) at 1.0 MiB/s"
UNKNOWN_PROGRESS = "Downloading runtime.tar.gz: 25.0 MiB downloaded at 1.0 MiB/s"
DIAGNOSTICS = ["resolving release", "Downloading runtime.tar.gz: retrying", "validating runtime"]
pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason = "Exercises POSIX setup.sh on macOS and Linux",
)


def run_download(tmp_path, component, env, exit_code):
    source = (ROOT / "studio/setup.sh").read_text()
    start = source.index(f'    _{component}_LOG="$(mktemp)"')
    end = source.index("    set -e", start) + len("    set -e")
    verbose = re.search(r"_is_verbose\(\) \{.*?\n\}", source, re.S).group()
    output_filter = re.search(r"_filter_download_output\(\) \{.*?\n\}", source, re.S).group()
    child = tmp_path / "install_node_prebuilt.py"
    child.write_text(
        "import os, sys, time\n"
        "from pathlib import Path\n"
        f"print({DIAGNOSTICS[0]!r}, file=sys.stderr, flush=True)\n"
        f"print({PROGRESS!r}, flush=True)\n"
        f"print({DIAGNOSTICS[1]!r}, flush=True)\n"
        f"print({UNKNOWN_PROGRESS!r}, file=sys.stderr, flush=True)\n"
        "while not Path(os.environ['RELEASE_FILE']).exists(): time.sleep(0.01)\n"
        f"print({DIAGNOSTICS[2]!r}, flush=True)\n"
        f"raise SystemExit({exit_code})\n"
    )
    script = tmp_path / "run.sh"
    script.write_text(
        verbose + "\n" + output_filter + '\n_NODE_PY="$1"\nSCRIPT_DIR="$2"\nNODE_DIR="$2"\n'
        '_PREBUILT_CMD=("$1" "$2/install_node_prebuilt.py")\n'
        '_WHISPER_CMD=("${_PREBUILT_CMD[@]}")\n'
        + source[start:end]
        + f'\ncp "$_{component}_LOG" "$2/captured.log"\n'
        + f'rm -f "$_{component}_LOG"\nexit "$_{component}_STATUS"\n'
    )
    release = tmp_path / "release"
    child_env = dict(os.environ, RELEASE_FILE = str(release), TMPDIR = str(tmp_path))
    child_env.pop("UNSLOTH_VERBOSE", None)
    child_env.pop("UNSLOTH_TAURI_UPDATE", None)
    child_env.update(env)
    process = subprocess.Popen(
        ["bash", str(script), sys.executable, str(tmp_path)],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        text = True,
        env = child_env,
    )
    return process, release


@pytest.mark.parametrize("component", ["NODE", "PREBUILT", "WHISPER"])
@pytest.mark.parametrize(
    "env",
    [{"UNSLOTH_TAURI_UPDATE": "1"}, {"UNSLOTH_TAURI_UPDATE": "true"}, {"UNSLOTH_VERBOSE": "1"}],
)
@pytest.mark.parametrize("exit_code", [0, 7])
def test_download_progress_arrives_before_the_installer_exits(tmp_path, component, env, exit_code):
    process, release = run_download(tmp_path, component, env, exit_code)
    lines = queue.Queue()

    def read_lines():
        for line in process.stdout:
            lines.put(line.strip())

    reader = threading.Thread(target = read_lines, daemon = True)
    reader.start()
    received = []
    try:
        while UNKNOWN_PROGRESS not in received:
            received.append(lines.get(timeout = 5))
        assert PROGRESS in received
        assert process.poll() is None
    finally:
        release.touch()
        process.wait(timeout = 5)
        reader.join(timeout = 5)
        process.stdout.close()
    while not lines.empty():
        received.append(lines.get_nowait())
    assert process.returncode == exit_code
    all_output = [DIAGNOSTICS[0], PROGRESS, DIAGNOSTICS[1], UNKNOWN_PROGRESS, DIAGNOSTICS[2]]
    assert received == (
        all_output if env.get("UNSLOTH_VERBOSE") == "1" else [PROGRESS, UNKNOWN_PROGRESS]
    )
    assert (tmp_path / "captured.log").read_text().splitlines() == all_output


@pytest.mark.parametrize("component", ["NODE", "PREBUILT", "WHISPER"])
def test_terminal_update_stays_quiet(tmp_path, component):
    process, release = run_download(tmp_path, component, {}, 0)
    release.touch()
    output, _ = process.communicate(timeout = 5)
    assert process.returncode == 0
    assert output == ""
    assert (tmp_path / "captured.log").read_text().splitlines() == [
        DIAGNOSTICS[0],
        PROGRESS,
        DIAGNOSTICS[1],
        UNKNOWN_PROGRESS,
        DIAGNOSTICS[2],
    ]
