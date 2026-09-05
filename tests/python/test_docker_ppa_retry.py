# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Launchpad answers `add-apt-repository ppa:deadsnakes/ppa` with a 504 now and then,
and every push to main builds the image, so a short outage turned into a red publish
twice in one day. Both apt stages retry the call; the loop is cut out of the
Dockerfile and run with a stubbed add-apt-repository.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"


def _retry_loops() -> list[str]:
    text = DOCKERFILE.read_text(encoding = "utf-8")
    joined = re.sub(r"\\\n\s*", " ", text)  # backslash continuations into one line
    loops = re.findall(r"for i in [^;]*; do add-apt-repository -y ppa:deadsnakes/ppa .*?; done", joined)
    return loops


def test_both_apt_stages_retry_the_ppa_add():
    text = DOCKERFILE.read_text(encoding = "utf-8")
    assert text.count("add-apt-repository -y ppa:deadsnakes/ppa") == 2
    assert len(_retry_loops()) == 2, "an add-apt-repository call lost its retry loop"


def _run_loop(loop: str, tmp_path: Path, *, failures: int) -> tuple[subprocess.CompletedProcess, int]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    counter = tmp_path / "calls"
    counter.write_text("0", encoding = "utf-8")
    (bin_dir / "add-apt-repository").write_text(
        "#!/usr/bin/env bash\n"
        f"n=$(cat {counter}); n=$((n + 1)); echo $n > {counter}\n"
        f"[ $n -gt {failures} ] || {{ echo '504 Gateway Time-out' >&2; exit 1; }}\n",
        encoding = "utf-8",
    )
    (bin_dir / "add-apt-repository").chmod(0o755)
    (bin_dir / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n", encoding = "utf-8")
    (bin_dir / "sleep").chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    res = subprocess.run(["sh", "-c", loop], capture_output = True, text = True, env = env, timeout = 60)
    return res, int(counter.read_text(encoding = "utf-8"))


@pytest.mark.parametrize("loop", _retry_loops(), ids = ["builder", "runtime"])
def test_two_outages_are_absorbed(loop: str, tmp_path: Path):
    res, calls = _run_loop(loop, tmp_path, failures = 2)
    assert res.returncode == 0, res.stdout + res.stderr
    assert calls == 3
    assert res.stdout.count("retrying") == 2


@pytest.mark.parametrize("loop", _retry_loops(), ids = ["builder", "runtime"])
def test_a_lasting_outage_still_fails_the_build(loop: str, tmp_path: Path):
    res, calls = _run_loop(loop, tmp_path, failures = 99)
    assert res.returncode != 0
    assert calls == 5
