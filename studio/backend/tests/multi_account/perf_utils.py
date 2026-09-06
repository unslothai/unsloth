# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Local Git-only baseline materialization; no network or other worker checkout is read."""

import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PROBE = REPO / "tests/studio/multi_account/perf/probe.py"


def materialize_revision(ref: str, destination: Path) -> Path:
    destination = destination.resolve()
    assert destination.is_relative_to(REPO), "Revision snapshots must stay in this clone"
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"], cwd = REPO, text = True
    ).strip()
    payload = subprocess.check_output(
        ["git", "archive", "--format=tar", revision, "studio/backend"], cwd = REPO
    )
    destination.mkdir(parents = True, exist_ok = False)
    with tarfile.open(fileobj = io.BytesIO(payload)) as archive:
        # Reject links and traversal before writing; all files are from this local Git object.
        for member in archive.getmembers():
            assert not member.issym() and not member.islnk()
            assert (destination / member.name).resolve().is_relative_to(destination)
        archive.extractall(destination, filter = "data")
    return destination / "studio/backend"


def run_probe(backend: Path, home: Path, *, mode: str, samples: int = 2000, history_samples: int = 200) -> dict:
    home = home.resolve()
    assert home.is_relative_to(REPO), "Probe data must stay in this clone"
    home.parent.mkdir(parents = True, exist_ok = True)
    runtime = home.parent / (home.name + "-runtime")
    for leaf in ("tmp", "hf", "cache", "Documents"):
        (runtime / leaf).mkdir(parents = True, exist_ok = True)
    env = dict(os.environ)
    env.update(
        PYTHONDONTWRITEBYTECODE = "1", PYTHONPATH = str(backend),
        UNSLOTH_STUDIO_HOME = str(home), HF_HOME = str(runtime / "hf"),
        HF_HUB_CACHE = str(runtime / "hf" / "hub"), HF_HUB_OFFLINE = "1",
        TMPDIR = str(runtime / "tmp"), XDG_CACHE_HOME = str(runtime / "cache"),
        UNSLOTH_STUDIO_DOCUMENTS_HOME = str(runtime / "Documents"),
        UNSLOTH_ALLOW_CPU = "1", UNSLOTH_IS_PRESENT = "1",
        UNSLOTH_STUDIO_DISABLE_DEVICE_PROBE = "1",
        UNSLOTH_DIFFUSION_ATTENTION_INSTALL = "0",
    )
    result = subprocess.run(
        [sys.executable, str(PROBE), "--backend", str(backend), "--helpers", str(Path(__file__).parent),
         "--mode", mode, "--samples", str(samples), "--history-samples", str(history_samples)],
        cwd = backend, env = env, capture_output = True, text = True, timeout = 300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.splitlines()[-1])
