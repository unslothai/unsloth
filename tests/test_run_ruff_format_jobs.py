# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""The Python passes run in bounded chunks around one ruff; chunking must lose, repeat or race nothing."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import run_ruff_format  # noqa: E402
from run_ruff_format import JOBS_ENV, chunk_files, job_count, run_chunks  # noqa: E402


def _files(tmp_path: Path, sizes: list[int]) -> list[str]:
    names = []
    for index, size in enumerate(sizes):
        path = tmp_path / f"f{index}.py"
        path.write_text("x" * size, encoding = "utf-8")
        names.append(str(path))
    return names


class TestJobCount:
    def test_few_files_stay_in_one_process(self):
        assert job_count(1, {}) == 1
        assert job_count(32, {}) == 1

    def test_bounded_by_the_cap_however_many_cpus(self, monkeypatch):
        monkeypatch.setattr(run_ruff_format, "usable_cpus", lambda: 192)
        assert job_count(100_000, {}) == run_ruff_format._MAX_JOBS

    def test_bounded_by_cpus(self, monkeypatch):
        monkeypatch.setattr(run_ruff_format, "usable_cpus", lambda: 2)
        assert job_count(100_000, {}) == 2

    def test_the_override_wins_and_a_bad_one_is_ignored(self):
        assert job_count(100_000, {JOBS_ENV: "1"}) == 1
        assert job_count(5, {JOBS_ENV: "3"}) == 3
        assert job_count(5, {JOBS_ENV: "0"}) == 1
        assert job_count(5, {JOBS_ENV: "many"}) == 1


class TestChunkFiles:
    def test_every_file_once_in_caller_order_within_a_chunk(self, tmp_path):
        names = _files(tmp_path, [5, 1, 9, 3, 7, 2, 8])
        chunks = chunk_files(names, 3)
        flat = [name for chunk in chunks for name in chunk]
        assert sorted(flat) == sorted(names)
        for chunk in chunks:
            assert chunk == sorted(chunk, key = names.index)

    def test_every_name_for_one_file_shares_a_chunk_in_order(self, tmp_path):
        names = _files(tmp_path, [3, 4])
        # A string, not a Path: pathlib would fold the "." away and make this the same name.
        alias = os.path.join(os.path.dirname(names[0]), ".", os.path.basename(names[0]))
        assert alias != names[0]
        argv = [names[0], names[1], alias, names[0]]
        chunks = chunk_files(argv, 4)
        # Kept, not dropped: the serial pass rewrote the file once per name.
        assert sorted(name for chunk in chunks for name in chunk) == sorted(argv)
        assert [names[0], alias, names[0]] in chunks

    @pytest.mark.skipif(not hasattr(os, "symlink") or os.name == "nt", reason = "POSIX symlinks")
    def test_a_symlink_shares_its_targets_chunk(self, tmp_path):
        names = _files(tmp_path, [3, 4])
        link = tmp_path / "link.py"
        link.symlink_to(names[0])
        chunks = chunk_files([str(link), names[1], names[0]], 4)
        assert [str(link), names[0]] in chunks

    def test_the_largest_file_gets_a_chunk_to_itself(self, tmp_path):
        names = _files(tmp_path, [1000, 10, 10, 10, 10])
        chunks = chunk_files(names, 2)
        assert [names[0]] in chunks

    def test_deterministic_and_never_more_chunks_than_files(self, tmp_path):
        names = _files(tmp_path, [4, 4, 4])
        assert chunk_files(names, 8) == chunk_files(names, 8)
        assert len(chunk_files(names, 8)) == 3
        assert chunk_files(names, 1) == [names]


class TestRunChunks:
    def test_output_in_chunk_order_and_the_first_failure_wins(self, capfd):
        head = [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[1]); sys.exit(int(sys.argv[1]) % 3)",
        ]
        code = run_chunks(head, [["3"], ["4"], ["5"]])
        assert code == 1
        assert capfd.readouterr().out.split() == ["3", "4", "5"]

    def test_all_clean_is_zero(self):
        assert run_chunks([sys.executable, "-c", "pass"], [["a"], ["b"]]) == 0


_ROOT = Path(__file__).resolve().parent.parent
_PINNED = run_ruff_format.pinned_ruff_version(run_ruff_format.CONFIG.read_text(encoding = "utf-8"))


@pytest.mark.skipif(
    run_ruff_format.installed_ruff_version() != _PINNED,
    reason = f"needs the pinned ruff {_PINNED}",
)
def test_chunked_and_serial_runs_write_the_same_bytes(tmp_path):
    """The hook takes the chunked path whenever it gets more than _FILES_PER_JOB files,
    which the fixed-point guard (pinned to one job) never exercises."""
    sources = sorted((_ROOT / "tests").glob("test_*.py"))[:70]
    assert len(sources) > 2 * run_ruff_format._FILES_PER_JOB
    outputs = {}
    for jobs in ("1", "3"):
        root = tmp_path / f"jobs{jobs}"
        root.mkdir()
        shutil.copy2(_ROOT / "pyproject.toml", root / "pyproject.toml")
        files, stripped = [], []
        for source in sources:
            target = root / source.name
            # Strip the kwarg spacing the post-pass has to put back.
            target.write_text(
                re.sub(r"(\w) = (\w)", r"\1=\2", source.read_text(encoding = "utf-8")),
                encoding = "utf-8",
            )
            files.append(str(target))
            stripped.append(target.read_bytes())
        proc = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "run_ruff_format.py"), *files],
            env = {**os.environ, JOBS_ENV: jobs},
            capture_output = True,
            text = True,
            timeout = 300,
        )
        assert proc.returncode == 0, proc.stderr
        outputs[jobs] = [Path(name).read_bytes() for name in files]
        assert outputs[jobs] != stripped, "the formatter rewrote nothing, so this proves nothing"
    assert outputs["1"] == outputs["3"]


class TestRunChunksStops:
    def test_an_interrupt_kills_the_chunks_still_running(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "pids"
        head = [
            sys.executable,
            "-c",
            "import os, sys, time\n"
            f"open({str(pid_file)!r}, 'a').write(str(os.getpid()) + '\\n')\n"
            "time.sleep(120)\n",
        ]
        real_popen = subprocess.Popen
        started = []

        def popen(*args, **kwargs):
            started.append(real_popen(*args, **kwargs))
            if len(started) == 2:
                raise KeyboardInterrupt  # as if Ctrl+C landed between two launches
            return started[-1]

        monkeypatch.setattr(run_ruff_format.subprocess, "Popen", popen)
        with pytest.raises(KeyboardInterrupt):
            run_chunks(head, [["a"], ["b"], ["c"]])
        # The first is in `running` and must have been killed; the second was started
        # by Popen but never recorded, so it is reaped here to keep the test clean.
        assert started[0].poll() is not None, "a chunk outlived the interrupt"
        started[1].kill()
        started[1].wait()

    def test_no_chunks_is_a_clean_no_op(self):
        assert run_chunks([sys.executable, "-c", "pass"], []) == 0

    def test_replay_works_without_a_byte_buffer(self, monkeypatch):
        import io

        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)
        head = [sys.executable, "-c", "import sys; print(sys.argv[1])"]
        assert run_chunks(head, [["x"], ["y"]]) == 0
        assert out.getvalue().split() == ["x", "y"]
