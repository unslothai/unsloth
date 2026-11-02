#!/usr/bin/env python3
"""Runs ruff-unsloth format, then enforce_kwargs_spacing.py --residual, which skips unneeded files."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE.parent / ".pre-commit-config.yaml"
# One-off experiments only: commits made under it get reformatted by the hook.
ANY_VERSION_ENV = "UNSLOTH_RUFF_FORMAT_ANY_VERSION"
USAGE = "usage: run_ruff_format.py FILE [FILE ...]  (formats in place; no options)"
# Worker count for the Python pass after ruff. The hook is require_serial (#14174), so
# these chunks are the only parallelism; ruff itself runs once.
JOBS_ENV = "UNSLOTH_RUFF_FORMAT_JOBS"
_MAX_JOBS = 8
# Below this many files per worker an extra interpreter costs more than it saves.
_FILES_PER_JOB = 32

DISTRIBUTION = "ruff-unsloth"
MODULE = "ruff_unsloth"
# The ruff-unsloth wheels under the hook's additional_dependencies, one per platform:
# `- "ruff-unsloth @ https://.../ruff_unsloth-0.6.9.post1-py3-none-<tag>.whl#sha256=... ; <marker>"`.
# `ruff-unsloth==0.6.9.post1` is read too. The version comes out of the config rather
# than a copy here, because a second copy of the pin is a second thing to forget; a
# regex rather than yaml.safe_load because this hook installs ruff-unsloth and nothing
# else, and adding PyYAML to run a version check would be the tail wagging the dog.
_REQUIREMENT_RE = re.compile(
    r"""^\s*-\s*(["']?)(ruff[-_]unsloth\b.*?)\1\s*(?:#.*)?$""", re.MULTILINE
)
_WHEEL_VERSION_RE = re.compile(r"ruff_unsloth-([0-9][^-\s]*)-py3")
_EXACT_VERSION_RE = re.compile(r"^ruff[-_]unsloth\s*==\s*([0-9][^\s;'\"]*)")


def pinned_requirements(config_text: str) -> list[str]:
    """The ruff-unsloth requirement lines, markers included, for installing it outside pre-commit."""
    return [match.group(2).strip() for match in _REQUIREMENT_RE.finditer(config_text)]


def pinned_ruff_version(config_text: str) -> str | None:
    """The ruff-unsloth this repo's formatting was produced with, or None if unpinned."""
    found = set()
    for requirement in pinned_requirements(config_text):
        match = _WHEEL_VERSION_RE.search(requirement) or _EXACT_VERSION_RE.match(requirement)
        if match:
            found.add(match.group(1))
    # Two different pins is not a question this script can answer, and guessing
    # would enforce the wrong one. Every platform's wheel carries the same version.
    return found.pop() if len(found) == 1 else None


def installed_ruff_version(python: str = sys.executable) -> str | None:
    """The ruff-unsloth the format below would actually run, or None when it cannot say.

    Read from the distribution, not `--version`: the binary reports the ruff it is
    built on (0.6.9), which does not change between ruff-unsloth releases.
    """
    probe = f"from importlib.metadata import version; print(version({DISTRIBUTION!r}))"
    try:
        out = subprocess.run([python, "-c", probe], capture_output = True, text = True, timeout = 60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def ruff_unavailable_reason(python: str = sys.executable) -> str | None:
    """Why `python -m ruff_unsloth` cannot run here, or None when it can.

    Separate from the version question because the answers differ: an unreadable
    version is survivable, a formatter that does not run at all is not.
    """
    try:
        out = subprocess.run(
            [python, "-m", MODULE, "--version"], capture_output = True, text = True, timeout = 60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"
    if out.returncode != 0:
        message = (out.stderr or out.stdout).strip()
        return message or f"`{MODULE} --version` exited {out.returncode}"
    return None


def version_mismatch(pinned: str | None, installed: str | None) -> bool:
    """Whether running this ruff would produce formatting the hook then undoes.

    An unreadable pin or an unreadable version string is not a mismatch:
    refusing on a question we could not ask would break the hook wherever the
    config moves. A ruff that cannot run at all is caught before this, by
    ruff_unavailable_reason.
    """
    return bool(pinned and installed and pinned != installed)


def parse_files(argv: list[str]) -> tuple[list[str], str | None]:
    """The paths to format, or an empty list plus a message saying why not.

    Every argument is a path to rewrite. Silently dropping the rest was worse
    than it sounds: `--check FILE` dropped the flag, kept the file, and wrote
    to it, and a typo'd path formatted nothing while exiting 0, which quietly
    passes any "the formatter is a fixed point" check.
    """
    if not argv:
        return [], f"no files given.\n{USAGE}"

    options = [arg for arg in argv if arg.startswith("-")]
    if options:
        message = f"unsupported option{'s' if len(options) > 1 else ''}: {' '.join(options)}"
        if any(opt in ("--check", "--diff") for opt in options):
            message += (
                "\n  There is no check mode: this script always rewrites the files"
                " it is given, and `ruff format --check` is not an equivalent."
                "\n  It checks only the ruff step, so a clean ruff says nothing"
                " about the string-merge and redundant-pass step after it."
                "\n  To preview a run, copy the file aside, run this script on the"
                " copy, and diff the two."
            )
        return [], f"{message}\n{USAGE}"

    missing = [arg for arg in argv if not Path(arg).exists()]
    if missing:
        return [], f"no such file{'s' if len(missing) > 1 else ''}: {' '.join(missing)}\n{USAGE}"

    return list(argv), None


def job_count(n_files: int, environ: dict[str, str] | None = None) -> int:
    """Workers for the Python passes: bounded by CPUs, _MAX_JOBS and the file count."""
    env = os.environ if environ is None else environ
    override = env.get(JOBS_ENV, "").strip()
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass
    return max(1, min(usable_cpus(), _MAX_JOBS, -(-n_files // _FILES_PER_JOB)))


def usable_cpus() -> int:
    """CPUs this process may run on, which on a shared runner is fewer than the host has."""
    if hasattr(os, "process_cpu_count"):  # 3.13+
        return os.process_cpu_count() or 1
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def chunk_files(files: list[str], jobs: int) -> list[list[str]]:
    """Split ``files`` into at most ``jobs`` chunks of similar total size.

    Every name that reaches one file (repeated, respelled, symlinked) lands in the
    same chunk, in the caller's order, so no two workers ever rewrite one file and
    each worker sees exactly what the single serial pass saw. Largest first into
    the lightest chunk keeps a 50k-line file from landing beside a pile of others;
    each chunk keeps the caller's order, and chunks are ordered by their first file.
    """
    groups: dict[object, list[int]] = {}
    for index, name in enumerate(files):
        try:
            # Device and inode: the one identity case-insensitive filesystems and
            # hard links cannot fool. realpath is the fallback for a missing file.
            info = os.stat(name)
            key: object = (info.st_dev, info.st_ino)
        except OSError:
            key = os.path.normcase(os.path.realpath(name))
        groups.setdefault(key, []).append(index)

    def size(indices: list[int]) -> int:
        try:
            return os.path.getsize(files[indices[0]])
        except OSError:
            return 0

    bins: list[list[int]] = [[] for _ in range(max(1, min(jobs, len(groups))))]
    loads = [0] * len(bins)
    for indices in sorted(groups.values(), key = lambda group: (-size(group), group[0])):
        lightest = loads.index(min(loads))
        bins[lightest].extend(indices)
        loads[lightest] += size(indices)
    chunks = [sorted(chunk) for chunk in bins if chunk]
    chunks.sort(key = lambda chunk: chunk[0])
    return [[files[index] for index in chunk] for chunk in chunks]


def run_chunks(head: list[str], chunks: list[list[str]]) -> int:
    """Run ``head + chunk`` for every chunk side by side; the first failure's code wins.

    One chunk runs as before, output straight through. Several write to temporary
    files that are replayed in chunk order, so the log reads the same from run to
    run. Anything that stops this process early (Ctrl+C, a timeout raising in this
    thread) kills the chunks still running instead of waiting for them.
    """
    if not chunks:
        return 0
    if len(chunks) == 1:
        return subprocess.run([*head, *chunks[0]]).returncode
    running: list[tuple[subprocess.Popen, object, object]] = []
    try:
        for chunk in chunks:
            out, err = tempfile.TemporaryFile(), tempfile.TemporaryFile()
            running.append((subprocess.Popen([*head, *chunk], stdout = out, stderr = err), out, err))
        codes = [proc.wait() for proc, _, _ in running]
    finally:
        for proc, _, _ in running:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    sys.stdout.flush()
    sys.stderr.flush()
    for _, out, err in running:
        for sink, stream in ((out, sys.stdout), (err, sys.stderr)):
            sink.seek(0)
            data = sink.read()
            sink.close()
            # Bytes where the stream has them: a path the locale codec cannot decode
            # would otherwise crash the replay after every file was formatted.
            if hasattr(stream, "buffer"):
                stream.buffer.write(data)
                stream.buffer.flush()
            else:
                stream.write(data.decode("utf-8", "replace"))
                stream.flush()
    return next((code for code in codes if code != 0), 0)


def main(argv: list[str]) -> int:
    files, error = parse_files(argv)
    if error is not None:
        print(f"run_ruff_format: {error}", file = sys.stderr)
        return 2

    pinned = pinned_ruff_version(CONFIG.read_text(encoding = "utf-8")) if CONFIG.exists() else None
    install_hint = (
        f"pip install the {DISTRIBUTION} wheel for this platform from the hook's "
        f"additional_dependencies in {CONFIG.name} "
        f"({DISTRIBUTION} {pinned or '<the pinned version>'}), or run the hook itself "
        f"(pre-commit run ruff-format-with-kwargs --files ...)"
    )

    # Both checks are made before anything is rewritten. The override below is
    # deliberately not honoured here: no ruff-unsloth formats nothing.
    unavailable = ruff_unavailable_reason()
    if unavailable is not None:
        print(
            f"run_ruff_format: cannot run `python -m {MODULE}` ({unavailable}).\n"
            f"  Refusing before rewriting anything.\n"
            f"  Fix: {install_hint}",
            file = sys.stderr,
        )
        return 1

    # ruff's own formatting is not stable across releases -- 0.9 changed which
    # half of an `assert cond, "msg"` gets wrapped -- and a ruff-unsloth release can
    # change one of the repo's rules, so running another version silently produces
    # a style the pinned hook reformats back, and the commit fails pre-commit on
    # files that are otherwise correct.
    installed = installed_ruff_version()
    if version_mismatch(pinned, installed) and not os.environ.get(ANY_VERSION_ENV):
        print(
            f"run_ruff_format: this would run {DISTRIBUTION} {installed}, but the repo is "
            f"formatted with {DISTRIBUTION} {pinned} ({CONFIG.name}).\n"
            f"  Their output differs, so the hook would undo this run and pre-commit would "
            f"fail on files you did not break.\n"
            f"  Fix: {install_hint}.\n"
            f"  Override with {ANY_VERSION_ENV}=1 if you really mean it.",
            file = sys.stderr,
        )
        return 1

    # One ruff over every file: it parallelises itself, and one per chunk would
    # stack its thread pools the way pre-commit's partitions did (#14174).
    ruff_proc = subprocess.run([sys.executable, "-m", MODULE, "format", *files])
    if ruff_proc.returncode != 0:
        return ruff_proc.returncode

    # What ruff-unsloth does not do: merge adjacent same-line string literals and drop
    # redundant `pass`. --residual sends only files that could need either to the AST.
    spacing_script = HERE / "enforce_kwargs_spacing.py"
    chunks = chunk_files(files, job_count(len(files)))
    return run_chunks([sys.executable, str(spacing_script), "--residual"], chunks)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
