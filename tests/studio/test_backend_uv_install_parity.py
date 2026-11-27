# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Backend CI installs moved from pip to uv without changing what gets installed.

That holds only while uv is told to resolve like pip: pick the best version across both
indexes (pip's --extra-index-url) and compile bytecode. Its cache stays in the runner's temp
dir, outside every saved cache, so the pip cache entries keep their size. Dropping any of these
settings changes the environment the tests run in, or grows a saved cache, without failing a step.
"""

import pathlib
import re
import shlex

import pytest
import yaml


REPO = pathlib.Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
STEPS = [
    ("studio-backend-ci.yml", "pytest", "Install backend test dependencies (CPU only)"),
    (
        "studio-backend-ci.yml",
        "repo-cpu-tests",
        "Install deps (shared shape with backend pytest job)",
    ),
    (
        "distributed-cpu-ci.yml",
        "distributed-cpu-tests",
        "Install deps (shared shape with backend pytest job)",
    ),
]
PARITY_ENV = {
    "UV_INDEX_STRATEGY": "unsafe-best-match",
    "UV_COMPILE_BYTECODE": "1",
    "UV_LINK_MODE": "copy",
    "UV_CACHE_DIR": "${{ runner.temp }}/uv-cache",
    # No uv.toml / user config can add an index or change resolution (pip reads none either here).
    "UV_NO_CONFIG": "1",
}
PIP_ALLOWED = (["--upgrade", "pip"], ["--target", "$RUNNER_TEMP/uv-tool", "uv"])
TORCH_INDEX = ["--index-url", "https://download.pytorch.org/whl/cpu"]
PYPI_EXTRA = ["--extra-index-url", "https://pypi.org/simple"]
_SEPARATORS = {";", "&&", "||", "|", "(", ")", "then", "do", "if", "!", "else", "fi", "done"}


def _installs(run: str):
    """(installer, args) for every `pip install` command in a run block, comments dropped."""
    out = []
    for line in run.replace("\\\n", " ").splitlines():
        lexer = shlex.shlex(line, posix = True, punctuation_chars = True)
        lexer.whitespace_split = True
        lexer.commenters = "#"
        try:
            tokens = list(lexer)
        except ValueError:
            continue
        segment = []
        for token in tokens + [";"]:
            if token not in _SEPARATORS:
                segment.append(token)
                continue
            for i in range(1, len(segment)):
                if segment[i] == "install" and segment[i - 1] in ("pip", "pip3"):
                    before = segment[: i - 1]
                    if before[-1:] in (["uv"], ["$UV"]):
                        out.append(("uv", segment[i + 1 :]))
                    else:
                        out.append(("pip", segment[i + 1 :]))
                    break
            segment = []
    return out


def _step(workflow, job, name):
    doc = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding = "utf-8"))
    steps = doc["jobs"][job]["steps"]
    found = [s for s in steps if s.get("name") == name]
    assert len(found) == 1, f"{workflow}:{job} has {len(found)} steps named {name!r}"
    return found[0], steps


@pytest.mark.parametrize("workflow,job,name", STEPS)
def test_uv_resolves_like_pip(workflow, job, name):
    step, _ = _step(workflow, job, name)
    env = {k: str(v) for k, v in (step.get("env") or {}).items()}
    for key, value in PARITY_ENV.items():
        assert env.get(key) == value, (
            f"{workflow}:{job} {name!r} sets {key}={env.get(key)!r}, expected {value!r}; "
            f"without it uv installs something other than what pip did"
        )


@pytest.mark.parametrize("workflow,job,name", STEPS)
def test_every_package_install_goes_through_uv(workflow, job, name):
    step, _ = _step(workflow, job, name)
    installs = _installs(str(step.get("run", "")))
    assert any(kind == "uv" for kind, _ in installs), f"{workflow}:{job} {name!r} no longer uses uv"
    for kind, args in installs:
        if kind == "pip":
            assert args in PIP_ALLOWED, (
                f"{workflow}:{job}: `pip install {' '.join(args)}` is a plain pip install next to "
                f"uv ones; mixing the two resolvers in one sequence is how the environment drifts"
            )
            continue
        assert "--system" in args, f"{workflow}:{job}: uv install {args} skips the job's Python"
        for override in ("--python", "-p", "--target", "--prefix"):
            assert override not in args, f"{workflow}:{job}: uv install {args} sets {override}"


@pytest.mark.parametrize("workflow,job,name", STEPS)
def test_torch_comes_from_the_cpu_index_with_pypi_as_extra(workflow, job, name):
    """Without the CPU index, uv picks the CUDA torch build and its CUDA wheels."""
    step, _ = _step(workflow, job, name)
    torch = [
        a
        for k, a in _installs(str(step.get("run", "")))
        if k == "uv" and any(re.match(r"torch(?:$|[<>=!~\[;])", x) for x in a)
    ]
    assert torch, f"{workflow}:{job} no longer installs torch through uv"
    for args in torch:
        joined = " ".join(args)
        assert (
            " ".join(TORCH_INDEX) in joined and " ".join(PYPI_EXTRA) in joined
        ), f"{workflow}:{job}: torch install {args} lost the CPU index or the PyPI extra index"


def test_the_parser_sees_through_comments_and_continuations():
    run = "# --system\nuv pip install foo\npip3 install bar\nif uv pip install --system \\\n  baz; then :; fi\n"
    assert _installs(run) == [("uv", ["foo"]), ("pip", ["bar"]), ("uv", ["--system", "baz"])]


@pytest.mark.parametrize("workflow,job,name", STEPS)
def test_the_uv_cache_is_never_saved(workflow, job, name):
    """uv's cache would grow the pip cache entries (unpacked wheels) if it sat inside them."""
    _, steps = _step(workflow, job, name)
    for s in steps:
        path = str((s.get("with") or {}).get("path", ""))
        assert (
            "uv-cache" not in path
        ), f"{workflow}:{job} saves or restores uv's cache: {s.get('name')}"


def test_the_pytest_job_keeps_uv_off_path():
    """The backend tests never had uv importable or on PATH; some skip without it."""
    step, steps = _step(*STEPS[0])
    run = str(step.get("run", ""))
    assert ("pip", ["--target", "$RUNNER_TEMP/uv-tool", "uv"]) in _installs(run)
    assert all(not (k == "uv" and "$UV" not in run) for k, _ in _installs(run))
    assert "uv pip install" not in run.replace('"$UV" pip install', "")
    assert str(step["env"]["UV"]) == "${{ runner.temp }}/uv-tool/bin/uv"
    for s in steps:
        body = str(s.get("run", ""))
        assert not ("uv-tool" in body and "GITHUB_PATH" in body), f"{s.get('name')} puts uv on PATH"
        assert "uv-tool" not in str((s.get("env") or {}).get("PATH", "")), s.get("name")
        assert "uv-tool" not in str((s.get("env") or {}).get("PYTHONPATH", "")), s.get("name")
        if s is not step:
            assert ("pip", ["uv"]) not in _installs(body), f"{s.get('name')} installs uv on PATH"
