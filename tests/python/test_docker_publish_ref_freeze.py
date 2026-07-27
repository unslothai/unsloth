# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The docker publish workflow must never forward an unfrozen ref.

`prepare` resolves unsloth, unsloth-zoo and notebooks to ONE commit each so the
amd64 leg, the arm64 leg and the Studio build all bake identical source; that is
the whole reason the job exists. Each resolver was

    SHA="$(git ls-remote <repo> "$REF" | awk 'NR==1{print $1}')"
    [ -n "$SHA" ] || SHA="$REF"

`git ls-remote` exits 0 whether or not a ref matched, so a non-zero exit means
the remote was never reached. That exit was lost twice over: it is the first
element of a pipeline, and a `run:` step with no explicit `shell:` runs under
`bash -e` WITHOUT pipefail, so the step exited 0 and published `ref=main`. Each
build then resolved `main` independently, and a branch advance between them
would ship one multi-arch tag containing different revisions. The stable-tag
gates key off the inputs, not off whether resolution worked, so `:latest` would
still be moved onto it.

Static plus behavioural: the resolver `run:` blocks are executed under `bash -e`
with a `git` stub. No docker, no network.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-publish.yml"

RESOLVER_STEPS = ("unsloth_ref", "zoo_ref", "notebooks")

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None, reason = "needs bash",
)


@pytest.fixture(scope = "module")
def steps() -> dict:
    assert WORKFLOW.is_file(), f"missing {WORKFLOW}"
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    found = {}
    for step in doc["jobs"]["prepare"]["steps"]:
        if step.get("id") in RESOLVER_STEPS:
            found[step["id"]] = step["run"]
    missing = set(RESOLVER_STEPS) - set(found)
    assert not missing, f"resolver steps missing from the prepare job: {missing}"
    return found


def test_the_workflow_never_pins_a_shell_so_bash_e_has_no_pipefail(steps: dict):
    # If someone later adds `shell: bash` the runner switches to
    # `bash --noprofile --norc -eo pipefail`, which would make the guards below
    # redundant rather than wrong -- but until then they are the only protection.
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    assert "shell" not in doc.get("defaults", {}).get("run", {}), (
        "this test models the default `bash -e` shell; update it if a default "
        "shell with pipefail is introduced"
    )


@pytest.mark.parametrize("step_id", RESOLVER_STEPS)
def test_an_unreachable_remote_fails_the_step(steps: dict, step_id: str, tmp_path: Path):
    script = _expand(steps[step_id])
    res = _run_with_failing_ls_remote(script, tmp_path)
    assert res.returncode != 0, (
        "a transport failure must fail the prepare job, not fall through to the "
        f"mutable ref:\nstdout={res.stdout}\nstderr={res.stderr}"
    )


@pytest.mark.parametrize("step_id", RESOLVER_STEPS)
def test_an_unreachable_remote_never_emits_a_mutable_ref(
    steps: dict, step_id: str, tmp_path: Path,
):
    script = _expand(steps[step_id])
    res = _run_with_failing_ls_remote(script, tmp_path)
    emitted = (tmp_path / "github_output").read_text(encoding = "utf-8") \
        if (tmp_path / "github_output").exists() else ""
    for line in emitted.splitlines():
        key, _, value = line.partition("=")
        assert re.fullmatch(r"[0-9a-f]{40}", value), (
            f"{step_id} published {key}={value!r}, which the three builds each "
            "resolve again, so they can bake different revisions"
        )
    assert res.returncode != 0


def _expand(run: str) -> str:
    """Replace the `${{ ... }}` expressions with the empty string the default
    (push to main, no dispatch inputs) trigger produces."""
    return re.sub(r"\$\{\{[^}]*\}\}", "", run)


def _run_with_failing_ls_remote(script: str, tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    stub = bin_dir / "git"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "ls-remote" ]; then\n'
        '  echo "fatal: unable to access: Could not resolve host" >&2\n'
        "  exit 128\n"
        "fi\n"
        "exit 0\n",
        encoding = "utf-8",
    )
    stub.chmod(0o755)
    out = tmp_path / "github_output"
    out.write_text("", encoding = "utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["GITHUB_OUTPUT"] = str(out)
    # Whatever the expansions above blanked out; the resolvers default to "main".
    for name in ("INPUT_REF", "TAG_REF", "PUSH_SHA"):
        env[name] = ""
    path = tmp_path / "step.sh"
    path.write_text(script, encoding = "utf-8")
    # Exactly how the runner invokes a `run:` step with no explicit `shell:`.
    return subprocess.run(
        ["bash", "-e", str(path)],
        capture_output = True, text = True, env = env, timeout = 60,
    )
