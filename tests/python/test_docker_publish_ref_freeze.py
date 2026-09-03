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
    shutil.which("bash") is None,
    reason = "needs bash",
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
def test_an_unreachable_remote_never_emits_a_mutable_ref(steps: dict, step_id: str, tmp_path: Path):
    script = _expand(steps[step_id])
    res = _run_with_failing_ls_remote(script, tmp_path)
    emitted = (
        (tmp_path / "github_output").read_text(encoding = "utf-8")
        if (tmp_path / "github_output").exists()
        else ""
    )
    for line in emitted.splitlines():
        key, _, value = line.partition("=")
        assert re.fullmatch(r"[0-9a-f]{40}", value), (
            f"{step_id} published {key}={value!r}, which the three builds each "
            "resolve again, so they can bake different revisions"
        )
    assert res.returncode != 0


# --- the llama.cpp prebuilt tag ----------------------------------------------
# Same hole, same job, different resolver: the tag step is
#
#     TAG="$(curl -fsSL -o /dev/null -w '%{url_effective}' .../releases/latest \
#         | sed -n 's#.*/releases/tag/##p')"
#     echo "tag=${TAG:-latest}" >> "$GITHUB_OUTPUT"
#
# `bash -e` without pipefail takes the exit status of `sed`, so an unreachable
# github.com made the step emit `tag=latest`. That value is NOT a pin: both
# matrix legs pass it to docker/fetch_llama_prebuilt.py, whose main() re-resolves
# "latest" per build, and Dockerfile.studio re-resolves it a third time, so a
# release published mid-run can put two different llama.cpp bundles under one
# multi-arch manifest -- with `:latest` moved onto it, because the stable-tag
# gates key off the dispatch inputs, not off whether resolution worked.


@pytest.fixture(scope = "module")
def llama_step() -> str:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    for step in doc["jobs"]["prepare"]["steps"]:
        if step.get("id") == "llama":
            return step["run"]
    raise AssertionError("the llama tag resolver step is missing from the prepare job")


def test_an_unresolvable_llama_release_fails_the_step(llama_step: str, tmp_path: Path):
    res = _run_llama_step(llama_step, tmp_path, curl_exit = 6)
    assert res.returncode != 0, (
        "a failed /releases/latest lookup must fail the prepare job:\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )


def test_an_unresolvable_llama_release_never_emits_a_mutable_tag(llama_step: str, tmp_path: Path):
    res = _run_llama_step(llama_step, tmp_path, curl_exit = 6)
    emitted = (tmp_path / "github_output").read_text(encoding = "utf-8")
    assert "latest" not in emitted, (
        f"the step published {emitted.strip()!r}; every consumer resolves that "
        "mutable tag again, so the two arch legs and Studio can bake different "
        "llama.cpp versions under one manifest"
    )
    assert res.returncode != 0


def test_a_resolved_llama_release_is_forwarded_verbatim(llama_step: str, tmp_path: Path):
    # The fix must not break the normal path.
    res = _run_llama_step(llama_step, tmp_path, curl_exit = 0)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert (tmp_path / "github_output").read_text(encoding = "utf-8").strip() == (
        "tag=b10107-mix-1911198"
    )


def _run_llama_step(script: str, tmp_path: Path, *, curl_exit: int):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    stub = bin_dir / "curl"
    if curl_exit:
        # How curl reports an unreachable github.com: nothing on stdout, non-zero.
        stub.write_text(
            "#!/usr/bin/env bash\n"
            'echo "curl: (6) Could not resolve host: github.com" >&2\n'
            f"exit {curl_exit}\n",
            encoding = "utf-8",
        )
    else:
        stub.write_text(
            "#!/usr/bin/env bash\n"
            "printf '%s' "
            "'https://github.com/unslothai/llama.cpp/releases/tag/b10107-mix-1911198'\n",
            encoding = "utf-8",
        )
    stub.chmod(0o755)
    out = tmp_path / "github_output"
    out.write_text("", encoding = "utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["GITHUB_OUTPUT"] = str(out)
    env["INPUT_TAG"] = ""  # the default (push / schedule) trigger
    path = tmp_path / "llama_step.sh"
    path.write_text(_expand(script), encoding = "utf-8")
    return subprocess.run(
        ["bash", "-e", str(path)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
    )


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
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
    )


# --- the zoo tag probe --------------------------------------------------------
# Item 3924993124. Before freezing, the zoo resolver asks whether the pushed
# unsloth tag exists in unsloth-zoo at all (it usually does not: unsloth's v*
# tags are Studio releases the zoo never cuts). `--exit-code` exists precisely so
# a caller can tell the two apart -- git documents status 2 for "talked to the
# remote, no matching refs", any other non-zero status for "never reached it" --
# but the probe treated every non-zero status as "tag absent". A transient
# DNS/TLS failure in that one call therefore published the requested unsloth tag
# against zoo `main`, and the freeze step below resolves `main` without
# complaint the moment the network recovers, so nothing else catches it.

ZOO_TAG = "v2026.9.1"
ZOO_MAIN_SHA = "1" * 40
ZOO_TAG_SHA = "2" * 40


def _expand_tag_trigger(run: str) -> str:
    """The expansions a push of refs/tags/$ZOO_TAG produces (the dispatch input
    is empty, so the probe branch is the one that runs)."""
    run = run.replace("${{ github.event.inputs.unsloth_zoo_ref }}", "")
    run = run.replace("${{ startsWith(github.ref, 'refs/tags/') }}", "true")
    run = run.replace("${{ github.ref_name }}", ZOO_TAG)
    return re.sub(r"\$\{\{[^}]*\}\}", "", run)


def _run_zoo_step(script: str, tmp_path: Path, *, probe_exit: int):
    """Drive the zoo resolver on a tag push with a git stub whose PROBE exits
    `probe_exit` and whose freeze lookup always succeeds."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    stub = bin_dir / "git"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'for a in "$@"; do\n'
        '  if [ "$a" = "--exit-code" ]; then\n'
        + (
            '    echo "fatal: unable to access: Could not resolve host" >&2\n'
            if probe_exit not in (0, 2)
            else ""
        )
        + f"    exit {probe_exit}\n"
        "  fi\n"
        "done\n"
        # The freeze lookup: whatever ref it was handed resolves to a sha.
        'if [ "${!#}" = "main" ]; then\n'
        f'  printf "%s\\trefs/heads/main\\n" "{ZOO_MAIN_SHA}"\n'
        "else\n"
        f'  printf "%s\\trefs/tags/{ZOO_TAG}\\n" "{ZOO_TAG_SHA}"\n'
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
    path = tmp_path / "zoo_step.sh"
    path.write_text(_expand_tag_trigger(script), encoding = "utf-8")
    res = subprocess.run(
        ["bash", "-e", str(path)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
    )
    return res, out.read_text(encoding = "utf-8")


def test_an_unreachable_zoo_probe_fails_instead_of_taking_main(steps: dict, tmp_path: Path):
    res, emitted = _run_zoo_step(steps["zoo_ref"], tmp_path, probe_exit = 128)
    assert res.returncode != 0, (
        "a transport failure in the tag probe must fail the prepare job; taking "
        "'main' pairs the requested unsloth tag with an unrelated zoo revision:\n"
        f"stdout={res.stdout}\nstderr={res.stderr}"
    )
    assert f"ref={ZOO_MAIN_SHA}" not in emitted, emitted


def test_a_missing_zoo_tag_still_falls_back_to_main(steps: dict, tmp_path: Path):
    # git's documented "reached the remote, no matching refs" status. This is the
    # common case for unsloth's Studio tags and must keep working.
    res, emitted = _run_zoo_step(steps["zoo_ref"], tmp_path, probe_exit = 2)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert emitted.strip() == f"ref={ZOO_MAIN_SHA}", emitted


def test_a_present_zoo_tag_is_mirrored(steps: dict, tmp_path: Path):
    res, emitted = _run_zoo_step(steps["zoo_ref"], tmp_path, probe_exit = 0)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert emitted.strip() == f"ref={ZOO_TAG_SHA}", emitted


# --- the base manifest digest -------------------------------------------------
# Item 3924399626. build-studio FROMs the digest this step exports, so it must be
# THIS run's manifest. metadata-action sorts its tags by priority (raw=200 above
# sha=100), so `.tags[0]` on a main push is the MUTABLE `:core`, and main runs are
# deliberately not serialised (the per-sha concurrency group at the top of the
# workflow). A second run retagging `:core` between the manifest creation and this
# inspection made the step export that run's digest instead.

OTHER_RUN_DIGEST = "sha256:" + "a" * 64
THIS_RUN_DIGEST = "sha256:" + "b" * 64
IMAGE = "docker.io/unsloth/unsloth"


@pytest.fixture(scope = "module")
def manifest_digest_step() -> str:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    for step in doc["jobs"]["merge"]["steps"]:
        if step.get("id") == "manifest_digest":
            return step["run"]
    raise AssertionError("the manifest digest export step is missing from the merge job")


@pytest.mark.skipif(shutil.which("jq") is None, reason = "needs jq")
def test_the_exported_digest_comes_from_this_runs_tag(manifest_digest_step: str, tmp_path: Path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    # `docker buildx imagetools inspect <TAG> --format ...`: the mutable :core has
    # already been retagged by the overlapping run, the run-specific tag has not.
    stub = bin_dir / "docker"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'case "$4" in\n'
        f"  *core-sha-*) printf '\"{THIS_RUN_DIGEST}\"' ;;\n"
        f"  *) printf '\"{OTHER_RUN_DIGEST}\"' ;;\n"
        "esac\n",
        encoding = "utf-8",
    )
    stub.chmod(0o755)
    out = tmp_path / "github_output"
    out.write_text("", encoding = "utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["GITHUB_OUTPUT"] = str(out)
    # What metadata-action emits for a default-branch push, in priority order.
    env["DOCKER_METADATA_OUTPUT_JSON"] = (
        '{"tags":["' + IMAGE + ':core","' + IMAGE + ':core-sha-abc1234"]}'
    )
    path = tmp_path / "digest_step.sh"
    path.write_text(_expand(manifest_digest_step), encoding = "utf-8")
    res = subprocess.run(
        ["bash", "-e", str(path)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert out.read_text(encoding = "utf-8").strip() == f"digest={THIS_RUN_DIGEST}", (
        "the step read the digest through the mutable :core tag, so an overlapping "
        "main run can hand build-studio another commit's base image"
    )


@pytest.mark.skipif(shutil.which("jq") is None, reason = "needs jq")
def test_the_digest_export_still_works_without_a_sha_tag(manifest_digest_step: str, tmp_path: Path):
    # A tag push publishes core-<version>; the fallback must keep resolving.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents = True, exist_ok = True)
    stub = bin_dir / "docker"
    stub.write_text(f"#!/usr/bin/env bash\nprintf '\"{THIS_RUN_DIGEST}\"'\n", encoding = "utf-8")
    stub.chmod(0o755)
    out = tmp_path / "github_output"
    out.write_text("", encoding = "utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["GITHUB_OUTPUT"] = str(out)
    env["DOCKER_METADATA_OUTPUT_JSON"] = '{"tags":["' + IMAGE + ':core-v2026.9.1"]}'
    path = tmp_path / "digest_step.sh"
    path.write_text(_expand(manifest_digest_step), encoding = "utf-8")
    res = subprocess.run(
        ["bash", "-e", str(path)], capture_output = True, text = True, env = env, timeout = 60
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert out.read_text(encoding = "utf-8").strip() == f"digest={THIS_RUN_DIGEST}"
