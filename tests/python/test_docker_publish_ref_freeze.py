# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""The docker publish workflow must never forward an unfrozen ref.

`git ls-remote` exits 0 whether or not a ref matched, so a non-zero exit means the
remote was never reached -- and that exit is lost twice over: it heads a pipeline, and
a `run:` step with no explicit `shell:` runs under `bash -e` WITHOUT pipefail. The
step then publishes `ref=main`, each build resolves it independently, and the
stable-tag gates still move `:latest` onto the result.
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
    # `shell: bash` would switch the runner to `-eo pipefail`; until then the guards
    # below are the only protection
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


# Same hole in the llama tag step, where `bash -e` without pipefail takes the exit
# status of the trailing `sed`. `tag=latest` is NOT a pin: every consumer re-resolves
# it, so a release published mid-run puts two bundles under one manifest.


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
    for name in ("INPUT_REF", "TAG_REF", "PUSH_SHA"):
        env[name] = ""
    path = tmp_path / "step.sh"
    path.write_text(script, encoding = "utf-8")
    return subprocess.run(
        ["bash", "-e", str(path)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
    )


# git documents status 2 for "talked to the remote, no matching refs" and any other
# non-zero for "never reached it", so treating every non-zero as "tag absent" lets a
# transient DNS/TLS failure pair the unsloth tag with zoo `main`.

ZOO_TAG = "v2026.9.1"
ZOO_MAIN_SHA = "1" * 40
ZOO_TAG_SHA = "2" * 40


def _expand_tag_trigger(run: str) -> str:
    run = run.replace("${{ github.event.inputs.unsloth_zoo_ref }}", "")
    run = run.replace("${{ startsWith(github.ref, 'refs/tags/') }}", "true")
    run = run.replace("${{ github.ref_name }}", ZOO_TAG)
    return re.sub(r"\$\{\{[^}]*\}\}", "", run)


def _run_zoo_step(script: str, tmp_path: Path, *, probe_exit: int):
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
    # git's "reached the remote, no matching refs" status: the common case
    res, emitted = _run_zoo_step(steps["zoo_ref"], tmp_path, probe_exit = 2)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert emitted.strip() == f"ref={ZOO_MAIN_SHA}", emitted


def test_a_present_zoo_tag_is_mirrored(steps: dict, tmp_path: Path):
    res, emitted = _run_zoo_step(steps["zoo_ref"], tmp_path, probe_exit = 0)
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert emitted.strip() == f"ref={ZOO_TAG_SHA}", emitted


# build-studio FROMs the digest this step exports. metadata-action sorts tags by
# priority (raw=200 above sha=100), so `.tags[0]` on a main push is the MUTABLE
# `:core`, which an unserialised second run can retag before this inspection.

OTHER_RUN_DIGEST = "sha256:" + "a" * 64
THIS_RUN_DIGEST = "sha256:" + "b" * 64
IMAGE = "docker.io/unsloth/unsloth"

# the per-arch digests this run pushed, one file per digest in /tmp/digests
ARCH_DIGESTS = ("a" * 64, "c" * 64)


def _digests_dir(tmp_path: Path, digests = ARCH_DIGESTS) -> Path:
    d = tmp_path / "digests"
    d.mkdir(parents = True, exist_ok = True)
    for h in digests:
        (d / h).write_text("", encoding = "utf-8")
    return d


def _docker_stub(bin_dir: Path, body: str) -> None:
    bin_dir.mkdir(parents = True, exist_ok = True)
    stub = bin_dir / "docker"
    stub.write_text("#!/usr/bin/env bash\n" + body, encoding = "utf-8")
    stub.chmod(0o755)


def _raw_index(children = ARCH_DIGESTS) -> str:
    inner = ",".join('{\\"digest\\":\\"sha256:%s\\"}' % h for h in children)
    return '{\\"manifests\\":[' + inner + "]}"


# What a build leg really pushes with provenance and SBOM on: not an image manifest
# but an OCI index per arch, holding the image manifest and its attestation. The
# merge flattens those children into the published index, so the per-arch index
# digest (the artifact file name) never appears there. The first publish runs on main
# failed on exactly that shape against a correct manifest.
PER_ARCH_CHILDREN = {
    ARCH_DIGESTS[0]: ("a1" * 32, "a2" * 32),
    ARCH_DIGESTS[1]: ("c1" * 32, "c2" * 32),
}
FLATTENED = tuple(h for pair in PER_ARCH_CHILDREN.values() for h in pair)


def _raw_case(merged_children) -> str:
    """A `case` over the ref for `inspect --raw`: per-arch index digests answer with
    their own children, anything else is the merged index."""
    arms = "".join(
        f'    *@sha256:{h}) printf "{_raw_index(kids)}" ;;\n'
        for h, kids in PER_ARCH_CHILDREN.items()
    )
    return (
        '  --raw) case "$5" in\n'
        + arms
        + f'    *) printf "{_raw_index(merged_children)}" ;;\n'
        + "  esac ;;\n"
    )


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
    _docker_stub(
        bin_dir,
        'case "$4" in\n'
        f'  --raw) printf "{_raw_index()}" ;;\n'
        f"  *core-build-*) printf '\"{THIS_RUN_DIGEST}\"' ;;\n"
        f"  *) printf '\"{OTHER_RUN_DIGEST}\"' ;;\n"
        "esac\n",
    )
    out = tmp_path / "github_output"
    out.write_text("", encoding = "utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["GITHUB_OUTPUT"] = str(out)
    env["DOCKER_METADATA_OUTPUT_JSON"] = (
        '{"tags":["' + IMAGE + ':core","' + IMAGE + ':core-build-123"]}'
    )
    path = tmp_path / "digest_step.sh"
    path.write_text(_expand(manifest_digest_step), encoding = "utf-8")
    res = subprocess.run(
        ["bash", "-e", str(path)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
        cwd = str(_digests_dir(tmp_path)),
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert out.read_text(encoding = "utf-8").strip() == f"digest={THIS_RUN_DIGEST}", (
        "the step read the digest through the mutable :core tag, so an overlapping "
        "main run can hand build-studio another commit's base image"
    )


@pytest.mark.skipif(shutil.which("jq") is None, reason = "needs jq")
def test_the_digest_export_still_works_without_a_handle_tag(
    manifest_digest_step: str, tmp_path: Path
):
    bin_dir = tmp_path / "bin"
    _docker_stub(
        bin_dir,
        'case "$4" in\n'
        f'  --raw) printf "{_raw_index()}" ;;\n'
        f"  *) printf '\"{THIS_RUN_DIGEST}\"' ;;\n"
        "esac\n",
    )
    out = tmp_path / "github_output"
    out.write_text("", encoding = "utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["GITHUB_OUTPUT"] = str(out)
    env["DOCKER_METADATA_OUTPUT_JSON"] = '{"tags":["' + IMAGE + ':core-v2026.9.1"]}'
    path = tmp_path / "digest_step.sh"
    path.write_text(_expand(manifest_digest_step), encoding = "utf-8")
    res = subprocess.run(
        ["bash", "-e", str(path)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
        cwd = str(_digests_dir(tmp_path)),
    )
    assert res.returncode == 0, f"stdout={res.stdout}\nstderr={res.stderr}"
    assert out.read_text(encoding = "utf-8").strip() == f"digest={THIS_RUN_DIGEST}"


@pytest.mark.skipif(shutil.which("jq") is None, reason = "needs jq")
def test_the_digest_export_refuses_another_runs_manifest(manifest_digest_step: str, tmp_path: Path):
    """Even under this run's own name, a manifest without the per-arch digests this run pushed must fail rather than hand build-studio another run's base."""
    bin_dir = tmp_path / "bin"
    # the tag now resolves to a manifest built from somebody else's arches, while
    # this run's own per-arch indexes still answer with their real children
    _docker_stub(
        bin_dir,
        'case "$4" in\n'
        + _raw_case(("d" * 64, "e" * 64))
        + f"  *) printf '\"{THIS_RUN_DIGEST}\"' ;;\n"
        "esac\n",
    )
    out = tmp_path / "github_output"
    out.write_text("", encoding = "utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["GITHUB_OUTPUT"] = str(out)
    env["DOCKER_METADATA_OUTPUT_JSON"] = (
        '{"tags":["' + IMAGE + ':core","' + IMAGE + ':core-build-123"]}'
    )
    path = tmp_path / "digest_step.sh"
    path.write_text(_expand(manifest_digest_step), encoding = "utf-8")
    res = subprocess.run(
        ["bash", "-e", str(path)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
        cwd = str(_digests_dir(tmp_path)),
    )
    assert res.returncode != 0, (
        "the step accepted a manifest that does not contain this run's arches, so an "
        "overlapping ref at the same commit silently becomes the published base:\n"
        + res.stdout
        + res.stderr
    )
    assert "digest=" not in out.read_text(
        encoding = "utf-8"
    ), "a digest was exported despite the mismatch"


@pytest.mark.skipif(shutil.which("jq") is None, reason = "needs jq")
def test_the_digest_export_accepts_the_flattened_per_arch_indexes(
    manifest_digest_step: str, tmp_path: Path
):
    """The real shape: each build leg pushed an index (image + attestation), and the
    merged index carries those CHILDREN, never the per-arch index digests the
    artifact files are named after. Runs 33935929946 and 33936467156 published a
    correct :core and then failed here, which skipped the Studio build."""
    bin_dir = tmp_path / "bin"
    _docker_stub(
        bin_dir,
        'case "$4" in\n' + _raw_case(FLATTENED) + f"  *) printf '\"{THIS_RUN_DIGEST}\"' ;;\n"
        "esac\n",
    )
    out = tmp_path / "github_output"
    out.write_text("", encoding = "utf-8")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}" + env["PATH"]
    env["GITHUB_OUTPUT"] = str(out)
    env["DOCKER_METADATA_OUTPUT_JSON"] = (
        '{"tags":["' + IMAGE + ':core","' + IMAGE + ':core-build-123"]}'
    )
    path = tmp_path / "digest_step.sh"
    path.write_text(_expand(manifest_digest_step), encoding = "utf-8")
    res = subprocess.run(
        ["bash", "-e", str(path)],
        capture_output = True,
        text = True,
        env = env,
        timeout = 60,
        cwd = str(_digests_dir(tmp_path)),
    )
    assert res.returncode == 0, (
        "the step rejected a merged index that holds every child of this run's "
        "per-arch indexes, i.e. the manifest buildx actually produces:\n" + res.stdout + res.stderr
    )
    assert f"digest={THIS_RUN_DIGEST}" in out.read_text(encoding = "utf-8")
