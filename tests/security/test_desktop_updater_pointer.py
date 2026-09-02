# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Desktop discovery must stay on a release with one complete asset set.

Clients poll the repo-wide `/releases/latest/download/latest.json`, so whichever
release holds GitHub's Latest label is the release that serves both the updater
manifest and the `Unsloth-Desktop-*` download links. Publishing anything without
Desktop bundles takes that label, so the repair moves it back.

Every test here runs the workflow's real `run:` text through bash against a
stateful fake `gh`. The fake refuses to answer an invocation it does not model,
because a shim that quietly succeeds turns a broken step into a passing test.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
UPDATER_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "publish-desktop-updater.yml"

REPOSITORY = "unslothai/unsloth"
STEP_NAME = "Restore latest complete Desktop release"
GATE_NAME = "Check for desktop bundles"
SUFFIXES = ("MacOS.dmg", "Linux.AppImage", "Ubuntu.deb", "Windows.exe")
STABLE_ASSETS = tuple(f"Unsloth-Desktop-{suffix}" for suffix in SUFFIXES)

# A `gh` that keeps state, so a PATCH is observable in the next read and an unmodelled call is a loud failure rather
# than a silent success.
# The body is Python because the state it keeps is JSON;
# it is reached through a /bin/sh shim like every other fake in this directory, because a `python3` shebang does not
# resolve on a Windows runner, where the interpreter is `python.exe`.
FAKE_GH_BODY = r"""
import json, os, pathlib, sys

state_path = pathlib.Path(os.environ["FAKE_STATE"])
state = json.loads(state_path.read_text())
argv = sys.argv[1:]
with open(os.environ["COMMAND_LOG"], "a") as handle:
    handle.write("gh " + " ".join(argv) + "\n")


def by_tag(tag):
    return next((r for r in state["releases"] if r.get("tag_name") == tag), None)


def emit(payload, argv):
    if "--jq" in argv:
        expression = argv[argv.index("--jq") + 1]
        if expression == ".tag_name":
            print(payload["tag_name"])
        elif expression == ".assets[].name":
            for asset in payload.get("assets", []):
                print(asset["name"])
        elif expression == "[.assets[].name]":
            print(json.dumps([a["name"] for a in payload.get("assets", [])]))
        else:
            sys.stderr.write("fake gh: unmodelled --jq %r\n" % expression)
            return 2
        return 0
    print(json.dumps(payload))
    return 0


if argv[:2] == ["release", "view"]:
    release = by_tag(argv[2])
    if release is None:
        sys.stderr.write("release not found\n")
        return_code = 1
    else:
        return_code = emit(release, argv)
    sys.exit(return_code)

if argv[:2] == ["release", "download"]:
    tag = argv[2]
    manifest = state.get("manifests", {}).get(tag)
    if manifest is None:
        sys.stderr.write("no latest.json on %s\n" % tag)
        sys.exit(1)
    directory = pathlib.Path(argv[argv.index("--dir") + 1])
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "latest.json").write_text(json.dumps(manifest))
    sys.exit(0)

if argv[:1] == ["api"]:
    if "--method" in argv and "PATCH" in argv:
        release_id = int(argv[argv.index("PATCH") + 1].rsplit("/", 1)[-1])
        release = next((r for r in state["releases"] if r.get("id") == release_id), None)
        if release is None:
            sys.stderr.write("HTTP 404\n")
            sys.exit(1)
        fields = dict(argv[i + 1].split("=", 1) for i, a in enumerate(argv) if a == "-f")
        if fields.get("make_latest") == "true":
            if release.get("draft") or release.get("prerelease"):
                sys.stderr.write("HTTP 422: cannot mark a draft or prerelease latest\n")
                sys.exit(1)
            state["latest"] = release["tag_name"]
        if state.get("race_latest"):
            state["latest"] = state.pop("race_latest")
        state_path.write_text(json.dumps(state))
        sys.exit(emit(release, argv))
    route = argv[1]
    if route.endswith("/releases/latest"):
        release = by_tag(state.get("latest"))
        if release is None or release.get("draft") or release.get("prerelease"):
            sys.stderr.write("HTTP 404: Not Found\n")
            sys.exit(1)
        sys.exit(emit(release, argv))
    if "releases?per_page=" in route:
        sys.exit(emit(state["releases"], argv))

sys.stderr.write("fake gh: unmodelled invocation %r\n" % (argv,))
sys.exit(2)
"""


def _step(name):
    job = yaml.safe_load(UPDATER_WORKFLOW.read_text(encoding = "utf-8"))["jobs"]["publish-updater"]
    steps = {step.get("name"): step for step in job["steps"]}
    assert name in steps, sorted(steps)
    return steps[name]


def _step_run():
    step = _step(STEP_NAME)
    # Only for a release that arrived without a complete publication of its own.
    assert "steps.gate.outputs.proceed == 'false'" in step["if"]
    # Nothing fires this workflow automatically, so the repair must be asked for.
    assert "inputs.repair_pointer" in step["if"]
    return step["run"]


def _release(
    tag,
    *,
    release_id,
    complete = True,
    draft = False,
    prerelease = False,
    published_at = "2026-01-01T00:00:00Z",
    legacy = False,
    manifest_only = False,
    drop = (),
    assets = None,
):
    if assets is not None:
        names = list(assets)
    elif manifest_only:
        names = ["latest.json"]
    elif not complete:
        names = ["notes.txt"]
    else:
        version = tag.removeprefix("v").replace(".", "_").replace("-", "_")
        base = f"Unsloth-Desktop-{version}" if legacy else "Unsloth-Desktop"
        names = ["latest.json", *(f"{base}-{suffix}" for suffix in SUFFIXES)]
        names.append(f"{base}-ARM64.app.tar.gz")
    names = [name for name in names if name not in drop]
    return {
        "id": release_id,
        "tag_name": tag,
        "draft": draft,
        "prerelease": prerelease,
        "published_at": published_at,
        "assets": [{"name": name} for name in names],
    }


def _manifest(tag, *, legacy = False):
    base = (
        f"Unsloth-Desktop-{tag.removeprefix('v').replace('.', '_').replace('-', '_')}"
        if legacy
        else "Unsloth-Desktop"
    )
    prefix = f"https://github.com/{REPOSITORY}/releases/download/{tag}/"
    return {
        "version": tag,
        "platforms": {
            "darwin-aarch64": {"url": f"{prefix}{base}-ARM64.app.tar.gz", "signature": "c2ln"},
            "linux-x86_64": {"url": f"{prefix}{base}-Linux.AppImage", "signature": "c2ln"},
            "windows-x86_64": {"url": f"{prefix}{base}-Windows.exe", "signature": "c2ln"},
        },
    }


def _world(
    tmp_path,
    *,
    releases,
    manifests,
    latest,
    race_latest = None,
):
    state = {"releases": releases, "manifests": manifests, "latest": latest}
    if race_latest:
        state["race_latest"] = race_latest
    tmp_path.mkdir(parents = True, exist_ok = True)
    path = tmp_path / "github.json"
    path.write_text(json.dumps(state), encoding = "utf-8")
    return path


def _run_step(
    tmp_path,
    script,
    *,
    release_tag,
    state_path,
    repair_pointer = "true",
):
    tmp_path.mkdir(parents = True, exist_ok = True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(exist_ok = True)
    body = fake_bin / "fake_gh.py"
    body.write_text(FAKE_GH_BODY, encoding = "utf-8")
    gh = fake_bin / "gh"
    gh.write_text(f'#!/bin/sh\nexec "$FAKE_GH_PYTHON" "{body}" "$@"\n', encoding = "utf-8")
    gh.chmod(0o755)

    log = tmp_path / "commands.log"
    log.write_text("", encoding = "utf-8")
    output = tmp_path / "github-output"
    output.write_text("", encoding = "utf-8")
    env = os.environ.copy()
    env.update(
        {
            "COMMAND_LOG": str(log),
            "FAKE_GH_PYTHON": sys.executable,
            "FAKE_STATE": str(state_path),
            "GITHUB_OUTPUT": str(output),
            "GITHUB_REPOSITORY": REPOSITORY,
            "GITHUB_STEP_SUMMARY": str(tmp_path / "step-summary.md"),
            "PATH": f"{fake_bin}:{env['PATH']}",
            "REPAIR_POINTER": repair_pointer,
            "RELEASE_TAG": release_tag,
            "RUNNER_TEMP": str(tmp_path),
        }
    )
    result = subprocess.run(
        ["bash", "-c", script],
        cwd = tmp_path,
        env = env,
        text = True,
        capture_output = True,
        check = False,
    )
    outputs = {}
    for line in output.read_text(encoding = "utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            outputs[key] = value
    return result, log.read_text(encoding = "utf-8").splitlines(), outputs


def _run(
    tmp_path,
    *,
    release_tag,
    releases,
    manifest = None,
    manifests = None,
    latest = None,
    race_latest = None,
):
    """Run the repair step. `latest` defaults to the release being repaired."""
    if manifests is None:
        manifests = {}
        if manifest is not None:
            # The manifest fixture belongs to whichever release would be chosen.
            for release in releases:
                if release["tag_name"] != release_tag:
                    manifests[release["tag_name"]] = manifest
    state_path = _world(
        tmp_path,
        releases = releases,
        manifests = manifests,
        latest = release_tag if latest is None else latest,
        race_latest = race_latest,
    )
    result, commands, _ = _run_step(
        tmp_path, _step_run(), release_tag = release_tag, state_path = state_path
    )
    return result, commands


def _latest_of(tmp_path):
    return json.loads((tmp_path / "github.json").read_text(encoding = "utf-8"))["latest"]


# ------------------------------------------------------------------ the gate


def _run_gate(tmp_path, *, release_tag, assets, repair_pointer):
    state_path = _world(
        tmp_path,
        releases = [_release(release_tag, release_id = 1, assets = assets)],
        manifests = {},
        latest = release_tag,
    )
    return _run_step(
        tmp_path,
        _step(GATE_NAME)["run"],
        release_tag = release_tag,
        state_path = state_path,
        repair_pointer = repair_pointer,
    )


LEGACY_ASSETS = tuple(f"Unsloth-Desktop-0_1_53_beta-{suffix}" for suffix in SUFFIXES)


def test_the_gate_classifies_every_shape_of_release(tmp_path):
    """The four-way classification, executed rather than grepped."""
    cases = (
        ((), "false", "false"),
        ((), "true", "false"),
        (("notes.txt",), "false", "false"),
        (("latest.json", *STABLE_ASSETS), "false", "true"),
        (("latest.json", *STABLE_ASSETS), "true", "true"),
        # Everything published before the rename is still a whole publication.
        (("latest.json", *LEGACY_ASSETS), "false", "true"),
        (("latest.json", *LEGACY_ASSETS), "true", "true"),
        # Partial publications are never validated...
        (("latest.json", *STABLE_ASSETS[:3]), "false", None),
        ((*STABLE_ASSETS,), "false", None),
        (("latest.json",), "false", None),
        (("latest.json", *STABLE_ASSETS[:2], *LEGACY_ASSETS[2:]), "false", None),
        # ...but they must never block the repair, which writes nothing here.
        (("latest.json", *STABLE_ASSETS[:3]), "true", "false"),
        ((*STABLE_ASSETS,), "true", "false"),
        (("latest.json",), "true", "false"),
    )
    for index, (assets, repair_pointer, expected) in enumerate(cases):
        result, _, outputs = _run_gate(
            tmp_path / f"case{index}",
            release_tag = "v0.1.53-beta",
            assets = assets,
            repair_pointer = repair_pointer,
        )
        label = (assets, repair_pointer)
        if expected is None:
            assert result.returncode == 1, label
            assert "refusing a partial publication" in result.stderr, label
        else:
            assert result.returncode == 0, (label, result.stderr)
            assert outputs.get("proceed") == expected, (label, outputs)


def test_the_gate_fails_closed_when_the_release_cannot_be_read(tmp_path):
    state_path = _world(tmp_path, releases = [], manifests = {}, latest = None)
    result, _, outputs = _run_step(
        tmp_path,
        _step(GATE_NAME)["run"],
        release_tag = "v0.1.53-beta",
        state_path = state_path,
        repair_pointer = "false",
    )
    assert result.returncode == 1
    assert "refusing to advance the channel" in result.stderr
    assert outputs == {}


# ---------------------------------------------------------------- the repair


def test_the_newest_complete_desktop_release_is_restored_without_copying_assets(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.52-beta", release_id = 52, published_at = "2026-03-01T00:00:00Z"),
            _release("v0.1.51-beta", release_id = 51, published_at = "2026-02-01T00:00:00Z"),
            _release(
                "v0.1.53-beta",
                release_id = 53,
                complete = False,
                published_at = "2026-04-01T00:00:00Z",
            ),
        ],
        manifests = {
            "v0.1.52-beta": _manifest("v0.1.52-beta"),
            "v0.1.51-beta": _manifest("v0.1.51-beta"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert "gh release download v0.1.52-beta --pattern latest.json" in "\n".join(commands)
    assert any(
        line.startswith(f"gh api --method PATCH repos/{REPOSITORY}/releases/52")
        and "-f make_latest=true" in line
        for line in commands
    ), commands
    # The binaries are never duplicated; only the label moves.
    assert not [line for line in commands if line.startswith("gh release upload")]
    assert _latest_of(tmp_path) == "v0.1.52-beta"

    restored = json.loads((tmp_path / "restore-latest" / "latest.json").read_text(encoding = "utf-8"))
    assert restored == _manifest("v0.1.52-beta")
    summary = (tmp_path / "step-summary.md").read_text(encoding = "utf-8")
    assert "points back to v0.1.52-beta" in summary
    assert "404" in summary


def test_legacy_downloads_are_restored_during_migration(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.802-beta",
        releases = [
            _release("v0.1.802-beta", release_id = 802, complete = False),
            _release("v0.1.801-beta", release_id = 801, legacy = True),
        ],
        manifests = {"v0.1.801-beta": _manifest("v0.1.801-beta", legacy = True)},
    )
    assert result.returncode == 0, result.stderr
    assert "gh release download v0.1.801-beta --pattern latest.json" in "\n".join(commands)
    assert any("releases/801 -f make_latest=true" in line for line in commands)
    assert _latest_of(tmp_path) == "v0.1.801-beta"
    # A pre-rename release cannot serve the stable links, and the summary says so rather than reporting the downloads as
    # repaired.
    summary = (tmp_path / "step-summary.md").read_text(encoding = "utf-8")
    assert "predates the stable asset names" in summary


def test_a_release_holding_only_a_manifest_is_not_a_restore_candidate(tmp_path):
    """`latest.json` alone is the state the old carry-forward repair left behind."""
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52, manifest_only = True),
        ],
        manifests = {"v0.1.52-beta": _manifest("v0.1.52-beta")},
    )
    assert result.returncode == 1
    assert "nothing to restore" in result.stderr
    assert not [line for line in commands if "--method PATCH" in line]


def test_incomplete_draft_and_prerelease_releases_are_never_restored(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52, draft = True),
            _release("v0.1.51-beta", release_id = 51, prerelease = True),
            _release("v0.1.50-beta", release_id = 50, complete = False),
        ],
        manifests = {"v0.1.52-beta": _manifest("v0.1.52-beta")},
    )
    # An explicit repair request that cannot be honoured leaves production broken, so it must go red rather than finish
    # green with nothing done.
    assert result.returncode == 1
    assert "nothing to restore" in result.stderr
    assert not [line for line in commands if "--method PATCH" in line]
    summary = (tmp_path / "step-summary.md").read_text(encoding = "utf-8")
    assert "still returning 404" in summary


def test_a_prebuilt_release_holding_the_pointer_is_repaired(tmp_path):
    """llama.cpp prebuilts (b8475) are normal releases in this same repository.

    They take the repo-wide pointer exactly as a bundleless v... release does, and
    with it the stable download links, so they have to be repairable too.
    """
    result, commands = _run(
        tmp_path,
        release_tag = "b8475",
        releases = [
            _release(
                "b8475",
                release_id = 8475,
                assets = ["llama-b8475-bin-ubuntu-x64.zip"],
                published_at = "2026-04-01T00:00:00Z",
            ),
            _release("v0.1.52-beta", release_id = 52, published_at = "2026-03-01T00:00:00Z"),
        ],
        manifests = {"v0.1.52-beta": _manifest("v0.1.52-beta")},
    )
    assert result.returncode == 0, result.stderr
    assert any("releases/52 -f make_latest=true" in line for line in commands)
    assert _latest_of(tmp_path) == "v0.1.52-beta"


def test_the_highest_version_wins_over_a_later_publish_time(tmp_path):
    """A republished older release must not drag the pointer onto an older build."""
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.54-beta",
        releases = [
            _release(
                "v0.1.54-beta", release_id = 54, complete = False, published_at = "2026-05-01T00:00:00Z"
            ),
            _release("v0.1.52-beta", release_id = 52, published_at = "2026-04-01T00:00:00Z"),
            _release("v0.1.53-beta", release_id = 53, published_at = "2026-03-01T00:00:00Z"),
        ],
        manifests = {
            "v0.1.52-beta": _manifest("v0.1.52-beta"),
            "v0.1.53-beta": _manifest("v0.1.53-beta"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert _latest_of(tmp_path) == "v0.1.53-beta", commands


def test_an_unsound_candidate_is_skipped_for_the_next_one(tmp_path):
    """One bad manifest must not end a repair that a good candidate could finish."""
    broken = _manifest("v0.1.53-beta")
    broken["platforms"]["linux-x86_64"]["url"] = (
        f"https://github.com/{REPOSITORY}/releases/latest/download/Unsloth-Desktop-Linux.AppImage"
    )
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.54-beta",
        releases = [
            _release("v0.1.54-beta", release_id = 54, complete = False),
            _release("v0.1.53-beta", release_id = 53),
            _release("v0.1.52-beta", release_id = 52),
        ],
        manifests = {
            "v0.1.53-beta": broken,
            "v0.1.52-beta": _manifest("v0.1.52-beta"),
        },
    )
    assert result.returncode == 0, result.stderr
    assert _latest_of(tmp_path) == "v0.1.52-beta"
    assert any("releases/52 -f make_latest=true" in line for line in commands)


def test_a_manifest_naming_a_missing_bundle_is_refused(tmp_path):
    """Prefix-matching a URL says nothing about the asset still being there."""
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52, drop = ("Unsloth-Desktop-ARM64.app.tar.gz",)),
        ],
        manifests = {"v0.1.52-beta": _manifest("v0.1.52-beta")},
    )
    assert result.returncode == 1
    assert "names a missing asset" in result.stderr
    assert not [line for line in commands if "--method PATCH" in line]


def test_a_manifest_with_an_empty_signature_is_refused(tmp_path):
    manifest = _manifest("v0.1.52-beta")
    manifest["platforms"]["windows-x86_64"]["signature"] = "   "
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52),
        ],
        manifests = {"v0.1.52-beta": manifest},
    )
    assert result.returncode == 1
    assert "has no signature" in result.stderr
    assert not [line for line in commands if "--method PATCH" in line]


def test_a_manifest_naming_another_release_is_refused(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.54-beta",
        releases = [
            _release("v0.1.54-beta", release_id = 54, complete = False),
            _release("v0.1.53-beta", release_id = 53),
        ],
        manifests = {"v0.1.53-beta": _manifest("v0.1.52-beta")},
    )
    assert result.returncode == 1
    assert "names v0.1.52-beta" in result.stderr
    assert not [line for line in commands if "--method PATCH" in line]


def test_a_manifest_with_a_moving_bundle_url_is_refused(tmp_path):
    manifest = _manifest("v0.1.52-beta")
    manifest["platforms"]["linux-x86_64"]["url"] = (
        f"https://github.com/{REPOSITORY}/releases/latest/download/Unsloth-Desktop-Linux.AppImage"
    )
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52),
        ],
        manifests = {"v0.1.52-beta": manifest},
    )
    assert result.returncode == 1
    assert "URL is not pinned to its release" in result.stderr
    assert not [line for line in commands if "--method PATCH" in line]


def test_a_manifest_without_a_usable_version_is_refused(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52),
        ],
        manifests = {"v0.1.52-beta": {"version": "latest", "platforms": {}}},
    )
    assert result.returncode == 1
    assert "declares invalid version" in result.stderr
    assert not [line for line in commands if "--method PATCH" in line]


def test_a_draft_or_prerelease_target_is_refused(tmp_path):
    # /releases/latest never resolves to a draft or prerelease, so a repair aimed at one cannot affect the endpoint. The
    # source filter already refuses them; the target is held to the same rule.
    for state in ("draft", "prerelease"):
        target = _release("v0.1.53-beta", release_id = 53, complete = False)
        target[state] = True
        result, commands = _run(
            tmp_path / state,
            release_tag = "v0.1.53-beta",
            releases = [target, _release("v0.1.52-beta", release_id = 52)],
            manifests = {"v0.1.52-beta": _manifest("v0.1.52-beta")},
            latest = "v0.1.52-beta",
        )
        assert result.returncode != 0, f"{state} target was accepted"
        assert f"is a {state}" in result.stderr, result.stderr
        assert not [line for line in commands if "--method PATCH" in line], commands


def test_a_target_missing_from_the_release_listing_is_refused(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [_release("v0.1.52-beta", release_id = 52)],
        manifests = {"v0.1.52-beta": _manifest("v0.1.52-beta")},
        latest = "v0.1.52-beta",
    )
    assert result.returncode != 0, result.stdout
    assert "is not among the 100 most recent releases" in result.stderr, result.stderr
    assert not [line for line in commands if "--method PATCH" in line], commands


def test_a_pointer_that_moved_on_is_left_alone(tmp_path):
    """Between the dispatch and the PATCH someone else fixed or replaced Latest."""
    # The listing is the only view of the target's draft state: /releases/tags/{tag} answers 404 for a draft, which has
    # no git tag until it is published.
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52),
            _release("v0.1.54-beta", release_id = 54, published_at = "2026-06-01T00:00:00Z"),
        ],
        manifests = {
            "v0.1.52-beta": _manifest("v0.1.52-beta"),
            "v0.1.54-beta": _manifest("v0.1.54-beta"),
        },
        latest = "v0.1.54-beta",
    )
    assert result.returncode == 0, result.stderr
    assert not [line for line in commands if "--method PATCH" in line], commands
    assert _latest_of(tmp_path) == "v0.1.54-beta"
    assert "nothing to repair" in result.stdout


def test_a_pointer_moved_by_someone_else_after_the_patch_fails_loudly(tmp_path):
    result, commands = _run(
        tmp_path,
        release_tag = "v0.1.53-beta",
        releases = [
            _release("v0.1.53-beta", release_id = 53, complete = False),
            _release("v0.1.52-beta", release_id = 52),
            _release(
                "v0.1.55-beta", release_id = 55, published_at = "2026-07-01T00:00:00Z", complete = False
            ),
        ],
        manifests = {"v0.1.52-beta": _manifest("v0.1.52-beta")},
        race_latest = "v0.1.55-beta",
    )
    assert result.returncode == 1
    assert "another run may have moved it" in result.stderr
    assert any("--method PATCH" in line for line in commands)
