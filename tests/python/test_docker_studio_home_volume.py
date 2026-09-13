# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The Studio image splits Studio's code from its data.

Studio keeps one root for both: the venv, source tree and Node next to auth/,
studio.db, outputs/ and exports/. Without a volume on it, `docker rm` lost every
account, chat and trained model; with one, the volume kept the first image's code and
every later image ran that old Studio. The image now keeps the code in
$UNSLOTH_STUDIO_APP and links it into $UNSLOTH_STUDIO_HOME, which the entrypoint
repairs at every start. Whatever is in the way is kept aside, never deleted, so a
volume can also go back to an older image.
"""

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCKER = REPO / "docker"
LINKER = DOCKER / "studio_home.sh"
STUDIO_DF = DOCKER / "Dockerfile.studio"
ENTRYPOINT = DOCKER / "entrypoint.sh"
LEGACY = ".unsloth-studio-legacy"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")


def _app(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    for name in ("unsloth_studio", "src", "node", "bin", "share", "cache", ".venv_t5_550"):
        (app / name).mkdir(parents = True)
    (app / "unsloth_studio" / "VERSION").write_text("new\n")
    (app / ".node.install.lock").write_text("")
    (app / "llama.cpp").symlink_to("/opt/unsloth/llama.cpp")
    return app


def _link(
    app: Path,
    home: Path,
    *args,
    env = None,
    path = None,
):
    full = dict(os.environ, UNSLOTH_STUDIO_APP = str(app), UNSLOTH_STUDIO_HOME = str(home))
    full.update(env or {})
    if path:
        full["PATH"] = f"{path}:{full['PATH']}"
    return subprocess.run(
        ["bash", str(LINKER), *args], env = full, capture_output = True, text = True, timeout = 60
    )


def _legacy_home(tmp_path: Path) -> Path:
    """A volume created by an image from before the split: real code dirs next to data."""
    home = tmp_path / "home"
    (home / "unsloth_studio").mkdir(parents = True)
    (home / "unsloth_studio" / "VERSION").write_text("old\n")
    (home / "src" / "studio").mkdir(parents = True)
    (home / "src" / "studio" / "uncommitted.py").write_text("mine")
    (home / ".node.install.lock").write_text("")
    (home / "auth").mkdir()
    (home / "auth" / "auth.db").write_text("users")
    (home / "studio.db").write_text("chats")
    return home


def test_an_empty_home_gets_every_code_entry_as_a_link(tmp_path):
    """A fresh bind mount or named volume starts empty."""
    app = _app(tmp_path)
    home = tmp_path / "home"
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    for entry in app.iterdir():
        target = home / entry.name
        assert target.is_symlink(), entry.name
        assert os.readlink(target) == str(entry)
    assert (home / "unsloth_studio" / "VERSION").read_text() == "new\n"
    assert not (home / LEGACY).exists()
    assert res.stderr == ""


def test_a_volume_from_an_earlier_image_runs_this_images_code(tmp_path):
    """The freeze: an earlier image's venv and source tree sat in the volume as real
    directories, so a new image kept running the old Studio."""
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert (home / "unsloth_studio").is_symlink()
    assert (home / "unsloth_studio" / "VERSION").read_text() == "new\n"
    assert (home / "src").is_symlink()
    assert "kept" in res.stderr and "aside" in res.stderr


def test_nothing_in_the_way_is_deleted_it_is_kept_aside(tmp_path):
    """What an earlier image left, or what the user put under a code entry's name, is
    a rename away, not gone: the old venv, the old source tree with an uncommitted
    file, a real file where the lock is, and a link the user pointed elsewhere."""
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    (home / "bin").mkdir()
    (home / "bin" / "mytool").write_text("tool")
    (home / "share").write_text("a file, not a dir")
    big = tmp_path / "big"
    big.mkdir()
    (home / "cache").symlink_to(big)
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    legacy = home / LEGACY
    assert stat.S_IMODE(legacy.stat().st_mode) == 0o700
    assert (legacy / "unsloth_studio" / "VERSION").read_text() == "old\n"
    assert (legacy / "src" / "studio" / "uncommitted.py").read_text() == "mine"
    assert (legacy / "bin" / "mytool").read_text() == "tool"
    assert (legacy / "share").read_text() == "a file, not a dir"
    assert (legacy / "cache").is_symlink() and os.readlink(legacy / "cache") == str(big)
    assert (legacy / ".node.install.lock").is_file()
    for name in ("unsloth_studio", "src", "bin", "share", "cache", ".node.install.lock"):
        assert os.readlink(home / name) == str(app / name), name
    assert "--restore" in res.stderr
    assert "rm -rf" in res.stderr


def test_restore_puts_the_kept_entries_back_for_an_older_image(tmp_path):
    """Rollback: an image from before the split expects real directories, so the links
    go and the kept entries return. Data is not touched either way."""
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    assert _link(app, home).returncode == 0
    res = _link(app, home, "--restore")
    assert res.returncode == 0, res.stderr
    assert not (home / "unsloth_studio").is_symlink()
    assert (home / "unsloth_studio" / "VERSION").read_text() == "old\n"
    assert (home / "src" / "studio" / "uncommitted.py").read_text() == "mine"
    assert not (home / LEGACY).exists()
    # links this image added for entries the old image never had are gone too
    assert not (home / "node").exists()
    assert (home / "auth" / "auth.db").read_text() == "users"
    assert (home / "studio.db").read_text() == "chats"
    # and the next start of this image migrates again
    assert _link(app, home).returncode == 0
    assert (home / "unsloth_studio").is_symlink()


def test_restore_refuses_to_overwrite_a_real_entry(tmp_path):
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    assert _link(app, home).returncode == 0
    os.unlink(home / "src")
    (home / "src").mkdir()
    (home / "src" / "current.py").write_text("keep")
    res = _link(app, home, "--restore")
    assert res.returncode == 1
    assert "real entry" in res.stderr
    assert (home / "src" / "current.py").read_text() == "keep"
    assert (home / LEGACY / "src" / "studio" / "uncommitted.py").read_text() == "mine"


def test_restore_without_a_legacy_dir_is_a_no_op(tmp_path):
    app = _app(tmp_path)
    home = tmp_path / "home"
    assert _link(app, home).returncode == 0
    res = _link(app, home, "--restore")
    assert res.returncode == 0, res.stderr
    assert (home / "src").is_symlink()


def test_keep_legacy_0_deletes_instead(tmp_path):
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    res = _link(app, home, env = {"UNSLOTH_STUDIO_KEEP_LEGACY": "0"})
    assert res.returncode == 0, res.stderr
    assert not (home / LEGACY).exists()
    assert (home / "src").is_symlink()
    assert "removed" in res.stderr and "KEEP_LEGACY=0" in res.stderr
    assert (home / "studio.db").read_text() == "chats"


def test_only_the_latest_generation_is_kept_aside(tmp_path):
    """A second upgrade replaces the copy the first one kept; the legacy dir does not
    accumulate."""
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    assert _link(app, home).returncode == 0
    os.unlink(home / "src")
    (home / "src").mkdir()
    (home / "src" / "gen2.py").write_text("2")
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert (home / LEGACY / "src" / "gen2.py").read_text() == "2"
    assert not (home / LEGACY / "src" / "studio").exists()


def test_runtime_data_is_never_touched(tmp_path):
    app = _app(tmp_path)
    home = tmp_path / "home"
    for name in ("auth", "outputs", "exports", "runs", "rag", "assets"):
        (home / name).mkdir(parents = True)
        (home / name / "keep.txt").write_text(name)
    (home / "studio.db").write_text("chats")
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    for name in ("auth", "outputs", "exports", "runs", "rag", "assets"):
        assert not (home / name).is_symlink()
        assert (home / name / "keep.txt").read_text() == name
    assert (home / "studio.db").read_text() == "chats"


def test_links_already_in_place_are_left_alone_and_stale_ones_retargeted(tmp_path):
    """A link into an older app path is ours and is simply re-pointed; a link the user
    made is kept aside like any other entry."""
    app = _app(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "node").symlink_to(app / "node")
    (home / "src").symlink_to(app / "src-from-an-older-layout")
    (home / "share").symlink_to(tmp_path / "somewhere-else")
    before = os.lstat(home / "node").st_ino
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert os.lstat(home / "node").st_ino == before
    assert os.readlink(home / "src") == str(app / "src")
    assert os.readlink(home / "share") == str(app / "share")
    assert not (home / LEGACY / "src").exists()
    assert os.readlink(home / LEGACY / "share") == str(tmp_path / "somewhere-else")


def test_links_to_entries_this_image_dropped_are_pruned(tmp_path):
    """An earlier image's volume can hold a link to an entry, e.g. a transformers tier,
    that this image no longer ships. Nothing the user created is pruned with it."""
    app = _app(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / ".venv_t5_510").symlink_to(app / ".venv_t5_510")
    (home / "outputs").symlink_to(tmp_path / "never-created")
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert not (home / ".venv_t5_510").is_symlink()
    assert (home / "outputs").is_symlink()
    assert (home / ".venv_t5_550").is_symlink()


def test_a_second_run_is_silent_and_changes_nothing(tmp_path):
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    assert _link(app, home).returncode == 0
    snapshot = {p: os.lstat(p).st_ino for p in home.rglob("*")}
    res = _link(app, home)
    assert res.returncode == 0
    assert res.stderr == ""
    assert {p: os.lstat(p).st_ino for p in home.rglob("*")} == snapshot


def test_names_with_spaces_and_dotfiles_are_handled(tmp_path):
    app = tmp_path / "app dir"
    (app / "my venv").mkdir(parents = True)
    (app / ".hidden lock").write_text("")
    home = tmp_path / "home dir"
    (home / "my venv").mkdir(parents = True)
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert os.readlink(home / "my venv") == str(app / "my venv")
    assert os.readlink(home / ".hidden lock") == str(app / ".hidden lock")
    assert (home / LEGACY / "my venv").is_dir()


def test_the_same_root_for_app_and_home_is_refused_with_the_install_intact(tmp_path):
    """`-e UNSLOTH_STUDIO_HOME=/opt/unsloth-studio-app` would otherwise move the whole
    install aside and leave links pointing at themselves."""
    app = _app(tmp_path)
    res = _link(app, app)
    assert res.returncode == 1
    assert "must not be" in res.stderr
    assert (app / "unsloth_studio" / "VERSION").read_text() == "new\n"
    assert not (app / "src").is_symlink()
    assert not (app / LEGACY).exists()
    # through a symlink too: paths are compared as the kernel sees them
    alias = tmp_path / "alias"
    alias.symlink_to(app)
    res = _link(app, alias)
    assert res.returncode == 1
    assert not (app / "src").is_symlink()


def test_nested_roots_are_refused(tmp_path):
    app = _app(tmp_path)
    inside = app / "home"
    inside.mkdir()
    assert _link(app, inside).returncode == 1
    assert not (inside / "src").exists()
    home = tmp_path / "home"
    home.mkdir()
    nested_app = home / "app"
    nested_app.mkdir()
    assert _link(nested_app, home).returncode == 1


def _stub(bindir: Path, name: str, body: str):
    bindir.mkdir(exist_ok = True)
    p = bindir / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(0o755)


def test_an_interrupted_migration_never_loses_an_entry(tmp_path):
    """The move completes before the link is made, so a failure between the two leaves
    the kept copy in place and names it; a rerun finishes the job."""
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    bindir = tmp_path / "bin"
    _stub(
        bindir,
        "ln",
        'case "$*" in */src) echo "ln: simulated failure" >&2; exit 1;; esac\n'
        'exec /bin/ln "$@"\n',
    )
    res = _link(app, home, path = str(bindir))
    assert res.returncode == 1
    assert "cannot link" in res.stderr
    assert "intact at" in res.stderr
    kept = home / LEGACY
    # src was moved and its link failed: not lost, its kept copy is named in the message
    assert not (home / "src").exists() and not (home / "src").is_symlink()
    assert (kept / "src" / "studio" / "uncommitted.py").read_text() == "mine"
    assert str(kept / "src") in res.stderr
    # entries after src in the loop were not reached and are still real
    assert not (home / "unsloth_studio").is_symlink()
    assert (home / "studio.db").read_text() == "chats"
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert (home / "src").is_symlink() and (home / "unsloth_studio").is_symlink()
    assert (kept / "src" / "studio" / "uncommitted.py").read_text() == "mine"


def test_a_failed_move_changes_nothing(tmp_path):
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    bindir = tmp_path / "bin"
    _stub(bindir, "mv", 'echo "mv: simulated failure" >&2; exit 1\n')
    res = _link(app, home, path = str(bindir))
    assert res.returncode == 1
    assert "cannot move" in res.stderr
    assert not (home / "unsloth_studio").is_symlink()
    assert (home / "unsloth_studio" / "VERSION").read_text() == "old\n"
    assert not any((home / LEGACY).iterdir())


@pytest.mark.skipif(os.geteuid() == 0, reason = "root ignores directory modes")
def test_a_read_only_home_fails_loudly_and_touches_nothing(tmp_path):
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    home.chmod(0o555)
    try:
        res = _link(app, home)
    finally:
        home.chmod(0o755)
    assert res.returncode == 1
    assert "ERROR" in res.stderr
    assert not (home / "unsloth_studio").is_symlink()
    assert (home / "unsloth_studio" / "VERSION").read_text() == "old\n"
    assert not (home / LEGACY).exists()


def test_the_linker_is_a_no_op_without_an_app_dir(tmp_path):
    """The entrypoint belongs to the base image too, which has no Studio."""
    home = tmp_path / "home"
    res = _link(tmp_path / "missing", home)
    assert res.returncode == 0, res.stderr
    assert not home.exists()


def test_the_code_moves_in_the_same_layer_that_installs_it():
    """A `mv` in a later RUN stores the 10+ GB Studio install twice."""
    body = STUDIO_DF.read_text(encoding = "utf-8")
    runs = re.split(r"\n(?=RUN |COPY |ENV |ARG |FROM |EXPOSE |CMD |USER )", body)
    install = [r for r in runs if r.startswith("RUN ") and "bash install.sh --local" in r]
    assert len(install) == 1
    assert '-exec mv -t "${UNSLOTH_STUDIO_APP}"' in install[0]
    assert "/usr/local/bin/unsloth-studio-home" in install[0]
    assert "UNSLOTH_STUDIO_APP=/opt/unsloth-studio-app" in body


def test_the_uv_cache_goes_with_the_code_and_cache_stays_data():
    """install.sh and setup.sh default the uv cache to $STUDIO_HOME/cache/uv only when
    UV_CACHE_DIR is unset; the venv hardlinks into it (9 GB). Studio's runtime caches
    (download-resume manifests, llama slots, dataset caches) also live under cache/, so
    linking the whole directory into the app dir would have deleted a volume's runtime
    state on upgrade and sent new state into the container layer. The image points uv
    at the app dir before the install RUN and leaves cache/ in the home."""
    body = STUDIO_DF.read_text(encoding = "utf-8")
    env_block = body[
        body.index("ENV UNSLOTH_STUDIO_HOME=") : body.index(
            "\n\n", body.index("ENV UNSLOTH_STUDIO_HOME=")
        )
    ]
    assert "UV_CACHE_DIR=/opt/unsloth-studio-app/uv-cache" in env_block
    install = [
        r
        for r in re.split(r"\n(?=RUN |COPY |ENV |ARG |FROM |EXPOSE |CMD |USER )", body)
        if r.startswith("RUN ") and "bash install.sh --local" in r
    ]
    assert (
        env_block in body[: body.index(install[0])]
    ), "UV_CACHE_DIR must be set before the install"
    assert "! -name cache -exec mv -t" in install[0], "cache/ must not move to the app dir"
    assert (
        'test ! -e "${UNSLOTH_STUDIO_HOME}/cache/uv"' in install[0]
    ), "the build must prove the uv cache did not land in the home"


def test_the_entrypoint_relinks_before_it_touches_the_studio_venv_and_stops_on_failure():
    """A half-linked home must not reach Studio: the hook exits instead of warning."""
    body = ENTRYPOINT.read_text(encoding = "utf-8")
    hook = body.index("/usr/local/bin/unsloth-studio-home")
    assert hook < body.index("select_cuda_jit_tools() {")
    block = body[hook : body.index("select_cuda_jit_tools() {")]
    assert "exit 1" in block
    assert "WARN" not in block


def test_the_docs_mount_the_studio_home():
    for doc in (DOCKER / "DOCKERHUB.md", REPO / "README.md"):
        assert "-v unsloth-studio:/opt/unsloth-studio" in doc.read_text(encoding = "utf-8"), doc


def test_the_linker_is_in_the_build_context():
    """docker/.dockerignore is an allowlist; a file missing from it fails the build
    with `"/studio_home.sh": not found`."""
    allowed = (DOCKER / ".dockerignore").read_text(encoding = "utf-8").splitlines()
    assert "!studio_home.sh" in allowed
