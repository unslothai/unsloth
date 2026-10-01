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

# The actionable half of the "nothing to restore" refusal, quoted from studio_home.sh. Kept as a
# named constant because it is asserted twice: once against the script's source, so a copy edit
# fails naming the file and the line to change, and once against the stderr an actual run
# produces, so a script that no longer reaches that branch cannot pass on the source check alone.
# #11254 changed this sentence ("the Studio code" -> "the Unsloth Studio code") and left the
# expectation behind, which failed as an opaque runtime mismatch in Repo tests (CPU, python).
RESTORE_NEEDS_APP_HINT = "run --restore under an image that has the Unsloth Studio code in"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")


def _app(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    for name in (
        "unsloth_studio",
        "src",
        "node",
        "bin",
        "share",
        "cache",
        ".venv_t5_550",
        "uv-cache",
    ):
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


def test_restore_without_a_legacy_dir_copies_this_images_code_in(tmp_path):
    """A volume first used after the split has no old code to put back. An older image
    cannot run links into an app dir it does not have, so --restore materialises copies."""
    app = _app(tmp_path)
    home = tmp_path / "home"
    (home / "outputs").mkdir(parents = True)
    (home / "outputs" / "model.bin").write_text("weights")
    assert _link(app, home).returncode == 0
    res = _link(app, home, "--restore")
    assert res.returncode == 0, res.stderr
    assert not (home / "src").is_symlink() and (home / "src").is_dir()
    assert (home / "unsloth_studio" / "VERSION").read_text() == "new\n"
    assert not (home / "llama.cpp").exists() or (home / "llama.cpp").is_symlink()
    assert (home / "outputs" / "model.bin").read_text() == "weights"
    assert "copied this image's code" in res.stderr
    # and a split image links its own code back in on the next start, keeping the copies aside
    assert _link(app, home).returncode == 0
    assert (home / "src").is_symlink()
    assert (home / LEGACY / "src").is_dir()


def test_restore_without_a_legacy_dir_or_an_app_dir_says_which_image_to_use(tmp_path):
    app = _app(tmp_path)
    home = tmp_path / "home"
    assert _link(app, home).returncode == 0
    import shutil

    shutil.rmtree(app)
    res = _link(app, home, "--restore")
    assert res.returncode == 1
    assert RESTORE_NEEDS_APP_HINT in res.stderr
    assert (home / "src").is_symlink()


def test_the_restore_hint_this_file_expects_is_the_one_the_script_prints() -> None:
    """Catch a copy edit at the source, not as a mismatch in someone else's run.

    The test above compares against stderr, so when the wording moves it fails with two long
    strings and no indication that the fix is one line of shell. This names the file.
    """
    assert RESTORE_NEEDS_APP_HINT in LINKER.read_text(encoding = "utf-8"), (
        f"{LINKER.relative_to(REPO)} no longer prints {RESTORE_NEEDS_APP_HINT!r}. If the wording "
        f"changed on purpose, update RESTORE_NEEDS_APP_HINT here to match."
    )


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


def test_a_link_where_the_kept_aside_copies_go_is_refused(tmp_path):
    """mkdir -p, rm -rf and mv all follow a symlink, so a link left at
    .unsloth-studio-legacy would send the kept-aside copies wherever it points, the app
    dir included. Both the link run and --restore stop before writing through it."""
    app = _app(tmp_path)
    home = _legacy_home(tmp_path)
    decoy = tmp_path / "decoy"
    (decoy / "src").mkdir(parents = True)
    (decoy / "src" / "app.py").write_text("this image's code")
    (home / LEGACY).symlink_to(decoy)
    for args in ((), ("--restore",)):
        res = _link(app, home, *args)
        assert res.returncode == 1, res.stderr
        assert LEGACY in res.stderr and "must be a directory" in res.stderr
        assert [p.name for p in decoy.iterdir()] == ["src"]
        assert (decoy / "src" / "app.py").read_text() == "this image's code"
        assert (home / LEGACY).is_symlink()
        assert not (home / "src").is_symlink()
        assert (home / "src" / "studio" / "uncommitted.py").read_text() == "mine"
        assert (home / "unsloth_studio" / "VERSION").read_text() == "old\n"
        assert not (home / "node").exists()
    # with the link out of the way the migration runs as usual
    os.unlink(home / LEGACY)
    assert _link(app, home).returncode == 0
    assert (home / "src").is_symlink()


def test_a_restore_copy_that_fails_leaves_the_link_and_no_half_tree(tmp_path):
    """The copy lands beside the link and is swapped in whole. A partial real directory
    at the entry's name would be skipped by the rerun's symlink-only loop, so --restore
    would then report success over code that is missing files."""
    app = _app(tmp_path)
    (app / "src" / "studio").mkdir()
    (app / "src" / "studio" / "main.py").write_text("code")
    home = tmp_path / "home"
    assert _link(app, home).returncode == 0
    bindir = tmp_path / "bin"
    _stub(
        bindir,
        "cp",
        'case "$*" in\n'
        '    *"/src "*) mkdir -p "${@: -1}"; echo half > "${@: -1}/half.py";\n'
        '           echo "cp: simulated failure" >&2; exit 1;;\n'
        "esac\n"
        'exec /bin/cp "$@"\n',
    )
    res = _link(app, home, "--restore", path = str(bindir))
    assert res.returncode == 1
    assert "rerun --restore" in res.stderr
    assert (home / "src").is_symlink()
    assert not (home / "src.restore-tmp").exists()
    # the rerun sees the link it left intact and finishes the job
    res = _link(app, home, "--restore")
    assert res.returncode == 0, res.stderr
    assert (home / "src").is_dir() and not (home / "src").is_symlink()
    assert (home / "src" / "studio" / "main.py").read_text() == "code"
    assert not (home / "src" / "half.py").exists()
    assert not (home / "src.restore-tmp").exists()


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0,
    reason = "needs POSIX directory modes, and root ignores them",
)
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


def test_updater_scratch_in_the_app_dir_is_never_linked_into_the_home(tmp_path):
    """unsloth-studio-update stages the new tree as .src-update.* and parks the old one as
    .src-prev.* next to src; a killed update leaves them behind. They are scratch."""
    app = _app(tmp_path)
    (app / ".src-update.abc123").mkdir()
    (app / ".src-prev.4242").mkdir()
    home = tmp_path / "home"
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert not (home / ".src-update.abc123").exists()
    assert not (home / ".src-prev.4242").exists()
    assert (home / "src").is_symlink()


def test_a_src_lost_between_the_updaters_two_renames_is_put_back(tmp_path):
    """SIGKILL between `mv src .src-prev.X` and `mv .src-update.Y src` leaves the previous
    tree as the only copy; the home's src link must not be pruned as dangling."""
    app = _app(tmp_path)
    (app / "src" / "studio").mkdir()
    (app / "src").rename(app / ".src-prev.k9x2Qa")
    (app / ".src-update.abc123").mkdir()
    home = tmp_path / "home"
    home.mkdir()
    (home / "src").symlink_to(app / "src")
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert (app / "src" / "studio").is_dir()
    assert not (app / ".src-prev.k9x2Qa").exists()
    assert (home / "src").is_symlink() and (home / "src" / "studio").is_dir()
    assert "put " in res.stderr


def test_a_killed_updates_record_is_recovered_before_studio_starts(tmp_path):
    """SIGKILL after the swap and the package replacement: no trap ran, so the record
    unsloth-studio-update keeps beside src is still there at the next container start.
    The linker hands it to the updater's --recover, with the home it just linked,
    before supervisord starts Studio on the unverified tree."""
    app = _app(tmp_path)
    (app / ".src-prev.k9x2Qa").mkdir()
    (app / ".src-update.rollback").write_text("-e file:///opt/prev-src\n")
    home = tmp_path / "home"
    stub = tmp_path / "stub" / "unsloth-studio-update"
    stub.parent.mkdir()
    stub.write_text(
        '#!/usr/bin/env bash\necho "UPDATER $* home=$UNSLOTH_STUDIO_HOME" >> "$STUB_LOG"\nexit "${STUB_RC:-0}"\n'
    )
    stub.chmod(0o755)
    log = tmp_path / "calls.log"
    env = {"UNSLOTH_STUDIO_UPDATER": str(stub), "STUB_LOG": str(log)}
    res = _link(app, home, env = env)
    assert res.returncode == 0, res.stderr
    assert log.read_text() == f"UPDATER --recover home={home}\n", log.read_text()
    assert "the previous install is back" in res.stderr, res.stderr
    assert (home / "unsloth_studio").is_symlink(), "recovery must run after the home is linked"
    # a failed recovery is loud but does not stop the container from starting
    log.unlink()
    res = _link(app, home, env = {**env, "STUB_RC": "1"})
    assert res.returncode == 0, res.stderr
    assert "WARNING" in res.stderr and "--recover" in res.stderr, res.stderr
    # no record: the updater is not run at all
    (app / ".src-update.rollback").unlink()
    log.unlink()
    res = _link(app, home, env = env)
    assert res.returncode == 0 and not log.exists()


def test_two_previous_trees_are_left_for_a_human(tmp_path):
    app = _app(tmp_path)
    (app / "src").rename(app / ".src-prev.aaaaaa")
    (app / ".src-prev.bbbbbb").mkdir()
    home = tmp_path / "home"
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert not (app / "src").exists()
    assert "several .src-prev" in res.stderr


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


def test_the_in_app_updates_know_the_images_code_tree():
    """whisper.cpp is discovered through the Studio home, where the linker leaves a link
    into the app dir, and the updater reads a linked component dir as the user's own
    checkout and offers nothing. It tells the two apart by the same variable the image
    sets, so the name must not drift apart from the image's."""
    flow = (REPO / "studio" / "backend" / "utils" / "prebuilt" / "update_flow.py").read_text(
        encoding = "utf-8"
    )
    assert 'os.environ.get("UNSLOTH_STUDIO_APP")' in flow
    assert "UNSLOTH_STUDIO_APP=/opt/unsloth-studio-app" in STUDIO_DF.read_text(encoding = "utf-8")


def test_the_docs_mount_the_studio_home():
    for doc in (DOCKER / "DOCKERHUB.md", REPO / "README.md"):
        assert "-v unsloth-studio:/opt/unsloth-studio" in doc.read_text(encoding = "utf-8"), doc


def test_the_linker_is_in_the_build_context():
    """docker/.dockerignore is an allowlist; a file missing from it fails the build
    with `"/studio_home.sh": not found`."""
    allowed = (DOCKER / ".dockerignore").read_text(encoding = "utf-8").splitlines()
    assert "!studio_home.sh" in allowed
