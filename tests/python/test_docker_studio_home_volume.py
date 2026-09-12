# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""The Studio image splits Studio's code from its data.

Studio keeps one root for both: the venv, source tree and Node next to auth/,
studio.db, outputs/ and exports/. Without a volume on it, `docker rm` lost every
account, chat and trained model; with one, the volume kept the first image's code and
every later image ran that old Studio. The image now keeps the code in
$UNSLOTH_STUDIO_APP and links it into $UNSLOTH_STUDIO_HOME, which the entrypoint
repairs at every start.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOCKER = REPO / "docker"
LINKER = DOCKER / "studio_home.sh"
STUDIO_DF = DOCKER / "Dockerfile.studio"
ENTRYPOINT = DOCKER / "entrypoint.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason = "needs bash")


def _app(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    for name in ("unsloth_studio", "src", "node", "bin", "share", "cache", ".venv_t5_550"):
        (app / name).mkdir(parents = True)
    (app / "unsloth_studio" / "VERSION").write_text("new\n")
    (app / ".node.install.lock").write_text("")
    (app / "llama.cpp").symlink_to("/opt/unsloth/llama.cpp")
    return app


def _link(app: Path, home: Path):
    env = dict(os.environ, UNSLOTH_STUDIO_APP = str(app), UNSLOTH_STUDIO_HOME = str(home))
    return subprocess.run(
        ["bash", str(LINKER)], env = env, capture_output = True, text = True, timeout = 60
    )


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


def test_a_volume_from_an_earlier_image_runs_this_images_code(tmp_path):
    """The freeze: an earlier image's venv and source tree sat in the volume as real
    directories, so a new image kept running the old Studio."""
    app = _app(tmp_path)
    home = tmp_path / "home"
    (home / "unsloth_studio").mkdir(parents = True)
    (home / "unsloth_studio" / "VERSION").write_text("old\n")
    (home / "src" / "studio").mkdir(parents = True)
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert (home / "unsloth_studio").is_symlink()
    assert (home / "unsloth_studio" / "VERSION").read_text() == "new\n"
    assert (home / "src").is_symlink()
    assert "earlier image" in res.stderr


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
    app = _app(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    (home / "node").symlink_to(app / "node")
    (home / "src").symlink_to(tmp_path / "somewhere-else")
    before = os.lstat(home / "node").st_ino
    res = _link(app, home)
    assert res.returncode == 0, res.stderr
    assert os.lstat(home / "node").st_ino == before
    assert os.readlink(home / "src") == str(app / "src")


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


def test_the_entrypoint_relinks_before_it_touches_the_studio_venv():
    body = ENTRYPOINT.read_text(encoding = "utf-8")
    hook = body.index("/usr/local/bin/unsloth-studio-home")
    assert hook < body.index("select_cuda_jit_tools() {")


def test_the_docs_mount_the_studio_home():
    for doc in (DOCKER / "DOCKERHUB.md", REPO / "README.md"):
        assert "-v unsloth-studio:/opt/unsloth-studio" in doc.read_text(encoding = "utf-8"), doc


def test_the_linker_is_in_the_build_context():
    """docker/.dockerignore is an allowlist; a file missing from it fails the build
    with `"/studio_home.sh": not found`."""
    allowed = (DOCKER / ".dockerignore").read_text(encoding = "utf-8").splitlines()
    assert "!studio_home.sh" in allowed
