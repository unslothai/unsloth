# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""stable-diffusion.cpp must not be installed OUTSIDE a flat portable root.

sd.cpp is the one managed runtime that lives UNDER the Studio root (#8226), and the
legacy default Studio root ``~/.unsloth/studio`` maps its tree to the SIBLING
``~/.unsloth/stable-diffusion.cpp`` so installs made before that fix are still found.
``install.sh --portable`` with UNSLOTH_STUDIO_HOME already set, and ``--root`` pointed at
the existing default install, both make that same directory the FLAT portable master
root, at which point the sibling is one level above everything ``rm -rf <root>`` removes.

So the remap has to survive in three shapes and disappear in exactly one, and both the
reader (sd_cpp_engine.managed_install_root) and the writer
(install_sd_cpp_prebuilt.default_install_dir) have to make the same call, or the finder
looks where the installer never wrote and every load re-downloads the bundle.

The write is the containment violation; the READ of an existing marked tree is not, and
it is kept, since the alternative is re-downloading hundreds of megabytes a converted
install already has on disk.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PORTABLE_MARKER = ".unsloth-portable-root"
STUDIO_OWNED_MARKER = ".unsloth-studio-owned"
MASTER_ROOT_RECORD = ".unsloth-master-root"

_PROBE = textwrap.dedent(
    """
    import importlib.util, json, sys
    from pathlib import Path

    repo = Path(sys.argv[1])
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "studio"))
    sys.path.insert(0, str(repo / "studio" / "backend"))

    from core.inference.sd_cpp_engine import (
        find_sd_cpp_binary, legacy_sibling_install_root, managed_install_root,
    )
    from utils.paths.storage_roots import studio_root, unsloth_home

    spec = importlib.util.spec_from_file_location(
        "_isd_probe", repo / "studio" / "install_sd_cpp_prebuilt.py"
    )
    installer = importlib.util.module_from_spec(spec)
    sys.modules["_isd_probe"] = installer
    spec.loader.exec_module(installer)

    master = unsloth_home()
    legacy = legacy_sibling_install_root()
    print(json.dumps({
        "studio_root": str(studio_root()),
        "master": None if master is None else str(master),
        "read": str(managed_install_root()),
        "write": str(installer.default_install_dir()),
        "legacy_read": None if legacy is None else str(legacy),
        "found": find_sd_cpp_binary(),
    }))
    """
)


def probe(home: Path, **env_extra: str) -> dict:
    """Resolve the sd.cpp roots in a child started with exactly *env_extra*.

    A child process, because both resolvers answer from the environment the process
    started with and the CLI resolves STUDIO_HOME at import time.
    """
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "")}
    env.update({key: value for key, value in env_extra.items() if value})
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(REPO_ROOT)],
        env = env,
        capture_output = True,
        text = True,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def make_venv(root: Path, *, owned: bool = False) -> None:
    (root / "unsloth_studio" / "bin").mkdir(parents = True, exist_ok = True)
    (root / "unsloth_studio" / "pyvenv.cfg").write_text("home = /usr\n", encoding = "utf-8")
    if owned:
        (root / "unsloth_studio" / STUDIO_OWNED_MARKER).write_text("", encoding = "utf-8")


def plant_managed_tree(root: Path) -> Path:
    """A marked managed sd.cpp tree with a runnable-looking sd-cli, as an install leaves it."""
    binary = root / "bin" / "sd-cli"
    binary.parent.mkdir(parents = True, exist_ok = True)
    binary.write_text("#!/bin/sh\n", encoding = "utf-8")
    binary.chmod(0o755)
    (root / STUDIO_OWNED_MARKER).write_text("", encoding = "utf-8")
    return binary


@pytest.fixture
def legacy_home(tmp_path: Path) -> Path:
    """(a) The plain default install: ~/.unsloth/studio, no portable root anywhere."""
    home = tmp_path / "legacy"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    make_venv(home / ".unsloth" / "studio")
    return home


@pytest.fixture
def flat_legacy_home(tmp_path: Path) -> Path:
    """(b) `install.sh --portable` over UNSLOTH_STUDIO_HOME=~/.unsloth/studio, or `--root` at it.

    _PORTABLE_FLAT is true for both, so ~/.unsloth/studio is the master root AND the
    Studio root, and ~/.unsloth is outside the install.
    """
    home = tmp_path / "flatlegacy"
    root = home / ".unsloth" / "studio"
    root.mkdir(parents = True)
    make_venv(root, owned = True)
    (root / PORTABLE_MARKER).write_text("", encoding = "utf-8")
    return home


@pytest.fixture
def flat_elsewhere(tmp_path: Path) -> tuple[Path, Path]:
    """(c) A flat portable root that is not the legacy path. Returns (home, master)."""
    home = tmp_path / "flatother" / "home"
    home.mkdir(parents = True)
    master = tmp_path / "flatother" / "opt" / "uns"
    master.mkdir(parents = True)
    make_venv(master, owned = True)
    (master / PORTABLE_MARKER).write_text("", encoding = "utf-8")
    return home, master


@pytest.fixture
def nested_elsewhere(tmp_path: Path) -> tuple[Path, Path]:
    """(d) `install.sh --root DIR`: Studio at DIR/studio. Returns (home, master)."""
    home = tmp_path / "nested" / "home"
    home.mkdir(parents = True)
    master = tmp_path / "nested" / "opt" / "uns"
    (master / "studio").mkdir(parents = True)
    make_venv(master / "studio")
    (master / PORTABLE_MARKER).write_text("", encoding = "utf-8")
    (master / "studio" / MASTER_ROOT_RECORD).write_text(f"{master}\n", encoding = "utf-8")
    return home, master


@pytest.fixture
def nested_legacy_home(tmp_path: Path) -> Path:
    """(e) `install.sh --root ~/.unsloth`: master ~/.unsloth, Studio ~/.unsloth/studio.

    The case that stops the fix collapsing into "portable means never remap": the
    sibling ~/.unsloth/stable-diffusion.cpp is INSIDE this master root, exactly where
    the master root's own llama.cpp, node and whisper.cpp are.
    """
    home = tmp_path / "nestedlegacy"
    (home / ".unsloth" / "studio").mkdir(parents = True)
    make_venv(home / ".unsloth" / "studio")
    (home / ".unsloth" / PORTABLE_MARKER).write_text("", encoding = "utf-8")
    (home / ".unsloth" / "studio" / MASTER_ROOT_RECORD).write_text(
        f"{home / '.unsloth'}\n", encoding = "utf-8"
    )
    return home


# ── the four layouts, reader and writer pinned together ──


def test_plain_legacy_install_keeps_the_sibling_placement(legacy_home):
    """(a) No portable root, so nothing promises containment and the pre-#8226 tree
    at ~/.unsloth/stable-diffusion.cpp stays exactly where every existing install has it."""
    found = probe(legacy_home)
    sibling = str(legacy_home / ".unsloth" / "stable-diffusion.cpp")
    assert found["master"] is None
    assert found["studio_root"] == str(legacy_home / ".unsloth" / "studio")
    assert found["read"] == sibling
    assert found["write"] == sibling


@pytest.mark.parametrize("with_unsloth_home", [False, True])
def test_flat_legacy_root_keeps_sd_cpp_inside_itself(flat_legacy_home, with_unsloth_home):
    """(b) The bug: ~/.unsloth/studio is the master root, so ~/.unsloth is outside the
    install and a managed sd.cpp there survives the advertised `rm -rf <root>`.

    Both spellings, because setup.sh exports UNSLOTH_HOME while a venv-activated
    process has only the on-disk marker.
    """
    root = flat_legacy_home / ".unsloth" / "studio"
    extra = {"UNSLOTH_HOME": str(root)} if with_unsloth_home else {}
    found = probe(flat_legacy_home, UNSLOTH_STUDIO_HOME = str(root), **extra)
    assert found["master"] == str(root)
    assert found["read"] == str(root / "stable-diffusion.cpp")
    assert found["write"] == str(root / "stable-diffusion.cpp")
    # The whole point: the write target is under the directory uninstall removes.
    assert Path(found["write"]).is_relative_to(root)


def test_flat_root_elsewhere_is_unchanged(flat_elsewhere):
    """(c) A flat root that was never the legacy path never had the remap to lose."""
    home, master = flat_elsewhere
    found = probe(home, UNSLOTH_HOME = str(master), UNSLOTH_STUDIO_HOME = str(master))
    assert found["read"] == str(master / "stable-diffusion.cpp")
    assert found["write"] == str(master / "stable-diffusion.cpp")


def test_nested_portable_keeps_sd_cpp_under_the_studio_root(nested_elsewhere):
    """(d) NOT the master root: sd.cpp is the one runtime that installs under studio/,
    and uninstall.sh removes it with the Studio root."""
    home, master = nested_elsewhere
    found = probe(home, UNSLOTH_HOME = str(master), UNSLOTH_STUDIO_HOME = str(master / "studio"))
    assert found["master"] == str(master)
    assert found["read"] == str(master / "studio" / "stable-diffusion.cpp")
    assert found["write"] == str(master / "studio" / "stable-diffusion.cpp")


def test_nested_master_at_the_legacy_parent_keeps_the_sibling(nested_legacy_home):
    """(e) `--root ~/.unsloth` owns ~/.unsloth, so the sibling is contained and the
    remap must stay. A fix keyed on "portable" rather than on "the root IS the master"
    would move this tree and strand the existing one."""
    root = nested_legacy_home / ".unsloth" / "studio"
    master = nested_legacy_home / ".unsloth"
    found = probe(nested_legacy_home, UNSLOTH_STUDIO_HOME = str(root))
    assert found["master"] == str(master)
    assert found["read"] == str(master / "stable-diffusion.cpp")
    assert found["write"] == str(master / "stable-diffusion.cpp")
    assert Path(found["read"]).is_relative_to(master)


# ── read versus write: an existing tree outside a converted root is still used ──


def test_converted_install_still_reads_its_pre_conversion_tree(flat_legacy_home):
    """The standing rule: pin only when there is nothing to strand.

    A default install converted with `--root ~/.unsloth/studio` already has a marked,
    multi-hundred-megabyte sd.cpp at ~/.unsloth/stable-diffusion.cpp. Writing new
    installs inside the root is containment; refusing to READ that tree would only
    re-download it. legacy_sibling_install_root() is marker-gated, so nothing but a
    real previous install of ours is picked up.
    """
    root = flat_legacy_home / ".unsloth" / "studio"
    sibling = flat_legacy_home / ".unsloth" / "stable-diffusion.cpp"
    binary = plant_managed_tree(sibling)

    found = probe(flat_legacy_home, UNSLOTH_STUDIO_HOME = str(root))
    assert found["write"] == str(root / "stable-diffusion.cpp")
    assert found["legacy_read"] == str(sibling)
    assert found["found"] == str(binary)


def test_unmarked_sibling_directory_is_not_adopted(flat_legacy_home):
    """A `git clone` of leejet's repo beside the root is the user's, not a stale install."""
    root = flat_legacy_home / ".unsloth" / "studio"
    sibling = flat_legacy_home / ".unsloth" / "stable-diffusion.cpp"
    (sibling / "bin").mkdir(parents = True)
    (sibling / "bin" / "sd-cli").write_text("#!/bin/sh\n", encoding = "utf-8")

    found = probe(flat_legacy_home, UNSLOTH_STUDIO_HOME = str(root))
    assert found["legacy_read"] is None


def test_a_tree_inside_the_converted_root_wins_over_the_legacy_one(flat_legacy_home):
    """Once the bundle has been installed inside the root, that is the one the finder
    returns; the pre-conversion tree is only the fallback."""
    root = flat_legacy_home / ".unsloth" / "studio"
    plant_managed_tree(flat_legacy_home / ".unsloth" / "stable-diffusion.cpp")
    inside = plant_managed_tree(root / "stable-diffusion.cpp")

    found = probe(flat_legacy_home, UNSLOTH_STUDIO_HOME = str(root))
    assert found["found"] == str(inside)
