# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The three managed-root helpers that sit outside storage_roots must agree with it.

Each helper is asked in a child process, because the CLI resolves STUDIO_HOME at
import time and the answer depends on the environment that process started with.
The runtimes split two ways and the split is the point:

  * node / llama.cpp / whisper.cpp are SIBLINGS of studio/ at the master root
    (studio/setup.sh derives them from UNSLOTH_HOME),
  * stable-diffusion.cpp installs UNDER the Studio root (#8226),

so a fix that collapsed everything onto one root would pass half of this file
and fail the other half. Legacy and plain custom roots are pinned as well, so no
fix can collapse into "always portable" either.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND = REPO_ROOT / "studio" / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unsloth_cli import _studio_stage  # noqa: E402

PORTABLE_MARKER = ".unsloth-portable-root"
STUDIO_OWNED_MARKER = ".unsloth-studio-owned"
MASTER_ROOT_RECORD = ".unsloth-master-root"

_PROBE = textwrap.dedent(
    """
    import importlib.util, json, os, sys
    from pathlib import Path

    repo = Path(sys.argv[1])
    # Before the imports: both resolvers read sys.prefix at import time, and an
    # activated venv is the only thing the symlinked layout can be reached by.
    _prefix = os.environ.get("_PREFIX")
    if _prefix:
        sys.prefix = sys.exec_prefix = _prefix
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "studio"))
    sys.path.insert(0, str(repo / "studio" / "backend"))

    from unsloth_cli import _studio_stage
    from unsloth_cli.commands.studio import STUDIO_HOME
    from core.inference.sd_cpp_engine import managed_install_root
    from utils.paths.storage_roots import studio_root, unsloth_home

    spec = importlib.util.spec_from_file_location(
        "_ill_probe", repo / "studio" / "install_llama_prebuilt.py"
    )
    llama = importlib.util.module_from_spec(spec)
    sys.modules["_ill_probe"] = llama
    spec.loader.exec_module(llama)

    master = unsloth_home()
    print(json.dumps({
        "studio_root": str(studio_root()),
        "master": None if master is None else str(master),
        "helper_root": str(_studio_stage.managed_helper_root(STUDIO_HOME)),
        "sd": str(managed_install_root()),
        "llama": str(llama.default_managed_llama_dir()),
    }))
    """
)


def probe(home: Path, **env_extra: str) -> dict:
    """Resolve every managed root in a child started with exactly *env_extra*."""
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


@pytest.fixture
def home(tmp_path: Path) -> Path:
    fake = tmp_path / "home"
    (fake / ".unsloth" / "studio").mkdir(parents = True)
    make_venv(fake / ".unsloth" / "studio")
    return fake


@pytest.fixture
def nested(tmp_path: Path) -> Path:
    """`install.sh --root DIR`: Studio at DIR/studio, runtimes beside it at DIR."""
    master = tmp_path / "master"
    (master / "studio").mkdir(parents = True)
    (master / PORTABLE_MARKER).write_text("", encoding = "utf-8")
    (master / "studio" / MASTER_ROOT_RECORD).write_text(f"{master}\n", encoding = "utf-8")
    make_venv(master / "studio")
    return master


@pytest.fixture
def symlinked(tmp_path: Path) -> Path:
    """`install.sh --root DIR` onto a DIR/studio that was ALREADY a symlink.

    _resolve_studio_destinations mkdir -p's through the link and _portable_escapes
    names studio/ in the closing summary instead of erroring, so the venv lands on
    the far volume while everything else the installer writes stays at the master
    root -- including node, llama.cpp and whisper.cpp.
    """
    master = tmp_path / "opt" / "uns"
    target = tmp_path / "bigvol" / "studio"
    (target / "unsloth_studio" / "bin").mkdir(parents = True)
    (target / "unsloth_studio" / "pyvenv.cfg").write_text("home = /usr\n", encoding = "utf-8")
    master.mkdir(parents = True)
    (master / "studio").symlink_to(target)
    (master / "bin").mkdir()
    (master / "bin" / "unsloth").write_text("#!/bin/sh\n", encoding = "utf-8")
    (master / PORTABLE_MARKER).write_text("", encoding = "utf-8")
    return master


@pytest.fixture
def flat(tmp_path: Path) -> Path:
    """`install.sh --root DIR --flat`: the master root IS the Studio root."""
    root = tmp_path / "flatroot"
    root.mkdir(parents = True)
    (root / PORTABLE_MARKER).write_text("", encoding = "utf-8")
    make_venv(root, owned = True)
    return root


# ── the sibling runtimes: node / llama.cpp / whisper.cpp live at the master root ──


def test_helper_root_is_the_master_root_for_a_nested_install(home, nested):
    """The bug this pins: <master>/studio holds no node/llama.cpp/whisper.cpp, so
    stage() cloned nothing and the staged update re-downloaded all three."""
    found = probe(home, UNSLOTH_HOME = str(nested), UNSLOTH_STUDIO_HOME = str(nested / "studio"))
    assert found["studio_root"] == str(nested / "studio")
    assert found["helper_root"] == str(nested)
    assert found["llama"] == str(nested / "llama.cpp")


def test_helper_root_uses_the_on_disk_record_without_unsloth_home(home, nested):
    """A venv-activated CLI carries no UNSLOTH_HOME; the in-root record is all it has."""
    found = probe(home, UNSLOTH_STUDIO_HOME = str(nested / "studio"))
    assert found["master"] == str(nested)
    assert found["helper_root"] == str(nested)


def test_helper_root_is_the_root_itself_for_a_flat_install(home, flat):
    found = probe(home, UNSLOTH_HOME = str(flat), UNSLOTH_STUDIO_HOME = str(flat))
    assert found["studio_root"] == str(flat)
    assert found["helper_root"] == str(flat)
    assert found["llama"] == str(flat / "llama.cpp")


def test_helper_root_is_the_parent_for_a_legacy_install(home):
    found = probe(home)
    assert found["master"] is None
    assert found["studio_root"] == str(home / ".unsloth" / "studio")
    assert found["helper_root"] == str(home / ".unsloth")
    assert found["llama"] == str(home / ".unsloth" / "llama.cpp")


def test_helper_root_is_the_studio_root_for_a_plain_custom_install(home, tmp_path):
    """No master root: setup.sh sets UNSLOTH_HOME="$STUDIO_HOME", so the runtimes
    really do sit under the custom root and the helper root must not walk up."""
    custom = tmp_path / "custom"
    custom.mkdir()
    make_venv(custom)
    found = probe(home, UNSLOTH_STUDIO_HOME = str(custom))
    assert found["master"] is None
    assert found["helper_root"] == str(custom)
    assert found["llama"] == str(custom / "llama.cpp")


def test_llama_dir_is_the_master_root_without_the_studio_home(home, nested):
    """setup.sh hands install_whisper_prebuilt.py only the master root, and the
    Studio home it inherited is <master>/studio, one level too deep for llama.cpp."""
    found = probe(home, UNSLOTH_HOME = str(nested))
    assert found["llama"] == str(nested / "llama.cpp")


def test_llama_dir_is_the_flat_root_without_the_studio_home(home, flat):
    found = probe(home, UNSLOTH_HOME = str(flat))
    assert found["llama"] == str(flat / "llama.cpp")


def test_llama_dir_honors_the_explicit_override(home, nested):
    found = probe(
        home,
        UNSLOTH_HOME = str(nested),
        UNSLOTH_LLAMA_CPP_PATH = str(nested / "elsewhere" / "llama.cpp"),
    )
    assert found["llama"] == str(nested / "elsewhere" / "llama.cpp")


def test_stage_clones_the_master_root_runtimes_for_a_nested_install(monkeypatch, tmp_path, nested):
    """End to end: with the helper root one level too deep, `helper.is_dir()` was
    false for all three and the staged update re-downloaded node, llama.cpp and
    whisper.cpp instead of reusing the copies the install already has."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "elsewhere")
    monkeypatch.setenv("UNSLOTH_HOME", str(nested))
    for name in _studio_stage.HELPER_NAMES:
        (nested / name).mkdir()
        (nested / name / "tag").write_text("live", encoding = "utf-8")
    monkeypatch.setattr(_studio_stage, "installed_version", lambda venv, env: "2026.9.1")
    monkeypatch.setattr(_studio_stage, "probe_cli", lambda venv, env: None)
    monkeypatch.setattr(_studio_stage, "probe_console_script", lambda venv, env: None)

    result = _studio_stage.stage(
        nested / "studio",
        update_args = ["--package", "unsloth"],
        echo = lambda message: None,
        run_update = lambda root, args: 0,
    )

    stage_dir = Path(result["root"])
    for name in _studio_stage.HELPER_NAMES:
        assert (stage_dir / name / "tag").read_text(encoding = "utf-8") == "live"


# ── the same master root, reached through a studio/ that was already a symlink ──


def test_helper_root_is_the_master_root_through_a_symlinked_studio(home, symlinked):
    """studio_home.resolve() is the far volume here, so neither it nor its parent
    is the master root and the containment check used to fall through to the
    Studio directory, where none of the three runtimes are."""
    prefix = symlinked / "studio" / "unsloth_studio"
    # No environment at all, which is what `source <venv>/bin/activate` leaves:
    # the marker at <master> is the only thing pointing back at the master root.
    # "llama" is not asserted here, since default_managed_llama_dir() reads the
    # environment only, by design; the exported case below is where it applies.
    found = probe(home, _PREFIX = str(prefix))
    assert found["studio_root"] == str(symlinked / "studio")
    assert found["master"] == str(symlinked)
    assert found["helper_root"] == str(symlinked)

    found = probe(home, UNSLOTH_HOME = str(symlinked))
    assert found["helper_root"] == str(symlinked)
    assert found["llama"] == str(symlinked / "llama.cpp")


def test_stage_clones_the_master_root_runtimes_through_a_symlinked_studio(
    monkeypatch, tmp_path, symlinked
):
    """End to end: stage() is handed the lexical <master>/studio the CLI resolved,
    and must still find node, llama.cpp and whisper.cpp beside it rather than
    re-downloading all three into the staged tree."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "elsewhere")
    monkeypatch.setenv("UNSLOTH_HOME", str(symlinked))
    for name in _studio_stage.HELPER_NAMES:
        (symlinked / name).mkdir()
        (symlinked / name / "tag").write_text("live", encoding = "utf-8")
    monkeypatch.setattr(_studio_stage, "installed_version", lambda venv, env: "2026.9.1")
    monkeypatch.setattr(_studio_stage, "probe_cli", lambda venv, env: None)
    monkeypatch.setattr(_studio_stage, "probe_console_script", lambda venv, env: None)

    result = _studio_stage.stage(
        symlinked / "studio",
        update_args = ["--package", "unsloth"],
        echo = lambda message: None,
        run_update = lambda root, args: 0,
    )

    stage_dir = Path(result["root"])
    for name in _studio_stage.HELPER_NAMES:
        assert (stage_dir / name / "tag").read_text(encoding = "utf-8") == "live"


# ── what the widened containment check still refuses ──


def test_helper_root_refuses_a_master_root_the_studio_root_is_not_under(
    monkeypatch, tmp_path, nested
):
    """The reason the check exists: `unsloth studio update` run inside one portable
    install must not hand a SECOND install's Studio root the first one's runtimes."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "elsewhere")
    monkeypatch.setenv("UNSLOTH_HOME", str(nested))
    unrelated = tmp_path / "unrelated" / "studio"
    unrelated.mkdir(parents = True)

    assert _studio_stage.managed_helper_root(unrelated) == unrelated


def test_helper_root_refuses_a_master_root_further_up_than_one_level(monkeypatch, tmp_path, nested):
    """One level, not any ancestor: <master>/studio/nested is not the Studio root
    of this install, and walking two up would clone from a tree it does not own."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "elsewhere")
    monkeypatch.setenv("UNSLOTH_HOME", str(nested))
    deeper = nested / "studio" / "deeper"
    deeper.mkdir(parents = True)

    assert _studio_stage.managed_helper_root(deeper) == deeper


def test_helper_root_refuses_an_unrelated_symlink_target(monkeypatch, tmp_path, nested):
    """The lexical spelling is offered, not blindly trusted: a link whose NAME sits
    outside the master root stays outside it even when it resolves inside."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "elsewhere")
    monkeypatch.setenv("UNSLOTH_HOME", str(nested))
    outside = tmp_path / "outside" / "studio"
    outside.parent.mkdir(parents = True)
    outside.symlink_to(tmp_path / "outside" / "real")
    (tmp_path / "outside" / "real").mkdir()

    assert _studio_stage.managed_helper_root(outside) == outside


# ── stable-diffusion.cpp: the one managed runtime that lives UNDER the Studio root ──


def test_sd_root_stays_under_the_studio_root_of_a_nested_install(home, nested):
    """NOT the master root. install_sd_cpp_prebuilt.default_install_dir() puts the
    tree at <UNSLOTH_STUDIO_HOME>/stable-diffusion.cpp, and uninstall.sh removes it
    with the Studio root, so resolving it beside studio/ would orphan the install."""
    found = probe(home, UNSLOTH_HOME = str(nested), UNSLOTH_STUDIO_HOME = str(nested / "studio"))
    assert found["sd"] == str(nested / "studio" / "stable-diffusion.cpp")


def test_sd_root_follows_the_studio_root_without_the_env_pair(home, nested):
    """run.py and main.py export UNSLOTH_STUDIO_HOME; nothing else has to."""
    found = probe(home, UNSLOTH_HOME = str(nested))
    assert found["studio_root"] == str(nested / "studio")
    assert found["sd"] == str(nested / "studio" / "stable-diffusion.cpp")


def test_sd_root_is_the_flat_root_itself(home, flat):
    found = probe(home, UNSLOTH_HOME = str(flat))
    assert found["studio_root"] == str(flat)
    assert found["sd"] == str(flat / "stable-diffusion.cpp")


def test_sd_root_keeps_the_legacy_sibling_placement(home):
    """The legacy default root maps to ~/.unsloth/stable-diffusion.cpp so an
    existing install is still found; that mapping predates the master root."""
    found = probe(home)
    assert found["sd"] == str(home / ".unsloth" / "stable-diffusion.cpp")


def test_sd_root_is_under_a_plain_custom_root(home, tmp_path):
    custom = tmp_path / "custom"
    custom.mkdir()
    make_venv(custom)
    found = probe(home, UNSLOTH_STUDIO_HOME = str(custom))
    assert found["sd"] == str(custom / "stable-diffusion.cpp")
