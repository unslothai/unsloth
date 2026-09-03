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
    import importlib.util, json, sys
    from pathlib import Path

    repo = Path(sys.argv[1])
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
