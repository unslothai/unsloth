# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A portable backend started directly must pin pip's cache like its launchers do.

`install.sh --portable` exports `PIP_CACHE_DIR=<root>/cache/pip` from
`_export_portable_roots`, writes the same line into `share/studio.conf`, and puts
it in the generated `bin/unsloth` wrapper; the CLI rebuilds it in
`_portable_root_env`. The supported `uvicorn main:app` path reaches none of
those: `main.py` calls `setup_cache_env()` and that was the whole environment, so
`python -m pip cache dir` answered `$HOME/.cache/pip` and the wheels below landed
outside the root.

Not an install-time-only concern. `utils/wheel_utils.install_wheel` falls back to
`python -m pip install <wheel_url>` whenever uv is missing or its attempt fails,
and `core/training/worker._pip_install_cmd` does the same for TileLang and
apache-tvm-ffi; both run inside a live Studio and both inherit this environment.

Nothing is stranded by the pin: an existing `~/.cache/pip` is left where it is
and merely stops being written to, and pip's cache is re-downloadable by
definition. That is why it is unconditional in portable mode, unlike MPLCONFIGDIR
or DATA_DESIGNER_HOME, which hold user configuration and are gated on the
directory being empty.
"""

import ast
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_STORAGE_ROOTS_PATH = _BACKEND_DIR / "utils" / "paths" / "storage_roots.py"
_MAIN_PATH = _BACKEND_DIR / "main.py"
_REPO = _BACKEND_DIR.parent.parent
_INSTALL_SH = _REPO / "install.sh"
_CLI_STUDIO = _REPO / "unsloth_cli" / "commands" / "studio.py"

_CACHE_KEYS = (
    "PIP_CACHE_DIR",
    "UV_CACHE_DIR",
    "TORCH_HOME",
    "HF_DATASETS_CACHE",
    "HF_ASSETS_CACHE",
    "UNSLOTH_COMPILE_LOCATION",
    "MPLCONFIGDIR",
    "DATA_DESIGNER_HOME",
    "DATA_DESIGNER_MANAGED_ASSETS_PATH",
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "HUGGINGFACE_HUB_CACHE",
)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch, tmp_path):
    for key in _CACHE_KEYS:
        monkeypatch.delenv(key, raising = False)
    for key in ("UNSLOTH_HOME", "UNSLOTH_PORTABLE", "STUDIO_HOME", "UNSLOTH_STUDIO_HOME"):
        monkeypatch.delenv(key, raising = False)
    for key in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TRITON_HOME"):
        monkeypatch.delenv(key, raising = False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _load_storage_roots():
    sys.modules.pop("utils.hf_cache_settings", None)
    spec = importlib.util.spec_from_file_location("storage_roots_pip_cache", _STORAGE_ROOTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_direct_uvicorn_path_really_is_setup_cache_env():
    """main.py's only cache seeding is this call; the test would be about nothing
    if it were removed or renamed."""
    tree = ast.parse(_MAIN_PATH.read_text(encoding = "utf-8"))
    imported = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "utils.paths.storage_roots"
        and any(alias.name == "setup_cache_env" for alias in node.names)
        for node in ast.walk(tree)
    )
    assert imported, "main.py no longer imports setup_cache_env from storage_roots"


def test_portable_direct_launch_pins_pip_cache_under_the_root(monkeypatch, tmp_path):
    master = tmp_path / "portable"
    (master / "unsloth_studio").mkdir(parents = True)
    (master / "unsloth_studio" / ".unsloth-studio-owned").write_text("")
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()

    # Flat layout, so the Studio root and the master root are the same directory.
    assert sr.studio_root() == master
    sr.setup_cache_env()

    assert os.environ["PIP_CACHE_DIR"] == str(master / "cache" / "pip")


def test_the_nested_layout_pins_pip_at_the_master_root_not_the_studio_root(monkeypatch, tmp_path):
    """install.sh exports `$UNSLOTH_ROOT/cache/pip`, which is one level ABOVE the
    Studio root under the default nested layout. Deriving it from cache_root()
    would give <root>/studio/cache/pip and split the cache in two."""
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()

    assert sr.studio_root() == master / "studio"
    sr.setup_cache_env()

    assert os.environ["PIP_CACHE_DIR"] == str(master / "cache" / "pip")


def test_a_normal_install_keeps_the_shared_pip_cache(monkeypatch, tmp_path):
    """~/.cache/pip is shared with every other tool on the machine; only a
    portable install has asked for it to move."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    sr = _load_storage_roots()

    assert sr.portable_mode() is False
    sr.setup_cache_env()

    assert "PIP_CACHE_DIR" not in os.environ


@pytest.mark.parametrize("preset", ("/tmp/chosen-pip-cache", "   "))
def test_an_existing_value_is_respected_and_a_blank_one_is_not(monkeypatch, tmp_path, preset):
    """setdefault semantics, and the same blank-counts-as-unset rule every other
    cache default here uses: a blank PIP_CACHE_DIR is a relative path pip would
    resolve against the working directory."""
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    monkeypatch.setenv("PIP_CACHE_DIR", preset)
    sr = _load_storage_roots()

    sr.setup_cache_env()

    expected = preset if preset.strip() else str(master / "cache" / "pip")
    assert os.environ["PIP_CACHE_DIR"] == expected


def test_the_pinned_path_matches_install_sh_and_the_cli(monkeypatch, tmp_path):
    """One shape, three writers. install.sh's export, the CLI's _portable_root_env
    and this default must name the same directory, or the launcher and the direct
    launch fill two different caches."""
    install_text = _INSTALL_SH.read_text(encoding = "utf-8", errors = "replace")
    assert re.search(
        r'^\s*export PIP_CACHE_DIR="\$UNSLOTH_ROOT/cache/pip"$', install_text, re.MULTILINE
    ), "install.sh no longer exports PIP_CACHE_DIR as $UNSLOTH_ROOT/cache/pip"

    cli_text = _CLI_STUDIO.read_text(encoding = "utf-8", errors = "replace")
    assert (
        '"PIP_CACHE_DIR": str(master / "cache" / "pip")' in cli_text
    ), "the CLI no longer derives PIP_CACHE_DIR as <master>/cache/pip"

    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()
    sr.setup_cache_env()

    assert os.environ["PIP_CACHE_DIR"] == str(master / "cache" / "pip")


def test_the_runtime_pip_fallbacks_this_pin_is_for_still_exist():
    """Both live installers really do fall back to `python -m pip install`. If
    they stop, this pin protects nothing and the comment above is stale."""
    wheel_text = (_BACKEND_DIR / "utils" / "wheel_utils.py").read_text(encoding = "utf-8")
    assert (
        '[python_executable, "-m", "pip", "install"' in wheel_text
    ), "wheel_utils no longer falls back to python -m pip install"

    worker_text = (_BACKEND_DIR / "core" / "training" / "worker.py").read_text(encoding = "utf-8")
    assert (
        '[sys.executable, "-m", "pip", "install", *args]' in worker_text
    ), "worker._pip_install_cmd no longer falls back to python -m pip install"
