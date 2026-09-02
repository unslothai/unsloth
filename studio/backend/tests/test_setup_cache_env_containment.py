# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Everything Unsloth writes should sit under one directory (issue #8865)."""

import contextlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_STORAGE_ROOTS_PATH = Path(__file__).resolve().parent.parent / "utils/paths/storage_roots.py"

_ALWAYS_PINNED = (
    "UV_CACHE_DIR",
    "VLLM_CACHE_ROOT",
    "UNSLOTH_COMPILE_LOCATION",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "CUDA_CACHE_PATH",
    "MPLCONFIGDIR",
    "NUMBA_CACHE_DIR",
    "DATA_DESIGNER_HOME",
    "DATA_DESIGNER_MANAGED_ASSETS_PATH",
)

# Shared user data / large re-downloads: portable mode only.
_PORTABLE_ONLY = ("HF_DATASETS_CACHE", "HF_ASSETS_CACHE", "TORCH_HOME")

_HF_ENV = ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HUGGINGFACE_HUB_CACHE")


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch, tmp_path):
    for key in _ALWAYS_PINNED + _PORTABLE_ONLY + _HF_ENV:
        monkeypatch.delenv(key, raising = False)
    for key in ("UNSLOTH_HOME", "UNSLOTH_PORTABLE", "STUDIO_HOME"):
        monkeypatch.delenv(key, raising = False)
    # _default_cache_home reads this before ~/.cache, and CI runners set it.
    monkeypatch.delenv("XDG_CACHE_HOME", raising = False)
    # Same for matplotlib's config dir, whose contents decide whether MPLCONFIGDIR
    # is ours to pin.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising = False)
    # Empty home: a real ~/.data-designer would change what the resolver pins.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))


@pytest.fixture(autouse = True)
def _restore_hf_cache_settings_module():
    # Left popped, the next import builds a second module object, so a later
    # test writes one and reads the other.
    import utils

    name = "utils.hf_cache_settings"
    saved = sys.modules.get(name)
    saved_attr = getattr(utils, "hf_cache_settings", None)
    try:
        yield
    finally:
        if saved is not None:
            sys.modules[name] = saved
        else:
            sys.modules.pop(name, None)
        if saved_attr is not None:
            utils.hf_cache_settings = saved_attr
        else:
            with contextlib.suppress(AttributeError):
                del utils.hf_cache_settings


def _load_storage_roots():
    # The resolver snapshots explicit env vars once per process.
    sys.modules.pop("utils.hf_cache_settings", None)
    spec = importlib.util.spec_from_file_location("storage_roots_under_test", _STORAGE_ROOTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_regenerable_caches_are_pinned_under_the_studio_root(tmp_path):
    sr = _load_storage_roots()

    sr._setup_cache_env()

    root = str(tmp_path / "studio")
    for key in _ALWAYS_PINNED:
        value = os.environ.get(key)
        assert value, f"{key} was not pinned"
        assert value.startswith(root), f"{key} escaped the studio root: {value}"


def test_default_install_leaves_the_shared_hf_cache_alone(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["HF_HUB_CACHE"] == str(home / ".cache" / "huggingface" / "hub")
    for key in _PORTABLE_ONLY:
        assert key not in os.environ


def test_portable_mode_moves_the_hf_and_torch_caches_under_the_root(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()

    assert sr.portable_mode() is True
    assert sr.studio_root() == master / "studio"
    sr._setup_cache_env()

    root = str(master)
    for key in _ALWAYS_PINNED + _PORTABLE_ONLY + ("HF_HUB_CACHE", "HF_XET_CACHE"):
        value = os.environ.get(key)
        assert value, f"{key} was not pinned"
        assert value.startswith(root), f"{key} escaped the portable root: {value}"


def test_portable_mode_still_leaves_hf_home_alone(monkeypatch, tmp_path):
    # HF_HOME owns the token path; credentials stay off a removable volume.
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "portable"))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["HF_HOME"] == str(home / ".cache" / "huggingface")


def test_unsloth_portable_alone_enables_portable_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSLOTH_PORTABLE", "1")
    sr = _load_storage_roots()

    assert sr.portable_mode() is True
    sr._setup_cache_env()

    assert os.environ["TORCH_HOME"].startswith(str(tmp_path / "studio"))


@pytest.mark.parametrize(
    "value", ("0", "false", "False", "FALSE", "off", "OFF", "no", "No", " false ", ""),
)
def test_unsloth_portable_off_values_do_not_enable_portable_mode(monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_PORTABLE", value)
    sr = _load_storage_roots()

    assert sr.portable_mode() is False


def test_explicit_env_beats_the_pinned_default(monkeypatch, tmp_path):
    chosen = tmp_path / "elsewhere" / "inductor"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(chosen))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(chosen)


def test_blank_inherited_value_counts_as_unset(monkeypatch, tmp_path):
    # "" would send the library to the working directory or the system temp dir.
    monkeypatch.setenv("MPLCONFIGDIR", "   ")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["MPLCONFIGDIR"] == str(tmp_path / "studio" / "cache" / "matplotlib")


def test_studio_home_outranks_unsloth_home(monkeypatch, tmp_path):
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(explicit))
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "portable"))
    sr = _load_storage_roots()

    assert sr.studio_root() == explicit.resolve()


def test_data_designer_home_is_set_before_the_library_would_read_it(tmp_path):
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["DATA_DESIGNER_MANAGED_ASSETS_PATH"] == str(
        tmp_path / "studio" / "data-designer" / "managed-assets"
    )


def test_an_existing_data_designer_home_is_left_where_it_is(tmp_path):
    legacy = tmp_path / "home" / ".data-designer"
    (legacy / "managed-assets").mkdir(parents = True)
    (legacy / "model_configs.yaml").write_text("models: []\n", encoding = "utf-8")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert "DATA_DESIGNER_HOME" not in os.environ
    assert "DATA_DESIGNER_MANAGED_ASSETS_PATH" not in os.environ


def test_managed_assets_follow_an_explicit_data_designer_home(monkeypatch, tmp_path):
    chosen = tmp_path / "mine" / ".data-designer"
    monkeypatch.setenv("DATA_DESIGNER_HOME", str(chosen))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["DATA_DESIGNER_HOME"] == str(chosen)
    assert "DATA_DESIGNER_MANAGED_ASSETS_PATH" not in os.environ


def test_an_explicit_triton_home_keeps_its_own_cache_dir(monkeypatch, tmp_path):
    chosen = tmp_path / "mine" / "triton-home"
    monkeypatch.setenv("TRITON_HOME", str(chosen))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["TRITON_HOME"] == str(chosen)
    assert "TRITON_CACHE_DIR" not in os.environ


def _matplotlib_config_dir(home: Path) -> Path:
    # matplotlib.__init__._get_config_or_cache_dir, Linux branch. The Windows and
    # macOS branches are exercised by _matplotlib_config_dir itself.
    return home / ".config" / "matplotlib"


def test_a_user_matplotlibrc_keeps_matplotlibs_own_config_dir(tmp_path):
    # MPLCONFIGDIR moves the config dir as well as the cache, so pinning it here
    # would drop the file silently and redraw every loss plot differently.
    config = _matplotlib_config_dir(tmp_path / "home")
    config.mkdir(parents = True)
    (config / "matplotlibrc").write_text("figure.dpi: 222\n", encoding = "utf-8")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert "MPLCONFIGDIR" not in os.environ


def test_a_user_style_library_keeps_matplotlibs_own_config_dir(tmp_path):
    styles = _matplotlib_config_dir(tmp_path / "home") / "stylelib"
    styles.mkdir(parents = True)
    (styles / "house.mplstyle").write_text("axes.facecolor: black\n", encoding = "utf-8")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert "MPLCONFIGDIR" not in os.environ


def test_an_empty_matplotlib_config_dir_is_still_pinned(tmp_path):
    # matplotlib mkdir -p's this on every import, so treating its existence as
    # user configuration would give up containment for nearly every install.
    _matplotlib_config_dir(tmp_path / "home").mkdir(parents = True)
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["MPLCONFIGDIR"] == str(tmp_path / "studio" / "cache" / "matplotlib")


@pytest.mark.skipif(
    not sys.platform.startswith(("linux", "freebsd")),
    reason = "XDG config base is the Linux/FreeBSD branch",
)
def test_the_matplotlib_config_dir_follows_xdg_config_home(monkeypatch, tmp_path):
    config = tmp_path / "xdg" / "matplotlib"
    config.mkdir(parents = True)
    (config / "matplotlibrc").write_text("figure.dpi: 222\n", encoding = "utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert "MPLCONFIGDIR" not in os.environ


def test_matplotlib_reads_the_config_the_pin_would_have_hidden(tmp_path):
    pytest.importorskip("matplotlib")
    config = _matplotlib_config_dir(tmp_path / "home")
    (config / "stylelib").mkdir(parents = True)
    (config / "matplotlibrc").write_text("figure.dpi: 222\n", encoding = "utf-8")
    (config / "stylelib" / "house.mplstyle").write_text("axes.facecolor: black\n", encoding = "utf-8")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    # A fresh interpreter: matplotlib caches the config dir on first read.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, matplotlib, matplotlib.style;"
            "print(json.dumps({'rc': matplotlib.matplotlib_fname(),"
            "'dpi': matplotlib.rcParams['figure.dpi'],"
            "'style': 'house' in matplotlib.style.available}))",
        ],
        env = dict(os.environ),
        capture_output = True,
        text = True,
        check = True,
    )
    result = json.loads(probe.stdout.strip().splitlines()[-1])

    assert result["rc"] == str(config / "matplotlibrc")
    assert result["dpi"] == 222.0
    assert result["style"] is True
