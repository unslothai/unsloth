# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Everything Unsloth writes should sit under one directory (issue #8865).

_setup_cache_env() pins the regenerable, process-scoped caches in every mode --
torch inductor, triton, matplotlib, numba, the CUDA JIT cache, and the
DataDesigner home, which the library reads at import time off Path.home().

The caches holding shared user data move only in portable mode. A default
install must keep the HF hub cache where the rest of the ecosystem looks, or
models fetched before Unsloth (and models shared with LM Studio / Ollama) get
downloaded a second time -- the reason the hub cache was taken out of the
Unsloth tree in the first place, see storage_roots.legacy_hf_cache_dir.
"""

import contextlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_STORAGE_ROOTS_PATH = Path(__file__).resolve().parent.parent / "utils/paths/storage_roots.py"

# Pinned in every mode: regenerable, and scoped to this process either way.
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
_PORTABLE_ONLY = ("HF_DATASETS_CACHE", "TORCH_HOME")

_HF_ENV = ("HF_HOME", "HF_HUB_CACHE", "HF_XET_CACHE", "HUGGINGFACE_HUB_CACHE")


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch, tmp_path):
    for key in _ALWAYS_PINNED + _PORTABLE_ONLY + _HF_ENV:
        monkeypatch.delenv(key, raising = False)
    for key in ("UNSLOTH_HOME", "UNSLOTH_PORTABLE", "STUDIO_HOME"):
        monkeypatch.delenv(key, raising = False)
    # hf_cache_settings._default_cache_home reads this before ~/.cache, and CI
    # runners set it, so a test asserting on the home default has to clear it.
    monkeypatch.delenv("XDG_CACHE_HOME", raising = False)
    # A developer machine may hold a real ~/.data-designer, which the resolver
    # reads before it pins a home of its own. Point Path.home() at an empty
    # directory so these assert on the code and not on who ran them.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))


@pytest.fixture(autouse = True)
def _restore_hf_cache_settings_module():
    # _load_storage_roots pops the resolver so each test models a fresh process.
    # Left popped, the next import builds a second module object and rebinds it
    # on the utils package, so a later test writes into one and reads the other.
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
    # The cache resolver snapshots explicit environment variables once per
    # process, so it has to be rebuilt per test.
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

    # Models downloaded before Unsloth, or shared with other local LLM tools,
    # must stay discoverable.
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
    # HF_HOME owns the token path. Credentials must not follow a cache onto a
    # removable volume just because the caches did.
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "portable"))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["HF_HOME"] == str(home / ".cache" / "huggingface")


def test_unsloth_portable_alone_enables_portable_mode(monkeypatch, tmp_path):
    # An existing UNSLOTH_STUDIO_HOME install can opt in without moving.
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
    # An inherited "KEY=" would otherwise pin the cache to "", which sends the
    # library to the working directory or the system temp dir.
    monkeypatch.setenv("MPLCONFIGDIR", "   ")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["MPLCONFIGDIR"] == str(tmp_path / "studio" / "cache" / "matplotlib")


def test_studio_home_outranks_unsloth_home(monkeypatch, tmp_path):
    # UNSLOTH_STUDIO_HOME names this exact directory; UNSLOTH_HOME only names
    # the tree it would sit in.
    explicit = tmp_path / "explicit"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(explicit))
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "portable"))
    sr = _load_storage_roots()

    assert sr.studio_root() == explicit.resolve()


def test_data_designer_home_is_set_before_the_library_would_read_it(tmp_path):
    # data_designer.config.utils.constants builds DATA_DESIGNER_HOME at import
    # time from Path.home() / ".data-designer" unless the variable is present,
    # and the Data Recipes worker is spawned, so it copies this environment.
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["DATA_DESIGNER_MANAGED_ASSETS_PATH"] == str(
        tmp_path / "studio" / "data-designer" / "managed-assets"
    )


def test_an_existing_data_designer_home_is_left_where_it_is(tmp_path):
    # DATA_DESIGNER_HOME holds model_configs.yaml, model_providers.yaml and the
    # multi-GB persona parquet under managed-assets. Repointing it does not move
    # that state, it hides it behind a freshly seeded default.
    legacy = tmp_path / "home" / ".data-designer"
    (legacy / "managed-assets").mkdir(parents = True)
    (legacy / "model_configs.yaml").write_text("models: []\n", encoding = "utf-8")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert "DATA_DESIGNER_HOME" not in os.environ
    assert "DATA_DESIGNER_MANAGED_ASSETS_PATH" not in os.environ


def test_managed_assets_follow_an_explicit_data_designer_home(monkeypatch, tmp_path):
    # The library derives <DATA_DESIGNER_HOME>/managed-assets itself, and its CLI
    # downloader writes there unconditionally. Forcing the assets path against
    # someone else's home splits the writer from the reader.
    chosen = tmp_path / "mine" / ".data-designer"
    monkeypatch.setenv("DATA_DESIGNER_HOME", str(chosen))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["DATA_DESIGNER_HOME"] == str(chosen)
    assert "DATA_DESIGNER_MANAGED_ASSETS_PATH" not in os.environ


def test_an_explicit_triton_home_keeps_its_own_cache_dir(monkeypatch, tmp_path):
    # triton/knobs.py: TRITON_CACHE_DIR outranks the <TRITON_HOME>/.triton/cache
    # derivation, so pinning it would move the kernels away from the dump and
    # override dirs that stay under the user's TRITON_HOME.
    chosen = tmp_path / "mine" / "triton-home"
    monkeypatch.setenv("TRITON_HOME", str(chosen))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["TRITON_HOME"] == str(chosen)
    assert "TRITON_CACHE_DIR" not in os.environ
