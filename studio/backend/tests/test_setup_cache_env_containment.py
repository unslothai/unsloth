# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Everything Unsloth writes should sit under one directory (issue #8865)."""

import contextlib
import errno
import importlib.util
import json
import os
import platform
import re
import shutil
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
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
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
    # Not pinned, but both change what Triton resolves under the test home.
    for key in ("TRITON_HOME", "TRITON_OVERRIDE_DIR"):
        monkeypatch.delenv(key, raising = False)
    # _default_cache_home reads this before ~/.cache, and CI runners set it.
    monkeypatch.delenv("XDG_CACHE_HOME", raising = False)
    # Same for matplotlib's config dir, whose contents decide whether MPLCONFIGDIR is ours.
    monkeypatch.delenv("XDG_CONFIG_HOME", raising = False)
    # Empty home: a real ~/.data-designer would change what the resolver pins.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))


@pytest.fixture(autouse = True)
def _restore_hf_cache_settings_module():
    # Left popped, the next import builds a second module object, so a later test writes one
    # and reads the other.
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
    "value",
    ("0", "false", "False", "FALSE", "off", "OFF", "no", "No", " false ", ""),
)
def test_unsloth_portable_off_values_do_not_enable_portable_mode(monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_PORTABLE", value)
    sr = _load_storage_roots()

    assert sr.portable_mode() is False


# Every spelling the installers refuse. "enabled" and "flase" are the shapes that matter: an
# intent the shell never acted on, and a typo.
_UNRECOGNIZED_PORTABLE = ("enabled", "flase", "2", "bogus", "y", "n", "disabled", "-1")


@pytest.mark.parametrize("value", _UNRECOGNIZED_PORTABLE)
def test_unrecognized_unsloth_portable_does_not_enable_portable_mode(monkeypatch, tmp_path, value):
    # A value the installers would have rejected must not put the runtime in portable mode on
    # its own: the caches would move for this launch and move back on the next one.
    monkeypatch.setenv("UNSLOTH_PORTABLE", value)
    sr = _load_storage_roots()

    assert sr.portable_mode() is False
    sr._setup_cache_env()

    for key in _PORTABLE_ONLY:
        assert key not in os.environ, f"{key} was redirected by UNSLOTH_PORTABLE={value!r}"


@pytest.mark.parametrize("value", _UNRECOGNIZED_PORTABLE + ("0", "false", "off", "no"))
def test_a_portable_root_stays_portable_whatever_unsloth_portable_says(
    monkeypatch, tmp_path, value
):
    # The root is what makes an install portable: neither an unrecognized value nor an off one
    # may strand its caches back in the home directory.
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    monkeypatch.setenv("UNSLOTH_PORTABLE", value)
    sr = _load_storage_roots()

    assert sr.portable_mode() is True
    assert sr.studio_root() == master / "studio"
    sr._setup_cache_env()

    for key in _PORTABLE_ONLY:
        assert os.environ[key].startswith(str(master)), f"{key} escaped the portable root"


@pytest.mark.parametrize("value", _UNRECOGNIZED_PORTABLE)
def test_an_on_disk_portable_root_outranks_an_unrecognized_value(monkeypatch, value):
    # unsloth_home() also resolves from the marker install.sh leaves at the root, the signal a
    # venv-activated launch carries no environment for.
    monkeypatch.setenv("UNSLOTH_PORTABLE", value)
    sr = _load_storage_roots()
    monkeypatch.setattr(sr, "unsloth_home", lambda: Path("/opt/unsloth-portable"))

    assert sr.portable_mode() is True


class _RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args):
        self.warnings.append(message % args if args else message)


def test_an_unrecognized_value_is_reported_once_not_once_per_call(monkeypatch):
    # portable_mode() runs on every cache-var lookup, so a per-call warning floods the log.
    monkeypatch.setenv("UNSLOTH_PORTABLE", "enabled")
    sr = _load_storage_roots()
    recorder = _RecordingLogger()
    monkeypatch.setattr(sr, "logger", recorder)

    for _ in range(200):
        assert sr.portable_mode() is False

    assert len(recorder.warnings) == 1
    warning = recorder.warnings[0]
    assert "enabled" in warning
    # Naming only the rejection leaves the user guessing at the spelling.
    for accepted in ("1", "true", "yes", "on", "0", "false", "off", "no"):
        assert accepted in warning


@pytest.mark.parametrize("value", ("1", "true", "TRUE", " on ", "0", "false", "off", "no", ""))
def test_accepted_spellings_are_silent(monkeypatch, value):
    monkeypatch.setenv("UNSLOTH_PORTABLE", value)
    sr = _load_storage_roots()
    recorder = _RecordingLogger()
    monkeypatch.setattr(sr, "logger", recorder)

    sr.portable_mode()

    assert recorder.warnings == []


def test_conflicting_roots_are_reported_once_per_conflict(monkeypatch, tmp_path):
    # studio_root() runs many times per request, so a per-call warning turns one configuration
    # mistake into a flooded log plus synchronous log I/O for the life of the backend.
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "elsewhere" / "studio"))
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "portable"))
    sr = _load_storage_roots()
    recorder = _RecordingLogger()
    monkeypatch.setattr(sr, "logger", recorder)

    for _ in range(200):
        sr.studio_root()

    assert len(recorder.warnings) == 1
    assert str(tmp_path / "elsewhere" / "studio") in recorder.warnings[0]
    assert str(tmp_path / "portable") in recorder.warnings[0]

    # A different pair of roots is a different mistake and must still be heard.
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "other-portable"))
    for _ in range(200):
        sr.studio_root()

    assert len(recorder.warnings) == 2
    assert str(tmp_path / "other-portable") in recorder.warnings[1]


@pytest.mark.parametrize("layout", ("nested", "flat"))
def test_a_self_contained_layout_never_warns(monkeypatch, tmp_path, layout):
    # Both supported shapes: studio/ under the master root, and one root naming itself.
    # Silencing the repeat must not silence the whole diagnostic.
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    monkeypatch.setenv(
        "UNSLOTH_STUDIO_HOME",
        str(master / "studio") if layout == "nested" else str(master),
    )
    sr = _load_storage_roots()
    recorder = _RecordingLogger()
    monkeypatch.setattr(sr, "logger", recorder)

    for _ in range(50):
        sr.studio_root()

    assert recorder.warnings == []


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


def test_portable_mode_keeps_project_workspaces_inside_the_root(monkeypatch, tmp_path):
    monkeypatch.delenv("UNSLOTH_STUDIO_PROJECTS_HOME", raising = False)
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert str(sr.project_workspaces_root()).startswith(str(master))
    assert not str(sr.documents_root()).startswith(str(master))


def test_default_install_leaves_project_workspaces_in_documents(monkeypatch, tmp_path):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("UNSLOTH_STUDIO_PROJECTS_HOME", raising = False)
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert sr.project_workspaces_root() == home / "Documents" / "Unsloth Studio" / "Projects"


def test_an_explicit_projects_home_beats_portable_mode(monkeypatch, tmp_path):
    chosen = tmp_path / "my-projects"
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(chosen))
    monkeypatch.delenv("UNSLOTH_STUDIO_HOME", raising = False)
    monkeypatch.setenv("UNSLOTH_HOME", str(tmp_path / "portable"))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert sr.project_workspaces_root() == chosen


def test_the_on_disk_marker_finds_the_root_without_any_environment(monkeypatch, tmp_path):
    # `source .../activate; unsloth studio` reaches the venv binary past the
    # shim that exports UNSLOTH_HOME.
    master = tmp_path / "portable"
    studio = master / "studio"
    studio.mkdir(parents = True)
    (master / ".unsloth-portable-root").write_text(str(master), encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio))
    sr = _load_storage_roots()

    assert sr.unsloth_home() == master
    assert sr.portable_mode() is True
    sr._setup_cache_env()
    assert os.environ["TORCH_HOME"].startswith(str(master))


def test_the_marker_also_works_when_the_root_is_the_studio_root(monkeypatch, tmp_path):
    root = tmp_path / "flat"
    root.mkdir()
    (root / ".unsloth-portable-root").write_text(str(root), encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(root))
    sr = _load_storage_roots()

    assert sr.unsloth_home() == root
    assert sr.portable_mode() is True


def test_no_marker_means_no_portable_mode(monkeypatch, tmp_path):
    # Upgrading a plain install would move its HF cache out from under it.
    studio = tmp_path / "plain"
    studio.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio))
    sr = _load_storage_roots()

    assert sr.unsloth_home() is None
    assert sr.portable_mode() is False


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


def _use_data_designer(home: Path) -> None:
    """Write what a Studio Data Designer session leaves behind."""
    (home / "managed-assets").mkdir(parents = True, exist_ok = True)
    (home / "model_configs.yaml").write_text("models: []\n", encoding = "utf-8")
    (home / "managed-assets" / "seeds.parquet").write_bytes(b"PAR1")


def test_a_used_managed_home_survives_a_legacy_dir_appearing_later(tmp_path):
    # The legacy probe re-runs every launch, so without this a standalone Data Designer run
    # creating ~/.data-designer hands the work under the Studio root to a re-seeded default.
    managed = tmp_path / "studio" / "data-designer"
    _use_data_designer(managed)
    (tmp_path / "home" / ".data-designer").mkdir(parents = True)

    sr = _load_storage_roots()
    sr._setup_cache_env()

    assert os.environ["DATA_DESIGNER_HOME"] == str(managed)
    assert os.environ["DATA_DESIGNER_MANAGED_ASSETS_PATH"] == str(managed / "managed-assets")


def test_a_used_managed_home_does_not_flip_when_the_legacy_dir_is_deleted(tmp_path):
    # Deleting and recreating ~/.data-designer used to toggle the active home, so which dataset
    # a run reads depended on whether that directory existed.
    managed = tmp_path / "studio" / "data-designer"
    _use_data_designer(managed)
    legacy = tmp_path / "home" / ".data-designer"

    seen = []
    for exists in (False, True, False, True):
        if exists:
            legacy.mkdir(parents = True, exist_ok = True)
        elif legacy.exists():
            shutil.rmtree(legacy)
        for key in ("DATA_DESIGNER_HOME", "DATA_DESIGNER_MANAGED_ASSETS_PATH"):
            os.environ.pop(key, None)
        sr = _load_storage_roots()
        sr._setup_cache_env()
        seen.append(os.environ.get("DATA_DESIGNER_HOME"))

    assert seen == [str(managed)] * 4


def test_an_unused_managed_home_still_defers_to_a_legacy_dir(tmp_path):
    # _setup_cache_env creates the managed home on the first launch, so its mere existence must
    # not claim a user's data. Nothing is stranded while it is empty.
    (tmp_path / "studio" / "data-designer" / "managed-assets").mkdir(parents = True)
    (tmp_path / "home" / ".data-designer").mkdir(parents = True)

    sr = _load_storage_roots()
    sr._setup_cache_env()

    assert "DATA_DESIGNER_HOME" not in os.environ
    assert "DATA_DESIGNER_MANAGED_ASSETS_PATH" not in os.environ


def test_an_unreadable_managed_home_keeps_its_pin(monkeypatch, tmp_path):
    # An inspection failure is not evidence of an empty home: reading it as one drops the pin
    # and runs against ~/.data-designer, hiding the recipes and datasets under the managed tree.
    managed = tmp_path / "studio" / "data-designer"
    _use_data_designer(managed)
    (tmp_path / "home" / ".data-designer").mkdir(parents = True)
    sr = _load_storage_roots()

    real_iterdir = Path.iterdir

    def deny(self):
        if self in (managed, managed / "managed-assets"):
            raise PermissionError(13, "Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", deny)
    sr._setup_cache_env()

    assert os.environ["DATA_DESIGNER_HOME"] == str(managed)
    assert os.environ["DATA_DESIGNER_MANAGED_ASSETS_PATH"] == str(managed / "managed-assets")


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason = "chmod 000 denies neither root nor Windows",
)
def test_an_unreadable_managed_assets_child_keeps_its_pin(tmp_path):
    # The same flip through the child: the home lists fine, and the walk trips on managed-assets
    # before it reaches the model_configs.yaml beside it.
    managed = tmp_path / "studio" / "data-designer"
    _use_data_designer(managed)
    (tmp_path / "home" / ".data-designer").mkdir(parents = True)
    sr = _load_storage_roots()

    assets = managed / "managed-assets"
    assets.chmod(0o000)
    try:
        sr._setup_cache_env()
    finally:
        assets.chmod(0o755)

    assert os.environ["DATA_DESIGNER_HOME"] == str(managed)
    assert os.environ["DATA_DESIGNER_MANAGED_ASSETS_PATH"] == str(assets)


def _fail_stat_on(monkeypatch, target: Path, error: OSError) -> None:
    """Make every stat of *target* raise *error*, and leave every other path alone.

    chmod covers EACCES; a failing mount answering EIO has no on-disk equivalent an unprivileged
    user can set up. Both os.stat and os.lstat are patched so the predicates this replaced see
    the same failure the fix does.
    """

    def denying(real):
        def deny(path, *args, **kwargs):
            if isinstance(path, (str, os.PathLike)) and Path(path) == target:
                raise error
            return real(path, *args, **kwargs)

        return deny

    for name in ("stat", "lstat"):
        monkeypatch.setattr(os, name, denying(getattr(os, name)))


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason = "chmod 000 denies neither root nor Windows",
)
def test_an_unreadable_legacy_data_designer_dir_keeps_the_home_unpinned(tmp_path):
    # Path.exists() cannot answer this: it raises for EACCES up to 3.13 and swallows it from
    # 3.14 on, and both readings ended at the pin, hiding the user's recipes and parquet.
    home = tmp_path / "home"
    _use_data_designer(home / ".data-designer")
    sr = _load_storage_roots()

    home.chmod(0o000)
    try:
        sr._setup_cache_env()
    finally:
        home.chmod(0o755)

    assert "DATA_DESIGNER_HOME" not in os.environ
    assert "DATA_DESIGNER_MANAGED_ASSETS_PATH" not in os.environ


def test_a_legacy_data_designer_dir_on_a_failing_volume_keeps_the_home_unpinned(
    monkeypatch, tmp_path
):
    legacy = tmp_path / "home" / ".data-designer"
    _use_data_designer(legacy)
    sr = _load_storage_roots()
    _fail_stat_on(monkeypatch, legacy, OSError(errno.EIO, "Input/output error"))

    sr._setup_cache_env()

    assert "DATA_DESIGNER_HOME" not in os.environ
    assert "DATA_DESIGNER_MANAGED_ASSETS_PATH" not in os.environ


def test_a_legacy_data_designer_symlink_we_cannot_follow_keeps_the_home_unpinned(tmp_path):
    # A loop answers ELOOP, which Path.exists() reports as absence on every release, so the
    # exception handler never ran and the pin was taken anyway.
    legacy = tmp_path / "home" / ".data-designer"
    legacy.parent.mkdir(parents = True, exist_ok = True)
    legacy.symlink_to(legacy)
    sr = _load_storage_roots()

    assert Path(legacy).exists() is False
    sr._setup_cache_env()

    assert "DATA_DESIGNER_HOME" not in os.environ


def test_an_absent_legacy_data_designer_dir_still_pins_the_home(tmp_path):
    # The inverse direction: hardening the probe must not turn into never containing anything.
    # Nothing is at ~/.data-designer here, so the pin is ours to take.
    (tmp_path / "home").mkdir(parents = True, exist_ok = True)
    sr = _load_storage_roots()

    sr._setup_cache_env()

    managed = tmp_path / "studio" / "data-designer"
    assert os.environ["DATA_DESIGNER_HOME"] == str(managed)
    assert os.environ["DATA_DESIGNER_MANAGED_ASSETS_PATH"] == str(managed / "managed-assets")


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


def test_triton_keeps_reading_the_default_override_dir(tmp_path):
    # Kernel overrides are user files, and TRITON_HOME would move their directory along with the
    # cache, so a TRITON_KERNEL_OVERRIDE=1 run would compile something else instead.
    pytest.importorskip("triton")
    override = tmp_path / "home" / ".triton" / "override" / "0123456789abcdef"
    override.mkdir(parents = True)
    (override / "kernel.ttir").write_text("// hand-tuned\n", encoding = "utf-8")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    # A fresh interpreter: torch may have imported Triton and pinned a cache dir already.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from triton import knobs;"
            "print(json.dumps({'cache': knobs.cache.dir,"
            "'dump': knobs.cache.dump_dir,"
            "'override': knobs.cache.override_dir}))",
        ],
        env = dict(os.environ),
        capture_output = True,
        text = True,
        check = True,
    )
    result = json.loads(probe.stdout.strip().splitlines()[-1])

    assert Path(result["override"]) == override.parent
    assert (Path(result["override"]) / override.name / "kernel.ttir").is_file()
    assert result["cache"] == str(tmp_path / "studio" / "cache" / "triton")
    assert result["dump"] == str(tmp_path / "studio" / "cache" / "triton-dump")


def test_the_macos_matplotlib_config_dir_matches_matplotlibs_own(monkeypatch, tmp_path):
    # macOS is matplotlib's "other platforms" branch, ~/.matplotlib and not ~/Library/Application
    # Support. Getting it wrong strands a real matplotlibrc or gives up containment for nothing.
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(sys, "platform", "darwin")
    sr = _load_storage_roots()

    ours = sr._matplotlib_config_dir()

    # sys.platform is read inside _get_config_or_cache_dir, so a fresh interpreter can be asked
    # what it would do on a Mac.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, matplotlib; sys.platform = 'darwin';"
            "print(matplotlib._get_config_or_cache_dir(matplotlib._get_xdg_config_dir))",
        ],
        env = {**os.environ, "HOME": str(tmp_path / "home"), "MPLCONFIGDIR": ""},
        capture_output = True,
        text = True,
        check = True,
    )
    theirs = Path(probe.stdout.strip().splitlines()[-1])

    assert ours is not None
    assert ours.resolve() == theirs.resolve()


def _matplotlib_config_dir(home: Path) -> Path:
    # _get_config_or_cache_dir, Linux branch. The other branches are covered above.
    return home / ".config" / "matplotlib"


def test_a_user_matplotlibrc_keeps_matplotlibs_own_config_dir(tmp_path):
    # MPLCONFIGDIR moves the config dir as well as the cache, so pinning it here would drop the
    # file silently and redraw every loss plot differently.
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
    # matplotlib mkdir -p's this on every import, so treating its existence as user configuration
    # would give up containment for nearly every install.
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


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason = "chmod 000 denies neither root nor Windows",
)
def test_an_uninspectable_matplotlib_config_dir_leaves_mplconfigdir_unset(tmp_path):
    # MPLCONFIGDIR lives for the whole process, so reading an unreadable config dir as an empty
    # one hides the matplotlibrc even once the mount recovers.
    config = _matplotlib_config_dir(tmp_path / "home")
    config.mkdir(parents = True)
    (config / "matplotlibrc").write_text("figure.dpi: 222\n", encoding = "utf-8")
    sr = _load_storage_roots()

    config.parent.chmod(0o000)
    try:
        sr._setup_cache_env()
    finally:
        config.parent.chmod(0o755)

    assert "MPLCONFIGDIR" not in os.environ


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason = "chmod 000 denies neither root nor Windows",
)
def test_an_uninspectable_style_library_leaves_mplconfigdir_unset(tmp_path):
    # Path.glob suppresses the scandir error and yields nothing on every release we support, so
    # this probe never reached its handler: an unreadable stylelib read as an empty one and every
    # custom style went missing from the loss plots.
    styles = _matplotlib_config_dir(tmp_path / "home") / "stylelib"
    styles.mkdir(parents = True)
    (styles / "house.mplstyle").write_text("axes.facecolor: black\n", encoding = "utf-8")
    sr = _load_storage_roots()

    styles.chmod(0o000)
    try:
        assert list(styles.glob("*.mplstyle")) == []
        sr._setup_cache_env()
    finally:
        styles.chmod(0o755)

    assert "MPLCONFIGDIR" not in os.environ


def test_a_matplotlib_config_dir_on_a_failing_volume_leaves_mplconfigdir_unset(
    monkeypatch, tmp_path
):
    config = _matplotlib_config_dir(tmp_path / "home")
    config.mkdir(parents = True)
    (config / "matplotlibrc").write_text("figure.dpi: 222\n", encoding = "utf-8")
    sr = _load_storage_roots()
    _fail_stat_on(monkeypatch, config / "matplotlibrc", OSError(errno.EIO, "Input/output error"))

    sr._setup_cache_env()

    assert "MPLCONFIGDIR" not in os.environ


@pytest.mark.skipif(
    not sys.platform.startswith(("linux", "freebsd")),
    reason = "XDG config base is the Linux/FreeBSD branch",
)
def test_an_xdg_config_dir_is_read_without_a_resolvable_home(monkeypatch, tmp_path):
    # _get_xdg_config_dir reads XDG_CONFIG_HOME before it needs a home, so bailing out on
    # Path.home() first pinned over a real matplotlibrc.
    config = tmp_path / "xdg" / "matplotlib"
    config.mkdir(parents = True)
    (config / "matplotlibrc").write_text("figure.dpi: 222\n", encoding = "utf-8")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    sr = _load_storage_roots()

    def no_home():
        raise RuntimeError("Could not determine home directory.")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))

    assert sr._matplotlib_defaults(tmp_path / "studio" / "cache") == {}


def test_only_a_positive_absence_counts_as_nothing_at_a_path(tmp_path):
    # The one rule every "pin only when there is nothing to strand" probe shares.
    sr = _load_storage_roots()
    present = tmp_path / "present"
    present.write_text("x", encoding = "utf-8")
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    dangling = tmp_path / "dangling"
    dangling.symlink_to(tmp_path / "not-mounted-yet" / "data")

    assert sr._nothing_at(tmp_path / "absent") is True
    assert sr._nothing_at(present) is False
    # Both of these answer False through Path.exists(), which is the bug.
    assert Path(loop).exists() is False and sr._nothing_at(loop) is False
    assert Path(dangling).exists() is False and sr._nothing_at(dangling) is False


def test_nothing_at_reports_an_uninspectable_directory_as_holding_something(monkeypatch, tmp_path):
    sr = _load_storage_roots()
    styles = tmp_path / "stylelib"
    styles.mkdir()
    real_scandir = os.scandir

    def deny(
        path = ".",
        *args,
        **kwargs,
    ):
        if isinstance(path, (str, os.PathLike)) and Path(path) == styles:
            raise OSError(errno.EIO, "Input/output error")
        return real_scandir(path, *args, **kwargs)

    monkeypatch.setattr(os, "scandir", deny)

    assert sr._nothing_at(styles, ending = ".mplstyle") is False
    assert sr._nothing_at(tmp_path / "absent", ending = ".mplstyle") is True


def _fake_torch_on_path(
    tmp_path,
    label,
    version,
    cuda = None,
    hip = None,
    debug = False,
):
    """A torch that has a version.py and explodes if anything imports it. version.py is written
    the way recent torch generates it, annotations and all, so the parser meets the real shape."""
    pkg = tmp_path / label / "torch"
    pkg.mkdir(parents = True)
    (pkg / "__init__.py").write_text("raise AssertionError('torch was imported')\n")
    (pkg / "version.py").write_text(
        "from typing import Optional\n\n"
        f"__version__ = {version!r}\n"
        f"debug = {debug!r}\n"
        f"cuda: Optional[str] = {cuda!r}\n"
        "git_version = '5811a8d7da873dd699ff6687092c225caffcf1bb'\n"
        f"hip: Optional[str] = {hip!r}\n"
        "xpu: Optional[str] = None\n"
    )
    return str(tmp_path / label)


@contextlib.contextmanager
def _only_torch(entry):
    saved_path = list(sys.path)
    saved_module = sys.modules.pop("torch", None)
    sys.path.insert(0, entry)
    importlib.invalidate_caches()
    try:
        yield
    finally:
        sys.path[:] = saved_path
        if saved_module is not None:
            sys.modules["torch"] = saved_module
        importlib.invalidate_caches()


def test_torch_extension_cache_keeps_an_abi_folder(tmp_path):
    # torch appends py<ver>_<accelerator> to its DEFAULT root only, so a pinned
    # TORCH_EXTENSIONS_DIR has to carry that isolation itself.
    sr = _load_storage_roots()

    sr._setup_cache_env()

    pinned = Path(os.environ["TORCH_EXTENSIONS_DIR"])
    assert str(pinned).startswith(str(tmp_path / "studio"))
    assert pinned.parent.name == "torch-extensions"
    assert pinned.name != "torch-extensions", "extension cache is shared across runtimes"
    assert pinned.name.startswith(f"py{sys.version_info.major}{sys.version_info.minor}")


def _interpreter_prefix() -> str:
    """The interpreter-and-ABI part of the tag, spelled out here rather than imported.

    Reusing storage_roots' own helper would make every assertion below agree with whatever
    it happens to produce, so the exact-string tests would stop being able to catch a change
    in the tag at all.
    """
    abi = getattr(sys, "abiflags", "")
    host = re.sub(r"[^A-Za-z0-9.]+", "-", f"{sys.platform}-{platform.machine() or 'unknown'}")
    return f"py{sys.version_info.major}{sys.version_info.minor}{abi}_{host}"


def test_torch_extension_cache_separates_incompatible_builds(tmp_path):
    sr = _load_storage_roots()

    tags = []
    for label, version, cuda in (("a", "2.9.1+cu128", "12.8"), ("b", "2.9.1+cu126", "12.6")):
        with _only_torch(_fake_torch_on_path(tmp_path, label, version, cuda = cuda)):
            tags.append(sr._torch_runtime_tag())

    assert tags[0] != tags[1], f"two torch builds shared one extension dir: {tags[0]}"
    assert "cu128" in tags[0] and "cu126" in tags[1]
    # Path-safe: no '+' or other separators survive into the directory name.
    assert all(part.replace(".", "").replace("-", "").replace("_", "").isalnum() for part in tags)


def test_torch_extension_tag_survives_a_missing_torch(tmp_path):
    # First launch, before the venv has torch: still isolated by interpreter.
    sr = _load_storage_roots()
    empty = tmp_path / "empty"
    empty.mkdir()

    with _only_torch(str(empty)):
        saved = list(sys.path)
        try:
            sys.path[:] = [str(empty)]
            tag = sr._torch_runtime_tag()
        finally:
            sys.path[:] = saved

    assert tag == _interpreter_prefix()


def test_torch_extension_cache_separates_builds_sharing_one_version_string(tmp_path):
    # conda-forge sets PYTORCH_BUILD_VERSION to the bare release, so its CPU and CUDA packages
    # of one version share a __version__ and differ only in a conda build string.
    sr = _load_storage_roots()

    tags = []
    for label, cuda in (("cpu", None), ("cu126", "12.6"), ("cu128", "12.8")):
        with _only_torch(_fake_torch_on_path(tmp_path, label, "2.9.1", cuda = cuda)):
            tags.append(sr._torch_runtime_tag())

    assert len(set(tags)) == 3, f"builds with different CUDA ABIs shared one dir: {tags}"
    prefix = _interpreter_prefix()
    assert tags[0] == f"{prefix}_cpu_2.9.1"
    assert tags[1] == f"{prefix}_cu126_2.9.1"
    assert tags[2] == f"{prefix}_cu128_2.9.1"


def test_torch_extension_cache_separates_a_rocm_build_from_a_cpu_build(tmp_path):
    # torch's own folder names a ROCm build 'cpu', since version.cuda is unset there. main
    # prioritises ROCm, so hip has to be read first.
    sr = _load_storage_roots()

    tags = []
    for label, hip in (("cpu", None), ("rocm", "6.4.43484-123eb5128")):
        with _only_torch(_fake_torch_on_path(tmp_path, label, "2.9.1", hip = hip)):
            tags.append(sr._torch_runtime_tag())

    assert tags[0] != tags[1], f"a ROCm build shared the CPU extension dir: {tags[0]}"
    assert "rocm6.4.43484-123eb5128" in tags[1]


def test_torch_extension_cache_separates_two_host_architectures(tmp_path, monkeypatch):
    # An arm64 python and a Rosetta x86_64 python on ONE Mac, sharing ONE $HOME, agree on
    # version_info, abiflags, torch.__version__ and 'cpu'. torch's own py<ver>_<cu_str> folder
    # gives them the same directory, ninja reads the other one's build as up to date, and the
    # .so fails to load. Nothing has to be moved between machines for this.
    sr = _load_storage_roots()

    tags = []
    for machine in ("arm64", "x86_64"):
        monkeypatch.setattr(platform, "machine", lambda m = machine: m)
        with _only_torch(_fake_torch_on_path(tmp_path, machine, "2.9.1")):
            tags.append(sr._torch_runtime_tag())

    assert tags[0] != tags[1], f"two host architectures shared one extension dir: {tags[0]}"
    assert "arm64" in tags[0] and "x86-64" in tags[1]


def test_torch_extension_cache_survives_an_unnameable_architecture(tmp_path, monkeypatch):
    # platform.machine() returns "" when the platform cannot answer. The tag has to stay a
    # usable directory name rather than growing an empty segment.
    sr = _load_storage_roots()

    monkeypatch.setattr(platform, "machine", lambda: "")
    with _only_torch(_fake_torch_on_path(tmp_path, "blank", "2.9.1")):
        tag = sr._torch_runtime_tag()

    assert "__" not in tag and not tag.endswith("_")
    assert "unknown" in tag


def test_torch_extension_cache_separates_a_debug_build(tmp_path):
    # A debug build keeps the soname of a release one but not its ABI.
    sr = _load_storage_roots()

    tags = []
    for label, debug in (("release", False), ("debug", True)):
        entry = _fake_torch_on_path(tmp_path, label, "2.9.1+cu128", cuda = "12.8", debug = debug)
        with _only_torch(entry):
            tags.append(sr._torch_runtime_tag())

    assert tags[0] != tags[1], f"a debug build shared the release extension dir: {tags[0]}"
    assert tags[1].endswith("_debug")


def test_torch_runtime_tag_never_imports_torch(tmp_path):
    # This runs before torch exists in a fresh venv, and importing it on the startup path would
    # cost seconds and pull CUDA in. The fake package raises on import, so a skip is a failure.
    sr = _load_storage_roots()
    entry = _fake_torch_on_path(tmp_path, "guard", "2.9.1+cu128", cuda = "12.8")

    with _only_torch(entry):
        tag = sr._torch_runtime_tag()
        assert "torch" not in sys.modules, "torch was imported to build the cache tag"

    assert "cu128" in tag
