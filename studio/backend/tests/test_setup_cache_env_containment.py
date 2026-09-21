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
import shlex
import shutil
import string
import subprocess
import sys
import tempfile
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
    # Left popped, the next import builds a second module object for a later test to read.
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


# Pinned to a path a compiler is handed on a command line: cpp_builder.py joins with spaces,
# interpolates unquoted and reparses with shlex.split, so a root containing a space splits
# mid-path and the C++ build fails outright.
_TOOLCHAIN_PINNED = (
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
    "CUDA_CACHE_PATH",
)


def _assert_no_unparseable_pin(sr, refused_root):
    """Whatever the resolver published for the toolchain keys, a compiler must be able to read
    it, and it must not sit inside the root we just refused.

    The rule is not "unset". torch's own fallback is <gettempdir>/torchinductor_<login> and it
    sanitises only [\\/:*?"<>|], so an o'brien or First Last login lands back on the character
    that caused the refusal. Unset is only acceptable when the temporary directory is no better.
    """
    for key in _TOOLCHAIN_PINNED:
        value = os.environ.get(key)
        if value is None:
            continue
        assert value.strip(), f"{key} was left blank, which Inductor reads as a relative path"
        assert not sr.toolchain_path_unparseable(value), f"{key} was pinned to {value!r}"
        assert not value.startswith(str(refused_root)), f"{key} stayed inside {refused_root}"


def test_a_spaced_root_leaves_the_compiler_caches_to_their_own_defaults(monkeypatch, tmp_path):
    """ "C:\\Users\\First Last" is an ordinary Windows account name, so the DEFAULT Studio root
    contains a space for a large share of installs. Before this file pinned these, Inductor used
    its own temporary directory and the build worked; pinning it into a spaced root broke
    torch.compile outright. The refusal now publishes a parseable directory rather than hoping
    torch's own default is one, since for a First Last login it is not. The rest of the caches,
    which nobody pastes into a command line, still move."""
    spaced = tmp_path / "my home" / "studio"
    spaced.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(spaced))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    _assert_no_unparseable_pin(sr, spaced)
    # Non-vacuity, and the point of the guard being narrow: everything else is still contained.
    for key in ("UV_CACHE_DIR", "NUMBA_CACHE_DIR", "MPLCONFIGDIR", "UNSLOTH_COMPILE_LOCATION"):
        assert os.environ[key].startswith(str(spaced.parent)), key


def test_a_root_without_spaces_still_pins_the_compiler_caches(monkeypatch, tmp_path):
    """The other half of the guard: it must fire on a path a compiler cannot take, not on
    every path."""
    plain = tmp_path / "plain_home" / "studio"
    plain.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(plain))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    for key in _TOOLCHAIN_PINNED:
        assert os.environ.get(key, "").startswith(str(plain)), key


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("o'brien", id = "an apostrophe, which shlex reads as an opening quote"),
        pytest.param('say"hi', id = "a double quote"),
    ],
)
def test_a_quoted_root_leaves_the_compiler_caches_to_their_own_defaults(
    name, monkeypatch, tmp_path
):
    """Whitespace was the only character the guard knew, and a quote is worse than a space: in
    POSIX mode shlex swallows the rest of the command into one argument and deletes the quote,
    so the build fails somewhere less obvious than a split path. "/home/o'brien" is an ordinary
    account name."""
    quoted = tmp_path / name / "studio"
    quoted.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(quoted))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    _assert_no_unparseable_pin(sr, quoted)
    # Non-vacuity: the caches nobody pastes into a command line still move.
    for key in ("UV_CACHE_DIR", "NUMBA_CACHE_DIR", "UNSLOTH_COMPILE_LOCATION"):
        assert os.environ[key].startswith(str(quoted.parent)), key


@pytest.mark.skipif(os.name == "nt", reason = "on Windows the separator is not an escape")
def test_a_backslash_in_a_posix_root_leaves_the_compiler_caches_alone(monkeypatch, tmp_path):
    """Legal in a POSIX filename and an escape to shlex, so the character is eaten and the
    compiler is handed a path that does not exist. Windows is exempt because cpp_builder
    rewrites the separator to "/" before it builds the command."""
    odd = tmp_path / "a\\b" / "studio"
    odd.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(odd))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    _assert_no_unparseable_pin(sr, odd)


@pytest.mark.parametrize(
    "name",
    [
        "plain",
        "my dir",
        "o'brien",
        'say"hi',
        "a\tb",
        pytest.param(
            "a\\b",
            marks = pytest.mark.skipif(
                os.name == "nt",
                reason = "cpp_builder rewrites the separator before building the command",
            ),
        ),
    ],
)
def test_the_guard_agrees_with_what_shlex_actually_does_to_the_command(name, tmp_path):
    """The test that keeps the character list honest. Rather than restating the predicate, this
    builds the command shape cpp_builder builds and asks shlex.split whether the path comes back
    whole, then requires the predicate to have said so. A character added to one and not the
    other fails here."""
    path = str(tmp_path / name / "cache" / "torchinductor")
    command = f"g++ {path}/main.cpp -o {path}/main.so"
    sr = _load_storage_roots()

    try:
        survives = shlex.split(command) == ["g++", f"{path}/main.cpp", "-o", f"{path}/main.so"]
    except ValueError:
        survives = False  # an odd number of quotes raises rather than mangling

    assert sr.toolchain_path_unparseable(path) is not survives, (
        f"{name!r}: predicate says {sr.toolchain_path_unparseable(path)}, "
        f"shlex.split round trip says {survives}"
    )


def test_no_character_at_all_lets_a_mangled_path_through(tmp_path):
    """The invariant that actually protects a user, swept over every printable ASCII character
    plus the whitespace Python recognises and shlex does not.

    The test above pins six characters both ways. This one allows the predicate to be too
    careful and forbids it being not careful enough, which is the only direction that breaks a
    build. It is deliberately one-sided: str.isspace() is true for a no-break space and the
    other Unicode spaces, while shlex splits on ASCII whitespace only, so the predicate refuses
    seven paths a compiler would in fact have accepted. That costs containment for those roots
    and nothing else, it predates this file, and closing it would mean re-deriving shlex's own
    whitespace set here."""
    sr = _load_storage_roots()
    specials = list(string.printable) + [" ", " ", " ", "　", " ", "é"]
    if os.name == "nt":
        specials.remove("\\")  # the separator, rewritten to "/" before the command is built

    leaked = []
    for char in specials:
        path = f"{tmp_path}/unsloth{char}root/cache/torchinductor"
        command = f"g++ {path}/main.cpp -o {path}/main.so"
        try:
            survives = shlex.split(command) == ["g++", f"{path}/main.cpp", "-o", f"{path}/main.so"]
        except ValueError:
            survives = False
        if not survives and not sr.toolchain_path_unparseable(path):
            leaked.append(char)

    assert leaked == [], (
        "these characters would be pinned into a compiler command line that mangles them: "
        + ", ".join(repr(c) for c in leaked)
    )


def test_a_refused_root_gets_a_parseable_cache_rather_than_torchs_own(monkeypatch, tmp_path):
    """Leaving the variable unset is not automatically safe, which is the whole reason this
    branch publishes something.

    torch's default is <gettempdir>/torchinductor_<login>, sanitised against [\\\\/:*?"<>|] only,
    so an o'brien or a First Last login is handed back the character that caused the refusal
    (torch/_inductor/runtime/cache_dir_utils.py::default_cache_dir). The replacement is named
    from a hex digest, so it cannot carry one itself, and it is keyed on the path we wanted, so
    the same install returns to the same cache every launch."""
    refused = tmp_path / "o'brien" / "studio"
    refused.mkdir(parents = True)
    temp_root = tmp_path / "tmp"
    temp_root.mkdir()
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(refused))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(temp_root))
    sr = _load_storage_roots()
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(temp_root))

    sr._setup_cache_env()
    first = os.environ["TORCHINDUCTOR_CACHE_DIR"]

    assert not sr.toolchain_path_unparseable(first)
    assert first.startswith(str(temp_root))
    assert Path(first).is_dir()

    # Stable: a cache that moved every launch would be a cold compile every launch.
    for key in _TOOLCHAIN_PINNED:
        monkeypatch.delenv(key, raising = False)
    sr._setup_cache_env()
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == first


def test_a_refused_root_with_no_usable_temp_root_still_publishes_nothing(monkeypatch, tmp_path):
    """The other end of the same branch. When the temporary directory carries the character too
    there is nowhere left to point, and an unset variable is better than a broken one."""
    refused = tmp_path / "o'brien" / "studio"
    refused.mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(refused))
    sr = _load_storage_roots()
    monkeypatch.setattr(sr.tempfile, "gettempdir", lambda: str(tmp_path / "also'bad"))

    sr._setup_cache_env()

    for key in _TOOLCHAIN_PINNED:
        assert key not in os.environ, key


def test_an_explicit_spaced_compiler_cache_is_left_alone(monkeypatch, tmp_path):
    """Only a default we invented is ours to withhold. A caller who set the variable chose it,
    and silently dropping it would send their cache somewhere they did not ask for."""
    chosen = tmp_path / "their choice"
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(chosen))
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(chosen)


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


# Every spelling the installers refuse: an intent the shell never acted on, and a typo.
_UNRECOGNIZED_PORTABLE = ("enabled", "flase", "2", "bogus", "y", "n", "disabled", "-1")


@pytest.mark.parametrize("value", _UNRECOGNIZED_PORTABLE)
def test_unrecognized_unsloth_portable_does_not_enable_portable_mode(monkeypatch, tmp_path, value):
    # A rejected value must not turn portable mode on: the caches would move for this launch
    # and move back on the next one.
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
    # The root is what makes an install portable; no UNSLOTH_PORTABLE value may veto it.
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
    # unsloth_home() also resolves from install.sh's root marker, which a venv-activated
    # launch has instead of an environment.
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
    # studio_root() runs many times per request, so a per-call warning floods the log and
    # adds synchronous log I/O for the life of the backend.
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
    # Both supported shapes; silencing the repeat must not silence the whole diagnostic.
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


def _use_data_designer(home: Path) -> None:
    """Write what a Studio Data Designer session leaves behind."""
    (home / "managed-assets").mkdir(parents = True, exist_ok = True)
    (home / "model_configs.yaml").write_text("models: []\n", encoding = "utf-8")
    (home / "managed-assets" / "seeds.parquet").write_bytes(b"PAR1")


def test_a_used_managed_home_does_not_flip_when_the_legacy_dir_is_deleted(tmp_path):
    # Deleting and recreating ~/.data-designer used to toggle which home a run read.
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


def _assert_the_resolver_really_ran(tmp_path: Path) -> None:
    """A decline only means something beside a pin that still happened.

    "MPLCONFIGDIR is absent" is equally true when no pinning code exists at all, so on its own
    every decline case passed against a reverted implementation. Naming a sibling that is pinned
    unconditionally makes the whole family falsifiable without a control test per case.
    """
    assert os.environ["TORCHINDUCTOR_CACHE_DIR"].startswith(str(tmp_path / "studio"))
    assert os.environ["NUMBA_CACHE_DIR"].startswith(str(tmp_path / "studio"))


def _mpl_managed(tmp_path: Path) -> Path:
    return tmp_path / "studio" / "cache" / "matplotlib"


def _write_rc(directory: Path) -> None:
    directory.mkdir(parents = True, exist_ok = True)
    (directory / "matplotlibrc").write_text("figure.dpi: 222\n", encoding = "utf-8")


def _write_style(stylelib: Path) -> None:
    stylelib.mkdir(parents = True, exist_ok = True)
    (stylelib / "house.mplstyle").write_text("axes.facecolor: black\n", encoding = "utf-8")


@contextlib.contextmanager
def _denied(*paths: Path):
    """chmod 000 for the duration of the call under test, restored however it ends."""
    for path in paths:
        path.chmod(0o000)
    try:
        yield
    finally:
        for path in paths:
            path.chmod(0o755)


_NEEDS_CHMOD = pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0,
    reason = "chmod 000 denies neither root nor Windows",
)
_XDG_ONLY = pytest.mark.skipif(
    not sys.platform.startswith(("linux", "freebsd")),
    reason = "XDG config base is the Linux/FreeBSD branch",
)


def _mpl_user_rc(tmp_path, monkeypatch):
    _write_rc(_matplotlib_config_dir(tmp_path / "home"))


def _mpl_user_styles(tmp_path, monkeypatch):
    _write_style(_matplotlib_config_dir(tmp_path / "home") / "stylelib")


def _mpl_empty_config(tmp_path, monkeypatch):
    _matplotlib_config_dir(tmp_path / "home").mkdir(parents = True)


def _mpl_xdg_rc(tmp_path, monkeypatch):
    _write_rc(tmp_path / "xdg" / "matplotlib")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def _mpl_denied_config(tmp_path, monkeypatch):
    config = _matplotlib_config_dir(tmp_path / "home")
    _write_rc(config)
    return _denied(config.parent)


def _mpl_denied_styles(tmp_path, monkeypatch):
    styles = _matplotlib_config_dir(tmp_path / "home") / "stylelib"
    _write_style(styles)

    @contextlib.contextmanager
    def guard():
        with _denied(styles):
            # Path.glob suppresses the scandir error, which is what read an unreadable stylelib
            # as empty; without this the case could pass while chmod did nothing.
            assert list(styles.glob("*.mplstyle")) == []
            yield

    return guard()


def _mpl_failing_volume(tmp_path, monkeypatch):
    config = _matplotlib_config_dir(tmp_path / "home")
    _write_rc(config)
    _fail_stat_on(monkeypatch, config / "matplotlibrc", OSError(errno.EIO, "Input/output error"))


def _mpl_managed_styles_and_user_rc(tmp_path, monkeypatch):
    _write_style(_mpl_managed(tmp_path) / "stylelib")
    _write_rc(_matplotlib_config_dir(tmp_path / "home"))


def _mpl_empty_managed_and_user_rc(tmp_path, monkeypatch):
    (_mpl_managed(tmp_path) / "stylelib").mkdir(parents = True)
    _write_rc(_matplotlib_config_dir(tmp_path / "home"))


def _mpl_linked_styles(tmp_path, monkeypatch):
    config = _matplotlib_config_dir(tmp_path / "home")
    config.mkdir(parents = True)
    elsewhere = tmp_path / "styles-volume"
    elsewhere.mkdir()  # empty, which is the whole point
    (config / "stylelib").symlink_to(elsewhere, target_is_directory = True)


def _mpl_dangling_styles(tmp_path, monkeypatch):
    config = _matplotlib_config_dir(tmp_path / "home")
    config.mkdir(parents = True)
    (config / "stylelib").symlink_to(tmp_path / "never-mounted", target_is_directory = True)


def _mpl_plain_empty_styles(tmp_path, monkeypatch):
    (_matplotlib_config_dir(tmp_path / "home") / "stylelib").mkdir(parents = True)


@pytest.mark.parametrize(
    "prepare, pinned",
    [
        # MPLCONFIGDIR moves the CONFIG directory as well as the cache, so a pin wins only when
        # there is nothing of the user's at matplotlib's own directory to hide. Each "must still
        # pin" row is the control for the row above it: matplotlib creates the config dir and
        # stylelib on import, so existence alone must never count as configuration.
        pytest.param(_mpl_user_rc, False, id = "a user matplotlibrc"),
        pytest.param(_mpl_user_styles, False, id = "a user style library"),
        pytest.param(_mpl_empty_config, True, id = "an empty config dir"),
        pytest.param(_mpl_xdg_rc, False, id = "an rc under XDG_CONFIG_HOME", marks = _XDG_ONLY),
        pytest.param(
            _mpl_denied_config, False, id = "a config dir we may not read", marks = _NEEDS_CHMOD
        ),
        pytest.param(
            _mpl_denied_styles, False, id = "a stylelib we may not read", marks = _NEEDS_CHMOD
        ),
        pytest.param(_mpl_failing_volume, False, id = "an rc on a failing volume"),
        pytest.param(_mpl_managed_styles_and_user_rc, True, id = "our own styles, already ours"),
        pytest.param(_mpl_empty_managed_and_user_rc, False, id = "an empty managed dir"),
        pytest.param(_mpl_linked_styles, False, id = "a stylelib linked to an empty volume"),
        pytest.param(_mpl_dangling_styles, False, id = "a dangling stylelib link"),
        pytest.param(_mpl_plain_empty_styles, True, id = "a plain empty stylelib"),
    ],
)
def test_mplconfigdir_is_pinned_only_when_it_would_hide_nothing(
    tmp_path, monkeypatch, prepare, pinned
):
    guard = prepare(tmp_path, monkeypatch) or contextlib.nullcontext()
    sr = _load_storage_roots()

    with guard:
        sr._setup_cache_env()

    if pinned:
        assert os.environ["MPLCONFIGDIR"] == str(_mpl_managed(tmp_path))
    else:
        assert "MPLCONFIGDIR" not in os.environ
        _assert_the_resolver_really_ran(tmp_path)


def _dd_managed(tmp_path: Path) -> Path:
    return tmp_path / "studio" / "data-designer"


def _dd_legacy(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".data-designer"


def _dd_nothing(tmp_path, monkeypatch):
    (tmp_path / "home").mkdir(parents = True, exist_ok = True)


def _dd_legacy_in_use(tmp_path, monkeypatch):
    _use_data_designer(_dd_legacy(tmp_path))


def _dd_managed_in_use(tmp_path, monkeypatch):
    _use_data_designer(_dd_managed(tmp_path))
    _dd_legacy(tmp_path).mkdir(parents = True)


def _dd_managed_untouched(tmp_path, monkeypatch):
    (_dd_managed(tmp_path) / "managed-assets").mkdir(parents = True)
    _dd_legacy(tmp_path).mkdir(parents = True)


def _dd_managed_untouched_legacy_in_use(tmp_path, monkeypatch):
    (_dd_managed(tmp_path) / "managed-assets").mkdir(parents = True)
    _use_data_designer(_dd_legacy(tmp_path))


def _dd_managed_unlistable(tmp_path, monkeypatch):
    managed = _dd_managed(tmp_path)
    _use_data_designer(managed)
    _dd_legacy(tmp_path).mkdir(parents = True)
    real_iterdir = Path.iterdir

    def deny(self):
        if self in (managed, managed / "managed-assets"):
            raise PermissionError(13, "Permission denied")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", deny)


def _dd_managed_assets_denied(tmp_path, monkeypatch):
    managed = _dd_managed(tmp_path)
    _use_data_designer(managed)
    _dd_legacy(tmp_path).mkdir(parents = True)
    return _denied(managed / "managed-assets")


def _dd_redirected_assets(tmp_path, monkeypatch):
    managed = _dd_managed(tmp_path)
    managed.mkdir(parents = True)
    elsewhere = tmp_path / "big-disk" / "assets"
    elsewhere.mkdir(parents = True)  # empty, which is the whole point
    (managed / "managed-assets").symlink_to(elsewhere, target_is_directory = True)
    _use_data_designer(_dd_legacy(tmp_path))


def _dd_legacy_denied(tmp_path, monkeypatch):
    _use_data_designer(_dd_legacy(tmp_path))
    return _denied(tmp_path / "home")


def _dd_legacy_failing_volume(tmp_path, monkeypatch):
    legacy = _dd_legacy(tmp_path)
    _use_data_designer(legacy)
    _fail_stat_on(monkeypatch, legacy, OSError(errno.EIO, "Input/output error"))


def _dd_legacy_symlink_loop(tmp_path, monkeypatch):
    legacy = _dd_legacy(tmp_path)
    legacy.parent.mkdir(parents = True, exist_ok = True)
    legacy.symlink_to(legacy)
    # ELOOP reads as absence through Path.exists() on every release, which is the bug.
    assert Path(legacy).exists() is False


@pytest.mark.parametrize(
    "prepare, pinned",
    [
        # Data Designer's home is not a cache: repointing one that holds yaml configs and
        # multi-GB parquet hides them behind a re-seeded default. Our own home wins once it has
        # been USED, since the legacy probe re-runs every launch and a standalone run creating
        # ~/.data-designer would otherwise take the work written under the Studio root.
        pytest.param(_dd_nothing, True, id = "no legacy home at all"),
        pytest.param(_dd_legacy_in_use, False, id = "a legacy home in use"),
        pytest.param(_dd_managed_in_use, True, id = "our home in use, legacy empty"),
        pytest.param(_dd_managed_untouched, False, id = "our home untouched, legacy empty"),
        pytest.param(
            _dd_managed_untouched_legacy_in_use, False, id = "our home untouched, legacy in use"
        ),
        pytest.param(_dd_managed_unlistable, True, id = "our home we may not list"),
        pytest.param(
            _dd_managed_assets_denied, True, id = "our managed-assets denied", marks = _NEEDS_CHMOD
        ),
        pytest.param(_dd_redirected_assets, True, id = "managed-assets redirected by a link"),
        pytest.param(
            _dd_legacy_denied, False, id = "a legacy home we may not read", marks = _NEEDS_CHMOD
        ),
        pytest.param(_dd_legacy_failing_volume, False, id = "a legacy home on a failing volume"),
        pytest.param(_dd_legacy_symlink_loop, False, id = "a legacy home that is a symlink loop"),
    ],
)
def test_the_data_designer_home_is_pinned_only_when_it_would_hide_nothing(
    tmp_path, monkeypatch, prepare, pinned
):
    guard = prepare(tmp_path, monkeypatch) or contextlib.nullcontext()
    sr = _load_storage_roots()

    with guard:
        sr._setup_cache_env()

    managed = _dd_managed(tmp_path)
    if pinned:
        assert os.environ["DATA_DESIGNER_HOME"] == str(managed)
        assert os.environ["DATA_DESIGNER_MANAGED_ASSETS_PATH"] == str(managed / "managed-assets")
    else:
        assert "DATA_DESIGNER_HOME" not in os.environ
        assert "DATA_DESIGNER_MANAGED_ASSETS_PATH" not in os.environ
        _assert_the_resolver_really_ran(tmp_path)


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
    # TRITON_HOME would move ~/.triton/override, which holds user files, along with the cache,
    # so a TRITON_KERNEL_OVERRIDE=1 run would compile something else instead.
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
    # macOS is matplotlib's "other platforms" branch: ~/.matplotlib, not Application Support.
    pytest.importorskip("matplotlib")
    monkeypatch.setattr(sys, "platform", "darwin")
    sr = _load_storage_roots()

    ours = sr._matplotlib_config_dir()

    # sys.platform is read inside _get_config_or_cache_dir, so a fresh interpreter can be
    # asked what it would do on a Mac.
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
    """Where matplotlib on THIS platform would look for a user matplotlibrc.

    _get_config_or_cache_dir: XDG on Linux and FreeBSD, a non-empty ~/.matplotlib ahead of
    %LOCALAPPDATA% on Windows, ~/.matplotlib everywhere else. The Linux branch alone was hardcoded
    here, so on a mac runner these tests wrote the rc into a directory matplotlib never reads: the
    guard correctly found nothing to strand, pinned MPLCONFIGDIR, and nine tests failed for a
    reason that was in them rather than in the code they cover. The two tests above hold this
    derivation against real matplotlib on each spoofed platform, so it cannot quietly drift into
    agreeing with storage_roots.py and nothing else.

    Windows resolves to ~/.matplotlib rather than %LOCALAPPDATA%: the callers write a file into
    whatever this returns, LOCALAPPDATA on a runner is outside tmp_path, and a non-empty
    ~/.matplotlib is the branch matplotlib itself prefers.
    """
    if sys.platform.startswith(("linux", "freebsd")):
        base = (os.environ.get("XDG_CONFIG_HOME") or "").strip()
        return (Path(base) if base else home / ".config") / "matplotlib"
    return home / ".matplotlib"


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
    not sys.platform.startswith(("linux", "freebsd")),
    reason = "XDG config base is the Linux/FreeBSD branch",
)
def test_an_xdg_config_dir_is_read_without_a_resolvable_home(monkeypatch, tmp_path):
    # _get_xdg_config_dir reads XDG_CONFIG_HOME before it needs a home, so bailing out on
    # Path.home() pinned over a real matplotlibrc.
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
    # torch appends py<ver>_<accelerator> to its DEFAULT root only, so a pin carries it.
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
    # conda-forge's CPU and CUDA packages of one release share a __version__, differing only
    # in a conda build string.
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
    # torch names a ROCm build 'cpu', since version.cuda is unset; hip is read first.
    sr = _load_storage_roots()

    tags = []
    for label, hip in (("cpu", None), ("rocm", "6.4.43484-123eb5128")):
        with _only_torch(_fake_torch_on_path(tmp_path, label, "2.9.1", hip = hip)):
            tags.append(sr._torch_runtime_tag())

    assert tags[0] != tags[1], f"a ROCm build shared the CPU extension dir: {tags[0]}"
    assert "rocm6.4.43484-123eb5128" in tags[1]


def test_torch_extension_cache_separates_two_host_architectures(tmp_path, monkeypatch):
    # An arm64 python and a Rosetta x86_64 python on ONE Mac agree on every other field, so
    # torch's py<ver>_<cu_str> folder gives them one directory and the .so fails to load.
    sr = _load_storage_roots()

    tags = []
    for machine in ("arm64", "x86_64"):
        monkeypatch.setattr(platform, "machine", lambda m = machine: m)
        with _only_torch(_fake_torch_on_path(tmp_path, machine, "2.9.1")):
            tags.append(sr._torch_runtime_tag())

    assert tags[0] != tags[1], f"two host architectures shared one extension dir: {tags[0]}"
    assert "arm64" in tags[0] and "x86-64" in tags[1]


def test_torch_extension_cache_survives_an_unnameable_architecture(tmp_path, monkeypatch):
    # platform.machine() returns "" when the platform cannot answer; no empty segment.
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
    # Runs before torch exists in a fresh venv, and importing it on the startup path would cost
    # seconds. The fake package raises on import, so a skip is a failure.
    sr = _load_storage_roots()
    entry = _fake_torch_on_path(tmp_path, "guard", "2.9.1+cu128", cuda = "12.8")

    with _only_torch(entry):
        tag = sr._torch_runtime_tag()
        assert "torch" not in sys.modules, "torch was imported to build the cache tag"

    assert "cu128" in tag


def test_a_managed_matplotlibrc_is_not_displaced_by_a_later_legacy_one(tmp_path):
    """The legacy probe re-runs every launch, so on its own it hands a matplotlibrc written
    HERE to a ~/.config/matplotlib created later by some other tool.

    Measured across four launches before the fix: dpi 177 from the managed config, then 222 once
    a legacy rc appeared, then 177 again when it was removed, with the managed rc present
    throughout. A style that changes on a later launch and changes back is worse than either
    choice made once. _data_designer_defaults already applies this rule to its own home.
    """
    managed = tmp_path / "studio" / "cache" / "matplotlib"
    managed.mkdir(parents = True)
    (managed / "matplotlibrc").write_text("figure.dpi: 177\n", encoding = "utf-8")
    config = _matplotlib_config_dir(tmp_path / "home")
    config.mkdir(parents = True)
    (config / "matplotlibrc").write_text("figure.dpi: 222\n", encoding = "utf-8")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["MPLCONFIGDIR"] == str(managed)


def test_a_blank_toolchain_override_is_dropped_on_a_spaced_root(monkeypatch, tmp_path):
    """ "blank counts as unset" has to hold for a root we refuse to pin, too.

    Inductor distinguishes an absent TORCHINDUCTOR_CACHE_DIR from a present one, so a leftover
    "   " becomes a relative compiler path and is then split by the very unquoted command
    construction the whitespace refusal exists to avoid.
    """
    spaced = tmp_path / "My Studio"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(spaced))
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", "   ")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert (
        os.environ.get("TORCHINDUCTOR_CACHE_DIR") or ""
    ).strip() != "" or "TORCHINDUCTOR_CACHE_DIR" not in os.environ
    _assert_no_unparseable_pin(sr, spaced)


def test_an_unusable_managed_inductor_path_is_not_published(monkeypatch, tmp_path):
    """torch treats the value as authoritative, so a path it cannot use fails every compile.

    Unset, it would have used its own temporary cache instead. The placement error is swallowed
    on purpose so startup survives, which is what let an unusable path stay published.
    """
    cache = tmp_path / "studio" / "cache"
    cache.mkdir(parents = True)
    # A regular file where the directory should go: mkdir raises FileExistsError, which the
    # best-effort handler treats as "already there".
    (cache / "torchinductor").write_text("not a directory", encoding = "utf-8")
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert "TORCHINDUCTOR_CACHE_DIR" not in os.environ
    # The other pins are unaffected: only the one that could not be made is withheld.
    assert os.environ["CUDA_CACHE_PATH"] == str(cache / "cuda")


@pytest.mark.skipif(
    os.name == "nt" or os.geteuid() == 0, reason = "chmod 555 denies neither root nor Windows"
)
def test_a_read_only_managed_cache_is_not_published(tmp_path):
    """A directory that exists but cannot be written to is the same defect as a missing one.

    mkdir passes exist_ok = False, so an existing directory takes the FileExistsError path and
    proves nothing about writability. torch only falls back to its own temporary cache when the
    variable is UNSET (torch/_inductor/runtime/cache_dir_utils.py), so a read-only pin fails
    every compile.
    """
    cache = tmp_path / "studio" / "cache"
    for name in ("torchinductor", "triton", "cuda"):
        (cache / name).mkdir(parents = True)
        (cache / name).chmod(0o555)
    sr = _load_storage_roots()
    try:
        sr._setup_cache_env()

        for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
            assert key not in os.environ, key
        # Withheld, not blanked: an empty value is a relative path to the library reading it.
        assert os.environ["NUMBA_CACHE_DIR"] == str(cache / "numba")
    finally:
        for name in ("torchinductor", "triton", "cuda"):
            (cache / name).chmod(0o755)


def test_the_write_probe_leaves_nothing_behind(tmp_path):
    """The probe creates a file in the directory it is testing; it must not accumulate."""
    sr = _load_storage_roots()

    sr._setup_cache_env()

    inductor = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    assert inductor.is_dir()
    assert list(inductor.iterdir()) == []


def test_a_usable_managed_inductor_path_is_still_published(tmp_path):
    """The rule above must not withhold the ordinary case."""
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert os.environ["TORCHINDUCTOR_CACHE_DIR"] == str(
        tmp_path / "studio" / "cache" / "torchinductor"
    )


@pytest.mark.skipif(os.name == "nt" or os.geteuid() == 0, reason = "chmod 000 denies neither")
def test_a_note_that_could_not_be_read_is_not_cached_as_a_missing_one(monkeypatch, tmp_path):
    """A miss is a fact about the install; a failure to LOOK is a fact about one instant.

    _recorded_master_root() memoises per studio path, and the backend is a long-lived process.
    Caching an EACCES on share/ -- a mount that came back, a permission fixed a second later --
    pinned that process to the legacy runtime paths for its whole lifetime with the trees it
    should have found sitting right beside it, and nothing in production calls
    forget_recorded_master_root() to get back out.
    """
    master = tmp_path / "root"
    studio = master / "studio"
    share = studio / "share"
    share.mkdir(parents = True)
    (share / ".unsloth-master-root").write_text(str(master) + "\n", encoding = "utf-8")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio))
    sr = _load_storage_roots()

    share.chmod(0o000)
    try:
        assert sr.unsloth_home() is None
    finally:
        share.chmod(0o755)

    assert sr.unsloth_home() == master, "a transient read failure was cached as a missing note"


def test_a_genuinely_absent_note_is_still_cached(monkeypatch, tmp_path):
    """The other half: the fix must not turn the memo off. unsloth_home() runs on every
    cache-var lookup, so re-stat'ing a note that is simply not there is a syscall per call."""
    studio = tmp_path / "root" / "studio"
    (studio / "share").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio))
    sr = _load_storage_roots()

    assert sr.unsloth_home() is None
    # Written after the miss was cached: a cached answer is the point, and
    # forget_recorded_master_root() is the way back out for an installer that writes one later.
    (studio / "share" / ".unsloth-master-root").write_text(
        str(tmp_path / "root") + "\n",
        encoding = "utf-8",
    )
    assert sr.unsloth_home() is None
    sr.forget_recorded_master_root()
    assert sr.unsloth_home() == tmp_path / "root"


def test_a_write_probe_that_cannot_close_its_handle_is_a_no_not_a_crash(monkeypatch, tmp_path):
    """_usable_dir answers yes or no. Every other failure in it returns False; an OSError from
    os.close escaping instead would take the whole _setup_cache_env() -- and the backend start
    that called it -- down with it."""
    sr = _load_storage_roots()

    real_close = os.close

    def _failing_close(fd):
        real_close(fd)
        raise OSError(errno.EIO, "close failed")

    monkeypatch.setattr(sr.os, "close", _failing_close)
    assert sr._usable_dir(str(tmp_path)) is False

    monkeypatch.setattr(sr.os, "close", real_close)
    assert sr._usable_dir(str(tmp_path)) is True


def _note_ancestor_of_a_legacy_tree(tmp_path, monkeypatch):
    """A note naming $HOME, which passes containment because ~/.unsloth/studio is inside it.

    process.rs scrubs UNSLOTH_HOME and UNSLOTH_PORTABLE from every managed spawn so Tauri uses
    the legacy root whatever the environment says. A note is a FILE, which no env_remove reaches,
    so this is how both variables came back and moved the packaged app's caches.
    """
    home = tmp_path / "home"
    studio = home / ".unsloth" / "studio"
    _write_note(studio, home)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio))
    for scrubbed in ("UNSLOTH_HOME", "UNSLOTH_PORTABLE", "STUDIO_HOME"):
        monkeypatch.delenv(scrubbed, raising = False)
    return None


def _note_naming_the_legacy_default(tmp_path, monkeypatch):
    """The reader has to decline it, not just the writer: setup only sweeps an old note when it
    runs again, and the user who hit this set UNSLOTH_HOME once, for one command."""
    home = tmp_path / "home"
    studio = home / ".unsloth" / "studio"
    _write_note(studio, home / ".unsloth")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio))
    return None


def _note_naming_a_real_master_root(tmp_path, monkeypatch):
    """The other side of the rule: a recorded root that is not the legacy default and that
    contains this Studio tree is exactly what the note is for."""
    master = tmp_path / "portable"
    _write_note(master / "studio", master)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(master / "studio"))
    return master


def _an_explicit_legacy_unsloth_home(tmp_path, monkeypatch):
    """Only the RECORDED root is declined. unsloth_home() returns an explicit UNSLOTH_HOME before
    it reads any note, and narrowing that would change what the variable means."""
    home = tmp_path / "home"
    (home / ".unsloth" / "studio" / "share").mkdir(parents = True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home / ".unsloth" / "studio"))
    monkeypatch.setenv("UNSLOTH_HOME", str(home / ".unsloth"))
    return home / ".unsloth"


def _write_note(studio: Path, root: Path) -> None:
    (studio / "share").mkdir(parents = True, exist_ok = True)
    (studio / "share" / ".unsloth-master-root").write_text(str(root) + "\n", encoding = "utf-8")


@pytest.mark.parametrize(
    "prepare",
    [
        pytest.param(_note_ancestor_of_a_legacy_tree, id = "a note naming an ancestor"),
        pytest.param(_note_naming_the_legacy_default, id = "a note naming the legacy default"),
        pytest.param(_note_naming_a_real_master_root, id = "a note naming a real master root"),
        pytest.param(_an_explicit_legacy_unsloth_home, id = "an explicit legacy UNSLOTH_HOME"),
    ],
)
def test_a_master_root_is_honoured_only_when_a_reader_should_honour_it(
    tmp_path, monkeypatch, prepare
):
    """`prepare` returns the root that must be honoured, or None when it must be declined.

    The hub cache is asserted either way, since declining is only meaningful if the shared cache
    really stays shared, and honouring is only meaningful if the caches really move.
    """
    expected = prepare(tmp_path, monkeypatch)
    sr = _load_storage_roots()

    assert sr.unsloth_home() == expected
    assert sr.portable_mode() is (expected is not None)

    sr._setup_cache_env()

    hub = Path(os.environ["HF_HUB_CACHE"])
    if expected is None:
        assert hub == Path(os.environ["HOME"]) / ".cache" / "huggingface" / "hub"
        assert "TORCH_HOME" not in os.environ
    else:
        assert expected in hub.parents
        assert Path(os.environ["TORCH_HOME"]).is_relative_to(expected)


# 25 of 113 cases stayed green with the production hunks reverted: each asserts a variable is
# ABSENT, which is equally true when no pinning code exists, so they were assertions rather than
# negative controls. Pairing a decline with the pins that must STILL happen makes them falsifiable.
_DECLINE_SIBLINGS = (
    "TORCHINDUCTOR_CACHE_DIR",
    "NUMBA_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "UV_CACHE_DIR",
    "UNSLOTH_COMPILE_LOCATION",
)


@pytest.mark.parametrize("declined", ["MPLCONFIGDIR", "DATA_DESIGNER_HOME"])
def test_declining_one_pin_never_means_the_resolver_did_nothing(tmp_path, declined):
    """A decline is only meaningful next to a pin. If the resolver were absent, or returned early,
    the sibling assertions below would fail and the decline would stop proving anything."""
    home = tmp_path / "home"
    if declined == "MPLCONFIGDIR":
        config = _matplotlib_config_dir(home)
        config.mkdir(parents = True)
        (config / "matplotlibrc").write_text("figure.dpi: 123\n", encoding = "utf-8")
    else:
        (home / ".data-designer" / "recipes").mkdir(parents = True)
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert declined not in os.environ, f"{declined} displaced user files"
    root = str(tmp_path / "studio")
    for sibling in _DECLINE_SIBLINGS:
        value = os.environ.get(sibling)
        assert value, f"{sibling} unpinned: the decline of {declined} proves nothing"
        assert value.startswith(root), f"{sibling} escaped the studio root: {value}"


def test_an_uninspectable_probe_declines_and_still_pins_everything_else(tmp_path):
    """Same rule for the os.lstat/os.scandir failure paths, which are the subtlest of the 25:
    an unreadable matplotlib config must leave MPLCONFIGDIR alone AND leave the rest pinned."""
    config = _matplotlib_config_dir(tmp_path / "home")
    config.parent.mkdir(parents = True, exist_ok = True)
    config.write_text("", encoding = "utf-8")  # a file where a directory belongs: ENOTDIR
    sr = _load_storage_roots()

    sr._setup_cache_env()

    assert "MPLCONFIGDIR" not in os.environ
    for sibling in _DECLINE_SIBLINGS:
        assert os.environ.get(sibling, "").startswith(str(tmp_path / "studio")), sibling


def test_a_file_where_the_config_dir_belongs_declines_the_pin_on_windows_too(monkeypatch, tmp_path):
    """_nothing_at reads a FileNotFoundError as absence, and on Windows that error also means
    "a parent component is a file" -- the case POSIX reports as NotADirectoryError and declines.
    So with a file where ~/.matplotlib belongs, POSIX declined the pin and Windows took it,
    hiding whatever the user had underneath. Caught by the windows-latest leg.

    Reproduced by the error shape rather than the platform, since the Linux runners cannot raise
    it; the POSIX arm of the same fixture is the test directly above.
    """
    sr = _load_storage_roots()
    config = _matplotlib_config_dir(tmp_path / "home")
    config.parent.mkdir(parents = True, exist_ok = True)
    config.write_text("", encoding = "utf-8")

    real_lstat = sr.os.lstat

    def windows_shaped_lstat(p, *a, **k):
        text = os.fspath(p)
        if text != str(config) and text.startswith(str(config) + os.sep):
            raise FileNotFoundError(2, "The system cannot find the path specified", text)
        return real_lstat(p, *a, **k)

    monkeypatch.setattr(sr.os, "lstat", windows_shaped_lstat)

    assert (
        sr._nothing_at(config / "matplotlibrc") is False
    ), "a file where the config dir belongs read as 'nothing there'"
    # A genuinely empty, genuinely present directory must still read as empty, or the fix above
    # would decline every pin and the guard would stop guarding anything.
    empty = tmp_path / "really-empty"
    empty.mkdir()
    assert sr._nothing_at(empty / "matplotlibrc") is True
    # And a path with nothing along it at all is still absence, which is the default install.
    assert sr._nothing_at(tmp_path / "never" / "created" / "matplotlibrc") is True
