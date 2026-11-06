# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`studio/setup.sh` and `unsloth_cli` must pick the same uv cache, so both run over the same fixtures."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_SETUP_SH = _REPO_ROOT / "studio" / "setup.sh"

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason = "studio/setup.sh only ever runs on POSIX; install.ps1 is Windows"
)

# Windows lacks os.geteuid and skipif decorators run at collection.
_IS_ROOT = getattr(os, "geteuid", lambda: 1)() == 0

# Sliced by name since setup.sh cannot be sourced; extraction is asserted.
_HELPERS = (
    "_uv_is_bucket_name",
    "_uv_cache_probe_writable",
    "_uv_cache_folds_case",
    "_uv_store_key",
    "_uv_cache_usable",
    "_uv_control_files_writable",
    "_uv_cache_warm",
)


def _shell_helpers() -> str:
    out: list[str] = []
    grab = False
    for line in _SETUP_SH.read_text(encoding = "utf-8").splitlines(keepends = True):
        if any(line.startswith(f"{name}() {{") for name in _HELPERS):
            grab = True
        if grab:
            out.append(line)
            if line.startswith("}"):
                grab = False
    text = "".join(out)
    missing = [name for name in _HELPERS if f"{name}() {{" not in text]
    assert not missing, f"could not extract {missing} from studio/setup.sh"
    return text


def _ask_shell(func: str, cache: Path) -> bool:
    script = _shell_helpers() + f'\nif {func} "$1"; then exit 0; else exit 1; fi\n'
    env = {k: v for k, v in os.environ.items() if not k.startswith("UV_")}
    return (
        subprocess.run(
            ["bash", "-c", script, "_", str(cache)], capture_output = True, env = env
        ).returncode
        == 0
    )


def _studio():
    from unsloth_cli.commands import studio as _studio_mod
    return _studio_mod


def _warm(cache: Path, bucket: str = "archive-v0") -> Path:
    (cache / bucket / "pkg").mkdir(parents = True, exist_ok = True)
    (cache / bucket / "pkg" / "torch.whl").write_bytes(b"\0" * 8)
    return cache


def _build(cache: Path, shape: str) -> None:
    cache.mkdir(parents = True, exist_ok = True)
    if shape == "cold":
        return
    if shape == "metadata only":
        (cache / "wheels-v6").mkdir()
        (cache / "wheels-v6" / "resolve.msgpack").write_bytes(b"\0")
        return
    if shape == "lookalike bucket":
        _warm(cache, "archive-v0.backup")
        return
    if shape == "unicode digit version":
        _warm(cache, "archive-v١")
        return
    _warm(cache)
    if shape == "warm":
        return
    if shape == "unrelated store":
        (cache / "osv-v0").mkdir()
    elif shape == "git store":
        (cache / "git-v0").mkdir()
    elif shape == "file at a store":
        (cache / "interpreter-v4").write_bytes(b"")
    elif shape == "lock is a directory":
        (cache / ".lock").mkdir()
    elif shape == "stray version-shaped dir":
        (cache / "unused-v999").mkdir()
        (cache / "unused-v999" / ".lock").write_bytes(b"")
    elif shape == "index leaf is a symlinked dir":
        (cache / "simple-v20").mkdir()
        (cache / "elsewhere").mkdir()
        (cache / "simple-v20" / "pypi").symlink_to(cache / "elsewhere")
    elif shape == "index leaf is a file":
        (cache / "simple-v20").mkdir()
        (cache / "simple-v20" / "pypi").write_bytes(b"")
    elif shape == "index leaf is a dangling link":
        (cache / "wheels-v6").mkdir()
        (cache / "wheels-v6" / "pypi").symlink_to(cache / "gone")
    elif shape == "custom index hash leaf":
        (cache / "simple-v20" / "index" / "e1d141a6ca947dff").mkdir(parents = True)
        (cache / "wheels-v6" / "index" / "e1d141a6ca947dff").mkdir(parents = True)
    elif shape == "index shard":
        (cache / "simple-v20" / "pypi").mkdir(parents = True)
        (cache / "wheels-v6" / "pypi").mkdir(parents = True)
        (cache / "interpreter-v4" / "abcd").mkdir(parents = True)
    elif shape == "control files present":
        (cache / ".lock").write_bytes(b"")
        (cache / "CACHEDIR.TAG").write_bytes(b"Signature")
    else:  # pragma: no cover - a typo in the parametrisation, not a product state
        raise AssertionError(f"unknown shape {shape!r}")


_SHAPES = [
    "warm",
    "cold",
    "metadata only",
    "lookalike bucket",
    "unicode digit version",
    "unrelated store",
    "git store",
    "file at a store",
    "lock is a directory",
    "stray version-shaped dir",
    "control files present",
    "index shard",
    "index leaf is a file",
    "index leaf is a dangling link",
    "index leaf is a symlinked dir",
    "custom index hash leaf",
]


@pytest.mark.parametrize("shape", _SHAPES)
def test_both_implementations_agree_on_warmth(tmp_path, shape):
    """Warmth decides WHICH cache is used, so a disagreement sends the two down different
    caches for the same install."""
    cache = tmp_path / "uv"
    _build(cache, shape)
    assert _ask_shell("_uv_cache_warm", cache) is _studio()._uv_cache_has_packages(cache)


@pytest.mark.parametrize("shape", _SHAPES)
def test_both_implementations_agree_on_usability(tmp_path, shape):
    """Usability decides whether a cache is handed to uv at all. One side accepting what the
    other rejects means uv aborts on one entry point and not the other."""
    cache = tmp_path / "uv"
    _build(cache, shape)
    assert _ask_shell("_uv_cache_usable", cache) is _studio()._uv_cache_is_writable(cache)


@pytest.mark.skipif(_IS_ROOT, reason = "root can write anywhere")
@pytest.mark.parametrize(
    "store, usable",
    [
        # Probed: `uv pip install` writes these, so a read-only one aborts uv.
        ("archive-v0", False),
        ("sdists-v9", False),
        ("git-v0", False),
        # Measured on uv 0.10.7: install succeeds with these at 0555.
        ("osv-v0", True),
        ("binaries-v0", True),
        ("environments-v2", True),
        ("python-v0", True),
        ("flat-index-v2", True),
    ],
)
def test_both_implementations_agree_on_which_stores_are_probed(tmp_path, store, usable):
    cache = _warm(tmp_path / "uv")
    (cache / store).mkdir(exist_ok = True)
    (cache / store).chmod(0o555)
    try:
        shell = _ask_shell("_uv_cache_usable", cache)
        python = _studio()._uv_cache_is_writable(cache)
    finally:
        (cache / store).chmod(0o755)
    assert shell is python, f"{store}: shell={shell} python={python}"
    assert shell is usable


@pytest.mark.skipif(_IS_ROOT, reason = "root can write anywhere")
@pytest.mark.parametrize(
    "shard, usable",
    [
        # uv rewrites index metadata on every resolve, so a read-only shard aborts it.
        ("simple-v20/pypi", False),
        ("wheels-v6/pypi", False),
        # Content-addressed stores are only added to; uv works with them at 0555.
        ("interpreter-v4/abcd", True),
        ("archive-v0/pkg", True),
        # index/<hash> holds custom --index-url metadata; a 0555 one aborts uv 0.12.1.
        ("simple-v20/index/e1d141a6ca947dff", False),
        ("wheels-v6/index/e1d141a6ca947dff", False),
        # Bounded: the per-package level below the hash was measured fine.
        ("wheels-v6/index/e1d141a6ca947dff/idna", True),
        ("simple-v20/pypi/deeper", True),
    ],
)
def test_both_implementations_agree_on_which_shards_are_probed(tmp_path, shard, usable):
    cache = _warm(tmp_path / "uv")
    (cache / shard).mkdir(parents = True, exist_ok = True)
    (cache / shard).chmod(0o555)
    try:
        shell = _ask_shell("_uv_cache_usable", cache)
        python = _studio()._uv_cache_is_writable(cache)
    finally:
        (cache / shard).chmod(0o755)
    assert shell is python, f"{shard}: shell={shell} python={python}"
    assert shell is usable
