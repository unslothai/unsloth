# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""share/studio.conf carries POSIX paths, so it is decoded as one.

A POSIX path is a byte string. install.sh writes the resolved root into
share/studio.conf verbatim, so a root holding a byte that is not valid UTF-8
reaches _portable_root_env as that byte. Decoded through errors="replace" it
comes back as U+FFFD, which is not an unusable value but a DIFFERENT and
perfectly creatable one -- the one byte re-encodes to the three bytes of
U+FFFD -- so an activated-venv `unsloth studio update` would export
UV_CACHE_DIR and PIP_CACHE_DIR under a SIBLING of the selected root and
repopulate the caches outside it, which is the escape this overlay exists to
prevent. os.fsdecode is what the master-root record beside it already uses.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from unsloth_cli.commands import studio as cli  # noqa: E402

# The variables install.sh restates in the portable block, each of which names a
# directory an update writes into.
_ROOTED = ("UV_CACHE_DIR", "UV_TOOL_DIR", "NPM_CONFIG_CACHE", "PIP_CACHE_DIR", "CUDA_CACHE_PATH")

_CONF_LINES = (
    (b"UV_CACHE_DIR", b"/cache/uv"),
    (b"UV_PYTHON_INSTALL_DIR", b"/cache/uv-python"),
    (b"UV_TOOL_DIR", b"/cache/uv-tools"),
    (b"UV_TOOL_BIN_DIR", b"/bin"),
    (b"UV_PYTHON_BIN_DIR", b"/bin"),
    (b"UV_INSTALL_DIR", b"/bin"),
    (b"NPM_CONFIG_CACHE", b"/cache/npm"),
    (b"BUN_INSTALL_CACHE_DIR", b"/cache/bun"),
    (b"CUDA_CACHE_PATH", b"/cache/cuda"),
    (b"PIP_CACHE_DIR", b"/cache/pip"),
)


def _write_installer_conf(root_bytes: bytes) -> None:
    """The portable block create_studio_shortcuts writes, byte for byte."""
    share = root_bytes + b"/share"
    os.makedirs(share, exist_ok = True)
    # The installer's own escaping: `sed "s/'/'\\''/g"`.
    quoted = root_bytes.replace(b"'", b"'\\''")
    blob = b"UNSLOTH_EXE='" + quoted + b"/studio/unsloth_studio/bin/unsloth'\n"
    blob += b"export UNSLOTH_HOME='" + quoted + b"'\n"
    blob += b"export UNSLOTH_PORTABLE=1\n"
    for name, suffix in _CONF_LINES:
        blob += b"export " + name + b"='" + quoted + suffix + b"'\n"
    blob += b"export UV_NO_MODIFY_PATH=1\n"
    with open(share + b"/studio.conf", "wb") as handle:
        handle.write(blob)


@pytest.mark.skipif(os.name != "posix", reason = "a non-UTF-8 path byte needs POSIX filenames")
def test_conf_paths_under_a_non_utf8_root_stay_inside_it(tmp_path: Path) -> None:
    root_bytes = os.fsencode(str(tmp_path)) + b"/opt\xffunsloth"
    try:
        os.makedirs(root_bytes)
    except (OSError, UnicodeError) as exc:  # a filesystem that refuses the byte
        pytest.skip(f"filesystem rejects a non-UTF-8 name: {exc}")
    _write_installer_conf(root_bytes)
    root = Path(os.fsdecode(root_bytes))

    env = cli._portable_root_env(root)

    # Every rooted value is the path install.sh wrote, and every one of them is
    # still INSIDE the root once re-encoded the way an update would use it.
    assert env["UV_CACHE_DIR"] == str(root / "cache" / "uv")
    assert env["PIP_CACHE_DIR"] == str(root / "cache" / "pip")
    for name in _ROOTED:
        assert os.fsencode(env[name]).startswith(root_bytes + b"/"), (
            f"{name} escaped the root: {os.fsencode(env[name])!r}"
        )
    # And specifically not the U+FFFD sibling, which exists as a distinct
    # directory name and would silently be created by the update.
    assert b"\xef\xbf\xbd" not in os.fsencode(env["UV_CACHE_DIR"])


@pytest.mark.skipif(os.name != "posix", reason = "a non-UTF-8 path byte needs POSIX filenames")
def test_a_non_utf8_root_keeps_the_derived_values_when_no_conf_exists(tmp_path: Path) -> None:
    """The no-conf path is unchanged: derived values, already correct."""
    root_bytes = os.fsencode(str(tmp_path)) + b"/opt\xffunsloth"
    try:
        os.makedirs(root_bytes)
    except (OSError, UnicodeError) as exc:
        pytest.skip(f"filesystem rejects a non-UTF-8 name: {exc}")
    root = Path(os.fsdecode(root_bytes))
    env = cli._portable_root_env(root)
    assert env["UV_CACHE_DIR"] == str(root / "cache" / "uv")
    assert env["UNSLOTH_PORTABLE"] == "1"


def test_a_utf8_root_is_still_read_from_the_conf(tmp_path: Path) -> None:
    """Non-ASCII but valid UTF-8, and an apostrophe: the overlay still applies.

    Guards the decode change against the opposite mistake -- reading bytes must
    not stop the conf from overriding the derived values at all.
    """
    root = tmp_path / "opt" / "unslöth's"
    root.mkdir(parents = True)
    _write_installer_conf(os.fsencode(str(root)))
    env = cli._portable_root_env(root)
    assert env["UV_CACHE_DIR"] == str(root / "cache" / "uv")
    assert env["NPM_CONFIG_CACHE"] == str(root / "cache" / "npm")
