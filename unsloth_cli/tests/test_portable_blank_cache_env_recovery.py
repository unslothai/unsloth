# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A whitespace-only cache override must not survive the portable env recovery.

`unsloth studio update` from an activated portable venv rebuilds the installer's
cache environment in _ensure_studio_env_exported(). install.sh (_trim_ws) and
storage_roots._setup_cache_env both treat a whitespace-only value as unset, so
this recovery has to as well: " " is a RELATIVE path, and the update subprocess
would resolve it against its working directory and write the uv/npm/pip caches
outside the portable root, which is the escape the recovery exists to prevent.

Each case runs in its own interpreter: STUDIO_HOME is resolved at import time and
the seeding mutates os.environ, so an in-process assertion would depend on
import and test ordering.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MARKER = ".unsloth-portable-root"

PROBE = textwrap.dedent(
    """
    import json, os, sys
    sys.path.insert(0, os.environ["_REPO"])
    from unsloth_cli.commands import studio as cli

    cli._ensure_studio_env_exported()
    print("__JSON__" + json.dumps({
        "studio_home": str(cli.STUDIO_HOME),
        "master": str(cli._portable_master_root() or ""),
        "env": {k: os.environ.get(k) for k in (
            "UNSLOTH_STUDIO_HOME", "UNSLOTH_HOME", "UNSLOTH_LLAMA_CPP_PATH",
            "UV_CACHE_DIR", "NPM_CONFIG_CACHE", "PIP_CACHE_DIR", "CUDA_CACHE_PATH",
        )},
    }))
    """
)


def _make_portable_install(root: Path) -> None:
    """The on-disk shape install.sh --root writes, minus the venv contents."""
    (root / "studio" / "unsloth_studio").mkdir(parents = True, exist_ok = True)
    (root / "bin").mkdir(parents = True, exist_ok = True)
    (root / "share").mkdir(parents = True, exist_ok = True)
    (root / "share" / "studio.conf").write_text(f"export UNSLOTH_HOME='{root}'\n")
    (root / MARKER).write_text(f"{root}\n")


def _run_probe(root: Path, home: Path, overrides: dict) -> dict:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "_REPO": str(REPO_ROOT),
        # The activated-venv launch: the marker, not the installer environment,
        # is what identifies the portable root.
        "UNSLOTH_STUDIO_HOME": str(root / "studio"),
    }
    env.update(overrides)
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    raise AssertionError(
        f"probe failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )


@pytest.fixture()
def portable_root(tmp_path: Path) -> Path:
    root = tmp_path / "opt" / "unsloth"
    _make_portable_install(root)
    (tmp_path / "home").mkdir(exist_ok = True)
    return root


@pytest.mark.parametrize("blank", [" ", "\t", "  \t "])
def test_whitespace_only_cache_override_is_replaced(portable_root: Path, blank: str) -> None:
    result = _run_probe(
        portable_root,
        portable_root.parents[1] / "home",
        {"UV_CACHE_DIR": blank, "NPM_CONFIG_CACHE": blank, "PIP_CACHE_DIR": blank},
    )
    env = result["env"]
    assert env["UV_CACHE_DIR"] == str(portable_root / "cache" / "uv")
    assert env["NPM_CONFIG_CACHE"] == str(portable_root / "cache" / "npm")
    assert env["PIP_CACHE_DIR"] == str(portable_root / "cache" / "pip")


def test_whitespace_only_roots_are_replaced(portable_root: Path) -> None:
    """UNSLOTH_HOME and UNSLOTH_LLAMA_CPP_PATH follow the same rule.

    _portable_master_root() strips UNSLOTH_HOME before using it, so keeping a
    whitespace-only value here would export a root the CLI itself ignored.
    """
    result = _run_probe(
        portable_root,
        portable_root.parents[1] / "home",
        {"UNSLOTH_HOME": " ", "UNSLOTH_LLAMA_CPP_PATH": " "},
    )
    env = result["env"]
    assert result["master"] == str(portable_root)
    assert env["UNSLOTH_HOME"] == str(portable_root)
    assert env["UNSLOTH_LLAMA_CPP_PATH"] == str(portable_root / "llama.cpp")


def test_whitespace_only_studio_home_does_not_suppress_the_resolved_root(tmp_path: Path) -> None:
    """A blank UNSLOTH_STUDIO_HOME is already ignored by _resolve_studio_home."""
    root = tmp_path / "opt" / "unsloth"
    _make_portable_install(root)
    home = tmp_path / "home"
    home.mkdir()
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "_REPO": str(REPO_ROOT),
        "UNSLOTH_STUDIO_HOME": " ",
        "UNSLOTH_HOME": str(root),
    }
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], env = env, capture_output = True, text = True, timeout = 300
    )
    payload = next(
        (
            json.loads(line[len("__JSON__") :])
            for line in proc.stdout.splitlines()
            if line.startswith("__JSON__")
        ),
        None,
    )
    assert payload is not None, f"{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    assert payload["env"]["UNSLOTH_STUDIO_HOME"] == payload["studio_home"]
    assert payload["env"]["UNSLOTH_STUDIO_HOME"].strip()


def test_genuine_overrides_are_preserved(portable_root: Path) -> None:
    """Only whitespace changes: a real user value still wins."""
    custom_uv = str(portable_root.parents[1] / "elsewhere" / "uv")
    result = _run_probe(
        portable_root,
        portable_root.parents[1] / "home",
        {"UV_CACHE_DIR": custom_uv},
    )
    assert result["env"]["UV_CACHE_DIR"] == custom_uv
    assert result["env"]["CUDA_CACHE_PATH"] == str(portable_root / "cache" / "cuda")
