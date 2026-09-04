# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The sd.cpp auto-install must write into the root the finder reads, in every layout.

``tests/python/test_portable_flat_legacy_sd_root.py`` pins the two DERIVATIONS --
``sd_cpp_engine.managed_install_root`` and ``install_sd_cpp_prebuilt.default_install_dir`` --
against each other, but every case there exports UNSLOTH_STUDIO_HOME. A portable Studio started
the documented direct way (``uvicorn main:app`` from the repo, or an activated venv) inherits
none of the launcher's exports: the resolver still finds the master root, off the in-root
``.unsloth-master-root`` record or the marker beside it, while the installer's standalone
derivation sees no environment at all and answers the legacy ``~/.unsloth/stable-diffusion.cpp``.

That is a WRITE outside the one directory ``rm -rf <root>`` is advertised to remove, and the
finder never looks there, so the bundle is re-downloaded on every single load.

So this pins the target the BACKEND actually installs to, end to end through the real
``install()``, with only the network stubbed. Both entry points are covered
(``ensure_sd_cpp_binary`` and ``ensure_sd_server_binary``), and every layout is asserted with and
without the launcher's exports, so the answer cannot collapse into one path for all of them:

  (a) legacy non-portable            -> ~/.unsloth/stable-diffusion.cpp
  (b) flat portable elsewhere        -> <master>/stable-diffusion.cpp
  (c) flat portable AT ~/.unsloth/studio -> <root>/stable-diffusion.cpp   (not the sibling)
  (d) nested portable elsewhere      -> <master>/studio/stable-diffusion.cpp
  (e) nested master at ~/.unsloth    -> ~/.unsloth/stable-diffusion.cpp   (the sibling IS inside)

(c) and (e) are the two earlier fixes this must not undo: (c) because the flat master root owns
nothing above itself, (e) because a ``--root ~/.unsloth`` master does own that level, exactly
where its own llama.cpp, node and whisper.cpp sit.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

PORTABLE_MARKER = ".unsloth-portable-root"
STUDIO_OWNED_MARKER = ".unsloth-studio-owned"
MASTER_ROOT_RECORD = ".unsloth-master-root"

_PROBE = textwrap.dedent(
    """
    import json, os, sys, zipfile
    from pathlib import Path

    repo = Path(os.environ["_REPO"])
    # What a direct `uvicorn main:app` or an activated venv gives the resolvers: the venv the
    # installer built, and nothing else.
    sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "studio"))
    sys.path.insert(0, str(repo / "studio" / "backend"))

    from core.inference import sd_cpp_backend as be
    from core.inference.sd_cpp_engine import (
        find_sd_cpp_binary, find_sd_server_binary, managed_install_root,
    )
    from utils.paths.storage_roots import studio_root, unsloth_home

    mod = be._installer_module()

    # A bundle shaped like a release zip: both binaries, and a --help that identifies.
    bundle = Path(os.environ["_BUNDLE"])
    script = '#!/bin/sh\\necho "stable-diffusion.cpp version test"\\n'
    with zipfile.ZipFile(bundle, "w") as zf:
        zf.writestr("bin/sd-cli", script)
        zf.writestr("bin/sd-server", script)

    # Only the network is stubbed; install() itself runs for real, so the target it picks is the
    # one the product would pick.
    mod._resolve_with_fallback = lambda accelerator, token: (
        "unslothai/stable-diffusion.cpp",
        {"tag_name": "test", "assets": [
            {"name": "sd-test.zip", "browser_download_url": "stub://sd-test.zip"}]},
        "sd-test.zip",
    )
    mod._download = lambda url, dest, timeout = 300.0: dest.write_bytes(bundle.read_bytes())
    mod._verify_sha256 = lambda path, digest: None

    if os.environ["_MODE"] == "server":
        installed = be.ensure_sd_server_binary(allow_install = True, accelerator = "cpu")
    else:
        installed = be.ensure_sd_cpp_binary(allow_install = True, accelerator = "cpu")

    master = unsloth_home()
    print("__JSON__" + json.dumps({
        "studio_root": str(studio_root()),
        "master": None if master is None else str(master),
        "read": str(managed_install_root()),
        "installer_default": str(mod.default_install_dir()),
        "installed": installed,
        "found_cli": find_sd_cpp_binary(),
        "found_server": find_sd_server_binary(),
    }))
    """
)


def probe(home: Path, prefix: Path, tmp_path: Path, *, mode: str, **env_extra: str) -> dict:
    """Install once in a child started with exactly *env_extra*, and report where it landed.

    A child process, because both resolvers answer from the environment and the venv prefix the
    process started with. PATH is emptied so a system ``sd`` can never stand in for the install.
    """
    empty_path = tmp_path / "emptybin"
    empty_path.mkdir(exist_ok = True)
    env = {
        "PATH": str(empty_path),
        "HOME": str(home),
        "_REPO": str(REPO_ROOT),
        "_PREFIX": str(prefix),
        "_BUNDLE": str(tmp_path / f"bundle-{mode}.zip"),
        "_MODE": mode,
    }
    env.update({key: value for key, value in env_extra.items() if value})
    result = subprocess.run(
        [sys.executable, "-c", _PROBE], env = env, capture_output = True, text = True
    )
    line = next((ln for ln in result.stdout.splitlines() if ln.startswith("__JSON__")), None)
    detail = f"rc={result.returncode}\n{result.stdout[-3000:]}\n{result.stderr[-4000:]}"
    assert line is not None, detail
    return json.loads(line[len("__JSON__") :])


def make_venv(root: Path, *, owned: bool = False) -> Path:
    (root / "unsloth_studio" / "bin").mkdir(parents = True, exist_ok = True)
    (root / "unsloth_studio" / "pyvenv.cfg").write_text("home = /usr\n", encoding = "utf-8")
    if owned:
        (root / "unsloth_studio" / STUDIO_OWNED_MARKER).write_text("", encoding = "utf-8")
    return root / "unsloth_studio"


def build_layout(tmp_path: Path, layout: str) -> tuple[Path, Path, Path, dict]:
    """(home, venv prefix, expected sd.cpp root, launcher exports) for *layout*."""
    home = tmp_path / layout / "home"
    home.mkdir(parents = True)
    if layout == "legacy":
        # (a) The plain default install: no portable root anywhere.
        root = home / ".unsloth" / "studio"
        root.mkdir(parents = True)
        prefix = make_venv(root)
        return home, prefix, home / ".unsloth" / "stable-diffusion.cpp", {
            "UNSLOTH_STUDIO_HOME": str(root),
        }
    if layout == "flat_elsewhere":
        # (b) `install.sh --portable --root /opt/uns`: the venv sits AT the master root.
        master = tmp_path / layout / "opt" / "uns"
        master.mkdir(parents = True)
        prefix = make_venv(master, owned = True)
        (master / PORTABLE_MARKER).write_text("", encoding = "utf-8")
        return home, prefix, master / "stable-diffusion.cpp", {
            "UNSLOTH_HOME": str(master),
            "UNSLOTH_STUDIO_HOME": str(master),
        }
    if layout == "flat_legacy":
        # (c) `--portable` over UNSLOTH_STUDIO_HOME=~/.unsloth/studio: the LEGACY path is itself
        # the master root, so the ~/.unsloth sibling is outside everything uninstall removes.
        root = home / ".unsloth" / "studio"
        root.mkdir(parents = True)
        prefix = make_venv(root, owned = True)
        (root / PORTABLE_MARKER).write_text("", encoding = "utf-8")
        return home, prefix, root / "stable-diffusion.cpp", {
            "UNSLOTH_HOME": str(root),
            "UNSLOTH_STUDIO_HOME": str(root),
        }
    if layout == "nested_elsewhere":
        # (d) `install.sh --root /opt/uns`: Studio at <master>/studio, record inside it.
        master = tmp_path / layout / "opt" / "uns"
        (master / "studio").mkdir(parents = True)
        prefix = make_venv(master / "studio")
        (master / PORTABLE_MARKER).write_text("", encoding = "utf-8")
        (master / "studio" / MASTER_ROOT_RECORD).write_text(f"{master}\n", encoding = "utf-8")
        return home, prefix, master / "studio" / "stable-diffusion.cpp", {
            "UNSLOTH_HOME": str(master),
            "UNSLOTH_STUDIO_HOME": str(master / "studio"),
        }
    if layout == "nested_legacy":
        # (e) `install.sh --root ~/.unsloth`: the master OWNS ~/.unsloth, so the sibling is
        # contained and the legacy remap must survive.
        root = home / ".unsloth" / "studio"
        root.mkdir(parents = True)
        prefix = make_venv(root)
        (home / ".unsloth" / PORTABLE_MARKER).write_text("", encoding = "utf-8")
        (root / MASTER_ROOT_RECORD).write_text(f"{home / '.unsloth'}\n", encoding = "utf-8")
        return home, prefix, home / ".unsloth" / "stable-diffusion.cpp", {
            "UNSLOTH_HOME": str(home / ".unsloth"),
            "UNSLOTH_STUDIO_HOME": str(root),
        }
    raise AssertionError(layout)


LAYOUTS = ("legacy", "flat_elsewhere", "flat_legacy", "nested_elsewhere", "nested_legacy")

# The three environments one install can be launched in. "launcher" is run.py / the bin/unsloth
# wrapper; "master_only" is setup.sh, which exports UNSLOTH_HOME and not UNSLOTH_STUDIO_HOME;
# "bare" is the direct `uvicorn main:app` / activated-venv launch, which has neither.
EXPORTS = ("launcher", "master_only", "bare")


def exports_for(kind: str, env: dict) -> dict:
    if kind == "launcher":
        return env
    if kind == "master_only":
        return {"UNSLOTH_HOME": env["UNSLOTH_HOME"]} if "UNSLOTH_HOME" in env else {}
    return {}


@pytest.mark.parametrize("mode", ["cli", "server"])
@pytest.mark.parametrize("exports", EXPORTS)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_install_lands_in_the_root_the_finder_reads(tmp_path, layout, exports, mode):
    """The one assertion that matters: the bundle is written where discovery looks.

    Asserted on the real install's return value and on a fresh finder pass afterwards, not on the
    two derivations agreeing, because agreement is only the mechanism.
    """
    home, prefix, expected, env = build_layout(tmp_path / exports / mode, layout)
    found = probe(home, prefix, tmp_path, mode = mode, **exports_for(exports, env))

    stem = "sd-server" if mode == "server" else "sd-cli"
    assert found["read"] == str(expected)
    # Where install() actually put it.
    assert found["installed"] == str(expected / "bin" / stem)
    # And a finder pass started from scratch sees that tree, so the next load does not re-download.
    assert found["found_cli"] == str(expected / "bin" / "sd-cli")
    assert found["found_server"] == str(expected / "bin" / "sd-server")


@pytest.mark.parametrize("layout", ["flat_elsewhere", "flat_legacy", "nested_elsewhere"])
def test_a_bare_launch_never_installs_outside_the_portable_root(tmp_path, layout):
    """Containment, stated as itself: ``rm -rf <master>`` must take the sd.cpp tree with it.

    Separate from the table above because this is the promise, not the placement: a future change
    that moved the tree somewhere else INSIDE the root would still be honest, and one that put it
    back beside the root would not.
    """
    home, prefix, expected, env = build_layout(tmp_path, layout)
    master = Path(env["UNSLOTH_HOME"])
    found = probe(home, prefix, tmp_path, mode = "cli")

    assert found["master"] == str(master)
    assert Path(found["installed"]).is_relative_to(master)
    assert not Path(found["installed"]).is_relative_to(home / ".unsloth" / "stable-diffusion.cpp")


@pytest.mark.parametrize("layout", ["legacy", "nested_legacy"])
def test_the_legacy_sibling_remap_survives_a_bare_launch(tmp_path, layout):
    """The two earlier fixes, from the other side: a root that is NOT the master root keeps
    mapping to ``~/.unsloth/stable-diffusion.cpp``.

    (a) has no master root at all, and (e)'s master root is ``~/.unsloth`` itself, which owns the
    sibling exactly as it owns its llama.cpp, node and whisper.cpp. A fix keyed on "portable"
    rather than on "the Studio root IS the master root" would move (e)'s tree and strand the one
    already installed."""
    home, prefix, expected, _env = build_layout(tmp_path, layout)
    found = probe(home, prefix, tmp_path, mode = "cli")

    assert expected == home / ".unsloth" / "stable-diffusion.cpp"
    assert found["read"] == str(expected)
    assert found["installed"] == str(expected / "bin" / "sd-cli")


def test_the_installer_default_is_what_diverges_on_a_bare_nested_launch(tmp_path):
    """The defect this file exists for, named directly.

    ``default_install_dir()`` has to answer standalone, so it can only be TOLD about a portable
    root through the environment; a bare launch tells it nothing and it falls back to the legacy
    sibling. That is fine as the standalone CLI's fallback and fatal as the backend's, which is
    why the backend passes the resolved root instead of relying on the two agreeing.
    """
    home, prefix, expected, _env = build_layout(tmp_path, "nested_elsewhere")
    found = probe(home, prefix, tmp_path, mode = "cli")

    assert found["installer_default"] == str(home / ".unsloth" / "stable-diffusion.cpp")
    assert found["read"] != found["installer_default"]
    # The install followed the reader, not the standalone default.
    assert found["installed"] == str(expected / "bin" / "sd-cli")
