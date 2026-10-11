# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The macOS "Unsloth Studio.app" shortcut must start the launcher natively on Apple Silicon (#9467).

Its executable is a shell script, which Launch Services may run under Rosetta; a translated shell
then runs the x86_64 slice of a universal2 venv python against arm64 wheels and `unsloth` dies on
`incompatible architecture (have 'arm64', need 'x86_64')`. The live tests rebuild the bundle from
install.sh's own heredocs and compare it with the pre-fix shape, which must still come up
translated on the same host, or the run proves nothing and skips.
"""

import os
import platform
import plistlib
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO_ROOT / "install.sh"

ON_APPLE_SILICON = (
    sys.platform == "darwin"
    and subprocess.run(
        ["/usr/sbin/sysctl", "-in", "hw.optional.arm64"], capture_output = True, text = True
    ).stdout.strip()
    == "1"
)


def _heredoc(tag: str) -> str:
    text = INSTALL_SH.read_text(encoding = "utf-8")
    match = re.search(rf"<< '{tag}'\n(.*?)\n{tag}\n", text, re.S)
    assert match, f"install.sh lost its {tag} heredoc"
    return match.group(1) + "\n"


def _plist() -> dict:
    return plistlib.loads(_heredoc("PLIST_EOF").encode())


def _old_stub() -> str:
    return "#!/bin/sh\nexec '@@DATA_DIR@@/launch-studio.sh' \"$@\"\n"


def _old_plist() -> dict:
    plist = _plist()
    plist.pop("LSArchitecturePriority", None)
    return plist


def test_plist_asks_for_arm64_first() -> None:
    assert _plist()["LSArchitecturePriority"][0] == "arm64"


def test_stub_parses_and_keeps_a_plain_exec_for_intel() -> None:
    stub = _heredoc("STUB_EOF")
    subprocess.run(["sh", "-n", "/dev/stdin"], input = stub, text = True, check = True)
    assert "/usr/bin/arch -arm64 '@@DATA_DIR@@/launch-studio.sh'" in stub
    assert stub.rstrip().endswith("exec '@@DATA_DIR@@/launch-studio.sh' \"$@\"")


def _rosetta_available() -> bool:
    return (
        subprocess.run(
            ["/usr/bin/arch", "-x86_64", "/usr/bin/true"], capture_output = True
        ).returncode
        == 0
    )


needs_rosetta = pytest.mark.skipif(
    not ON_APPLE_SILICON or not _rosetta_available(),
    reason = "needs an Apple Silicon Mac with Rosetta 2 installed",
)


def _data_dir(tmp_path: Path, name: str) -> Path:
    data = tmp_path / name
    data.mkdir()
    probe = data / "launch-studio.sh"
    # proc_translated reads 1 only inside a Rosetta-translated process.
    probe.write_text(
        "#!/bin/bash\n"
        'printf "%s %s\\n" "$(/usr/sbin/sysctl -in sysctl.proc_translated)" "$(uname -m)" '
        f"> '{data}/arch.txt'\n",
        encoding = "utf-8",
    )
    probe.chmod(0o755)
    return data


def _write_stub(path: Path, stub: str, data: Path) -> None:
    path.write_text(stub.replace("@@DATA_DIR@@", str(data)), encoding = "utf-8")
    path.chmod(0o755)


def _read_arch(data: Path, timeout: float = 30.0) -> str:
    out = data / "arch.txt"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if out.exists() and out.read_text(encoding = "utf-8").strip():
            return out.read_text(encoding = "utf-8").strip()
        time.sleep(0.25)
    pytest.fail(f"launcher never wrote {out}")


@needs_rosetta
def test_stub_returns_a_translated_launch_to_arm64(tmp_path: Path) -> None:
    results = {}
    for name, stub in (("old", _old_stub()), ("new", _heredoc("STUB_EOF"))):
        data = _data_dir(tmp_path, name)
        stub_path = tmp_path / f"{name}-stub"
        _write_stub(stub_path, stub, data)
        subprocess.run(["/usr/bin/arch", "-x86_64", "/bin/sh", str(stub_path)], check = True)
        results[name] = _read_arch(data)
    assert results["old"].startswith("1 "), f"control did not run translated: {results}"
    assert results["new"] == "0 arm64", results


def _build_app(tmp_path: Path, name: str, plist: dict, stub: str) -> tuple[Path, Path]:
    data = _data_dir(tmp_path, name)
    app = tmp_path / f"{name}.app"
    (app / "Contents" / "MacOS").mkdir(parents = True)
    # A fresh bundle id per run: Launch Services caches its architecture choice per id.
    plist = dict(plist, CFBundleIdentifier = f"ai.unsloth.studio.test.{name}.{uuid.uuid4().hex}")
    (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps(plist))
    _write_stub(app / "Contents" / "MacOS" / plist["CFBundleExecutable"], stub, data)
    return app, data


@needs_rosetta
def test_app_bundle_launches_natively(tmp_path: Path) -> None:
    old_app, old_data = _build_app(tmp_path, "old", _old_plist(), _old_stub())
    new_app, new_data = _build_app(tmp_path, "new", _plist(), _heredoc("STUB_EOF"))
    subprocess.run(["open", "-n", str(old_app)], check = True)
    old = _read_arch(old_data)
    subprocess.run(["open", "-n", str(new_app)], check = True)
    new = _read_arch(new_data)
    print(f"pre-fix bundle: {old!r}, fixed bundle: {new!r}, macOS {platform.mac_ver()[0]}")
    assert new == "0 arm64", f"fixed bundle ran {new!r} (pre-fix bundle ran {old!r})"
    if not old.startswith("1 "):
        pytest.skip(
            reason = f"Launch Services ran the pre-fix bundle natively here ({old!r}): "
            "this host does not reproduce the Rosetta launch, only the fixed result is shown"
        )
