#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A flat sentinel has to NAME the venv it vouches for, in both resolvers.

`unsloth` is an ordinary word and `unsloth_studio` is an ordinary venv name, so a
reused `--root` can hold somebody's own `bin/unsloth` helper beside their own
`unsloth_studio` virtualenv. install.sh already refuses to read that pair as one
of our flat installs: `_resolve_studio_destinations` requires the in-venv
`.unsloth-studio-owned` marker, or a `share/studio.conf` carrying
`UNSLOTH_EXE='<venv>/bin/unsloth'`, or a `bin/unsloth` symlink resolving to
`<venv>/bin/unsloth`, or the generated wrapper's `exec '<venv>/bin/unsloth' "$@"`
line.

Both Python resolvers kept the older existence-only rule, so the installer and
the runtime disagreed in the direction that costs most: the installer creates
`<root>/studio`, while `storage_roots.studio_root()` and the CLI's `STUDIO_HOME`
select `<root>` and go on to launch or update the unrelated venv sitting there.

Non-vacuous by construction: the expected layout is not typed in, it is produced
by running the REAL `_resolve_studio_destinations` lifted out of install.sh
against the same fixture. A rewrite that weakens either side is a failure here
even if both sides are weakened the same way, because [2] pins the genuine flat
install and [3] pins each sentinel on its own.

Subprocess per case: both resolvers read the environment at import time.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio" / "backend"
INSTALL = REPO / "install.sh"
OWNED = ".unsloth-studio-owned"

PROBE = r"""
import json, os, sys
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
out = {"backend": str(sr.studio_root())}
sys.path.insert(0, os.environ["_REPO"])
from unsloth_cli.commands import studio as cli
out["cli"] = str(cli.STUDIO_HOME)
print("__JSON__" + json.dumps(out))
"""

FAILS: list[str] = []


def _resolver_blocks() -> tuple[str, str]:
    """`_trim_ws` and `_resolve_studio_destinations`, lifted out of install.sh.

    Same extraction tests/sh/test_install_flat_layout_ownership.sh uses, so the
    expected layout below comes from the installer itself rather than from this
    file's opinion of it.
    """
    text = INSTALL.read_text(encoding = "utf-8", errors = "replace")
    trim = next(line for line in text.splitlines() if line.startswith("_trim_ws() "))
    body = re.search(
        r"^_resolve_studio_destinations\(\) \{$.*?^\}$", text, re.MULTILINE | re.DOTALL
    )
    assert body is not None, "could not extract _resolve_studio_destinations from install.sh"
    resolve = body.group(0)
    assert "_PORTABLE_FLAT=true" in resolve, "the extracted resolver lost its flat branch"
    assert "UNSLOTH_EXE='" in resolve, "the extracted resolver no longer matches studio.conf"
    assert "exec '" in resolve, "the extracted resolver no longer matches the generated shim"
    assert "-ef" in resolve, "the extracted resolver no longer resolves a bin/unsloth symlink"
    return trim, resolve


def installer_layout(root: Path, home: Path) -> str:
    """STUDIO_HOME install.sh itself would choose for `--root <root>`."""
    trim, resolve = _resolver_blocks()
    script = (
        "set -e\n"
        f"{trim}\n"
        f"{resolve}\n"
        "substep() { :; }\n"
        "_PORTABLE_MODE=true\n"
        "_PORTABLE_FLAT=false\n"
        '_UNSLOTH_ROOT="$_RSD_ROOT"\n'
        "_resolve_studio_destinations\n"
        'printf "%s\\n" "$STUDIO_HOME"\n'
    )
    proc = subprocess.run(
        ["sh", "-c", script],
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(home),
            "USER": os.environ.get("USER", "tester"),
            "_RSD_ROOT": str(root),
        },
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert proc.returncode == 0, f"installer resolver failed: {proc.stderr[-2000:]}"
    return proc.stdout.strip()


def resolvers(root: Path, home: Path) -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        "_BACKEND": str(BACKEND),
        "_REPO": str(REPO),
        "UNSLOTH_HOME": str(root),
    }
    proc = subprocess.run(
        [sys.executable, "-c", PROBE], env = env, capture_output = True, text = True, timeout = 300
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    raise RuntimeError(
        f"probe failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )


def agree(label: str, root: Path, home: Path, *, expect_flat: bool) -> None:
    """The installer and both resolvers must name the same Studio root."""
    installer = installer_layout(root, home)
    wanted = str(root) if expect_flat else str(root / "studio")
    got = resolvers(root, home)
    for who, value in (("installer", installer), ("backend", got["backend"]), ("CLI", got["cli"])):
        if value == wanted:
            print(f"  PASS  {label}: {who}")
        else:
            print(f"  FAIL  {label}: {who} expected [{wanted}] got [{value}]")
            FAILS.append(f"{label}/{who}")


def sq(value: str) -> str:
    """The `'` -> `'\\''` escaping every install.sh writer applies."""
    return value.replace("'", "'\\''")


def gen_conf(root: Path, venv: Path) -> None:
    """create_studio_shortcuts' share/studio.conf, byte for byte."""
    (root / "share").mkdir(parents = True, exist_ok = True)
    (root / "share" / "studio.conf").write_text(
        f"UNSLOTH_EXE='{sq(str(venv))}/bin/unsloth'\nexport UNSLOTH_PORTABLE=1\n"
    )


def gen_portable_shim(root: Path, venv: Path) -> None:
    """The wrapper the --portable block generates, ending in its exec line."""
    (root / "bin").mkdir(parents = True, exist_ok = True)
    shim = root / "bin" / "unsloth"
    shim.write_text(
        "#!/bin/sh\n"
        "# Generated by install.sh --portable. Keeps every Unsloth path inside\n"
        f"export UNSLOTH_HOME='{sq(str(root))}'\n"
        "export UNSLOTH_PORTABLE=1\n"
        f"exec '{sq(str(venv))}/bin/unsloth' \"$@\"\n"
    )
    shim.chmod(0o755)


def gen_symlink_shim(root: Path, venv: Path) -> None:
    """The `ln -sfn` a non-portable env-mode install leaves at bin/unsloth."""
    (root / "bin").mkdir(parents = True, exist_ok = True)
    (venv / "bin").mkdir(parents = True, exist_ok = True)
    (venv / "bin" / "unsloth").write_text("")
    (root / "bin" / "unsloth").symlink_to(venv / "bin" / "unsloth")


def main() -> int:
    FAILS.clear()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        home = tmp / "home"
        home.mkdir()
        elsewhere = tmp / "elsewhere" / "unsloth_studio"
        elsewhere.mkdir(parents = True)
        counter = {"n": 0}

        def new_root(name: str) -> Path:
            counter["n"] += 1
            root = tmp / f"{counter['n']:02d}-{name}"
            root.mkdir()
            return root

        def dev_venv(root: Path) -> Path:
            """Somebody's own virtualenv that happens to be called unsloth_studio."""
            venv = root / "unsloth_studio"
            (venv / "lib").mkdir(parents = True)
            (venv / "pyvenv.cfg").write_text("")
            return venv

        print("\n[1] a sentinel of the right NAME that is not ours stays NESTED")
        # The reported case: the user's own helper script beside their own venv.
        r = new_root("foreign-shim")
        dev_venv(r)
        (r / "bin").mkdir()
        (r / "bin" / "unsloth").write_text("#!/bin/sh\n# my own helper\necho hi\n")
        (r / "bin" / "unsloth").chmod(0o755)
        agree("unrelated bin/unsloth script", r, home, expect_flat = False)

        r = new_root("foreign-conf")
        dev_venv(r)
        (r / "share").mkdir()
        (r / "share" / "studio.conf").write_text("some other tool wrote this\n")
        agree("unrelated share/studio.conf", r, home, expect_flat = False)

        # Our own generated shapes, naming a DIFFERENT venv. Evidence for some
        # other install is not evidence for this one.
        r = new_root("names-elsewhere")
        dev_venv(r)
        gen_conf(r, elsewhere)
        gen_portable_shim(r, elsewhere)
        agree("sentinels naming another venv", r, home, expect_flat = False)

        r = new_root("symlink-elsewhere")
        dev_venv(r)
        (r / "bin").mkdir()
        (elsewhere / "bin").mkdir(parents = True, exist_ok = True)
        (elsewhere / "bin" / "unsloth").write_text("")
        (r / "bin" / "unsloth").symlink_to(elsewhere / "bin" / "unsloth")
        agree("bin/unsloth symlink into another tree", r, home, expect_flat = False)

        # Whole-line matching, not substring: a conf that merely MENTIONS the exe
        # is not the record create_studio_shortcuts writes.
        r = new_root("mentions-only")
        venv = dev_venv(r)
        (r / "share").mkdir()
        (r / "share" / "studio.conf").write_text(
            f"# see UNSLOTH_EXE='{venv}/bin/unsloth' for details\n"
        )
        agree("a conf that only mentions the exe", r, home, expect_flat = False)

        print("\n[2] the older existence-only cases still stay NESTED")
        r = new_root("empty-leftover")
        (r / "unsloth_studio").mkdir()
        agree("empty leftover named unsloth_studio", r, home, expect_flat = False)

        r = new_root("nested-with-stray")
        (r / "studio" / "unsloth_studio").mkdir(parents = True)
        (r / "unsloth_studio").mkdir()
        gen_conf(r, r / "studio" / "unsloth_studio")
        gen_portable_shim(r, r / "studio" / "unsloth_studio")
        agree("a stray venv beside a real nested install", r, home, expect_flat = False)

        print("\n[3] the requirement must not collapse into 'never flat'")
        for sentinel in ("owner", "conf", "shim", "link"):
            r = new_root(f"flat-{sentinel}")
            venv = r / "unsloth_studio"
            (venv / "bin").mkdir(parents = True)
            if sentinel == "owner":
                (venv / OWNED).write_text("")
            elif sentinel == "conf":
                gen_conf(r, venv)
            elif sentinel == "shim":
                gen_portable_shim(r, venv)
            else:
                gen_symlink_shim(r, venv)
            agree(f"the {sentinel} sentinel alone selects flat", r, home, expect_flat = True)

        # The writers single-quote every path and escape `'` as `'\''`. A root
        # with an apostrophe is where the two conventions would diverge, and
        # divergence means a genuine install stops recognising itself.
        for sentinel in ("conf", "shim"):
            r = tmp / f"o'brien-{sentinel}"
            r.mkdir()
            venv = r / "unsloth_studio"
            (venv / "bin").mkdir(parents = True)
            (gen_conf if sentinel == "conf" else gen_portable_shim)(r, venv)
            agree(f"an apostrophe in the root keeps the {sentinel} readable", r, home,
                  expect_flat = True)

        print("\n[4] the nested layout install.sh builds by default")
        r = new_root("plain-nested")
        (r / "studio" / "unsloth_studio").mkdir(parents = True)
        gen_conf(r, r / "studio" / "unsloth_studio")
        agree("plain nested install", r, home, expect_flat = False)

        r = new_root("fresh")
        agree("a root with no venv at all", r, home, expect_flat = False)

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("Installer and both resolvers agree on every flat-layout fixture.")
    return 0


@pytest.mark.skipif(shutil.which("sh") is None, reason = "needs a POSIX sh to run install.sh")
def test_flat_sentinels_must_identify_the_candidate_venv():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
