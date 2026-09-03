#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The in-root master root record is a path, so it is read as the filesystem
spells one.

install.sh writes the record verbatim (`printf '%s\\n' "$UNSLOTH_ROOT"`) and a
POSIX path is a byte string, not text. Read back through errors="replace", a root
holding a byte that is not valid UTF-8 -- which every POSIX filesystem permits,
and which a directory copied off a latin-1 or Shift-JIS volume really carries --
came back as U+FFFD, named a directory that does not exist, and was refused. The
record is the ONLY signal left on the case it was added for, the group-writable
master root whose parent marker _parent_marker_is_trustworthy deliberately does
not believe, so refusing it there sends the whole install back to ~/.unsloth: the
caches, the projects root and studio.db land outside the root that was selected,
and the CLI exports that answer into the backend.

Every direction is pinned, since a reader that decoded nothing and accepted
everything would pass a one-sided version of this:

  [1] a non-UTF-8 root resolves, in the backend and in the CLI (the fix),
  [2] an ordinary ASCII root still resolves (not "accept anything"),
  [3] a multi-line record is still refused (the truncated-prefix fix holds),
  [4] surrounding ASCII whitespace is still stripped, and a NON-ASCII space is
      still not (the installer refuses a root that its own `sed`-based _trim_ws
      would change, and leaves U+00A0 alone, so the reader must too or the two
      sides of the record disagree again).

Subprocess per case, as the sibling record tests do: these resolvers read
sys.prefix and the environment at import time. POSIX only -- Windows has no
non-UTF-8 filenames to decode, and install.ps1 refuses portable mode there.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio" / "backend"
MARKER = b".unsloth-portable-root"
RECORD = b".unsloth-master-root"
OWNED = b".unsloth-studio-owned"

# latin-1 as the transport for every path in and out of the probes: it is the one
# codec that round-trips all 256 byte values, so a name the test cannot spell in
# UTF-8 still survives a JSON hop.
BACKEND_PROBE = r"""
import json, os, sys
sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
master = sr.unsloth_home()
print("__JSON__" + json.dumps({
    "studio_root": os.fsencode(sr.studio_root()).decode("latin-1"),
    "unsloth_home": None if master is None else os.fsencode(master).decode("latin-1"),
    "portable": sr.portable_mode(),
}))
"""

CLI_PROBE = r"""
import json, os, sys
sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
sys.path.insert(0, os.environ["_REPO"])
from unsloth_cli.commands import studio as cli
master = cli._portable_master_root()
print("__JSON__" + json.dumps({
    "cli_master": None if master is None else os.fsencode(master).decode("latin-1"),
    "cli_studio_home": os.fsencode(cli.STUDIO_HOME).decode("latin-1"),
}))
"""

FAILS: list[str] = []


def check(label: str, expected, actual) -> None:
    if expected == actual:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} : expected [{expected}] got [{actual}]")
        FAILS.append(label)


def _probe(probe: str, prefix: bytes, home: bytes) -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.fsdecode(home),
        "USERPROFILE": os.fsdecode(home),
        "_BACKEND": str(BACKEND),
        "_REPO": str(REPO),
        "_PREFIX": os.fsdecode(prefix),
    }
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        env = env,
        capture_output = True,
        timeout = 300,
    )
    # Bytes, not text=True: a traceback naming one of these roots is itself
    # undecodable, and a crash in the probe must report as a failure rather than
    # as a UnicodeDecodeError in the harness.
    out = proc.stdout.decode("utf-8", "surrogateescape")
    err = proc.stderr.decode("utf-8", "surrogateescape")
    for line in out.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    raise RuntimeError(f"probe failed rc={proc.returncode}\n{out[-2000:]}\n{err[-3000:]}")


def _run(prefix: bytes, home: bytes) -> dict:
    # Two subprocesses: _ensure_studio_env_exported puts UNSLOTH_HOME into the
    # environment, and storage_roots reads that first, so one process would
    # measure the CLI twice.
    merged = _probe(BACKEND_PROBE, prefix, home)
    merged.update(_probe(CLI_PROBE, prefix, home))
    return merged


def _nested(master: bytes, *, record: bytes | None, mode: int = 0o775) -> bytes:
    """What `install.sh --root <master>` leaves behind, nested layout.

    0o775 by default: what `umask 002` produces on a multi-user box, where the
    parent marker fails the trust check and the in-root record is the only signal
    left. 0o700 keeps that marker believable, so a case about a REFUSED record
    can show it falling through to the marker rather than to nothing. The
    contents are passed in so a malformed record can be tested against the same
    layout; None writes the path itself, the way the installer does.
    """
    studio = master + b"/studio"
    os.makedirs(studio + b"/unsloth_studio/bin", exist_ok = True)
    with open(studio + b"/unsloth_studio/" + OWNED, "wb"):
        pass
    os.makedirs(master + b"/share", exist_ok = True)
    os.makedirs(master + b"/bin", exist_ok = True)
    with open(master + b"/share/studio.conf", "wb") as handle:
        handle.write(b"")
    with open(master + b"/" + MARKER, "wb") as handle:
        handle.write(master + b"\n")
    with open(studio + b"/" + RECORD, "wb") as handle:
        handle.write(master + b"\n" if record is None else record)
    os.chmod(master, mode)
    return studio


def main() -> int:
    if os.name == "nt":
        print("SKIP: POSIX-only (Windows has no non-UTF-8 filenames, and no portable mode)")
        return 0

    with tempfile.TemporaryDirectory() as td:
        tmp = os.fsencode(Path(td).resolve())
        home = tmp + b"/home"
        os.mkdir(home, mode = 0o700)
        legacy = (home + b"/.unsloth/studio").decode("latin-1")

        print("\n[1] roots whose names are not valid UTF-8 resolve to themselves")
        # 0xff is invalid UTF-8 in any position; 0x93 0xfa is the Shift-JIS
        # spelling of a common kanji, i.e. what a volume authored on Japanese
        # Windows and copied onto Linux really holds.
        cases = (
            ("latin-1 tail byte", b"vol-\xff"),
            ("shift-jis pair", b"vol-\x93\xfa"),
            ("truncated utf-8 lead", b"vol-\xe2\x82"),
        )
        for label, name in cases:
            master = tmp + b"/" + name
            studio = _nested(master, record = None)
            want = master.decode("latin-1")
            r = _run(studio + b"/unsloth_studio", home)
            check(f"{label}: backend finds the master root", want, r["unsloth_home"])
            check(f"{label}: portable mode on", True, r["portable"])
            check(
                f"{label}: Studio root is the installed one",
                studio.decode("latin-1"),
                r["studio_root"],
            )
            check(f"{label}: CLI agrees", want, r["cli_master"])
            check(
                f"{label}: CLI does not fall back to the home",
                False,
                r["cli_studio_home"] == legacy,
            )

        print("\n[2] an ordinary ASCII root is unaffected")
        master = tmp + b"/plain-root"
        studio = _nested(master, record = None)
        r = _run(studio + b"/unsloth_studio", home)
        check("ascii: backend finds the master root", master.decode("latin-1"), r["unsloth_home"])
        check("ascii: CLI agrees", master.decode("latin-1"), r["cli_master"])

        print("\n[3] a multi-line record is still refused, decoding or not")
        # The truncated-prefix case, in its genuine shape: master root
        # <tmp>/decoy\nevil, whose prefix <tmp>/decoy exists beside it. Reading
        # the first line alone hands UNSLOTH_HOME to that neighbour.
        # 0o700 so the parent marker stays believable and the refusal shows as a
        # fall-through to it, not as an install that lost every signal at once.
        os.mkdir(tmp + b"/decoy")
        master = tmp + b"/decoy\nevil"
        studio = _nested(master, record = None, mode = 0o700)
        r = _run(studio + b"/unsloth_studio", home)
        check(
            "multi-line: backend refuses it and takes the marker",
            master.decode("latin-1"),
            r["unsloth_home"],
        )
        check(
            "multi-line: and never the decoy",
            False,
            r["unsloth_home"] == (tmp + b"/decoy").decode("latin-1"),
        )
        check("multi-line: CLI refuses it too", master.decode("latin-1"), r["cli_master"])
        # And the same one line short of a newline: a record naming a real
        # directory, with a second line after it, must decline rather than win.
        master = tmp + b"/two-line"
        studio = _nested(master, record = master + b"\n" + tmp + b"/decoy\n")
        os.unlink(master + b"/" + MARKER)
        r = _run(studio + b"/unsloth_studio", home)
        check("two-line: backend declines", None, r["unsloth_home"])
        check("two-line: CLI declines", None, r["cli_master"])

        print("\n[4] the record and the installer agree on what whitespace is")
        # install.sh refuses a root that its own _trim_ws would change, so a
        # recorded path with surrounding ASCII space cannot come from us and is
        # read stripped, as it always was.
        master = tmp + b"/spaced"
        studio = _nested(master, record = b"  \t" + master + b" \n")
        r = _run(studio + b"/unsloth_studio", home)
        check(
            "ascii whitespace: still stripped",
            master.decode("latin-1"),
            r["unsloth_home"],
        )
        # U+00A0 is NOT whitespace to sed's [[:space:]], so the installer accepts
        # a root ending in one and records it. str.strip() would eat it and name
        # a different directory -- the exact disagreement the refusal exists to
        # prevent, arriving from the other side.
        master = tmp + b"/nbsp-\xc2\xa0"
        studio = _nested(master, record = None)
        r = _run(studio + b"/unsloth_studio", home)
        check(
            "non-ascii space: kept, so the root is the one installed into",
            master.decode("latin-1"),
            r["unsloth_home"],
        )
        check(
            "non-ascii space: CLI agrees",
            master.decode("latin-1"),
            r["cli_master"],
        )

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("All master root record decoding checks passed.")
    return 0


def test_master_record_is_decoded_with_filesystem_semantics():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
