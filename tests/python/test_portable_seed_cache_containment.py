#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A portable Data Designer run keeps the user's text inside the root.

`tmp_root()` is the documented exception to containment: short-lived scratch that
a portable install is allowed to put in the system temp dir, alongside the
XDG_RUNTIME_DIR socket. The unstructured seed cache was hanging off it and is not
scratch. Each file holds the FULL text of a .txt/.md the user uploaded, split
into a chunk_text column; UnstructuredSeedReader.get_dataset_uri() hands that
parquet to duckdb for the whole generation run rather than for a preview; and
nothing deletes it, including the route that removes the upload it was derived
from. So a portable run wrote a persistent copy of user data outside the selected
root, it survived `rm -rf <root>`, and it sat at one path shared with every other
user and installation on the machine.

Both directions are pinned, since "always redirect" would pass a one-sided
version of this:

  [1] portable: the cache is under the root, the parquet really lands there, and
      `rm -rf <root>` leaves nothing of it behind,
  [2] non-portable: the cache is exactly the system temp path it has always been,
      byte for byte,
  [3] tmp_root() itself does NOT move in portable mode, so the exception it
      documents stays an exception and per-example scratch is not churned onto a
      removable volume,
  [4] hub.utils.paths.tmp_root() answers identically in both modes, which is what
      collapsing that duplicate onto the canonical one buys.

Subprocess per case: storage_roots reads the environment at import, and
chunking.py captures the cache directory at import too, so a monkeypatch after
the fact would measure nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BACKEND = REPO / "studio" / "backend"
CHUNKING = (
    BACKEND
    / "plugins"
    / "data-designer-unstructured-seed"
    / "src"
    / "data_designer_unstructured_seed"
    / "chunking.py"
)
MARKER = ".unsloth-portable-root"
OWNED = ".unsloth-studio-owned"

# A string that appears nowhere else, so finding it on disk is proof it came from
# the seed the probe uploaded and not from some other artefact.
SECRET = "CONFIDENTIAL-SEED-MARKER-8865-do-not-leak"

PROBE = r"""
import importlib.util, json, os, sys
sys.prefix = sys.exec_prefix = os.environ["_PREFIX"]
sys.path.insert(0, os.environ["_BACKEND"])
from utils.paths import storage_roots as sr
import hub.utils.paths as hub

out = {
    "portable": sr.portable_mode(),
    "studio_root": str(sr.studio_root()),
    "tmp_root": str(sr.tmp_root()),
    "hub_tmp_root": str(hub.tmp_root()),
    "seed_cache": str(sr.unstructured_seed_cache_root()),
    "oxc": str(sr.oxc_validator_tmp_root()),
    "parquet": None,
}

if os.environ.get("_MATERIALIZE") == "1":
    spec = importlib.util.spec_from_file_location("chunking", os.environ["_CHUNKING"])
    chunking = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(chunking)
    # Where routes/data_recipe/seed.py puts an upload: inside the Studio root.
    uploads = sr.datasets_root() / "unstructured-uploads" / "blk1"
    uploads.mkdir(parents = True, exist_ok = True)
    source = uploads / "notes.extracted.txt"
    source.write_text(os.environ["_SECRET"] + "\n" + ("filler sentence. " * 400))
    parquet, _rows = chunking.materialize_unstructured_seed_dataset(
        source_path = source, chunk_size = 1200, chunk_overlap = 200,
    )
    out["parquet"] = str(parquet)
    out["parquet_exists"] = parquet.exists()

print("__JSON__" + json.dumps(out))
"""

FAILS: list[str] = []


def check(label: str, expected, actual) -> None:
    if expected == actual:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} : expected [{expected}] got [{actual}]")
        FAILS.append(label)


def _probe(
    prefix: Path,
    home: Path,
    systmp: Path,
    *,
    materialize: bool = False,
) -> dict:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "USERPROFILE": str(home),
        # tempfile.gettempdir() reads these, so the "system" temp dir is one this
        # test owns rather than the machine's real /tmp.
        "TMPDIR": str(systmp),
        "TMP": str(systmp),
        "TEMP": str(systmp),
        "_BACKEND": str(BACKEND),
        "_PREFIX": str(prefix),
        "_CHUNKING": str(CHUNKING),
        "_SECRET": SECRET,
        "_MATERIALIZE": "1" if materialize else "0",
    }
    proc = subprocess.run(
        [sys.executable, "-c", PROBE],
        env = env,
        capture_output = True,
        text = True,
        timeout = 600,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__JSON__"):
            return json.loads(line[len("__JSON__") :])
    raise RuntimeError(
        f"probe failed rc={proc.returncode}\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}"
    )


def _nested(master: Path) -> Path:
    """What `install.sh --root <master>` leaves behind. No environment is set:
    the documented launch path is an activated venv, which carries none."""
    (master / "studio" / "unsloth_studio" / "bin").mkdir(parents = True, exist_ok = True)
    (master / "studio" / "unsloth_studio" / OWNED).write_text("")
    (master / "share").mkdir(exist_ok = True)
    (master / "bin").mkdir(exist_ok = True)
    (master / "share" / "studio.conf").write_text(f"export UNSLOTH_HOME={master}\n")
    (master / MARKER).write_text(f"{master}\n")
    master.chmod(0o755)
    return master / "studio"


def _holds_secret(root: Path) -> list[str]:
    """Every file under *root* whose bytes contain the seed text."""
    found: list[str] = []
    if not root.exists():
        return found
    for path in root.rglob("*"):
        try:
            if path.is_file() and SECRET.encode() in path.read_bytes():
                found.append(str(path))
        except OSError:
            continue
    return found


def main() -> int:
    if not CHUNKING.is_file():
        print(f"SKIP: no chunking module at {CHUNKING}")
        return 0
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as exc:
        print(f"SKIP: parquet stack unavailable ({exc})")
        return 0

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td).resolve()
        home = tmp / "home"
        home.mkdir(mode = 0o700)
        systmp = tmp / "systemtemp"
        systmp.mkdir()
        global_tmp = systmp / "unsloth-studio"

        print("\n[1] portable: the seed cache is inside the root and dies with it")
        master = tmp / "UnslothPortable"
        master.mkdir()
        studio = _nested(master)
        r = _probe(studio / "unsloth_studio", home, systmp, materialize = True)
        check("portable mode on", True, r["portable"])
        check("Studio root is the installed one", str(studio), r["studio_root"])
        check(
            "seed cache under the root",
            str(studio / "cache" / "unstructured-seed-cache"),
            r["seed_cache"],
        )
        check("the parquet was really written", True, r.get("parquet_exists"))
        check(
            "the parquet is inside the root",
            True,
            Path(r["parquet"]).is_relative_to(master),
        )
        # The upload itself holds it too, so this asks that the DERIVED copy does
        # -- the point being that a copy exists at all, not that only one does.
        check("the parquet holds the seed text", True, r["parquet"] in _holds_secret(master))
        check("nothing of it in the system temp dir", [], _holds_secret(global_tmp))
        shutil.rmtree(master)
        check("after rm -rf <root>: the root is gone", False, master.exists())
        check(
            "after rm -rf <root>: no user text anywhere outside it",
            [],
            _holds_secret(tmp),
        )

        print("\n[2] tmp_root() itself is unchanged in portable mode")
        # The documented exception stays one. Redirecting it wholesale would put
        # every per-example OuteTTS wav and audio decode on a removable volume,
        # and leak them into the root when a run is killed, since nothing reaps
        # this directory.
        check("portable tmp_root is the system one", str(global_tmp), r["tmp_root"])
        check("portable oxc scratch follows it", str(global_tmp / "oxc-validator"), r["oxc"])
        check("hub agrees with the canonical tmp_root", r["tmp_root"], r["hub_tmp_root"])

        print("\n[3] a non-portable install keeps the system temp dir exactly as before")
        plain = tmp / "plain" / "studio"
        (plain / "unsloth_studio" / "bin").mkdir(parents = True)
        (plain / "share").mkdir()
        (plain / "share" / "studio.conf").write_text("")
        r = _probe(plain / "unsloth_studio", home, systmp, materialize = True)
        check("not portable", False, r["portable"])
        check("non-portable tmp_root", str(global_tmp), r["tmp_root"])
        check(
            "non-portable seed cache stays in the system temp dir",
            str(global_tmp / "unstructured-seed-cache"),
            r["seed_cache"],
        )
        check("the parquet was really written", True, r.get("parquet_exists"))
        check(
            "and it landed there",
            True,
            Path(r["parquet"]).is_relative_to(global_tmp / "unstructured-seed-cache"),
        )
        check("hub agrees with the canonical tmp_root", r["tmp_root"], r["hub_tmp_root"])

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
        return 1
    print("All portable seed cache containment checks passed.")
    return 0


def test_portable_seed_cache_stays_inside_the_root():
    assert main() == 0, "checks failed: " + ", ".join(FAILS)


if __name__ == "__main__":
    raise SystemExit(main())
