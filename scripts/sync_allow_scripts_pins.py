#!/usr/bin/env python3
"""Keep `allowScripts` pins in studio/frontend/package.json in sync with
package-lock.json.

`npm approve-scripts` writes version-pinned entries ("pkg@1.2.3": true).
A dependency bump strands the pin, so the approval (or denial) silently
stops matching and the package's install scripts fall back to
"unreviewed". This tool re-pins existing entries to the versions the
lockfile actually resolves; it never adds or removes entries, so
approving a brand-new script-bearing package stays a human decision.

Usage:
  python scripts/sync_allow_scripts_pins.py --check   # CI: exit 1 on drift
  python scripts/sync_allow_scripts_pins.py --fix     # rewrite package.json

Pinned keys follow npm's allowScripts grammar: "name@1.2.3" or
"name@1.2.3 || 1.2.4". Bare names (no version) match every version and
are left alone. Entries whose range is not an exact-version disjunction
(wildcards, tags) are left alone too.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO_ROOT / "studio" / "frontend"

EXACT_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?$")


def split_spec(key: str) -> tuple[str, str | None]:
    """'@scope/name@1.2.3' -> ('@scope/name', '1.2.3'); bare names -> (key, None)."""
    if key.startswith("@"):
        rest = key[1:]
        if "@" not in rest:
            return key, None
        name, rng = rest.split("@", 1)
        return "@" + name, rng
    if "@" not in key:
        return key, None
    name, rng = key.split("@", 1)
    return name, rng


def is_exact_disjunction(rng: str) -> bool:
    parts = [p.strip() for p in rng.split("||")]
    return all(EXACT_VERSION_RE.match(p) for p in parts) and bool(parts)


def version_sort_key(version: str) -> tuple:
    release = version.split("-", 1)[0].split("+", 1)[0]
    return tuple(int(x) for x in release.split(".")), version


def script_versions_from_lock(lock: dict) -> dict[str, list[str]]:
    """Map package name -> sorted versions that carry install scripts."""
    out: dict[str, set[str]] = {}
    for path, meta in (lock.get("packages") or {}).items():
        if not path or not meta.get("hasInstallScript"):
            continue
        name = path.rsplit("node_modules/", 1)[-1]
        version = meta.get("version")
        if name and version:
            out.setdefault(name, set()).add(version)
    return {n: sorted(vs, key = version_sort_key) for n, vs in out.items()}


def desired_key(name: str, versions: list[str]) -> str:
    return f"{name}@{' || '.join(versions)}"


def compute_renames(policy: dict, lock_versions: dict[str, list[str]]) -> dict[str, str]:
    renames: dict[str, str] = {}
    for key in policy:
        name, rng = split_spec(key)
        if rng is None or not is_exact_disjunction(rng):
            continue  # bare name or non-exact spec: matches by name, never stale
        versions = lock_versions.get(name)
        if not versions:
            continue  # package gone or script-free now: stale pin is inert
        want = desired_key(name, versions)
        if key != want:
            renames[key] = want
    return renames


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    mode = ap.add_mutually_exclusive_group(required = True)
    mode.add_argument("--check", action = "store_true", help = "exit 1 if pins are stale")
    mode.add_argument("--fix", action = "store_true", help = "rewrite package.json in place")
    ap.add_argument(
        "--dir",
        type = Path,
        default = DEFAULT_DIR,
        help = "directory holding package.json + package-lock.json",
    )
    args = ap.parse_args(argv)

    pkg_path = args.dir / "package.json"
    lock_path = args.dir / "package-lock.json"
    if not pkg_path.exists() or not lock_path.exists():
        print(f"sync-allow-scripts: nothing to do ({args.dir} has no package.json + lockfile)")
        return 0

    pkg = json.loads(pkg_path.read_text(encoding = "utf-8"))
    policy = pkg.get("allowScripts")
    if not isinstance(policy, dict) or not policy:
        print("sync-allow-scripts: no allowScripts policy in package.json, nothing to do")
        return 0

    lock = json.loads(lock_path.read_text(encoding = "utf-8"))
    renames = compute_renames(policy, script_versions_from_lock(lock))

    if not renames:
        print(f"sync-allow-scripts: {len(policy)} allowScripts entries in sync with the lockfile")
        return 0

    for old, new in renames.items():
        print(f'  stale pin: "{old}" -> "{new}"')

    if args.check:
        print(
            "sync-allow-scripts: pins are stale; run "
            "`python scripts/sync_allow_scripts_pins.py --fix` and commit the result"
        )
        return 1

    pkg["allowScripts"] = {renames.get(k, k): v for k, v in policy.items()}
    pkg_path.write_text(json.dumps(pkg, indent = 2, ensure_ascii = False) + "\n", encoding = "utf-8")
    print(
        f"sync-allow-scripts: re-pinned {len(renames)} entr{'y' if len(renames) == 1 else 'ies'} in {pkg_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
