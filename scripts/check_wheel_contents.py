#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Content checks for a built unsloth wheel + sdist, shared by wheel-smoke.yml and pypi-publish.yml.

Usage: check_wheel_contents.py [DIST_DIR]  (default: dist). Exit 0 pass, 1 a check failed,
2 an expected artifact is missing.
"""

import glob
import sys
import tarfile
import zipfile


def wheel_content_ok(wheel):
    print(f"wheel: {wheel}")
    with zipfile.ZipFile(wheel) as z:
        n = z.namelist()
        checks = {
            # The lockfile used to be asserted PRESENT. It is a build input for
            # `npm ci`, not something an installed Unsloth reads, so it now
            # has to be absent along with the rest of the pre-build tree.
            "no package-lock": not any(s.endswith("studio/frontend/package-lock.json") for s in n),
            "no frontend public": not any("studio/frontend/public/" in s for s in n),
            "no frontend src": not any("studio/frontend/src/" in s for s in n),
            "frontend dist shipped": any(s.endswith("studio/frontend/dist/index.html") for s in n),
            # The installers read these out of site-packages before falling back to GitHub.
            "shortcut icons shipped": all(
                any(s.endswith(f"studio/frontend/dist/{f}") for s in n)
                for f in ("rounded-512.png", "unsloth.ico")
            ),
            "no node_modules": not any("studio/frontend/node_modules/" in s for s in n),
            "no bun.lock": not any(s.endswith("studio/frontend/bun.lock") for s in n),
        }
        js = [
            s
            for s in n
            if "studio/frontend/dist/assets/" in s and s.endswith(".js") and "/index-" in s
        ]
        if not js:
            print("FAIL: no main bundle index-*.js in wheel")
            sys.exit(2)
        data = z.read(js[0]).decode("utf-8", "replace")
        hits = data.count("unstable_Provider:")
        print(f"main bundle: {js[0]}")
        print(f"unstable_Provider hits: {hits} (>=4 indicates 2026.5.1 regression)")
        checks["bundle has no Unsloth unstable_Provider call site"] = hits < 4

    print()
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return all(checks.values())


def _is_test(path):
    return "tests" in path.strip("/").split("/")


def _no_tests_report(label, names):
    offenders = sorted(n for n in names if _is_test(n))
    ioc = sorted(n for n in names if "malicious_" in n)
    print(f"{label}: {len(offenders)} test paths, {len(ioc)} IOC fixtures")
    for n in offenders[:20]:
        print(f"    {n}")
    if len(offenders) > 20:
        print(f"    ... and {len(offenders) - 20} more")
    for n in ioc:
        print(f"    IOC {n}")
    return not offenders and not ioc


def no_tests_ok(wheel, sdist):
    # tests/security/fixtures embeds a real supply-chain IOC literal, which got
    # the published sdist flagged by antivirus vendors (discussion #9577).
    # Both artifacts: `python -m build` builds the wheel FROM the sdist, so a
    # test suite reaching one reaches the other.
    ok = True
    with zipfile.ZipFile(wheel) as z:
        ok &= _no_tests_report(f"wheel {wheel}", z.namelist())
    with tarfile.open(sdist) as t:
        # Every sdist member carries an "unsloth-<version>/" prefix.
        ok &= _no_tests_report(
            f"sdist {sdist}", [n.split("/", 1)[1] for n in t.getnames() if "/" in n]
        )
    print("PASS" if ok else "FAIL: test files are being shipped")
    return ok


def main(argv):
    dist = argv[1] if len(argv) > 1 else "dist"
    wheels = glob.glob(f"{dist}/unsloth-*.whl")
    sdists = glob.glob(f"{dist}/unsloth-*.tar.gz")
    if not wheels or not sdists:
        print(f"FAIL: expected one wheel and one sdist in {dist}/")
        return 2
    content = wheel_content_ok(wheels[0])
    print()
    tests = no_tests_ok(wheels[0], sdists[0])
    return 0 if content and tests else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
