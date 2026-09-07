# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Build the component test driver from a pinned, explicitly supplied Detours checkout.

No download, installation or production artifact publication occurs here.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
DETOURS_COMMIT = "adb07604aa56508448b95bf037c2a6d0d3b6831a"
MSVC = "14.44.35207"
SDK = "10.0.22621.0"


def build(detours, output, vs, sdk):
    detours, output = Path(detours).resolve(), Path(output).resolve()
    if output == ROOT or ROOT in output.parents or output == detours or detours in output.parents:
        raise ValueError("Use a separate build output directory.")
    revision = subprocess.check_output(
        ["git", "-C", str(detours), "rev-parse", "HEAD"], text = True
    ).strip()
    if revision != DETOURS_COMMIT:
        raise ValueError("Detours checkout does not match the pinned revision.")
    if subprocess.check_output(["git", "-C", str(detours), "diff", "HEAD", "--", "src"]):
        raise ValueError("Detours source has local modifications.")
    output.mkdir(parents = True, exist_ok = False)
    # Copy only tracked source bytes, preventing generated/untracked headers
    # from becoming compiler inputs. Build no pre-existing detours.lib.
    tracked = subprocess.check_output(
        ["git", "-C", str(detours), "ls-files", "-z", "src"], text = True
    ).split("\0")
    vendor = output / "vendor"
    hashes = {}
    for name in filter(None, tracked):
        data = subprocess.check_output(
            ["git", "-C", str(detours), "show", f"{DETOURS_COMMIT}:{name}"]
        )
        target = vendor / name
        target.parent.mkdir(parents = True, exist_ok = True)
        target.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    compiler = Path(vs) / "VC/Tools/MSVC" / MSVC
    sdk = Path(sdk)
    cl = compiler / "bin/Hostx64/x64/cl.exe"
    includes = [
        compiler / "include",
        *(sdk / "Include" / SDK / x for x in ("ucrt", "shared", "um")),
    ]
    libs = [compiler / "lib/x64", *(sdk / "Lib" / SDK / x / "x64" for x in ("ucrt", "um"))]
    env = dict(os.environ)
    for name in ("CL", "_CL_", "LINK", "_LINK_", "INCLUDE", "LIB", "LIBPATH"):
        env.pop(name, None)
    env["PATH"] = os.pathsep.join(
        [str(cl.parent), str(sdk / "bin" / SDK / "x64"), env.get("PATH", "")]
    )
    scratch = output / "temp"
    scratch.mkdir()
    env["TEMP"] = env["TMP"] = str(scratch)
    flags = [
        "/nologo",
        "/W4",
        "/WX",
        "/O2",
        "/MT",
        "/GS",
        "/guard:cf",
        *[f"/I{x}" for x in includes],
    ]
    names = (
        "detours",
        "modules",
        "disasm",
        "image",
        "creatwth",
        "disolx86",
        "disolx64",
        "disolia64",
        "disolarm",
        "disolarm64",
    )
    subprocess.run(
        [str(cl), *flags, "/c", *[str(vendor / "src" / (x + ".cpp")) for x in names]],
        cwd = output,
        env = env,
        timeout = 120,
        check = True,
    )
    sources = [ROOT / "src/activation_context.c", ROOT / "tests/native/activation_driver.c"]
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [*sources, ROOT / "src/activation_context.h"]
    }
    subprocess.run(
        [
            str(cl),
            *flags,
            "/std:c11",
            "/DUNICODE",
            "/D_UNICODE",
            "/D_WIN32_WINNT=0x0A00",
            f"/I{ROOT / 'src'}",
            f"/I{vendor / 'src'}",
            *map(str, sources),
            *[str(output / (x + ".obj")) for x in names],
            f"/Fe{output / 'activation_driver.exe'}",
            "/link",
            "/DYNAMICBASE",
            "/HIGHENTROPYVA",
            "/NXCOMPAT",
            "/guard:cf",
            *[f"/LIBPATH:{x}" for x in libs],
            "kernel32.lib",
            "advapi32.lib",
            "psapi.lib",
        ],
        cwd = output,
        env = env,
        timeout = 120,
        check = True,
    )
    binary = output / "activation_driver.exe"
    if any(
        hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
        for name, digest in source_hashes.items()
    ):
        raise RuntimeError("Activation adapter sources changed during compilation.")
    (output / "build.json").write_text(
        json.dumps(
            {
                "detours_commit": revision,
                "detours_sources": hashes,
                "sources": source_hashes,
                "msvc": MSVC,
                "sdk": SDK,
                "binary": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "qualification": False,
            },
            indent = 2,
        ),
        encoding = "utf-8",
    )
    return binary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--detours-source", required = True, type = Path)
    parser.add_argument("--output", required = True, type = Path)
    parser.add_argument("--vs-root", required = True, type = Path)
    parser.add_argument("--sdk-root", required = True, type = Path)
    args = parser.parse_args()
    print(build(args.detours_source, args.output, args.vs_root, args.sdk_root))
