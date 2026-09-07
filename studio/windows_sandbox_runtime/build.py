# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Explicit development/CI build. Never imported by the tool launcher.

The compiler and SDK versions are pinned; output must be outside the sources.
This entrypoint builds diagnostic binaries, not qualified release artifacts.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "backend"))
from core.inference.windows_sandbox.artifacts import MSVC_VERSION, SDK_VERSION, SOURCE_NAMES
from core.inference.windows_sandbox.profiles import PYTHON_PROFILE, select_abi_adapter


def build(
    vs_root,
    sdk_root,
    output,
    python_home = None,
    detours_source = None,
):
    compiler = vs_root / "VC" / "Tools" / "MSVC" / MSVC_VERSION
    include = sdk_root / "Include" / SDK_VERSION
    libraries = sdk_root / "Lib" / SDK_VERSION
    cl = compiler / "bin" / "Hostx64" / "x64" / "cl.exe"
    required = (
        cl,
        compiler / "include",
        include / "um",
        include / "ucrt",
        include / "shared",
        libraries / "um" / "x64",
        libraries / "ucrt" / "x64",
    )
    if os.name != "nt" or any(not path.exists() for path in required):
        raise RuntimeError("Pinned MSVC x64 compiler or Windows SDK is unavailable.")
    output = output.resolve()
    if output == ROOT or ROOT in output.parents:
        raise ValueError("Build output must be outside the runtime sources.")
    output.mkdir(parents = True, exist_ok = True)
    args = [
        str(cl),
        "/nologo",
        "/std:c11",
        "/W4",
        "/WX",
        "/O2",
        "/MT",
        "/GS",
        "/guard:cf",
        "/sdl",
        "/DUNICODE",
        "/D_UNICODE",
        "/D_WIN32_WINNT=0x0A00",
        "/DWINVER=0x0A00",
    ]
    args.extend(
        f"/I{path}"
        for path in (
            compiler / "include",
            include / "ucrt",
            include / "um",
            include / "shared",
            ROOT / "src",
        )
    )
    linker = [
        "/MANIFEST:EMBED",
        f"/MANIFESTINPUT:{ROOT / 'src/runtime.manifest.xml'}",
        "/INCREMENTAL:NO",
        "/DYNAMICBASE",
        "/HIGHENTROPYVA",
        "/NXCOMPAT",
        "/guard:cf",
        "/MACHINE:X64",
        "/SUBSYSTEM:CONSOLE",
        f"/LIBPATH:{compiler / 'lib' / 'x64'}",
        f"/LIBPATH:{libraries / 'um' / 'x64'}",
        f"/LIBPATH:{libraries / 'ucrt' / 'x64'}",
        "kernel32.lib",
        "advapi32.lib",
        "OneCoreUAP.lib",
    ]
    env = dict(os.environ)
    # Keep compiler scratch on the explicitly selected output drive as well.
    # A full system TEMP volume must not corrupt manifest/resource conversion.
    scratch = output / "temp"
    scratch.mkdir(exist_ok = True)
    env["TEMP"] = env["TMP"] = str(scratch)
    for name in ("CL", "_CL_", "LINK", "_LINK_", "INCLUDE", "LIB", "LIBPATH"):
        env.pop(name, None)
    env["PATH"] = os.pathsep.join(
        (
            str(compiler / "bin/Hostx64/x64"),
            str(sdk_root / "bin" / SDK_VERSION / "x64"),
            env.get("PATH", ""),
        )
    )
    targets = [
        ("gate_driver", [ROOT / "src/gate.c", ROOT / "tests/native/gate_driver.c"], [], []),
        ("detach_fixture", [ROOT / "tests/native/detach_fixture.c"], ["/LD"], []),
        ("detach_control", [ROOT / "tests/native/detach_control.c"], [], []),
    ]
    if python_home is not None:
        if detours_source is None:
            raise ValueError("Python hosts require an explicit pinned Detours source checkout.")
        from build_activation_context import build as build_activation

        activation_output = output / "activation"
        build_activation(detours_source, activation_output, vs_root, sdk_root)
        activation_evidence = json.loads(
            (activation_output / "build.json").read_text(encoding = "utf-8")
        )
        detours_objects = [
            activation_output / (name + ".obj")
            for name in (
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
        ]
        header = (python_home / "include/patchlevel.h").read_text(encoding = "utf-8")
        version = tuple(
            int(re.search(rf"^#define\s+PY_{part}_VERSION\s+(\d+)", header, re.MULTILINE)[1])
            for part in ("MAJOR", "MINOR", "MICRO")
        )
        adapter = select_abi_adapter(implementation = "cpython", version = version, architecture = "x64")
        targets.append(
            (
                f"python_host-{adapter.identity}",
                [
                    ROOT / "src/gate.c",
                    ROOT / "src/host_config.c",
                    ROOT / "src/python_host.c",
                    ROOT / "src/activation_context.c",
                    ROOT / "src/activation_plan.c",
                    ROOT / "src/authority_audit.c",
                    *detours_objects,
                ],
                [
                    f"/I{python_home / 'include'}",
                    f"/I{activation_output / 'vendor/src'}",
                    f"/DUS_PY_MAJOR={adapter.major}",
                    f"/DUS_PY_MINOR={adapter.minor}",
                ],
                [
                    f"/NODEFAULTLIB:python{adapter.major}{adapter.minor}.lib",
                    "ws2_32.lib",
                    "psapi.lib",
                    "bcrypt.lib",
                ],
            )
        )
    for name, sources, compile_flags, link_flags in targets:
        if name.startswith("python_host-"):
            source_hashes = {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in SOURCE_NAMES
            }
            headers = {
                str(path.relative_to(python_home / "include")): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted((python_home / "include").rglob("*.h"))
            }
        command = [
            *args,
            *compile_flags,
            *map(str, sources),
            f"/Fe{output / (name + '.exe')}",
            "/link",
            *linker,
            *link_flags,
        ]
        subprocess.run(command, cwd = output, env = env, check = True, timeout = 120)
        if name.startswith("python_host-"):
            if source_hashes != {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in SOURCE_NAMES
            } or headers != {
                str(path.relative_to(python_home / "include")): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in sorted((python_home / "include").rglob("*.h"))
            }:
                raise RuntimeError("Native sources or CPython headers changed during compilation.")
            binary = (output / (name + ".exe")).read_bytes()
            evidence = {
                "schema": 1,
                "abi": adapter.identity,
                "headers": list(version),
                "header_digest": hashlib.sha256(
                    json.dumps(headers, sort_keys = True).encode()
                ).hexdigest(),
                "profile": PYTHON_PROFILE.digest,
                "protocol": PYTHON_PROFILE.protocol_version,
                "compiler": {
                    "version": MSVC_VERSION,
                    "sha256": hashlib.sha256(cl.read_bytes()).hexdigest(),
                },
                "sdk": SDK_VERSION,
                "sources": source_hashes,
                "binary": {"sha256": hashlib.sha256(binary).hexdigest(), "size": len(binary)},
                "detours": {
                    "commit": activation_evidence["detours_commit"],
                    "source_digest": hashlib.sha256(
                        json.dumps(activation_evidence["detours_sources"], sort_keys = True).encode()
                    ).hexdigest(),
                },
            }
            (output / (name + ".build.json")).write_text(
                json.dumps(evidence, sort_keys = True, separators = (",", ":")), encoding = "utf-8"
            )
    return output / ((targets[-1][0] if python_home is not None else "gate_driver") + ".exe")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--vs-root", type = Path, required = True)
    parser.add_argument("--sdk-root", type = Path, required = True)
    parser.add_argument("--output", type = Path, required = True)
    parser.add_argument(
        "--python-home", type = Path, help = "Build the matching host from static CPython headers"
    )
    parser.add_argument("--detours-source", type = Path)
    options = parser.parse_args()
    print(
        build(
            options.vs_root,
            options.sdk_root,
            options.output,
            options.python_home,
            options.detours_source,
        )
    )
