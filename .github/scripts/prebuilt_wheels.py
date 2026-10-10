#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The build plan for the prebuilt CUDA wheels, and the release notes that describe them.

prebuilt-cuda-wheels.yml needs three things that are all awkward inside YAML and all worth a
unit test: a matrix expanded from free-text dispatch inputs, the upstream wheel filename for a
cell, and a release body regenerated from whatever is on the release right now. They live here
together because they share one table -- SPECS below -- and a drift between the name a cell
builds and the name the notes advertise is the one failure nobody would notice until a user's
pip install 404s.

Every value that reaches a shell command in the workflow is validated against that table rather
than interpolated from the dispatch input. The workflow is dispatch-only and gated, but an
input that becomes `git checkout $REF` is a command injection whether or not the door in front
of it is locked, so package, torch and python are resolved to known-good constants here and the
raw input is never used again.

Subcommands:

  matrix         read UW_PACKAGES / UW_TORCH_VERSIONS / UW_PYTHON_VERSIONS from the environment
                 and print the `include` list for the build matrix as JSON.
  wheel-name     print the single upstream-style filename for one cell.
  notes          read `sha256  filename` lines on stdin and print the release body.

Usage:
  prebuilt_wheels.py matrix
  prebuilt_wheels.py wheel-name --package flash-attn --torch 2.13.0 --python 3.13
  prebuilt_wheels.py notes --tag prebuilt-wheels-cu13 --repo unslothai/unsloth < SHA256SUMS
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Local version segment (cu13), not the toolkit patch level.
CUDA_TAG = "13"

CUDA_TOOLKIT = "13.0"

# torch 2.7+ wheels are cxx11 ABI only; the tag stays to match upstream filenames.
CXX11_ABI = "TRUE"

# Tagged linux_x86_64 like upstream; the practical glibc floor is the runner's, hence ubuntu-22.04.
PLATFORM_TAG = "linux_x86_64"

# Pinned to commits. flash-attn has no 2.8.4 release; the pin carries the c++20 switch torch 2.13 needs.
SPECS = {
    "flash-attn": {
        "dist": "flash_attn",
        "version": "2.8.4",
        "repo": "Dao-AILab/flash-attention",
        "ref": "edb5c76ee329b18ed95d1f7ea9aa522a1331ab7d",
        "submodules": True,
        # nvcc 13 OOMs above one job on a 16 GB hosted runner.
        "max_jobs": "1",
        "nvcc_threads": "2",
        "env": {
            "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
            "FLASH_ATTENTION_FORCE_CXX11_ABI": CXX11_ABI,
            # sm_86/89 run sm_80 cubins and PTX covers future cards; 110 is not targeted.
            "FLASH_ATTN_CUDA_ARCHS": "80;90;100;120",
        },
        # Shard the compile to stay below GitHub's 6-hour job limit.
        "shards": 8,
        # Leave time to upload partial caches after a build timeout.
        "build_timeout": "300m",
        "import_names": ["flash_attn", "flash_attn_2_cuda"],
    },
    "causal-conv1d": {
        "dist": "causal_conv1d",
        "version": "1.7.0",
        "repo": "Dao-AILab/causal-conv1d",
        "ref": "cd81f0413cad2fc1e6f17e785ac39f59aae690cd",
        "submodules": False,
        "max_jobs": "4",
        "nvcc_threads": "2",
        "env": {
            "CAUSAL_CONV1D_FORCE_BUILD": "TRUE",
        },
        "build_timeout": "120m",
        "import_names": ["causal_conv1d", "causal_conv1d_cuda"],
    },
    "mamba-ssm": {
        "dist": "mamba_ssm",
        "version": "2.3.2.post1",
        "repo": "state-spaces/mamba",
        "ref": "e9594ce1c732d97440f0332fdc43170a2294dbfa",
        "submodules": False,
        "max_jobs": "4",
        "nvcc_threads": "2",
        "env": {
            "MAMBA_FORCE_BUILD": "TRUE",
            # Mamba-1's selective-scan kernels are opt-in upstream and are the reason for this wheel.
            "MAMBA_KEEP_CUDA_BUILD": "TRUE",
        },
        # See patch_mamba_cxx20.py.
        "patch": "cxx20",
        "build_timeout": "180m",
        "import_names": ["mamba_ssm", "selective_scan_cuda"],
    },
}

TORCH_VERSIONS = ("2.13.0", "2.14.0")

# 3.14 is absent: nothing in the Unsloth stack is tested on it yet.
PYTHON_VERSIONS = ("3.11", "3.12", "3.13")

DEFAULT_PACKAGES = tuple(SPECS)
DEFAULT_TORCH = TORCH_VERSIONS
DEFAULT_PYTHON = ("3.13",)


def torch_minor(torch_version: str) -> str:
    """2.13.0 -> 2.13. The local version segment carries the minor and nothing finer."""
    major, minor = torch_version.split(".")[:2]
    return f"{major}.{minor}"


def python_tag(python_version: str) -> str:
    """3.13 -> cp313."""
    major, minor = python_version.split(".")[:2]
    return f"cp{major}{minor}"


def local_version(torch_version: str) -> str:
    """The `+cu13torch2.13cxx11abiTRUE` segment, byte for byte as upstream writes it."""
    return f"+cu{CUDA_TAG}torch{torch_minor(torch_version)}cxx11abi{CXX11_ABI}"


def wheel_name(package: str, torch_version: str, python_version: str) -> str:
    """The published filename for one cell.

    This is the whole point of the local version segment: pip refuses to install a wheel whose
    local version does not match what was requested, and a direct URL install of
    `...torch2.13...` into a torch 2.14 environment is a mistake the filename can prevent and
    an `undefined symbol` traceback at import time cannot.
    """
    spec = SPECS[package]
    tag = python_tag(python_version)
    return (
        f"{spec['dist']}-{spec['version']}{local_version(torch_version)}"
        f"-{tag}-{tag}-{PLATFORM_TAG}.whl"
    )


def _split(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace("\n", ",").split(",") if item.strip()]


def _resolve(raw: str, allowed, default, label: str) -> list[str]:
    """Free text in, allowlisted constants out, in the order the allowlist declares them.

    Order matters for more than tidiness: the matrix is emitted in this order and GitHub
    dispatches cells in it, so the longest build in the set starts first rather than last.
    """
    wanted = _split(raw) or list(default)
    unknown = [item for item in wanted if item not in allowed]
    if unknown:
        raise SystemExit(
            f"unknown {label}: {', '.join(sorted(unknown))}. " f"Allowed: {', '.join(allowed)}."
        )
    return [item for item in allowed if item in wanted]


def build_matrix(
    packages: str = "",
    torches: str = "",
    pythons: str = "",
) -> list[dict]:
    chosen_packages = _resolve(packages, DEFAULT_PACKAGES, DEFAULT_PACKAGES, "package")
    chosen_torch = _resolve(torches, TORCH_VERSIONS, DEFAULT_TORCH, "torch version")
    chosen_python = _resolve(pythons, PYTHON_VERSIONS, DEFAULT_PYTHON, "python version")

    include = []
    for torch_version in chosen_torch:
        for python_version in chosen_python:
            for package in chosen_packages:
                spec = SPECS[package]
                include.append(
                    {
                        "package": package,
                        "dist": spec["dist"],
                        "version": spec["version"],
                        "repo": spec["repo"],
                        "ref": spec["ref"],
                        "submodules": "recursive" if spec["submodules"] else "false",
                        "patch": spec.get("patch", ""),
                        "torch": torch_version,
                        "torch_mm": torch_minor(torch_version),
                        "python": python_version,
                        "python_tag": python_tag(python_version),
                        "cuda_tag": CUDA_TAG,
                        "abi": CXX11_ABI,
                        "max_jobs": spec["max_jobs"],
                        "nvcc_threads": spec["nvcc_threads"],
                        "build_timeout": spec["build_timeout"],
                        "shards": spec.get("shards", 0),
                        "build_env": " ".join(
                            f"{key}={value}" for key, value in spec["env"].items()
                        ),
                        "import_names": " ".join(spec["import_names"]),
                        "wheel_name": wheel_name(package, torch_version, python_version),
                        "label": f"{package} / torch {torch_minor(torch_version)} / "
                        f"{python_tag(python_version)}",
                    }
                )
    return include


def warm_matrix(include: list[dict]) -> list[dict]:
    """One warm job per (cell, shard), for the cells whose package is sharded."""
    return [
        {
            **cell,
            "shard": shard,
            "label": f"{cell['label']} / shard {shard + 1} of {cell['shards']}",
        }
        for cell in include
        for shard in range(cell["shards"])
    ]


def parse_wheel_name(name: str) -> dict | None:
    """Filename back to the facts the notes table needs, or None if it is not one of ours.

    Deliberately strict. The release holds SHA256SUMS and .sigstore.json bundles beside the
    wheels, and a loose parse would put them in the table as packages.
    """
    if not name.endswith(f"-{PLATFORM_TAG}.whl"):
        return None
    stem = name[: -len(f"-{PLATFORM_TAG}.whl")]
    parts = stem.split("-")
    if len(parts) != 4:
        return None
    dist, version_local, tag, abi_tag = parts
    if tag != abi_tag or "+" not in version_local:
        return None
    _, local = version_local.split("+", 1)
    if not local.startswith(f"cu{CUDA_TAG}torch") or "cxx11abi" not in local:
        return None
    torch_part, abi = local[len(f"cu{CUDA_TAG}torch") :].split("cxx11abi", 1)
    package = next((key for key, spec in SPECS.items() if spec["dist"] == dist), None)
    if package is None:
        return None
    return {
        "package": package,
        "dist": dist,
        "version": SPECS[package]["version"],
        "torch": torch_part,
        "cuda": CUDA_TAG,
        "python": tag,
        "abi": abi,
        "name": name,
    }


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def render_notes(entries: list[tuple[str, str]], tag: str, repo: str) -> str:
    """One-sentence release body from `(sha256, filename)` pairs.

    Regenerated from the release's current assets on every publish rather than appended to, so
    a second run that adds the torch 2.14 half produces a sentence describing both halves.
    """
    parsed = [row for row in (parse_wheel_name(name) for _, name in entries) if row is not None]
    if not parsed:
        return "No wheels are attached to this release yet.\n"
    present = {row["package"] for row in parsed}
    packages = [
        f"{package} {spec['version']}" for package, spec in SPECS.items() if package in present
    ]
    torches = sorted({row["torch"] for row in parsed}, key = lambda v: tuple(map(int, v.split("."))))
    pythons = [
        f"{cp[2]}.{cp[3:]}"
        for cp in sorted({row["python"] for row in parsed}, key = lambda cp: int(cp[3:]))
    ]
    return (
        f"Prebuilt Linux x86_64 CUDA {CUDA_TAG} wheels for {_join(packages)}, "
        f"built for PyTorch {_join(torches)} on Python {_join(pythons)}.\n"
    )


def _cmd_matrix(args: argparse.Namespace) -> int:
    include = build_matrix(
        packages = os.environ.get("UW_PACKAGES", ""),
        torches = os.environ.get("UW_TORCH_VERSIONS", ""),
        pythons = os.environ.get("UW_PYTHON_VERSIONS", ""),
    )
    matrix = json.dumps({"include": include}, separators = (",", ":"))
    warm = warm_matrix(include)
    print(matrix)

    # Written here to keep quote-heavy JSON out of a shell round trip.
    if args.github:
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a", encoding = "utf-8") as handle:
                handle.write(f"matrix={matrix}\n")
                handle.write(f"count={len(include)}\n")
                warm_json = json.dumps({"include": warm}, separators = (",", ":"))
                handle.write(f"warm_matrix={warm_json}\n")
                handle.write(f"warm_count={len(warm)}\n")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            listing = "\n".join(f"- `{cell['wheel_name']}`" for cell in include)
            with open(summary, "a", encoding = "utf-8") as handle:
                handle.write(f"### Build plan\n\n{len(include)} cells:\n\n{listing}\n")
    return 0


def _cmd_wheel_name(args: argparse.Namespace) -> int:
    if args.package not in SPECS:
        raise SystemExit(f"unknown package: {args.package}")
    if args.torch not in TORCH_VERSIONS:
        raise SystemExit(f"unknown torch version: {args.torch}")
    if args.python not in PYTHON_VERSIONS:
        raise SystemExit(f"unknown python version: {args.python}")
    print(wheel_name(args.package, args.torch, args.python))
    return 0


def _cmd_notes(args: argparse.Namespace) -> int:
    entries = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        if not name:
            digest, _, name = line.partition(" ")
        entries.append((digest.strip(), name.strip().lstrip("*")))
    sys.stdout.write(render_notes(entries, tag = args.tag, repo = args.repo))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    sub = parser.add_subparsers(dest = "command", required = True)

    matrix_parser = sub.add_parser("matrix", help = "print the build matrix as JSON")
    matrix_parser.add_argument(
        "--github",
        action = "store_true",
        help = "also append matrix/count and warm_matrix/warm_count to $GITHUB_OUTPUT and a "
        "listing to $GITHUB_STEP_SUMMARY",
    )
    matrix_parser.set_defaults(func = _cmd_matrix)

    name_parser = sub.add_parser("wheel-name", help = "print the filename for one cell")
    name_parser.add_argument("--package", required = True)
    name_parser.add_argument("--torch", required = True)
    name_parser.add_argument("--python", required = True)
    name_parser.set_defaults(func = _cmd_wheel_name)

    notes_parser = sub.add_parser("notes", help = "print the release body, digests on stdin")
    notes_parser.add_argument("--tag", required = True)
    notes_parser.add_argument("--repo", required = True)
    notes_parser.set_defaults(func = _cmd_notes)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
