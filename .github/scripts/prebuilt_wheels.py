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

  matrix         read UW_PACKAGES / UW_TORCH_VERSIONS / UW_PYTHON_VERSIONS / UW_PLATFORMS from
                 the environment and print the `include` list for the build matrix as JSON.
  wheel-name     print the single upstream-style filename for one cell.
  notes          read `sha256  filename` lines on stdin and print the release body.
  sums           read `sha256  filename` lines on stdin and print SHA256SUMS for the wheels.

Usage:
  prebuilt_wheels.py matrix
  prebuilt_wheels.py wheel-name --package flash-attn --torch 2.13.0 --python 3.13 [--platform win_amd64]
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

# The Linux tag. The wheel is tagged linux_x86_64 rather than manylinux_*, exactly as upstream
# tags its own, so pip installs it on any glibc without a floor check; the practical floor is the
# runner's glibc, which is why the build runs on ubuntu-22.04 (glibc 2.35) and not on
# ubuntu-latest.
PLATFORM_TAG = "linux_x86_64"

# Keyed by the wheel's pip platform tag. No upstream release has ever shipped a Windows wheel.
# `abi` is torch._C._GLIBCXX_USE_CXX11_ABI, which upstream setup.py copies into the name; the
# Windows cu130 torch reports True too, and the build asserts it. `pytorch_nvcc` goes into
# build.ninja unquoted and is the only way to cache nvcc on Windows (_wrap_compiler is a no-op
# there), hence the space-free paths.
PLATFORMS = {
    "linux_x86_64": {
        "os": "linux",
        "runner": "ubuntu-22.04",
        "abi": "TRUE",
        "label": "",
        "pytorch_nvcc": "ccache /usr/local/cuda-13.0/bin/nvcc",
    },
    "win_amd64": {
        "os": "windows",
        # Visual Studio 2022 (MSVC 14.4x), a host compiler CUDA 13.0's Windows guide supports.
        "runner": "windows-2022",
        "abi": "TRUE",
        "label": " / win_amd64",
        "pytorch_nvcc": "C:/ccache/ccache.exe C:/cuda/bin/nvcc.exe",
    },
    # Upstream's aarch64 wheels stop at torch 2.10. ubuntu-22.04-arm keeps the glibc 2.35 floor
    # the x86_64 build has, and the build asserts it.
    "linux_aarch64": {
        "os": "linux",
        "runner": "ubuntu-22.04-arm",
        "abi": "TRUE",
        "label": " / linux_aarch64",
        "pytorch_nvcc": "ccache /usr/local/cuda-13.0/bin/nvcc",
    },
    # NVIDIA publishes flash-attn for Windows on ARM itself (studio resolves it), so only the other
    # two are built here, against the only CUDA torch that exists for it: NVIDIA's 2.14.0+cu134,
    # which needs CUDA 13.4, the first toolkit with a Windows ARM64 nvcc.
    "win_arm64": {
        "os": "windows",
        "runner": "windows-11-arm",
        "abi": "TRUE",
        "label": " / win_arm64",
        "pytorch_nvcc": "C:/ccache/ccache.exe C:/cuda/bin/nvcc.exe",
        "torches": ("2.14.0",),
        "packages": ("causal-conv1d", "mamba-ssm"),
        "torch_local": "cu134",
        "torch_index_args": "--extra-index-url https://pypi.nvidia.com/nvtorch_oot",
    },
}

# Per platform, what differs from the x86_64 defaults below without changing any x86_64 cell.
_PLATFORM_DEFAULTS = {
    "torches": None,  # every TORCH_VERSIONS entry
    "packages": None,  # every SPECS entry
    "torch_local": "",
    "torch_index_args": "--index-url https://download.pytorch.org/whl/cu130",
}
# What the build asserts about its own interpreter and output, so an emulated x64 Python on an
# ARM runner cannot quietly produce an x64 wheel under an arm64 name.
_PY_PLATFORM = {
    "linux_x86_64": "linux-x86_64",
    "win_amd64": "win-amd64",
    "linux_aarch64": "linux-aarch64",
    "win_arm64": "win-arm64",
}


def _platform(platform: str, key: str):
    return PLATFORMS[platform].get(key, _PLATFORM_DEFAULTS.get(key))


# torch 2.13 pins triton==3.7.1 and 2.14 triton~=3.8.0 (Linux-only markers); exact pins keep
# the Windows smoke test reproducible.
TRITON_WINDOWS = {"2.13": "3.7.1.post27", "2.14": "3.8.0.post29"}
WINDOWS_SMOKE_EXCLUDE = ("triton", "tilelang", "apache-tvm-ffi", "quack-kernels")

# Source revisions, pinned to a commit rather than a branch or a tag.
#
# flash-attn 2.8.4 does not exist as an upstream release: 2.8.3.post1 is the newest tag, and
# the version in flash_attn/__init__.py on main has already moved to 2.8.4. The pin is the
# commit that carries that version AND the c++20 switch (Dao-AILab/flash-attention#2899),
# which is what makes it build against torch 2.13 at all.
#
# mamba-ssm and causal-conv1d are pinned to the commit their released version was cut from.
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
# ARM is opt-in, so an ordinary dispatch builds exactly what it did before ARM existed.
DEFAULT_PLATFORMS = ("linux_x86_64", "win_amd64")


def torch_minor(torch_version: str) -> str:
    """2.13.0 -> 2.13. The local version segment carries the minor and nothing finer."""
    major, minor = torch_version.split(".")[:2]
    return f"{major}.{minor}"


def python_tag(python_version: str) -> str:
    """3.13 -> cp313."""
    major, minor = python_version.split(".")[:2]
    return f"cp{major}{minor}"


def local_version(torch_version: str, platform: str = PLATFORM_TAG) -> str:
    """The `+cu13torch2.13cxx11abiTRUE` segment, byte for byte as upstream writes it."""
    abi = PLATFORMS[platform]["abi"]
    return f"+cu{CUDA_TAG}torch{torch_minor(torch_version)}cxx11abi{abi}"


def wheel_name(
    package: str,
    torch_version: str,
    python_version: str,
    platform: str = PLATFORM_TAG,
) -> str:
    """The published filename for one cell.

    This is the whole point of the local version segment: pip refuses to install a wheel whose
    local version does not match what was requested, and a direct URL install of
    `...torch2.13...` into a torch 2.14 environment is a mistake the filename can prevent and
    an `undefined symbol` traceback at import time cannot.
    """
    spec = SPECS[package]
    tag = python_tag(python_version)
    return (
        f"{spec['dist']}-{spec['version']}{local_version(torch_version, platform)}"
        f"-{tag}-{tag}-{platform}.whl"
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


def _build_env(spec: dict, platform: str) -> str:
    """The package's NAME=VALUE pairs for one platform.

    The only per-platform value is flash-attn's FORCE_CXX11_ABI, which makes its setup.py set
    torch._C._GLIBCXX_USE_CXX11_ABI before it builds and names the wheel. It follows the platform's
    `abi`, so the name it writes always agrees with the one the build asserts. Today that is TRUE on
    both platforms, which on a torch already reporting True is a no-op.
    """
    abi = PLATFORMS[platform]["abi"]
    pairs = []
    for key, value in spec["env"].items():
        if key.endswith("FORCE_CXX11_ABI"):
            value = abi
        pairs.append(f"{key}={value}")
    return " ".join(pairs)


def _smoke(package: str, torch_version: str, platform: str) -> tuple[str, str]:
    """(dependencies to leave out, extra requirement to add) for the import smoke test."""
    if PLATFORMS[platform]["os"] != "windows" or package != "mamba-ssm":
        return "", ""
    return (
        " ".join(WINDOWS_SMOKE_EXCLUDE),
        f"triton-windows=={TRITON_WINDOWS[torch_minor(torch_version)]}",
    )


def build_matrix(
    packages: str = "",
    torches: str = "",
    pythons: str = "",
    platforms: str = "",
) -> list[dict]:
    chosen_packages = _resolve(packages, DEFAULT_PACKAGES, DEFAULT_PACKAGES, "package")
    chosen_torch = _resolve(torches, TORCH_VERSIONS, DEFAULT_TORCH, "torch version")
    chosen_python = _resolve(pythons, PYTHON_VERSIONS, DEFAULT_PYTHON, "python version")
    chosen_platforms = _resolve(platforms, tuple(PLATFORMS), DEFAULT_PLATFORMS, "platform")

    include = []
    for platform in chosen_platforms:
        target = PLATFORMS[platform]
        torches = _platform(platform, "torches") or TORCH_VERSIONS
        packages_here = _platform(platform, "packages") or tuple(SPECS)
        local = _platform(platform, "torch_local")
        for torch_version in chosen_torch:
            if torch_version not in torches:
                continue
            for python_version in chosen_python:
                for package in chosen_packages:
                    if package not in packages_here:
                        continue
                    spec = SPECS[package]
                    smoke_exclude, smoke_extra = _smoke(package, torch_version, platform)
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
                            "abi": target["abi"],
                            "max_jobs": spec["max_jobs"],
                            "nvcc_threads": spec["nvcc_threads"],
                            "build_timeout": spec["build_timeout"],
                            "shards": spec.get("shards", 0),
                            "build_env": _build_env(spec, platform),
                            "import_names": " ".join(spec["import_names"]),
                            "wheel_name": wheel_name(
                                package, torch_version, python_version, platform
                            ),
                            # Job name in the Actions UI; Linux keeps its old label.
                            "label": f"{package} / torch {torch_minor(torch_version)} / "
                            f"{python_tag(python_version)}{target['label']}",
                            "platform": platform,
                            "os": target["os"],
                            "runner": target["runner"],
                            "pytorch_nvcc": target["pytorch_nvcc"],
                            "smoke_exclude": smoke_exclude,
                            "smoke_extra": smoke_extra,
                            "torch_spec": f"torch=={torch_version}"
                            + (f"+{local}" if local else ""),
                            "torch_index_args": _platform(platform, "torch_index_args"),
                            "py_platform": _PY_PLATFORM[platform],
                        }
                    )
    if not include:
        raise SystemExit(
            "no cells: none of the chosen platforms builds that package / torch combination "
            "(win_arm64 builds causal-conv1d and mamba-ssm for torch 2.14.0 only)."
        )
    return include


def gpu_matrix(include: list[dict]) -> list[dict]:
    """Cells the (x86_64 Linux) self-hosted GPU runner can load."""
    return [cell for cell in include if cell["platform"] == "linux_x86_64"]


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
    wheels, and a loose parse would put them in the table as packages. The ABI must match the
    platform's, so a stray `cxx11abiFALSE` wheel is not one of ours.
    """
    platform = next((tag for tag in PLATFORMS if name.endswith(f"-{tag}.whl")), None)
    if platform is None:
        return None
    stem = name[: -len(f"-{platform}.whl")]
    parts = stem.split("-")
    if len(parts) != 4:
        return None
    dist, version_local, tag, abi_tag = parts
    if tag != abi_tag or "+" not in version_local:
        return None
    version, local = version_local.split("+", 1)
    if not local.startswith(f"cu{CUDA_TAG}torch") or "cxx11abi" not in local:
        return None
    torch_part, abi = local[len(f"cu{CUDA_TAG}torch") :].split("cxx11abi", 1)
    if abi != PLATFORMS[platform]["abi"]:
        return None
    package = next((key for key, spec in SPECS.items() if spec["dist"] == dist), None)
    if package is None:
        return None
    return {
        "package": package,
        "dist": dist,
        # From the file, not SPECS: an asset left over from an older pin is a different version.
        "version": version,
        "torch": torch_part,
        "cuda": CUDA_TAG,
        "python": tag,
        "abi": abi,
        "platform": platform,
        "name": name,
    }


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


# How the release body names each platform.
PLATFORM_NAMES = {
    "linux_x86_64": "Linux x86_64",
    "win_amd64": "Windows x86_64",
    "linux_aarch64": "Linux aarch64",
    "win_arm64": "Windows ARM64",
}


def _coverage(rows: list[dict]) -> str:
    """`PyTorch 2.13 and 2.14 on Python 3.13` for a set of parsed rows."""
    torches = sorted({row["torch"] for row in rows}, key = lambda v: tuple(map(int, v.split("."))))
    # cp313 -> 3.13
    pythons = [
        f"{cp[2]}.{cp[3:]}"
        for cp in sorted({row["python"] for row in rows}, key = lambda cp: int(cp[3:]))
    ]
    return f"PyTorch {_join(torches)} on Python {_join(pythons)}"


def _packages(rows: list[dict]) -> str:
    """`flash-attn 2.8.4 and mamba-ssm 2.3.2.post1` for a set of parsed rows, in SPECS order."""
    order = {package: index for index, package in enumerate(SPECS)}
    pairs = sorted(
        {(row["package"], row["version"]) for row in rows}, key = lambda pv: (order[pv[0]], pv[1])
    )
    return _join([f"{package} {version}" for package, version in pairs])


def render_notes(entries: list[tuple[str, str]], tag: str, repo: str) -> str:
    """One-sentence release body from `(sha256, filename)` pairs.

    Regenerated from the release's current assets on every publish, so a run that adds one
    torch minor or one platform still describes everything attached.
    """
    parsed = [row for row in (parse_wheel_name(name) for _, name in entries) if row is not None]
    if not parsed:
        return "No wheels are attached to this release yet.\n"
    platforms = [tag for tag in PLATFORMS if any(row["platform"] == tag for row in parsed)]
    if len(platforms) == 1:
        return (
            f"Prebuilt {PLATFORM_NAMES[platforms[0]]} CUDA {CUDA_TAG} wheels for "
            f"{_packages(parsed)}, built for {_coverage(parsed)}.\n"
        )
    # Per platform: separate runs need not cover the same packages or torch minors.
    per_platform = " ".join(
        f"{PLATFORM_NAMES[platform]}: {_packages(rows)}, built for {_coverage(rows)}."
        for platform in platforms
        for rows in [[row for row in parsed if row["platform"] == platform]]
    )
    return f"Prebuilt CUDA {CUDA_TAG} wheels. {per_platform}\n"


def render_sums(entries: list[tuple[str, str]]) -> str:
    """SHA256SUMS for every wheel on the release, in `sha256sum` format, sorted by filename.

    Built from the whole release, not from one run's output, because runs add to the tag one
    platform or one torch minor at a time and the file is replaced on every publish: a run that
    only built Windows wheels must not leave a SHA256SUMS that no longer lists the Linux ones. A
    digest that is not a 64-digit hex string is refused rather than written, because a checksum
    file with a placeholder in it is worse than none.
    """
    rows = []
    for digest, name in entries:
        if not name.endswith(".whl"):
            continue
        digest = digest.removeprefix("sha256:").lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise SystemExit(f"no usable sha256 for {name}: {digest!r}")
        rows.append((name, digest))
    if not rows:
        raise SystemExit("no wheels to list")
    return "".join(f"{digest}  {name}\n" for name, digest in sorted(rows))


def _cmd_matrix(args: argparse.Namespace) -> int:
    include = build_matrix(
        packages = os.environ.get("UW_PACKAGES", ""),
        torches = os.environ.get("UW_TORCH_VERSIONS", ""),
        pythons = os.environ.get("UW_PYTHON_VERSIONS", ""),
        platforms = os.environ.get("UW_PLATFORMS", ""),
    )
    matrix = json.dumps({"include": include}, separators = (",", ":"))
    warm = warm_matrix(include)
    gpu = gpu_matrix(include)
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
                gpu_json = json.dumps({"include": gpu}, separators = (",", ":"))
                handle.write(f"gpu_matrix={gpu_json}\n")
                handle.write(f"gpu_count={len(gpu)}\n")
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
    if args.platform not in PLATFORMS:
        raise SystemExit(f"unknown platform: {args.platform}")
    print(wheel_name(args.package, args.torch, args.python, args.platform))
    return 0


def _read_digest_lines() -> list[tuple[str, str]]:
    """`digest  filename` lines on stdin, in sha256sum's layout or with a single space."""
    entries = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition("  ")
        if not name:
            digest, _, name = line.partition(" ")
        entries.append((digest.strip(), name.strip().lstrip("*")))
    return entries


def _cmd_notes(args: argparse.Namespace) -> int:
    sys.stdout.write(render_notes(_read_digest_lines(), tag = args.tag, repo = args.repo))
    return 0


def _cmd_sums(args: argparse.Namespace) -> int:
    sys.stdout.write(render_sums(_read_digest_lines()))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    sub = parser.add_subparsers(dest = "command", required = True)

    matrix_parser = sub.add_parser("matrix", help = "print the build matrix as JSON")
    matrix_parser.add_argument(
        "--github",
        action = "store_true",
        help = "also append matrix/count, warm_matrix/warm_count and gpu_matrix/gpu_count to "
        "$GITHUB_OUTPUT and a listing to $GITHUB_STEP_SUMMARY",
    )
    matrix_parser.set_defaults(func = _cmd_matrix)

    name_parser = sub.add_parser("wheel-name", help = "print the filename for one cell")
    name_parser.add_argument("--package", required = True)
    name_parser.add_argument("--torch", required = True)
    name_parser.add_argument("--python", required = True)
    name_parser.add_argument("--platform", default = PLATFORM_TAG)
    name_parser.set_defaults(func = _cmd_wheel_name)

    notes_parser = sub.add_parser("notes", help = "print the release body, digests on stdin")
    notes_parser.add_argument("--tag", required = True)
    notes_parser.add_argument("--repo", required = True)
    notes_parser.set_defaults(func = _cmd_notes)

    sums_parser = sub.add_parser("sums", help = "print SHA256SUMS for every wheel, digests on stdin")
    sums_parser.set_defaults(func = _cmd_sums)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
