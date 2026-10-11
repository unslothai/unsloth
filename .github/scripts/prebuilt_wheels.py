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

# The CUDA major that every wheel here is built against, and the only one. It is the local
# version segment upstream writes (cu13), not the toolkit patch level: upstream normalises
# 13.x to "13" in get_wheel_url(), and a wheel built with 13.0 and one built with 13.2 are
# interchangeable for the purpose the tag serves, which is "do not install this on cu12".
CUDA_TAG = "13"

# The toolkit actually installed on the runner. Separate from CUDA_TAG because this one is a
# real apt package version and is reported in the notes, where "cu13" alone would be vague.
CUDA_TOOLKIT = "13.0"

# torch 2.7 and newer ship pip wheels built with _GLIBCXX_USE_CXX11_ABI=1, so there is no
# abiFALSE variant to build for 2.13 or 2.14 -- upstream's own matrix excludes it from 2.7 on.
# The tag is still in the filename because it is in every upstream filename, and a resolver
# that pattern-matches upstream names has to find it here too.
CXX11_ABI = "TRUE"

# The Linux tag. The wheel is tagged linux_x86_64 rather than manylinux_*, exactly as upstream
# tags its own, so pip installs it on any glibc without a floor check; the practical floor is the
# runner's glibc, which is why the build runs on ubuntu-22.04 (glibc 2.35) and not on
# ubuntu-latest.
PLATFORM_TAG = "linux_x86_64"

# Every platform a cell can be built for, keyed by the pip platform tag the wheel carries.
#
# win_amd64 exists because nobody else builds these three for Windows at all: across every release
# Dao-AILab and state-spaces have published there is not one Windows asset, so a Windows user's
# only option today is a multi-hour local compile with MSVC and the CUDA toolkit installed.
#
# `abi` is what torch._C._GLIBCXX_USE_CXX11_ABI reports on that platform, and it goes into the
# filename because that is what each upstream setup.py writes there on every platform: their own
# get_wheel_url() names a Windows wheel `...cxx11abiFALSE-cp313-cp313-win_amd64.whl`, because the
# attribute is False on an MSVC build of torch. Keeping their naming means the resolver builds the
# filename the same way on both platforms and needs no Windows special case.
#
# `pytorch_nvcc` is the compiler line torch's cpp_extension writes into build.ninja (it honours
# PYTORCH_NVCC on every OS, verbatim and unquoted, and on Windows that variable is the only way to
# put a compiler cache in front of nvcc, because _wrap_compiler is a no-op there). Hence the
# space-free C:/cuda junction and C:/ccache directory the Windows setup steps create.
#
# `smoke_exclude` / `smoke_extra` adjust the import smoke test's dependency install. mamba-ssm's
# METADATA requires `triton`, `tilelang==0.1.8`, `apache-tvm-ffi` and `quack-kernels`, none of
# which has a Windows wheel at that version; the import needs only `triton`, which triton-windows
# provides, and mamba_ssm imports its tilelang and CuTe kernels inside try/except.
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
        "abi": "FALSE",
        "label": " / win_amd64",
        "pytorch_nvcc": "C:/ccache/ccache.exe C:/cuda/bin/nvcc.exe",
    },
}

# Windows only. torch pins the triton release it was built against (2.13: triton==3.7.1, 2.14:
# triton~=3.8.0, both behind a Linux-only marker), and triton-windows tracks those releases with a
# .postN suffix. Exact pins, so the smoke test is the same test on every run.
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
        # The build is one nvcc invocation per (kernel, arch) and the kernels are large. On a
        # 4-core, 16 GB hosted runner nvcc 13 goes OOM above one job; this is upstream's own
        # value for the cu13 legs of its publish matrix, arrived at the same way.
        "max_jobs": "1",
        "nvcc_threads": "2",
        "env": {
            "FLASH_ATTENTION_FORCE_BUILD": "TRUE",
            "FLASH_ATTENTION_FORCE_CXX11_ABI": CXX11_ABI,
            # Upstream's default is "80;90;100;110;120". 110 (Thor) is dropped because nothing
            # Unsloth targets runs it and each arch is a full pass over every kernel. 86 and 89
            # are absent for a different reason: they are not needed. A cubin is compatible
            # forward across the minor versions of its major, so sm_80 code runs on sm_86 and
            # sm_89 hardware, and setup.py additionally emits PTX for the newest arch so an
            # unlisted future card JITs rather than failing.
            "FLASH_ATTN_CUDA_ARCHS": "80;90;100;120",
        },
        # Split the 6-8 hour compile across jobs to stay below GitHub's 6-hour limit.
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
            # Mamba-1's selective-scan CUDA kernels are opt-in upstream. They are the reason
            # this wheel is worth building: without them mamba_ssm falls back to the reference
            # path, and selective_scan_cuda -- the extension whose missing symbols are the
            # whole ABI problem -- is not in the wheel at all.
            "MAMBA_KEEP_CUDA_BUILD": "TRUE",
        },
        # The one source edit in this workflow. See patch_mamba_cxx20.py.
        "patch": "cxx20",
        "build_timeout": "180m",
        "import_names": ["mamba_ssm", "selective_scan_cuda"],
    },
}

# torch minors, not patch levels, are what the ABI is keyed on, but the build needs an exact
# version to pip install, so the table is keyed by the full version and the minor is derived.
TORCH_VERSIONS = ("2.13.0", "2.14.0")

# cp313 is the default and the only one the dispatch defaults to, because each extra
# interpreter is a whole extra flash-attn build. 3.11 and 3.12 are here so a run can add them
# as separate cells when the queue can afford it. 3.14 is deliberately absent: torch publishes
# a cu130 wheel for it, but nothing in the Unsloth stack is tested on 3.14 yet.
PYTHON_VERSIONS = ("3.11", "3.12", "3.13")

DEFAULT_PACKAGES = tuple(SPECS)
DEFAULT_TORCH = TORCH_VERSIONS
DEFAULT_PYTHON = ("3.13",)
DEFAULT_PLATFORMS = tuple(PLATFORMS)


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
    real ABI, so a Windows build does not claim, or compile with -D_GLIBCXX_USE_CXX11_ABI=1, a
    libstdc++ ABI that MSVC does not have. Linux keeps the exact string it always had.
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
    chosen_platforms = _resolve(platforms, DEFAULT_PLATFORMS, DEFAULT_PLATFORMS, "platform")

    include = []
    for platform in chosen_platforms:
        target = PLATFORMS[platform]
        for torch_version in chosen_torch:
            for python_version in chosen_python:
                for package in chosen_packages:
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
                            # Only used for the job name in the Actions UI, where "flash-attn /
                            # torch 2.13 / cp313" is the difference between reading the matrix
                            # and counting the cells. Linux keeps its old label; Windows says so.
                            "label": f"{package} / torch {torch_minor(torch_version)} / "
                            f"{python_tag(python_version)}{target['label']}",
                            "platform": platform,
                            "os": target["os"],
                            "runner": target["runner"],
                            "pytorch_nvcc": target["pytorch_nvcc"],
                            "smoke_exclude": smoke_exclude,
                            "smoke_extra": smoke_extra,
                        }
                    )
    return include


def gpu_matrix(include: list[dict]) -> list[dict]:
    """The cells the self-hosted GPU runner can test, which is a Linux machine: a Windows wheel
    does not even load there, so handing it one would only turn an optional check red."""
    return [cell for cell in include if cell["os"] == "linux"]


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
    wheels, and a loose parse would put them in the table as packages. The ABI must also be the
    one that platform actually has, so a stray `cxx11abiTRUE-...-win_amd64` is not one of ours.
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
        # Read from the file, not from SPECS: the notes describe what is attached, and an asset
        # left over from an older pin is a different version whatever the table says today.
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
PLATFORM_NAMES = {"linux_x86_64": "Linux x86_64", "win_amd64": "Windows x86_64"}


def _coverage(rows: list[dict]) -> str:
    """`PyTorch 2.13 and 2.14 on Python 3.13` for a set of parsed rows."""
    torches = sorted({row["torch"] for row in rows}, key = lambda v: tuple(map(int, v.split("."))))
    # cp313 -> 3.13
    pythons = [
        f"{cp[2]}.{cp[3:]}"
        for cp in sorted({row["python"] for row in rows}, key = lambda cp: int(cp[3:]))
    ]
    return f"PyTorch {_join(torches)} on Python {_join(pythons)}"


def render_notes(entries: list[tuple[str, str]], tag: str, repo: str) -> str:
    """One-sentence release body from `(sha256, filename)` pairs.

    Regenerated from the release's current assets on every publish rather than appended to, so
    a second run that adds the torch 2.14 half, or the Windows half, produces a sentence that
    describes everything attached. A release holding one platform reads exactly as it always has.
    """
    parsed = [row for row in (parse_wheel_name(name) for _, name in entries) if row is not None]
    if not parsed:
        return "No wheels are attached to this release yet.\n"
    order = {package: index for index, package in enumerate(SPECS)}
    packages = [
        f"{package} {version}"
        for package, version in sorted(
            {(row["package"], row["version"]) for row in parsed},
            key = lambda pv: (order[pv[0]], pv[1]),
        )
    ]
    platforms = [tag for tag in PLATFORMS if any(row["platform"] == tag for row in parsed)]
    if len(platforms) == 1:
        return (
            f"Prebuilt {PLATFORM_NAMES[platforms[0]]} CUDA {CUDA_TAG} wheels for "
            f"{_join(packages)}, built for {_coverage(parsed)}.\n"
        )
    # Per platform, because the two halves are published by separate runs and need not cover
    # the same torch minors; one merged list would advertise combinations that are not attached.
    per_platform = "; ".join(
        f"{PLATFORM_NAMES[platform]} for "
        f"{_coverage([row for row in parsed if row['platform'] == platform])}"
        for platform in platforms
    )
    return f"Prebuilt CUDA {CUDA_TAG} wheels for {_join(packages)}: {per_platform}.\n"


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

    # Writing the step output here rather than echoing it in YAML keeps the JSON -- which is
    # full of quotes and braces -- out of a shell round trip entirely.
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
