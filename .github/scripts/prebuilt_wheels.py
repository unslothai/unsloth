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

# Linux x86_64 only. The wheel is tagged linux_x86_64 rather than manylinux_*, exactly as
# upstream tags its own, so pip installs it on any glibc without a floor check; the practical
# floor is the runner's glibc, which is why the build runs on ubuntu-22.04 (glibc 2.35) and
# not on ubuntu-latest.
PLATFORM_TAG = "linux_x86_64"

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
        # ~4.5 h for the four architectures at MAX_JOBS=1. GitHub kills a job at 6 h, so the
        # build is wrapped in `timeout` below that, to fail as a build timeout with logs
        # rather than as a job cancellation with none.
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
                        "build_env": " ".join(
                            f"{key}={value}" for key, value in spec["env"].items()
                        ),
                        "import_names": " ".join(spec["import_names"]),
                        "wheel_name": wheel_name(package, torch_version, python_version),
                        # Only used for the job name in the Actions UI, where "flash-attn /
                        # torch 2.13 / cp313" is the difference between reading the matrix and
                        # counting the cells.
                        "label": f"{package} / torch {torch_minor(torch_version)} / "
                        f"{python_tag(python_version)}",
                    }
                )
    return include


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


NOTES_PREAMBLE = """\
Prebuilt Linux x86_64 CUDA 13 wheels for flash-attn, causal-conv1d and mamba-ssm, built against \
PyTorch 2.13 and 2.14.

### Why these exist

Upstream publishes prebuilt wheels for these three packages, but only against PyTorch 2.10 and \
2.11. Those wheels were ABI compatible through torch 2.12 and stopped being so at 2.13:

- `torch 2.13` changed `c10::impl::cow::materialize_cow_storage` and the signature of \
`c10::cuda::c10_cuda_check_implementation`, so an upstream wheel imports on 2.13 as \
`ImportError: undefined symbol: ...`.
- `torch 2.14` changed it again, so a 2.13 wheel is not usable on 2.14 either.

There is no version of these wheels that covers several torch minors from 2.13 on. Every torch \
minor needs its own build, which is what this release is: one wheel per (package, torch minor, \
interpreter), named so it cannot be installed against the wrong one.

Building them from source instead takes roughly five hours of nvcc for flash-attn alone, needs \
the CUDA toolkit present, and is out of reach on most machines that want to run the result.

### Compatibility

| | |
| --- | --- |
| Platform | Linux x86_64 only |
| CUDA | 13.0 toolkit, `cu13` wheels, usable with any CUDA 13.x runtime |
| PyTorch | exactly the minor in the filename, any patch level of it |
| GPU architectures | sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 |
| C++11 ABI | `TRUE`, matching every PyTorch pip wheel from 2.7 on |
| glibc | 2.35 or newer (built on Ubuntu 22.04) |

sm_86 and sm_89 are covered by the sm_80 cubin, which is forward compatible across the minor \
versions of its major. Newer architectures fall back to the embedded PTX and JIT on first use.

Not covered: Windows, macOS, ROCm, CUDA 12 or earlier, aarch64, and free-threaded interpreters.

### Installing

```
pip install https://github.com/{repo}/releases/download/{tag}/<wheel>
```

The `+cu13torch2.13cxx11abiTRUE` segment in each filename is a PEP 440 local version. pip \
refuses to install a wheel whose local version does not match the environment it was built \
for, so picking the wrong torch is a clean install error rather than an `undefined symbol` \
traceback the first time a kernel is called.

### Verifying

Every wheel is signed with [Sigstore](https://www.sigstore.dev/) through GitHub OIDC, with the \
bundle published beside it as `<wheel>.sigstore.json`, and carries an \
[SLSA build provenance](https://slsa.dev/) attestation.

Sigstore, which proves the wheel was produced by this workflow in this repository:

```
python -m pip install sigstore
python -m sigstore verify identity \\
  --cert-identity "https://github.com/{repo}/.github/workflows/prebuilt-cuda-wheels.yml@refs/heads/main" \\
  --cert-oidc-issuer https://token.actions.githubusercontent.com \\
  --bundle <wheel>.sigstore.json \\
  <wheel>
```

`verify identity` rather than `verify github`, because it takes both the identity and the issuer \
on every sigstore-python version; `verify github` dropped `--cert-oidc-issuer` in 4.0 and takes \
`--repository` and `--ref` instead. Use whichever your installed version accepts.

Build provenance, which reports the exact run, commit and workflow that produced it. Needs \
GitHub CLI 2.49 or newer:

```
gh attestation verify <wheel> --repo {repo}
```

Digests, which is the cheap check and still worth doing:

```
sha256sum -c SHA256SUMS --ignore-missing
```

### Wheels
"""


def render_notes(entries: list[tuple[str, str]], tag: str, repo: str) -> str:
    """Release body from `(sha256, filename)` pairs.

    Regenerated from the release's current assets on every publish rather than appended to, so
    a second run that adds the torch 2.14 half produces notes describing both halves, and a
    third run that replaces one wheel does not leave the old digest in the table.
    """
    rows = []
    for digest, name in entries:
        parsed = parse_wheel_name(name)
        if parsed is None:
            continue
        rows.append((parsed, digest))
    rows.sort(key=lambda row: (row[0]["torch"], row[0]["python"], row[0]["package"]))

    lines = [NOTES_PREAMBLE.format(repo=repo, tag=tag).rstrip("\n"), ""]
    if not rows:
        lines.append("No wheels are attached to this release yet.")
        return "\n".join(lines) + "\n"

    lines.append("| Wheel | Package | Version | torch | CUDA | Python | C++11 ABI | sha256 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for parsed, digest in rows:
        lines.append(
            "| `{name}` | {package} | {version} | {torch} | {cuda} | {python} | {abi} | `{digest}` |".format(
                digest=digest, **parsed
            )
        )
    lines.append("")
    lines.append(
        f"{len(rows)} wheels, each with a `.sigstore.json` bundle beside it. "
        "`SHA256SUMS` lists the same digests in `sha256sum -c` form."
    )
    return "\n".join(lines) + "\n"


def _cmd_matrix(args: argparse.Namespace) -> int:
    include = build_matrix(
        packages=os.environ.get("UW_PACKAGES", ""),
        torches=os.environ.get("UW_TORCH_VERSIONS", ""),
        pythons=os.environ.get("UW_PYTHON_VERSIONS", ""),
    )
    matrix = json.dumps({"include": include}, separators=(",", ":"))
    print(matrix)

    # Writing the step output here rather than echoing it in YAML keeps the JSON -- which is
    # full of quotes and braces -- out of a shell round trip entirely.
    if args.github:
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a", encoding="utf-8") as handle:
                handle.write(f"matrix={matrix}\n")
                handle.write(f"count={len(include)}\n")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            listing = "\n".join(f"- `{cell['wheel_name']}`" for cell in include)
            with open(summary, "a", encoding="utf-8") as handle:
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
    sys.stdout.write(render_notes(entries, tag=args.tag, repo=args.repo))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    matrix_parser = sub.add_parser("matrix", help="print the build matrix as JSON")
    matrix_parser.add_argument(
        "--github",
        action="store_true",
        help="also append matrix/count to $GITHUB_OUTPUT and a listing to $GITHUB_STEP_SUMMARY",
    )
    matrix_parser.set_defaults(func=_cmd_matrix)

    name_parser = sub.add_parser("wheel-name", help="print the filename for one cell")
    name_parser.add_argument("--package", required=True)
    name_parser.add_argument("--torch", required=True)
    name_parser.add_argument("--python", required=True)
    name_parser.set_defaults(func=_cmd_wheel_name)

    notes_parser = sub.add_parser("notes", help="print the release body, digests on stdin")
    notes_parser.add_argument("--tag", required=True)
    notes_parser.add_argument("--repo", required=True)
    notes_parser.set_defaults(func=_cmd_notes)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
