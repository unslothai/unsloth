#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compile one shard of a torch extension's objects into ccache, then stop before linking.

Only ninja's targets change; compiler commands must stay identical to the full build.

Windows targets are `D:\\...\\kernel.obj: cuda_compile`, so split on the LAST ": ". ninja with no
targets builds everything, so an empty inventory is an error and an empty slice skips ninja.
"""

import os
import runpy
import subprocess
import sys


OBJECT_SUFFIXES = (".o", ".obj")


def list_objects(ninja_targets: str) -> list[str]:
    """Every object target in `ninja -t targets` output, sorted and unique."""
    return sorted(
        {
            target
            for target in (line.rsplit(": ", 1)[0].strip() for line in ninja_targets.splitlines())
            if target.endswith(OBJECT_SUFFIXES)
        }
    )


def slice_objects(ninja_targets: str, shard: int, shards: int) -> list[str]:
    """Select every SHARDS-th object from sorted, unique ninja targets, starting at SHARD."""
    if not 0 <= shard < shards:
        raise SystemExit(f"shard {shard} is outside 0..{shards - 1}")
    objects = list_objects(ninja_targets)
    if not objects:
        raise SystemExit("ninja listed no object targets; refusing to run a targetless build")
    return objects[shard::shards]


def main() -> None:
    import torch.utils.cpp_extension as cpp_extension

    shard, shards = int(sys.argv[1]), int(sys.argv[2])

    def compile_shard(build_directory, verbose, error_prefix):
        listing = subprocess.run(
            ["ninja", "-t", "targets", "all"],
            cwd = build_directory,
            capture_output = True,
            text = True,
            check = True,
        ).stdout
        mine = slice_objects(listing, shard, shards)
        total = len(list_objects(listing))
        print(f"shard {shard + 1} of {shards}: {len(mine)} of {total} objects", flush = True)
        if not mine:
            raise SystemExit(0)
        jobs = os.environ.get("MAX_JOBS", "1")
        subprocess.run(["ninja", "-v", "-j", jobs, *mine], cwd = build_directory, check = True)
        raise SystemExit(0)

    cpp_extension._run_ninja_build = compile_shard
    sys.argv = ["setup.py", "build_ext"]
    runpy.run_path("setup.py", run_name = "__main__")
    raise SystemExit("setup.py returned without reaching the ninja step")


if __name__ == "__main__":
    main()
