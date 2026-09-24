#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Compile one shard of a torch extension's objects into ccache, then stop before linking.

Run `prebuilt_wheels_shard.py SHARD SHARDS` from the package's source directory.
Only ninja's targets change; torch's compiler commands stay identical to the full build.
"""

import os
import runpy
import subprocess
import sys


def slice_objects(ninja_targets: str, shard: int, shards: int) -> list[str]:
    """Select every SHARDS-th object from sorted, unique ninja targets, starting at SHARD."""
    if not 0 <= shard < shards:
        raise SystemExit(f"shard {shard} is outside 0..{shards - 1}")
    objects = sorted(
        {
            target
            for target in (line.split(":")[0] for line in ninja_targets.splitlines())
            if target.endswith(".o")
        }
    )
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
        total = len(slice_objects(listing, 0, 1))
        print(f"shard {shard + 1} of {shards}: {len(mine)} of {total} objects", flush = True)
        jobs = os.environ.get("MAX_JOBS", "1")
        subprocess.run(["ninja", "-v", "-j", jobs, *mine], cwd = build_directory, check = True)
        # Nothing to link: the objects are what ccache keeps.
        raise SystemExit(0)

    cpp_extension._run_ninja_build = compile_shard
    sys.argv = ["setup.py", "build_ext"]
    runpy.run_path("setup.py", run_name = "__main__")
    raise SystemExit("setup.py returned without reaching the ninja step")


if __name__ == "__main__":
    main()
