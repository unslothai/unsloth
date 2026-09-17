#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pick the legs of an install workflow for the event that triggered it.

Reads a matrix file (`.github/ci/*-matrix.yml`: one list of legs per matrix job, each
leg carrying a `pr` flag) and prints one `<job>={"include": [...]}` line plus a
`<job>_count=<n>` line per job, in the form `$GITHUB_OUTPUT` takes. On `pull_request` only the `pr: true` legs are
emitted; on every other event (schedule, push, workflow_dispatch) all of them are. The
`pr` key never reaches the workflow: what a job sees through `matrix.<key>` is exactly
what the leg declares, so the steps are unchanged by the selection.

The output is JSON on one line per job, which is what `fromJSON()` in the consuming
`strategy.matrix` needs, and it is a mapping with a single `include` key rather than a
bare list because `matrix:` must be a mapping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


def select(legs_by_job: dict, event_name: str) -> dict[str, dict]:
    """The `{"include": [...]}` mapping per job, `pr` stripped, for one event."""
    subset = event_name == "pull_request"
    picked: dict[str, dict] = {}
    for job, legs in legs_by_job.items():
        if not isinstance(legs, list):
            raise SystemExit(f"{job}: expected a list of legs, got {type(legs).__name__}")
        include = []
        for leg in legs:
            if not isinstance(leg, dict) or "pr" not in leg:
                raise SystemExit(f"{job}: every leg needs a `pr` flag: {leg!r}")
            if subset and not leg["pr"]:
                continue
            include.append({k: v for k, v in leg.items() if k != "pr"})
        picked[job] = {"include": include}
    return picked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument("--file", required = True, type = Path, help = "the matrix file")
    parser.add_argument("--event", required = True, help = "github.event_name")
    args = parser.parse_args(argv)
    legs_by_job = yaml.safe_load(args.file.read_text(encoding = "utf-8"))
    if not isinstance(legs_by_job, dict):
        raise SystemExit(f"{args.file}: expected a mapping of job -> legs")
    for job, matrix in select(legs_by_job, args.event).items():
        # One line per output: a newline inside the JSON would end the value early.
        print(f"{job}={json.dumps(matrix, separators = (',', ':'))}")
        # An empty matrix is a workflow error, so an all-nightly job gates on this count.
        print(f"{job}_count={len(matrix['include'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
