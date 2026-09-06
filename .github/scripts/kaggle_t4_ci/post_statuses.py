# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Post the commit statuses collect.py decided on, and say which ones landed.

The only step in the Kaggle CI holding a GitHub credential, and it holds no
Kaggle one: collect.py judges, this posts, neither can leak the other's token.
A script rather than a shell loop in three workflows so the record stays JSON
end to end and each value reaches `gh` as an argument, never through a shell.

THE SHA HAS TO BE EXPANDED FIRST. The slug carries only 8 hex characters (a
full sha plus the prefix does not fit Kaggle's slug limit) and the statuses
API refuses an abbreviation, measured against a real repository:

    POST /statuses/2ecb19df
    422 "Sha must be a valid hex object ID"

A sha that cannot be resolved (force-pushed away) is recorded as `unresolved`
so the kernel is released rather than retried forever.

The outcome is written as ``posted.json`` for collect.py --delete-collected: a
kernel whose status did not post is KEPT on Kaggle for the next pass. That
ordering (post, then delete) is why this is a separate step.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from collect import STATUS_CONTEXTS  # noqa: E402

STATES = ("error", "failure", "pending", "success")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


def _gh(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(["gh", *args], capture_output = True, text = True)
    return proc.returncode, (proc.stdout or "").strip(), (proc.stderr or "").strip()


# What GitHub says when the commit is genuinely not there, measured:
#
#     gh api repos/unslothai/unsloth/commits/deadbeef00
#     {"message":"No commit found for SHA: deadbeef00", ..., "status":"422"}
#
# Only that message releases the kernel. Status codes do not say it: 404 is
# also an unreadable repository and 422 an ambiguous abbreviation, and 5xx /
# rate limiting / network say nothing at all. Reading any of them as "gone"
# would delete the only copy of the result.
MISSING_MARKERS = ("no commit found",)


def resolve_sha(repo: str, sha: str) -> tuple[str, str | None]:
    """``("ok", full_sha)``, ``("missing", None)`` when the repository no
    longer has the commit, or ``("error", None)`` when the lookup itself
    failed and nothing is known about the commit."""
    code, out, err = _gh(["api", f"repos/{repo}/commits/{sha}", "-q", ".sha"])
    if code == 0 and FULL_SHA.match(out):
        return "ok", out
    text = f"{out} {err}".lower()
    if any(m in text for m in MISSING_MARKERS):
        return "missing", None
    return "error", None


def post_one(repo: str, full_sha: str, status: dict) -> bool:
    code, _out, _err = _gh(
        [
            "api",
            f"repos/{repo}/statuses/{full_sha}",
            "-f",
            f"state={status['state']}",
            "-f",
            f"context={status['context']}",
            "-f",
            f"description={status['description']}",
            "-f",
            f"target_url={status.get('target_url') or ''}",
            "--silent",
        ]
    )
    return code == 0


def valid(status: dict) -> str | None:
    """Why this record must not be posted, or None if it is well formed.

    Checked here because this is the process holding the token: a state outside
    the API's four, or a context that is not ours, is a record we must not sign.
    """
    if status.get("state") not in STATES:
        return f"state {status.get('state')!r} is not one of {STATES}"
    if status.get("context") not in STATUS_CONTEXTS.values():
        return f"context {status.get('context')!r} is not a Kaggle CI context"
    if not re.fullmatch(r"[0-9a-f]{8,40}", str(status.get("sha") or "")):
        return f"sha {status.get('sha')!r} is not a hex commit id"
    return None


def post_all(statuses: list[dict], repo: str) -> dict:
    outcome: dict[str, list[str]] = {"ok": [], "failed": [], "unresolved": [], "invalid": []}
    for status in statuses:
        slugs = list(status.get("slugs") or [status.get("slug")])
        why = valid(status)
        if why:
            print(f"::warning title=Malformed status record::{why}; not posted")
            outcome["invalid"].extend(slugs)
            continue
        found, full = resolve_sha(repo, status["sha"])
        if found == "missing":
            print(
                f"::warning title=Could not resolve a collected commit::{status['sha']} is "
                f"no longer a commit in this repository, so its {status['context']} result "
                "cannot be posted. The kernel is released; nothing will ever post for it."
            )
            outcome["unresolved"].extend(slugs)
            continue
        if found != "ok":
            print(
                f"::warning title=Commit lookup failed::could not ask GitHub about "
                f"{status['sha']} this pass; the kernel is kept so the next pass can try again"
            )
            outcome["failed"].extend(slugs)
            continue
        print(f"posting {status['context']}={status['state']} for {full}")
        if post_one(repo, full, status):
            outcome["ok"].extend(slugs)
        else:
            print(
                f"::warning title=Could not post a commit status::{status['context']} for "
                f"{full}; the kernel is kept so the next pass can try again"
            )
            outcome["failed"].extend(slugs)
    return outcome


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required = True, help = "collect_result.json from collect.py")
    ap.add_argument("--out", required = True, help = "where to write posted.json")
    ap.add_argument("--repo", default = os.environ.get("GITHUB_REPOSITORY", ""))
    args = ap.parse_args()
    if not args.repo:
        ap.error("--repo (or GITHUB_REPOSITORY) is required")

    data = json.loads(Path(args.result).read_text(encoding = "utf-8"))
    statuses = data.get("statuses") or []
    outcome = post_all(statuses, args.repo)
    Path(args.out).write_text(json.dumps(outcome, indent = 2), encoding = "utf-8")
    if not statuses:
        print("no statuses to post this pass")
    # Red when a verdict could not be delivered. The kernel is kept for the next
    # pass, but a delivery failure must not look like a quiet account.
    return 1 if outcome["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
