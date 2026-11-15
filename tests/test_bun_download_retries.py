"""The frontend CI Bun download must outlast a minutes-long burst of 504s from GitHub releases."""

from __future__ import annotations

import re
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "studio-frontend-ci.yml"
STEP = "Typecheck through bun check"


def _curl_args():
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]
    runs = [
        s["run"] for job in jobs.values() for s in job.get("steps", []) if s.get("name") == STEP
    ]
    assert len(runs) == 1, f"expected one {STEP!r} step, found {len(runs)}"
    script = re.sub(r"\\\n", " ", runs[0])
    curls = [line for line in script.splitlines() if line.strip().startswith("curl ")]
    assert len(curls) == 1, curls
    return shlex.split(curls[0])


def _backoff_seconds(args):
    """Total wait curl spends between attempts on a server that keeps failing."""
    retries = int(args[args.index("--retry") + 1]) if "--retry" in args else 0
    if "--retry-delay" in args:
        return retries * int(args[args.index("--retry-delay") + 1])
    # curl doubles from one second, capped at ten minutes per wait.
    return sum(min(2**i, 600) for i in range(retries))


def test_the_bun_download_backs_off_for_minutes():
    args = _curl_args()
    assert "--retry-all-errors" in args, "a 504 is only retried with --retry-all-errors"
    assert _backoff_seconds(args) >= 240, args


def test_the_backoff_reader_matches_curl():
    assert _backoff_seconds(["curl", "--retry", "3"]) == 7
    assert _backoff_seconds(["curl", "--retry", "5", "--retry-delay", "3"]) == 15
