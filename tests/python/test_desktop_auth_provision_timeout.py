"""provision_desktop_auth must have a bounded timeout; DESKTOP_AUTH_LOCK cannot pin forever."""

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DESKTOP_AUTH_RS = REPO_ROOT / "studio" / "src-tauri" / "src" / "desktop_auth.rs"


def _find_fn(source: str, name: str) -> str:
    pattern = re.compile(
        rf"(?:async\s+)?fn\s+{re.escape(name)}\s*\([^)]*\)[^{{]*\{{",
        re.MULTILINE,
    )
    m = pattern.search(source)
    assert m, f"function {name!r} not found in {DESKTOP_AUTH_RS}"
    start = m.start()
    # Walk braces from the opening `{` to find the matching close.
    depth = 0
    body_start = source.index("{", start)
    for i in range(body_start, len(source)):
        c = source[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
    raise AssertionError("unbalanced braces while scanning function body")


def test_provision_desktop_auth_wraps_output_in_timeout():
    src = DESKTOP_AUTH_RS.read_text()
    body = _find_fn(src, "provision_desktop_auth")
    # Timeout must wrap the subprocess output call.
    assert "tokio::time::timeout" in body, (
        "provision_desktop_auth must use tokio::time::timeout around cmd.output()"
    )
    # There must be no bare `cmd.output().await` left unbounded.
    bare = re.search(r"\bcmd\s*\.\s*output\s*\(\s*\)\s*\.\s*await", body)
    if bare is not None:
        # The only acceptable occurrence is as the future passed to timeout(...).
        preceding = body[: bare.start()]
        assert "timeout" in preceding.splitlines()[-1] or "timeout(" in preceding[-120:], (
            "cmd.output().await must only appear as the future arg to tokio::time::timeout"
        )


def test_provision_desktop_auth_timeout_is_bounded_small():
    src = DESKTOP_AUTH_RS.read_text()
    body = _find_fn(src, "provision_desktop_auth")
    # Extract the Duration::from_secs(N) inside the timeout call.
    m = re.search(r"Duration::from_secs\(\s*(\d+)\s*\)", body)
    assert m, "provision_desktop_auth must specify a concrete timeout in seconds"
    seconds = int(m.group(1))
    # A subprocess that takes >2 min is almost certainly hung; 30-120s is sane.
    assert 5 <= seconds <= 120, f"provision timeout {seconds}s is outside sane bounds"


def test_provision_desktop_auth_handles_timeout_error_path():
    src = DESKTOP_AUTH_RS.read_text()
    body = _find_fn(src, "provision_desktop_auth")
    # The Elapsed -> Err path must exist so callers see a real error, not a panic.
    assert "timed out" in body.lower() or "map_err" in body, (
        "timeout's Elapsed branch must be converted to an Err with context"
    )
