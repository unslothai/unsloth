# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The artifact page prelude regex runs on the event loop for any signed-in account, so a short
run of empty comments must not backtrack exponentially and stall every other account."""

import os
import subprocess
import sys

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_empty_comment_run_without_doctype_is_fast():
    # Time only the match: importing routes.inference is slow, so measure inside the child.
    probe = (
        "import time, routes.inference as inf\n"
        "t = time.monotonic()\n"
        "inf._artifact_page_with_prelude('<!---->' * 60 + 'x')\n"
        "print(time.monotonic() - t)\n"
    )
    child = subprocess.Popen(
        [sys.executable, "-c", probe],
        cwd = _BACKEND,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    try:
        out, err = child.communicate(timeout = 90)
    except subprocess.TimeoutExpired:
        child.kill()
        child.communicate()
        raise AssertionError("prelude regex did not finish: exponential backtracking")
    assert child.returncode == 0, err[-2000:]
    assert float(out.strip().splitlines()[-1]) < 1.0


def test_doctype_after_comments_still_found():
    import routes.inference as inf

    page = "<!-- a -->\n<!-- b --><!DOCTYPE html><p>x</p>"
    out = inf._artifact_page_with_prelude(page)
    assert out.startswith("<!-- a -->\n<!-- b --><!DOCTYPE html>")
    assert out.endswith("<p>x</p>")
    assert out != page
    unclosed = "<!-- never closed <!doctype html>"
    # No closed comment, no leading doctype: the prelude goes first, page untouched after it.
    assert inf._artifact_page_with_prelude(unclosed).endswith(unclosed)
