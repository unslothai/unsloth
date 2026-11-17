# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The browser panel's frame shell is one script assembled from several Python strings (BROWSER_NETWORK_JS and
_PAGE_STORAGE_JS are pasted into _FRAME_HTML's boot function). Each piece can parse on its own while the assembled
script does not, for example a `const route` in the network code beside the history code's `let route`. Then no
page renders in the panel at all. Parse every inline script exactly as it is served."""

import os
import re
import shutil
import subprocess

import pytest

import routes.browser as browser_mod
import routes.inference as inference_mod

_SCRIPT = re.compile(r"<script(?P<attrs>[^>]*)>(?P<body>.*?)</script>", re.S | re.I)


def _node():
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail("node is not on PATH; this check must not skip in CI")
        pytest.skip("node is not installed")
    return node


def _served_pages():
    return {
        "browser frame": browser_mod._FRAME_HTML,
        "browser print": browser_mod._PRINT_HTML,
        "artifact preview frame": inference_mod._artifact_preview_frame_html(),
    }


def _inline_scripts(html):
    scripts = []
    for match in _SCRIPT.finditer(html):
        attrs = match.group("attrs")
        if "src=" in attrs:
            continue
        assert "module" not in attrs, "a module script would need --input-type=module"
        scripts.append(match.group("body"))
    return scripts


def _parse_error(node, source, tmp_path, name):
    path = tmp_path / f"{re.sub(r'[^a-z0-9]+', '_', name)}.js"
    path.write_text(source, encoding = "utf-8")
    result = subprocess.run(
        [node, "--check", str(path)], capture_output = True, text = True, timeout = 60
    )
    return None if result.returncode == 0 else result.stderr.strip()


@pytest.mark.parametrize("page", sorted(_served_pages()))
def test_every_inline_script_parses_as_served(page, tmp_path):
    node = _node()
    scripts = _inline_scripts(_served_pages()[page])
    assert scripts, f"{page}: no inline script found, so the pattern no longer matches the template"
    for i, source in enumerate(scripts):
        error = _parse_error(node, source, tmp_path, f"{page} {i}")
        assert error is None, f"{page} script {i} does not parse:\n{error}"


def test_the_frame_check_catches_a_name_two_pasted_pieces_both_declare(tmp_path):
    # The failure this file guards: network code declaring a name the frame's history code already declares.
    node = _node()
    frame = browser_mod._FRAME_HTML
    assert "let route = 0;" in frame
    network = browser_mod.BROWSER_NETWORK_JS.strip()
    assert network in frame
    broken = frame.replace(network, network + "\nconst route = () => {};", 1)
    errors = [
        _parse_error(node, s, tmp_path, f"broken {i}")
        for i, s in enumerate(_inline_scripts(broken))
    ]
    assert any(e and "route" in e for e in errors), errors
