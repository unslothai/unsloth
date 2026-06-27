# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0
"""Tests for the Unsloth Docker Studio branding / AGPLv3 integrity guard.

verify_branding() is exercised against a staged temp tree that mirrors the
installed image layout, so no container or built labextension is required:
  * positive: a faithful tree passes (no problems).
  * negative: removing/altering each attribution marker is detected.
  * no-encoding: the attribution sources carry no base64/decoder obfuscation
    (plain readable strings only -- the only data URI is the logo *image*).
"""

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "docker", "jupyter"))

import unsloth_branding as ub  # noqa: E402


def _stage(tmp_path):
    """Create a faithful copy of the installed branding layout; return paths."""
    venv_share = tmp_path / "venv-share"
    js_dir = tmp_path / "jupyter_server"

    (venv_share).mkdir(parents = True)
    (venv_share / "UNSLOTH_LICENSE.AGPL-3.0").write_text(
        "                    GNU AFFERO GENERAL PUBLIC LICENSE\n"
        "                       Version 3, 19 November 2007\n"
        "  Copyright (C) 2007 Free Software Foundation, Inc.\n",
        encoding = "utf-8",
    )

    (venv_share / "lab" / "settings").mkdir(parents = True)
    (venv_share / "lab" / "settings" / "overrides.json").write_text(
        json.dumps({"@jupyterlab/apputils-extension:themes": {"theme": ub.THEME_NAME}}),
        encoding = "utf-8",
    )

    labext = venv_share / "labextensions" / ub.LABEXT_NAME
    (labext / "static").mkdir(parents = True)
    (labext / "package.json").write_text(json.dumps({"name": ub.LABEXT_NAME}), encoding = "utf-8")
    bundle = " ".join(
        [
            ub.PHRASE,
            ub.SHORT_LABEL,
            ub.COPYRIGHT,
            ub.AGPL_URL,
            ub.ABOUT_PLUGIN_ID,
            ub.SPLASH_PLUGIN_ID,
            ub.LOGO_DATA_URI_PREFIX + "AAAAdummyimagebytes",
        ]
    )
    (labext / "static" / "remoteEntry.abc123.js").write_text(bundle, encoding = "utf-8")

    (js_dir / "templates").mkdir(parents = True)
    (js_dir / "templates" / "login.html").write_text(
        "Built by Unsloth. Licensed under the GNU AGPLv3.\n"
        "Copyright 2026-Present the Unsloth team.\n"
        "https://github.com/unslothai/unsloth\n",
        encoding = "utf-8",
    )

    (js_dir / "static" / "favicons").mkdir(parents = True)
    (js_dir / "static" / "favicons" / "favicon.ico").write_bytes(b"\x00\x00\x01\x00icon")
    (js_dir / "static" / "logo").mkdir(parents = True)
    (js_dir / "static" / "logo" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\nlogo")

    return ub.resolve_paths(venv_share = str(venv_share), jupyter_server_dir = str(js_dir))


def test_positive_clean_tree_passes(tmp_path):
    paths = _stage(tmp_path)
    assert ub.verify_branding(paths) == []


# --- negative mutations: each strips one attribution marker --------------------
def _remove_license(paths):
    os.remove(paths["license"])


def _blank_license(paths):
    with open(paths["license"], "w", encoding = "utf-8") as f:
        f.write("All rights reserved. Proprietary. Resold by someone else.\n")


def _remove_login(paths):
    os.remove(paths["login"])


def _strip_login_source(paths):
    with open(paths["login"], encoding = "utf-8") as f:
        text = f.read()
    with open(paths["login"], "w", encoding = "utf-8") as f:
        f.write(text.replace(ub.SOURCE_URL, "https://example.com/forks"))


def _strip_login_copyright(paths):
    with open(paths["login"], encoding = "utf-8") as f:
        text = f.read()
    with open(paths["login"], "w", encoding = "utf-8") as f:
        f.write(text.replace(ub.COPYRIGHT, "Copyright someone else"))


def _drop_theme(paths):
    with open(paths["overrides"], "w", encoding = "utf-8") as f:
        f.write("{}")


def _rebrand_labext(paths):
    with open(paths["labext_pkg"], "w", encoding = "utf-8") as f:
        f.write(json.dumps({"name": "totally-not-unsloth"}))


def _strip_bundle_phrase(paths):
    import glob
    for path in glob.glob(os.path.join(paths["labext_static"], "*.js")):
        with open(path, encoding = "utf-8") as f:
            text = f.read()
        with open(path, "w", encoding = "utf-8") as f:
            f.write(text.replace(ub.PHRASE, "").replace(ub.SHORT_LABEL, ""))


def _strip_bundle_logo(paths):
    import glob
    for path in glob.glob(os.path.join(paths["labext_static"], "*.js")):
        with open(path, encoding = "utf-8") as f:
            text = f.read()
        with open(path, "w", encoding = "utf-8") as f:
            f.write(text.replace(ub.LOGO_DATA_URI_PREFIX, "data:image/png;base64,XXXX"))


def _remove_logo_png(paths):
    os.remove(paths["logo"])


def _empty_favicon(paths):
    open(paths["favicon"], "w").close()


@pytest.mark.parametrize(
    "mutate",
    [
        _remove_license,
        _blank_license,
        _remove_login,
        _strip_login_source,
        _strip_login_copyright,
        _drop_theme,
        _rebrand_labext,
        _strip_bundle_phrase,
        _strip_bundle_logo,
        _remove_logo_png,
        _empty_favicon,
    ],
)
def test_negative_each_marker_is_enforced(tmp_path, mutate):
    paths = _stage(tmp_path)
    assert ub.verify_branding(paths) == [], "baseline should be clean before mutation"
    mutate(paths)
    problems = ub.verify_branding(paths)
    assert problems, "stripping " + mutate.__name__ + " must be detected"


def test_attribution_sources_have_no_encoded_obfuscation():
    """Plain readable strings only -- no base64/decoder tricks (antivirus-safe)."""
    src_dir = os.path.join(REPO, "docker", "jupyter")
    files = [
        os.path.join(src_dir, "unsloth_branding.py"),
        os.path.join(src_dir, "unsloth_labext", "src", "branding.ts"),
        os.path.join(src_dir, "unsloth_labext", "src", "about.ts"),
        os.path.join(src_dir, "unsloth_labext", "src", "splash.ts"),
    ]
    forbidden = [
        "b64decode",
        "b64encode",
        "atob(",
        "btoa(",
        "fromCharCode",
        "unescape(",
        "rot13",
        "codecs.decode",
    ]
    for path in files:
        with open(path, encoding = "utf-8") as f:
            text = f.read()
        for token in forbidden:
            assert token not in text, path + " uses obfuscation token: " + token


def test_canonical_phrase_is_plain_text_in_definition_files():
    """The attribution lives as plain readable text in both definition files.

    branding.ts holds the full PHRASE as ONE contiguous literal (so webpack keeps
    it whole in the bundle for the guard to grep). unsloth_branding.py keeps the
    markers as plain constants (the runtime PHRASE value matches, even though the
    source wraps it across adjacent literals)."""
    src_dir = os.path.join(REPO, "docker", "jupyter")
    ts = open(
        os.path.join(src_dir, "unsloth_labext", "src", "branding.ts"), encoding = "utf-8"
    ).read()
    assert ub.PHRASE in ts, "branding.ts must hold the full PHRASE as one literal"
    py = open(os.path.join(src_dir, "unsloth_branding.py"), encoding = "utf-8").read()
    for marker in (ub.SHORT_LABEL, ub.COPYRIGHT, ub.SOURCE_URL, ub.AGPL_URL, ub.THEME_NAME):
        assert marker in py, "unsloth_branding.py missing plain marker: " + marker
