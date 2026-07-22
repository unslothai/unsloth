#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-Present the Unsloth team. See /studio/LICENSE.AGPL-3.0

"""Cross-platform validation of the Unsloth Docker JupyterLab/notebook features.

Runs WITHOUT Docker or a GPU, so it can execute on the Linux/macOS/Windows CI
lanes. It exercises the actual notebook-helper logic (not just py_compile) and
checks the shipped JupyterLab config + labextension source, so a regression in
the notebook organisation, Colab compatibility, Colab-intro/widget stripping,
sidecar-log gating, the labextension plugins, the JupyterLab defaults, or the
login branding fails CI on every device.

Usage:  python tests/validate_studio_features.py
Exit 0 = all checks pass; non-zero = at least one failed.
"""

from __future__ import annotations

import importlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKER = os.path.join(ROOT, "docker")
JUPYTER = os.path.join(DOCKER, "jupyter")
LABEXT = os.path.join(JUPYTER, "unsloth_labext")
sys.path.insert(0, DOCKER)

_failures: list[str] = []


def check(
    name: str,
    cond: bool,
    detail: str = "",
) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail and not cond else ""))
    if not cond:
        _failures.append(name)


# --------------------------------------------------------------------------
# 1. Colab cell-magic compatibility (#@title then %%capture)
# --------------------------------------------------------------------------
def test_colab_compat() -> None:
    print("colab cell-magic compat (unsloth_colab_compat):")
    m = importlib.import_module("unsloth_colab_compat")
    out = m.colab_cell_magic_fix(["#@title Setup\n", "%%capture\n", "!pip install x\n"])
    check("magic hoisted above #@title", out[0] == "%%capture\n" and "#@title Setup\n" in out)
    # idempotent / already on top
    same = ["%%capture\n", "print(1)\n"]
    check("no-op when magic already first", m.colab_cell_magic_fix(same) == same)
    # non-magic cell untouched
    plain = ["x = 1\n", "y = 2\n"]
    check("plain cell untouched", m.colab_cell_magic_fix(plain) == plain)
    # content magic (%%writefile) NOT hoisted into the written file body
    wf = ["#@title Config\n", "%%writefile config.json\n", "{}\n"]
    check("content magic (%%writefile) left untouched", m.colab_cell_magic_fix(wf) == wf)
    # safe magic with arg still hoisted
    bash = ["#@title Run\n", "%%bash\n", "echo hi\n"]
    check("safe magic (%%bash) hoisted", m.colab_cell_magic_fix(bash)[0] == "%%bash\n")


# --------------------------------------------------------------------------
# 2. Notebook categorisation (clean_section) + README parsing
# --------------------------------------------------------------------------
def test_nb_view() -> None:
    print("notebook view (unsloth_nb_view):")
    v = importlib.import_module("unsloth_nb_view")
    check(
        "clean_section dash/slash -> space",
        v.clean_section("### GRPO-Reinforcement/Learning Notebooks")
        == "GRPO Reinforcement Learning Notebooks",
        v.clean_section("### GRPO-Reinforcement/Learning Notebooks"),
    )
    check(
        "clean_section strips hashes/space",
        v.clean_section("##  Main Notebooks  ") == "Main Notebooks",
    )


# --------------------------------------------------------------------------
# 3. Colab-intro + stale-widget stripping
# --------------------------------------------------------------------------
def test_strip() -> None:
    print("notebook strip (unsloth_nb_strip_colab):")
    s = importlib.import_module("unsloth_nb_strip_colab")
    nb = {
        "metadata": {"widgets": {"application/vnd.jupyter.widget-state+json": {"x": 1}}},
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    'To run this, press "Runtime" ... Tesla T4 Google Colab instance!\n',
                    "\n",
                    "You will learn how to ...\n",
                ],
            },
            {
                "cell_type": "code",
                "source": ["print(1)\n"],
                "outputs": [
                    {"output_type": "stream", "name": "stdout", "text": "ok\n"},
                    {
                        "output_type": "display_data",
                        "data": {
                            "application/vnd.jupyter.widget-view+json": {"model_id": "abc"},
                            "text/plain": "0%| | 0/10",
                        },
                    },
                ],
            },
        ],
    }
    changed1 = s._strip_intro(nb)
    changed2 = s._clean_widgets(nb)
    check(
        "intro line stripped",
        changed1 and not any("to run this, press" in (l.lower()) for l in nb["cells"][0]["source"]),
    )
    check("intro body kept", any("You will learn" in l for l in nb["cells"][0]["source"]))
    wv = sum(
        1
        for c in nb["cells"]
        for o in (c.get("outputs", []) or [])
        if "application/vnd.jupyter.widget-view+json" in (o.get("data", {}) or {})
    )
    check("widget-view outputs removed", changed2 and wv == 0)
    check(
        "non-widget outputs kept",
        any(
            o.get("output_type") == "stream"
            for c in nb["cells"]
            for o in (c.get("outputs", []) or [])
        ),
    )
    check("metadata.widgets removed", "widgets" not in nb["metadata"])
    # idempotent
    check("strip idempotent", not s._strip_intro(nb) and not s._clean_widgets(nb))


# --------------------------------------------------------------------------
# 4. Sidecar-log gating
# --------------------------------------------------------------------------
def test_sidecar_log_gate() -> None:
    print("sidecar log gate (unsloth_nb_compat):")
    c = importlib.import_module("unsloth_nb_compat")
    old = os.environ.pop("UNSLOTH_ENABLE_LOGGING", None)
    try:
        check("logging off by default", c._logging_enabled() is False)
        os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"
        check("logging on with env=1", c._logging_enabled() is True)
        os.environ["UNSLOTH_ENABLE_LOGGING"] = "0"
        check("logging off with env=0", c._logging_enabled() is False)
    finally:
        os.environ.pop("UNSLOTH_ENABLE_LOGGING", None)
        if old is not None:
            os.environ["UNSLOTH_ENABLE_LOGGING"] = old


# --------------------------------------------------------------------------
# 5. JupyterLab defaults (overrides.json)
# --------------------------------------------------------------------------
def test_overrides() -> None:
    print("jupyterlab defaults (jupyter/overrides.json):")
    path = os.path.join(JUPYTER, "overrides.json")
    check("overrides.json exists", os.path.isfile(path))
    if not os.path.isfile(path):
        return
    with open(path, encoding = "utf-8") as f:
        d = json.load(f)  # raises -> CI fails if invalid JSON
    themes = d.get("@jupyterlab/apputils-extension:themes", {})
    check(
        "default theme = Unsloth Dark",
        themes.get("theme") == "Unsloth Dark",
        str(themes.get("theme")),
    )
    check("adaptive theme on", themes.get("adaptive-theme") is True)
    check("preferred dark = Unsloth Dark", themes.get("preferred-dark-theme") == "Unsloth Dark")
    tracker = d.get("@jupyterlab/notebook-extension:tracker", {})
    check(
        "windowingMode none",
        tracker.get("windowingMode") == "none",
        str(tracker.get("windowingMode")),
    )
    notif = d.get("@jupyterlab/apputils-extension:notification", {})
    check(
        "news prompt off",
        str(notif.get("fetchNews")) == "false" and notif.get("checkForUpdates") is False,
    )
    panel = d.get("@jupyterlab/notebook-extension:panel", {})
    labels = [t.get("label", "") for t in panel.get("toolbar", [])]
    check(
        "Restart & Run All label (single >>)",
        any(l == "Restart & Run All" for l in labels) and not any(">>" in l for l in labels),
        str(labels),
    )


# --------------------------------------------------------------------------
# 6. Labextension source (plugins) + login branding assets
# --------------------------------------------------------------------------
def test_labext_and_branding() -> None:
    print("labextension + branding assets:")
    pkg = os.path.join(LABEXT, "package.json")
    check("labext package.json exists", os.path.isfile(pkg))
    if os.path.isfile(pkg):
        with open(pkg, encoding = "utf-8") as f:
            p = json.load(f)
        check("labext name unsloth-jupyterlab", p.get("name") == "unsloth-jupyterlab")
        check("labext themePath set", bool(p.get("jupyterlab", {}).get("themePath")))
    # Concatenate every .ts module under src/ so plugins defined in their own
    # files (cellNav, colabTitle, outputSelect, uiChrome) are all covered.
    src_dir = os.path.join(LABEXT, "src")
    all_src = ""
    if os.path.isdir(src_dir):
        for fn in sorted(os.listdir(src_dir)):
            if fn.endswith(".ts"):
                with open(os.path.join(src_dir, fn), encoding = "utf-8") as f:
                    all_src += f.read() + "\n"
    for plug in [
        "unsloth-jupyterlab:theme",
        "unsloth-jupyterlab:cell-nav",
        "unsloth-jupyterlab:logo",
        "unsloth-jupyterlab:colab-title",
        "unsloth-jupyterlab:output-select-all",
        "unsloth-jupyterlab:ui-chrome",
    ]:
        check(f"plugin present: {plug}", plug in all_src)
    # The two newest plugins are also exported from index.ts (wired in).
    index = os.path.join(src_dir, "index.ts")
    index_src = open(index, encoding = "utf-8").read() if os.path.isfile(index) else ""
    check("outputSelect wired in index.ts", "outputSelectPlugin" in index_src)
    check("uiChrome wired in index.ts", "uiChromePlugin" in index_src)
    # uiChrome hides the right activity bar; CTRL+A output-select selects nodes.
    check("right activity bar hidden", "jp-mod-right" in all_src and "display: none" in all_src)
    check("ctrl+A output select", "selectNodeContents" in all_src)
    # branding assets
    login = os.path.join(JUPYTER, "login.html")
    login_src = open(login, encoding = "utf-8").read() if os.path.isfile(login) else ""
    check("login.html branded", "unsloth-login-card" in login_src)
    check(
        "login.html uses sloth stickers",
        'static_url("sloth/' in login_src or "static_url('sloth/" in login_src,
    )
    check("favicon.ico present", os.path.isfile(os.path.join(JUPYTER, "favicon.ico")))
    check("logo.png present", os.path.isfile(os.path.join(JUPYTER, "logo.png")))
    check(
        "sloth sticker installer present",
        os.path.isfile(os.path.join(JUPYTER, "install_sloth_stickers.py")),
    )


def main() -> int:
    print("=== Unsloth Studio/notebook feature validation ===")
    for t in (
        test_colab_compat,
        test_nb_view,
        test_strip,
        test_sidecar_log_gate,
        test_overrides,
        test_labext_and_branding,
    ):
        try:
            t()
        except Exception as e:  # a thrown exception is a failure, not a crash
            _failures.append(f"{t.__name__}: {e!r}")
            print(f"  [FAIL] {t.__name__} raised {e!r}")
    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
