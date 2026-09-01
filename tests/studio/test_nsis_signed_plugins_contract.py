# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep the NSIS plugin DLLs coming from the signed copy.

NSIS runs its plugin DLLs out of $PLUGINSDIR, so the installer being signed says
nothing about them. tauri-bundler signs a copy and points at it two different
ways depending on version: the NSISPLUGINS env var up to bundler 2.8.1, and the
signed_plugins_path template variable from 2.9.3 (tauri-apps/tauri#15422). The
template carries both, because a missing plugin directory is a silent no-op, so
dropping the wrong one ships unsigned DLLs with no build error.
"""

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
TEMPLATE = REPO / "studio/src-tauri/windows/installer.nsi"

ENV_FORM = '!addplugindir "$%NSISPLUGINS%\\x86-unicode"'
TEMPLATE_FORM = '!addplugindir "{{signed_plugins_path}}"'


@pytest.fixture(scope = "module")
def template() -> str:
    return TEMPLATE.read_text(encoding = "utf-8")


def test_both_signed_plugin_directory_forms_are_present(template: str) -> None:
    assert ENV_FORM in template, "the pinned bundler only exports NSISPLUGINS"
    assert TEMPLATE_FORM in template, "bundler 2.9.3+ only exports signed_plugins_path"


def test_the_template_form_is_guarded(template: str) -> None:
    # Unsigned builds omit signed_plugins_path entirely;
    guard = template.index("{{#if signed_plugins_path}}")
    assert guard < template.index(TEMPLATE_FORM) < template.index("{{/if}}", guard)


def _line_of(template: str, needle: str) -> int:
    lines = template.splitlines()
    return next(i for i, line in enumerate(lines) if line.strip() == needle)


def _signed_dir_lines(template: str) -> dict[str, int]:
    # Each form is checked on its own.
    return {
        "env var": _line_of(template, ENV_FORM),
        "template var": _line_of(template, TEMPLATE_FORM),
    }


@pytest.mark.parametrize("form", ["env var", "template var"])
def test_signed_plugin_dir_precedes_every_plugin_use(template: str, form: str) -> None:
    # !addplugindir after a plugin call raises "conflicts with a plugin in another directory" at compile time, or
    lines = template.splitlines()
    plugin_calls = [
        i
        for i, l in enumerate(lines)
        if "::" in l
        and not l.strip().startswith((";", "#", "!"))
        and any(
            p in l
            for p in ("System::", "nsDialogs::", "StartMenu::", "NSISdl::", "nsis_tauri_utils::")
        )
    ]
    assert plugin_calls, "expected the template to call NSIS plugins"
    assert _signed_dir_lines(template)[form] < min(plugin_calls)


@pytest.mark.parametrize("form", ["env var", "template var"])
def test_includes_come_after_the_signed_plugin_dir(template: str, form: str) -> None:
    mui = _line_of(template, "!include MUI2.nsh")
    assert _signed_dir_lines(template)[form] < mui
