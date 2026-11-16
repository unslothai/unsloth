# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import re
from pathlib import Path

import unsloth_cli.commands.start as start_cli

REPO_ROOT = Path(__file__).resolve().parents[2]
EN = REPO_ROOT / "studio" / "frontend" / "src" / "i18n" / "locales" / "en.ts"


def test_agents_tab_note_names_every_agent_the_cli_limits_to_gguf():
    source = EN.read_text(encoding = "utf-8")
    note = re.search(r'modelNote:\s*"([^"]*llama-server[^"]*)"', source).group(1)
    gguf_only = [
        value.label for value in vars(start_cli).values() if isinstance(value, start_cli._GgufAgent)
    ]
    assert gguf_only
    for label in gguf_only:
        assert label in note.split(". Other agents")[0], label
