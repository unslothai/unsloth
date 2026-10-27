# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Every `unsloth` invocation (`--version`, `--help`, the `studio` launcher) imports all command
modules, so their module scope stays lean: YAML, asyncio and rich load where they are used."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFERRED = ("yaml", "asyncio", "rich.console")


def _deferred_loaded_by(statement: str) -> set[str]:
    probe = (
        f"{statement}\n"
        "import sys\n"
        f"print(json.dumps(sorted(m for m in {DEFERRED!r} if m in sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", "import json\n" + probe],
        capture_output = True,
        text = True,
        cwd = str(_REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    return set(json.loads(result.stdout.strip().splitlines()[-1]))


def test_importing_the_cli_does_not_load_deferred_modules():
    # Only what the CLI itself adds counts: typer, click or pydantic may load these on some versions.
    preloaded = _deferred_loaded_by("import typer, click, pydantic")
    assert _deferred_loaded_by("import unsloth_cli") - preloaded == set()


def test_config_fields_are_available_before_the_schema_is_built():
    from unsloth_cli.config import Config, load_config
    from unsloth_cli.options import _collect_config_fields

    names = {name for name, _ in _collect_config_fields(Config)}
    assert {"model", "dataset", "learning_rate", "lora_r", "hf_token"} <= names
    cfg = load_config(None)
    cfg.apply_overrides(learning_rate = 1e-4)
    assert cfg.training.learning_rate == 1e-4


def test_yaml_config_still_loads(tmp_path):
    from unsloth_cli.config import load_config

    path = tmp_path / "config.yaml"
    path.write_text("model: unsloth/foo\ntraining:\n  max_steps: 7\n", encoding = "utf-8")
    cfg = load_config(path)
    assert cfg.model == "unsloth/foo"
    assert cfg.training.max_steps == 7
