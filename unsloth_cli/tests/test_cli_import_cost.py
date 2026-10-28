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


def _child_json(code: str):
    # cwd alone does not select this checkout under PYTHONSAFEPATH=1, so put it first on sys.path.
    probe = f"import json, sys\nsys.path.insert(0, {str(_REPO_ROOT)!r})\n{code}\n"
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output = True,
        text = True,
        cwd = str(_REPO_ROOT),
        timeout = 300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _deferred_loaded_by(statement: str) -> set[str]:
    return set(
        _child_json(
            f"{statement}\n"
            f"print(json.dumps(sorted(m for m in {DEFERRED!r} if m in sys.modules)))"
        )
    )


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


_MODELS = "(c.DataConfig, c.TrainingConfig, c.LoraConfig, c.LoggingConfig, c.Config)"


def test_importing_the_cli_leaves_every_config_schema_unbuilt():
    built = _child_json(
        "import unsloth_cli\n"
        "import unsloth_cli.config as c\n"
        f"print(json.dumps([m.__pydantic_complete__ for m in {_MODELS}]))"
    )
    assert built == [False] * 5


def test_concurrent_first_loads_build_the_schemas_once():
    # Racing first builds of the deferred schemas can leave a validator broken for every later call.
    errors, ok = _child_json(
        "import threading\n"
        "sys.setswitchinterval(1e-6)\n"
        "from unsloth_cli.config import load_config\n"
        "errors = []\n"
        "barrier = threading.Barrier(8)\n"
        "def run():\n"
        "    try:\n"
        "        barrier.wait()\n"
        "        cfg = load_config(None)\n"
        "        cfg.apply_overrides(learning_rate = 1e-4, lora_r = 8, max_steps = 3)\n"
        "    except Exception as error:\n"
        "        errors.append(repr(error))\n"
        "threads = [threading.Thread(target = run) for _ in range(8)]\n"
        "for t in threads: t.start()\n"
        "for t in threads: t.join()\n"
        "cfg = load_config(None)\n"
        "cfg.apply_overrides(learning_rate = 2e-4)\n"
        "print(json.dumps([errors, cfg.training.learning_rate == 2e-4]))"
    )
    assert errors == []
    assert ok is True
