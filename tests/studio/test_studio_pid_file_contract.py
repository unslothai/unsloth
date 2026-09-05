# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""run.py writes the Unsloth PID files; `unsloth studio stop` globs for them.

Nothing else ties the writer's filename to the reader's glob, and each side's own
tests hardcode the names they expect, so a rename on either side alone leaves both
suites green while `stop` silently finds nothing. `unsloth_cli/tests/` also runs
in no workflow, so this lives here, where the repo CPU job discovers it.

AST + exec of the writer, so no backend dependency stack is imported.
"""

import ast
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_RUN_SRC = (_ROOT / "studio" / "backend" / "run.py").read_text(encoding = "utf-8")


def _func_source(source: str, name: str) -> str:
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"function {name!r} not found")


def _backend_pid_path(root: Path, port: int) -> Path:
    """The path run.py's own _pid_file_for_port builds, without importing run.py."""
    ns = {"os": os, "Path": Path, "_studio_root": lambda: root}
    exec(_func_source(_RUN_SRC, "_pid_file_for_port"), ns)
    return ns["_pid_file_for_port"](port)


def test_stop_finds_a_pid_file_named_the_way_the_backend_writes_it(tmp_path, monkeypatch):
    from unsloth_cli.commands import studio as cli

    path = _backend_pid_path(tmp_path, 8901)
    # The same three-line body _write_pid_file emits (create_time is blank when psutil is unavailable, and the CLI must
    # tolerate that).
    path.write_text(f"{os.getpid()}\n\n127.0.0.1", encoding = "utf-8")

    monkeypatch.setattr(cli, "STUDIO_HOME", tmp_path)
    monkeypatch.setattr(cli, "_PID_FILE", tmp_path / "studio.pid")

    assert [pid for pid, _times, _files in cli._pid_file_entries()] == [os.getpid()]


def test_the_legacy_file_stays_a_bare_pid_an_older_cli_can_parse(tmp_path, monkeypatch):
    # An older `unsloth studio stop` reads studio.pid and requires str.isdigit(), so the compatibility file must never
    # gain the extra metadata lines.
    ns = {
        "os": os,
        "Path": Path,
        "_studio_root": lambda: tmp_path,
        "_PID_FILE": tmp_path / "studio.pid",
        "_pid_file_for_port": lambda port: _backend_pid_path(tmp_path, port),
        "_process_create_time": lambda pid: None,
        "_bind_addresses": lambda host, port: {host},
        # _write_pid_file consults these before taking over studio.pid.
        "_read_pid_record": lambda path: None,
        "_pid_alive": lambda pid: False,
        "_OWN_PID_FILE": None,
    }
    exec(_func_source(_RUN_SRC, "_write_pid_file"), ns)
    ns["_write_pid_file"](8901, "127.0.0.1")

    assert (tmp_path / "studio.pid").read_text(encoding = "utf-8").strip().isdigit()
