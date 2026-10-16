# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Post-export hooks: config loading, choice resolution, path/option validation, streaming and the output marker."""

import json
import sys

import pytest

from core.export import post_hooks


def _write_hooks(tmp_path, monkeypatch, hooks):
    path = tmp_path / "post_export_hooks.json"
    path.write_text(json.dumps({"hooks": hooks}))
    monkeypatch.setenv("UNSLOTH_POST_EXPORT_HOOKS", str(path))
    return path


def _merged_export(root, name = "run/checkpoint"):
    d = root / name
    d.mkdir(parents = True)
    (d / "config.json").write_text(json.dumps({"_name_or_path": "org/base-model"}))
    (d / "model.safetensors").write_bytes(b"")
    return d


ECHO = [
    sys.executable,
    "-c",
    "import os, sys; print('converted', sys.argv[1:], os.environ.get('UNSLOTH_POST_EXPORT'), os.environ.get('UNSLOTH_EXPORT_TYPE')); print('UNSLOTH_HOOK_OUTPUT=/tmp/out.bin')",
]


def test_no_file_means_no_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_POST_EXPORT_HOOKS", str(tmp_path / "missing.json"))
    assert post_hooks.list_hooks() == []


def test_list_filters_by_export_type_and_hides_command(tmp_path, monkeypatch):
    _write_hooks(
        tmp_path,
        monkeypatch,
        [
            {"id": "a", "label": "A", "applies_to": ["merged"], "command": ECHO},
            {"id": "b", "label": "B", "applies_to": ["lora"], "command": ECHO},
            {"id": "broken"},
        ],
    )
    ids = [h["id"] for h in post_hooks.list_hooks("merged")]
    assert ids == ["a"]
    assert "command" not in post_hooks.list_hooks()[0]


def test_choices_from_command(tmp_path, monkeypatch):
    cmd = [sys.executable, "-c", "import json; print(json.dumps({'recipes': ['auto', 'x']}))"]
    _write_hooks(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "a",
                "command": ECHO,
                "options": [
                    {
                        "name": "recipe",
                        "default": "auto",
                        "choices_command": cmd,
                        "choices_key": "recipes",
                    }
                ],
            }
        ],
    )
    assert post_hooks.list_hooks()[0]["options"][0]["choices"] == ["auto", "x"]


def test_run_substitutes_streams_and_reports_output(tmp_path, monkeypatch):
    _write_hooks(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "a",
                "command": ECHO + ["{export_dir}", "{export_type}", "{base_model}", "{level}"],
                "options": [{"name": "level", "default": "1", "choices": ["1", "2"]}],
            }
        ],
    )
    root = tmp_path / "exports"
    export = _merged_export(root)
    lines = []
    ok, _, details = post_hooks.run_hook(
        "a", str(export.parent), {"level": "2"}, root, lambda l, s: lines.append(l)
    )
    assert ok, lines
    assert details["output_path"] == "/tmp/out.bin"
    assert any(
        str(export) in l and "merged" in l and "org/base-model" in l and "'2'" in l for l in lines
    )
    assert any(
        l.endswith("1 merged") for l in lines
    ), lines  # UNSLOTH_POST_EXPORT / UNSLOTH_EXPORT_TYPE reach the command
    assert not any(l.startswith("UNSLOTH_HOOK_OUTPUT=") for l in lines)


@pytest.mark.parametrize(
    "hook_id,options,where,msg",
    [
        ("nope", {}, "inside", "Unknown"),
        ("a", {"level": "9"}, "inside", "choices"),
        ("a", {}, "outside", "not inside"),
    ],
)
def test_run_rejects(tmp_path, monkeypatch, hook_id, options, where, msg):
    _write_hooks(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "a",
                "command": ECHO,
                "options": [{"name": "level", "default": "1", "choices": ["1", "2"]}],
            }
        ],
    )
    root = tmp_path / "exports"
    export = _merged_export(root if where == "inside" else tmp_path / "elsewhere")
    ok, message, _ = post_hooks.run_hook(hook_id, str(export), options, root, lambda l, s: None)
    assert not ok and msg in message


def test_failing_command_is_reported(tmp_path, monkeypatch):
    _write_hooks(
        tmp_path,
        monkeypatch,
        [{"id": "a", "command": [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"]}],
    )
    root = tmp_path / "exports"
    export = _merged_export(root)
    ok, message, details = post_hooks.run_hook("a", str(export), {}, root, lambda l, s: None)
    assert not ok and "exit 3" in message and details["tail"] == ["boom"]


@pytest.mark.parametrize("root", [[], None, {"hooks": None}, {"hooks": {"id": "a"}}, "hooks", 3])
def test_wrong_file_shape_means_no_hooks(tmp_path, monkeypatch, root):
    path = tmp_path / "post_export_hooks.json"
    path.write_text(json.dumps(root))
    monkeypatch.setenv("UNSLOTH_POST_EXPORT_HOOKS", str(path))
    assert post_hooks.list_hooks() == []
    ok, message, _ = post_hooks.run_hook("a", str(tmp_path), {}, tmp_path, lambda l, s: None)
    assert not ok and "Unknown" in message


@pytest.mark.parametrize(
    "bad",
    [
        {"id": 7, "command": ECHO},
        {"id": "x", "command": "echo hi"},
        {"id": "x", "command": [sys.executable, 3]},
        {"id": "x", "command": ECHO, "applies_to": "merged"},
        {"id": "x", "command": ECHO, "options": {"name": "level"}},
        {"id": "x", "command": ECHO, "options": [{"label": "no name"}]},
        {"id": "x", "command": ECHO, "options": ["level"]},
        {"id": "x", "command": ECHO, "timeout_s": "soon"},
        {"id": "x", "command": ECHO, "label": ["A"]},
    ],
)
def test_malformed_entry_skipped_beside_a_valid_one(tmp_path, monkeypatch, bad):
    _write_hooks(tmp_path, monkeypatch, [bad, {"id": "a", "command": ECHO}])
    assert [h["id"] for h in post_hooks.list_hooks()] == ["a"]
    ok, message, _ = post_hooks.run_hook("x", str(tmp_path), {}, tmp_path, lambda l, s: None)
    assert not ok and "Unknown" in message


@pytest.mark.parametrize(
    "printed", ['{"recipes": "auto"}', '{"recipes": null}', '"auto"', "not json"]
)
def test_choices_command_with_wrong_output_falls_back(tmp_path, monkeypatch, printed):
    cmd = [sys.executable, "-c", f"print({printed!r})"]
    _write_hooks(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "a",
                "command": ECHO,
                "options": [
                    {
                        "name": "recipe",
                        "choices": ["auto"],
                        "choices_command": cmd,
                        "choices_key": "recipes",
                    }
                ],
            }
        ],
    )
    assert post_hooks.list_hooks()[0]["options"][0]["choices"] == ["auto"]


def test_silent_command_still_times_out(tmp_path, monkeypatch):
    _write_hooks(
        tmp_path,
        monkeypatch,
        [
            {
                "id": "a",
                "command": [sys.executable, "-c", "import time; time.sleep(30)"],
                "timeout_s": 1,
            }
        ],
    )
    root = tmp_path / "exports"
    export = _merged_export(root)
    t0 = __import__("time").time()
    ok, message, _ = post_hooks.run_hook("a", str(export), {}, root, lambda l, s: None)
    assert not ok and "timeout" in message
    assert __import__("time").time() - t0 < 10
