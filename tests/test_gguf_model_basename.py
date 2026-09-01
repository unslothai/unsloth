# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""GGUF filenames must never carry a directory, drive or UNC prefix (#7897).

When a LoRA's base model is a local Windows directory, ``config._name_or_path`` has
no forward slash, so the old ``split("/")[-1]`` kept the whole path as the model
name. Joining that onto the output directory relocated the file, because
``ntpath.join`` discards its first argument when the second carries a drive:

    ntpath.join(r"C:\\exp\\_gguf", r"D:\\M\\X.Q5_K_M.gguf") == r"D:\\M\\X.Q5_K_M.gguf"

The GGUF landed next to the base model and Unsloth logged ``(none)``.

These run on Linux and still assert the Windows answers, deliberately:
``os.path.basename`` returns the whole ``D:\\...`` string on POSIX, so a fix built
on it would pass here while Windows stayed broken.
"""

from __future__ import annotations

import ast
import ntpath
import posixpath
import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SAVE_PY = _REPO_ROOT / "unsloth" / "save.py"


def _load_helper():
    """Exec just ``_model_basename`` out of save.py: save.py cannot be imported
    without the full ML stack, and the helper is pure string/stat logic. Same ast
    lift as test_export_capability.py.
    """
    src = _SAVE_PY.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_model_basename":
            namespace: dict = {"os": __import__("os")}
            exec(compile(ast.Module([node], []), str(_SAVE_PY), "exec"), namespace)
            return namespace["_model_basename"]
    raise AssertionError(
        "unsloth/save.py does not define a module-level _model_basename helper. "
        "GGUF filenames are still derived with .split('/')[-1], which keeps the "
        "whole path for Windows base-model directories (issue #7897)."
    )


# Table A: the basename contract.
# (label, config._name_or_path, expected model_name) Rows marked REGRESSION must be byte-identical to the old
# .split("/")[-1].
_TABLE_A = [
    # -- REGRESSION rows: behaviour must not change -------------------------
    ("hf_id", "unsloth/Qwen3-8B", "Qwen3-8B"),
    ("hf_id_nested", "meta-llama/Llama-2-7b-hf", "Llama-2-7b-hf"),
    ("bare_name", "Qwen3-8B", "Qwen3-8B"),
    ("posix_abs", "/home/u/models/MyModel", "MyModel"),
    ("posix_rel", "./models/MyModel", "MyModel"),
    ("wsl_mount", "/mnt/d/Models/MyModel", "MyModel"),
    ("win_forward_slashes", "D:/Models/MyModel", "MyModel"),
    ("posix_trailing_sep", "/home/u/models/MyModel/", "MyModel"),
    ("win_drive_abs", r"D:\Models\Merged Models\MyModel", "MyModel"),
    ("win_drive_trailing", "D:\\Models\\MyModel\\", "MyModel"),
    ("win_mixed_seps", "D:\\Models/Merged Models\\MyModel", "MyModel"),
    ("unc_share", r"\\server\share\Models\MyModel", "MyModel"),
    ("unc_extended", r"\\?\UNC\server\share\MyModel", "MyModel"),
    ("win_extended", r"\\?\D:\Models\MyModel", "MyModel"),
    ("win_dot_rel", r".\models\MyModel", "MyModel"),
    ("win_spaces", r"C:\Users\Ada\OneDrive - X\Llama 3.1 8B", "Llama 3.1 8B"),
    ("win_unicode", "D:\\Модели\\Модель", "Модель"),
    ("win_short_name", r"C:\Models\MYMODE~1", "MYMODE~1"),
    ("win_double_dot_name", r"C:\Models\MyModel..v2", "MyModel..v2"),
    ("repeated_seps", "D:\\Models\\\\MyModel", "MyModel"),
    # -- degenerate rows: must never yield "" (a hidden .Q4_K_M.gguf) --------
    ("drive_root", "D:\\", "model"),
    ("bare_drive", "D:", "model"),
    ("empty", "", "model"),
    ("none", None, "model"),
    ("dot", ".", "model"),
    ("dotdot", "..", "model"),
    ("posix_root", "/", "model"),
]

_REGRESSION_LABELS = {
    "hf_id",
    "hf_id_nested",
    "bare_name",
    "posix_abs",
    "posix_rel",
    "wsl_mount",
    "win_forward_slashes",
}


@pytest.mark.parametrize(
    "label, name_or_path, expected",
    _TABLE_A,
    ids = [row[0] for row in _TABLE_A],
)
def test_model_basename(label, name_or_path, expected):
    assert _load_helper()(name_or_path) == expected


@pytest.mark.parametrize(
    "label, name_or_path, expected",
    [row for row in _TABLE_A if row[0] in _REGRESSION_LABELS],
    ids = [row[0] for row in _TABLE_A if row[0] in _REGRESSION_LABELS],
)
def test_working_inputs_are_unchanged(label, name_or_path, expected):
    """The fix must be inert for every input that already worked."""
    assert _load_helper()(name_or_path) == name_or_path.split("/")[-1] == expected


def test_result_never_starts_with_a_dot():
    """An empty stem produced hidden files that glob.glob(*.gguf) cannot see."""
    helper = _load_helper()
    for _label, name_or_path, _expected in _TABLE_A:
        stem = helper(name_or_path)
        assert stem, f"empty stem for {name_or_path!r}"
        assert not stem.startswith("."), f"hidden GGUF stem {stem!r} for {name_or_path!r}"


def test_result_is_never_a_path():
    """No separator, drive or UNC prefix may survive into a filename."""
    helper = _load_helper()
    for _label, name_or_path, _expected in _TABLE_A:
        stem = helper(name_or_path)
        assert "/" not in stem and "\\" not in stem, f"{stem!r} from {name_or_path!r}"
        assert not ntpath.isabs(stem) and not posixpath.isabs(stem)
        assert ntpath.splitdrive(stem)[0] == "", f"drive survived in {stem!r}"


def test_helper_is_idempotent():
    helper = _load_helper()
    for _label, name_or_path, _expected in _TABLE_A:
        once = helper(name_or_path)
        assert helper(once) == once


# The join arithmetic this protects (real ntpath, no mocking).
_GGUF_DIR = r"C:\Users\u\.unsloth\exports\MyModel\_tmp_model_ab12_gguf"


@pytest.mark.parametrize(
    "label, name_or_path, _expected",
    [row for row in _TABLE_A if row[0] not in _REGRESSION_LABELS],
    ids = [row[0] for row in _TABLE_A if row[0] not in _REGRESSION_LABELS],
)
def test_quantize_output_stays_inside_gguf_directory(label, name_or_path, _expected):
    """save.py:2073 joins the stem onto gguf_directory. Prove it cannot escape.

    Exact arithmetic from _quantize_one: the unfixed stem silently relocates the
    output to another drive/UNC share, the fixed stem stays put.
    """
    stem = _load_helper()(name_or_path)
    out = ntpath.join(_GGUF_DIR, f"{stem}.Q5_K_M.gguf")
    assert (
        ntpath.dirname(out) == _GGUF_DIR
    ), f"{name_or_path!r} -> stem {stem!r} -> output escaped to {out!r}"
    assert ntpath.basename(out) != ".Q5_K_M.gguf"


def test_unfixed_derivation_really_did_escape():
    """Pin the failure mode itself, so nobody 'simplifies' the helper back."""
    broken = r"D:\Models\Merged Models\MyModel".split("/")[-1]
    escaped = ntpath.join(_GGUF_DIR, f"{broken}.Q5_K_M.gguf")
    assert escaped == r"D:\Models\Merged Models\MyModel.Q5_K_M.gguf"
    assert ntpath.dirname(escaped) != _GGUF_DIR


def _gguf_func_src(name: str) -> str:
    src = _SAVE_PY.read_text(encoding = "utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError(f"{name} not found in unsloth/save.py")


def test_derivation_no_longer_splits_on_forward_slash_only():
    body = _gguf_func_src("unsloth_save_pretrained_gguf")
    assert 'split("/")[-1]' not in body, (
        "unsloth_save_pretrained_gguf still derives the GGUF stem with "
        "split('/')[-1]; Windows base-model directories keep the whole path."
    )
    assert "_model_basename(" in body


def test_lora_gguf_path_shares_the_helper():
    """save.py already had Windows-safe basename logic here; keep the two in sync."""
    body = _gguf_func_src("_unsloth_save_lora_gguf")
    assert "_model_basename(" in body


def test_helper_is_module_level_and_adds_no_locals_to_the_gguf_entrypoint():
    """save.py does `arguments = dict(locals())` and splats it into
    unsloth_generic_save(**arguments). Any NEW local bound before that line
    becomes an unexpected keyword argument and breaks every GGUF export on
    every OS, so the helper must live at module level."""
    src = _SAVE_PY.read_text(encoding = "utf-8")
    tree = ast.parse(src)

    assert any(
        isinstance(n, ast.FunctionDef) and n.name == "_model_basename" for n in tree.body
    ), "_model_basename must be module level, not nested inside the entrypoint"

    body = _gguf_func_src("unsloth_save_pretrained_gguf")
    fn = ast.parse(body).body[0]

    def _is_locals_snapshot(stmt) -> bool:
        if not isinstance(stmt, ast.Assign) or not isinstance(stmt.value, ast.Call):
            return False
        call = stmt.value
        return (
            isinstance(call.func, ast.Name)
            and call.func.id == "dict"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Call)
            and isinstance(call.args[0].func, ast.Name)
            and call.args[0].func.id == "locals"
        )

    def _always_exits(stmts) -> bool:
        """True if this block can never fall through to the snapshot."""
        for stmt in stmts:
            if isinstance(stmt, (ast.Return, ast.Raise)):
                return True
            if isinstance(stmt, ast.If) and stmt.orelse:
                if _always_exits(stmt.body) and _always_exits(stmt.orelse):
                    return True
        return False

    # that always returns/raises (the save_method="lora" early exit) never reach it.
    # Locals bound before `arguments = dict(locals())`.
    bound: set[str] = set()
    saw_snapshot = False
    for stmt in fn.body:
        if _is_locals_snapshot(stmt):
            saw_snapshot = True
            break
        if isinstance(stmt, ast.If) and _always_exits(stmt.body) and not stmt.orelse:
            continue
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Store):
                bound.add(sub.id)
    assert saw_snapshot, (
        "expected `arguments = dict(locals())` in unsloth_save_pretrained_gguf; "
        "if that changed, this guard needs updating"
    )

    # Names deleted from `arguments` before the splat.
    deleted = set(re.findall(r'del\s+arguments\[[\'"](\w+)[\'"]\]', body))

    params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    leaked = bound - deleted - params
    assert not leaked, (
        f"these locals reach unsloth_generic_save(**arguments) as unexpected "
        f"kwargs and will break every GGUF export: {sorted(leaked)}"
    )
