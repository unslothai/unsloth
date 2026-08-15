"""_get_statistics must pass its force_download argument through to snapshot_download.

Regression test for #8899: the signature accepted force_download but the
snapshot_download call site hard-coded True, so the caller's explicit
force_download = False for the "repeat" statistics check was discarded.
"""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
UTILS_PATH = REPO_ROOT / "unsloth" / "models" / "_utils.py"


def _function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _snapshot_download_calls(func):
    calls = []
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "snapshot_download"
        ):
            calls.append(node)
    return calls


def test_get_statistics_signature_accepts_force_download():
    tree = ast.parse(UTILS_PATH.read_text(encoding = "utf-8"))
    func = _function(tree, "_get_statistics")
    assert [arg.arg for arg in func.args.args] == ["statistics", "force_download"]


def test_snapshot_download_receives_the_parameter_not_a_constant():
    tree = ast.parse(UTILS_PATH.read_text(encoding = "utf-8"))
    func = _function(tree, "_get_statistics")
    calls = _snapshot_download_calls(func)
    assert calls, "snapshot_download call not found inside _get_statistics"
    for call in calls:
        keywords = {kw.arg: kw.value for kw in call.keywords}
        assert "force_download" in keywords, "force_download keyword missing"
        value = keywords["force_download"]
        assert not isinstance(
            value, ast.Constant
        ), "force_download is hard-coded to a constant; the caller's argument is discarded"
        assert (
            isinstance(value, ast.Name) and value.id == "force_download"
        ), "force_download must forward the function parameter"


def test_repeat_caller_still_requests_no_forced_download():
    tree = ast.parse(UTILS_PATH.read_text(encoding = "utf-8"))
    func = _function(tree, "get_statistics")
    for node in ast.walk(func):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_get_statistics"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "repeat"
        ):
            keywords = {kw.arg: kw.value for kw in node.keywords}
            assert "force_download" in keywords
            value = keywords["force_download"]
            assert isinstance(value, ast.Constant) and value.value is False
            return
    raise AssertionError('_get_statistics("repeat", ...) call not found in get_statistics')
