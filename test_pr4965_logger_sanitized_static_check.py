import ast
from pathlib import Path


def _route_tree():
    src = (Path(__file__).resolve().parent
           / "studio" / "backend" / "routes" / "datasets.py").read_text()
    return ast.parse(src), src


def _find_get_splits():
    tree, _ = _route_tree()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_dataset_splits":
            return node
    return None


def test_logger_error_calls_have_no_exc_info_true():
    fn = _find_get_splits()
    assert fn is not None
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "error"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "logger"
        ):
            for kw in n.keywords:
                if kw.arg == "exc_info":
                    # Any truthy exc_info is forbidden post-fix
                    assert isinstance(kw.value, ast.Constant)
                    assert kw.value.value is False or kw.value.value is None


def test_logger_error_messages_include_type_name_formatter():
    fn = _find_get_splits()
    assert fn is not None
    found_any = False
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "error"
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "logger"
            and n.args
        ):
            found_any = True
            # Arg 0 should be a JoinedStr (f-string) containing a FormattedValue
            # whose expression is `type(e).__name__`.
            arg0 = n.args[0]
            assert isinstance(arg0, ast.JoinedStr), (
                "logger.error first arg must be an f-string"
            )
            uses_type_name = False
            for part in arg0.values:
                if isinstance(part, ast.FormattedValue):
                    src_text = ast.unparse(part.value)
                    if ".__name__" in src_text and "type(" in src_text:
                        uses_type_name = True
            assert uses_type_name, (
                "logger.error f-string should format type(e).__name__"
            )
    assert found_any, "expected at least one logger.error call in function"


def test_partial_failure_format_is_generic():
    _, src = _route_tree()
    # New message is generic
    assert "of {len(configs)} config(s) could not be loaded" in src
    assert "Some subset options may be missing" in src
    # Old message fragment removed
    assert "could not be fetched:" not in src
