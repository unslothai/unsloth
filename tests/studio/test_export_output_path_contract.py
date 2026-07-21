# SPDX-License-Identifier: AGPL-3.0-only

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT = REPO_ROOT / "studio" / "backend" / "core" / "export" / "export.py"

EXPORT_FNS = (
    "export_merged_model",
    "export_base_model",
    "export_gguf",
    "export_lora_adapter",
)


def _find_method(tree, cls_name, method_name):
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == cls_name:
            for item in cls.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    return None


def _return_tuple_arity(fn):
    arities = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple):
            arities.append(len(node.value.elts))
    return arities


def test_export_methods_return_three_tuple_annotation():
    tree = ast.parse(EXPORT.read_text())
    for fn_name in EXPORT_FNS:
        fn = _find_method(tree, "ExportBackend", fn_name)
        assert fn is not None, f"missing ExportBackend.{fn_name}"
        ret = fn.returns
        assert isinstance(ret, ast.Subscript), f"{fn_name} return must be Tuple[...]"
        slc = ret.slice
        elts = slc.elts if isinstance(slc, ast.Tuple) else None
        assert (
            elts is not None and len(elts) == 3
        ), f"{fn_name} return annotation must be a 3-tuple, got {ast.dump(ret)}"


def test_export_methods_return_three_element_tuples():
    tree = ast.parse(EXPORT.read_text())
    for fn_name in EXPORT_FNS:
        fn = _find_method(tree, "ExportBackend", fn_name)
        assert fn is not None
        arities = _return_tuple_arity(fn)
        assert arities, f"{fn_name} has no tuple-return statements"
        for arity in arities:
            assert arity == 3, f"{fn_name} return tuple arity {arity}, expected 3"


def test_local_save_assigns_output_path():
    tree = ast.parse(EXPORT.read_text())
    for fn_name in EXPORT_FNS:
        fn = _find_method(tree, "ExportBackend", fn_name)
        assert fn is not None
        assigns = []
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "output_path":
                        assigns.append(node)
        non_none = [
            a
            for a in assigns
            if not (isinstance(a.value, ast.Constant) and a.value.value is None)
        ]
        assert non_none, f"{fn_name} never assigns a non-None output_path"


def test_gpu_save_method_bound_for_hub_only():
    tree = ast.parse(EXPORT.read_text())
    fn = _find_method(tree, "ExportBackend", "export_merged_model")
    assert fn is not None
    found_pre_save_method = False
    for node in ast.walk(fn):
        if isinstance(node, ast.Try):
            for stmt in node.body:
                if isinstance(stmt, ast.If):
                    test = stmt.test
                    if isinstance(test, ast.Name) and test.id == "_IS_MLX":
                        for sub in ast.walk(
                            ast.Module(body = stmt.orelse, type_ignores = [])
                        ):
                            if isinstance(sub, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == "save_method"
                                for t in sub.targets
                            ):
                                found_pre_save_method = True
                                break
                        if found_pre_save_method:
                            break
            if found_pre_save_method:
                break
    assert found_pre_save_method, (
        "GPU save_method must be assigned at the top of the try block, "
        "before the `if save_directory:` guard, so Hub-only export does not "
        "raise UnboundLocalError."
    )


def test_mlx_hub_only_uses_temp_directory():
    src = EXPORT.read_text()
    assert (
        src.count("tempfile.TemporaryDirectory") >= 3
    ), "expected TemporaryDirectory in merged, base, and lora hub-push paths"
    assert "import tempfile" in src.split("class ExportBackend")[0]


def test_is_mlx_imported_from_unsloth():
    src = EXPORT.read_text()
    assert "from unsloth import" in src
    head = src.split("class ExportBackend")[0]
    assert "_IS_MLX" in head
    assert "_IS_MLX = platform.system()" not in src
