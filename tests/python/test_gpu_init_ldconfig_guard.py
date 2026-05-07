import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_INIT = REPO_ROOT / "unsloth" / "_gpu_init.py"


def _find_geteuid_guard(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        for sub in ast.walk(node.test):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute):
                if sub.func.attr == "geteuid":
                    return node
    return None


def test_gpu_init_has_geteuid_guard():
    tree = ast.parse(GPU_INIT.read_text())
    guard = _find_geteuid_guard(tree)
    assert (
        guard is not None
    ), "_gpu_init.py must guard ldconfig recovery on os.geteuid()"


def test_ldconfig_calls_only_inside_geteuid_guard():
    src = GPU_INIT.read_text()
    tree = ast.parse(src)
    guard = _find_geteuid_guard(tree)
    assert guard is not None
    guard_src = ast.get_source_segment(src, guard) or ""
    ldconfig_lines = [
        line for line in src.splitlines() if "ldconfig" in line and "os.system" in line
    ]
    for line in ldconfig_lines:
        assert line.strip() in guard_src, (
            "os.system('ldconfig ...') must live inside the geteuid guard, "
            f"but found unguarded: {line!r}"
        )


def test_non_root_branch_warns_when_bnb_present():
    src = GPU_INIT.read_text()
    assert "elif bnb is not None" in src
    assert "sudo ldconfig" in src
