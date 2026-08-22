import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
GPU_INIT = REPO_ROOT / "unsloth" / "_gpu_init.py"
MODEL_UTILS = REPO_ROOT / "unsloth" / "models" / "_utils.py"


def _run_hip_bf16_branch(arch, torch_reports_bf16 = True):
    source = GPU_INIT.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    branch = next(
        node
        for node in tree.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "cuda"
            for comparator in node.test.comparators
        )
    )
    hip_body = next(
        node.body
        for node in branch.orelse
        if isinstance(node, ast.If)
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "hip"
            for comparator in ast.walk(node.test)
        )
    )

    cuda = SimpleNamespace(
        get_device_properties = lambda *_: SimpleNamespace(gcnArchName = arch),
        is_bf16_supported = lambda *_, **__: torch_reports_bf16,
    )
    namespace = {"torch": SimpleNamespace(cuda = cuda)}
    exec(compile(ast.Module(body = hip_body, type_ignores = []), str(GPU_INIT), "exec"), namespace)
    return namespace["SUPPORTS_BFLOAT16"], cuda.is_bf16_supported()


@pytest.mark.parametrize("arch", ["gfx1010", "gfx1030", "gfx1032:sramecc-:xnack-"])
def test_gfx10_disables_bf16_even_when_torch_reports_support(arch):
    detected, patched_torch_probe = _run_hip_bf16_branch(arch)
    assert detected is False
    assert patched_torch_probe is False


@pytest.mark.parametrize("arch", ["gfx1100", "gfx1200", "gfx90a", "gfx942"])
def test_newer_rdna_and_cdna_keep_torch_bf16_detection(arch):
    detected, patched_torch_probe = _run_hip_bf16_branch(arch)
    assert detected is True
    assert patched_torch_probe is True


def test_torch_bf16_rejection_is_preserved_on_other_hip_architectures():
    detected, patched_torch_probe = _run_hip_bf16_branch("gfx1100", False)
    assert detected is False
    assert patched_torch_probe is False


def test_model_utils_uses_the_patched_hip_bf16_probe():
    source = MODEL_UTILS.read_text(encoding = "utf-8")
    tree = ast.parse(source)
    hip_branch = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and any(
            isinstance(comparator, ast.Constant) and comparator.value == "hip"
            for comparator in getattr(node.test, "comparators", [])
        )
        and any(
            isinstance(child, ast.Name) and child.id == "SUPPORTS_BFLOAT16"
            for child in ast.walk(node)
        )
    )
    branch_source = "\n".join(
        ast.get_source_segment(source, child) or "" for child in hip_branch.body
    )
    assert "SUPPORTS_BFLOAT16 = torch.cuda.is_bf16_supported()" in branch_source
    assert "SUPPORTS_BFLOAT16 = True" not in branch_source
