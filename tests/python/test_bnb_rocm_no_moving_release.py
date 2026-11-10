"""bitsandbytes for ROCm / XPU comes from PyPI, never from bnb's continuous-release_main.

That release is deleted and re-uploaded under the same file names on every push to bnb main,
so installing from it ran whatever bnb main last built.
"""

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MOVING = "releases/download/continuous-release"


@pytest.mark.parametrize(
    "rel",
    [
        "install.sh",
        "studio/install_python_stack.py",
        "pyproject.toml",
        "docker/Dockerfile.rocm",
        "studio/setup.ps1",
        "install.ps1",
    ],
)
def test_no_installer_pulls_bnb_from_the_moving_release(rel):
    path = ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} absent")
    assert MOVING not in path.read_text(encoding = "utf-8")


def test_rocm_bnb_falls_back_to_the_pypi_release_with_the_fix():
    tree = ast.parse((ROOT / "studio/install_python_stack.py").read_text(encoding = "utf-8"))
    values = {}
    for node in tree.body:
        target = getattr(node, "target", None) or (
            node.targets[0] if isinstance(node, ast.Assign) else None
        )
        if isinstance(target, ast.Name) and target.id in (
            "_BNB_ROCM_PRERELEASE_URLS",
            "_BNB_ROCM_PYPI_FALLBACK",
        ):
            values[target.id] = ast.literal_eval(node.value)
    assert values["_BNB_ROCM_PRERELEASE_URLS"] == {}
    assert values["_BNB_ROCM_PYPI_FALLBACK"] == "bitsandbytes>=0.50.0"
