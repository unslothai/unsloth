"""`torch_amp_custom_fwd` and `torch_amp_custom_bwd` are exported from
`unsloth.models._utils.__all__`, so `from ._utils import *` raises AttributeError
on any device whose branch never binds them.

The chain covered cuda, hip and xpu only. On the MLX runtime, where
`DEVICE_TYPE` is `"mlx"` and `DEVICE_TYPE_TORCH` is `"mps"`, neither ran, so
importing `unsloth.models`, `unsloth.save` or `unsloth.utils.attention_dispatch`
died with `module 'unsloth.models._utils' has no attribute
'torch_amp_custom_fwd'`.

The block is sliced out with `ast` rather than imported: `unsloth.models._utils`
pulls in the rest of the package, which needs a GPU toolchain, while this branch
is a handful of assignments with no such requirement.
"""

import ast
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "unsloth" / "models" / "_utils.py"
AMP_NAMES = ("torch_amp_custom_fwd", "torch_amp_custom_bwd")

# (DEVICE_TYPE, DEVICE_TYPE_TORCH), mirroring device_type.py's mapping.
DEVICES = [("cuda", "cuda"), ("hip", "cuda"), ("xpu", "xpu"), ("mlx", "mps")]


def _binds(node, name):
    return any(
        isinstance(child, ast.Name) and child.id == name and isinstance(child.ctx, ast.Store)
        for child in ast.walk(node)
    )


def _amp_branch():
    """The top-level `if` that assigns the amp helpers."""
    tree = ast.parse(UTILS_PATH.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.If) and _binds(node, AMP_NAMES[0]):
            return node
    raise AssertionError(f"no top-level branch in {UTILS_PATH.name} assigns {AMP_NAMES[0]}")


def _fake_torch():
    amp = types.SimpleNamespace(
        custom_fwd = lambda device_type: f"fwd:{device_type}",
        custom_bwd = lambda device_type: f"bwd:{device_type}",
    )
    cuda_amp = types.SimpleNamespace(custom_fwd = "fwd:legacy", custom_bwd = "bwd:legacy")
    return types.SimpleNamespace(amp = amp, cuda = types.SimpleNamespace(amp = cuda_amp))


def test_the_amp_helpers_are_exported():
    """Without this the rest of the file would be testing nothing."""
    tree = ast.parse(UTILS_PATH.read_text(encoding = "utf-8"))
    exported = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and _binds(node, "__all__"):
            exported |= {
                element.value
                for element in ast.walk(node)
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    for name in AMP_NAMES:
        assert name in exported, f"{name} is no longer in __all__; this test needs updating"


@pytest.mark.parametrize(("device_type", "device_type_torch"), DEVICES)
def test_amp_helpers_are_bound_on_every_device(device_type, device_type_torch):
    from packaging.version import Version

    namespace = {
        "torch": _fake_torch(),
        "Version": Version,
        "torch_version": "2.9.0",
        "DEVICE_TYPE": device_type,
        "DEVICE_TYPE_TORCH": device_type_torch,
    }
    exec(compile(ast.Module(body = [_amp_branch()], type_ignores = []), "<amp>", "exec"), namespace)

    for name in AMP_NAMES:
        assert name in namespace, (
            f"DEVICE_TYPE={device_type!r} leaves {name} unbound, so "
            f"`from ._utils import *` raises AttributeError on that runtime"
        )
