# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""AMD Windows wheels ship no Triton: `import unsloth` must still work. Unsloth's own kernel modules see unsloth_zoo's
triton stub while they are defined; no other module may see it afterwards."""

import os
import subprocess
import sys
import textwrap

import pytest

# Triton made unfindable for the child, as on a host where the package is not installed.
_NO_TRITON = textwrap.dedent("""
    import importlib.machinery, sys
    _find = importlib.machinery.PathFinder.find_spec.__func__

    def _find_spec(cls, name, path = None, target = None):
        if name == "triton" or name.startswith("triton."):
            return None
        return _find(cls, name, path, target)

    importlib.machinery.PathFinder.find_spec = classmethod(_find_spec)
    for _k in [k for k in sys.modules if k == "triton" or k.startswith("triton.")]:
        del sys.modules[_k]
""")

_CHECK = textwrap.dedent("""
    import builtins, importlib.util, sys
    real_import = builtins.__import__
    import unsloth  # noqa: F401
    import unsloth.kernels.cross_entropy_loss as ce
    from unsloth_zoo.stubs import triton_stub
    assert builtins.__import__ is real_import
    assert importlib.util.find_spec("triton") is None
    assert not [k for k in sys.modules if k == "triton" or k.startswith("triton.")]
    assert ce.triton is triton_stub
    holders = [n for n, m in list(sys.modules.items())
               if m is not None and not n.startswith(("unsloth", "__main__", "__mp_main__"))
               and any(v is triton_stub or type(v).__module__ == triton_stub.__name__ for v in list(vars(m).values()))]
    assert not holders, holders
    import unsloth_zoo.fused_losses.fused_linear as fl
    assert not fl._HAS_TRITON
    print("NO_TRITON_OK")
""")


def test_unsloth_imports_without_triton():
    try:
        import unsloth_zoo.stubs.triton_stub  # noqa: F401
    except ImportError:
        pytest.skip("unsloth_zoo has no triton stub")
    import unsloth_zoo

    if getattr(unsloth_zoo, "DEVICE_TYPE", None) == "mlx":
        pytest.skip("Apple Silicon / MLX stubs Triton for every importer by design")
    # UNSLOTH_ALLOW_CPU keeps the GPU import path (models, kernels) on a host without a GPU.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_ALLOW_CPU = "1")
    # That variable also installs zoo's global stub (download-only child), which this test must not see.
    env.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
    out = subprocess.run(
        [sys.executable, "-c", _NO_TRITON + _CHECK],
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert out.returncode == 0 and "NO_TRITON_OK" in out.stdout, (
        out.stdout[-2000:],
        out.stderr[-4000:],
    )
