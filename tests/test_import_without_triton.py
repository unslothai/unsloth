# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""On a host without Triton (for example a Windows install without triton-windows) `import unsloth` must still work.
Unsloth's own kernel modules see unsloth_zoo's triton stub while they are defined; no other module, in this thread or
another, may see it."""

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


# The hook lifted from unsloth/_gpu_init.py, run while another thread imports triton. The other thread imports at the
# point where the unsloth.* import is inside the hook's call to the real __import__ (if it makes one), else right after
# it returned: either way it must see no Triton, and nothing may be left in sys.modules / sys.meta_path.
_CONCURRENT = textwrap.dedent("""
    import ast, builtins, importlib.util, os, sys, textwrap, threading
    from unsloth_zoo.stubs import triton_stub

    path = os.path.join(importlib.util.find_spec("unsloth").submodule_search_locations[0], "_gpu_init.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_define_kernels_against_triton_stub")
    ns = {"sys": sys}
    exec(compile(ast.Module(body = [node], type_ignores = []), path, "exec"), ns)

    real_import = builtins.__import__
    checkpoint, release = threading.Event(), threading.Event()

    def paused_import(name, globals = None, locals = None, fromlist = (), level = 0):
        if name.split(".")[0] == "triton" and (globals or {}).get("__name__", "").startswith("unsloth."):
            checkpoint.set()
            assert release.wait(30)
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = paused_import
    meta_path = list(sys.meta_path)
    remove = ns["_define_kernels_against_triton_stub"]()
    kernel = {"__name__": "unsloth.kernels.example"}

    def define_kernels():
        try:
            exec("import triton", kernel)
        finally:
            checkpoint.set()

    worker = threading.Thread(target = define_kernels)
    worker.start()
    assert checkpoint.wait(30)
    outsider = {"__name__": "some_library"}
    try:
        exec("import triton", outsider)
    except ModuleNotFoundError:
        pass
    leaked = [k for k in sys.modules if k == "triton" or k.startswith("triton.")]
    finders = [f for f in sys.meta_path if f not in meta_path]
    release.set()
    worker.join(30)
    assert outsider.get("triton") is None, "another thread received the triton stub"
    assert not leaked and not finders, (leaked, finders)
    assert kernel["triton"] is triton_stub

    # Every import form an unsloth.* module uses, handed the stub without publishing it.
    forms = '''
        import triton
        import triton.language as tl
        import triton.language
        from triton import jit, cdiv
        from triton.language import constexpr
        from triton.language.extra import libdevice
        import triton.runtime.jit as rjit
        import triton.experimental.deep.module as deep
        from triton.experimental.deep import module as deep2
    '''
    mod = {"__name__": "unsloth.kernels.forms"}
    exec(textwrap.dedent(forms), mod)
    assert mod["triton"] is triton_stub and mod["tl"] is triton_stub.language
    assert mod["jit"] is triton_stub.jit and mod["cdiv"] is triton_stub.cdiv
    assert mod["constexpr"] is triton_stub._ConstExpr
    assert mod["rjit"] is triton_stub.runtime.jit
    assert mod["deep"].__name__ == "triton.experimental.deep.module" and mod["deep2"] is mod["deep"]
    assert repr(mod["libdevice"]).startswith("<triton_stub")
    assert not [k for k in sys.modules if k == "triton" or k.startswith("triton.")]
    assert sys.meta_path == meta_path
    remove()
    assert builtins.__import__ is paused_import
    print("NO_TRITON_CONCURRENT_OK")
""")


def test_triton_stub_never_reaches_other_importers():
    try:
        import unsloth_zoo.stubs.triton_stub  # noqa: F401
    except ImportError:
        pytest.skip("unsloth_zoo has no triton stub")
    import unsloth_zoo

    if getattr(unsloth_zoo, "DEVICE_TYPE", None) == "mlx":
        pytest.skip("Apple Silicon / MLX stubs Triton for every importer by design")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES = "", UNSLOTH_ALLOW_CPU = "1")
    env.pop("UNSLOTH_ZOO_DISABLE_GPU_INIT", None)
    out = subprocess.run(
        [sys.executable, "-c", _NO_TRITON + _CONCURRENT],
        env = env,
        capture_output = True,
        text = True,
        timeout = 900,
    )
    assert out.returncode == 0 and "NO_TRITON_CONCURRENT_OK" in out.stdout, (
        out.stdout[-2000:],
        out.stderr[-4000:],
    )
