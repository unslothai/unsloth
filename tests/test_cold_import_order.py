# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A cold `import unsloth.save` has to work, whatever imported first.

`unsloth/save.py` imports `.models.loader_utils`, which executes
`unsloth/models/__init__.py`. Three modules under `unsloth/models/` used to
import names back out of `unsloth.save` at module scope, so that chain closed a
cycle: save.py ran as far as its own `.models` import, `models/__init__.py`
reached `llama.py`, `llama.py` reached `vision.py`, and vision's
`from ..save import patch_saving_functions` landed in a half-built
`unsloth.save` and raised

    ImportError: cannot import name 'patch_saving_functions' from partially
    initialized module 'unsloth.save' (most likely due to a circular import)

`unsloth/__init__.py` hid this on the GPU path, and only there, because
`_gpu_init.py` imports `.models` before `.save`, so by the time save.py runs the
package it needs is already complete. The MLX branch of `unsloth/__init__.py`
never reaches `_gpu_init`, so on Apple Silicon a cold `import unsloth.save` hit
the cycle for real, and `import unsloth` first did not help.

Each of the three modules now defines a one-line shim of the same name that
imports the real function on first call, so the module attribute is still there
for anything that reads or patches it, and the call sites are unchanged. Two
source gates below keep the module-scope imports out (they are what fails on the
unfixed tree, on any platform, with no torch), and the behavioural cases reproduce
the original ordering in a child interpreter.
"""

import ast
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MODELS_DIR = _ROOT / "unsloth" / "models"

# The names that used to be bound at module scope, and the modules that bound them.
_DEFERRED_NAMES = (
    "patch_saving_functions",
    "unsloth_save_pretrained_torchao",
    "unsloth_save_pretrained_gguf",
)


def _model_sources():
    for path in sorted(_MODELS_DIR.rglob("*.py")):
        yield path, ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))


def _is_save_module(node, package_depth):
    """Whether an ImportFrom names `unsloth.save`.

    Both spellings count: the absolute `from unsloth.save import ...` and the
    relative `from ..save import ...` that the files under `unsloth/models/`
    actually use (`level` 2 from a module one directory below the package root).
    """
    if node.module == "unsloth.save":
        return True
    return node.level == package_depth and node.module == "save"


def _module_scope_imports(tree):
    """Only the `import`s the interpreter runs while the module object is built."""
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        # `try: from x import y` / `if TYPE_CHECKING:` at module scope still runs.
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)) and not isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                yield child


def test_no_module_under_models_imports_unsloth_save_at_module_scope():
    """The gate. `unsloth.save` imports this package, so the edge may only run
    lazily, from inside a function."""
    offenders = []
    for path, tree in _model_sources():
        depth = len(path.relative_to(_ROOT / "unsloth").parts)
        for node in _module_scope_imports(tree):
            if isinstance(node, ast.ImportFrom) and _is_save_module(node, depth):
                offenders.append(
                    f"{path.relative_to(_ROOT)}:{node.lineno} "
                    f"imports {', '.join(a.name for a in node.names)}"
                )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "unsloth.save":
                        offenders.append(f"{path.relative_to(_ROOT)}:{node.lineno} imports unsloth.save")
    assert offenders == [], (
        "a module-scope import of unsloth.save from unsloth/models/ closes the cycle that "
        "breaks a cold `import unsloth.save`; move it into the function that needs it, the "
        "way unsloth/chat_templates.py does:\n  " + "\n  ".join(offenders)
    )


def _module_level_functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_every_module_that_calls_a_deferred_name_defines_the_shim_for_it():
    """The other half of the gate. Deleting a shim must fail here rather than as a
    NameError the first time a user saves a model, and a shim that stops importing
    from `unsloth.save` is no longer a hand-off."""
    problems = []
    for path, tree in _model_sources():
        depth = len(path.relative_to(_ROOT / "unsloth").parts)
        module_functions = _module_level_functions(tree)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for name in _DEFERRED_NAMES:
            if name not in called:
                continue
            shim = module_functions.get(name)
            if shim is None:
                problems.append(f"{path.relative_to(_ROOT)} calls {name} and defines no shim for it")
                continue
            imports = {
                alias.name
                for node in ast.walk(shim)
                if isinstance(node, ast.ImportFrom) and _is_save_module(node, depth)
                for alias in node.names
            }
            if name not in imports:
                problems.append(
                    f"{path.relative_to(_ROOT)}:{shim.lineno} {name}() does not import "
                    f"{name} from unsloth.save"
                )
    assert problems == [], "\n  ".join([""] + problems)


def test_the_deferred_names_are_still_exported_by_unsloth_save():
    """Nothing here may rename a public symbol; the gate above would otherwise
    pass on a tree where the names had simply gone away."""
    source = (_ROOT / "unsloth" / "save.py").read_text(encoding = "utf-8")
    tree = ast.parse(source)
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    for name in _DEFERRED_NAMES:
        assert name in defined, f"unsloth.save no longer defines {name}"
    exported = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "__all__" for t in node.targets)
    )
    listed = {elt.value for elt in exported.elts if isinstance(elt, ast.Constant)}
    # Only `patch_saving_functions` is in `__all__`; the two save_pretrained helpers are
    # reached by name and never through `from unsloth.save import *`. Pinned so a later
    # change to `__all__` cannot silently drop the one that is exported.
    assert "patch_saving_functions" in listed


# ---------------------------------------------------------------------------
# Behavioural: the original ordering, in a child interpreter
# ---------------------------------------------------------------------------

def _has_torch():
    import importlib.util

    return all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "peft"))


_needs_torch = pytest.mark.skipif(
    not _has_torch(),
    reason = "importing unsloth.models needs torch, transformers and peft",
)


def _run(code):
    """Fresh interpreter with this checkout first on the path."""
    path = [str(_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output = True,
        text = True,
        env = dict(os.environ, PYTHONPATH = os.pathsep.join(path)),
        timeout = 900,
    )


# The MLX ordering without MLX: put a bare `unsloth` package object in sys.modules, which
# is all that `unsloth.models` has ever needed from the parent, then import `unsloth.save`
# first. Faking the MLX branch itself would need mlx, mlx_lm and an Apple Silicon uname.
_COLD_SAVE_FIRST = """
    import importlib
    import importlib.machinery
    import os
    import sys
    import types

    root = os.path.dirname(os.path.dirname(os.path.abspath(%r)))
    pkg_path = os.path.join(root, "unsloth")
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [pkg_path]
    pkg.__package__ = "unsloth"
    pkg.__file__ = os.path.join(pkg_path, "__init__.py")
    pkg.__spec__ = importlib.machinery.ModuleSpec(
        "unsloth", loader = None, origin = pkg.__file__, is_package = True,
    )
    pkg.__spec__.submodule_search_locations = [pkg_path]
    sys.modules["unsloth"] = pkg

    import unsloth.save as save

    assert callable(save.patch_saving_functions)
    for name in ("vision", "llama", "sentence_transformer"):
        assert "unsloth.models." + name in sys.modules, name
    vision = importlib.import_module("unsloth.models.vision")
    sentence = importlib.import_module("unsloth.models.sentence_transformer")
    # The shims are still module attributes, and they reach the real functions.
    assert callable(vision.patch_saving_functions)
    assert callable(sentence.unsloth_save_pretrained_gguf)
    assert callable(sentence.unsloth_save_pretrained_torchao)
    exec("from ..save import patch_saving_functions as _direct", vars(vision))
    assert vision._direct is save.patch_saving_functions
    print("COLD_SAVE_FIRST_OK")
    """


@_needs_torch
def test_a_cold_import_of_unsloth_save_does_not_hit_the_cycle():
    result = _run(_COLD_SAVE_FIRST % (str(__file__),))
    assert "COLD_SAVE_FIRST_OK" in result.stdout, (
        "a cold `import unsloth.save` with unsloth.models not yet loaded failed; this is the "
        f"Apple Silicon ordering.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "partially initialized module" not in result.stderr


@_needs_torch
@pytest.mark.parametrize(
    "first, second",
    [
        ("unsloth.save", "unsloth.models"),
        ("unsloth.models", "unsloth.save"),
        ("unsloth.models.sentence_transformer", "unsloth.save"),
        ("unsloth.save", "unsloth.models.sentence_transformer"),
    ],
)
def test_either_import_order_works_through_the_real_package_init(first, second):
    """The supported path too: `unsloth/__init__.py` runs, then both modules are
    imported in either order. This passes on the unfixed tree as well, and is here
    so the fix cannot be mistaken for having changed it."""
    result = _run(
        f"""
        import {first}
        import {second}
        import unsloth.save
        assert callable(unsloth.save.patch_saving_functions)
        print("ORDER_OK")
        """
    )
    assert "ORDER_OK" in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@_needs_torch
def test_the_star_export_from_gpu_init_still_carries_the_names():
    """`unsloth/_gpu_init.py` re-exports `unsloth.save`, so `from unsloth import
    patch_saving_functions` has to keep working for existing scripts."""
    result = _run(
        """
        import unsloth
        from unsloth import patch_saving_functions
        from unsloth.save import patch_saving_functions as direct
        assert patch_saving_functions is direct
        print("STAR_OK")
        """
    )
    assert "STAR_OK" in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def test_save_publishes_the_deferred_names_back_over_its_own_shims():
    """Source gate for the last block of `unsloth/save.py`.

    The deferral would otherwise be visible after import: `inspect.signature` on the
    module attribute would read `(*args, **kwargs)` rather than the real one, which is a
    change to what a caller can see even though every call still works. save.py ends by
    replacing its own shims with the real objects, and it identifies them by the
    `_unsloth_deferred_shim` marker so it can never overwrite somebody else's function.
    """
    source = (_ROOT / "unsloth" / "save.py").read_text(encoding = "utf-8")
    assert "_unsloth_deferred_shim" in source, (
        "unsloth/save.py no longer publishes the deferred names back to unsloth.models, so "
        "the shims stay visible after import"
    )
    for module_name in (
        "unsloth.models.vision",
        "unsloth.models.llama",
        "unsloth.models.sentence_transformer",
    ):
        assert module_name in source, f"save.py does not publish back to {module_name}"

    for path, tree in _model_sources():
        module_functions = _module_level_functions(tree)
        marked = {
            node.targets[0].value.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and node.targets[0].attr == "_unsloth_deferred_shim"
            and isinstance(node.targets[0].value, ast.Name)
        }
        for name in _DEFERRED_NAMES:
            if name not in module_functions:
                continue
            assert name in marked, (
                f"{path.relative_to(_ROOT)} defines the {name} shim but never sets "
                f"{name}._unsloth_deferred_shim, so save.py will not replace it"
            )


@_needs_torch
def test_the_module_attributes_are_the_real_functions_once_save_is_imported():
    """The behavioural half. After `import unsloth.save`, on any import order, the three
    modules expose the exact objects they exposed before the deferral: same identity, so
    same signature, same docstring and same result from anything that introspects them."""
    result = _run(
        """
        import importlib
        import inspect
        import unsloth
        import unsloth.save as save

        for module_name, names in (
            ("unsloth.models.vision", ("patch_saving_functions",)),
            ("unsloth.models.llama", ("patch_saving_functions",)),
            (
                "unsloth.models.sentence_transformer",
                ("unsloth_save_pretrained_torchao", "unsloth_save_pretrained_gguf"),
            ),
        ):
            module = importlib.import_module(module_name)
            for name in names:
                attribute = getattr(module, name)
                implementation = getattr(save, name)
                assert attribute is implementation, (module_name, name)
                assert not getattr(attribute, "_unsloth_deferred_shim", False), (
                    module_name, name,
                )
                assert inspect.signature(attribute) == inspect.signature(implementation)
        print("PUBLISHED_OK")
        """
    )
    assert "PUBLISHED_OK" in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


_SHIM_ONLY = """
    import importlib
    import importlib.machinery
    import os
    import sys
    import types

    root = os.path.dirname(os.path.dirname(os.path.abspath(%r)))
    pkg_path = os.path.join(root, "unsloth")
    pkg = types.ModuleType("unsloth")
    pkg.__path__ = [pkg_path]
    pkg.__package__ = "unsloth"
    pkg.__file__ = os.path.join(pkg_path, "__init__.py")
    pkg.__spec__ = importlib.machinery.ModuleSpec(
        "unsloth", loader = None, origin = pkg.__file__, is_package = True,
    )
    pkg.__spec__.submodule_search_locations = [pkg_path]
    sys.modules["unsloth"] = pkg

    import unsloth.models.vision as vision

    shim = vision.patch_saving_functions
    assert callable(shim)
    assert getattr(shim, "_unsloth_deferred_shim", False), (
        "unsloth.save has not been imported, so this attribute should still be the shim"
    )

    import unsloth.save as save

    # The shim captured before save.py published over it still reaches unsloth.save, with
    # the arguments untouched.
    seen = []
    save.patch_saving_functions = lambda *a, **k: seen.append((a, k))
    shim("model", vision = True)
    assert seen == [(("model",), {"vision": True})], seen
    print("SHIM_OK")
    """


@_needs_torch
def test_a_module_that_never_imports_save_still_gets_a_working_name():
    """The case the shim exists for. Import `unsloth.models.vision` and never
    `unsloth.save`, and the attribute is still there and still forwards, unchanged, to
    the real function the moment anything calls it."""
    result = _run(_SHIM_ONLY % (str(__file__),))
    assert "SHIM_OK" in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
