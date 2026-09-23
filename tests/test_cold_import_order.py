# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A cold `import unsloth.save` has to work, whatever imported first.

save.py -> .models.loader_utils -> models/__init__ -> llama -> vision, whose module-scope
`from ..save import ...` landed back in the half-built save.py. `_gpu_init.py` hid that by
importing `.models` first, but the MLX branch never reaches it.
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
        yield path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _is_save_module(node, package_depth):
    """Absolute or relative: `from ..save` is level 2 one directory below the root."""
    if node.module == "unsloth.save":
        return True
    return node.level == package_depth and node.module == "save"


def _module_scope_imports(tree):
    # Only the `import`s the interpreter runs while the module object is built.
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
                        offenders.append(
                            f"{path.relative_to(_ROOT)}:{node.lineno} imports unsloth.save"
                        )
    assert offenders == [], (
        "a module-scope import of unsloth.save from unsloth/models/ closes the cycle that "
        "breaks a cold `import unsloth.save`; move it into the function that needs it, the "
        "way unsloth/chat_templates.py does:\n  " + "\n  ".join(offenders)
    )


def _module_level_functions(tree):
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_every_module_that_calls_a_deferred_name_defines_the_shim_for_it():
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
                problems.append(
                    f"{path.relative_to(_ROOT)} calls {name} and defines no shim for it"
                )
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
    """The gate above also passes on a tree where the names simply went away."""
    source = (_ROOT / "unsloth" / "save.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    defined = {
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
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
    # Only this one is in `__all__`; the save_pretrained helpers are reached by name.
    assert "patch_saving_functions" in listed


def _run(code):
    path = [str(_ROOT)]
    if os.environ.get("PYTHONPATH"):
        path.append(os.environ["PYTHONPATH"])
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(path)),
        timeout=900,
    )


def _models_importable():
    """Not `find_spec("torch")`: a GitHub macOS runner has torch and still cannot import
    this, because `unsloth_zoo.device_type` raises NotImplementedError with no accelerator."""
    result = _run(
        """
        import unsloth.models  # noqa: F401
        print("MODELS_OK")
        """
    )
    return "MODELS_OK" in result.stdout


_needs_torch = pytest.mark.skipif(
    not _models_importable(),
    reason=(
        "importing unsloth.models does not complete in this environment (no torch, or no "
        "accelerator unsloth_zoo recognises)"
    ),
)


# The MLX ordering without MLX: faking the branch needs mlx, mlx_lm and an Apple uname.
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
    """Passes on the unfixed tree too, so the fix cannot be mistaken for changing it."""
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
    """`from unsloth import patch_saving_functions` has to keep working for old scripts."""
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
    """Without the publish-back, `inspect.signature` would read `(*args, **kwargs)`."""
    source = (_ROOT / "unsloth" / "save.py").read_text(encoding="utf-8")
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
    result = _run(_SHIM_ONLY % (str(__file__),))
    assert "SHIM_OK" in result.stdout, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
