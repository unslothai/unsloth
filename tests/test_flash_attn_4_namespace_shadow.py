# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""flash-attn 4 installs the CuTe build as `flash_attn.cute` with no `flash_attn/__init__.py`,
so `flash_attn` becomes an implicit namespace package with no `flash_attn_func`,
no `flash_attn_varlen_func` and no `flash_attn.flash_attn_interface`.

xFormers imports `flash_attn.flash_attn_interface` unconditionally once `find_spec("flash_attn")`
hits, so that layout makes `import xformers.ops` raise, unsloth swallows it into
`xformers = None`, `HAS_XFORMERS` goes False and every fast-path model silently degrades to
plain SDPA (measured on a B200 at seq_len 8192 with Qwen3-0.6B + LoRA: 547 -> 2154 ms/step,
2.69 -> 19.02 GB peak).

These tests build the three module layouts on disk and check the classifier, because the
whole fix hangs off getting the classification exactly right: a real flash-attn 2 install must
not be touched, and neither must a machine with BOTH installed.
"""

import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_import_fixes():
    """The module by path, not `from unsloth import import_fixes`.

    Importing the package runs unsloth/__init__.py, which refuses to import without an
    accelerator, so the package form cannot be collected on the CPU-only CI job. This
    module is stdlib plus packaging at import time, so it loads on its own.
    """
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_under_test", _REPO_ROOT / "unsloth" / "import_fixes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import_fixes = _load_import_fixes()


def _write_layout(tmp_path, layout):
    """Materialise a site-packages-like directory for one flash_attn layout."""
    root = tmp_path / layout
    pkg = root / "flash_attn"
    if layout == "absent":
        root.mkdir(parents = True, exist_ok = True)
        return root
    if layout in ("flash_attn_4_only", "both"):
        # flash-attn 4: a `cute` subpackage and deliberately NO flash_attn/__init__.py.
        (pkg / "cute").mkdir(parents = True, exist_ok = True)
        (pkg / "cute" / "__init__.py").write_text("def flash_attn_func(*a, **k): ...\n")
    if layout in ("flash_attn_2", "both"):
        pkg.mkdir(parents = True, exist_ok = True)
        if layout == "flash_attn_2":
            # A real flash-attn 2 wheel is a regular package.
            (pkg / "__init__.py").write_text(
                "__version__ = '2.8.3'\n"
                "def flash_attn_func(*a, **k): ...\n"
                "def flash_attn_varlen_func(*a, **k): ...\n"
            )
        (pkg / "flash_attn_interface.py").write_text("flash_attn_gpu = object()\n")
    return root


# The same by-path load as _load_import_fixes, for the subprocess cases.
_LOAD_MODULE = """
    import importlib.util, pathlib, sys
    _path = pathlib.Path(sys.argv[2]) / "unsloth" / "import_fixes.py"
    _spec = importlib.util.spec_from_file_location("unsloth_import_fixes", _path)
    import_fixes = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(import_fixes)
"""

_CLASSIFY = textwrap.dedent(
    _LOAD_MODULE
    + """
    sys.path.insert(0, sys.argv[1])
    print("LAYOUT=" + import_fixes._flash_attn_layout())
    """
)


@pytest.mark.parametrize(
    "layout, expected",
    [
        ("absent", "absent"),
        ("flash_attn_2", "flash_attn_2"),
        ("flash_attn_4_only", "flash_attn_4_only"),
        # Both installed: the regular FA2 package wins the path search, so nothing to repair.
        ("both", "flash_attn_2"),
    ],
)
def test_layout_is_classified_correctly(tmp_path, layout, expected):
    root = _write_layout(tmp_path, layout)
    # A subprocess per layout: `flash_attn` cannot be un-imported cleanly between cases.
    out = subprocess.run(
        [sys.executable, "-c", _CLASSIFY, str(root), str(_REPO_ROOT)],
        capture_output = True,
        text = True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert f"LAYOUT={expected}" in out.stdout, out.stdout + out.stderr


_NO_EAGER_IMPORT = textwrap.dedent(
    # Load the module BEFORE the layout is visible, so this measures the classifier only.
    _LOAD_MODULE
    + """
    sys.modules.pop("flash_attn", None)
    sys.path.insert(0, sys.argv[1])
    import importlib
    importlib.invalidate_caches()
    import_fixes._flash_attn_layout()
    import_fixes._flash_attn_4_present()
    mod = sys.modules.get("flash_attn")
    # A namespace package has no __file__ and executes nothing; a REGULAR flash-attn 2 package
    # would have run its __init__ (and loaded flash_attn_2_cuda) if we had imported it.
    print("EXECUTED=" + str(getattr(mod, "__file__", None) is not None))
    """
)


@pytest.mark.parametrize("layout", ["absent", "flash_attn_2", "flash_attn_4_only", "both"])
def test_classification_never_imports_flash_attn(tmp_path, layout):
    """`import unsloth` must not drag in flash-attn.

    `importlib.util.find_spec("flash_attn.flash_attn_interface")` resolves the dotted name by
    IMPORTING the parent first, so classifying that way would execute `flash_attn/__init__.py`
    (and its CUDA extension) on every machine that has flash-attn 2, for users who never asked
    for it. The classifier probes the package's search locations on disk instead.
    """
    root = _write_layout(tmp_path, layout)
    out = subprocess.run(
        [sys.executable, "-c", _NO_EAGER_IMPORT, str(root), str(_REPO_ROOT)],
        capture_output = True,
        text = True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "EXECUTED=False" in out.stdout, out.stdout + out.stderr


def test_fix_is_a_noop_when_flash_attn_is_absent(monkeypatch):
    """Nothing here may change behaviour on a machine with no flash-attn."""
    monkeypatch.setattr(import_fixes, "_flash_attn_layout", lambda: "absent")
    called = []
    monkeypatch.setattr(
        import_fixes.importlib.util,
        "find_spec",
        lambda *a, **k: called.append(a) or None,
    )
    import_fixes.fix_flash_attn_4_namespace_shadow()
    assert called == [], "the fix probed for xformers on a machine with no flash-attn"


def test_fix_is_a_noop_for_a_real_flash_attn_2(monkeypatch):
    monkeypatch.setattr(import_fixes, "_flash_attn_layout", lambda: "flash_attn_2")
    called = []
    monkeypatch.setattr(
        import_fixes.importlib.util,
        "find_spec",
        lambda *a, **k: called.append(a) or None,
    )
    import_fixes.fix_flash_attn_4_namespace_shadow()
    assert called == []


def test_fix_is_a_noop_without_xformers(monkeypatch):
    """With no xformers there is nothing to protect: this machine was on SDPA either way, and
    the `attn_implementation=` delegation path never reaches flash-attn (transformers'
    `is_flash_attn_2_available()` looks up metadata for `flash_attn`, which the `flash-attn-4`
    distribution does not provide)."""
    monkeypatch.setattr(import_fixes, "_flash_attn_layout", lambda: "flash_attn_4_only")
    asked = []
    real_find_spec = import_fixes.importlib.util.find_spec

    def _find_spec(name, package = None):
        asked.append(name)
        return None if name == "xformers" else real_find_spec(name, package)

    monkeypatch.setattr(import_fixes.importlib.util, "find_spec", _find_spec)
    import_fixes._FA4_NAMESPACE_WARNED[0] = False
    import_fixes.fix_flash_attn_4_namespace_shadow()
    # Asked once, got nothing, stopped: no import, no warning, find_spec never left swapped.
    assert asked == ["xformers"]
    assert import_fixes.importlib.util.find_spec is _find_spec
    assert import_fixes._FA4_NAMESPACE_WARNED[0] is False


_BROKEN_XFORMERS = textwrap.dedent(
    _LOAD_MODULE
    + """
    import importlib.machinery, importlib.util

    # An xformers whose `ops` submodule explodes on import, standing in for an install the
    # repair cannot rescue. The top-level package is stubbed too, because the repair gates on
    # `find_spec("xformers")` first: without the stub this case would silently become the
    # no-xformers early return on any machine that has no xformers (the CPU-only CI job).
    class _StubLoader:
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            pass

    class _Boom:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "xformers.ops":
                raise RuntimeError("boom")
            if fullname == "xformers":
                spec = importlib.machinery.ModuleSpec("xformers", _StubLoader())
                spec.submodule_search_locations = []
                return spec
            return None
    sys.meta_path.insert(0, _Boom())
    for name in [m for m in sys.modules if m == "xformers" or m.startswith("xformers.")]:
        sys.modules.pop(name, None)

    import_fixes._flash_attn_layout = lambda: "flash_attn_4_only"
    import_fixes._flash_attn_4_present = lambda: True
    import_fixes._FA4_NAMESPACE_WARNED[0] = False

    before = importlib.util.find_spec
    import_fixes.fix_flash_attn_4_namespace_shadow()
    print("RESTORED=" + str(importlib.util.find_spec is before))
    print("WARNED=" + str(import_fixes._FA4_NAMESPACE_WARNED[0]))
    """
)


def test_find_spec_is_restored_even_when_xformers_import_fails():
    """The repair patches `importlib.util.find_spec` for exactly one import. If that import
    raises, the patch must still come off, or every later find_spec in the process lies --
    and the unrepairable state must warn rather than degrade in silence."""
    out = subprocess.run(
        [sys.executable, "-c", _BROKEN_XFORMERS, "", str(_REPO_ROOT)],
        capture_output = True,
        text = True,
    )
    assert "RESTORED=True" in out.stdout, out.stdout + out.stderr
    assert "WARNED=True" in out.stdout, out.stdout + out.stderr


def test_the_warning_names_the_cost_and_the_remedy(monkeypatch):
    messages = []
    monkeypatch.setattr(import_fixes.logger, "warning", lambda m: messages.append(m))
    monkeypatch.setattr(import_fixes, "_flash_attn_4_present", lambda: True)
    import_fixes._FA4_NAMESPACE_WARNED[0] = False
    import_fixes._warn_flash_attn_4_shadow_once("test")
    import_fixes._warn_flash_attn_4_shadow_once("test")
    assert len(messages) == 1
    text = messages[0]
    for needed in ("flash-attn 4", "SDPA", "3.9x", "flash_attn.cute", "pip uninstall"):
        assert needed in text, f"missing {needed!r} from:\n{text}"
