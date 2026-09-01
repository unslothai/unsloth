# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-CPU, no-network tests pinning unsloth/_version.py as the one source of the version.

Three consumers read that literal by different means and each breaks differently when it
moves: pyproject resolves it with a static AST parse (a non-literal makes setuptools import
torch inside the build env), Unsloth's version fallback scans the file line by line for
``__version__ = `` (a re-export line silently reports "dev"), and the MLX branch of
unsloth/__init__.py imports the module directly (anything with imports defeats its
torch-free boot, which is what drove it to borrow unsloth_zoo's number instead -- unsloth#8171).

None of that is visible from `unsloth.__version__` on a GPU host, so it is pinned here.
"""

import ast
import re
import sys
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from pathlib import Path

import pytest


def _raise_not_found(name):
    raise _PackageNotFoundError(name)


_REPO_ROOT = Path(__file__).resolve().parents[1]
_VERSION_FILE = _REPO_ROOT / "unsloth" / "_version.py"

_HEAVY = ("torch", "transformers", "trl", "peft", "triton", "unsloth_zoo")


def _load_version_module_standalone():
    """Load _version.py off disk, bypassing the unsloth package __init__."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_unsloth_version_probe", _VERSION_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_version_file_is_a_literal_in_a_module_with_no_imports():
    tree = ast.parse(_VERSION_FILE.read_text(encoding = "utf-8"))

    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == [], (
        "unsloth/_version.py must import nothing: pyproject reads it with a static AST "
        "parse, and the torch-free MLX path imports it directly."
    )

    assigns = [n for n in tree.body if isinstance(n, ast.Assign)]
    assert len(assigns) == 1, "the module should hold the version and nothing else"
    target, value = assigns[0].targets[0], assigns[0].value
    assert target.id == "__version__"
    assert isinstance(value, ast.Constant) and isinstance(value.value, str), (
        "__version__ must stay a plain string literal, else setuptools falls back to "
        "importing unsloth.models._utils (and torch) at build time."
    )


def test_importing_the_version_alone_pulls_in_no_heavy_dependency():
    before = {name for name in _HEAVY if name in sys.modules}
    module = _load_version_module_standalone()
    after = {name for name in _HEAVY if name in sys.modules}

    assert isinstance(module.__version__, str) and module.__version__
    assert after - before == set(), "reading the version must not import the world"


def test_models_utils_still_re_exports_the_same_version():
    # Every banner, every saved config's unsloth_version, and unsloth.__version__ on the GPU path come through here.
    from unsloth.models._utils import __version__ as via_utils
    assert via_utils == _load_version_module_standalone().__version__


def test_pyproject_reads_the_version_from_the_leaf_module():
    pyproject = (_REPO_ROOT / "pyproject.toml").read_text(encoding = "utf-8")
    match = re.search(r"^version\s*=\s*\{attr\s*=\s*\"([^\"]+)\"\}", pyproject, re.MULTILINE)
    assert match, "pyproject must keep deriving the distribution version from an attr"
    assert match.group(1) == "unsloth._version.__version__"


def test_setuptools_resolves_the_version_without_importing_torch():
    read_attr = pytest.importorskip("setuptools.config.expand").read_attr

    before = {name for name in _HEAVY if name in sys.modules}
    resolved = read_attr("unsloth._version.__version__", root_dir = str(_REPO_ROOT))
    after = {name for name in _HEAVY if name in sys.modules}

    assert resolved == _load_version_module_standalone().__version__
    assert after - before == set(), (
        "the build must resolve the version statically; importing unsloth.models._utils "
        "here would drag torch into the build environment."
    )


def _get_unsloth_version_with_metadata_missing(main_py_path):
    """Run studio/backend/main.py::get_unsloth_version with the distribution metadata
    forced absent, which is the source-checkout case its file scan exists for.

    The function body is exec'd rather than imported because importing main.py starts the
    whole FastAPI backend.
    """
    src = _Path_read(main_py_path)
    start = src.index("def get_unsloth_version()")
    body = src[start : src.index("\n\n\n", start)]
    namespace = {
        "_Path": Path,
        "PackageNotFoundError": _PackageNotFoundError,
        "package_version": _raise_not_found,
        "__file__": str(main_py_path),
    }
    exec(compile(body, "main.py", "exec"), namespace)
    return namespace["get_unsloth_version"]()


def _Path_read(p):
    return Path(p).read_text(encoding = "utf-8")


def test_studio_version_fallback_reports_the_real_version_on_a_source_checkout():
    # get_unsloth_version falls back to scanning the source when distribution metadata is missing.
    reported = _get_unsloth_version_with_metadata_missing(
        _REPO_ROOT / "studio" / "backend" / "main.py"
    )
    assert reported == _load_version_module_standalone().__version__, (
        f"the fallback reported {reported!r}; Unsloth would show that instead of the "
        "real version on a source checkout."
    )


def test_the_studio_fallback_survives_a_half_updated_tree(tmp_path):
    # main.py beside an old models/_utils.py or the reverse. Either file alone must still
    # The fallback is what a source checkout relies on, and a checkout can be half updated:
    import re as _re

    main_py = (_REPO_ROOT / "studio" / "backend" / "main.py").read_text(encoding = "utf-8")
    start = main_py.index("def get_unsloth_version()")
    body = main_py[start : main_py.index("\n\n\n", start)]

    def _version_for(layout):
        root = tmp_path / layout
        (root / "unsloth" / "models").mkdir(parents = True)
        (root / "studio" / "backend").mkdir(parents = True)
        if layout in ("current", "only_version"):
            (root / "unsloth" / "_version.py").write_text('__version__ = "9.9.9"\n')
        if layout == "only_utils":
            (root / "unsloth" / "models" / "_utils.py").write_text('__version__ = "9.9.9"\n')
        else:
            (root / "unsloth" / "models" / "_utils.py").write_text(
                "from .._version import __version__\n"
            )
        namespace = {
            "_Path": Path,
            "PackageNotFoundError": _PackageNotFoundError,
            # Force the metadata lookup to miss, which is the source-checkout case.
            "package_version": _raise_not_found,
            "__file__": str(root / "studio" / "backend" / "main.py"),
        }
        exec(compile(body, "main.py", "exec"), namespace)
        return namespace["get_unsloth_version"]()

    assert _version_for("current") == "9.9.9"
    assert _version_for("only_version") == "9.9.9"
    assert _version_for("only_utils") == "9.9.9"
    # Neither file carries a literal: reporting "dev" is correct, not a silent wrong number.
    assert _version_for("neither") == "dev"


def test_the_mlx_branch_no_longer_borrows_the_zoo_version():
    # unsloth#8171:
    # unsloth#8171: the MLX path reported unsloth_zoo's number, which is a different package pinned with >=, so it was
    init_py = (_REPO_ROOT / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    assert "__version__ = unsloth_zoo.__version__" not in init_py
    assert "from ._version import __version__" in init_py
