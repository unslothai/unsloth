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
torch inside the build env), Studio's version fallback scans the file line by line for
``__version__ = `` (a re-export line silently reports "dev"), and the MLX branch of
unsloth/__init__.py imports the module directly (anything with imports defeats its
torch-free boot, which is what drove it to borrow unsloth_zoo's number instead -- unsloth#8171).

None of that is visible from `unsloth.__version__` on a GPU host, so it is pinned here.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

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
    # Every banner, every saved config's unsloth_version, and unsloth.__version__ on the
    # GPU path come through here.
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


def test_studio_version_fallback_scans_a_file_that_holds_the_literal():
    # studio/backend/main.py::get_unsloth_version falls back to scanning this file when
    # the distribution metadata is missing (source checkouts). It matches the line
    # prefix, so pointing it at a module that only re-exports the name reports "dev".
    main_py = (_REPO_ROOT / "studio" / "backend" / "main.py").read_text(encoding = "utf-8")
    match = re.search(
        r"version_file\s*=\s*_Path\(__file__\)\.resolve\(\)\.parents\[2\]\s*/(.+)", main_py
    )
    assert match, "get_unsloth_version's fallback path moved; re-pin it here"

    parts = re.findall(r"\"([^\"]+)\"", match.group(1))
    scanned = _REPO_ROOT.joinpath(*parts)
    assert scanned.exists(), f"the fallback scans {scanned}, which does not exist"

    scraped = None
    for line in scanned.read_text(encoding = "utf-8").splitlines():
        if line.startswith("__version__ = "):
            scraped = line.split("=", 1)[1].strip().strip('"').strip("'")
            break
    assert scraped == _load_version_module_standalone().__version__, (
        f"{scanned.name} has no `__version__ = ` literal for the fallback to find, so "
        "Studio would report its version as 'dev' on a source checkout."
    )


def test_the_mlx_branch_no_longer_borrows_the_zoo_version():
    # unsloth#8171: the MLX path reported unsloth_zoo's number, which is a different
    # package pinned with >=, so it was neither the installed core nor the latest zoo.
    init_py = (_REPO_ROOT / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    assert "__version__ = unsloth_zoo.__version__" not in init_py
    assert "from ._version import __version__" in init_py
