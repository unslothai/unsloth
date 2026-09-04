# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A portable backend started directly must use the SAME cache directories as its launchers.

`install.sh --portable` exports `<root>/cache/uv`, `<root>/cache/cuda` and
`<root>/cache/pip` from `_export_portable_roots`, restates all three in the
`share/studio.conf` it writes and in the generated `bin/unsloth` wrapper, and the
CLI rebuilds them in `_portable_root_env`. The supported `uvicorn main:app` path
and an activated-venv launch reach none of those: `main.py` calls
`setup_cache_env()`, which derived these from `cache_root()`.

Under the NESTED layout (`install.sh --root DIR`, and bare `--portable`, whose
root is `~/.unsloth` with Studio at `~/.unsloth/studio`) `cache_root()` is
`<master>/studio/cache`, one level BELOW the master root, so the direct launch
built a second cache tree beside the installer's. uv's is the expensive half:
`utils/wheel_utils.install_wheel` and `core/training/worker._pip_install_cmd`
both prefer `uv pip install` from inside a live Studio, so they re-downloaded
wheels the installer had already cached. Under the FLAT layout the master root IS
the Studio root, so the two spellings already named one directory; these tests
pin that down rather than pretend the defect was universal.

Nothing is stranded. A `<master>/studio/cache/uv` left by an earlier direct
launch stays on disk, inside the same portable root, and merely stops being
written to; both caches are re-downloadable, which is the PIP_CACHE_DIR case and
not the MPLCONFIGDIR one.
"""

import ast
import importlib.util
import os
import re
import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

_STORAGE_ROOTS_PATH = _BACKEND_DIR / "utils" / "paths" / "storage_roots.py"
_REPO = _BACKEND_DIR.parent.parent
_INSTALL_SH = _REPO / "install.sh"
_CLI_STUDIO = _REPO / "unsloth_cli" / "commands" / "studio.py"

# Everything _setup_cache_env may set, so one test cannot see another's leftovers
# through the blank-counts-as-unset guard.
_CACHE_KEYS = (
    "UV_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "PIP_CACHE_DIR",
    "VLLM_CACHE_ROOT",
    "TORCH_HOME",
    "HF_DATASETS_CACHE",
    "HF_ASSETS_CACHE",
    "UNSLOTH_COMPILE_LOCATION",
    "UNSLOTH_STUDIO_PROJECTS_HOME",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "NUMBA_CACHE_DIR",
    "MPLCONFIGDIR",
    "DATA_DESIGNER_HOME",
    "DATA_DESIGNER_MANAGED_ASSETS_PATH",
    "TRITON_CACHE_DIR",
    "TRITON_DUMP_DIR",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_XET_CACHE",
    "HUGGINGFACE_HUB_CACHE",
)


@pytest.fixture(autouse = True)
def _clean_env(monkeypatch, tmp_path):
    for key in _CACHE_KEYS:
        monkeypatch.delenv(key, raising = False)
    # Every variable the CLI pins, so "is it in os.environ after the call" means "did the
    # resolver set it" and not "did this developer's shell already export it". NPM_CONFIG_CACHE
    # and UV_PYTHON_INSTALL_DIR are commonly set machine-wide and did exactly that here.
    for key in _cli_master_cache_vars():
        monkeypatch.delenv(key, raising = False)
    for key in ("UNSLOTH_HOME", "UNSLOTH_PORTABLE", "STUDIO_HOME", "UNSLOTH_STUDIO_HOME"):
        monkeypatch.delenv(key, raising = False)
    for key in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "TRITON_HOME"):
        monkeypatch.delenv(key, raising = False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))


def _load_storage_roots():
    sys.modules.pop("utils.hf_cache_settings", None)
    spec = importlib.util.spec_from_file_location(
        "storage_roots_master_caches", _STORAGE_ROOTS_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_flat_root(master: Path) -> None:
    """The venv-plus-marker pair _is_flat_portable_root requires to believe a flat layout."""
    venv = master / "unsloth_studio"
    venv.mkdir(parents = True)
    (venv / ".unsloth-studio-owned").write_text("")


def _cli_master_cache_vars() -> dict[str, str]:
    """`{VAR: leaf}` for every `<master>/cache/<leaf>` entry in the CLI's _portable_root_env.

    Read out of the source rather than imported: this file is under studio/backend and the
    CLI package pulls in its own dependencies at import time. Parsed with ast rather than a
    regex so an entry that stops being a `master / "cache" / ...` expression drops out of the
    contract instead of being matched by accident.
    """
    tree = ast.parse(_CLI_STUDIO.read_text(encoding = "utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name == "_portable_root_env"):
            continue
        found = {}
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Dict):
                continue
            for key, value in zip(sub.keys, sub.values):
                leaf = _master_cache_leaf(value)
                if isinstance(key, ast.Constant) and isinstance(key.value, str) and leaf:
                    found[key.value] = leaf
        return found
    return {}


def _master_cache_leaf(value: ast.AST) -> str | None:
    """The `X` of `str(master / "cache" / "X")`, or None if that is not the shape."""
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "str"
        and len(value.args) == 1
    ):
        return None
    parts = []
    node = value.args[0]
    while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
            return None
        parts.append(node.right.value)
        node = node.left
    if not (isinstance(node, ast.Name) and node.id == "master"):
        return None
    parts.reverse()
    return parts[1] if len(parts) == 2 and parts[0] == "cache" else None


def test_the_nested_layout_pins_uv_at_the_master_root_not_the_studio_root(monkeypatch, tmp_path):
    """The reported defect. install.sh exports `$UNSLOTH_ROOT/cache/uv`, which is one level
    ABOVE the Studio root here, so deriving it from cache_root() gave <master>/studio/cache/uv
    and split the installer's multi-gigabyte cache in two."""
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()

    assert sr.studio_root() == master / "studio"
    assert sr.cache_root() == master / "studio" / "cache"
    sr.setup_cache_env()

    assert os.environ["UV_CACHE_DIR"] == str(master / "cache" / "uv")


def test_the_nested_layout_pins_the_cuda_jit_cache_at_the_master_root(monkeypatch, tmp_path):
    """The same split in the same dict. _export_portable_roots exports
    `CUDA_CACHE_PATH=$UNSLOTH_ROOT/cache/cuda` right beside the uv one, so fixing only the
    reported variable would leave a second cache tree behind."""
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()

    sr.setup_cache_env()

    assert os.environ["CUDA_CACHE_PATH"] == str(master / "cache" / "cuda")


def test_the_legacy_path_portable_install_is_nested_too(monkeypatch, tmp_path):
    """Bare `install.sh --portable` with no root and no STUDIO_HOME uses `$HOME/.unsloth`,
    with Studio at `$HOME/.unsloth/studio`. That is the nested shape, so it has the split
    even though the paths look like a default install's."""
    master = Path(os.environ["HOME"]) / ".unsloth"
    (master / "studio").mkdir(parents = True)
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()

    assert sr.studio_root() == master / "studio"
    sr.setup_cache_env()

    assert os.environ["UV_CACHE_DIR"] == str(master / "cache" / "uv")
    assert os.environ["CUDA_CACHE_PATH"] == str(master / "cache" / "cuda")


def test_the_flat_layout_was_already_correct_and_stays_correct(monkeypatch, tmp_path):
    """`--portable` pointed at an existing STUDIO_HOME makes the master root the Studio root,
    so `<master>/cache/uv` and `cache_root()/uv` were always the same directory. The change
    must not move a layout that never disagreed."""
    master = tmp_path / "portable"
    _make_flat_root(master)
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()

    assert sr.studio_root() == master
    root = sr.cache_root()
    sr.setup_cache_env()

    assert os.environ["UV_CACHE_DIR"] == str(master / "cache" / "uv") == str(root / "uv")
    assert os.environ["CUDA_CACHE_PATH"] == str(master / "cache" / "cuda") == str(root / "cuda")


def test_a_normal_install_keeps_both_caches_under_the_studio_root(monkeypatch, tmp_path):
    """No master root to share with, so the portable overlay must not leak out and hoist
    these into the Studio root's parent, which Unsloth does not own on a default install."""
    studio = tmp_path / "elsewhere" / "studio"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio))
    sr = _load_storage_roots()

    assert sr.portable_mode() is False
    sr.setup_cache_env()

    assert os.environ["UV_CACHE_DIR"] == str(studio / "cache" / "uv")
    assert os.environ["CUDA_CACHE_PATH"] == str(studio / "cache" / "cuda")


def test_portable_without_a_master_root_stays_inside_the_studio_root(monkeypatch, tmp_path):
    """UNSLOTH_PORTABLE=1 on its own has no master to ask about. Containment is still what
    was asked for, so the pins fall back under cache_root() rather than onto ~/.cache/uv."""
    studio = tmp_path / "opted-in"
    monkeypatch.setenv("UNSLOTH_PORTABLE", "1")
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio))
    sr = _load_storage_roots()

    assert sr.portable_mode() is True
    assert sr.unsloth_home() is None
    sr.setup_cache_env()

    assert os.environ["UV_CACHE_DIR"] == str(studio / "cache" / "uv")
    assert os.environ["CUDA_CACHE_PATH"] == str(studio / "cache" / "cuda")


@pytest.mark.parametrize("key", ("UV_CACHE_DIR", "CUDA_CACHE_PATH"))
@pytest.mark.parametrize("preset", ("/tmp/chosen-cache", "   "))
def test_an_existing_value_is_respected_and_a_blank_one_is_not(monkeypatch, tmp_path, key, preset):
    """setdefault semantics, and the same blank-counts-as-unset rule the rest of the dict
    uses: a blank UV_CACHE_DIR is a relative path uv resolves against the working directory."""
    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    monkeypatch.setenv(key, preset)
    sr = _load_storage_roots()

    sr.setup_cache_env()

    leaf = "uv" if key == "UV_CACHE_DIR" else "cuda"
    expected = preset if preset.strip() else str(master / "cache" / leaf)
    assert os.environ[key] == expected


def test_all_four_launcher_sites_really_do_use_the_master_root(monkeypatch, tmp_path):
    """One shape, five writers. install.sh's _export_portable_roots, the share/studio.conf it
    writes, the generated bin/unsloth wrapper, the CLI's _portable_root_env and this default
    must name one directory, or a launcher and a direct launch fill two caches."""
    install_text = _INSTALL_SH.read_text(encoding = "utf-8", errors = "replace")
    for leaf, var in (("uv", "UV_CACHE_DIR"), ("cuda", "CUDA_CACHE_PATH")):
        assert re.search(
            rf'^\s*export {var}="\$UNSLOTH_ROOT/cache/{leaf}"$', install_text, re.MULTILINE
        ), f"_export_portable_roots no longer exports {var} as $UNSLOTH_ROOT/cache/{leaf}"
        # share/studio.conf, written by the printf block in _create_studio_shortcuts.
        assert (
            f"""printf '%s\\n' "export {var}='$_css_quoted_root/cache/{leaf}'\"""" in install_text
        ), f"share/studio.conf no longer records {var} as <root>/cache/{leaf}"
        # The generated bin/unsloth wrapper.
        assert (
            f"""\"export {var}='$_shim_root/cache/{leaf}'\"""" in install_text
        ), f"the generated bin/unsloth wrapper no longer sets {var} to <root>/cache/{leaf}"

    assert _cli_master_cache_vars().get("UV_CACHE_DIR") == "uv"
    assert _cli_master_cache_vars().get("CUDA_CACHE_PATH") == "cuda"

    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()
    sr.setup_cache_env()

    assert os.environ["UV_CACHE_DIR"] == str(master / "cache" / "uv")
    assert os.environ["CUDA_CACHE_PATH"] == str(master / "cache" / "cuda")


def test_every_shared_variable_agrees_with_the_cli_in_the_nested_layout(monkeypatch, tmp_path):
    """The whole class, not the two known members. Any variable that BOTH the resolver pins
    and the CLI derives as `<master>/cache/<leaf>` must resolve to the same directory; a third
    one added to either side in the nested layout fails here without anyone remembering this
    file exists. Empty would make the test vacuous, so the count is asserted too."""
    shared_leaves = _cli_master_cache_vars()
    assert len(shared_leaves) >= 4, f"parsed too few CLI cache vars: {shared_leaves}"

    master = tmp_path / "portable"
    monkeypatch.setenv("UNSLOTH_HOME", str(master))
    sr = _load_storage_roots()
    assert sr.studio_root() == master / "studio"
    sr.setup_cache_env()

    overlap = {key: leaf for key, leaf in shared_leaves.items() if key in os.environ}
    # The CLI also pins NPM_CONFIG_CACHE, BUN_INSTALL_CACHE_DIR, UV_PYTHON_INSTALL_DIR and
    # UV_TOOL_DIR, which the backend deliberately does not set: nothing inside a running
    # Studio installs a node package, a uv tool or an interpreter, so those belong to
    # setup.sh and reach it through a launcher. They are absent here rather than split.
    assert set(overlap) == {"UV_CACHE_DIR", "CUDA_CACHE_PATH", "PIP_CACHE_DIR"}, (
        "the set of cache variables the resolver and the CLI both pin has changed; route the "
        f"new one through _portable_master_cache_dir or explain why it differs: {sorted(overlap)}"
    )
    disagree = {
        key: (os.environ[key], str(master / "cache" / leaf))
        for key, leaf in overlap.items()
        if os.environ[key] != str(master / "cache" / leaf)
    }
    assert not disagree, f"resolver and CLI name different directories: {disagree}"


def test_the_runtime_uv_call_sites_this_pin_is_for_still_exist():
    """`uv pip install` really is preferred at runtime from inside a live Studio. If it stops,
    the uv half of this pin protects nothing and the comment above is stale."""
    wheel_text = (_BACKEND_DIR / "utils" / "wheel_utils.py").read_text(encoding = "utf-8")
    assert (
        'uv_cmd = ["uv", "pip", "install"]' in wheel_text
    ), "wheel_utils.install_wheel no longer shells out to uv pip install"
