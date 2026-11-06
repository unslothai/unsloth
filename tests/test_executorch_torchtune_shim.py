# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""fix_executorch must never hide an installed torchtune (fake packages, fresh interpreters)."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IMPORT_FIXES = ROOT / "unsloth" / "import_fixes.py"

EXECUTORCH_MODELS_INIT = """\
# Copyright (c) Meta Platforms, Inc. and affiliates.

from enum import Enum


class Model(str, Enum):
    Llama = "llama"
"""

LEGACY_MAPPED_KEY_IMPORT = "from torchtune.models.convert_weights import get_mapped_key\n"


def _make_executorch(root: Path, legacy_import: bool = True) -> Path:
    models = root / "executorch" / "examples" / "models"
    (models / "qwen2_5").mkdir(parents = True)
    (root / "executorch" / "__init__.py").write_text("")
    (root / "executorch" / "examples" / "__init__.py").write_text("")
    (models / "__init__.py").write_text(EXECUTORCH_MODELS_INIT)
    (models / "qwen2_5" / "__init__.py").write_text("")
    src = (
        LEGACY_MAPPED_KEY_IMPORT
        if legacy_import
        else ("from executorch.examples.models.checkpoint import get_mapped_key\n")
    )
    (models / "qwen2_5" / "convert_weights.py").write_text(src)
    return models / "__init__.py"


def _make_real_torchtune(root: Path) -> None:
    cw = root / "torchtune" / "models"
    cw.mkdir(parents = True)
    (root / "torchtune" / "__init__.py").write_text("REAL = True\n")
    (cw / "__init__.py").write_text("")
    (cw / "convert_weights.py").write_text(
        "def get_mapped_key(key, mapping_dict):\n    return 'real'\n"
        "def tune_to_meta(sd):\n    return sd\n"
        "def hf_to_tune(sd, **kw):\n    return sd\n"
    )


# An installed torchtune must not leak into the "absent" cases: only the fake packages under `path` count.
_HIDE_SITE_TORCHTUNE = (
    "import os as _os, sys as _sys\n"
    "_sys.path[:] = [_p for _p in _sys.path if _p == {path!r} or not "
    "_os.path.exists(_os.path.join(_p or '.', 'torchtune'))]\n"
)


def _run(code: str, path: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(path)
    out = subprocess.run(
        [sys.executable, "-c", _HIDE_SITE_TORCHTUNE.format(path = str(path)) + textwrap.dedent(code)],
        env = env,
        capture_output = True,
        text = True,
        timeout = 120,
        cwd = str(path),
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def _apply_fix(path: Path) -> None:
    _run(
        f"""
        import importlib.util, json
        s = importlib.util.spec_from_file_location("_uif", {str(IMPORT_FIXES)!r})
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        m.fix_executorch()
        print(json.dumps({{}}))
        """,
        path,
    )


PROBE = """
import json, sys
{pre}
import executorch.examples.models
tt = sys.modules.get("torchtune")
from torchtune.models.convert_weights import get_mapped_key
import torchtune
res = {{
    "real": getattr(torchtune, "REAL", False),
    "mapped": get_mapped_key("layers.3.attention.wq.weight",
                             {{"layers.{{}}.attention.wq.weight": "model.layers.{{}}.q"}}),
}}
try:
    from torchtune.models import convert_weights
    convert_weights.tune_to_meta
    res["tune_to_meta"] = "ok"
except AttributeError as e:
    res["tune_to_meta"] = "missing: " + str(e)
print(json.dumps(res))
"""


def test_torchtune_absent_gets_stub(tmp_path):
    _make_executorch(tmp_path)
    _apply_fix(tmp_path)
    res = _run(PROBE.format(pre = ""), tmp_path)
    assert res["real"] is False
    assert res["mapped"] == "model.layers.3.q"
    assert "torchtune is not installed" in res["tune_to_meta"]


def test_torchtune_installed_is_not_hidden(tmp_path):
    _make_executorch(tmp_path)
    _make_real_torchtune(tmp_path)
    _apply_fix(tmp_path)
    res = _run(PROBE.format(pre = ""), tmp_path)
    assert res["real"] is True
    assert res["mapped"] == "real"
    assert res["tune_to_meta"] == "ok"


def test_torchtune_imported_first_is_kept(tmp_path):
    _make_executorch(tmp_path)
    _make_real_torchtune(tmp_path)
    _apply_fix(tmp_path)
    res = _run(PROBE.format(pre = "import torchtune.models.convert_weights"), tmp_path)
    assert res["real"] is True and res["tune_to_meta"] == "ok"


def test_torchtune_installed_after_patch(tmp_path):
    # Patch written while torchtune was absent; user installs torchtune later.
    _make_executorch(tmp_path)
    _apply_fix(tmp_path)
    _make_real_torchtune(tmp_path)
    res = _run(PROBE.format(pre = ""), tmp_path)
    assert res["real"] is True and res["tune_to_meta"] == "ok"


def test_idempotent(tmp_path):
    init = _make_executorch(tmp_path)
    _apply_fix(tmp_path)
    once = init.read_text()
    assert once != EXECUTORCH_MODELS_INIT
    _apply_fix(tmp_path)
    _apply_fix(tmp_path)
    assert init.read_text() == once


def test_no_patch_when_executorch_has_own_get_mapped_key(tmp_path):
    init = _make_executorch(tmp_path, legacy_import = False)
    _make_real_torchtune(tmp_path)
    _apply_fix(tmp_path)
    assert init.read_text() == EXECUTORCH_MODELS_INIT
    res = _run(PROBE.format(pre = ""), tmp_path)
    assert res["real"] is True


@pytest.mark.parametrize("legacy_import", [True, False])
def test_repairs_legacy_unguarded_block(tmp_path, legacy_import):
    init = _make_executorch(tmp_path, legacy_import = legacy_import)
    _make_real_torchtune(tmp_path)
    s = __import__("importlib.util").util.spec_from_file_location("_uif_t", IMPORT_FIXES)
    m = __import__("importlib.util").util.module_from_spec(s)
    s.loader.exec_module(m)
    legacy = m._LEGACY_EXECUTORCH_TORCHTUNE_SHIM
    init.write_text(
        EXECUTORCH_MODELS_INIT.replace(
            "from enum import Enum\n\n", "from enum import Enum\n\n" + legacy, 1
        )
    )
    assert _run(PROBE.format(pre = ""), tmp_path)["real"] is False
    _apply_fix(tmp_path)
    assert legacy not in init.read_text()
    if not legacy_import:
        assert init.read_text() == EXECUTORCH_MODELS_INIT
    res = _run(PROBE.format(pre = ""), tmp_path)
    assert res["real"] is True and res["tune_to_meta"] == "ok"


@pytest.mark.parametrize("editable", [False, True])
def test_legacy_stub_already_imported_is_dropped(tmp_path, editable):
    # An old patched executorch imported before Unsloth in the same process: the stub is already in sys.modules.
    init = _make_executorch(tmp_path)
    site = tmp_path / "editable" if editable else tmp_path
    site.mkdir(exist_ok = True)
    _make_real_torchtune(site)
    # PEP 660 editable installs resolve through a meta path finder, not a sys.path entry.
    finder = (
        f"""
        import importlib.machinery
        class _Editable:
            @classmethod
            def find_spec(cls, name, path = None, target = None):
                if name == "torchtune":
                    return importlib.machinery.PathFinder.find_spec(name, [{str(site)!r}])
        sys.meta_path.append(_Editable)
        """
        if editable
        else ""
    )
    s = __import__("importlib.util").util.spec_from_file_location("_uif_t2", IMPORT_FIXES)
    m = __import__("importlib.util").util.module_from_spec(s)
    s.loader.exec_module(m)
    init.write_text(
        EXECUTORCH_MODELS_INIT.replace(
            "from enum import Enum\n\n",
            "from enum import Enum\n\n" + m._LEGACY_EXECUTORCH_TORCHTUNE_SHIM,
            1,
        )
    )
    res = _run(
        f"""
        import importlib.util, json, sys
        import executorch.examples.models
        assert getattr(sys.modules["torchtune"], "REAL", False) is False
        {textwrap.indent(textwrap.dedent(finder), " " * 8).strip()}
        s = importlib.util.spec_from_file_location("_uif", {str(IMPORT_FIXES)!r})
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        m.fix_executorch()
        import torchtune
        from torchtune.models import convert_weights
        print(json.dumps({{"real": getattr(torchtune, "REAL", False), "hf_to_tune": hasattr(convert_weights, "hf_to_tune")}}))
        """,
        tmp_path,
    )
    assert res == {"real": True, "hf_to_tune": True}
