# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`import unsloth` must not import the openai SDK itself (about 0.6 s), while a broken openai
install still reads as unavailable through `transformers.utils.is_openai_available`.

Before: unsloth/models/_utils.py ran `from openai import OpenAI` at module level whenever openai
was installed, only to flip the transformers check to False when that import failed."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

_TESTS = str(Path(__file__).resolve().parent)
_MSG = "Unsloth: OpenAI failed to import - ignoring for now."

# Records the frame whose import statement loaded openai (the first frame outside importlib), so an
# eager import by a dependency (trl<0.26 judges, during `import unsloth`) is not blamed on unsloth.
_HEALTHY = """
import sys
_importers = []
_f = sys._getframe(1)
while _f is not None and "importlib" in _f.f_code.co_filename:
    _f = _f.f_back
if _f is not None:
    _importers.append((_f.f_code.co_filename.replace("\\\\", "/"), _f.f_code.co_name))
class OpenAI:
    pass
"""

_BROKEN = 'raise TypeError("simulated broken openai install")\n'

# Slow to fail, so a second thread's first call arrives while the first thread is still importing.
_SLOW_BROKEN = 'import time\ntime.sleep(1.0)\nraise TypeError("simulated broken openai install")\n'

_CHECK = r"""
import os, sys
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
sys.path.insert(0, sys.argv[1])
import _zoo_aggressive_cuda_spoof as _spoof
_spoof.apply()
import unsloth
import transformers.utils as tu
import transformers.utils.import_utils as iu
mode = sys.argv[2]
if mode == "threads":
    import threading
    print("CALLS", flush = True)
    from unsloth.models._utils import _lazy_openai_import_check
    # A fresh wrapper: trl<0.26 judges already made the first call during import unsloth.
    check, answers = _lazy_openai_import_check(iu.is_openai_available), []
    workers = [threading.Thread(target = lambda: answers.append(check())) for _ in range(4)]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    assert answers == [False] * 4, answers
elif mode == "healthy":
    importers = getattr(sys.modules.get("openai"), "_importers", [])
    eager = [f for f in importers if f[0].endswith("unsloth/models/_utils.py") and f[1] == "<module>"]
    assert not eager, eager
    assert tu.is_openai_available() is True
    assert "openai" in sys.modules
else:
    print("CALLS", flush = True)
    assert tu.is_openai_available() is False
    assert tu.is_openai_available() is False
    assert iu.is_openai_available() is True
# Exactly one wrapper layer around transformers' own check (idempotent patch).
cells = [c.cell_contents for c in (tu.is_openai_available.__closure__ or ())]
assert getattr(tu.is_openai_available, "_unsloth_lazy_openai_check", False), tu.is_openai_available
assert any(getattr(c, "__module__", "") == iu.__name__ and c.__name__ == "is_openai_available" for c in cells), cells
print("OK", flush = True)
"""


def _fake_openai(root, body):
    pkg = root / "openai"
    pkg.mkdir()
    (pkg / "__init__.py").write_text(body, encoding = "utf-8")
    dist = root / "openai-9.9.9.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: openai\nVersion: 9.9.9\n", encoding = "utf-8"
    )
    (dist / "top_level.txt").write_text("openai\n", encoding = "utf-8")


def _run(tmp_path, mode, body):
    _fake_openai(tmp_path, body)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(tmp_path), env.get("PYTHONPATH")) if p)
    # Keep an Apple Silicon host with mlx installed on the GPU path, where unsloth/models/_utils.py runs.
    env["UNSLOTH_FORCE_GPU_PATH"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", _CHECK, _TESTS, mode],
        capture_output = True,
        text = True,
        timeout = 900,
        env = env,
    )
    assert result.returncode == 0 and "OK" in result.stdout, (
        result.stdout[-2000:] + result.stderr[-4000:]
    )
    return result.stdout


def test_import_unsloth_does_not_import_openai_itself(tmp_path):
    _run(tmp_path, "healthy", _HEALTHY)


def test_broken_openai_still_reads_unavailable_and_warns_once(tmp_path):
    out = _run(tmp_path, "broken", _BROKEN)
    # Printed once per process, by whichever caller asks first (trl<0.26 judges ask during import).
    assert out.count(_MSG) == 1, out[-2000:]


def test_concurrent_first_calls_on_a_broken_openai_warn_once(tmp_path):
    out = _run(tmp_path, "threads", _SLOW_BROKEN)
    # Concurrent first calls make one import attempt and print the message once.
    assert out.split("CALLS", 1)[1].count(_MSG) == 1, out[-2000:]
