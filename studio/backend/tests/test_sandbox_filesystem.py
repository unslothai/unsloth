# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Stage 3: filesystem-confinement policy in the sandbox static classifier."""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import _check_code_safety, _resolve_path
import ast


def _blocked(code):
    assert _check_code_safety(code) is not None, code


def _ok(code):
    assert _check_code_safety(code) is None, code


def _verdict(expr):
    return _resolve_path(ast.parse(expr, mode="eval").body)


class TestPathResolver:
    @pytest.mark.parametrize(
        "expr, expect",
        [
            ('"out.txt"', "LOCAL"),
            ('"outputs/run/m.bin"', "LOCAL"),
            ('"/etc/passwd"', "ESCAPE"),
            ('"../secret"', "ESCAPE"),
            ('"~/.bashrc"', "ESCAPE"),
            ('"C:\\\\Windows"', "ESCAPE"),
            ('os.path.join("out", "a.txt")', "LOCAL"),
            ('os.path.join("out", "..", "etc")', "ESCAPE"),
            ('os.path.join("/home/u", ".ssh", "authorized_keys")', "ESCAPE"),
            ('os.path.join("sub", name)', "UNKNOWN"),
            ('Path("results") / "m.json"', "LOCAL"),
            ('Path("/tmp/x")', "ESCAPE"),
            ('os.path.expanduser("~/.bashrc")', "UNKNOWN"),
            ('f"/var/log/{name}"', "ESCAPE"),
            ('f"out/{name}"', "UNKNOWN"),
            ("fname", "UNKNOWN"),
        ],
    )
    def test_resolve(self, expr, expect):
        assert _verdict(expr) == expect, expr


class TestMutatingBlocked:
    @pytest.mark.parametrize(
        "code",
        [
            'import shutil; shutil.rmtree("/home/user")',
            'import os; os.remove("../secret.txt")',
            'open("/etc/cron.d/x", "w").write("* * * * *")',
            'import os; open(os.path.expanduser("~/.bashrc"), "a")',
            "import os; os.remove(user_path)",
            'from pathlib import Path; Path("/tmp/x").write_text("hi")',
            'import os; os.rename("data.csv", "/root/data.csv")',
            'import os; os.symlink("/etc", "link")',
            'import os; os.chdir("/")',
            'import os; os.chmod("/usr/bin/python", 0o777)',
            'import pandas as pd; df.to_csv(os.path.join("/home/u", ".ssh", "authorized_keys"))',
            'open(f"/var/log/{name}", "w")',
            'import tempfile; tempfile.mkstemp(dir="/tmp")',
            'import numpy as np; np.save("/etc/x.npy", a)',
            'import os; os.makedirs("/opt/evil")',
            'open("out/" + name, "w")',
        ],
    )
    def test_block(self, code):
        _blocked(code)


class TestReadEscapeBlocked:
    @pytest.mark.parametrize(
        "code",
        [
            'open("../../etc/passwd").read()',
            'open("/etc/shadow").read()',
            'import numpy as np; np.load("/etc/shadow")',
            'import pandas as pd; pd.read_csv("/etc/passwd")',
            'open("~/.ssh/id_rsa").read()',
        ],
    )
    def test_block(self, code):
        _blocked(code)


class TestFilesystemAllowed:
    @pytest.mark.parametrize(
        "code",
        [
            'open("out.txt", "w").write("hi")',
            'from pathlib import Path; (Path("results") / "m.json").write_text(s)',
            'import os; os.makedirs("run/ckpt", exist_ok=True)',
            'import shutil; shutil.copy("a.csv", "b.csv")',
            'import numpy as np; np.save("emb.npy", arr)',
            'import pandas as pd; df.to_parquet("out/data.parquet")',
            'import json; json.dump(d, open("r.json", "w"))',
            'import tempfile; f = tempfile.NamedTemporaryFile()',
            'import pandas as pd; pd.read_csv("/data/train.csv")',
            "open(fname).read()",
            'p = "ckpt.pt"\nimport torch\ntorch.save(m, p)',
            'df.to_csv("results/summary.csv")',
            'open("data_" + str(i) + ".csv").read()',
        ],
    )
    def test_allow(self, code):
        _ok(code)


class TestReadStrictKnob:
    def test_dynamic_read_allowed_by_default(self, monkeypatch):
        monkeypatch.delenv("FS_READ_STRICT", raising=False)
        _ok("open(fname).read()")

    def test_dynamic_read_blocked_when_strict(self, monkeypatch):
        monkeypatch.setenv("FS_READ_STRICT", "1")
        _blocked("open(fname).read()")
