# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Stage 3: static sensitive-read scanner in the sandbox classifier.

Filesystem WRITE confinement is enforced at runtime by the realpath backstop (see
test_sandbox_runtime_backstop.py), which is strictly more robust than static path
proving. This static pass only blocks host-secret READS, which the backstop leaves
unpatched, so writes/deletes must pass the static gate and be confined at runtime.
"""

import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference.tools import _check_code_safety


def _blocked(code):
    assert _check_code_safety(code) is not None, code


def _ok(code):
    assert _check_code_safety(code) is None, code


class TestSensitiveReadBlocked:
    """Host-secret reads must block statically (the runtime backstop skips reads)."""

    @pytest.mark.parametrize(
        "code",
        [
            'open("/etc/passwd").read()',
            'open("/etc/shadow").read()',
            'open("../../etc/passwd").read()',
            'open("~/.ssh/id_rsa").read()',
            'open("/proc/self/environ").read()',
            'open("~/.aws/credentials").read()',
            # library loaders that internally open() the path
            'import numpy as np; np.load("/etc/shadow")',
            'import pandas as pd; pd.read_csv("/etc/passwd")',
            'from pathlib import Path; Path("/root/.ssh/id_rsa").read_text()',
            # a sensitive path anywhere (incl. write targets) is caught by the
            # callee-independent scan, which is fine (also runtime-confined)
            'import os; os.rename("data.csv", "/root/data.csv")',
        ],
    )
    def test_block(self, code):
        _blocked(code)


class TestWritesPassStaticGate:
    """Writes/deletes/renames to non-secret paths are no longer statically blocked;
    the runtime realpath backstop confines them. They must pass the static gate so
    benign in-workdir I/O is never over-blocked."""

    @pytest.mark.parametrize(
        "code",
        [
            'import shutil; shutil.rmtree("/home/user")',
            'import os; os.remove("../secret.txt")',
            'open("/etc/cron.d/x", "w").write("* * * * *")',
            'import os; os.symlink("/etc", "link")',
            'import os; os.chmod("/usr/bin/python", 0o777)',
            'open(f"/var/log/{name}", "w")',
            'import tempfile; tempfile.mkstemp(dir="/tmp")',
            'import numpy as np; np.save("/etc/x.npy", a)',
            'import os; os.makedirs("/opt/evil")',
            'import os; os.rename("data.csv", "backup/data.csv")',
        ],
    )
    def test_static_allow(self, code):
        _ok(code)


class TestBenignFilesystemAllowed:
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
            "import tempfile; f = tempfile.NamedTemporaryFile()",
            'import pandas as pd; pd.read_csv("/data/train.csv")',
            "open(fname).read()",
            'p = "ckpt.pt"\nimport torch\ntorch.save(m, p)',
            'df.to_csv("results/summary.csv")',
            'open("data_" + str(i) + ".csv").read()',
        ],
    )
    def test_allow(self, code):
        _ok(code)
