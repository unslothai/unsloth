# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The float-seed code signature is the same in every process (CPU, no torch needed).

`x in {"a", "b"}` compiles to a frozenset constant whose repr order follows
PYTHONHASHSEED. Hashed by repr, a block forward holding one got a new signature in
every process, so the saved float seeds were never read back and the SymFloat
recompile came back on every start.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from core.inference import diffusion_bg_compile as bg

BACKEND = Path(__file__).resolve().parents[1]

_FORWARD = """
def forward(self, x, mode):
    if mode in {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"}:
        return x
    if (mode, 1) in {("a", 1), ("b", 1), ("c", 1), ("d", 1), ("e", 1)}:
        return -x
    return x * 2
"""

_CHILD = (
    "import sys\n"
    "from core.inference import diffusion_bg_compile as bg\n"
    "namespace = {}\n"
    "exec(compile(sys.argv[1], 'block.py', 'exec'), namespace)\n"
    "print(bg._code_digest(namespace['forward'].__code__))\n"
)


def _digest_under_seed(seed: str) -> str:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed
    env["PYTHONPATH"] = os.pathsep.join(
        [str(BACKEND)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, "-c", _CHILD, _FORWARD],
        cwd = str(BACKEND),
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout.strip().splitlines()[-1]


def test_frozenset_constants_do_not_depend_on_hash_seed():
    digests = {seed: _digest_under_seed(seed) for seed in ("0", "1", "2", "3", "4", "5")}
    assert len(set(digests.values())) == 1, digests


def test_constants_without_a_frozenset_keep_their_repr():
    """Seed files written before the fix stay valid: only frozenset constants changed."""

    def forward(x, scale = 0.5):
        return x * 1e-6 + scale, ("a", 1, None, (2.5, b"z"))

    code = forward.__code__
    for const in code.co_consts:
        if not hasattr(const, "co_code"):
            assert bg._const_repr(const) == repr(const)


def test_frozenset_digest_still_tells_contents_apart():
    namespace_a, namespace_b = {}, {}
    exec(compile(_FORWARD, "a.py", "exec"), namespace_a)
    exec(compile(_FORWARD.replace('"theta"', '"iota"'), "b.py", "exec"), namespace_b)
    assert bg._code_digest(namespace_a["forward"].__code__) != bg._code_digest(
        namespace_b["forward"].__code__
    )
