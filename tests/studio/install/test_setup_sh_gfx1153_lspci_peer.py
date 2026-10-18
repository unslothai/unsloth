# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""studio/setup.sh treats a Krackan Point 2 APU's OWN lspci line as a covered gfx1152
peer when lspci prints the Krackan die name shared with Krackan Point ("Krackan [Radeon 840M /
860M Graphics]"), so a runtime-less lone gfx1153 host gets no "gfx1153" from
_setup_unsupported_gfx_any and setup.sh forwards no --rocm-gfx gfx1153 to
install_llama_prebuilt.py (whose gfx1153 HIP bundle exists). install.sh (_infer_linux_amd_gfx_arch)
and install_python_stack.py both skip that line beside a Krackan Point 2 CPU; setup.sh must too.
Same gap in the marketing-name block (_setup_mkt names the K2, lspci Krackan line = "peer").

Each case runs in its own bash subprocess with lspci stubbed and /proc/cpuinfo redirected.
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SRC = (ROOT / "studio" / "setup.sh").read_text(encoding = "utf-8")


def _func(name):
    m = re.search(rf"^[ \t]*{re.escape(name)}\(\) \{{", SRC, re.M)
    if not m:
        return ""
    i, depth = SRC.index("{", m.start()), 0
    while i < len(SRC):
        c = SRC[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return SRC[m.start() : i + 1]
        i += 1
    raise AssertionError(name)


def _mkt_block():
    lines = SRC.split("\n")
    s = next(i for i, l in enumerate(lines) if "_setup_k2_name_ere=" in l)
    e = next(
        i
        for i, l in enumerate(lines)
        if i > s and "_setup_gfx=gfx1153" in l and "_setup_k2_peer" in l
    )
    assert lines[e + 1].strip() == "fi"
    return "\n".join(lines[s : e + 2])


K2_CPU = "model name : AMD Ryzen AI 5 330 w/ Radeon 820M\n"
KP_CPU = "model name : AMD Ryzen AI 7 350 w/ Radeon 860M\n"
SHARED = "c4:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] Krackan [Radeon 840M / 860M Graphics] [1002:1114]"
K2NAME = "c4:00.0 Display controller [0380]: Advanced Micro Devices, Inc. [AMD/ATI] Krackan Point 2 [1002:1114]"
RX7900 = "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XTX] [1002:744c]"


def _run(snippet, cpu, lspci_lines):
    d = Path(tempfile.mkdtemp(dir = os.environ.get("TMPDIR")))
    (d / "cpuinfo").write_text(cpu)
    (d / "lspci.txt").write_text("\n".join(lspci_lines) + "\n")
    funcs = "\n".join(
        _func(n)
        for n in (
            "_setup_supported_gfx_from_name",
            "_setup_unsupported_gfx_from_name",
            "_setup_unsupported_gfx_any",
        )
    ).replace("/proc/cpuinfo", '"$STUB_CPUINFO"')
    if not _func("_setup_unsupported_gfx_any"):
        pytest.skip("tree has no _setup_unsupported_gfx_any (no gfx1153 handling)")
    script = f'lspci() {{ cat "$STUB_LSPCI"; }}\n{funcs}\n{snippet}\n'
    r = subprocess.run(
        ["bash", "-c", script],
        capture_output = True,
        text = True,
        timeout = 30,
        env = {
            "PATH": os.environ["PATH"],
            "STUB_CPUINFO": str(d / "cpuinfo"),
            "STUB_LSPCI": str(d / "lspci.txt"),
        },
    )
    return r.stdout.strip()


ANY = 'out=$(_setup_unsupported_gfx_any "$MKT") && echo "HIT:$out" || echo "MISS"'


@pytest.mark.parametrize(
    "case,cpu,lspci,mkt,want",
    [
        ("lone K2, shared Krackan lspci name", K2_CPU, [SHARED], "", "HIT:gfx1153"),
        (
            "lone K2, amd-smi says 820M, shared lspci",
            K2_CPU,
            [SHARED],
            "AMD Radeon 820M",
            "HIT:gfx1153",
        ),
        ("lone K2, K2-named lspci (already fine)", K2_CPU, [K2NAME], "", "HIT:gfx1153"),
        ("K2 + RX 7900 peer keeps covered route", K2_CPU, [SHARED, RX7900], "", "MISS"),
        ("lone Krackan Point gfx1152 is supported", KP_CPU, [SHARED], "", "MISS"),
    ],
)
def test_unsupported_any(case, cpu, lspci, mkt, want):
    got = _run(f"MKT='{mkt}'\n{ANY}", cpu, lspci)
    print(case, "->", got)
    assert got == want, case


@pytest.mark.parametrize(
    "case,cpu,lspci,mkt,want",
    [
        (
            "mkt names K2 CPU, shared lspci line only",
            K2_CPU,
            [SHARED],
            "AMD Ryzen AI 5 330 w/ Radeon 820M",
            "gfx1153",
        ),
        (
            "mkt names K2 CPU, real RX 7900 peer",
            K2_CPU,
            [SHARED, RX7900],
            "AMD Ryzen AI 5 330 w/ Radeon 820M",
            "gfx1152",
        ),
    ],
)
def test_marketing_name_block(case, cpu, lspci, mkt, want):
    snippet = (
        f"_setup_mkt='{mkt}'\n"
        '_setup_gfx=$(_setup_supported_gfx_from_name "$_setup_mkt") || _setup_gfx=""\n'
        f"{_mkt_block()}\n"
        'echo "$_setup_gfx"'
    )
    got = _run(snippet, cpu, lspci)
    print(case, "->", got)
    assert got == want, case
