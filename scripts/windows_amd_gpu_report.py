# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Standalone Windows AMD GPU report. No install, no checkout, no dependencies.

WHAT THIS IS FOR

Unsloth Studio reports per-GPU VRAM on Windows AMD by reading two things and
joining them: the `GPU Adapter Memory` performance counters, and the adapter
records DirectX keeps in the registry. We have no Windows machine with an AMD
GPU in CI, so several plain questions about that pairing are open, and they are
questions about YOUR MACHINE rather than about any particular patch. This script
answers them and prints a report to paste back into the pull request.

WHAT IT DOES, EXACTLY

  * reads three registry keys under HKLM\\SOFTWARE\\Microsoft\\DirectX (read
    only; it never writes, creates or deletes anything)
  * runs `Get-Counter` twice, which is the same thing Task Manager reads
  * prints text

It does not need administrator. It does not need Python packages: standard
library only, so any Python 3.9+ will do, including the one from the Microsoft
Store. It does not import torch, does not import Unsloth, does not touch the
GPU, does not read your files, and makes no network connection of any kind.
Nothing is uploaded; you decide what to paste.

HOW TO RUN

    python windows_amd_gpu_report.py

Takes a couple of seconds. If you also want to see the counters move, hold some
VRAM in any other program (a game, a model load, anything) and run it again.
"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys

GB = 1024**3
AMD_PCI_VENDOR_ID = 0x1002

# 1 = REG_SZ, 3 = REG_BINARY, 4 = REG_DWORD, 11 = REG_QWORD.
REG_TYPES = {1: "REG_SZ", 2: "REG_EXPAND_SZ", 3: "REG_BINARY", 4: "REG_DWORD",
             7: "REG_MULTI_SZ", 11: "REG_QWORD"}


def directx_records() -> dict:
    """Every GUID-named DirectX adapter record, with registry and Python types."""
    if platform.system() != "Windows":
        return {"error": f"this is {platform.system()}; the DirectX key is Windows-only"}
    try:
        import winreg
    except ImportError as e:
        return {"error": f"winreg unavailable: {e}"}

    records, err = [], None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\DirectX") as dx:
            for i in range(winreg.QueryInfoKey(dx)[0]):
                subkey = winreg.EnumKey(dx, i)
                if not (subkey.startswith("{") and subkey.endswith("}")):
                    continue  # ShaderCache and friends are not adapters
                rec = {"subkey": subkey}
                try:
                    with winreg.OpenKey(dx, subkey) as k:
                        for field in ("VendorId", "DeviceId", "AdapterLuid",
                                      "Description", "AdapterFamily", "DriverVersion"):
                            try:
                                value, regtype = winreg.QueryValueEx(k, field)
                            except OSError:
                                rec[field] = None
                                rec[field + "_type"] = "ABSENT"
                                continue
                            rec[field] = value.hex() if isinstance(value, bytes) else value
                            rec[field + "_type"] = REG_TYPES.get(regtype, "type %s" % regtype)
                            rec[field + "_py"] = type(value).__name__
                except OSError as e:
                    rec["error"] = str(e)
                records.append(rec)
    except FileNotFoundError:
        err = r"HKLM\SOFTWARE\Microsoft\DirectX does not exist on this machine"
    except PermissionError as e:
        err = f"permission denied reading the DirectX key: {e}"
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    return {"records": records, "error": err}


def counter(name: str) -> dict:
    """One `GPU Adapter Memory` counter for every adapter instance."""
    if platform.system() != "Windows":
        return {"error": f"this is {platform.system()}; these counters are Windows-only"}
    ps = (
        f"$s=(Get-Counter '\\GPU Adapter Memory(*)\\{name}'"
        " -ErrorAction SilentlyContinue).CounterSamples;"
        "if($s){$s|ForEach-Object{'{0}|{1}' -f $_.InstanceName,[int64]$_.CookedValue}}"
        "else{'__NONE__'}"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output = True, text = True, encoding = "utf-8",
            errors = "replace", timeout = 60,
        )
    except FileNotFoundError:
        return {"error": "powershell not found on PATH"}
    except subprocess.TimeoutExpired:
        return {"error": "Get-Counter timed out after 60s"}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    if r.returncode != 0:
        return {"error": f"powershell exited {r.returncode}: {r.stderr.strip()[:300]}"}
    if "__NONE__" in r.stdout:
        return {"error": "the counter is not available (this is how a localized "
                         "Windows looks from here, and is itself a useful answer)"}
    rows = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        instance, _, raw = line.rpartition("|")
        try:
            rows.append({"instance": instance.strip(), "bytes": float(raw.strip())})
        except ValueError:
            continue
    return {"rows": rows} if rows else {"error": "no instances parsed from the counter"}


def parse_instance_luid(instance: str):
    """`luid_0x<high>_0x<low>_phys_<n>` -> the 64-bit LUID, or None.

    Deliberately character for character what `_parse_adapter_luid` does in
    `studio/backend/utils/hardware/hardware.py`, including the strip and the
    case-insensitive match. If this were stricter, a machine reporting the
    instance names in a different case would be reported here as "the LUIDs do
    not line up" while the real code paired them perfectly well, and we would go
    chasing a bug that is not there.
    """
    m = re.match(r"luid_0x([0-9a-f]+)_0x([0-9a-f]+)", instance.strip(), re.IGNORECASE)
    if m is None:
        return None
    try:
        return (int(m.group(1), 16) << 32) | int(m.group(2), 16)
    except ValueError:
        return None


def main() -> int:
    obs = {
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "directx": directx_records(),
        "dedicated": counter("Dedicated Usage"),
        "shared": counter("Shared Usage"),
    }

    L = []
    A = L.append
    A("## Windows AMD GPU report")
    A("")
    A(f"- {obs['platform']}, Python {obs['python']}")
    A("")

    A("### 1. DirectX adapter records")
    A("")
    dx = obs["directx"]
    if dx.get("error"):
        A(f"**Could not read:** {dx['error']}")
    recs = dx.get("records") or []
    amd = [r for r in recs if r.get("VendorId") is not None
           and str(r["VendorId"]).isdigit() and int(r["VendorId"]) == AMD_PCI_VENDOR_ID]
    A(f"{len(recs)} adapter record(s), {len(amd)} of them AMD (vendor 0x1002).")
    A("")
    if recs:
        A("| record | VendorId | AdapterLuid | LUID type | Python type | Description | AdapterFamily |")
        A("|---|---|---|---|---|---|---|")
        for r in recs:
            A("| `{}` | {} | {} | {} | {} | {} | {} |".format(
                r["subkey"][:9] + "...", r.get("VendorId"), r.get("AdapterLuid"),
                r.get("AdapterLuid_type"), r.get("AdapterLuid_py"),
                r.get("Description"), r.get("AdapterFamily")))
    A("")
    A("The **LUID type** column is the one to look at: `REG_QWORD` with Python "
      "type `int` is what the code assumes. `REG_BINARY` with `bytes` would mean "
      "it is wrong. `ABSENT` would mean the join has nothing to work with.")
    A("")

    A("### 2. GPU Adapter Memory counters")
    A("")
    for label, key in (("Dedicated Usage", "dedicated"), ("Shared Usage", "shared")):
        c = obs[key]
        if c.get("error"):
            A(f"**{label}:** not read ({c['error']})")
            A("")
            continue
        A(f"**{label}**")
        A("")
        A("| instance | LUID from the name | GB |")
        A("|---|---|---|")
        for row in c["rows"]:
            A(f"| `{row['instance']}` | {parse_instance_luid(row['instance'])} "
              f"| {round(row['bytes'] / GB, 3)} |")
        A("")

    A("### 3. Do the two line up?")
    A("")
    reg_luids = set()
    for r in amd:
        try:
            reg_luids.add(int(r["AdapterLuid"]))
        except (TypeError, ValueError):
            pass
    ded = obs["dedicated"].get("rows") or []
    counter_luids = {parse_instance_luid(row["instance"]) for row in ded}
    counter_luids.discard(None)
    if not reg_luids or not counter_luids:
        A("Cannot tell: one of the two sides came back empty above.")
    else:
        matched = reg_luids & counter_luids
        A(f"- AMD LUIDs in the registry: {sorted(reg_luids)}")
        A(f"- LUIDs in the counter names: {sorted(counter_luids)}")
        A(f"- **in both: {sorted(matched) if matched else 'NONE'}**")
        A("")
        A("Every AMD adapter appearing on both sides is the result the code needs. "
          "`NONE` would mean the join cannot work on this machine at all, which is "
          "the single most useful thing this script can tell us.")
    A("")
    A("<details><summary>raw JSON</summary>")
    A("")
    A("```json")
    A(json.dumps(obs, indent = 2, default = str))
    A("```")
    A("")
    A("</details>")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main())
