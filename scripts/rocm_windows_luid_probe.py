# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Diagnostic for the Windows AMD adapter LUID join (#8863).

TEMPORARY. Delete this once a reading from a Windows AMD machine is on the PR.

#8863 attributes each `Dedicated Usage` counter to a torch device by IDENTITY
rather than by capacity ranking: it reads the DirectX adapter records out of
`HKLM\\SOFTWARE\\Microsoft\\DirectX`, joins each counter to a record on the LUID
in the counter's instance name, and then joins that record to a torch device on
whatever the two agree about -- the model name first, the gfx target second.

Every one of those joins is against data no CI machine has. `windows-latest`
has a real registry but no AMD adapter in it, and the AMD CI runner has a real
AMD adapter but is Linux, where the whole path is inert. So the tests can only
assert that the code does the right thing with invented registry values, and
the four questions below stay open until somebody runs this:

  1. Are the values even there? `VendorId`, `AdapterLuid`, `Description`, and
     `AdapterFamily` on a GUID-named subkey. `AdapterFamily` is optional in the
     code; the other three are not.
  2. Is `AdapterLuid` a REG_QWORD? The code does `int(luid)`, which is a no-op
     for an int and a TypeError for the bytes a REG_BINARY would hand back.
  3. Do the registry LUIDs actually match the LUIDs in the counter instance
     names? The whole join is that equality. Instances are named
     `luid_0x<high>_0x<low>_phys_<n>`.
  4. Does either key connect a record to a torch device? `Description` comes
     from the driver INF and `props.name` from the ASIC record, so they are
     spelled differently often enough that the gfx fallback exists. If NEITHER
     lands, the join returns None on your machine and you silently get the old
     capacity ranking.

Run it from a checkout, with the Studio venv's python if you have one:

    python scripts/rocm_windows_luid_probe.py
    python scripts/rocm_windows_luid_probe.py --allocate 4

`--allocate` is the end-to-end check: it holds memory on one GPU and re-reads,
so the report shows whether the usage was attributed to the card that actually
has it. That is the part a two-GPU machine answers and a one-GPU machine cannot.

Reads the registry, never writes it. Nothing is uploaded. Prints a report to
paste into the PR; `--json` writes the same data as JSON.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

GB = 1024**3
AMD_PCI_VENDOR_ID = 0x1002


# --------------------------------------------------------------------------- #
# The DirectX registry, read raw
# --------------------------------------------------------------------------- #
def read_directx_records() -> dict[str, Any]:
    """Every GUID-named DirectX adapter record, with the raw types.

    Deliberately does NOT reuse the module's reader: that one is all-or-nothing
    and returns `{}` on any surprise, which is correct for it and useless here.
    A diagnostic has to show WHICH value was missing or the wrong type.
    """
    if platform.system() != "Windows":
        return {"error": "not Windows; this key does not exist here"}
    try:
        import winreg
    except ImportError as e:
        return {"error": f"winreg unavailable: {e}"}

    out: list[dict[str, Any]] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\DirectX") as dx:
            n_subkeys = winreg.QueryInfoKey(dx)[0]
            for i in range(n_subkeys):
                subkey = winreg.EnumKey(dx, i)
                rec: dict[str, Any] = {"subkey": subkey, "guid_named": subkey.startswith("{")}
                if not rec["guid_named"]:
                    out.append(rec)
                    continue
                with winreg.OpenKey(dx, subkey) as k:
                    for field in ("VendorId", "AdapterLuid", "Description", "AdapterFamily",
                                  "DeviceId", "DriverVersion"):
                        try:
                            value, regtype = winreg.QueryValueEx(k, field)
                        except OSError:
                            rec[field] = None
                            rec[f"{field}_type"] = "absent"
                            continue
                        rec[field] = value if not isinstance(value, bytes) else value.hex()
                        # 4 = REG_DWORD, 11 = REG_QWORD, 1 = REG_SZ, 3 = REG_BINARY.
                        rec[f"{field}_type"] = {
                            1: "REG_SZ", 3: "REG_BINARY", 4: "REG_DWORD", 11: "REG_QWORD",
                        }.get(regtype, f"type {regtype}")
                        rec[f"{field}_python"] = type(value).__name__
                out.append(rec)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}", "partial": out}
    amd = [r for r in out if r.get("VendorId") is not None
           and int(r["VendorId"]) == AMD_PCI_VENDOR_ID]
    return {"records": out, "amd_count": len(amd), "subkey_count": len(out)}


# --------------------------------------------------------------------------- #
# The performance counters, read raw
# --------------------------------------------------------------------------- #
def read_counter(name: str = "Dedicated Usage") -> Optional[list[tuple[str, float]]]:
    if platform.system() != "Windows":
        return None
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
            errors = "replace", timeout = 15,
        )
    except Exception:  # noqa: BLE001
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    out: list[tuple[str, float]] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line or line == "__NONE__" or "|" not in line:
            continue
        instance, _, raw = line.rpartition("|")
        try:
            out.append((instance.strip(), float(raw.strip())))
        except ValueError:
            continue
    return out or None


# --------------------------------------------------------------------------- #
# Studio, on this checkout
# --------------------------------------------------------------------------- #
def read_studio(repo: Path, allocate_gb: float, device: Optional[int]) -> dict[str, Any]:
    backend = repo / "studio" / "backend"
    if not backend.is_dir():
        return {"error": f"no Studio backend at {backend}"}
    sys.path.insert(0, str(backend))
    obs: dict[str, Any] = {}
    try:
        import utils.hardware.hardware as hw
    except Exception as e:  # noqa: BLE001
        return {"error": f"cannot import hardware module: {type(e).__name__}: {e}"}

    # IS_ROCM is False at module scope until detection runs, and this whole path
    # short-circuits on it. Without this the probe reports the join as declining
    # and never exercises the changed code at all, silently.
    try:
        obs["detected_device"] = str(hw.ensure_hardware_detected())
    except Exception as e:  # noqa: BLE001
        obs["detect_error"] = f"{type(e).__name__}: {e}"
    obs["is_rocm"] = bool(getattr(hw, "IS_ROCM", False))

    try:
        obs["records_by_luid"] = {
            str(luid): rec for luid, rec in
            hw._windows_amd_adapter_records_by_luid().items()
        }
    except Exception as e:  # noqa: BLE001
        obs["records_error"] = f"{type(e).__name__}: {e}"

    try:
        import torch
        indices = list(range(torch.cuda.device_count()))
        obs["torch_devices"] = [
            {
                "ordinal": i,
                "name": torch.cuda.get_device_properties(i).name,
                "gfx": getattr(torch.cuda.get_device_properties(i), "gcnArchName", None),
                "total_gb": round(torch.cuda.get_device_properties(i).total_memory / GB, 2),
            }
            for i in indices
        ]
    except Exception as e:  # noqa: BLE001
        obs["torch_error"] = f"{type(e).__name__}: {e}"
        indices = [0]

    held = None
    if allocate_gb > 0:
        try:
            import torch
            ordinal = device if device is not None else 0
            buf = torch.empty(int(allocate_gb * GB), dtype = torch.uint8,
                              device = f"cuda:{ordinal}")
            buf.fill_(1)
            torch.cuda.synchronize(ordinal)
            held = {"gb": round(allocate_gb, 2), "ordinal": ordinal,
                    "name": torch.cuda.get_device_properties(ordinal).name, "_buf": buf}
            obs["allocation"] = {k: v for k, v in held.items() if k != "_buf"}
            # The counters lag by tens of seconds; give them a chance to catch up.
            import time
            time.sleep(20)
        except Exception as e:  # noqa: BLE001
            obs["allocation_error"] = f"{type(e).__name__}: {e}"

    adapters = read_counter()
    obs["counter_adapters"] = adapters
    try:
        obs["parsed_luids"] = (
            [{"instance": inst, "luid": hw._parse_adapter_luid(inst), "used_gb": round(u / GB, 3)}
             for inst, u in adapters] if adapters else None
        )
    except Exception as e:  # noqa: BLE001
        obs["parse_error"] = f"{type(e).__name__}: {e}"

    try:
        devices, aggregate = hw._rocm_windows_per_device_vram(indices)
        obs["per_device_vram"] = devices
        obs["aggregate_gb"] = aggregate
    except Exception as e:  # noqa: BLE001
        obs["per_device_vram_error"] = f"{type(e).__name__}: {e}"

    try:
        obs["visible_devices"] = hw.get_visible_gpu_utilization().get("devices")
    except Exception as e:  # noqa: BLE001
        obs["visible_error"] = f"{type(e).__name__}: {e}"

    del held
    return obs


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def render(obs: dict[str, Any]) -> str:
    L: list[str] = []
    L.append("### Host")
    L.append("")
    L.append(f"- platform: {platform.system()} {platform.release()}")
    s = obs.get("studio") or {}
    L.append(f"- detected device: {s.get('detected_device')}, IS_ROCM: {s.get('is_rocm')}")
    for d in s.get("torch_devices") or []:
        L.append(f"- torch {d['ordinal']}: {d['name']} / {d['gfx']} / {d['total_gb']} GB")
    L.append("")

    L.append("### 1. DirectX adapter records")
    L.append("")
    dx = obs.get("directx") or {}
    if dx.get("error"):
        L.append(f"- not read: {dx['error']}")
    else:
        L.append(f"- {dx.get('subkey_count')} subkeys, {dx.get('amd_count')} AMD adapter(s)")
        L.append("")
        L.append("| subkey | VendorId | AdapterLuid | type | python | Description | AdapterFamily |")
        L.append("|---|---|---|---|---|---|---|")
        for r in dx.get("records") or []:
            if not r.get("guid_named"):
                continue
            L.append(
                f"| `{r['subkey'][:10]}...` | {r.get('VendorId')} | {r.get('AdapterLuid')} "
                f"| {r.get('AdapterLuid_type')} | {r.get('AdapterLuid_python')} "
                f"| {r.get('Description')} | {r.get('AdapterFamily')} |"
            )
    L.append("")
    L.append("The `AdapterLuid` type column answers whether `int(luid)` is a no-op "
             "or a TypeError. `REG_QWORD` / `int` is the expected pair.")
    L.append("")

    L.append("### 2. Counter instances and their LUIDs")
    L.append("")
    parsed = s.get("parsed_luids")
    if not parsed:
        L.append("- no counter reading (localized Windows, or the counter is unavailable)")
    else:
        L.append("| instance | parsed LUID | matches a record | used |")
        L.append("|---|---|---|---|")
        records = s.get("records_by_luid") or {}
        for p in parsed:
            luid = p["luid"]
            L.append(f"| `{p['instance']}` | {luid} "
                     f"| {'yes' if luid is not None and str(luid) in records else 'NO'} "
                     f"| {p['used_gb']} GB |")
    L.append("")

    L.append("### 3. What the module made of it")
    L.append("")
    recs = s.get("records_by_luid")
    accepted = recs if recs else "NONE (join declines, falls back to capacity ranking)"
    L.append(f"- records the module accepted: {accepted}")
    L.append(f"- per-device VRAM: {s.get('per_device_vram')}")
    L.append(f"- aggregate: {s.get('aggregate_gb')}")
    L.append(f"- System tab would show: {s.get('visible_devices')}")
    L.append("")

    if s.get("allocation"):
        a = s["allocation"]
        L.append("### 4. End to end")
        L.append("")
        L.append(f"Held {a['gb']} GB on ordinal {a['ordinal']} ({a['name']}).")
        L.append("")
        L.append("The reading above should show that GB on THAT device and not on "
                 "another one. On a single-GPU machine this only confirms the counter "
                 "moves; it takes two GPUs to show the attribution is right.")
        L.append("")
    for key in ("allocation_error", "records_error", "torch_error", "per_device_vram_error"):
        if s.get(key):
            L.append(f"- **{key}**: {s[key]}")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument("--repo", type = Path, default = Path(__file__).resolve().parents[1],
                    help = "checkout to import Studio from (default: this one)")
    ap.add_argument("--allocate", type = float, default = 0.0,
                    help = "hold N GB on a GPU and re-read, to check attribution")
    ap.add_argument("--device", type = int, default = None,
                    help = "torch ordinal to allocate on (default 0)")
    ap.add_argument("--json", type = Path, default = None, help = "also write raw JSON here")
    args = ap.parse_args()

    obs: dict[str, Any] = {"directx": read_directx_records()}
    obs["studio"] = read_studio(args.repo, args.allocate, args.device)

    report = render(obs)
    print(report)
    if args.json:
        args.json.write_text(json.dumps(obs, indent = 2, default = str))
        print(f"\n(raw JSON written to {args.json})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
