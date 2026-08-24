# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Diagnostic for Windows ROCm VRAM reporting on a unified-memory APU (#9314).

TEMPORARY. This exists because the change in #9314 rests on one premise that no
machine in CI can check: that on Windows, `props.total_memory` reports the BIOS
carve-out while `hipMemGetInfo`'s total spans the whole pool. On Linux the two
agree exactly, so a Linux APU cannot settle it. Delete this once a Windows APU
reading is on the PR.

It answers three questions, and needs a Windows AMD APU for any of them:

  1. Do the two totals actually disagree here? That is the premise. If they
     agree, #9314 is widening a total that was already correct.
  2. Which counter moves when the GPU actually holds memory? `--allocate` takes
     the reading, allocates, and takes it again. `Dedicated Usage` is what Studio
     reads today; if the working set lands in `Shared Usage` instead, that is the
     numerator the PR deferred for want of hardware.
  3. What does Studio report end to end, before and after the change?

Run it from a checkout, with the Studio venv's python if you have one:

    python scripts/rocm_windows_apu_probe.py
    python scripts/rocm_windows_apu_probe.py --allocate 8

`GPU Adapter Memory` lags by tens of seconds, so both readings wait for it to
stop moving first and the report says whether it did. Without that, a run
started while an earlier one is still draining reads an inflated baseline and a
still-climbing "after", and the delta can be wrong by enough to blame the wrong
counter. `--no-settle` skips the wait and is only for seeing that effect.

Nothing is uploaded. It prints a report to paste into the PR, and `--json`
writes the same data as JSON.
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


# --------------------------------------------------------------------------- #
# Windows performance counters
# --------------------------------------------------------------------------- #
def _counter(name: str) -> Optional[list[tuple[str, float]]]:
    """Read one `GPU Adapter Memory` counter for every adapter instance.

    Returns [(instance, value_bytes)], or None when the counter is unavailable,
    which is what a localized (non-English) Windows looks like from here.
    """
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
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
            timeout = 15,
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


def read_counters() -> dict[str, Any]:
    """Dedicated and Shared usage side by side, keyed by adapter instance."""
    dedicated = _counter("Dedicated Usage") or []
    shared = _counter("Shared Usage") or []
    # Union, not the dedicated list: the two are separate samples, and an adapter
    # the dedicated query happened to drop would otherwise vanish from the report
    # entirely, taking the shared reading that may hold the allocation with it.
    names = list(dict.fromkeys([n for n, _ in dedicated] + [n for n, _ in shared]))
    ded, shr = dict(dedicated), dict(shared)
    return {
        "available": bool(dedicated or shared),
        "per_adapter": [
            {
                "instance": n,
                "dedicated_gb": round(ded.get(n, 0.0) / GB, 3) if n in ded else None,
                "shared_gb": round(shr.get(n, 0.0) / GB, 3) if n in shr else None,
            }
            for n in names
        ],
    }


def _usage_totals_gb(counters: dict[str, Any]) -> tuple[float, float]:
    """Summed Dedicated and Shared usage across adapters, as a stability key.

    Both, not just Dedicated: past the carve-out Dedicated plateaus while Shared
    is still climbing, so a dedicated-only test calls that sample settled in the
    middle of the very overflow this script exists to measure.
    """
    per = counters.get("per_adapter", [])
    return (
        sum(a["dedicated_gb"] or 0.0 for a in per),
        sum(a["shared_gb"] or 0.0 for a in per),
    )


def read_counters_settled(
    timeout_s: float = 90.0,
    tol_gb: float = 0.25,
    need_stable: int = 3,
    interval_s: float = 3.0,
    baseline: Optional[tuple[float, float]] = None,
    move_gb: float = 0.0,
    min_wait_s: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Poll until Dedicated and Shared Usage both stop moving, then take the reading.

    These counters lag badly: a `GPU Process Memory` instance has been observed
    still reporting 47 GB for a pid that had already exited, and an adapter
    reading takes tens of seconds to fall back to idle. Reading them cold gives a
    "before" that is already inflated by whatever ran last and an "after" that is
    still climbing, and the resulting delta can be wrong by enough to argue for
    the wrong counter entirely. So the baseline is established rather than
    assumed, and whether it settled is reported either way: a contaminated run
    has to be visible as contaminated instead of looking like an answer.

    ``baseline`` is the pre-allocation reading. The same lag that inflates a cold
    "before" also means the counters can still be sitting at their OLD values for
    tens of seconds after an allocation lands, and three identical stale samples
    look exactly like three identical settled ones. So with a baseline the window
    is only trusted once the reading has actually left it; if it never does,
    that is reported rather than dressed up as a settled zero delta.

    These totals are host-wide, so a display adapter twitching would otherwise
    pass for the allocation landing. ``move_gb`` is therefore sized from the
    amount actually held rather than from the noise tolerance, and ``min_wait_s``
    holds the window shut for long enough that a lagged counter which happens to
    be stable at its old value cannot be mistaken for a settled one.
    """
    import time

    samples: list[tuple[float, float]] = []
    counters = read_counters()
    started = time.monotonic()
    deadline = started + timeout_s
    settled = False
    moved = baseline is None
    threshold = max(move_gb, tol_gb)
    while True:
        d, s = _usage_totals_gb(counters)
        samples.append((round(d, 3), round(s, 3)))
        if baseline is not None and not moved:
            moved = any(abs(now - was) >= threshold for now, was in zip((d, s), baseline))
        window = samples[-need_stable:]
        if (
            moved
            and time.monotonic() - started >= min_wait_s
            and len(window) >= need_stable
            and all(max(col) - min(col) <= tol_gb for col in zip(*window))
        ):
            settled = True
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(interval_s)
        counters = read_counters()
    return counters, {
        "settled": settled,
        "moved_from_baseline": moved,
        "movement_threshold_gb": round(threshold, 3),
        "samples_gb": samples,
        "tolerance_gb": tol_gb,
        "waited_s": round((len(samples) - 1) * interval_s, 1),
    }


# --------------------------------------------------------------------------- #
# torch / driver
# --------------------------------------------------------------------------- #
def read_torch() -> dict[str, Any]:
    obs: dict[str, Any] = {}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"error": f"torch not importable: {type(e).__name__}: {e}"}

    obs["torch_version"] = getattr(torch, "__version__", None)
    obs["hip"] = getattr(getattr(torch, "version", None), "hip", None)
    obs["cuda"] = getattr(getattr(torch, "version", None), "cuda", None)
    try:
        obs["device_count"] = torch.cuda.device_count()
    except Exception as e:  # noqa: BLE001
        obs["device_count"] = 0
        obs["device_count_error"] = f"{type(e).__name__}: {e}"

    devices = []
    for i in range(obs.get("device_count") or 0):
        d: dict[str, Any] = {"ordinal": i}
        try:
            p = torch.cuda.get_device_properties(i)
            d["name"] = p.name
            d["gcn_arch"] = getattr(p, "gcnArchName", None)
            d["is_integrated"] = getattr(p, "is_integrated", None)
            # THE PREMISE. On Windows these are expected to disagree on an APU.
            d["props_total_gb"] = round(p.total_memory / GB, 3)
            free_b, total_b = torch.cuda.mem_get_info(i)
            d["driver_total_gb"] = round(total_b / GB, 3)
            d["driver_free_gb"] = round(free_b / GB, 3)
            d["driver_exceeds_props"] = bool(total_b > p.total_memory)
            d["gap_gb"] = round((total_b - p.total_memory) / GB, 3)
        except Exception as e:  # noqa: BLE001
            d["error"] = f"{type(e).__name__}: {e}"
        devices.append(d)
    obs["devices"] = devices
    return obs


def read_studio(repo: Path) -> dict[str, Any]:
    """What this checkout reports, and what it would have reported before #9314."""
    backend = repo / "studio" / "backend"
    if not backend.is_dir():
        return {"error": f"no Studio backend at {backend}"}
    sys.path.insert(0, str(backend))
    obs: dict[str, Any] = {}
    try:
        import utils.hardware.hardware as hw
    except Exception as e:  # noqa: BLE001
        return {"error": f"cannot import hardware module: {type(e).__name__}: {e}"}

    # IS_ROCM is False at module scope until detection runs, and
    # _rocm_props_total_is_carve_out short-circuits on it, so without this the
    # probe reports the classifier as not firing and never exercises the changed
    # path at all -- silently, and worst on exactly the machines that matter.
    try:
        obs["detected_device"] = str(hw.ensure_hardware_detected())
    except Exception as e:  # noqa: BLE001
        obs["detect_error"] = f"{type(e).__name__}: {e}"

    obs["is_rocm"] = bool(getattr(hw, "IS_ROCM", False))
    try:
        import torch

        n = torch.cuda.device_count()
        obs["classifier_says_carve_out"] = [
            bool(hw._rocm_props_total_is_carve_out(torch.cuda.get_device_properties(i)))
            for i in range(n)
        ]
        indices = list(range(n))
    except Exception as e:  # noqa: BLE001
        obs["classifier_error"] = f"{type(e).__name__}: {e}"
        indices = [0]

    try:
        devices, aggregate = hw._rocm_windows_per_device_vram(indices)
        obs["per_device_vram"] = devices
        obs["aggregate_gb"] = aggregate
    except Exception as e:  # noqa: BLE001
        obs["per_device_vram_error"] = f"{type(e).__name__}: {e}"

    # Why a used_gb came back unknown. On a single-APU box the pairing declines
    # before the widened-total rule is ever consulted, and without this you have
    # to reconstruct that by hand to tell the two causes apart.
    try:
        adapters = hw._rocm_windows_perf_counter_vram_by_adapter()
        obs["raw_adapters"] = adapters
        if adapters:
            import torch

            totals = [int(torch.cuda.get_device_properties(i).total_memory) for i in indices]
            useds = [u for _, u in adapters]
            obs["matcher_result"] = hw._match_adapter_used_to_devices(useds, totals)
            obs["aggregate_helper"] = hw._rocm_windows_aggregate_used_bytes(useds, totals)
    except Exception as e:  # noqa: BLE001
        obs["pairing_error"] = f"{type(e).__name__}: {e}"
    try:
        obs["inventory"] = hw._torch_get_device_inventory(indices)
    except Exception as e:  # noqa: BLE001
        obs["inventory_error"] = f"{type(e).__name__}: {e}"
    try:
        obs["visible_devices"] = hw.get_visible_gpu_utilization().get("devices")
    except Exception as e:  # noqa: BLE001
        obs["visible_error"] = f"{type(e).__name__}: {e}"
    return obs


def allocate(gib: float) -> dict[str, Any]:
    """Hold `gib` on the GPU so the counters can be re-read against it.

    This is the experiment for the open question. Studio reads Dedicated Usage;
    on a unified APU the working set is expected to land in the shared segment
    instead, and only a real APU can say which counter actually moves.
    """
    import torch

    n = int(gib * GB)
    buf = torch.empty(n, dtype = torch.uint8, device = "cuda")
    buf.fill_(1)
    torch.cuda.synchronize()
    return {
        "held_gb": round(n / GB, 3),
        "torch_allocated_gb": round(torch.cuda.memory_allocated() / GB, 3),
        "torch_reserved_gb": round(torch.cuda.memory_reserved() / GB, 3),
        "_buf": buf,  # keep alive; stripped before serialising
    }


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def render(obs: dict[str, Any]) -> str:
    L: list[str] = []
    host = obs["host"]
    L.append("### Host")
    L.append("")
    L.append(f"- platform: {host['platform']} {host.get('release', '')}".rstrip())
    t = obs["torch"]
    L.append(f"- torch: {t.get('torch_version')} (hip={t.get('hip')}, cuda={t.get('cuda')})")
    L.append(f"- visible GPUs: {t.get('device_count')}")
    L.append("")

    L.append("### 1. The premise: do the two totals disagree?")
    L.append("")
    L.append("| ordinal | name | arch | integrated | props.total_memory | driver total | gap |")
    L.append("|---|---|---|---|---|---|---|")
    for d in t.get("devices", []):
        if d.get("error"):
            L.append(f"| {d['ordinal']} | error | | | | | {d['error']} |")
            continue
        L.append(
            f"| {d['ordinal']} | {d.get('name')} | {d.get('gcn_arch')} "
            f"| {d.get('is_integrated')} | {d.get('props_total_gb')} GB "
            f"| {d.get('driver_total_gb')} GB | {d.get('gap_gb')} GB |"
        )
    L.append("")
    any_gap = any(d.get("driver_exceeds_props") for d in t.get("devices", []))
    L.append(
        f"**driver total exceeds props.total_memory: {any_gap}**"
        + ("" if any_gap else "  <- if this is False on an APU, the premise does not hold here")
    )
    L.append("")

    L.append("### 2. Which counter holds the working set?")
    L.append("")
    for label in ("counters_before", "counters_after"):
        c = obs.get(label)
        if not c:
            continue
        when = "before allocation" if label == "counters_before" else "after allocation"
        if not c.get("available"):
            why = (
                "these counters are Windows-only"
                if host["platform"] != "Windows"
                else "counter unavailable, which is how a localized Windows reads from here"
            )
            L.append(f"- {when}: no reading ({why})")
            continue
        s = obs.get("settle_before" if label == "counters_before" else "settle_after")
        L.append(f"**{when}**")
        if s:
            if s["settled"]:
                L.append(
                    f"  (baseline settled after {s['waited_s']}s; "
                    f"(dedicated, shared) totals {s['samples_gb']} GB)"
                )
            elif not s.get("moved_from_baseline", True):
                L.append(
                    f"  **WARNING: never moved {s['movement_threshold_gb']} GB off "
                    f"the pre-allocation reading in {s['waited_s']}s** "
                    f"(samples {s['samples_gb']} GB). The "
                    f"allocation is resident but no counter has caught up, so the "
                    f"delta below understates it. Raise --settle-timeout and re-run."
                )
            else:
                L.append(
                    f"  **WARNING: still moving after {s['waited_s']}s** "
                    f"(samples {s['samples_gb']} GB). These counters lag, so this "
                    f"reading is contaminated by whatever ran last and the delta "
                    f"below is not trustworthy. Wait for the GPU to go idle and "
                    f"re-run."
                )
        L.append("")
        L.append("| adapter instance | Dedicated Usage | Shared Usage |")
        L.append("|---|---|---|")
        for a in c["per_adapter"]:
            L.append(f"| `{a['instance']}` | {a['dedicated_gb']} GB | {a['shared_gb']} GB |")
        L.append("")
    if obs.get("allocation"):
        a = obs["allocation"]
        L.append(
            f"Held {a['held_gb']} GB on the GPU "
            f"(torch allocated {a['torch_allocated_gb']} GB, "
            f"reserved {a['torch_reserved_gb']} GB)."
        )
        L.append("")
        before = {
            x["instance"]: x for x in (obs.get("counters_before") or {}).get("per_adapter", [])
        }
        after = {x["instance"]: x for x in (obs.get("counters_after") or {}).get("per_adapter", [])}
        if before and after:
            L.append("| adapter instance | Dedicated delta | Shared delta |")
            L.append("|---|---|---|")

            # A missing side is not zero. The report tells the reader to pick the
            # numerator by whichever delta matches the amount held, so an absent
            # baseline coerced to 0.0 would promote an absolute reading to a
            # delta and hand back a confident wrong answer.
            def _delta(key: str, b: dict, aft: dict) -> str:
                x, y = b.get(key), aft.get(key)
                if x is None or y is None:
                    return "unknown (counter missing on one side)"
                return f"{round(y - x, 3):+} GB"

            # Union of both snapshots. Dropping out of either one has to show as
            # a named unknown: an adapter absent from `after` silently vanishing
            # is how the report ends up claiming nothing moved.
            for name in dict.fromkeys(list(before) + list(after)):
                b, aft = before.get(name), after.get(name)
                if b is None or aft is None:
                    missing = "before" if b is None else "after"
                    cell = f"unknown (absent {missing})"
                    L.append(f"| `{name}` | {cell} | {cell} |")
                    continue
                L.append(
                    f"| `{name}` | {_delta('dedicated_gb', b, aft)} "
                    f"| {_delta('shared_gb', b, aft)} |"
                )
            L.append("")
        L.append(
            "Whichever counter moved by roughly the amount held is the one that tracks a "
            "resident model, and therefore the numerator Studio should be reading. A delta "
            "materially larger than the amount held means the baseline was not idle; "
            "re-run rather than believing it."
        )
        L.append("")

    L.append("### 3. What Studio reports")
    L.append("")
    s = obs.get("studio") or {}
    if s.get("error"):
        L.append(f"- not read: {s['error']}")
    else:
        L.append(f"- detected device: {s.get('detected_device')}")
        L.append(f"- IS_ROCM: {s.get('is_rocm')}")
        L.append(f"- carve-out classifier fires: {s.get('classifier_says_carve_out')}")
        L.append("")
        L.append("```json")
        L.append(
            json.dumps(
                {
                    "per_device_vram": s.get("per_device_vram"),
                    "aggregate_gb": s.get("aggregate_gb"),
                    "inventory": s.get("inventory"),
                    "raw_adapters": s.get("raw_adapters"),
                    "matcher_result": s.get("matcher_result"),
                    "aggregate_helper": s.get("aggregate_helper"),
                    "visible_devices": s.get("visible_devices"),
                },
                indent = 2,
                default = str,
            )
        )
        L.append("```")
    L.append("")
    L.append("### Before this PR, the same machine would have reported")
    L.append("")
    for d in t.get("devices", []):
        if d.get("error"):
            continue
        L.append(
            f"- ordinal {d['ordinal']}: total **{d.get('props_total_gb')} GB** "
            f"(props.total_memory verbatim), against **{d.get('driver_total_gb')} GB** now"
        )
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description = __doc__)
    ap.add_argument(
        "--allocate",
        type = float,
        default = 0.0,
        help = "GiB to hold on the GPU, to see which counter moves",
    )
    ap.add_argument(
        "--repo",
        type = Path,
        default = Path(__file__).resolve().parents[1],
        help = "checkout to read the Studio hardware module from",
    )
    ap.add_argument("--json", type = Path, default = None, help = "also write raw JSON here")
    ap.add_argument(
        "--settle-timeout",
        type = float,
        default = 90.0,
        help = "seconds to wait for the counters to stop moving before reading",
    )
    ap.add_argument(
        "--no-settle",
        action = "store_true",
        help = "read the counters cold; the reading may be contaminated by whatever ran last",
    )
    args = ap.parse_args()

    obs: dict[str, Any] = {
        "host": {
            "platform": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "torch": read_torch(),
    }
    if obs["host"]["platform"] != "Windows":
        print(
            "NOTE: this is a Windows diagnostic. On any other platform the "
            "counters are unavailable and the Windows code path is inert, "
            "which is exactly why it cannot be validated off Windows.\n"
        )

    if args.no_settle:
        obs["counters_before"] = read_counters()
    else:
        obs["counters_before"], obs["settle_before"] = read_counters_settled(
            timeout_s = args.settle_timeout
        )

    held = None
    if args.allocate > 0:
        try:
            held = allocate(args.allocate)
            obs["allocation"] = {k: v for k, v in held.items() if k != "_buf"}
            # The reading climbs for a while after the allocation lands, so the
            # "after" needs settling every bit as much as the "before".
            if args.no_settle:
                obs["counters_after"] = read_counters()
            else:
                obs["counters_after"], obs["settle_after"] = read_counters_settled(
                    timeout_s = args.settle_timeout,
                    baseline = _usage_totals_gb(obs["counters_before"]),
                    # Half the held amount: enough that a display adapter's blip
                    # cannot pass for the allocation, loose enough that a counter
                    # which only ever accounts for part of it still qualifies.
                    move_gb = 0.5 * held["held_gb"],
                    min_wait_s = 15.0,
                )
        except Exception as e:  # noqa: BLE001
            obs["allocation_error"] = f"{type(e).__name__}: {e}"

    obs["studio"] = read_studio(args.repo)

    report = render(obs)
    print(report)
    if args.json:
        args.json.write_text(json.dumps(obs, indent = 2, default = str))
        print(f"\nraw JSON written to {args.json}")
    del held
    return 0


if __name__ == "__main__":
    sys.exit(main())
