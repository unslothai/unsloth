# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe: do GPU-to-GPU peer copies on this host actually transfer the data?

On bare-metal Linux with a translating IOMMU, a PCIe peer-to-peer copy between two
NVIDIA GPUs can be discarded by the chipset while CUDA still reports
``cudaSuccess``, and ``torch.cuda.can_device_access_peer`` still returns True, so
nothing upstream of the data notices. In llama.cpp with ``GGML_CUDA_P2P`` set, the
model emits ``!!!!!``, ``/////``, a repeated token or word salad, which looks
exactly like a broken quant or chat template. See issue #10613.

The destination is filled with a sentinel before each copy, so a transfer that
moves nothing is distinguishable from one that legitimately writes zeros. Every
ordered pair is tested at several sizes, because small copies can succeed through
a different path than large ones.

Run with no arguments on the host in question::

    python scripts/p2p_integrity_probe.py

Exit 0: every peer copy transferred intact. Exit 1: at least one dropped data, so
do NOT set ``GGML_CUDA_P2P`` here. Exit 2: could not run (no torch, <2 GPUs).

``GGML_CUDA_P2P=0`` does not disable the flag upstream: llama.cpp tests it for
presence, not value, so it must be unset entirely. Unsloth Studio unsets it for
you and gates the flag on verified NVLink.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

SENTINEL = -7.0
SIZES_MIB = (1, 4, 16, 64)
REPEATS = 3


def _print_topology() -> None:
    """Show `nvidia-smi topo -m`, the cheap read that answers this in advance. NV#
    means NVLink, which does not traverse the PCIe root complex and is not exposed
    to this failure; NODE / PHB / PXB / PIX / SYS all go over PCIe."""
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found; skipping topology.\n")
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output = True, text = True, timeout = 30,
        )
    except Exception as e:  # noqa: BLE001 -- diagnostic, never fatal
        print(f"topology read failed: {e}\n")
        return
    if out.returncode != 0:
        print("topology read returned non-zero; skipping.\n")
        return
    print("=== nvidia-smi topo -m ===")
    # The legend after the matrix is long and not what the reader needs.
    for line in out.stdout.splitlines():
        if line.strip().startswith("Legend"):
            break
        print(line)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument(
        "--sizes-mib", type = int, nargs = "+", default = list(SIZES_MIB),
        help = "copy sizes to test, in MiB",
    )
    parser.add_argument(
        "--repeats", type = int, default = REPEATS,
        help = "copies per pair per size",
    )
    args = parser.parse_args()

    # A probe that tests nothing must not print PASS: its whole job is to license
    # setting GGML_CUDA_P2P, so an empty run is the worst answer it could give.
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if any(size <= 0 for size in args.sizes_mib):
        parser.error("--sizes-mib values must all be positive")

    _print_topology()

    try:
        import torch
    except ImportError:
        print("torch is not installed; cannot probe peer copies.")
        return 2

    if not (torch.cuda.is_available() and torch.cuda.device_count() >= 2):
        print(
            f"need at least 2 visible CUDA GPUs, found "
            f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}. "
            "Peer copies are not used on a single GPU, so this host is unaffected."
        )
        return 2

    count = torch.cuda.device_count()
    print(f"=== {count} visible CUDA GPUs ===")
    for i in range(count):
        print(f"  cuda:{i}  {torch.cuda.get_device_name(i)}")
    print()

    failures = 0
    checked = 0
    print(f"{'pair':>10}  {'size':>8}  {'peer access':>11}  verdict")
    for src in range(count):
        for dst in range(count):
            if src == dst:
                continue
            can = torch.cuda.can_device_access_peer(src, dst)
            for mib in args.sizes_mib:
                elements = mib * 1024 * 1024 // 4
                worst_dropped = 0
                worst_wrong = 0
                error = None
                for _ in range(args.repeats):
                    s = d = None
                    try:
                        s = torch.arange(
                            elements, dtype = torch.float32, device = f"cuda:{src}"
                        )
                        d = torch.full(
                            (elements,), SENTINEL,
                            dtype = torch.float32, device = f"cuda:{dst}",
                        )
                        d.copy_(s)
                        torch.cuda.synchronize(src)
                        torch.cuda.synchronize(dst)
                        # Compare against the source, not just the sentinel: a
                        # scrambled transfer overwrites it with wrong data, which
                        # a sentinel-only test scores as a pass. The sentinel
                        # count stays because "never written" is a different
                        # diagnosis, pointing at a dropped DMA specifically.
                        host_dst = d.cpu()
                        wrong = int((host_dst != s.cpu()).sum())
                        dropped = int((host_dst == SENTINEL).sum())
                        worst_wrong = max(worst_wrong, wrong)
                        worst_dropped = max(worst_dropped, dropped)
                    except Exception as e:  # noqa: BLE001 -- a raising copy is also a failure
                        error = e
                        break
                    finally:
                        del s, d
                        torch.cuda.empty_cache()

                checked += 1
                pair = f"{src}->{dst}"
                access = "yes" if can else "no"
                if error is not None:
                    failures += 1
                    print(f"{pair:>10}  {str(mib) + ' MiB':>8}  {access:>11}  ERROR: {error}")
                elif worst_wrong:
                    failures += 1
                    pct = 100.0 * worst_wrong / elements
                    detail = (
                        f"{worst_dropped} never written"
                        if worst_dropped
                        else "written with the wrong data"
                    )
                    print(
                        f"{pair:>10}  {str(mib) + ' MiB':>8}  {access:>11}  "
                        f"CORRUPT: {worst_wrong}/{elements} elements wrong "
                        f"({pct:.0f}%, {detail})"
                    )
                else:
                    print(f"{pair:>10}  {str(mib) + ' MiB':>8}  {access:>11}  ok")

    print()
    if failures:
        print(
            f"FAIL: {failures} of {checked} peer-copy tests lost data.\n"
            "\n"
            "Peer-to-peer copies are NOT safe on this host. Do not set GGML_CUDA_P2P,\n"
            "and note that GGML_CUDA_P2P=0 does not disable it (llama.cpp tests the\n"
            "variable for presence, not value) -- it must be unset entirely.\n"
            "\n"
            "If CUDA reported success above while dropping data, check the kernel log\n"
            "for IOMMU faults:  dmesg | grep -i -E 'DMAR|AMD-Vi'\n"
            "NVIDIA does not support bare-metal PCIe peer-to-peer with the IOMMU in\n"
            "translating mode; booting with intel_iommu=off / amd_iommu=off (or\n"
            "iommu=pt) is the usual remedy where that is acceptable."
        )
        return 1

    print(
        f"PASS: all {checked} peer-copy tests transferred intact.\n"
        "P2P is safe on this host. Unsloth Studio still gates GGML_CUDA_P2P on\n"
        "verified NVLink; set UNSLOTH_FORCE_DC_P2P=1 to enable it here anyway."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
