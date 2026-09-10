# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Probe: do GPU-to-GPU peer copies on this host actually transfer the data?

On bare-metal Linux with a translating IOMMU, a PCIe peer-to-peer copy between two
NVIDIA GPUs can be discarded by the chipset while CUDA reports ``cudaSuccess`` and
``torch.cuda.can_device_access_peer`` returns True, so nothing notices. In
llama.cpp with ``GGML_CUDA_P2P`` set, the model emits ``!!!!!``, a repeated token
or word salad, looking exactly like a broken quant or chat template (#10613).

The destination is sentinel-filled before each copy, so a transfer that moves
nothing is distinguishable from one that legitimately writes zeros, and every
ordered pair is tested at several sizes, since small copies can take a different
path than large ones. Run with no arguments on the host in question::

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
    is NVLink, which never traverses the PCIe root complex; NODE / PHB / PXB / PIX
    / SYS all do."""
    if shutil.which("nvidia-smi") is None:
        print("nvidia-smi not found; skipping topology.\n")
        return
    try:
        out = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output = True,
            text = True,
            timeout = 30,
        )
    except Exception as e:  # noqa: BLE001 -- diagnostic, never fatal
        print(f"topology read failed: {e}\n")
        return
    if out.returncode != 0:
        print("topology read returned non-zero; skipping.\n")
        return
    print("=== nvidia-smi topo -m ===")
    # The legend after the matrix is not what the reader needs.
    for line in out.stdout.splitlines():
        if line.strip().startswith("Legend"):
            break
        print(line)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description = __doc__.splitlines()[0])
    parser.add_argument(
        "--sizes-mib",
        type = int,
        nargs = "+",
        default = list(SIZES_MIB),
        help = "copy sizes to test, in MiB",
    )
    parser.add_argument(
        "--repeats",
        type = int,
        default = REPEATS,
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

    # AMD SDK wheels leave version.hip unset and only encode "rocm" in
    # __version__, which is why the backend's _torch_is_rocm checks both.
    if (
        getattr(torch.version, "hip", None) is not None
        or "rocm" in getattr(torch, "__version__", "").lower()
    ):
        # ROCm reuses the torch.cuda namespace, so every test below would run and
        # PASS on AMD, then recommend GGML_CUDA_P2P, which only the CUDA backend
        # reads. A pass ending in a no-op instruction is worse than no answer.
        print(
            "This is a ROCm/HIP build of torch. GGML_CUDA_P2P is read only by\n"
            "llama.cpp's CUDA backend, so this probe has no advice to give for AMD\n"
            "GPUs and does not test them."
        )
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
    inconclusive = 0
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
                setup_error = None
                for _ in range(args.repeats):
                    s = d = None
                    try:
                        try:
                            s = torch.arange(elements, dtype = torch.float32, device = f"cuda:{src}")
                            d = torch.full(
                                (elements,),
                                SENTINEL,
                                dtype = torch.float32,
                                device = f"cuda:{dst}",
                            )
                        except Exception as alloc_exc:  # noqa: BLE001
                            # No transfer was attempted (usually a resident model
                            # holds the memory), so this is inconclusive, NOT
                            # evidence that peer copies drop data.
                            setup_error = alloc_exc
                            break
                        d.copy_(s)
                        torch.cuda.synchronize(src)
                        torch.cuda.synchronize(dst)
                        # Compare against the source, not just the sentinel: a
                        # scrambled transfer overwrites it and would score as a
                        # pass. The sentinel count stays because "never written"
                        # is a different diagnosis, a dropped DMA.
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
                # Corruption seen on an earlier repeat outranks a later allocation
                # failure: repeats exist to catch INTERMITTENT drops, so downgrading
                # a real mismatch to SKIPPED would hide what they are for.
                if setup_error is not None and not worst_wrong:
                    inconclusive += 1
                    print(
                        f"{pair:>10}  {str(mib) + ' MiB':>8}  {access:>11}  "
                        f"SKIPPED (could not allocate): {setup_error}"
                    )
                elif error is not None:
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

    if inconclusive:
        # Exit 1 is for copies that were tested and lost data. Nothing was tested
        # here, so "unsafe" would push someone off a working optimisation.
        print(
            f"INCONCLUSIVE: {inconclusive} of {checked} tests could not allocate "
            "their buffers,\n"
            f"and {checked - inconclusive} completed intact. No peer copy was "
            "shown to lose data,\n"
            "but the host was not fully tested. Free the GPUs (stop any resident\n"
            "model) and re-run, or lower --sizes-mib."
        )
        return 2

    print(
        f"PASS: all {checked} peer-copy tests transferred intact.\n"
        "P2P is safe on this host. Unsloth Studio still gates GGML_CUDA_P2P on\n"
        "verified NVLink, so to enable it here anyway:\n"
        "\n"
        "  data-center cards (A100, H100, L40S, RTX 6000 Ada, ...):\n"
        "      UNSLOTH_FORCE_DC_P2P=1\n"
        "  anything else, including GeForce:\n"
        "      GGML_CUDA_P2P=1\n"
        "\n"
        "UNSLOTH_FORCE_DC_P2P only reaches the fabric check, which sits behind the\n"
        "data-center gate, so it does nothing on a consumer card. Setting\n"
        "GGML_CUDA_P2P yourself is passed through on every card (any value enables\n"
        "it upstream, so unset it to turn it back off; 0 will NOT)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
