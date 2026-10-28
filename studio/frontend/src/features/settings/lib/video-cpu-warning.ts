// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Developer mode CPU video: what the confirm dialog says about this host. Pure, so it is unit tested. */

const GIB = 1024 ** 3;

export type VideoCpuRamCheck = {
  /** "below" warns; "unknown" (no reading) says nothing either way. */
  status: "below" | "ok" | "unknown";
  hostGib: number | null;
  minGib: number;
};

/** Host RAM against the backend's minimum, both rounded to whole GiB for display. */
export function videoCpuRamCheck(
  hostRamBytes: number | null | undefined,
  minRamBytes: number,
): VideoCpuRamCheck {
  const minGib = Math.round(minRamBytes / GIB);
  if (
    typeof hostRamBytes !== "number" ||
    !Number.isFinite(hostRamBytes) ||
    hostRamBytes <= 0
  ) {
    return { status: "unknown", hostGib: null, minGib };
  }
  const hostGib = Math.round(hostRamBytes / GIB);
  // Compared in bytes with 5% slack: a 32 GB machine reports ~31.3 GiB after firmware and kernel
  // reservations, and must not be told it is below a 32 GiB minimum.
  const status = hostRamBytes >= minRamBytes * 0.95 ? "ok" : "below";
  return { status, hostGib, minGib };
}

/** Only turning the opt-in ON asks first; turning it off restores the GPU gate at once. */
export function videoCpuToggleNeedsConfirm(next: boolean): boolean {
  return next;
}
