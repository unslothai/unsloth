// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface IgpuCarveoutAdvice {
  current_gb: number;
  needed_gb: number;
  suggested_gb: number;
  /** Visible RAM plus the current allocation. */
  machine_gb: number;
  host_left_gb: number;
  message: string;
}

/** A partial payload from an older backend is treated as no advice. */
export function parseCarveoutAdvice(value: unknown): IgpuCarveoutAdvice | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Record<string, unknown>;
  const num = (key: string): number | null => {
    const n = raw[key];
    return typeof n === "number" && Number.isFinite(n) ? n : null;
  };
  const current_gb = num("current_gb");
  const needed_gb = num("needed_gb");
  const suggested_gb = num("suggested_gb");
  const machine_gb = num("machine_gb");
  const host_left_gb = num("host_left_gb");
  const message = raw.message;
  if (
    current_gb === null ||
    needed_gb === null ||
    suggested_gb === null ||
    machine_gb === null ||
    host_left_gb === null ||
    typeof message !== "string" ||
    !message.trim()
  ) {
    return null;
  }
  return { current_gb, needed_gb, suggested_gb, machine_gb, host_left_gb, message };
}
