// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isLlamaRuntimeReason } from "./backend-preflight-message.ts";

export const RUNTIME_REPAIR_KEY = "unsloth-llama-runtime-repair";
const RECURRENCE_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;

export function recordRuntimeRepair(
  reason: string | null,
  now = Date.now(),
): void {
  if (!isLlamaRuntimeReason(reason)) return;
  try {
    localStorage.setItem(
      RUNTIME_REPAIR_KEY,
      JSON.stringify({ reason, repairedAt: now }),
    );
  } catch {
    // A failed write must not fail the repair.
  }
}

export function wasRuntimeRepairedRecently(
  reason: string | null,
  now = Date.now(),
): boolean {
  if (!isLlamaRuntimeReason(reason)) return false;
  try {
    const raw = localStorage.getItem(RUNTIME_REPAIR_KEY);
    if (!raw) return false;
    const record: unknown = JSON.parse(raw);
    if (!record || typeof record !== "object") return false;
    const { reason: previousReason, repairedAt } = record as Record<
      string,
      unknown
    >;
    return (
      typeof previousReason === "string" &&
      isLlamaRuntimeReason(previousReason) &&
      typeof repairedAt === "number" &&
      Number.isFinite(repairedAt) &&
      now >= repairedAt &&
      now - repairedAt <= RECURRENCE_WINDOW_MS
    );
  } catch {
    return false;
  }
}
