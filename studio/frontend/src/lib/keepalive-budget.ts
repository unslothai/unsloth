// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Fetch caps total in-flight keepalive bodies at 64 KiB; larger bodies go without keepalive.
const KEEPALIVE_BODY_BUDGET_BYTES = 60 * 1024;

export function isUnderKeepaliveBudget(body: string): boolean {
  // JSON is at least one byte per UTF-16 unit, so short bodies skip encoding.
  if (body.length <= KEEPALIVE_BODY_BUDGET_BYTES / 3) return true;
  return new TextEncoder().encode(body).byteLength <= KEEPALIVE_BODY_BUDGET_BYTES;
}

let reservedBytes = 0;

/** Claims room for one keepalive body among those still in flight; null when it would not fit. */
export function reserveKeepaliveBody(body: string): (() => void) | null {
  const bytes = new TextEncoder().encode(body).byteLength;
  if (reservedBytes + bytes > KEEPALIVE_BODY_BUDGET_BYTES) return null;
  reservedBytes += bytes;
  let released = false;
  return () => {
    if (released) return;
    released = true;
    reservedBytes -= bytes;
  };
}
