// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Fetch caps total in-flight keepalive bodies at 64 KiB; larger bodies go without keepalive.
const KEEPALIVE_BODY_BUDGET_BYTES = 60 * 1024;

export function isUnderKeepaliveBudget(body: string, copies = 1): boolean {
  const budget = KEEPALIVE_BODY_BUDGET_BYTES / copies;
  // JSON is at least one byte per UTF-16 unit, so short bodies skip encoding.
  if (body.length <= budget / 3) return true;
  return new TextEncoder().encode(body).byteLength <= budget;
}
