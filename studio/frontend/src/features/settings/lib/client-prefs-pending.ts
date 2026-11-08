// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Relative with .ts: the node tests import this file and cannot resolve @/.
import { accountDatabaseName } from "../../../lib/account-transition.ts";

export const PENDING_SECTIONS = [
  "scale",
  "browser",
  "chat",
  "plusMenu",
  "timeFormat",
  "spellcheck",
] as const;

export type PendingSection = (typeof PENDING_SECTIONS)[number];
export type Pending = Record<PendingSection, boolean>;

const pendingKey = () => accountDatabaseName("unsloth_client_prefs_pending");

export function readPending(): Pending {
  let saved: Record<string, unknown> | null = null;
  try {
    saved = JSON.parse(localStorage.getItem(pendingKey()) ?? "{}");
  } catch {
    // nothing pending
  }
  return Object.fromEntries(
    PENDING_SECTIONS.map((section) => [section, saved?.[section] === true]),
  ) as Pending;
}

export function writePending(pending: Pending): void {
  try {
    if (Object.values(pending).some(Boolean)) {
      localStorage.setItem(pendingKey(), JSON.stringify(pending));
    } else {
      localStorage.removeItem(pendingKey());
    }
  } catch {
    // storage unavailable
  }
}

// Re-read first: other tabs of this account share the marker.
export function markPending(sections: readonly PendingSection[], value: boolean): void {
  const pending = readPending();
  for (const section of sections) pending[section] = value;
  writePending(pending);
}
