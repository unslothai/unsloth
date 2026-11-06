// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Relative with .ts: the node tests import this file and cannot resolve @/.
import { accountDatabaseName } from "../../../lib/account-transition.ts";

export const PENDING_SECTIONS = [
  "scale",
  "browser",
  "chat",
  "plusMenu",
  "material",
  "timeFormat",
  "spellcheck",
] as const;

export type PendingSection = (typeof PENDING_SECTIONS)[number];
export type Pending = Record<PendingSection, boolean>;

const pendingKey = () => accountDatabaseName("unsloth_client_prefs_pending");

export function nothingPending(): Pending {
  return Object.fromEntries(
    PENDING_SECTIONS.map((section) => [section, false]),
  ) as Pending;
}

export function anyPending(pending: Pending): boolean {
  return PENDING_SECTIONS.some((section) => pending[section]);
}

export function readPending(): Pending {
  const pending = nothingPending();
  try {
    const saved = JSON.parse(localStorage.getItem(pendingKey()) ?? "{}");
    for (const section of PENDING_SECTIONS) {
      pending[section] = saved?.[section] === true;
    }
  } catch {
    // A corrupt marker reads as nothing pending.
  }
  return pending;
}

export function writePending(pending: Pending): void {
  try {
    if (anyPending(pending)) {
      localStorage.setItem(pendingKey(), JSON.stringify(pending));
    } else {
      localStorage.removeItem(pendingKey());
    }
  } catch {
  }
}
