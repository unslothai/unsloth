// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Relative with .ts: the node tests import this file and cannot resolve @/.
import { accountDatabaseName } from "../../../lib/account-transition.ts";

export type Pending = { scale: boolean; browser: boolean };

const pendingKey = () => accountDatabaseName("unsloth_client_prefs_pending");

export function readPending(): Pending {
  try {
    const saved = JSON.parse(localStorage.getItem(pendingKey()) ?? "{}");
    return { scale: saved?.scale === true, browser: saved?.browser === true };
  } catch {
    return { scale: false, browser: false };
  }
}

export function writePending(pending: Pending): void {
  try {
    if (pending.scale || pending.browser) {
      localStorage.setItem(pendingKey(), JSON.stringify(pending));
    } else {
      localStorage.removeItem(pendingKey());
    }
  } catch {
  }
}
