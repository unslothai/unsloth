// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ThreadRecord } from "../types";

const claimed = new Set<string>();

/** Once per untitled fork per page; forks open as "regular", so assistant-ui never names them. */
export function claimUntitledForkTitle(
  record:
    | Pick<ThreadRecord, "id" | "title" | "forkedFromThreadId">
    | null
    | undefined,
): boolean {
  if (
    !record?.forkedFromThreadId ||
    record.title !== "New Chat" ||
    claimed.has(record.id)
  ) {
    return false;
  }
  claimed.add(record.id);
  return true;
}
