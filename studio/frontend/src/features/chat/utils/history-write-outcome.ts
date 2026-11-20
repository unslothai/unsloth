// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord } from "../types";

/**
 * `saveStoredChatMessage` returns its input by reference for incognito,
 * rejected-payload and protected-message early returns. Those messages still
 * reach the run, but no new row was written.
 */
export type HistoryWriteOutcome = "persisted" | "delivered";

export function historyWriteOutcome(
  record: MessageRecord,
  saved: MessageRecord,
): HistoryWriteOutcome {
  return saved === record ? "delivered" : "persisted";
}
