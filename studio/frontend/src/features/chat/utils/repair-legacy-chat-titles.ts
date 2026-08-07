// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { batchListChatMessages } from "../api/chat-api";
import type { ThreadRecord } from "../types";
import { updateStoredChatThread } from "./chat-history-storage";
import {
  couldBeLegacyClippedTitle,
  planLegacyTitleRepairs,
} from "./chat-title";

/** Threads already tried. A repaired title stops matching anyway; this keeps
 *  the ones that could not be rewritten off every later refresh. */
const attempted = new Set<string>();

/** Rewrite titles stored pre-cut at 48 chars so they grow with the sidebar
 *  again. Takes whichever threads the caller just loaded. */
export async function repairLegacyChatTitles(
  threads: ThreadRecord[],
): Promise<number> {
  const candidates = threads.filter(
    (thread) =>
      couldBeLegacyClippedTitle(thread.title) && !attempted.has(thread.id),
  );
  if (candidates.length === 0) return 0;
  for (const thread of candidates) attempted.add(thread.id);

  // One batched call, not one per row.
  const messagesByThreadId = await batchListChatMessages(
    candidates.map((thread) => thread.id),
  ).catch(() => null);
  if (!messagesByThreadId) return 0;

  const repairs = planLegacyTitleRepairs(candidates, messagesByThreadId);
  // A title patch leaves updatedAt alone, so Recents keeps its order.
  const results = await Promise.all(
    repairs.map((repair) =>
      updateStoredChatThread(repair.threadId, { title: repair.title })
        .then(() => true)
        .catch(() => false),
    ),
  );
  return results.filter(Boolean).length;
}
