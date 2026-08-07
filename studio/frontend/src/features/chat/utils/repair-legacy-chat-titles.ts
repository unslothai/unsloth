// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { batchListChatMessages } from "../api/chat-api";
import type { ThreadRecord } from "../types";
import {
  listStoredChatThreads,
  updateStoredChatThread,
} from "./chat-history-storage";
import {
  couldBeLegacyClippedTitle,
  planLegacyTitleRepairs,
} from "./chat-title";
import { runWithConcurrency } from "./run-with-concurrency";

/** Threads already tried. A repaired title stops matching anyway; this keeps
 *  the ones that could not be rewritten off every later refresh. */
const attempted = new Set<string>();

/** Rows per pass, so a long history drains over a few refreshes instead of
 *  putting its whole backlog on the backend at once. */
const REPAIR_PER_PASS = 100;

/** Writes in flight. Each PATCH is a synchronous SQLite call server side. */
const REPAIR_CONCURRENCY = 4;

/** Rewrite titles stored pre-cut at 48 chars so they grow with the sidebar
 *  again. Takes whichever threads the caller just loaded. */
export async function repairLegacyChatTitles(
  threads: ThreadRecord[],
): Promise<number> {
  const ids = threads
    .filter(
      (thread) =>
        couldBeLegacyClippedTitle(thread.title) && !attempted.has(thread.id),
    )
    .map((thread) => thread.id)
    .slice(0, REPAIR_PER_PASS);
  if (ids.length === 0) return 0;
  for (const id of ids) attempted.add(id);

  let repairs: ReturnType<typeof planLegacyTitleRepairs>;
  try {
    // One batched call, not one per row.
    const messagesByThreadId = await batchListChatMessages(ids);
    // Read the titles last. The caller's list is a snapshot by now, and a
    // rename that landed since has to win over the rewrite.
    const current = await listStoredChatThreads({ includeArchived: true });
    const byId = new Map(current.map((thread) => [thread.id, thread]));
    repairs = planLegacyTitleRepairs(
      ids
        .map((id) => byId.get(id))
        .filter((thread): thread is ThreadRecord => thread !== undefined),
      messagesByThreadId,
    );
  } catch {
    // Nothing was decided, so let a later refresh try these again.
    for (const id of ids) attempted.delete(id);
    return 0;
  }

  let repaired = 0;
  // A title patch leaves updatedAt alone, so Recents keeps its order.
  await runWithConcurrency(repairs, REPAIR_CONCURRENCY, async (repair) => {
    try {
      await updateStoredChatThread(repair.threadId, { title: repair.title });
      repaired += 1;
    } catch {
      attempted.delete(repair.threadId);
    }
  });
  return repaired;
}
