// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { batchListChatMessages } from "../api/chat-api";
import type { MessageRecord, ThreadRecord } from "../types";
import {
  listLegacyChatMessages,
  updateStoredChatThread,
} from "./chat-history-storage";
import {
  mergeMessagesById,
  planLegacyTitleRepairs,
  selectLegacyRepairPage,
  threadsMissingMessages,
  threadsWithoutRepairs,
} from "./chat-title";
import { runWithConcurrency } from "./run-with-concurrency";

/** Threads already tried. A repaired title stops matching anyway; this keeps
 *  the ones that could not be rewritten off every later refresh. */
const attempted = new Set<string>();

/** Rows per pass, so a long history drains in pages instead of putting its
 *  whole backlog on the backend at once. */
const REPAIR_PER_PASS = 100;

/** Writes in flight. Each PATCH is a synchronous SQLite call server side. */
const REPAIR_CONCURRENCY = 4;

/** Breather between pages. */
const REPAIR_PAGE_PAUSE_MS = 500;

/** Rewrite titles stored pre-cut at 48 chars so they grow with the sidebar
 *  again. `known` is the caller's own message map, when it already has one. */
export async function repairLegacyChatTitles(
  threads: ThreadRecord[],
  known?: Map<string, MessageRecord[]>,
): Promise<number> {
  const { candidates, rest, hasMore } = selectLegacyRepairPage(
    threads,
    attempted,
    REPAIR_PER_PASS,
  );
  if (candidates.length === 0) return 0;
  const ids = candidates.map((thread) => thread.id);
  for (const id of ids) attempted.add(id);

  let messages: Map<string, MessageRecord[]>;
  let repairs: ReturnType<typeof planLegacyTitleRepairs>;
  try {
    // One batched call, and none at all when the caller already fetched them.
    messages = known
      ? new Map(known)
      : await batchListChatMessages(ids);
    repairs = planLegacyTitleRepairs(candidates, messages);

    // A row nothing could be made of may just be missing its opening message
    // here: a legacy chat can sit in Dexie entirely, or hold its first turn
    // there while later ones have already been imported. Look locally before
    // writing any of them off.
    const unexplained = threadsWithoutRepairs(candidates, repairs);
    if (unexplained.length > 0) {
      await runWithConcurrency(unexplained, REPAIR_CONCURRENCY, async (id) => {
        const local = await listLegacyChatMessages(id);
        if (local.length === 0) return;
        messages.set(id, mergeMessagesById(messages.get(id) ?? [], local));
      });
      repairs = planLegacyTitleRepairs(candidates, messages);
    }
  } catch {
    // Nothing was decided, so let a later refresh try these again.
    for (const id of ids) attempted.delete(id);
    return 0;
  }

  // Nothing known at all reads as an incomplete answer, not an empty chat: a
  // clipped title means there was an opening message once. Leave those rows for
  // a later refresh rather than writing them off for the session.
  for (const id of threadsMissingMessages(ids, messages)) attempted.delete(id);
  let repaired = 0;
  await runWithConcurrency(repairs, REPAIR_CONCURRENCY, async (repair) => {
    try {
      // expectedTitle guards the write: a rename that landed since this list
      // was read answers 409 and keeps the user's title. A title patch leaves
      // updatedAt alone, so Recents keeps its order.
      await updateStoredChatThread(
        repair.threadId,
        { title: repair.title },
        { expectedTitle: repair.previousTitle },
      );
      repaired += 1;
    } catch {
      attempted.delete(repair.threadId);
    }
  });

  // A page that wrote nothing fires no history update, so the next page has to
  // be scheduled here rather than left waiting on an unrelated refresh. It runs
  // on `rest`, so rows this page unmarked cannot be drawn again by this drain.
  if (hasMore) {
    setTimeout(() => {
      void repairLegacyChatTitles(rest, known).catch(() => undefined);
    }, REPAIR_PAGE_PAUSE_MS);
  }
  return repaired;
}
