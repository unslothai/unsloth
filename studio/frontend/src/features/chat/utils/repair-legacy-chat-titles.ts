// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { batchListChatMessages } from "../api/chat-api";
import type { MessageRecord, ThreadRecord } from "../types";
import { updateStoredChatThread } from "./chat-history-storage";
import {
  planLegacyTitleRepairs,
  selectLegacyRepairPage,
  threadsMissingMessages,
} from "./chat-title";
import { runWithConcurrency } from "./run-with-concurrency";
import { createSerialQueue } from "./serial-queue";

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

/** Several sidebars can be mounted at once, so REPAIR_CONCURRENCY only holds
 *  if their passes queue rather than run alongside each other. */
const serial = createSerialQueue();

/** Rewrite titles stored pre-cut at 48 chars so they grow with the sidebar
 *  again. `known` is the caller's own message map, when it already has one. */
export function repairLegacyChatTitles(
  threads: ThreadRecord[],
  known?: Map<string, MessageRecord[]>,
): Promise<number> {
  return serial(() => runRepairPass(threads, known));
}

async function runRepairPass(
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
  try {
    // One batched call, and none at all when the caller already fetched them.
    messages = known ? known : await batchListChatMessages(ids);
  } catch {
    // Nothing was decided, so let a later refresh try these again.
    for (const id of ids) attempted.delete(id);
    return 0;
  }

  // Stored messages are the only source here. Dexie holds rows the backend has
  // pruned, since deleting a message never clears them, so reading it could put
  // a deleted prompt back into a title. A chat whose messages have not been
  // imported yet reads as unknown below and is retried once they land.
  const repairs = planLegacyTitleRepairs(candidates, messages);

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
