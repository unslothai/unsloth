// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { batchListChatMessages, listChatThreads } from "../api/chat-api";
import type { MessageRecord, ThreadRecord } from "../types";
import { updateStoredChatThread } from "./chat-history-storage";
import {
  planLegacyTitleRepairs,
  repairsStillValid,
  selectLegacyRepairPage,
  threadsMissingMessages,
} from "./chat-title";
import { type GuardProbe, readGuardProbe } from "./openapi-support";
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

/** Cached once the served schema answers. A failed probe is not cached, so a
 *  hiccup at startup does not park the migration for the session. */
let guardSupport: Promise<boolean> | null = null;

/** Whether this backend enforces the conditional title patch.
 *
 *  Probing by sending one is not an option: an older backend would apply the
 *  write, which is the harm being checked for. The served schema says so
 *  without touching anything. Anything unreadable reads as unsupported. */
function backendEnforcesTitleGuard(): Promise<boolean> {
  guardSupport ??= (async () => {
    let probe: GuardProbe = { supported: false, settled: false };
    try {
      const response = await authFetch("/openapi.json");
      probe = readGuardProbe(
        response.ok,
        response.ok ? await response.json() : null,
      );
    } catch {
      probe = { supported: false, settled: false };
    }
    // Only a settled answer is worth remembering.
    if (!probe.settled) guardSupport = null;
    return probe.supported;
  })();
  return guardSupport;
}

/** Rewrite titles stored pre-cut at 48 chars so they grow with the sidebar
 *  again. */
export function repairLegacyChatTitles(
  threads: ThreadRecord[],
): Promise<number> {
  return serial(() => runRepairPass(threads));
}

async function runRepairPass(threads: ThreadRecord[]): Promise<number> {
  // Nothing is claimed or marked until the guard is known to be enforced: a
  // rewrite that can silently beat a rename is not worth a tidier title.
  if (!(await backendEnforcesTitleGuard())) return 0;

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
    // Read here, not from a map the caller fetched earlier: the backend's own
    // view, taken as late as possible, is what the rewrite has to be based on.
    // One batched call for the page.
    messages = await batchListChatMessages(ids);
  } catch {
    // Nothing was decided, so let a later refresh try these again.
    for (const id of ids) attempted.delete(id);
    return 0;
  }

  // Backend messages only. Dexie keeps rows the backend has pruned, since
  // deleting a message never clears them, so anything that merges the two could
  // put a deleted prompt back into a title. A chat whose messages have not been
  // imported yet reads as unknown below and is retried once they land.
  const repairs = planLegacyTitleRepairs(candidates, messages);

  // Nothing known at all reads as an incomplete answer, not an empty chat: a
  // clipped title means there was an opening message once. Leave those rows for
  // a later refresh rather than writing them off for the session.
  for (const id of threadsMissingMessages(ids, messages)) attempted.delete(id);

  // expectedTitle is only enforced by a backend that knows the field. The
  // desktop app ships its own frontend and can meet an older one, which would
  // apply the write regardless, so confirm the titles here too. One call for
  // the page, taken last, so a rename is respected either way.
  let live: ReturnType<typeof repairsStillValid>;
  try {
    const current = await listChatThreads({ includeArchived: true });
    live = repairsStillValid(
      repairs,
      new Map(current.map((thread) => [thread.id, thread.title])),
    );
  } catch {
    for (const id of ids) attempted.delete(id);
    return 0;
  }

  let repaired = 0;
  await runWithConcurrency(live, REPAIR_CONCURRENCY, async (repair) => {
    try {
      // expectedTitle makes this atomic where the backend enforces it: a
      // rename landing now answers 409 and keeps the user's title. A title
      // patch leaves updatedAt alone, so Recents keeps its order.
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
      void repairLegacyChatTitles(rest).catch(() => undefined);
    }, REPAIR_PAGE_PAUSE_MS);
  }
  return repaired;
}
