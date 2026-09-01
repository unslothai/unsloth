// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  batchListChatMessages,
  listChatImportLedger,
  updateChatThread,
} from "../api/chat-api";
import type { MessageRecord, ThreadRecord } from "../types";
import {
  planLegacyTitleRepairs,
  selectLegacyRepairPage,
  threadsAwaitingImport,
  threadsMissingMessages,
} from "./chat-title";
import { type GuardProbe, readGuardProbe } from "./openapi-support";
import { runWithConcurrency } from "./run-with-concurrency";
import { createSerialQueue } from "./serial-queue";

/** Threads already tried, so ones that could not be rewritten stay off every later refresh. */
const attempted = new Set<string>();

/** Rows per pass, so a long history drains in pages. */
const REPAIR_PER_PASS = 100;

/** Writes in flight. Each PATCH is a synchronous SQLite call server side. */
const REPAIR_CONCURRENCY = 4;

/** Breather between pages. */
const REPAIR_PAGE_PAUSE_MS = 500;

/** Several sidebars can be mounted at once, so REPAIR_CONCURRENCY only holds if their passes
 *  queue rather than overlap. */
const serial = createSerialQueue();

/** Cached once the served schema answers. A failed probe is not cached, so a startup hiccup does
 *  not park the migration for the session. */
let guardSupport: Promise<boolean> | null = null;

/** Whether this backend enforces the conditional title patch. Probing by sending one would let
 *  an older backend apply the write, the very harm being checked for, so the served schema
 *  answers; anything unreadable is a no. */
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

/** Rewrite titles stored pre-cut at 48 chars so they grow with the sidebar. */
export function repairLegacyChatTitles(
  threads: ThreadRecord[],
): Promise<number> {
  return serial(() => runRepairPass(threads));
}

async function runRepairPass(threads: ThreadRecord[]): Promise<number> {
  // Claim nothing until the guard is known: a rewrite that can silently beat a rename is not worth a tidier title.
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
    // Not the caller's earlier map: the backend's own view, taken as late as possible, is what the
    // rewrite has to be based on. One batched call.
    messages = await batchListChatMessages(ids);
  } catch {
    // Nothing was decided, so let a later refresh try these again.
    for (const id of ids) attempted.delete(id);
    return 0;
  }

  // Backend messages only: Dexie keeps rows the backend has pruned, so merging the two could put
  // a deleted prompt back into a title. A chat not imported yet reads as unknown below.
  const repairs = planLegacyTitleRepairs(candidates, messages);

  // Nothing stored means either mid-import or emptied by the user; the ledger tells them apart,
  // and is only fetched when there is something to decide.
  const withoutMessages = threadsMissingMessages(ids, messages);
  if (withoutMessages.length > 0) {
    let imported = new Set<string>();
    try {
      imported = await listChatImportLedger();
    } catch {
      // Undecided, so keep every one of them retryable.
    }
    for (const id of threadsAwaitingImport(ids, messages, imported)) {
      attempted.delete(id);
    }
  }

  let repaired = 0;
  await runWithConcurrency(repairs, REPAIR_CONCURRENCY, async (repair) => {
    try {
      // The backend PATCH directly, not updateStoredChatThread: that ensures the thread first,
      // re-importing one deleted on another client, and a migration must never create anything.
      // Both guards answer 409 if a rename or a delete of the opening prompt lands first, and a
      // title patch leaves updatedAt alone, so Recents keeps its order.
      await updateChatThread(
        repair.threadId,
        { title: repair.title },
        {
          expectedTitle: repair.previousTitle,
          expectedOpeningMessageId: repair.openingMessageId,
        },
      );
      repaired += 1;
    } catch {
      attempted.delete(repair.threadId);
    }
  });

  // A page that wrote nothing fires no history update, so schedule the next one here. It runs on
  // `rest`, so rows this page unmarked are not drawn again.
  if (hasMore) {
    setTimeout(() => {
      void repairLegacyChatTitles(rest).catch(() => undefined);
    }, REPAIR_PAGE_PAUSE_MS);
  }
  return repaired;
}
