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

/** Threads already tried, so unrewritable ones are not retried on every refresh. */
const attempted = new Set<string>();

const REPAIR_PER_PASS = 100;

/** Each PATCH is a synchronous SQLite call server side. */
const REPAIR_CONCURRENCY = 4;

const REPAIR_PAGE_PAUSE_MS = 500;

/** Several sidebars may mount at once; passes must queue to honour REPAIR_CONCURRENCY. */
const serial = createSerialQueue();

/** A failed probe is not cached, so a startup hiccup does not park the migration. */
let guardSupport: Promise<boolean> | null = null;

/** Read from the served schema: probing with a real patch could let an old backend apply it. */
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
  // Do nothing until the guard is known: an unguarded rewrite could beat a rename.
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
    // Use the backend's own messages, fetched as late as possible, as the rewrite base.
    messages = await batchListChatMessages(ids);
  } catch {
    // Nothing was decided, so let a later refresh try these again.
    for (const id of ids) attempted.delete(id);
    return 0;
  }

  // Backend messages only: Dexie may hold pruned rows that would resurrect a deleted prompt.
  const repairs = planLegacyTitleRepairs(candidates, messages);

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
      // Direct PATCH, not updateStoredChatThread, which would re-import a thread deleted elsewhere.
      // Guards answer 409 on a racing rename or delete; updatedAt is untouched.
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

  // A page that wrote nothing fires no history update, so schedule the next one here.
  if (hasMore) {
    setTimeout(() => {
      void repairLegacyChatTitles(rest).catch(() => undefined);
    }, REPAIR_PAGE_PAUSE_MS);
  }
  return repaired;
}
