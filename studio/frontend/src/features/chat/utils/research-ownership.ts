// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which messages own a research reply, derived once per thread revision.
 *
 * Asking that per message meant exporting the whole thread inside every message's
 * render body, so one render pass over N messages cost N exports of N messages.
 */

/** The research run id a message carries, or null. */
export const getResearchRunId = (metadata: unknown): string | null => {
  const custom = (
    metadata as
      | {
          custom?: {
            researchRunId?: unknown;
            researchRun?: { id?: unknown };
          };
        }
      | undefined
  )?.custom;
  const runId = custom?.researchRunId ?? custom?.researchRun?.id;
  return typeof runId === "string" ? runId : null;
};

type ExportedItem = { parentId: string | null; message: { metadata?: unknown } };

// Keyed by the thread's message list, which assistant-ui replaces on every change, so
// the entry lives exactly as long as the revision it was derived from.
const byRevision = new WeakMap<object, ReadonlySet<string>>();

const EMPTY: ReadonlySet<string> = new Set();

/**
 * Ids of the messages that have a research reply, for one revision of a thread.
 * `revision` is the thread's message list; `exportItems` runs at most once per revision.
 */
export function researchOwnerIds(
  revision: object,
  exportItems: () => readonly ExportedItem[],
): ReadonlySet<string> {
  const cached = byRevision.get(revision);
  if (cached) return cached;
  const owners = new Set<string>();
  for (const { parentId, message } of exportItems()) {
    if (parentId && getResearchRunId(message.metadata)) owners.add(parentId);
  }
  const result: ReadonlySet<string> = owners.size === 0 ? EMPTY : owners;
  byRevision.set(revision, result);
  return result;
}
