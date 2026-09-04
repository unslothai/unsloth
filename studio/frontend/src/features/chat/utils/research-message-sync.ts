// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord } from "../types";

/** Mirrors studio_db._RESEARCH_LINK_KEYS: what marks a message as owned by a research run. */
export const RESEARCH_METADATA_KEYS = [
  "researchRunId",
  "researchRun",
  "researchStatus",
  "researchPlanRevision",
  "serverManaged",
] as const;

export function hasResearchMetadata(metadata: unknown): boolean {
  if (!metadata || typeof metadata !== "object") return false;
  const keys = metadata as Record<string, unknown>;
  return RESEARCH_METADATA_KEYS.some((key) => key in keys);
}

/** Replace the client's copy of every server-managed research message with the stored one. The
 *  runtime keeps client-only fields on a research turn (the live `researchRun`, a
 *  `serverRevision`, streamed content parts), so mirroring the repository verbatim asks the
 *  backend to edit messages it owns. It answers 409 and drops the whole payload, which cost the
 *  thread its autosave. A faithful copy is the no-op the guard accepts. The research prompt
 *  carries no metadata of its own, so it is identified as the parent of the report. */
export function reconcileServerManagedMessages(
  records: MessageRecord[],
  stored: MessageRecord[],
): MessageRecord[] {
  const storedById = new Map(stored.map((message) => [message.id, message]));
  const serverManaged = new Set<string>();
  for (const message of stored) {
    if (!hasResearchMetadata(message.metadata)) continue;
    serverManaged.add(message.id);
    if (message.parentId) serverManaged.add(message.parentId);
  }
  if (serverManaged.size === 0) return records;
  return records.map((record) => {
    if (!serverManaged.has(record.id)) return record;
    const stored = storedById.get(record.id);
    // parentId stays the client's: deleting the message a research prompt hung off relinks it, and
    // echoing the stored parent would persist a link to a row the same sync then prunes.
    return stored ? { ...stored, parentId: record.parentId } : record;
  });
}
