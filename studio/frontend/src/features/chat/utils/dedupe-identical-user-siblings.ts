// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord } from "../types";

type UserTurnFingerprint = string;

function stableJson(value: unknown): string {
  return JSON.stringify(value, (_, nested) => {
    if (nested && typeof nested === "object" && !Array.isArray(nested)) {
      return Object.keys(nested as Record<string, unknown>)
        .sort()
        .reduce<Record<string, unknown>>((acc, key) => {
          acc[key] = (nested as Record<string, unknown>)[key];
          return acc;
        }, {});
    }
    return nested;
  });
}

function attachmentsFingerprint(
  attachments: MessageRecord["attachments"] | undefined,
): string {
  if (!Array.isArray(attachments) || attachments.length === 0) {
    return "[]";
  }
  const normalized = attachments.map((attachment) => {
    if (!attachment || typeof attachment !== "object") {
      return attachment;
    }
    const { id: _id, status: _status, ...rest } = attachment as Record<
      string,
      unknown
    >;
    return rest;
  });
  return stableJson(normalized);
}

function userTurnFingerprint(record: MessageRecord): UserTurnFingerprint {
  return stableJson([
    record.parentId ?? null,
    record.content,
    attachmentsFingerprint(record.attachments),
  ]);
}

function pickCanonicalUser(records: MessageRecord[]): MessageRecord {
  return [...records].sort((left, right) => {
    const createdAtDelta = left.createdAt - right.createdAt;
    if (createdAtDelta !== 0) {
      return createdAtDelta;
    }
    return left.id.localeCompare(right.id);
  })[0];
}

/**
 * Collapse accidental duplicate user siblings that share the same parent,
 * content, and attachments. Intentional branches that differ in text or files
 * are left alone. Returns remapped records and ids dropped from the payload.
 */
export function dedupeIdenticalUserSiblings(records: MessageRecord[]): {
  records: MessageRecord[];
  collapsedIds: string[];
} {
  const usersByFingerprint = new Map<UserTurnFingerprint, MessageRecord[]>();
  for (const record of records) {
    if (record.role !== "user") {
      continue;
    }
    const fingerprint = userTurnFingerprint(record);
    const group = usersByFingerprint.get(fingerprint) ?? [];
    group.push(record);
    usersByFingerprint.set(fingerprint, group);
  }

  const collapsedIds = new Set<string>();
  const idRemap = new Map<string, string>();

  for (const group of usersByFingerprint.values()) {
    if (group.length <= 1) {
      continue;
    }
    const canonical = pickCanonicalUser(group);
    for (const duplicate of group) {
      if (duplicate.id === canonical.id) {
        continue;
      }
      collapsedIds.add(duplicate.id);
      idRemap.set(duplicate.id, canonical.id);
    }
  }

  if (collapsedIds.size === 0) {
    return { records, collapsedIds: [] };
  }

  const remapParentId = (parentId: string | null | undefined): string | null => {
    let current = parentId ?? null;
    const seen = new Set<string>();
    while (current && idRemap.has(current) && !seen.has(current)) {
      seen.add(current);
      current = idRemap.get(current) ?? current;
    }
    return current;
  };

  const dedupedRecords = records
    .filter((record) => !collapsedIds.has(record.id))
    .map((record) => {
      const parentId = remapParentId(record.parentId);
      if (parentId === (record.parentId ?? null)) {
        return record;
      }
      return { ...record, parentId };
    });

  return { records: dedupedRecords, collapsedIds: [...collapsedIds].sort() };
}
