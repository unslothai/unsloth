// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which messages own a research reply, computed once per thread revision: per-message exports
 * made thread changes quadratic. Keyed on the revision alone, so it answers ONE question only.
 */

export type ExportedReplyItem = {
  parentId: string | null;
  message: { metadata?: unknown };
};

// Keyed on the revision object, so a stale entry is unreachable; weak, so dead threads are freed.
const ownersByRevision = new WeakMap<object, ReadonlySet<string>>();

/**
 * @param revision identity that changes whenever the exported repository could have.
 * @param exportItems reads the repository; called at most once per revision.
 */
export function researchReplyOwners(
  revision: object,
  exportItems: () => readonly ExportedReplyItem[],
  isResearchReply: (metadata: unknown) => boolean,
): ReadonlySet<string> {
  const known = ownersByRevision.get(revision);
  if (known) {
    return known;
  }
  const owners = new Set<string>();
  for (const { parentId, message } of exportItems()) {
    if (parentId !== null && isResearchReply(message.metadata)) {
      owners.add(parentId);
    }
  }
  ownersByRevision.set(revision, owners);
  return owners;
}
