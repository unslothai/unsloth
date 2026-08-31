// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which messages own a research reply, answered once per thread revision instead of once per
 * message.
 *
 * A reply can sit on a branch the view is not showing, so this needs the whole repository, not
 * the visible message list. Every user message's action bar asks it, and one export per message
 * made any thread change quadratic in thread length -- paid on every delete and every generated
 * token. One pass per revision removes the per-message factor (not the export itself).
 *
 * Keyed on the revision ALONE, so it assumes ONE question: a second caller asking something else
 * of the same repository would get this answer. Give it its own map, not a second predicate.
 */

export type ExportedReplyItem = {
  parentId: string | null;
  message: { metadata?: unknown };
};

// Keyed on the revision object, not a counter, so a stale entry is unreachable: a current
// revision is the same object, a stale one has been replaced. Weak, so a dead thread's entry goes
// with it.
const ownersByRevision = new WeakMap<object, ReadonlySet<string>>();

/**
 * Ids of the messages that have at least one research reply.
 *
 * @param revision identity that changes whenever the exported repository could have. assistant-ui
 *   rebuilds its message array on every repository mutation (add, delete, branch switch, reset),
 *   so that array is one.
 * @param exportItems reads the repository. Called at most once per revision.
 * @param isResearchReply whether a message's metadata marks it as a research reply.
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
