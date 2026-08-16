// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which messages own a research reply, answered once per thread revision instead of once per
 * message.
 *
 * "Does this prompt own a research reply?" is a question about the whole repository: a reply can
 * sit on a branch the current view is not showing, so it cannot be read off the visible message
 * list. The action bar of every user message asks it, and answering it by exporting the
 * repository per message made the cost of any thread change quadratic in thread length. That is
 * paid on every delete and on every token of a generation, not once per thread.
 *
 * The answer changes only when the repository does, so one pass over one export serves every
 * message at that revision. What is removed is the per-message factor, not the export: a
 * generation still pays one export per token rather than one per user message per token.
 *
 * The cache is keyed on the revision ALONE, so it assumes one question. A second caller asking a
 * different question of the same repository would be handed this one's answer; give it its own
 * map rather than a second predicate here.
 */

export type ExportedReplyItem = {
  parentId: string | null;
  message: { metadata?: unknown };
};

// Keyed on the caller's revision object rather than on a counter, so a stale entry is not
// reachable: a revision that is still current is the same object, and one that is not has been
// replaced. Weak, so a thread that goes away takes its entry with it.
const ownersByRevision = new WeakMap<object, ReadonlySet<string>>();

/**
 * Ids of the messages that have at least one research reply.
 *
 * @param revision object identity that changes whenever the exported repository could have
 *   changed. assistant-ui rebuilds its message array on every repository mutation (add, delete,
 *   branch switch, reset all dirty the cached list), so that array is such an identity.
 * @param exportItems reads the repository. Called at most once per revision, and not at all when
 *   the answer for this revision is already known.
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
