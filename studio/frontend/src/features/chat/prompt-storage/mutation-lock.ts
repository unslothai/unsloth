// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One in-flight mutation per row, tracked by id.
//
// The detail panes are keyed by row id, so selecting another row unmounts them.
// A lock held in the pane comes back false on return, and since the save is an
// unconditional PUT it can then land after a DELETE and write the deleted row
// straight back. The set therefore lives above the key, and these are its two
// transitions, kept here so they can be tested without a renderer.

export type LockSet = ReadonlySet<string>;

/** Returns the next set, and whether this caller took the lock. */
export function acquire(held: LockSet, id: string): [LockSet, boolean] {
  if (held.has(id)) return [held, false];
  return [new Set(held).add(id), true];
}

/** Releasing an id nobody holds returns the same set, so React skips the render. */
export function release(held: LockSet, id: string): LockSet {
  if (!held.has(id)) return held;
  const next = new Set(held);
  next.delete(id);
  return next;
}
