// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One in-flight mutation per row, tracked by id. The detail panes are keyed by row id, so
// selecting another row unmounts them and a lock held in the pane comes back false; since the
// save is an unconditional PUT it can then land after a DELETE and write the deleted row back.
// The set therefore lives above the key, and these are its two transitions.

export type LockSet = ReadonlySet<string>;

/** Returns the next set, and whether this caller took the lock. */
export function acquire(held: LockSet, id: string): [LockSet, boolean] {
  if (held.has(id)) return [held, false];
  return [new Set(held).add(id), true];
}

// Prompts and prompt lists are separate tables with independent ids, so the same id can name one
// of each and an import can produce that. One set keyed on the raw id would let a prompt's save
// disable the unrelated list's controls.
export function lockKey(kind: "prompt" | "list", id: string): string {
  return `${kind}:${id}`;
}

/** Releasing an id nobody holds returns the same set, so React skips the render. */
export function release(held: LockSet, id: string): LockSet {
  if (!held.has(id)) return held;
  const next = new Set(held);
  next.delete(id);
  return next;
}

// A save is async and the editor stays usable while it runs, so by the time the PUT resolves the
// draft may have moved on. Clearing it unconditionally throws away whatever was typed meanwhile,
// which is the user's most recent intent. Only clear what was actually sent.

export function samePromptDraft(
  a: { name: string; text: string },
  b: { name: string; text: string },
): boolean {
  return a.name === b.name && a.text === b.text;
}

export function sameListDraft(
  a: { name: string; items: readonly string[] },
  b: { name: string; items: readonly string[] },
): boolean {
  return (
    a.name === b.name &&
    a.items.length === b.items.length &&
    a.items.every((item, i) => item === b.items[i])
  );
}
