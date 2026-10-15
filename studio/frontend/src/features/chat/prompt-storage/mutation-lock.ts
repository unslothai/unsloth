// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Locks live above the row-keyed panes, or a save PUT can land after a DELETE and resurrect it.

export type LockSet = ReadonlySet<string>;

export function acquire(held: LockSet, id: string): [LockSet, boolean] {
  if (held.has(id)) return [held, false];
  return [new Set(held).add(id), true];
}

// Prompts and lists have independent ids, so keys are namespaced by kind.
export function lockKey(kind: "prompt" | "list", id: string): string {
  return `${kind}:${id}`;
}

export function release(held: LockSet, id: string): LockSet {
  if (!held.has(id)) return held;
  const next = new Set(held);
  next.delete(id);
  return next;
}

// Only clear the draft if it still equals what was sent.

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
