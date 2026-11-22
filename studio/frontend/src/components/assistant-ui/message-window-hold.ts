// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Scoped per-message holds shared by the body and row windowing layers.

type HoldListener = (viewport: HTMLElement, messageId: string) => void;

const holds = new WeakMap<HTMLElement, Map<string, number>>();
const listeners = new Set<HoldListener>();
const EMPTY_IDS: ReadonlySet<string> = new Set();

function notify(viewport: HTMLElement, messageId: string): void {
  for (const listener of [...listeners]) {
    listener(viewport, messageId);
  }
}

/** Keeps one message mounted until the returned, idempotent release runs. */
export function holdMessage(
  viewport: HTMLElement,
  messageId: string,
): () => void {
  if (messageId.length === 0) {
    return () => undefined;
  }
  let byMessage = holds.get(viewport);
  if (byMessage === undefined) {
    byMessage = new Map();
    holds.set(viewport, byMessage);
  }
  byMessage.set(messageId, (byMessage.get(messageId) ?? 0) + 1);
  notify(viewport, messageId);

  let released = false;
  return () => {
    // Cleanup can run twice (cancel -> highlight clear -> finish). That must
    // not decrement another holder's claim.
    if (released) {
      return;
    }
    released = true;
    const current = holds.get(viewport);
    if (current === undefined) {
      return;
    }
    const count = current.get(messageId) ?? 0;
    if (count <= 1) {
      current.delete(messageId);
    } else {
      current.set(messageId, count - 1);
    }
    if (current.size === 0) {
      holds.delete(viewport);
    }
    notify(viewport, messageId);
  };
}

export function isMessageHeld(
  viewport: HTMLElement,
  messageId: string,
): boolean {
  return (holds.get(viewport)?.get(messageId) ?? 0) > 0;
}

/** Empty when nothing is held, so callers avoid allocating per update. */
export function heldMessageIds(viewport: HTMLElement): ReadonlySet<string> {
  const byMessage = holds.get(viewport);
  return byMessage === undefined || byMessage.size === 0
    ? EMPTY_IDS
    : new Set(byMessage.keys());
}

export function subscribeMessageHolds(listener: HoldListener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
