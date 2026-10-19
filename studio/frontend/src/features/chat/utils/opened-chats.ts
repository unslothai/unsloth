// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** How many recently opened chats keep their sidebar rows loaded. */
export const OPENED_CHATS_LIMIT = 20;

/** Most recently opened first. */
export function touchOpenedChat(opened: readonly string[], id: string): string[] {
  return [id, ...opened.filter((other) => other !== id)];
}

/** Evicts the least recently opened chats past `limit`, skipping running ones and the open one: a
 *  running chat is never evicted, so the list passes `limit` when more than `limit` are running. */
export function keepOpenedChats(
  opened: readonly string[],
  running: ReadonlySet<string>,
  limit = OPENED_CHATS_LIMIT,
): string[] {
  let over = opened.length - limit;
  if (over <= 0) return [...opened];
  const kept: string[] = [];
  for (let index = opened.length - 1; index >= 0; index--) {
    const id = opened[index];
    if (over > 0 && index > 0 && !running.has(id)) {
      over -= 1;
      continue;
    }
    kept.push(id);
  }
  return kept.reverse();
}
