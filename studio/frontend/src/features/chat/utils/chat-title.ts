// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord, ThreadRecord } from "../types";

/** The sidebar clips the title with CSS, so store the whole first line and a
 *  wider sidebar shows more of it. Caps a pasted wall of text, and matches the
 *  rename input's maxLength, which counts UTF-16 units and includes any
 *  ellipsis this adds. */
export const FALLBACK_TITLE_MAX = 120;

/** Older titles were stored pre-cut at 48 chars with a literal "...", so no
 *  width could reveal more. Kept to find and rewrite those rows. */
export const LEGACY_FALLBACK_TITLE_MAX = 48;
const LEGACY_FALLBACK_SUFFIX = "...";

function firstLineOf(text: string): string {
  const firstLine = (text || "").split(/\r?\n/, 1)[0] ?? "";
  return firstLine.replace(/\s+/g, " ").trim();
}

/** Cut to at most `maxUnits` UTF-16 units, the unit maxLength counts, without
 *  splitting an astral character in half. A lone surrogate parses fine and then
 *  fails the backend's SQLite bind. */
function cutToUnits(text: string, maxUnits: number): string {
  let out = "";
  for (const character of text) {
    if (out.length + character.length > maxUnits) break;
    out += character;
  }
  return out;
}

export function fallbackTitleFromUserText(userText: string): string {
  const cleaned = firstLineOf(userText);
  if (!cleaned) return "New Chat";
  if (cleaned.length <= FALLBACK_TITLE_MAX) return cleaned;
  // The ellipsis takes one of the budget, so the whole title still fits what
  // the rename input accepts.
  return cutToUnits(cleaned, FALLBACK_TITLE_MAX - 1).trimEnd() + "…";
}

/** Pre-filter on the title alone: only these are worth fetching messages for. */
export function couldBeLegacyClippedTitle(title: string | undefined): boolean {
  return (
    typeof title === "string" &&
    title.endsWith(LEGACY_FALLBACK_SUFFIX) &&
    title.length === LEGACY_FALLBACK_TITLE_MAX + LEGACY_FALLBACK_SUFFIX.length
  );
}

/** True when `title` is exactly the old 48-character cut of `userText`. */
export function isLegacyClippedTitle(
  title: string | undefined,
  userText: string,
): boolean {
  if (!couldBeLegacyClippedTitle(title)) return false;
  const kept = (title as string).slice(0, LEGACY_FALLBACK_TITLE_MAX);
  const cleaned = firstLineOf(userText);
  return (
    cleaned.length > LEGACY_FALLBACK_TITLE_MAX &&
    cleaned.slice(0, LEGACY_FALLBACK_TITLE_MAX) === kept
  );
}

function textOf(message: MessageRecord | undefined): string {
  if (!message) return "";
  const content = message.content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter(
      (part): part is Extract<typeof part, { type: "text" }> =>
        part.type === "text",
    )
    .map((part) => part.text)
    .join("")
    .trim();
}

export interface LegacyTitleRepair {
  threadId: string;
  /** The clipped title the rewrite is based on, guarding the write. */
  previousTitle: string;
  /** The message the new title came from, guarding it too: deleting it must
   *  not leave its text expanded into the title. */
  openingMessageId: string;
  title: string;
}

export interface LegacyRepairPage {
  candidates: ThreadRecord[];
  /** Everything this page did not take. The next page reads from here, so a
   *  row this page failed on cannot be picked up again by the same drain. */
  rest: ThreadRecord[];
  hasMore: boolean;
}

/** One page of rows to look at, skipping the ones already tried. */
export function selectLegacyRepairPage(
  threads: ThreadRecord[],
  attempted: ReadonlySet<string>,
  limit: number,
): LegacyRepairPage {
  const pending = threads.filter(
    (thread) =>
      couldBeLegacyClippedTitle(thread.title) && !attempted.has(thread.id),
  );
  const candidates = pending.slice(0, limit);
  const taken = new Set(candidates.map((thread) => thread.id));
  return {
    candidates,
    rest: threads.filter((thread) => !taken.has(thread.id)),
    hasMore: pending.length > limit,
  };
}

/** Threads the backend holds no messages for. */
export function threadsMissingMessages(
  ids: readonly string[],
  messagesByThreadId: ReadonlyMap<string, MessageRecord[]>,
): string[] {
  return ids.filter((id) => (messagesByThreadId.get(id) ?? []).length === 0);
}

/** Of those, the ones still worth retrying: no record of their import
 *  finishing, so their messages may yet land. One the ledger knows is simply
 *  empty, every message deleted, which no later pass can change. Retrying
 *  those would re-read them on every refresh for the session, since the title
 *  stays clipped and keeps matching the pre-filter. */
export function threadsAwaitingImport(
  ids: readonly string[],
  messagesByThreadId: ReadonlyMap<string, MessageRecord[]>,
  importedThreadIds: ReadonlySet<string>,
): string[] {
  return threadsMissingMessages(ids, messagesByThreadId).filter(
    (id) => !importedThreadIds.has(id),
  );
}

/** Rows to rewrite. A title must be the exact old cut of its own first
 *  message, so a rename ending in "..." is left alone. */
export function planLegacyTitleRepairs(
  threads: ThreadRecord[],
  messagesByThreadId: Map<string, MessageRecord[]>,
): LegacyTitleRepair[] {
  const repairs: LegacyTitleRepair[] = [];
  for (const thread of threads) {
    const messages = messagesByThreadId.get(thread.id) ?? [];
    // Earliest, not first in the array: a local read is in index order.
    // Ties break on id, the order the backend reads it in.
    const opening = messages
      .filter((m) => m.role === "user")
      .reduce<MessageRecord | undefined>((earliest, m) => {
        if (earliest === undefined) return m;
        if (m.createdAt !== earliest.createdAt) {
          return m.createdAt < earliest.createdAt ? m : earliest;
        }
        return m.id < earliest.id ? m : earliest;
      }, undefined);
    const userText = textOf(opening);
    if (opening === undefined) continue;
    if (!isLegacyClippedTitle(thread.title, userText)) continue;
    const title = fallbackTitleFromUserText(userText);
    if (title === thread.title) continue;
    repairs.push({
      threadId: thread.id,
      previousTitle: thread.title,
      openingMessageId: opening.id,
      title,
    });
  }
  return repairs;
}
