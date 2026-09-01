// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord, ThreadRecord } from "../types";

/** Store the whole first line and let the sidebar clip it with CSS, so a wider one shows more.
 *  Matches the rename input's maxLength: UTF-16 units, ellipsis included. */
export const FALLBACK_TITLE_MAX = 120;

/** Older titles were stored pre-cut at 48 chars with a literal "...". Kept to find and rewrite those rows. */
export const LEGACY_FALLBACK_TITLE_MAX = 48;
const LEGACY_FALLBACK_SUFFIX = "...";

/** Drop unpaired surrogates: they render as nothing, and one reaching the backend fails its
 *  SQLite bind and 500s the title write. Iteration yields a valid pair whole, so a length-1
 *  unit in the range is a lone surrogate. */
function dropLoneSurrogates(text: string): string {
  let out = "";
  for (const character of text) {
    const code = character.codePointAt(0) ?? 0;
    if (character.length === 1 && code >= 0xd800 && code <= 0xdfff) continue;
    out += character;
  }
  return out;
}

function firstLineOf(text: string): string {
  const firstLine = (text || "").split(/\r?\n/, 1)[0] ?? "";
  // Drop surrogates first, or removing one can leave a double or trailing space.
  return dropLoneSurrogates(firstLine).replace(/\s+/g, " ").trim();
}

/** Cut to at most `maxUnits` UTF-16 units without splitting an astral character: a lone surrogate
 *  parses fine and then fails the backend's SQLite bind. */
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
  // The ellipsis takes one of the budget, so the title still fits the input.
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
  /** The message the title came from, guarded too: deleting it must not leave its text expanded into the title. */
  openingMessageId: string;
  title: string;
}

export interface LegacyRepairPage {
  candidates: ThreadRecord[];
  /** What this page skipped. The next page reads it, so a row this page failed on is not redrawn by the same drain. */
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

/** Of those, the ones still worth retrying: no record of their import finishing, so their
 *  messages may yet land. One the ledger knows is simply empty, and retrying it would re-read it
 *  on every refresh for the session, since its title stays clipped. */
export function threadsAwaitingImport(
  ids: readonly string[],
  messagesByThreadId: ReadonlyMap<string, MessageRecord[]>,
  importedThreadIds: ReadonlySet<string>,
): string[] {
  return threadsMissingMessages(ids, messagesByThreadId).filter(
    (id) => !importedThreadIds.has(id),
  );
}

/** Rows to rewrite. A title must be the exact old cut of its own first message, so a rename
 *  ending in "..." is left alone. */
export function planLegacyTitleRepairs(
  threads: ThreadRecord[],
  messagesByThreadId: Map<string, MessageRecord[]>,
): LegacyTitleRepair[] {
  const repairs: LegacyTitleRepair[] = [];
  for (const thread of threads) {
    const messages = messagesByThreadId.get(thread.id) ?? [];
    // Earliest, not first in the array; ties break on id, as the backend does.
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
