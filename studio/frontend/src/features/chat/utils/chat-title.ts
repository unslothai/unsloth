// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { MessageRecord, ThreadRecord } from "../types";

/** The sidebar clips the title with CSS, so store the whole first line and a
 *  wider sidebar shows more of it. Caps a pasted wall of text, at the rename
 *  input's maxLength. */
export const FALLBACK_TITLE_MAX = 120;

/** Older titles were stored pre-cut at 48 chars with a literal "...", so no
 *  width could reveal more. Kept to find and rewrite those rows. */
export const LEGACY_FALLBACK_TITLE_MAX = 48;
const LEGACY_FALLBACK_SUFFIX = "...";

function firstLineOf(text: string): string {
  const firstLine = (text || "").split(/\r?\n/, 1)[0] ?? "";
  return firstLine.replace(/\s+/g, " ").trim();
}

export function fallbackTitleFromUserText(userText: string): string {
  const cleaned = firstLineOf(userText);
  if (!cleaned) return "New Chat";
  // Cut on code points: a UTF-16 cut can halve an emoji, and the lone
  // surrogate that leaves fails the backend's SQLite bind.
  const points = Array.from(cleaned);
  if (points.length <= FALLBACK_TITLE_MAX) return cleaned;
  return points.slice(0, FALLBACK_TITLE_MAX).join("").trimEnd() + "…";
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
  title: string;
}

export interface LegacyRepairPage {
  candidates: ThreadRecord[];
  /** Rows left for the next page, which the caller has to schedule itself. */
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
  return {
    candidates: pending.slice(0, limit),
    hasMore: pending.length > limit,
  };
}

/** Threads the backend has nothing for. A chat still only in Dexie reads empty
 *  there, so it needs a local read before it can be written off. */
export function threadsMissingMessages(
  ids: readonly string[],
  messagesByThreadId: ReadonlyMap<string, MessageRecord[]>,
): string[] {
  return ids.filter((id) => (messagesByThreadId.get(id) ?? []).length === 0);
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
    const userText = textOf(messages.find((m) => m.role === "user"));
    if (!isLegacyClippedTitle(thread.title, userText)) continue;
    const title = fallbackTitleFromUserText(userText);
    if (title === thread.title) continue;
    repairs.push({
      threadId: thread.id,
      previousTitle: thread.title,
      title,
    });
  }
  return repairs;
}
