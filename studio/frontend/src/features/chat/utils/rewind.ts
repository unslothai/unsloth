// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Pure helpers for /rewind: no runtime imports, so the node tests load this file directly. */

export type RewindMode = "both" | "conversation" | "files";

export interface RewindResult {
  has_changes: boolean;
  restored: string[];
  deleted: string[];
  recreated: string[];
  conflicts: string[];
  untracked: string[];
  errors: string[];
  undo_turn_id: string | null;
}

interface BranchMessage {
  readonly id: string;
  readonly role: string;
  readonly content?: unknown;
}

const REWIND_COMMAND = "/rewind";
const PREVIEW_MAX_CHARS = 80;

/** Exactly `/rewind`, ignoring surrounding whitespace and case. */
export function isRewindCommand(text: string): boolean {
  return text.trim().toLowerCase() === REWIND_COMMAND;
}

/** The chosen user message and every later user message on the branch, oldest first. */
export function collectRewindTurnIds(
  branch: readonly BranchMessage[],
  userMessageId: string,
): string[] {
  const start = branch.findIndex((message) => message.id === userMessageId);
  if (start < 0 || branch[start].role !== "user") return [];
  return branch
    .slice(start)
    .filter((message) => message.role === "user")
    .map((message) => message.id);
}

/** Omitted without a user message, so a turn with none sends nothing new. */
export function userMessageIdRequestField(
  userMessageId: string | undefined,
): { user_message_id?: string } {
  return userMessageId ? { user_message_id: userMessageId } : {};
}

function messageText(content: unknown): string {
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((part) =>
      part && typeof part === "object" && (part as { type?: unknown }).type === "text"
        ? String((part as { text?: unknown }).text ?? "")
        : "",
    )
    .join(" ");
}

/** One line per user message on the branch, newest first; an empty preview is attachment-only. */
export function rewindPickerEntries(
  branch: readonly BranchMessage[],
): { id: string; preview: string; text: string }[] {
  return branch
    .filter((message) => message.role === "user")
    .map((message) => {
      const line = messageText(message.content).replace(/\s+/g, " ").trim();
      return {
        id: message.id,
        // Untruncated, for the confirm dialog's line-clamped quote.
        text: line,
        preview:
          line.length > PREVIEW_MAX_CHARS
            ? `${line.slice(0, PREVIEW_MAX_CHARS - 1).trimEnd()}…`
            : line,
      };
    })
    .reverse();
}

export function defaultRewindMode(hasChanges: boolean): RewindMode {
  return hasChanges ? "both" : "conversation";
}

export function rewindTouchesFiles(mode: RewindMode): boolean {
  return mode !== "conversation";
}

export function rewindTouchesConversation(mode: RewindMode): boolean {
  return mode !== "files";
}

export function rewindChangedCount(result: RewindResult): number {
  return (
    result.restored.length + result.deleted.length + result.recreated.length
  );
}

export function rewindSkippedCount(result: RewindResult): number {
  return result.conflicts.length + result.untracked.length;
}
