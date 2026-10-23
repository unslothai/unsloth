// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { FlowDocument } from "../../types";

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

export interface ChatEntry {
  /** The Chat Output block the player invokes for each message. */
  output: string;
  welcome: string;
}

/** Turns sent back to the flow with each message; the server keeps 50 at most. */
export const HISTORY_SENT = 40;
const SAVE_VERSION = 1;

/** A flow plays as a chat when it has an enabled Chat Output block. */
export function chatEntry(
  doc: FlowDocument | null | undefined,
): ChatEntry | null {
  if (!doc) return null;
  const output = doc.nodes.find((n) => n.type === "chat.output" && !n.disabled);
  if (!output) return null;
  const input = doc.nodes.find((n) => n.type === "chat.input" && !n.disabled);
  const welcome = input?.params?.welcome;
  return {
    output: output.id,
    welcome:
      typeof welcome === "string" && welcome.trim()
        ? welcome.trim()
        : "Hi! How can I help?",
  };
}

/** What the flow's Chat Input receives: the new message and the conversation before it. */
export function turnParams(
  turns: ChatTurn[],
  message: string,
): { message: string; history: ChatTurn[] } {
  return { message, history: turns.slice(-HISTORY_SENT) };
}

/** Turns kept in a save: far more than the model sees, and well under the session size limit. */
export const SAVED_TURNS = 200;
const SAVED_TURN_CHARS = 20000;

export function serializeChat(turns: ChatTurn[]): unknown {
  return {
    v: SAVE_VERSION,
    turns: turns.slice(-SAVED_TURNS).map((t) => ({
      role: t.role,
      content: t.content.slice(0, SAVED_TURN_CHARS),
    })),
  };
}

/** A saved conversation, or an empty one when the save is from another version or malformed. */
export function restoreChat(saved: unknown): ChatTurn[] {
  if (!saved || typeof saved !== "object") return [];
  const s = saved as { v?: unknown; turns?: unknown };
  if (s.v !== SAVE_VERSION || !Array.isArray(s.turns)) return [];
  return s.turns.filter(
    (t): t is ChatTurn =>
      !!t &&
      typeof t === "object" &&
      ((t as ChatTurn).role === "user" ||
        (t as ChatTurn).role === "assistant") &&
      typeof (t as ChatTurn).content === "string",
  );
}
