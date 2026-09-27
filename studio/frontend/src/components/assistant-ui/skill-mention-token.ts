// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MentionToken = { start: number; end: number; query: string };

// The @name token around the caret; accepting replaces all of it, not just up to the caret.
export function mentionTokenAt(text: string, caret: number): MentionToken | null {
  const prefix = text.slice(0, caret);
  const match = /(?:^|\s)@([a-z0-9-]*)$/i.exec(prefix);
  if (!match) return null;
  const tail = /^[a-z0-9-]*/i.exec(text.slice(caret))?.[0] ?? "";
  return { start: prefix.lastIndexOf("@"), end: caret + tail.length, query: match[1] ?? "" };
}

export function replaceMentionToken(
  text: string,
  token: MentionToken,
  directive: string,
): { text: string; caret: number } {
  const after = text.slice(token.end);
  return {
    text: `${text.slice(0, token.start)}${directive} ${after.startsWith(" ") ? after.slice(1) : after}`,
    caret: token.start + directive.length + 1,
  };
}
