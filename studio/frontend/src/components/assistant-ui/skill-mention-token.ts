// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type MentionToken = { start: number; end: number; query: string };

// The @name under the caret. It starts at an @ at the start of the text or after whitespace, the
// query is what sits between the @ and the caret, and the token runs on past the caret when the
// caret was moved back into it. Accepting a skill replaces the whole token, so a caret in the
// middle of "@calculator" cannot leave "tor" behind.
export function mentionTokenAt(text: string, caret: number): MentionToken | null {
  const prefix = text.slice(0, caret);
  const match = /(?:^|\s)@([a-z0-9-]*)$/i.exec(prefix);
  if (!match) return null;
  const tail = /^[a-z0-9-]*/i.exec(text.slice(caret))?.[0] ?? "";
  return { start: prefix.lastIndexOf("@"), end: caret + tail.length, query: match[1] ?? "" };
}

// The text once `directive` stands in for the token, and where the caret lands: after the
// directive and the one space that separates it from what follows.
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
