// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ConversationMarkdownMessage } from "./conversation-markdown.ts";

/** Inverse of roleLabel() in conversation-markdown.ts for exports this app wrote. */
const KNOWN_LABEL_TO_ROLE: Readonly<Record<string, string>> = {
  Assistant: "assistant",
  System: "system",
  User: "user",
  Message: "",
  Tool: "tool",
};

export function markdownLabelToRole(label: string): string {
  const trimmed = label.trim();
  const known = KNOWN_LABEL_TO_ROLE[trimmed];
  if (known !== undefined) return known;
  if (trimmed.length === 0) return "";
  return `${trimmed[0]?.toLowerCase() ?? ""}${trimmed.slice(1)}`;
}

const MULTI_CHAT_SEPARATOR = "\n---\n\n";

function splitMarkdownChats(text: string): string[] {
  const normalized = text.replace(/\r\n/g, "\n").trim();
  if (!normalized) return [];
  return normalized.split(MULTI_CHAT_SEPARATOR).map((chunk) => chunk.trim()).filter(Boolean);
}

function stripLeadingDocumentTitle(chunk: string): {
  title?: string;
  body: string;
} {
  const normalized = chunk.replace(/\r\n/g, "\n").trimStart();
  if (!normalized.startsWith("# ")) {
    return { body: chunk };
  }
  const lineEnd = normalized.indexOf("\n");
  if (lineEnd === -1) {
    return { body: chunk };
  }
  const title = normalized.slice(2, lineEnd).trim();
  let rest = normalized.slice(lineEnd + 1);
  if (rest.startsWith("\n")) rest = rest.slice(1);
  if (!rest.startsWith("## ")) {
    return { body: chunk };
  }
  return { title, body: rest };
}

/** Reads ## Role sections produced by buildConversationMarkdown(). */
export function parseConversationMarkdownMessages(
  body: string,
): ConversationMarkdownMessage[] {
  const normalized = body.replace(/\r\n/g, "\n").trimEnd();
  if (!normalized.startsWith("## ")) {
    return [];
  }

  const messages: ConversationMarkdownMessage[] = [];
  let pos = 0;
  while (pos < normalized.length) {
    if (!normalized.startsWith("## ", pos)) break;
    const labelEnd = normalized.indexOf("\n", pos + 3);
    if (labelEnd === -1) break;
    const label = normalized.slice(pos + 3, labelEnd);
    let contentStart = labelEnd + 1;
    if (normalized[contentStart] !== "\n") break;
    contentStart += 1;

    const nextHeading = normalized.indexOf("\n\n## ", contentStart);
    const rawContent =
      nextHeading === -1
        ? normalized.slice(contentStart)
        : normalized.slice(contentStart, nextHeading);
    const content = rawContent.replace(/\n$/, "");
    if (content.trim()) {
      messages.push({
        role: markdownLabelToRole(label),
        content,
      });
    }
    pos = nextHeading === -1 ? normalized.length : nextHeading + 2;
  }
  return messages;
}

export type ParsedMarkdownConversation = {
  title: string;
  messages: ConversationMarkdownMessage[];
};

/** One or more chats from a markdown export (single thread or bulk combined file). */
export function parseConversationMarkdownDocument(
  text: string,
  fallbackTitle: string,
): ParsedMarkdownConversation[] {
  const chats = splitMarkdownChats(text);
  const results: ParsedMarkdownConversation[] = [];
  for (const [index, chunk] of chats.entries()) {
    const { title: docTitle, body } = stripLeadingDocumentTitle(chunk);
    const messages = parseConversationMarkdownMessages(body);
    if (messages.length === 0) continue;
    const title =
      docTitle ??
      (chats.length > 1 ? `${fallbackTitle} ${index + 1}` : fallbackTitle);
    results.push({ title, messages });
  }
  return results;
}
