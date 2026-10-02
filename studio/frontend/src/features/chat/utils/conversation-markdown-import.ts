// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fromMarkdown } from "mdast-util-from-markdown";
import {
  CONVERSATION_MARKDOWN_FRAME_PREFIX,
  type ConversationMarkdownMessage,
} from "./conversation-markdown.ts";

type MarkdownNode = ReturnType<typeof fromMarkdown>["children"][number];

const LABEL_TO_ROLE = new Map([
  ["Assistant", "assistant"],
  ["System", "system"],
  ["User", "user"],
  ["Developer", "developer"],
  ["Message", ""],
  ["Tool", "tool"],
]);

const FRAME_END = " -->\n\n";
const CHAT_SEPARATOR = "\n---\n\n";
const TITLE_LINE = /^# ([^\n]*)\n\n/;
const ROLE_LINE = /^## ([^\n]+)\n\n/;

function parseFramedDocument(
  text: string,
  fallbackTitle: string,
): ParsedMarkdownConversation[] | undefined {
  const conversations: ParsedMarkdownConversation[] = [];
  let offset = 0;
  while (offset < text.length) {
    const title = TITLE_LINE.exec(text.slice(offset));
    const frameStart = offset + (title?.[0].length ?? 0);
    if (!text.startsWith(CONVERSATION_MARKDOWN_FRAME_PREFIX, frameStart)) {
      if (offset === 0) return;
      throw new Error("Missing Studio Markdown conversation framing.");
    }
    const metadataStart =
      frameStart + CONVERSATION_MARKDOWN_FRAME_PREFIX.length;
    const metadataEnd = text.indexOf(FRAME_END, metadataStart);
    let lengths: unknown;
    try {
      lengths = JSON.parse(text.slice(metadataStart, metadataEnd));
    } catch {
      throw new Error("Invalid Studio Markdown conversation framing.");
    }
    if (
      metadataEnd < 0 ||
      !Array.isArray(lengths) ||
      lengths.length === 0 ||
      !lengths.every((length) => Number.isSafeInteger(length) && length > 0)
    ) {
      throw new Error("Invalid Studio Markdown message lengths.");
    }
    offset = metadataEnd + FRAME_END.length;
    const messages: ConversationMarkdownMessage[] = [];
    for (const [index, length] of lengths.entries()) {
      const section = text.slice(offset, offset + length);
      const heading = ROLE_LINE.exec(section);
      if (section.length !== length || !heading) {
        throw new Error("Incomplete Studio Markdown message.");
      }
      const label = heading[1];
      messages.push({
        role:
          LABEL_TO_ROLE.get(label) ??
          `${label[0].toLowerCase()}${label.slice(1)}`,
        content: section.slice(heading[0].length),
      });
      offset += length;
      const separator = index === lengths.length - 1 ? "\n" : "\n\n";
      if (!text.startsWith(separator, offset)) {
        throw new Error("Invalid Studio Markdown message boundary.");
      }
      offset += separator.length;
    }
    conversations.push({ title: title?.[1] ?? fallbackTitle, messages });
    if (!text.slice(offset).trim()) break;
    if (!text.startsWith(CHAT_SEPARATOR, offset)) {
      throw new Error("Invalid Studio Markdown conversation boundary.");
    }
    offset += CHAT_SEPARATOR.length;
    if (offset === text.length) {
      throw new Error("Missing Studio Markdown conversation framing.");
    }
  }
  return conversations;
}

function roleHeading(node: MarkdownNode | undefined, text: string) {
  if (node?.type !== "heading" || node.depth !== 2) return;
  const start = node.position?.start.offset;
  const end = node.position?.end.offset;
  if (
    start === undefined ||
    end === undefined ||
    !text.startsWith("## ", start) ||
    (start > 0 && text.slice(start - 2, start) !== "\n\n") ||
    text.slice(end, end + 2) !== "\n\n"
  )
    return;
  const role = LABEL_TO_ROLE.get(text.slice(start + 3, end));
  return role === undefined
    ? undefined
    : { role, start, contentStart: end + 2 };
}

function documentTitle(nodes: MarkdownNode[], index: number, text: string) {
  const node = nodes[index];
  if (node?.type !== "heading" || node.depth !== 1) return;
  const start = node.position?.start.offset;
  const end = node.position?.end.offset;
  const firstMessage = roleHeading(nodes[index + 1], text);
  if (
    start === undefined ||
    end === undefined ||
    !text.startsWith("# ", start) ||
    !firstMessage ||
    text.slice(end, firstMessage.start).trim()
  )
    return;
  return { title: text.slice(start + 2, end).trim(), start };
}

function messagesFromNodes(
  text: string,
  nodes: MarkdownNode[],
  end: number,
): ConversationMarkdownMessage[] {
  if (!roleHeading(nodes[0], text)) return [];
  // only root headings delimit turns; headings in code, quotes and lists remain content.
  const headings = nodes.flatMap((node) => {
    const heading = roleHeading(node, text);
    return heading ? [heading] : [];
  });
  return headings.flatMap(({ role, contentStart }, index) => {
    const next = headings[index + 1];
    const contentEnd = next ? next.start - 2 : end;
    const content = text.slice(contentStart, contentEnd);
    const value = next ? content : content.replace(/\n$/, "");
    return value.trim() ? [{ role, content: value }] : [];
  });
}

export function parseConversationMarkdownMessages(
  body: string,
): ConversationMarkdownMessage[] {
  const text = body.replace(/\r\n?/g, "\n");
  const framed = parseFramedDocument(text, "");
  return framed
    ? framed.flatMap((conversation) => conversation.messages)
    : messagesFromNodes(text, fromMarkdown(text).children, text.length);
}

export type ParsedMarkdownConversation = {
  title: string;
  messages: ConversationMarkdownMessage[];
};

export function parseConversationMarkdownDocument(
  input: string,
  fallbackTitle: string,
): ParsedMarkdownConversation[] {
  const text = input.replace(/\r\n?/g, "\n").trimStart();
  const framed = parseFramedDocument(text, fallbackTitle);
  if (framed) return framed;
  const nodes = fromMarkdown(text).children;
  const firstTitle = documentTitle(nodes, 0, text);
  let title = firstTitle?.title ?? fallbackTitle;
  let firstNode = firstTitle === undefined ? 0 : 1;
  const results: ParsedMarkdownConversation[] = [];
  const append = (endNode: number, end: number) => {
    const messages = messagesFromNodes(
      text,
      nodes.slice(firstNode, endNode),
      end,
    );
    if (messages.length > 0) results.push({ title, messages });
  };

  for (let index = firstNode; index < nodes.length; index++) {
    const node = nodes[index];
    if (firstTitle === undefined || node.type !== "thematicBreak") continue;
    const nextTitle = documentTitle(nodes, index + 1, text);
    const start = node.position?.start.offset;
    const end = node.position?.end.offset;
    // bulk exports frame each chat with a rule, title and first role heading.
    if (
      start === undefined ||
      end === undefined ||
      nextTitle === undefined ||
      text.slice(start, end) !== "---" ||
      text.slice(end, nextTitle.start).trim()
    )
      continue;
    append(index, start - 1);
    title = nextTitle.title;
    firstNode = index + 2;
    index++;
  }
  append(nodes.length, text.length);
  return results;
}
