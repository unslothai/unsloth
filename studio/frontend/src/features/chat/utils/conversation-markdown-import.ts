// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { fromMarkdown } from "mdast-util-from-markdown";
import type { ConversationMarkdownMessage } from "./conversation-markdown.ts";

type MarkdownNode = ReturnType<typeof fromMarkdown>["children"][number];

const LABEL_TO_ROLE = new Map([
  ["Assistant", "assistant"],
  ["System", "system"],
  ["User", "user"],
  ["Developer", "developer"],
  ["Message", ""],
  ["Tool", "tool"],
]);

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
  const text = body.replace(/\r\n/g, "\n");
  return messagesFromNodes(text, fromMarkdown(text).children, text.length);
}

export type ParsedMarkdownConversation = {
  title: string;
  messages: ConversationMarkdownMessage[];
};

export function parseConversationMarkdownDocument(
  input: string,
  fallbackTitle: string,
): ParsedMarkdownConversation[] {
  const text = input.replace(/\r\n/g, "\n").trimStart();
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
