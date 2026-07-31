// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type ConversationMarkdownMessage = {
  readonly role: string;
  readonly content: string;
};

export const CONVERSATION_MARKDOWN_FORMAT = "markdown";
export const CONVERSATION_MARKDOWN_LABEL = "Markdown";
export const CONVERSATION_MARKDOWN_EXTENSION = "md";
export const CONVERSATION_MARKDOWN_MIME_TYPE = "text/markdown";

const ROLE_LABELS: Readonly<Record<string, string>> = {
  assistant: "Assistant",
  system: "System",
  user: "User",
};

function roleLabel(role: string): string {
  const knownLabel = ROLE_LABELS[role];
  if (knownLabel) {
    return knownLabel;
  }
  return role.length > 0
    ? `${role[0]?.toUpperCase()}${role.slice(1)}`
    : "Message";
}

// A message split into renderable pieces. The training exports flatten
// everything to plain text, which is fine for JSONL/CSV but leaves tool calls
// as a one-line JSON blob and thinking as literal [thinking] markers. Markdown
// is a document format, so it renders each piece on its own terms.
export type ConversationMarkdownBlock =
  | { readonly kind: "text"; readonly text: string }
  | { readonly kind: "thinking"; readonly text: string }
  | {
      readonly kind: "tool-call";
      readonly name: string;
      readonly args?: unknown;
      readonly result?: unknown;
    }
  | { readonly kind: "attachment"; readonly label: string }
  | { readonly kind: "source"; readonly title: string; readonly url: string };

const GENERATED_IMAGE_PLACEHOLDER = "[generated image omitted]";
const GENERATED_IMAGE_BYTES_KEY = "image_b64";
const LINE_BREAK_PATTERN = /[\r\n]/;

// Long enough to not be closed early by backticks inside the body.
function fence(body: string, language = ""): string {
  const longestRun = [...body.matchAll(/`+/g)].reduce(
    (max, [run]) => Math.max(max, run.length),
    0,
  );
  const ticks = "`".repeat(Math.max(3, longestRun + 1));
  return `${ticks}${language}\n${body}\n${ticks}`;
}

function inlineCode(value: string): string {
  const longestRun = [...value.matchAll(/`+/g)].reduce(
    (max, [run]) => Math.max(max, run.length),
    0,
  );
  const ticks = "`".repeat(longestRun + 1);
  // Padding keeps a leading or trailing backtick from closing the span.
  const pad = value.startsWith("`") || value.endsWith("`") ? " " : "";
  return `${ticks}${pad}${value}${pad}${ticks}`;
}

function escapeMarkdownLabel(value: string): string {
  return value.replace(/([\\[\]])/g, "\\$1");
}

function safeSourceUrl(raw: string): string {
  const value = raw.trim();
  if (!value || LINE_BREAK_PATTERN.test(value)) {
    return "";
  }
  if (value.startsWith("#")) {
    return encodeURI(value).replaceAll("<", "%3C").replaceAll(">", "%3E");
  }
  try {
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:"
      ? parsed.href.replaceAll("<", "%3C").replaceAll(">", "%3E")
      : "";
  } catch {
    return "";
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// A multi-line string is almost always source: html, a patch, a script. Fence
// it as-is so it reads as code instead of a JSON string full of \n. No language
// tag, since guessing one wrong is worse than none.
function renderValue(label: string, value: unknown): string[] {
  const escapedLabel = escapeMarkdownLabel(label);
  if (typeof value === "string") {
    if (value.includes("\n")) return [`**${escapedLabel}:**`, fence(value)];
    // Prose stays prose; anything that could be read as markup or a fence
    // becomes an inline code span so it renders as itself.
    const inert = /[<`]/.test(value) ? inlineCode(value) : value;
    return [`**${escapedLabel}:** ${inert}`];
  }
  if (value === undefined) return [];
  return [
    `**${escapedLabel}:**`,
    fence(JSON.stringify(value, null, 2), "json"),
  ];
}

function renderBlock(block: ConversationMarkdownBlock): string {
  // Emitted verbatim: leading indentation can be an indented code block, so
  // trimming is only ever an emptiness test, never applied to the output.
  if (block.kind === "text") {
    return block.text.trim() ? block.text : "";
  }
  if (block.kind === "thinking") {
    if (!block.text.trim()) return "";
    // Collapsed so a transcript reads as the conversation first, with the
    // reasoning still there for anyone who wants it.
    return `<details>\n<summary>thinking</summary>\n\n${block.text}\n\n</details>`;
  }
  if (block.kind === "attachment") {
    return block.label;
  }
  if (block.kind === "source") {
    const title = escapeMarkdownLabel(block.title);
    const url = safeSourceUrl(block.url);
    return url ? `**source:** [${title}](<${url}>)` : `**source:** ${title}`;
  }
  const parts: string[] = [`**tool call:** \`${block.name}\``];
  if (isPlainObject(block.args)) {
    for (const [key, value] of Object.entries(block.args)) {
      parts.push(...renderValue(key, value));
    }
  } else if (block.args !== undefined) {
    parts.push(...renderValue("args", block.args));
  }
  if (block.result !== undefined) {
    parts.push(...renderValue("result", block.result));
  }
  return parts.join("\n\n");
}

function withoutGeneratedImageBytes(value: unknown): unknown {
  if (!isPlainObject(value) || typeof value.image_b64 !== "string") {
    return value;
  }
  const metadata = Object.fromEntries(
    Object.entries(value).filter(([key]) => key !== GENERATED_IMAGE_BYTES_KEY),
  );
  return { image: GENERATED_IMAGE_PLACEHOLDER, ...metadata };
}

export function contentBlocksToMarkdownBlocks(
  content: unknown,
  normalizeToolResult: (
    result: unknown,
  ) => unknown = withoutGeneratedImageBytes,
): ConversationMarkdownBlock[] {
  if (typeof content === "string") {
    return [{ kind: "text", text: content }];
  }
  if (!Array.isArray(content)) {
    return [{ kind: "text", text: JSON.stringify(content) }];
  }

  const blocks: ConversationMarkdownBlock[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const p = part as Record<string, unknown>;
    if (p.type === "text" && typeof p.text === "string") {
      blocks.push({ kind: "text", text: p.text });
    } else if (p.type === "reasoning" || p.type === "thinking") {
      const thinkText =
        typeof p.thinking === "string"
          ? p.thinking
          : typeof p.text === "string"
            ? p.text
            : "";
      if (thinkText) blocks.push({ kind: "thinking", text: thinkText });
    } else if (p.type === "tool-call") {
      blocks.push({
        kind: "tool-call",
        name: typeof p.toolName === "string" ? p.toolName : "unknown",
        args: p.args,
        result: withoutGeneratedImageBytes(normalizeToolResult(p.result)),
      });
    } else if (
      p.type === "source" &&
      typeof p.title === "string" &&
      typeof p.url === "string"
    ) {
      blocks.push({ kind: "source", title: p.title, url: p.url });
    } else if (p.type === "image") {
      blocks.push({ kind: "attachment", label: "[image attachment]" });
    } else if (p.type === "audio") {
      blocks.push({ kind: "attachment", label: "[audio attachment]" });
    }
  }
  return blocks;
}

export function renderConversationBlocks(
  blocks: readonly ConversationMarkdownBlock[],
): string {
  return blocks
    .map((block) => renderBlock(block))
    .filter((rendered) => rendered.length > 0)
    .join("\n\n");
}

export function buildConversationMarkdown(
  messages: readonly ConversationMarkdownMessage[],
): string {
  const sections = messages.flatMap(({ role, content }) => {
    if (!content.trim()) {
      return [];
    }
    const label = roleLabel(role);
    return [`## ${label}\n\n${content}`];
  });
  return sections.length > 0 ? `${sections.join("\n\n")}\n` : "";
}
