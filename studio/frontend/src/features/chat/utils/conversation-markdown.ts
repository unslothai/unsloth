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
const GENERATED_AUDIO_PLACEHOLDER = "[generated audio omitted]";
const INLINE_DATA_PLACEHOLDER = "[inline data omitted]";
const LINE_BREAK_PATTERN = /[\r\n]/;
// Chat stores generated audio as a custom tag holding a base64 data URI; the
// bytes are for the player, not for a transcript.
const AUDIO_DATA_URI_PATTERN = /<audio-player\s+src="data:[^"]*"\s*\/>/g;
const DETAILS_CLOSE_PATTERN = /<\/(details)\s*>/gi;
const FENCE_PATTERN = /^ {0,3}(`{3,}|~{3,})/;

// A message that ends mid-fence or mid-comment swallows everything after it,
// including the next role heading, so each turn closes what it opened.
function closeOpenBlocks(text: string): string {
  let open: string | null = null;
  for (const line of text.split("\n")) {
    const run = FENCE_PATTERN.exec(line)?.[1];
    if (!run) continue;
    if (open === null) open = run;
    else if (run[0] === open[0] && run.length >= open.length) open = null;
  }
  let out = open === null ? text : `${text}\n${open}`;
  if (out.lastIndexOf("<!--") > out.lastIndexOf("-->")) out += "-->";
  return out;
}

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
  // Padding keeps a leading or trailing backtick from closing the span, and
  // keeps an empty value from collapsing into one delimiter run.
  const pad = !value || value.startsWith("`") || value.endsWith("`") ? " " : "";
  return `${ticks}${pad}${value}${pad}${ticks}`;
}

// Labels sit inside ** ** and [ ]: a line break ends either one, letting the
// rest of the label out as markdown, a backtick opens a code span, and < starts
// inline html that a permissive local viewer will render.
function escapeMarkdownLabel(value: string): string {
  // Not _: it cannot emphasise inside a word, and escaping it would turn every
  // snake_case tool key into snake\_case.
  return value.replace(/[\r\n]+/g, " ").replace(/([\\[\]*`<])/g, "\\$1");
}

function safeSourceUrl(raw: string): string {
  const value = raw.trim();
  if (!value || LINE_BREAK_PATTERN.test(value)) {
    return "";
  }
  try {
    // encodeURI throws on a lone surrogate, so it stays inside the guard.
    if (value.startsWith("#")) {
      return encodeURI(value).replaceAll("<", "%3C").replaceAll(">", "%3E");
    }
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
  if (value === undefined) return [];
  const escapedLabel = escapeMarkdownLabel(label);
  if (typeof value === "string") {
    if (LINE_BREAK_PATTERN.test(value)) return [`**${escapedLabel}:**`, fence(value)];
    // A code span rather than escaping: tool values are data, and any list of
    // markdown metacharacters to escape is one another syntax slips past.
    return [`**${escapedLabel}:** ${inlineCode(value)}`];
  }
  // Scalars inline too. A json fence spends three lines on `10`, and a tool
  // call with four of them reads as a wall of fences instead of a call.
  if (value === null || typeof value !== "object") {
    return [`**${escapedLabel}:** ${inlineCode(String(value))}`];
  }
  return [
    `**${escapedLabel}:**`,
    fence(JSON.stringify(value, null, 2), "json"),
  ];
}

function renderBlock(block: ConversationMarkdownBlock): string {
  // Emitted verbatim: leading indentation can be an indented code block, so
  // trimming is only ever an emptiness test, never applied to the output.
  if (block.kind === "text") {
    return block.text.trim() ? closeOpenBlocks(block.text) : "";
  }
  if (block.kind === "thinking") {
    if (!block.text.trim()) return "";
    // Reasoning that quotes a closing details tag would end the block early;
    // the entity renders as that same literal text.
    const text = closeOpenBlocks(
      block.text.replace(DETAILS_CLOSE_PATTERN, "&lt;/$1>"),
    );
    // Collapsed so a transcript reads as the conversation first, with the
    // reasoning still there for anyone who wants it.
    return `<details>\n<summary>thinking</summary>\n\n${text}\n\n</details>`;
  }
  if (block.kind === "attachment") {
    // Escaped: a link reference definition elsewhere in the transcript would
    // otherwise resolve the bare [label] into a link.
    return escapeMarkdownLabel(block.label);
  }
  if (block.kind === "source") {
    const url = safeSourceUrl(block.url);
    // A rejected destination falls back to a code span, not plain text: a
    // title that is itself a bare url would just get autolinked instead.
    return url
      ? `**source:** [${escapeMarkdownLabel(block.title)}](<${url}>)`
      : `**source:** ${inlineCode(block.title)}`;
  }
  const parts: string[] = [`**tool call:** ${inlineCode(block.name)}`];
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
  if (
    !isPlainObject(value) ||
    typeof value[GENERATED_IMAGE_BYTES_KEY] !== "string"
  ) {
    return value;
  }
  const metadata = Object.fromEntries(
    Object.entries(value).filter(([key]) => key !== GENERATED_IMAGE_BYTES_KEY),
  );
  // Placeholder last: a result carrying its own image key must not shadow it.
  return { ...metadata, image: GENERATED_IMAGE_PLACEHOLDER };
}

function withoutInlineDataBytes(
  part: Record<string, unknown>,
): Record<string, unknown> {
  const inlineData = part.inlineData;
  if (!isPlainObject(inlineData) || typeof inlineData.data !== "string") {
    return part;
  }
  return {
    ...part,
    inlineData: { ...inlineData, data: INLINE_DATA_PLACEHOLDER },
  };
}

// Gemini keeps a code-execution turn replayable by stashing the raw part in
// args.google.native_part, base64 inlineData included. Same reasoning as
// generated images: keep the metadata, drop the bytes.
function withoutNativePartBytes(args: unknown): unknown {
  if (!isPlainObject(args) || !isPlainObject(args.google)) return args;
  const google = args.google;
  const native = google.native_part;
  if (!isPlainObject(native)) return args;
  const cleaned = Array.isArray(native.parts)
    ? {
        ...native,
        parts: native.parts.map((part) =>
          isPlainObject(part) ? withoutInlineDataBytes(part) : part,
        ),
      }
    : withoutInlineDataBytes(native);
  return { ...args, google: { ...google, native_part: cleaned } };
}

function withoutGeneratedAudioBytes(text: string): string {
  return text.replace(AUDIO_DATA_URI_PATTERN, GENERATED_AUDIO_PLACEHOLDER);
}

export function contentBlocksToMarkdownBlocks(
  content: unknown,
  normalizeToolResult: (result: unknown) => unknown = (result) => result,
): ConversationMarkdownBlock[] {
  if (typeof content === "string") {
    return [{ kind: "text", text: withoutGeneratedAudioBytes(content) }];
  }
  // A record written without content at all would otherwise stringify to the
  // undefined value, not a string, and take the whole export down with it.
  if (content == null) {
    return [];
  }
  if (!Array.isArray(content)) {
    return [{ kind: "text", text: JSON.stringify(content) }];
  }

  const blocks: ConversationMarkdownBlock[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    const p = part as Record<string, unknown>;
    if (p.type === "text" && typeof p.text === "string") {
      blocks.push({ kind: "text", text: withoutGeneratedAudioBytes(p.text) });
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
        args: withoutNativePartBytes(p.args),
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
