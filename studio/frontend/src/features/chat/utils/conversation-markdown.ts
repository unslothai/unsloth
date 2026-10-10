// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  formatMcpToolName,
  mcpServerFromProvenance,
  mcpToolFromProvenance,
} from "./mcp-tool-name.ts";

export type ConversationMarkdownMessage = {
  readonly role: string;
  readonly content: string;
};

export const CONVERSATION_MARKDOWN_FORMAT = "markdown";
export const CONVERSATION_MARKDOWN_LABEL = "Markdown";
export const CONVERSATION_MARKDOWN_EXTENSION = "md";
export const CONVERSATION_MARKDOWN_MIME_TYPE = "text/markdown";
export const CONVERSATION_MARKDOWN_FRAME_PREFIX = "<!-- unsloth-chat-v1:";

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
  // Imported role strings go into a heading: strip line breaks and markup characters.
  const label = role.replace(/[\s]+/g, " ").replace(/[<>&\\[\]`*_#]/g, "").trim();
  return label.length > 0
    ? `${label[0]?.toUpperCase()}${label.slice(1)}`
    : "Message";
}

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
const AUDIO_DATA_URI_PATTERN = /<audio-player\s+src="data:[^"]*"\s*\/>/g;
// Match both ends, or quoted details in reasoning would spend this block's closer.
const DETAILS_TAG_PATTERN = /<(\/?)(details)((?:\s[^>]*)?)>/gi;
// Each message closes the HTML it opened, since a browser keeps reading an open element.
// Best-effort line scanner over a fixed tag set; the export is not a security boundary.
const FENCE_LINE_PATTERN = /^( {0,3})(`{3,}|~{3,})([\s\S]*)$/;
const INDENTED_CODE_PATTERN = /^(?: {4}|\t)/;
// Leading capture instead of lookbehind: Safari < 16.4 cannot parse lookbehind.
const CODE_SPAN_PATTERN = /(^|[^`])(`+)(?!`)[\s\S]*?\2(?!`)/g;
const TAG_OPEN_PATTERN = /^<(\/?)([A-Za-z][^\s/>]*)/;
const BLOCKQUOTE_PATTERN = /^(?: {0,3}>)+/;
// CommonMark 6.3: an angle-bracket link destination is a url, not an element.
const LINK_DESTINATION_PATTERN = /\]\(\s*<[^<>]*>/g;
const IMAGE_DESCRIPTION_PATTERN = /!\[[^\]]*\]/g;
const OPEN_RUN_PATTERN = /(^|[^`])(`+)(?!`)/;
const HTML_BLOCK_START_PATTERN = /^ {0,3}<[A-Za-z!/?]/;

function runClosesLater(lines: readonly string[], from: number, run: string): boolean {
  const closer = new RegExp("(^|[^`])(" + run + ")(?!`)");
  for (let index = from; index < lines.length; index += 2) {
    const line = lines[index] as string;
    const content = line.slice((BLOCKQUOTE_PATTERN.exec(line)?.[0] ?? "").length);
    if (line.trim() === "" || HTML_BLOCK_START_PATTERN.test(content)) return false;
    if (closer.test(line)) return true;
  }
  return false;
}
// Comments, PIs and cdata run to their own terminator, so an open one swallows the file.
const LITERAL_BLOCKS = [
  { opener: "<!--", terminator: "-->", ownLine: false },
  { opener: "<?", terminator: "?>", ownLine: true },
  { opener: "<![CDATA[", terminator: "]]>", ownLine: true },
] as const;
const CLOSED_LITERAL_PATTERN = /<!--[\s\S]*?-->|<\?[\s\S]*?\?>|<!\[CDATA\[[\s\S]*?\]\]>/g;
// Nothing closes plaintext; only the opener can be undone.
const UNCLOSABLE_ELEMENTS: ReadonlySet<string> = new Set(["plaintext"]);
// Containers need a blank line before their closer or it reads as a lazy continuation.
const CONTAINER_ELEMENTS: ReadonlySet<string> = new Set(["details", "select"]);
const RAW_TEXT_ELEMENTS: ReadonlySet<string> = new Set([
  "iframe",
  "noembed",
  "noframes",
  "script",
  "style",
  "textarea",
  "title",
  "xmp",
]);
// CommonMark start condition 1: the block runs to the end tag.
const CONDITION_1_ELEMENTS: ReadonlySet<string> = new Set([
  "pre",
  "script",
  "style",
  "textarea",
]);
// CommonMark 2.4: only an odd backslash run escapes the <.
const ESCAPED_LT_PATTERN = /\\+</g;

const PERSISTENT_ELEMENTS: ReadonlySet<string> = new Set([
  "details",
  "iframe",
  "noembed",
  "noframes",
  "pre",
  "script",
  "select",
  "style",
  "template",
  "textarea",
  "title",
  "xmp",
]);

function closeOpenBlocks(text: string): string {
  const parts = text.split(/(\r\n|[\r\n])/);
  let fence: { readonly run: string; readonly indent: string; readonly quoted: boolean } | null =
    null;
  let block: (typeof LITERAL_BLOCKS)[number] | null = null;
  let blankBefore = true;
  let indented = false;
  let span = "";
  let unfinished: { readonly part: number; readonly column: number }[] = [];
  let quote = "";
  const open: string[] = [];
  const escapes: { readonly part: number; readonly column: number }[] = [];

  for (let part = 0; part < parts.length; part += 2) {
    let line = parts[part] as string;
    const blankLine = line.trim() === "";
    const afterBlank = blankBefore;
    blankBefore = blankLine;
    if (block !== null) {
      const end = line.indexOf(block.terminator);
      if (end === -1) continue;
      const consumed = end + block.terminator.length;
      line = " ".repeat(consumed) + line.slice(consumed);
      block = null;
    }
    // Blank the quote marker rather than slicing, so columns stay intact.
    const marker = BLOCKQUOTE_PATTERN.exec(line)?.[0] ?? "";
    const content = line.slice(marker.length);
    if (fence !== null && fence.quoted && marker === "") fence = null;
    const literal = open.some((name) => CONDITION_1_ELEMENTS.has(name));
    const [, indent = "", run, info = ""] =
      unfinished.length || literal ? [] : FENCE_LINE_PATTERN.exec(content) ?? [];
    if (fence !== null) {
      if (run && run[0] === fence.run[0] && run.length >= fence.run.length && !info.trim()) {
        fence = null;
      }
      continue;
    }
    if (run && (run[0] === "~" || !info.includes("`"))) {
      fence = { run, indent, quoted: marker !== "" };
      continue;
    }
    if (!unfinished.length && !literal) {
      indented = indented
        ? blankLine || INDENTED_CODE_PATTERN.test(content)
        : afterBlank && INDENTED_CODE_PATTERN.test(content);
      if (indented) continue;
    }
    if (span && (blankLine || HTML_BLOCK_START_PATTERN.test(content))) span = "";
    if (span) {
      const closer = new RegExp("(^|[^`])(" + span + ")(?!`)").exec(line);
      if (closer === null) continue;
      const consumed = (closer.index ?? 0) + closer[0].length;
      line = " ".repeat(consumed) + line.slice(consumed);
      span = "";
    }
    // Every substitution below preserves length, so columns still map to the original.
    let prose = line
      .replace(BLOCKQUOTE_PATTERN, (marker) => " ".repeat(marker.length))
      .replace(CODE_SPAN_PATTERN, (span: string, lead: string) =>
        lead + " ".repeat(span.length - lead.length),
      )
      .replace(IMAGE_DESCRIPTION_PATTERN, (alt) => " ".repeat(alt.length))
      .replace(LINK_DESTINATION_PATTERN, (link) => " ".repeat(link.length))
      .replace(CLOSED_LITERAL_PATTERN, (span) => " ".repeat(span.length));
    if (!literal) {
      prose = prose.replace(ESCAPED_LT_PATTERN, (backslashes) =>
        backslashes.length % 2 === 0 ? `${backslashes.slice(0, -1)} ` : backslashes,
      );
    }
    const leftover = OPEN_RUN_PATTERN.exec(prose);
    if (leftover !== null && runClosesLater(parts, part + 2, leftover[2] as string)) {
      span = leftover[2] as string;
      const from = (leftover.index ?? 0) + (leftover[1] as string).length;
      prose = prose.slice(0, from) + " ".repeat(prose.length - from);
    }
    let blockAt = -1;
    if (!unfinished.length) {
      for (const candidate of LITERAL_BLOCKS) {
        const at = prose.lastIndexOf(candidate.opener);
        if (at > blockAt && at > prose.lastIndexOf(candidate.terminator)) {
          blockAt = at;
          block = candidate;
        }
      }
    }
    const tags = blockAt === -1 ? prose : prose.slice(0, blockAt);

    let at = 0;
    while (at <= tags.length) {
      if (!unfinished.length) {
        const start = tags.indexOf("<", at);
        if (start === -1) break;
        if (!TAG_OPEN_PATTERN.test(tags.slice(start))) {
          at = start + 1;
          continue;
        }
        unfinished = [{ part, column: start }];
        quote = "";
        at = start + 1;
      }
      let end = at;
      for (; end < tags.length; end += 1) {
        const character = tags[end] as string;
        if (character === "<" && TAG_OPEN_PATTERN.test(tags.slice(end))) {
          unfinished.push({ part, column: end });
        } else if (quote) {
          if (character === quote) quote = "";
        } else if (character === '"' || character === "'") {
          quote = character;
        } else if (character === ">") {
          break;
        }
      }
      if (end === tags.length) break;
      const start = unfinished[0] as { part: number; column: number };
      const [, slash, name = ""] =
        TAG_OPEN_PATTERN.exec((parts[start.part] as string).slice(start.column)) ?? [];
      unfinished = [];
      at = end + 1;
      const tag = name.toLowerCase();
      const rawText = [...open].reverse().find((name_) => RAW_TEXT_ELEMENTS.has(name_));
      if (rawText !== undefined && !(slash && tag === rawText)) continue;
      if (!slash && UNCLOSABLE_ELEMENTS.has(tag)) {
        escapes.push(start);
        continue;
      }
      if (!PERSISTENT_ELEMENTS.has(tag)) continue;
      if (!slash) {
        open.push(tag);
        continue;
      }
      const index = open.lastIndexOf(tag);
      if (index !== -1) open.splice(index, 1);
    }
  }

  // An unfinished tag cannot be closed, only neutralised; apply last first so columns hold.
  escapes.push(...unfinished);
  escapes.sort((a, b) => b.part - a.part || b.column - a.column);
  for (const { part, column } of escapes) {
    const line = parts[part] as string;
    parts[part] = `${line.slice(0, column)}&lt;${line.slice(column + 1)}`;
  }

  const eol = !text.includes("\n") && text.includes("\r") ? "\r" : "\n";
  let out = parts.join("");
  if (block !== null) out += block.ownLine ? `${eol}${block.terminator}` : block.terminator;
  // Indented to the opener's column: at column zero the closer would end its list.
  if (fence !== null && !fence.quoted) out += `${eol}${fence.indent}${fence.run}`;
  for (let index = open.length - 1; index >= 0; index -= 1) {
    const name = open[index] as string;
    out += CONTAINER_ELEMENTS.has(name)
      ? `${eol}${eol}</${name}>`
      : `${eol}</${name}>`;
  }
  return out;
}

function fence(body: string, language = ""): string {
  const longestRun = [...body.matchAll(/`+/g)].reduce(
    (max, [run]) => Math.max(max, run.length),
    0,
  );
  const ticks = "`".repeat(Math.max(3, longestRun + 1));
  return `${ticks}${language}\n${body}\n${ticks}`;
}

function inlineCode(raw: string): string {
  const value = raw.replace(/[\r\n]+/g, " ");
  const longestRun = [...value.matchAll(/`+/g)].reduce(
    (max, [run]) => Math.max(max, run.length),
    0,
  );
  const ticks = "`".repeat(longestRun + 1);
  // CommonMark 6.1 strips one padding space per end, so padded values need a spare pair.
  const stripped = value.startsWith(" ") && value.endsWith(" ") && value.trim() !== "";
  const pad = !value || value.startsWith("`") || value.endsWith("`") || stripped ? " " : "";
  return `${ticks}${pad}${value}${pad}${ticks}`;
}

function escapeMarkdownLabel(value: string): string {
  // Not _: snake_case keys stay readable.
  return value.replace(/[\r\n]+/g, " ").replace(/([\\[\]*`<])/g, "\\$1");
}

// Escape entity-opening & and backslashes: CommonMark decodes them and can redirect the link.
const ENTITY_REFERENCE_PATTERN =
  /&(?=(?:[A-Za-z][A-Za-z0-9]{1,31}|#\d{1,7}|#[Xx][0-9A-Fa-f]{1,6});)/g;

function escapeMarkdownDestination(url: string): string {
  return url
    .replaceAll("<", "%3C")
    .replaceAll(">", "%3E")
    .replaceAll("\\", "%5C")
    .replace(ENTITY_REFERENCE_PATTERN, "&amp;");
}

function safeSourceUrl(raw: string): string {
  const value = raw.trim();
  if (!value || LINE_BREAK_PATTERN.test(value)) {
    return "";
  }
  try {
    if (value.startsWith("#")) {
      return escapeMarkdownDestination(encodeURI(value));
    }
    const parsed = new URL(value);
    return parsed.protocol === "http:" || parsed.protocol === "https:"
      ? escapeMarkdownDestination(parsed.href)
      : "";
  } catch {
    return "";
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function renderValue(label: string, value: unknown): string[] {
  if (value === undefined) return [];
  const escapedLabel = escapeMarkdownLabel(label);
  if (value === null || typeof value !== "object") {
    const text = typeof value === "string" ? value : String(value);
    if (LINE_BREAK_PATTERN.test(text))
      return [`**${escapedLabel}:**`, fence(text)];
    return [`**${escapedLabel}:** ${inlineCode(text)}`];
  }
  return [
    `**${escapedLabel}:**`,
    fence(JSON.stringify(value, null, 2), "json"),
  ];
}

function renderBlock(block: ConversationMarkdownBlock): string {
  if (block.kind === "text") {
    return block.text.trim() ? closeOpenBlocks(block.text) : "";
  }
  if (block.kind === "thinking") {
    if (!block.text.trim()) return "";
    const text = closeOpenBlocks(
      block.text.replace(DETAILS_TAG_PATTERN, "&lt;$1$2$3>"),
    );
    return `<details>\n<summary>thinking</summary>\n\n${text}\n\n</details>`;
  }
  if (block.kind === "attachment") {
    return escapeMarkdownLabel(block.label);
  }
  if (block.kind === "source") {
    const url = safeSourceUrl(block.url);
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

// Gemini stashes the raw part (with base64 bytes) in args.google.native_part; drop the bytes.
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
  normalizeToolResult: (result: unknown, toolName?: string) => unknown = (
    result,
  ) => result,
): ConversationMarkdownBlock[] {
  if (typeof content === "string") {
    return [{ kind: "text", text: withoutGeneratedAudioBytes(content) }];
  }
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
      const toolName = typeof p.toolName === "string" ? p.toolName : "unknown";
      blocks.push({
        kind: "tool-call",
        name:
          formatMcpToolName(
            toolName,
            mcpServerFromProvenance(p.provenance),
            mcpToolFromProvenance(p.provenance),
          ) ?? toolName,
        args: withoutNativePartBytes(p.args),
        result: withoutGeneratedImageBytes(
          normalizeToolResult(p.result, toolName),
        ),
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
  options: { includeImportMetadata?: boolean } = {},
): string {
  const sections = messages.flatMap(({ role, content }) => {
    if (!content.trim()) {
      return [];
    }
    const label = roleLabel(role);
    const text = options.includeImportMetadata
      ? content.replace(/\r\n?/g, "\n")
      : content;
    return [`## ${label}\n\n${text}`];
  });
  if (sections.length === 0) return "";
  const body = `${sections.join("\n\n")}\n`;
  if (!options.includeImportMetadata) return body;
  const lengths = JSON.stringify(sections.map((section) => section.length));
  return `${CONVERSATION_MARKDOWN_FRAME_PREFIX}${lengths} -->\n\n${body}`;
}
