// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { formatMcpToolName, mcpServerFromProvenance } from "./mcp-tool-name.ts";

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
  // Imported transcripts keep their role strings verbatim, and this one is interpolated into a
  // ## heading closeOpenBlocks never sees: a line break would end the heading, a tag would
  // open an element nothing later closes.
  const label = role.replace(/[\s]+/g, " ").replace(/[<>&\\[\]`*_#]/g, "").trim();
  return label.length > 0
    ? `${label[0]?.toUpperCase()}${label.slice(1)}`
    : "Message";
}

// A message split into renderable pieces. JSONL/CSV exports flatten to plain text; markdown
// is a document format, so each piece renders on its own terms.
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
// Generated audio is a custom tag holding a base64 data URI: player bytes, not transcript.
const AUDIO_DATA_URI_PATTERN = /<audio-player\s+src="data:[^"]*"\s*\/>/g;
// Both ends, not just the closer: reasoning quoting a whole details element would keep its
// opener and spend this block's closer, hiding every later turn.
const DETAILS_TAG_PATTERN = /<(\/?)(details)((?:\s[^>]*)?)>/gi;
// A blank line and a ## heading separate turns for the markdown parser but not the browser:
// an element left open keeps reading, so each message closes what it opened. This is a
// line scanner over a fixed tag set and is meant to stay one: it covers what people
// actually leave open. An unfinished tag is neutralised rather than closed, since nothing
// can close it. Shapes needing a real tokenizer are repaired best effort or not at all:
// the export is a document, not a security boundary.
const FENCE_LINE_PATTERN = /^( {0,3})(`{3,}|~{3,})([\s\S]*)$/;
const INDENTED_CODE_PATTERN = /^(?: {4}|\t)/;
// Equal-length delimiters, so a wide span may hold a narrower run. CommonMark 6.1 reads a
// contiguous run as one delimiter, so neither end may sit inside a longer one. The leading
// capture pins the opener to the start of its run in place of a lookbehind, which Safari
// only gained in 16.4 and would take the whole bundle down if unparseable.
const CODE_SPAN_PATTERN = /(^|[^`])(`+)(?!`)[\s\S]*?\2(?!`)/g;
// The html tokenizer ends a tag name at whitespace, a solidus or >.
const TAG_OPEN_PATTERN = /^<(\/?)([A-Za-z][^\s/>]*)/;
// Block quote markers are structure, not content: a > opening a quoted line is not the >
// that finishes a tag left open above.
const BLOCKQUOTE_PATTERN = /^(?: {0,3}>)+/;
// CommonMark 6.3: an angle-bracket destination is a url, so [x](<details>) is a link, not an
// element; reading it as one emits or spends a closer wrongly.
const LINK_DESTINATION_PATTERN = /\]\(\s*<[^<>]*>/g;
// An image description becomes an escaped alt attribute, so a tag there is text. A link label
// is not: CommonMark parses raw html inside one.
const IMAGE_DESCRIPTION_PATTERN = /!\[[^\]]*\]/g;
// A run left over once a line's closed spans are masked opens a span into the next.
const OPEN_RUN_PATTERN = /(^|[^`])(`+)(?!`)/;
// Block structure is read before inlines, so a line starting with a tag begins an html block
// and ends any span still open above it.
const HTML_BLOCK_START_PATTERN = /^ {0,3}<[A-Za-z!/?]/;

// A run only opens a span if its match arrives; an unmatched run is live text. The search
// stops where a span would, at a blank line or an html block.
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
// Comments, processing instructions and cdata run to their own terminator, not to a blank
// line or tag, so one left open swallows the file. ownLine: a ?> appended to the last
// line would land inside the instruction.
const LITERAL_BLOCKS = [
  { opener: "<!--", terminator: "-->", ownLine: false },
  { opener: "<?", terminator: "?>", ownLine: true },
  { opener: "<![CDATA[", terminator: "]]>", ownLine: true },
] as const;
// One that opens and closes on the same line is literal text, not an opener.
const CLOSED_LITERAL_PATTERN = /<!--[\s\S]*?-->|<\?[\s\S]*?\?>|<!\[CDATA\[[\s\S]*?\]\]>/g;
// Nothing closes plaintext: the browser reads every byte after it as that element's text.
// The opener itself is the only thing that can be undone.
const UNCLOSABLE_ELEMENTS: ReadonlySet<string> = new Set(["plaintext"]);
// Containers need a blank line before their closer or it reads as a lazy continuation; in a
// raw text element that blank line is content instead.
const CONTAINER_ELEMENTS: ReadonlySet<string> = new Set(["details", "select"]);
// Read as text until their own end tag, so a </details> inside one must not close a details
// open outside it. pre is absent: it holds real markup.
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
// CommonMark start condition 1: inside one of these the block runs to the end tag, so a fence
// or indented code line in there is literal text.
const CONDITION_1_ELEMENTS: ReadonlySet<string> = new Set([
  "pre",
  "script",
  "style",
  "textarea",
]);
// CommonMark 2.4: a backslash before ascii punctuation makes it literal, so \<script> opens
// nothing. Only an odd run escapes.
const ESCAPED_LT_PATTERN = /\\+</g;

// Elements a blank line does not close: the browser reads their content as text or holds them
// open as a container, so whatever follows is swallowed. GFM's tagfilter lists nearly the
// same set for the same reason.
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
  // Split keeping the separators, so a repaired line can be put back verbatim.
  const parts = text.split(/(\r\n|[\r\n])/);
  let fence: { readonly run: string; readonly indent: string; readonly quoted: boolean } | null =
    null;
  let block: (typeof LITERAL_BLOCKS)[number] | null = null;
  let blankBefore = true;
  // An indented code block runs through blank lines, so its state outlives its opener.
  let indented = false;
  // A code span may cross a soft line break, so an unclosed run keeps masking.
  let span = "";
  // Tag starts whose > has not arrived, and the quote a value is inside. Every < after an
  // unfinished tag is inside it until that one is neutralised.
  let unfinished: { readonly part: number; readonly column: number }[] = [];
  let quote = "";
  const open: string[] = [];
  // Openers to neutralise, because no closer would reach them.
  const escapes: { readonly part: number; readonly column: number }[] = [];

  for (let part = 0; part < parts.length; part += 2) {
    let line = parts[part] as string;
    const blankLine = line.trim() === "";
    const afterBlank = blankBefore;
    blankBefore = blankLine;
    if (block !== null) {
      const end = line.indexOf(block.terminator);
      if (end === -1) continue;
      // The rest of the line is live again, so blank what the block held and rescan.
      const consumed = end + block.terminator.length;
      line = " ".repeat(consumed) + line.slice(consumed);
      block = null;
    }
    // Fences, indents and tags mean different things inside a block quote, so read them against
    // the quoted content. Blanking, not slicing, keeps columns intact.
    const marker = BLOCKQUOTE_PATTERN.exec(line)?.[0] ?? "";
    const content = line.slice(marker.length);
    // The quote ending ends the fence, so an unclosed one never reaches the next turn.
    if (fence !== null && fence.quoted && marker === "") fence = null;
    const literal = open.some((name) => CONDITION_1_ELEMENTS.has(name));
    const [, indent = "", run, info = ""] =
      unfinished.length || literal ? [] : FENCE_LINE_PATTERN.exec(content) ?? [];
    if (fence !== null) {
      // A closer repeats the opener's character, is at least as long, and has no info.
      if (run && run[0] === fence.run[0] && run.length >= fence.run.length && !info.trim()) {
        fence = null;
      }
      continue;
    }
    if (run && (run[0] === "~" || !info.includes("`"))) {
      fence = { run, indent, quoted: marker !== "" };
      continue;
    }
    // A backtick opener with a backtick in its info string is not a fence: the line is prose and
    // falls through, since its tags are live. Indented code, code spans and closed comments are
    // literal, but indented code only starts after a blank line, else the line is a lazy
    // continuation and live.
    if (!unfinished.length && !literal) {
      indented = indented
        ? blankLine || INDENTED_CODE_PATTERN.test(content)
        : afterBlank && INDENTED_CODE_PATTERN.test(content);
      if (indented) continue;
    }
    // A span in progress swallows this line up to its closing run. A blank line ends it, and so
    // does an html block, which the parser sees first.
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
    // The backslash is markdown's, so it holds only where that parser reads inlines.
    if (!literal) {
      prose = prose.replace(ESCAPED_LT_PATTERN, (backslashes) =>
        backslashes.length % 2 === 0 ? `${backslashes.slice(0, -1)} ` : backslashes,
      );
    }
    // A run left after masking closed spans runs on, so the rest of the line is code.
    const leftover = OPEN_RUN_PATTERN.exec(prose);
    if (leftover !== null && runClosesLater(parts, part + 2, leftover[2] as string)) {
      span = leftover[2] as string;
      const from = (leftover.index ?? 0) + (leftover[1] as string).length;
      prose = prose.slice(0, from) + " ".repeat(prose.length - from);
    }
    // These first: whichever opens last swallows every tag after it.
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
      // The > only ends a tag outside an attribute value, so an unbalanced quote reads on.
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
      // Innermost matching opener, so </div></details> still closes the details. A closer with no
      // opener is inert, so it must not license a later opener.
      const index = open.lastIndexOf(tag);
      if (index !== -1) open.splice(index, 1);
    }
  }

  // An unfinished tag cannot be closed: the tokenizer is still reading its name or attribute
  // value, so a later </script> is swallowed. Neutralising the < is the only repair.
  // Last first, so columns do not shift.
  escapes.push(...unfinished);
  escapes.sort((a, b) => b.part - a.part || b.column - a.column);
  for (const { part, column } of escapes) {
    const line = parts[part] as string;
    parts[part] = `${line.slice(0, column)}&lt;${line.slice(column + 1)}`;
  }

  // Close on the body's own line ending. A renderer that ignores bare carriage returns would
  // read a \n-prefixed closer as a fresh fence instead.
  const eol = !text.includes("\n") && text.includes("\r") ? "\r" : "\n";
  let out = parts.join("");
  if (block !== null) out += block.ownLine ? `${eol}${block.terminator}` : block.terminator;
  // Indented to the opener's column: at column zero the closer would end its list.
  if (fence !== null && !fence.quoted) out += `${eol}${fence.indent}${fence.run}`;
  // Innermost first, so closers nest as the openers did; each on its own line after a blank one
  // so it is a block, not a lazy continuation.
  for (let index = open.length - 1; index >= 0; index -= 1) {
    const name = open[index] as string;
    out += CONTAINER_ELEMENTS.has(name)
      ? `${eol}${eol}</${name}>`
      : `${eol}</${name}>`;
  }
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

function inlineCode(raw: string): string {
  // A code span cannot hold a line break: it would leave the rest loose as markdown.
  const value = raw.replace(/[\r\n]+/g, " ");
  const longestRun = [...value.matchAll(/`+/g)].reduce(
    (max, [run]) => Math.max(max, run.length),
    0,
  );
  const ticks = "`".repeat(longestRun + 1);
  // Padding keeps an edge backtick from closing the span and an empty value from collapsing
  // into one delimiter run. CommonMark 6.1 also strips one space from each padded end
  // unless all spaces, so such a value needs a spare pair.
  const stripped = value.startsWith(" ") && value.endsWith(" ") && value.trim() !== "";
  const pad = !value || value.startsWith("`") || value.endsWith("`") || stripped ? " " : "";
  return `${ticks}${pad}${value}${pad}${ticks}`;
}

// Labels sit inside ** ** and [ ]: a line break ends either and lets the rest out as
// markdown, a backtick opens a code span, and < starts inline html.
function escapeMarkdownLabel(value: string): string {
  // Not _: it cannot emphasise inside a word, so snake_case keys stay readable.
  return value.replace(/[\r\n]+/g, " ").replace(/([\\[\]*`<])/g, "\\$1");
}

// A destination is parsed, not copied: CommonMark 6.2 resolves entity references inside it
// and 2.4 consumes a backslash before ASCII punctuation, and both move where the link goes
// (`https://docs.unsloth.ai&commat;evil.test/` decodes to credentials on evil.test). Only
// an entity-opening ampersand is escaped, and &amp; not %26 because encoding one folds it
// into the value before.
const ENTITY_REFERENCE_PATTERN =
  /&(?=(?:[A-Za-z][A-Za-z0-9]{1,31}|#\d{1,7}|#[Xx][0-9A-Fa-f]{1,6});)/g;

function escapeMarkdownDestination(url: string): string {
  // Percent-encoded, not doubled: %5C is what a renderer writes for it anyway.
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
    // encodeURI throws on a lone surrogate, so it stays inside the guard.
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

// A multi-line string is almost always source, so fence it as-is rather than let it read as
// a JSON string full of \n. No language tag: guessing wrong is worse than none.
function renderValue(label: string, value: unknown): string[] {
  if (value === undefined) return [];
  const escapedLabel = escapeMarkdownLabel(label);
  // Scalars as text too: a json fence spends three lines on `10`.
  if (value === null || typeof value !== "object") {
    const text = typeof value === "string" ? value : String(value);
    if (LINE_BREAK_PATTERN.test(text))
      return [`**${escapedLabel}:**`, fence(text)];
    // A code span, not escaping: any list of metacharacters to escape misses one.
    return [`**${escapedLabel}:** ${inlineCode(text)}`];
  }
  return [
    `**${escapedLabel}:**`,
    fence(JSON.stringify(value, null, 2), "json"),
  ];
}

function renderBlock(block: ConversationMarkdownBlock): string {
  // Verbatim: leading indentation can be code, so trim is only an emptiness test.
  if (block.kind === "text") {
    return block.text.trim() ? closeOpenBlocks(block.text) : "";
  }
  if (block.kind === "thinking") {
    if (!block.text.trim()) return "";
    // A quoted </details> would end the block early; the entity renders the same.
    const text = closeOpenBlocks(
      block.text.replace(DETAILS_TAG_PATTERN, "&lt;$1$2$3>"),
    );
    // Collapsed so the transcript reads as the conversation first.
    return `<details>\n<summary>thinking</summary>\n\n${text}\n\n</details>`;
  }
  if (block.kind === "attachment") {
    // Escaped: a link reference definition elsewhere would turn [label] into a link.
    return escapeMarkdownLabel(block.label);
  }
  if (block.kind === "source") {
    const url = safeSourceUrl(block.url);
    // Code span, not plain text: a title that is itself a bare url would autolink.
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
// args.google.native_part, base64 inlineData included. Keep metadata, drop bytes.
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
  // Otherwise a content-less record stringifies to undefined and breaks the export.
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
          formatMcpToolName(toolName, mcpServerFromProvenance(p.provenance)) ??
          toolName,
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
