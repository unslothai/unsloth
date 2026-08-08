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
// Both ends, not just the closer: reasoning that quotes a whole details
// element would otherwise keep its opener, hand its closer to the entity, and
// leave this block's own closer to the inner element, hiding every later turn.
const DETAILS_TAG_PATTERN = /<(\/?)(details)((?:\s[^>]*)?)>/gi;
// details is a condition 6 tag, so its markdown block ends at the blank line
// between two turns, but the element the browser opened does not: an unmatched
// opener collapses every later turn inside it. Counted rather than escaped,
// because a matched details element is a disclosure widget the chat itself
// renders and the transcript has to keep it. Ends the way the html tokenizer
// ends a tag name, not the way DETAILS_TAG_PATTERN does: <details/> and a bare
// <details at the end of a line both still open an element.
const DETAILS_TAG_SCAN_PATTERN = /<(\/?)details(?=[\s/>]|$)/gi;
// A code span is literal text, so a <!-- inside one opens nothing.
const CODE_SPAN_PATTERN = /(`+)[^`]*\1/g;
// Same for a comment: <!-- <details> --> opens nothing either.
const CLOSED_COMMENT_PATTERN = /<!--[\s\S]*?-->/g;
// Four spaces of indentation is an indented code block, so a fence or a raw
// html block only starts within the first three columns. Splitting the
// indentation off once keeps the opener's container context, which the
// synthesized closer has to be written back into.
// [\s\S] rather than .: a line separator is an ordinary character to a
// markdown parser, and . would drop the whole line here.
const INDENT_PATTERN = /^( {0,3})([\s\S]*)$/;
const FENCE_PATTERN = /^(`{3,}|~{3,})(.*)$/;
const EOL_PATTERN = /\r\n|[\r\n]/;

type HtmlBlockRule = {
  readonly start: RegExp;
  readonly end: RegExp;
  readonly close: (match: RegExpExecArray) => string;
};

// Raw html blocks that a blank line does not close (CommonMark 4.6, start
// conditions 1, 3, 4 and 5): they run to the end of the document. Condition 2
// is the comment tracked below; 6 and 7 end at the blank line that already
// separates every turn, so they need no repair.
const HTML_BLOCK_RULES: readonly HtmlBlockRule[] = [
  {
    start: /^<(pre|script|style|textarea)(?=[ \t>]|$)/i,
    end: /<\/(?:pre|script|style|textarea)>/i,
    // Any of the four end tags satisfies the parser, but a viewer that renders
    // the raw html needs the tag that was actually opened.
    close: (match) => `</${(match[1] ?? "pre").toLowerCase()}>`,
  },
  { start: /^<\?/, end: /\?>/, close: () => "?>" },
  { start: /^<![A-Za-z]/, end: />/, close: () => ">" },
  { start: /^<!\[CDATA\[/, end: /\]\]>/, close: () => "]]>" },
];

function openedHtmlBlock(
  rest: string,
  line: string,
): { readonly rule: HtmlBlockRule; readonly close: string } | null {
  for (const rule of HTML_BLOCK_RULES) {
    const match = rule.start.exec(rest);
    // A line that meets the start and the end condition is the whole block.
    if (!match || rule.end.test(line)) continue;
    return { rule, close: rule.close(match) };
  }
  return null;
}

// A message that ends mid-fence, mid-comment, inside a raw html block or
// inside a details element swallows everything after it, including the next
// role heading, so each turn closes what it opened. The first three are
// tracked as one state: inside any of them, the other two are literal, and so
// is a details tag.
function closeOpenBlocks(text: string): string {
  let open: string | null = null;
  let openIndent = "";
  let comment = false;
  let html: HtmlBlockRule | null = null;
  let htmlClose = "";
  let htmlIndent = "";
  let details = 0;
  // Split the way a parser does. Splitting on \n alone hides a fence opened in
  // a body that uses bare carriage returns.
  for (const line of text.split(EOL_PATTERN)) {
    if (comment) {
      if (line.includes("-->")) comment = false;
      continue;
    }
    if (html) {
      if (html.end.test(line)) html = null;
      continue;
    }
    const [, indent = "", rest = ""] = INDENT_PATTERN.exec(line) ?? [];
    const [, run, info] = FENCE_PATTERN.exec(rest) ?? [];
    if (open !== null) {
      // A closer repeats the opener's character, is at least as long, and
      // carries no info string.
      if (run && run[0] === open[0] && run.length >= open.length && !info?.trim()) {
        open = null;
      }
      continue;
    }
    if (run) {
      // A backtick opener cannot carry a backtick in its info string; that
      // line is a paragraph, and treating it as a fence would open a real one.
      if (run[0] === "~" || !info?.includes("`")) {
        open = run;
        openIndent = indent;
      }
      continue;
    }
    const opened = openedHtmlBlock(rest, line);
    if (opened) {
      html = opened.rule;
      htmlClose = opened.close;
      htmlIndent = indent;
      continue;
    }
    // Deliberately not limited to a line-initial <!--: a mid-line one inside a
    // raw html block still opens a comment in the browser.
    const prose = line.replace(CODE_SPAN_PATTERN, "");
    const commentStart = prose.lastIndexOf("<!--");
    comment = commentStart > prose.lastIndexOf("-->");
    // Mid-line too, and for the same reason: `hello <details>` opens the
    // element even though no line here begins a raw html block.
    const tags = (comment ? prose.slice(0, commentStart) : prose).replace(
      CLOSED_COMMENT_PATTERN,
      "",
    );
    for (const [, slash] of tags.matchAll(DETAILS_TAG_SCAN_PATTERN)) {
      // Clamped, not signed: a stray closer is inert in every html parser, so
      // it must not license an opener later in the message.
      details = slash ? Math.max(0, details - 1) : details + 1;
    }
  }
  let out = text;
  if (comment) out += "-->";
  // Close on the body's own line ending. A renderer that ignores bare carriage
  // returns would read a \n-prefixed closer as a fresh fence instead.
  const eol = !text.includes("\n") && text.includes("\r") ? "\r" : "\n";
  // Indented back to the opener's column: a closer written at column zero ends
  // the list the opener sits in, and then opens a block of its own instead.
  if (open !== null) out += `${eol}${openIndent}${open}`;
  if (html) out += `${eol}${htmlIndent}${htmlClose}`;
  // Last, and after a blank line: the closer has to sit outside every other
  // construct repaired above, and at column zero as its own html block rather
  // than as a lazy continuation of the paragraph it follows.
  out += `${eol}${eol}</details>`.repeat(details);
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
  // A code span cannot hold a line break: a blank line ends it outright and
  // leaves the rest of the value loose as markdown.
  const value = raw.replace(/[\r\n]+/g, " ");
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

// A destination is parsed, not copied: CommonMark 6.2 resolves entity
// references inside it and 2.4 consumes a backslash before ASCII punctuation.
// Search results are attacker-controllable, and both rules move where the link
// goes: `https://docs.unsloth.ai&commat;evil.test/` is a url whose host is
// docs.unsloth.ai until a viewer decodes it into `docs.unsloth.ai@evil.test`,
// which is credentials on evil.test. Only the ampersand that opens an entity
// reference is escaped, so an ordinary query separator stays readable, and the
// escape is &amp; rather than %26 because percent-encoding one would fold the
// separator into the preceding value.
const ENTITY_REFERENCE_PATTERN =
  /&(?=(?:[A-Za-z][A-Za-z0-9]{1,31}|#\d{1,7}|#[Xx][0-9A-Fa-f]{1,6});)/g;

function escapeMarkdownDestination(url: string): string {
  // Percent-encoded rather than doubled: %5C is what a renderer writes for a
  // surviving backslash anyway, so the destination stays a plain url.
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

// A multi-line string is almost always source: html, a patch, a script. Fence
// it as-is so it reads as code instead of a JSON string full of \n. No language
// tag, since guessing one wrong is worse than none.
function renderValue(label: string, value: unknown): string[] {
  if (value === undefined) return [];
  const escapedLabel = escapeMarkdownLabel(label);
  // Scalars render as text too. A json fence spends three lines on `10`, and a
  // tool call with four of them reads as a wall of fences instead of a call.
  if (value === null || typeof value !== "object") {
    const text = typeof value === "string" ? value : String(value);
    if (LINE_BREAK_PATTERN.test(text)) return [`**${escapedLabel}:**`, fence(text)];
    // A code span rather than escaping: tool values are data, and any list of
    // markdown metacharacters to escape is one another syntax slips past.
    return [`**${escapedLabel}:** ${inlineCode(text)}`];
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
      block.text.replace(DETAILS_TAG_PATTERN, "&lt;$1$2$3>"),
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
