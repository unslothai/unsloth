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
  // An imported transcript carries its own role strings (oaiMessagesToRecords
  // keeps them verbatim), and this one is interpolated into a ## heading that
  // sits outside closeOpenBlocks. A line break would end the heading and a tag
  // would open an element nothing later closes, so the label is reduced to one
  // line of text before it is trusted with the document's own structure.
  const label = role.replace(/[\s]+/g, " ").replace(/[<>&\\[\]`*_#]/g, "").trim();
  return label.length > 0
    ? `${label[0]?.toUpperCase()}${label.slice(1)}`
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
// A turn is separated from the next by a blank line and a ## heading. That is
// enough for the markdown parser but not for the browser: an element left open
// keeps reading, so a message that forgets a closing fence, comment or tag
// takes every later turn inside it and the transcript silently loses them.
// Each message therefore closes what it opened.
//
// This is a line scanner over a fixed tag set, not a parser, and it is meant to
// stay one. It covers what a person actually leaves open: an unclosed code
// fence, an unterminated comment, a quoted details element, a pasted script.
// A tag left unfinished is neutralised rather than closed, since nothing can
// close it. Shapes that need a real html tokenizer to see (a processing
// instruction, self-closing tags in svg or mathml) are repaired best effort or
// not at all: closing that gap costs an html parser, and the export is a
// document, not a security boundary.
const FENCE_LINE_PATTERN = /^( {0,3})(`{3,}|~{3,})([\s\S]*)$/;
const INDENTED_CODE_PATTERN = /^(?: {4}|\t)/;
// Equal-length delimiters, so a wide span may hold a narrower run: `` `x` ``
// is one span, not two. The lookahead keeps a closer from matching a prefix of
// a longer run.
// CommonMark 6.1 reads the whole contiguous run as one delimiter, so neither
// end may sit inside a longer one: ```x`` is not a two-backtick span, it is
// live text, and masking it would hide the tags it holds. The leading capture
// pins the opener to the start of its run the way a lookbehind would, without
// needing one -- Safari only gained those in 16.4, and an unparseable pattern
// takes the whole bundle down rather than this one line.
const CODE_SPAN_PATTERN = /(^|[^`])(`+)(?!`)[\s\S]*?\2(?!`)/g;
// The html tokenizer ends a tag name at whitespace, a solidus or >.
const TAG_OPEN_PATTERN = /^<(\/?)([A-Za-z][^\s/>]*)/;
// Block quote markers are structure, not content: a > that opens a quoted line
// is not the > that finishes a tag left open on the line above.
const BLOCKQUOTE_PATTERN = /^(?: {0,3}>)+/;
// CommonMark 6.3: a destination in angle brackets is a url, so [x](<details>)
// is a link and not an element. Reading it as one would emit a closer the
// message never opened, or spend the closer of one it did.
const LINK_DESTINATION_PATTERN = /\]\(\s*<[^<>]*>/g;
// An image description becomes the alt attribute, escaped, so a tag in there
// is text. A link label is not: CommonMark parses raw html inside one.
const IMAGE_DESCRIPTION_PATTERN = /!\[[^\]]*\]/g;
// A backtick run left over once the closed spans on a line are masked. It opens
// a span that carries on to the next line.
const OPEN_RUN_PATTERN = /(^|[^`])(`+)(?!`)/;
// Block structure is read before inlines, so a line starting with a tag begins
// an html block and ends any span that was still open above it.
const HTML_BLOCK_START_PATTERN = /^ {0,3}<[A-Za-z!/?]/;

// A run only opens a span if its match actually arrives, so look ahead before
// masking anything: an unmatched run is ordinary live text. The search stops
// where a span would, at a blank line or an html block.
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
// A comment, a processing instruction and a cdata section are read to their
// own terminator, not
// to a blank line or a tag, so one left open swallows the rest of the file.
// ownLine because a ?> appended to the last line would land inside the
// instruction's own text rather than end it.
const LITERAL_BLOCKS = [
  { opener: "<!--", terminator: "-->", ownLine: false },
  { opener: "<?", terminator: "?>", ownLine: true },
  { opener: "<![CDATA[", terminator: "]]>", ownLine: true },
] as const;
// One that opens and closes on the same line is literal text, not an opener.
const CLOSED_LITERAL_PATTERN = /<!--[\s\S]*?-->|<\?[\s\S]*?\?>|<!\[CDATA\[[\s\S]*?\]\]>/g;
// Nothing at all closes plaintext: the browser reads every byte after it as
// that element's text. The opener itself is the only thing that can be undone.
const UNCLOSABLE_ELEMENTS: ReadonlySet<string> = new Set(["plaintext"]);
// Container elements need a blank line before their closer or it is read as a
// lazy continuation of the paragraph above. Raw text elements must not have
// one: there the blank line is content the element would render.
const CONTAINER_ELEMENTS: ReadonlySet<string> = new Set(["details", "select"]);
// The tokenizer reads these as text, not markup, until it meets their own end
// tag, so a </details> inside one is script data and must not close a details
// that is genuinely open outside it. pre is not here: it holds real markup.
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
// CommonMark start condition 1: inside one of these the markdown block runs to
// the end tag, so a fence or an indented code line in there is literal text.
const CONDITION_1_ELEMENTS: ReadonlySet<string> = new Set([
  "pre",
  "script",
  "style",
  "textarea",
]);
// CommonMark 2.4: a backslash before ascii punctuation makes it literal, so
// \<script> opens nothing. Only an odd run of backslashes escapes, which is an
// even match length once the < is counted.
const ESCAPED_LT_PATTERN = /\\+</g;

// Elements a blank line does not close. The browser either reads their content
// as text (script, style, textarea, title, xmp, iframe, noembed, noframes) or
// keeps them open as a container (details, template, pre), so whatever follows
// is swallowed rather than rendered. GFM's tagfilter extension lists nearly the
// same set for the same reason. plaintext is in it because it is worth naming:
// nothing closes it, so a message that opens one loses the rest regardless.
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
  // An indented code block runs through blank lines to the first line that is
  // not indented, so the state has to outlive the line that opened it.
  let indented = false;
  // A code span may cross a soft line break, so an unclosed run keeps masking
  // until its match arrives.
  let span = "";
  // Tag starts whose > has not arrived yet, and the quote a value is inside.
  // Once a tag is unfinished every < after it is inside that tag, but each one
  // becomes a tag start again as soon as the one before it is neutralised.
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
      // The rest of the line is live again, so blank what the block held and
      // keep scanning rather than skipping to the next line.
      const consumed = end + block.terminator.length;
      line = " ".repeat(consumed) + line.slice(consumed);
      block = null;
    }
    // A fence, an indent and a tag all mean different things inside a block
    // quote than at the top level, so read them against the quoted content.
    // Blanking rather than slicing keeps every column where it was.
    const marker = BLOCKQUOTE_PATTERN.exec(line)?.[0] ?? "";
    const content = line.slice(marker.length);
    // The quote ending ends the fence with it, so an unclosed fence inside one
    // never reaches the next turn.
    if (fence !== null && fence.quoted && marker === "") fence = null;
    const literal = open.some((name) => CONDITION_1_ELEMENTS.has(name));
    const [, indent = "", run, info = ""] =
      unfinished.length || literal ? [] : FENCE_LINE_PATTERN.exec(content) ?? [];
    if (fence !== null) {
      // A closer repeats the opener's character, is at least as long, and
      // carries no info string.
      if (run && run[0] === fence.run[0] && run.length >= fence.run.length && !info.trim()) {
        fence = null;
      }
      continue;
    }
    if (run && (run[0] === "~" || !info.includes("`"))) {
      fence = { run, indent, quoted: marker !== "" };
      continue;
    }
    // A backtick opener carrying a backtick in its info string is not a fence,
    // so the line is an ordinary paragraph and falls through: its tags are
    // live, and skipping it would leave whatever it opens unclosed.
    // Indented code, code spans and closed comments are literal text, so a <
    // or a <!-- inside one opens nothing. Indented code only starts after a
    // blank line; otherwise the line is a lazy continuation of the paragraph
    // above and its content is live.
    if (!unfinished.length && !literal) {
      indented = indented
        ? blankLine || INDENTED_CODE_PATTERN.test(content)
        : afterBlank && INDENTED_CODE_PATTERN.test(content);
      if (indented) continue;
    }
    // A span in progress swallows this line up to its closing run. A blank line
    // ends it, and so does an html block, which the parser sees first.
    if (span && (blankLine || HTML_BLOCK_START_PATTERN.test(content))) span = "";
    if (span) {
      const closer = new RegExp("(^|[^`])(" + span + ")(?!`)").exec(line);
      if (closer === null) continue;
      const consumed = (closer.index ?? 0) + closer[0].length;
      line = " ".repeat(consumed) + line.slice(consumed);
      span = "";
    }
    // Every substitution below is length preserving, so a column found here is
    // still the right column in the original line.
    let prose = line
      .replace(BLOCKQUOTE_PATTERN, (marker) => " ".repeat(marker.length))
      .replace(CODE_SPAN_PATTERN, (span: string, lead: string) =>
        lead + " ".repeat(span.length - lead.length),
      )
      .replace(IMAGE_DESCRIPTION_PATTERN, (alt) => " ".repeat(alt.length))
      .replace(LINK_DESTINATION_PATTERN, (link) => " ".repeat(link.length))
      .replace(CLOSED_LITERAL_PATTERN, (span) => " ".repeat(span.length));
    // The backslash is the markdown parser's, so it only holds where that
    // parser is reading inlines, not inside a raw text element.
    if (!literal) {
      prose = prose.replace(ESCAPED_LT_PATTERN, (backslashes) =>
        backslashes.length % 2 === 0 ? `${backslashes.slice(0, -1)} ` : backslashes,
      );
    }
    // Whatever run is left once the closed spans are masked opens a span that
    // runs on, so the rest of this line is code as well.
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
      // The > only ends the tag outside an attribute value, so an unbalanced
      // quote keeps reading and no closing tag can ever land.
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
      // Innermost matching opener, so </div></details> still closes the
      // details. A closer with no opener is inert in every parser, so it must
      // not license an opener later in the message.
      const index = open.lastIndexOf(tag);
      if (index !== -1) open.splice(index, 1);
    }
  }

  // An unfinished tag cannot be closed: the tokenizer is still reading its name
  // or an attribute value, so a later </script> is swallowed as more of the
  // same tag. Neutralising the < is the only repair.
  // Last first, so an earlier replacement does not move a later column.
  escapes.push(...unfinished);
  escapes.sort((a, b) => b.part - a.part || b.column - a.column);
  for (const { part, column } of escapes) {
    const line = parts[part] as string;
    parts[part] = `${line.slice(0, column)}&lt;${line.slice(column + 1)}`;
  }

  // Close on the body's own line ending. A renderer that ignores bare carriage
  // returns would read a \n-prefixed closer as a fresh fence instead.
  const eol = !text.includes("\n") && text.includes("\r") ? "\r" : "\n";
  let out = parts.join("");
  if (block !== null) out += block.ownLine ? `${eol}${block.terminator}` : block.terminator;
  // Indented back to the opener's column: a closer at column zero would end the
  // list the opener sits in, then open a block of its own.
  if (fence !== null && !fence.quoted) out += `${eol}${fence.indent}${fence.run}`;
  // Innermost first, so the closers nest the way the openers did, and each on
  // its own line after a blank one so it is a block rather than a lazy
  // continuation of the paragraph above.
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
  // A code span cannot hold a line break: a blank line ends it outright and
  // leaves the rest of the value loose as markdown.
  const value = raw.replace(/[\r\n]+/g, " ");
  const longestRun = [...value.matchAll(/`+/g)].reduce(
    (max, [run]) => Math.max(max, run.length),
    0,
  );
  const ticks = "`".repeat(longestRun + 1);
  // Padding keeps a leading or trailing backtick from closing the span, and
  // keeps an empty value from collapsing into one delimiter run. CommonMark
  // 6.1 also strips one space from each end of a span that has one at both,
  // unless it is all spaces, so a padded argument needs a spare pair or it
  // renders with its own whitespace missing.
  const stripped = value.startsWith(" ") && value.endsWith(" ") && value.trim() !== "";
  const pad = !value || value.startsWith("`") || value.endsWith("`") || stripped ? " " : "";
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
    if (LINE_BREAK_PATTERN.test(text))
      return [`**${escapedLabel}:**`, fence(text)];
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
