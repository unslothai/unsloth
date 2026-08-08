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
// A code span is literal text, so a <!-- inside one opens nothing.
const CODE_SPAN_PATTERN = /(`+)[^`]*\1/g;
// Four spaces of indentation is an indented code block, so a fence or a raw
// html block only starts within the first three columns. Splitting the
// indentation off once keeps the opener's container context, which the
// synthesized closer has to be written back into.
// [\s\S] rather than .: a line separator is an ordinary character to a
// markdown parser, and . would drop the whole line here.
const INDENT_PATTERN = /^( {0,3})([\s\S]*)$/;
// Same [\s\S], and for the same reason: U+2028 in an info string is an
// ordinary character to a markdown parser but a line terminator to a
// JavaScript dot, so . misses the opener and leaves the fence unrepaired.
const FENCE_PATTERN = /^(`{3,}|~{3,})([\s\S]*)$/;
const EOL_PATTERN = /\r\n|[\r\n]/g;
const BLANK_LINE_PATTERN = /^[ \t]*$/;
const INDENTED_CODE_PATTERN = /^(?: {4}|\t)/;
// A list marker or a block quote owns the indentation that follows it, so a
// four-space line inside one is that container's content and not a code block.
// Whether the message has such a marker at all is the test, because reading
// prose as code would stop scanning text a viewer really does render as html,
// while reading code as prose only adds a repair the message did not need.
const CONTAINER_MARKER_PATTERN =
  /^ {0,3}(?:[-*+](?:[ \t]|$)|\d{1,9}[.)](?:[ \t]|$)|>)/;
// The html tokenizer ends a tag name at whitespace, a solidus or >, so
// <details/> and a bare <details at the end of a line are both this tag.
const TAG_PATTERN = /<(\/?)([A-Za-z][^\s/>]*)/g;
// The renderer strips block quote markers, so their > is not a > the tokenizer
// ever sees. Left in, it would close a tag the browser still has open and hide
// the closer the message went on to write.
const BLOCK_QUOTE_PREFIX_PATTERN = /^ {0,3}(?:> ?)+/;
const NON_BREAK_PATTERN = /[^\r\n]/g;
// Everything that begins a tag or a bogus comment, and so everything that
// needs a > the message may never write.
const TAG_OPENER_PATTERN = /<[A-Za-z!/?]/g;

// Elements whose content the html tokenizer reads as text rather than markup,
// so nothing but their own end tag closes them (WHATWG 13.2.5, the RAWTEXT,
// RCDATA and script data states). CommonMark start condition 1 ends its block
// at whichever of pre, script, style and textarea closes first, which is a
// different answer, and the element is the one that hides the next turn.
const RAW_TEXT_ELEMENTS: ReadonlySet<string> = new Set([
  "script",
  "style",
  "textarea",
  "title",
  "xmp",
  "iframe",
  "noembed",
  "noframes",
]);
// pre is ordinary content to the tokenizer, but CommonMark condition 1 still
// runs its markdown block to the end of the document; details is condition 6,
// whose block ends at the blank line between two turns while the element does
// not. Both are closed rather than escaped, because both are things the chat
// renders and a transcript has to keep.
const CLOSED_ELEMENTS: ReadonlySet<string> = new Set([
  ...RAW_TEXT_ELEMENTS,
  "pre",
  "details",
]);
// The four whose markdown block also runs to the end of the document, so
// nothing inside one is a fence or an indented code block either.
const CONDITION_1_ELEMENTS: ReadonlySet<string> = new Set([
  "pre",
  "script",
  "style",
  "textarea",
]);
// CommonMark start condition 1 (4.6), which is a separate question from which
// element the browser has open: the block ends at the first line holding any
// of the four end tags, and the spec says outright that it "need not match the
// start tag". A message that ends inside one turns every later role heading
// into raw text even where the element itself is closed.
const HTML_BLOCK_START_PATTERN = /^<(pre|script|style|textarea)(?=[ \t>]|$)/i;
const HTML_BLOCK_END_PATTERN = /<\/(?:pre|script|style|textarea)>/i;

type OpenElement = { readonly name: string; readonly indent: string };

// Only this element's own end tag closes it, and the tokenizer ends the name
// at whitespace, a solidus or >, so </scriptx> is not it.
function endTagIndex(line: string, name: string, from: number): number {
  const pattern = new RegExp(`</${name}(?=[\\s/>]|$)`, "gi");
  pattern.lastIndex = from;
  return pattern.exec(line)?.index ?? -1;
}

// Everything a message can leave open that would run on into the next turn.
type OpenState = {
  readonly escapes: readonly number[];
  readonly open: readonly OpenElement[];
  readonly comment: boolean;
  readonly fence: OpenElement | null;
  readonly block: OpenElement | null;
  readonly bogus: OpenElement | null;
};

// One pass over a message, tracking both machines that decide whether it can
// hide the turn after it.
//
// CommonMark decides where a raw html BLOCK ends; the browser decides where
// the ELEMENT ends; and the two disagree. <script> ... </pre> ends the block
// and not the element (4.6 condition 1 says the end tag "need not match the
// start tag", WHATWG 13.2.5 wants the appropriate one). A <script> written
// after prose opens the element and no block at all. A message that stops
// inside an unfinished tag reads whatever is written after it as attributes,
// closer included. So the element is tracked by name, wherever it was opened,
// and the block is tracked beside it.
function scanOpenBlocks(text: string): OpenState {
  const escapes: number[] = [];
  const open: OpenElement[] = [];
  let raw: string | null = null;
  let comment = false;
  let fence: OpenElement | null = null;
  let block: OpenElement | null = null;
  // A tag runs to its >, which can be lines away, so where it started and what
  // it is are carried across them, as is the quote an attribute value is
  // inside: a > within one is part of the value, not the end of the tag.
  let tag: {
    readonly at: number;
    readonly name: string;
    readonly closing: boolean;
    // Set on the end tag that took the tokenizer out of a raw text element:
    // inside one a &lt; is the literal four characters, so an unfinished one
    // there has to be terminated rather than neutralised.
    readonly fromRaw: boolean;
    readonly indent: string;
    readonly close: string;
  } | null = null;
  let quote = "";
  let indentedCode = false;
  let blankBefore = true;
  let container = false;
  // The message with every literal region blanked to spaces: what a renderer
  // will actually hand to the browser as markup, indexed exactly like the
  // message itself.
  let live = "";

  const closeTo = (name: string): void => {
    for (let index = open.length - 1; index >= 0; index -= 1) {
      if (open[index]?.name === name) {
        open.length = index;
        return;
      }
    }
    // An end tag with no open element is inert in every html parser, so it
    // must not license an opener later in the message.
  };

  // Walked the way a parser does. Splitting on \n alone hides a fence opened
  // in a body that uses bare carriage returns.
  EOL_PATTERN.lastIndex = 0;
  let lineStart = 0;
  for (;;) {
    const lineBreak = EOL_PATTERN.exec(text);
    const line = text.slice(
      lineStart,
      lineBreak ? lineBreak.index : text.length,
    );
    const [, indent = "", rest = ""] = INDENT_PATTERN.exec(line) ?? [];
    const blankLine = BLANK_LINE_PATTERN.test(line);
    let literal = false;

    // Block structure, suspended only while the markdown parser itself is
    // inside something that stops it reading markdown. A tag it has not
    // finished reading is a state the browser has and it does not: a <script>
    // on the next line still opens a block for it, and a fence still opens a
    // fence, so neither may be suspended on that account.
    const insideRawBlock = comment || block !== null;
    if (!insideRawBlock) {
      if (fence !== null) {
        literal = true;
        const [, run, info] = FENCE_PATTERN.exec(rest) ?? [];
        // A closer repeats the opener's character, is at least as long, and
        // carries no info string.
        if (
          run &&
          run[0] === fence.name[0] &&
          run.length >= fence.name.length &&
          !info?.trim()
        ) {
          fence = null;
        }
      } else {
        // An indented code block runs until a non-blank line that is not
        // itself indented.
        if (indentedCode && !blankLine && !INDENTED_CODE_PATTERN.test(line)) {
          indentedCode = false;
        }
        const [, run, info] = FENCE_PATTERN.exec(rest) ?? [];
        if (indentedCode) {
          literal = true;
        } else if (run && (run[0] === "~" || !info?.includes("`"))) {
          // A backtick opener cannot carry a backtick in its info string; that
          // line is a paragraph, and treating it as a fence would open a real
          // one.
          fence = { name: run, indent };
          literal = true;
        } else if (
          !container &&
          blankBefore &&
          INDENTED_CODE_PATTERN.test(line)
        ) {
          // An indented code block cannot interrupt a paragraph (CommonMark
          // 4.4), so it only starts after a blank line.
          indentedCode = true;
          literal = true;
        }
      }
    }

    // Code spans, block quote markers and anything markdown reads as literal
    // are blanked rather than removed, so an index into what is scanned is
    // still an index into the message. Annotated because raw is assigned from
    // this scan, which would otherwise make the inference circular.
    const prose: string =
      literal && raw === null
        ? line.replace(NON_BREAK_PATTERN, " ")
        : line
            .replace(CODE_SPAN_PATTERN, (span) => " ".repeat(span.length))
            .replace(BLOCK_QUOTE_PREFIX_PATTERN, (marker) =>
              " ".repeat(marker.length),
            );
    live += prose;
    // Raw text is scanned even where markdown calls the line literal: a fence
    // inside an <xmp> is a fence to the parser and text to the tokenizer, and
    // the end tag hiding in it is the only thing that closes the element.
    if (!literal || raw !== null) {
      let index = 0;
      for (;;) {
        if (tag !== null) {
          let end = index;
          while (end < prose.length) {
            const character = prose[end] as string;
            if (quote) {
              if (character === quote) quote = "";
            } else if (character === '"' || character === "'") {
              quote = character;
            } else if (character === ">") {
              break;
            }
            end += 1;
          }
          // The tag is unfinished: it continues on the next line, or, if there
          // is none, it is still open when the message ends.
          if (end >= prose.length) break;
          if (tag.name === "") {
            // A bogus comment opened nothing.
          } else if (tag.closing) {
            if (raw === null || raw === tag.name) {
              raw = null;
              closeTo(tag.name);
            }
          } else if (raw === null) {
            if (tag.name === "plaintext") {
              // No end tag exists for it in any parser: the rest of the
              // document is its text, so the opener itself has to go.
              escapes.push(tag.at);
            } else if (CLOSED_ELEMENTS.has(tag.name)) {
              open.push({ name: tag.name, indent: tag.indent });
              if (RAW_TEXT_ELEMENTS.has(tag.name)) raw = tag.name;
            }
          }
          tag = null;
          index = end + 1;
          continue;
        }
        if (comment) {
          const end = line.indexOf("-->", index);
          if (end < 0) break;
          comment = false;
          index = end + 3;
          continue;
        }
        if (raw !== null) {
          // Inside a condition 1 block markdown parses no inlines, so the
          // backticks are literal and an end tag between them is a real one.
          // Opened from inline html instead, the same code span is rendered as
          // an escaped <code>, and the browser never sees a closer at all.
          const end = endTagIndex(block !== null ? line : prose, raw, index);
          if (end < 0) break;
          // The raw text ends here, but the end tag itself still has to reach
          // its >, and everything before that is eaten as its attributes, so
          // it goes through the same machinery as any other tag.
          index = end + 2 + raw.length;
          tag = {
            at: lineStart + end,
            name: raw,
            closing: true,
            fromRaw: true,
            indent,
            close: ">",
          };
          quote = "";
          raw = null;
          continue;
        }
        const start = prose.indexOf("<", index);
        if (start < 0) break;
        if (prose.startsWith("<!--", start)) {
          comment = true;
          index = start + 4;
          continue;
        }
        TAG_PATTERN.lastIndex = start;
        const match = TAG_PATTERN.exec(prose);
        if (match && match.index === start) {
          tag = {
            at: lineStart + start,
            name: (match[2] ?? "").toLowerCase(),
            closing: match[1] === "/",
            fromRaw: false,
            indent,
            close: "",
          };
          quote = "";
          index = start + match[0].length;
          continue;
        }
        // <!, <? and </ before anything that is not a tag name are a bogus
        // comment, which the tokenizer ends at the first > (CommonMark start
        // conditions 3, 4 and 5 end their blocks there too). Its terminator
        // carries that >, so an unfinished one can be closed rather than
        // escaped.
        const opener = prose.slice(start, start + 2);
        if (opener === "<!" || opener === "<?" || opener === "</") {
          tag = {
            at: lineStart + start,
            name: "",
            closing: false,
            fromRaw: false,
            indent,
            close: prose.startsWith("<![CDATA[", start)
              ? "]]>"
              : opener === "<?"
                ? "?>"
                : ">",
          };
          quote = "";
          index = start + 2;
          continue;
        }
        index = start + 1;
      }
    }

    if (block !== null) {
      if (HTML_BLOCK_END_PATTERN.test(line)) block = null;
    } else if (!insideRawBlock && !literal) {
      const start = HTML_BLOCK_START_PATTERN.exec(rest);
      // A line meeting both conditions is the whole block.
      if (start && !HTML_BLOCK_END_PATTERN.test(line)) {
        block = { name: (start[1] ?? "pre").toLowerCase(), indent };
      }
    }

    blankBefore = blankLine;
    if (!blankLine && CONTAINER_MARKER_PATTERN.test(line)) container = true;
    if (!lineBreak) break;
    live += lineBreak[0];
    lineStart = lineBreak.index + lineBreak[0].length;
  }

  // An unfinished tag consumes every closer written after it as attribute
  // text, and no terminator can be appended past a quote it left open, so the
  // opener is neutralised: escaping its < makes the rest of the message
  // ordinary text again. Two kinds are left alone. A bogus comment is
  // terminated below instead, because its terminator carries the > it is
  // waiting for. An unfinished end tag that took the tokenizer out of a raw
  // text element is left to the element closer, which supplies that > as well:
  // escaping there would produce the literal characters &lt;, since a raw text
  // element resolves no character references.
  if (tag !== null && tag.name !== "" && !tag.fromRaw) {
    // Nothing behind an unfinished tag can be finished either: it is unfinished
    // because no > follows it, and every tag, comment and bogus comment after
    // it needs that same >. So the whole tail is neutralised here, in one
    // linear pass, rather than one rescan per opener hiding inside it. This is
    // also what a renderer prints for that tail anyway, since CommonMark
    // passes through only complete tags and escapes the rest.
    TAG_OPENER_PATTERN.lastIndex = tag.at;
    for (
      let opener = TAG_OPENER_PATTERN.exec(live);
      opener !== null;
      opener = TAG_OPENER_PATTERN.exec(live)
    ) {
      escapes.push(opener.index);
    }
  }
  return {
    escapes,
    open,
    comment,
    fence,
    block,
    bogus:
      tag !== null && tag.name === ""
        ? { name: tag.close, indent: tag.indent }
        : null,
  };
}

// A message that ends mid-fence, mid-comment or inside an html element
// swallows everything after it, including the next role heading, so each turn
// closes what it opened. What can be closed gets its own closer; what cannot
// -- an unfinished tag, or <plaintext>, which no parser ever closes -- has its
// opening < escaped instead, so it opens nothing and still reads back as the
// text the message contained. That is also what GFM's own tagfilter extension
// does with this set of tags, and for the same stated reason: they change how
// the html after them is interpreted.
function closeOpenBlocks(text: string): string {
  let out = text;
  let state = scanOpenBlocks(out);
  // Neutralising an opener hands back the text the tokenizer had swallowed
  // into it, which can reveal another opener that was hiding inside, so the
  // scan repeats until a pass finds nothing left to neutralise. It terminates:
  // every pass removes at least one < and no pass adds one.
  while (state.escapes.length > 0) {
    // Rebuilt once rather than spliced per position, so a message carrying
    // thousands of them costs one pass over it rather than one each.
    const at = [...new Set(state.escapes)].sort(
      (first, second) => first - second,
    );
    const parts: string[] = [];
    let read = 0;
    for (const index of at) {
      parts.push(out.slice(read, index), "&lt;");
      read = index + 1;
    }
    parts.push(out.slice(read));
    out = parts.join("");
    state = scanOpenBlocks(out);
  }
  const { open, comment, fence, block, bogus } = state;
  if (comment) out += "-->";
  // Close on the body's own line ending. A renderer that ignores bare carriage
  // returns would read a \n-prefixed closer as a fresh fence instead.
  const eol = !text.includes("\n") && text.includes("\r") ? "\r" : "\n";
  // Before every element closer: an unfinished bogus comment would otherwise
  // swallow them all.
  if (bogus !== null) out += `${eol}${bogus.indent}${bogus.name}`;
  // Indented back to the opener's column: a closer written at column zero ends
  // the list the opener sits in, and then opens a block of its own instead.
  if (fence !== null) out += `${eol}${fence.indent}${fence.name}`;
  // Any of the four end tags satisfies condition 1, so an element closer
  // written below already ends the block; only a block still open without one
  // needs a closer of its own.
  if (
    block !== null &&
    !open.some((element) => CONDITION_1_ELEMENTS.has(element.name))
  ) {
    out += `${eol}${block.indent}</${block.name}>`;
  }
  // Innermost first, so the closers nest the way the openers did. details is
  // written last and after a blank line: its closer has to sit outside every
  // other construct repaired above, and at column zero as its own html block
  // rather than as a lazy continuation of the paragraph it follows.
  for (let index = open.length - 1; index >= 0; index -= 1) {
    const element = open[index] as OpenElement;
    out +=
      element.name === "details"
        ? `${eol}${eol}</details>`
        : `${eol}${element.indent}</${element.name}>`;
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
