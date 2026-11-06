// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Turning assistant text into what the voice actually speaks, and into the chunks
// it speaks one at a time. Dependency-free: these are the only part of the voice
// path testable without a mic or a GPU.

const SPEECH_STRIP_RE =
  /[\p{Extended_Pictographic}\u{1F1E6}-\u{1F1FF}\u{20E3}\u{FE00}-\u{FE0F}\u{200D}]/gu;

// Normalize text for TTS: some characters derail Orpheus (like the colon it reads
// as a speaker tag) or get voiced literally by any TTS model. Em/en dashes and the
// single-char ellipsis are the main offenders (an em dash breaks the voice), and
// markdown/markup symbols get read out ("asterisk asterisk"). All of these are just
// dropped (replaced with a space, not a comma, so no phantom pauses are inserted);
// smart quotes are normalized. Speech ONLY -- the on-screen chat text keeps everything.
export function stripForSpeech(text: string): string {
  const out = text
    .replace(/\[([^\]\n]*)\]\([^)\s]*\)/g, "$1") //               [text](url) -> text
    .replace(/\b(?:https?:\/\/|www\.)\S+/gi, " ") //                 bare URLs -> drop
    // LaTeX: OmniVoice reads ")$." as a burst of noise, and "\frac" letter by letter.
    .replace(/\\(?:times|cdot)(?![A-Za-z])/g, " times ")
    .replace(/\\(?:frac|left|right|mathrm|mathbf|text|displaystyle|begin|end|quad)(?![A-Za-z])/g, " ")
    .replace(/\\[()[\]]/g, " ")
    .replace(/\\([A-Za-z]+)/g, " $1 ")
    .replace(/\$(?!\d)/g, " ") //                                    math $, not "$5"
    .replace(/(?<!\d)\/|\/(?!\d)/g, " ") //                          "and/or", "/s"; not 1/2
    .replace(SPEECH_STRIP_RE, "")
    .replace(/[^\S\n]*[—–―‒−][^\S\n]*/g, " ") // — – ― ‒ −  -> drop
    .replace(/[^\S\n]*…[^\S\n]*/g, " ") //                 …  -> drop
    .replace(/\.{2,}/g, " ") //                                 ... -> drop
    .replace(/[‐‑]/g, "-") //                          unicode hyphens -> ASCII
    .replace(/[*_`~^|#<>\\{}[\]]/g, " ") //                      markdown / markup -> space
    .replace(/[‘’‚‛]/g, "'") //             smart single quotes
    .replace(/[“”„‟]/g, '"') //             smart double quotes
    // Collapse horizontal whitespace only. Newlines are sentence boundaries to
    // splitIntoSentences, and flattening them here (as this used to) made a
    // multi-paragraph reply one enormous sentence: the loop then waited on a
    // single long synth instead of speaking the first line straight away.
    .replace(/[^\S\n]+/g, " ")
    .replace(/ ([.,!?;:])/g, "$1")
    .replace(/\n{2,}/g, "\n")
    .trim();
  // Nothing to say: a TTS model handed bare punctuation voices it as noise.
  return /[\p{L}\p{N}]/u.test(out) ? out : "";
}

// A tool call or JSON written as text: its keys and quotes are not words.
const JSON_OPEN_RE = /^[ \t]*(?:\{|\[[ \t]*(?:[{["]|$))/;
const JSON_KEY_LINE_RE =
  /^[ \t]*"[^"\n]+"[ \t]*:[ \t]*(?:"[^"\n]*"|-?\d[\d.eE+-]*|true|false|null|\{\}|\[\]|[{[])[ \t]*,?[ \t]*$/;
// What a key line can look like before its end has streamed in. Only the last line gets
// this: dropped while it may still be JSON, so it never queues a sentence it later loses.
const JSON_KEY_PREFIX_RE =
  /^[ \t]*"[^"\n]+"[ \t]*:[ \t]*(?:"[^"\n]*(?:"[ \t]*,?)?|[-\d.eE+]*,?|(?:t(?:r(?:ue?)?)?|f(?:a(?:l(?:se?)?)?)?|n(?:u(?:ll?)?)?),?|[{[])?[ \t]*$/;

function bracketDepth(line: string): number {
  let depth = 0;
  let inString = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inString) {
      if (c === "\\") i++;
      else if (c === '"') inString = false;
    } else if (c === '"') inString = true;
    else if (c === "{" || c === "[") depth++;
    else if (c === "}" || c === "]") depth--;
  }
  return depth;
}

/** Blanks JSON lines: a block from its opening line to its matching close (or the end,
 *  mid-stream), and lone `"key": value,` lines. */
function dropJsonLines(): (line: string, i: number, lines: string[]) => string {
  let depth = 0;
  return (line, i, lines) => {
    if (depth > 0) {
      depth += bracketDepth(line);
      return "";
    }
    if (JSON_OPEN_RE.test(line)) {
      depth = Math.max(0, bracketDepth(line));
      return "";
    }
    if (JSON_KEY_LINE_RE.test(line)) return "";
    if (i === lines.length - 1 && JSON_KEY_PREFIX_RE.test(line)) return "";
    return line;
  };
}

/**
 * The part of a reply that is meant to be heard, applied to the whole text before it is
 * split. Every rule only looks backwards or drops to the end, so a streamed prefix splits
 * the same way the finished reply does.
 */
export function speakableReply(text: string): string {
  return text
    .replace(/<(think|thinking|reasoning|tool_call)>[\s\S]*?(<\/\1>|$)/gi, " ")
    .replace(/<\|channel\|?>[\s\S]*?(<\|?channel\|>|$)/g, " ")
    .replace(/```[\s\S]*?(```|$)/g, "\n")
    .split("\n")
    .map(dropJsonLines())
    .join("\n")
    // List markers: "1." on its own reads as a one-word sentence.
    .replace(/^[ \t]*(?:\d{1,2}[.)]|[-*+•])(?=[ \t]|$)[ \t]*/gm, "");
}

// Words whose trailing period ends an abbreviation, not a sentence. Splitting
// there hands the codec a fragment ("Dr.") and starts the next clip mid-thought.
const SPEECH_ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "eg",
  "ie", "approx", "dept", "est", "fig", "no", "vol", "inc", "ltd", "co",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
  "nov", "dec",
]);

/** The streamed text stops right after this sentence's period, which may not end it yet:
 *  "about 2." becomes "about 2.5 inches", "Dr." becomes "Dr. Smith". */
export function sentenceMayContinue(sentence: string, textEndsWithSpace: boolean): boolean {
  if (!textEndsWithSpace) return true;
  const tail = sentence.replace(/["')\]]+$/, "");
  const word = /(?:^|\s)([A-Za-z]+)\.$/.exec(tail)?.[1];
  return word !== undefined && (word.length === 1 || SPEECH_ABBREVIATIONS.has(word.toLowerCase()));
}

export function splitIntoSentences(text: string): string[] {
  // Horizontal whitespace only: a newline is a boundary the regex below relies on.
  const norm = text
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n{2,}/g, "\n")
    .trim();
  const raw = norm.match(/[^.!?\n]*[.!?]+|\S[^.!?\n]*$|\S[^.!?\n]*(?=\n)/g);
  if (!raw) return [text];
  const out: string[] = [];
  let at = 0;
  for (const chunk of raw) {
    const start = norm.indexOf(chunk, at);
    at = start + chunk.length;
    const part = chunk.trim();
    // Whether a space or newline came before it: "2.5" has none, "in 2020. 30 people" has one.
    const spaced = /^\s/.test(chunk) || (start > 0 && /\s/.test(norm[start - 1] ?? ""));
    if (!part) continue;
    const prev = out[out.length - 1];
    if (prev !== undefined) {
      // Quotes and brackets only CLOSE when whitespace or the end follows one.
      // Followed by a letter it opens a quotation, which is a real boundary.
      const closer = /^["')\]]+(\s|$)/.test(part);
      const tail = prev.replace(/["')\]]+$/, "");
      const lastWord = /([A-Za-z]+)\.$/.exec(tail)?.[1];
      const decimal = /\d\.$/.test(tail) && !spaced && /^\d/.test(part);
      const initial = /(^|\s)[A-Za-z]\.$/.test(tail) && /^[A-Za-z]\./.test(part);
      const abbrev =
        lastWord !== undefined && SPEECH_ABBREVIATIONS.has(lastWord.toLowerCase());
      if (decimal || initial || abbrev || closer) {
        // "3.50" and "D.C." close up; "Dr. Smith" keeps the space it was written with.
        out[out.length - 1] = prev + (decimal || initial || closer ? "" : " ") + part;
        continue;
      }
    }
    out.push(part);
  }
  return out.filter(Boolean);
}

// While the LLM is still writing, only terminated sentences are safe to speak;
// the trailing chunk is the sentence in progress.
export function splitStreaming(raw: string): {
  complete: string[];
  partial: string;
} {
  const text = speakableReply(raw);
  const parts = splitIntoSentences(text);
  if (parts.length === 0) return { complete: [], partial: "" };
  // "about 2." may be "2.5" and "Dr." may be "Dr. Smith" mid-stream: speaking either as a
  // sentence drops what the split later merges into it and shifts every later index.
  const last = parts[parts.length - 1] ?? "";
  if (/[.!?]["')\]]?\s*$/.test(text) && !sentenceMayContinue(last, /\s$/.test(text))) {
    return { complete: parts, partial: "" };
  }
  return { complete: parts.slice(0, -1), partial: parts[parts.length - 1] ?? "" };
}
