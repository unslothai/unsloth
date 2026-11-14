// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


const SPEECH_STRIP_RE =
  /[\p{Extended_Pictographic}\u{1F1E6}-\u{1F1FF}\u{20E3}\u{FE00}-\u{FE0F}\u{200D}]/gu;

export function stripForSpeech(text: string): string {
  const out = text
    .replace(/\[([^\]\n]*)\]\([^)\s]*\)/g, "$1") //               [text](url) -> text
    .replace(/\b(?:https?:\/\/|www\.)\S+/gi, " ") //                 bare URLs -> drop
    .replace(/\\(?:times|cdot)(?![A-Za-z])/g, " times ")
    .replace(/\\(?:frac|left|right|mathrm|mathbf|text|displaystyle|begin|end|quad)(?![A-Za-z])/g, " ")
    .replace(/\\[()[\]]/g, " ")
    .replace(/\\([A-Za-z]+)/g, " $1 ")
    .replace(/\$(?!\d)/g, " ") //                                    math $, not "$5"
    .replace(/(?<!\d)\/|\/(?!\d)/g, " ") //                          "and/or", "/s"; not 1/2
    .replace(SPEECH_STRIP_RE, "")
    .replace(/−(?=\d)/g, "-") //                                 a numeric minus is not a dash
    .replace(/[^\S\n]*[—–―‒−][^\S\n]*/g, " ")
    .replace(/[^\S\n]*…[^\S\n]*/g, " ")
    .replace(/\.{2,}/g, " ")
    .replace(/[‐‑]/g, "-")
    .replace(/[*_`~^|#<>\\{}[\]]/g, " ")
    .replace(/[‘’‚‛]/g, "'")
    .replace(/[“”„‟]/g, '"')
    // Horizontal only: newlines are sentence boundaries for splitIntoSentences.
    .replace(/[^\S\n]+/g, " ")
    .replace(/ ([.,!?;:])/g, "$1")
    .replace(/\n{2,}/g, "\n")
    .trim();
  return /[\p{L}\p{N}]/u.test(out) ? out : "";
}

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

export function speakableReply(text: string): string {
  return text
    .replace(/<(think|thinking|reasoning|tool_call)>[\s\S]*?(<\/\1>|$)/gi, " ")
    .replace(/<\|channel\|?>[\s\S]*?(<\|?channel\|>|$)/g, " ")
    .replace(/```[\s\S]*?(```|$)/g, "\n")
    .split("\n")
    .map(dropJsonLines())
    .join("\n")
    .replace(/^[ \t]*(?:\d{1,2}[.)]|[-*+•])(?=[ \t]|$)[ \t]*/gm, "")
    .split("\n")
    .map((line) => (speakable(line) ? line : ""))
    .join("\n");
}

// A trailing period on these ends an abbreviation, not a sentence ("Dr.").
const SPEECH_ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "eg",
  "ie", "approx", "dept", "est", "fig", "no", "vol", "inc", "ltd", "co",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
  "nov", "dec",
]);

const SENTENCE_FINAL_ABBREVIATIONS = new Set(["etc", "inc", "ltd", "co", "jr", "sr"]);

const speakable = (s: string) => /[\p{L}\p{N}]/u.test(s);

export function sentenceMayContinue(sentence: string, textEndsWithSpace: boolean): boolean {
  if (!textEndsWithSpace) return true;
  const tail = sentence.replace(/["')\]]+$/, "");
  if (/(?:^|\s)(?:[A-Za-z]\.){2,}$/.test(tail)) return true;
  const word = /(?:^|\s)([A-Za-z]+)\.$/.exec(tail)?.[1];
  return word !== undefined && (word.length === 1 || SPEECH_ABBREVIATIONS.has(word.toLowerCase()));
}

export function splitIntoSentences(text: string): string[] {
  const norm = text
    .replace(/[^\S\n]+/g, " ")
    .replace(/\n{2,}/g, "\n")
    .trim();
  // A sentence ends at .!? (plus closing quotes) only before whitespace or the end, so "3.5",
  // ".env", "./train.py" and "D.C." stay whole; a line without one ends at the newline.
  const raw = norm.match(/\S[^\n]*?(?:[.!?]+["'”’»)\]]*(?=\s|$)|(?=\n|$))/g);
  if (!raw) return [text];
  const out: string[] = [];
  let lead = "";
  for (let part of raw) {
    if (lead) {
      part = `${lead} ${part}`;
      lead = "";
    }
    const prev = out[out.length - 1];
    // Punctuation alone ("...", "--") is no request of its own.
    if (!speakable(part)) {
      if (prev === undefined) lead = part;
      else out[out.length - 1] = `${prev} ${part}`;
      continue;
    }
    if (prev !== undefined) {
      const tail = prev.replace(/["'”’»)\]]+$/, "");
      const word = /([A-Za-z]+)\.$/.exec(tail)?.[1]?.toLowerCase();
      // "J. Smith", "John F. Kennedy" are names; "pick option A. Then" is a boundary.
      const nameInitial =
        /(?:^|(?:^|\s)(?:[A-Z][a-z]+|[A-Z]\.)\s)[A-Z]\.$/.test(tail) && /^[A-Z][a-z]/.test(part);
      const initial = /(^|\s)[A-Za-z]\.$/.test(tail) && /^[A-Za-z]\./.test(part);
      // "etc. Then I left": these also end sentences, so only a lowercase word or digit continues one.
      const abbrev =
        word !== undefined &&
        SPEECH_ABBREVIATIONS.has(word) &&
        (!SENTENCE_FINAL_ABBREVIATIONS.has(word) || /^[a-z\d]/.test(part));
      // "The U.S. is large", "e.g. the docs": a lowercase word after an initialism continues it.
      const initialism = /(?:^|\s)(?:[A-Za-z]\.){2,}$/.test(tail) && /^[a-z]/.test(part);
      if (nameInitial || initial || abbrev || initialism) {
        out[out.length - 1] = `${prev} ${part}`;
        continue;
      }
    }
    out.push(part);
  }
  if (lead) out.push(lead);
  return out;
}

export function splitStreaming(raw: string): {
  complete: string[];
  partial: string;
} {
  const text = speakableReply(raw);
  const parts = splitIntoSentences(text);
  if (parts.length === 0) return { complete: [], partial: "" };
  const last = parts[parts.length - 1] ?? "";
  if (/[.!?]["')\]]?\s*$/.test(text) && !sentenceMayContinue(last, /\s$/.test(text))) {
    return { complete: parts, partial: "" };
  }
  return { complete: parts.slice(0, -1), partial: parts[parts.length - 1] ?? "" };
}
