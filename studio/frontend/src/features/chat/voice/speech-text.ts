// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


const SPEECH_STRIP_RE =
  /[\p{Extended_Pictographic}\u{1F1E6}-\u{1F1FF}\u{20E3}\u{FE00}-\u{FE0F}\u{200D}]/gu;

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
    .replace(/`([^`\n]+)`/g, (_m, code: string) =>
      code.replace(/\(\s*\)/g, "").replace(/\[[^\]\n]*\]/g, " ").replace(/(\w)\.(?=[A-Za-z_])/g, "$1 dot ").replace(/_/g, " "))
    .replace(/(?<!\d)\/|\/(?!\d)/g, " ") //                          "and/or", "/s"; not 1/2
    .replace(SPEECH_STRIP_RE, "")
    .replace(/−(?=\d)/g, "-") //                                 a numeric minus is not a dash
    .replace(/(\d)[^\S\n]*[–—‒][^\S\n]*(?=\d)/g, "$1 to ")
    .replace(/[^\S\n]*[–―‒−][^\S\n]*/g, " ")
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
  // A TTS model handed bare punctuation voices it as noise.
  return speakable(out) ? out : "";
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
      return blankLine(line);
    }
    if (JSON_OPEN_RE.test(line)) {
      depth = Math.max(0, bracketDepth(line));
      return blankLine(line);
    }
    if (JSON_KEY_LINE_RE.test(line)) return blankLine(line);
    if (i === lines.length - 1 && JSON_KEY_PREFIX_RE.test(line)) return blankLine(line);
    return line;
  };
}

// Every rule only looks backwards or drops to the end, so a streamed prefix splits as the finished reply does.
// Blanked, never cut: offsets must still point into the reply.
const blankAs = (first: string) => (region: string) =>
  region ? first + " ".repeat(region.length - 1) : "";
const blankLine = (line: string) => " ".repeat(line.length);

export function speakableReply(text: string): string {
  return text
    .replace(/<(think|thinking|reasoning|tool_call)>[\s\S]*?(<\/\1>|$)/gi, blankAs(" "))
    .replace(/<\|channel\|?>[\s\S]*?(<\|?channel\|>|$)/g, blankAs(" "))
    .replace(/```[\s\S]*?(```|$)/g, blankAs("\n"))
    .split("\n")
    .map(dropJsonLines())
    .join("\n")
    .replace(/^[ \t]*(?:\d{1,2}[.)]|[-*+•])(?=[ \t]|$)[ \t]*/gm, blankLine)
    .split("\n")
    .map((line) => (speakable(line) ? line : blankLine(line)))
    .join("\n");
}

// A trailing period on these ends an abbreviation, not a sentence ("Dr.").
const SPEECH_ABBREVIATIONS = new Set([
  "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "eg",
  "ie", "approx", "dept", "est", "fig", "no", "vol", "inc", "ltd", "co",
  "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
  "nov", "dec",
]);

// "no" usually ends a sentence: "I said no. Then".
const SENTENCE_FINAL_ABBREVIATIONS = new Set(["etc", "inc", "ltd", "co", "jr", "sr", "no"]);

const speakable = (s: string) => /[\p{L}\p{N}]/u.test(s);

/** Offsets into the reply, not into text. */
export interface SpokenSentence {
  text: string;
  start: number;
  end: number;
}

function sentenceSpans(text: string): SpokenSentence[] {
  // A sentence ends at .!? (plus closing quotes) only before whitespace or the end, so "3.5",
  // ".env", "./train.py" and "D.C." stay whole; a line without one ends at the newline.
  const body = text.trimEnd();
  const raw = [...body.matchAll(/\S[^\n]*?(?:[.!?]+["'”’»)\]]*(?=\s|$)|(?=\n|$))/g)].map((m) => ({
    text: m[0].replace(/[^\S\n]+/g, " "),
    start: m.index,
    end: m.index + m[0].length,
  }));
  if (raw.length === 0) return [{ text, start: 0, end: text.length }];
  const out: SpokenSentence[] = [];
  let lead: SpokenSentence | null = null;
  const join = (into: SpokenSentence, next: SpokenSentence) => {
    into.text = `${into.text} ${next.text}`;
    into.end = next.end;
  };
  for (const piece of raw) {
    const part: SpokenSentence = lead ? { ...lead } : { ...piece };
    if (lead) {
      join(part, piece);
      lead = null;
    }
    const prev = out[out.length - 1];
    // Punctuation alone ("...", "--") is no request of its own.
    if (!speakable(part.text)) {
      if (prev === undefined) lead = part;
      else join(prev, part);
      continue;
    }
    if (prev !== undefined) {
      const tail = prev.text.replace(/["'”’»)\]]+$/, "");
      const word = /([A-Za-z]+)\.$/.exec(tail)?.[1]?.toLowerCase();
      // "J. Smith", "John F. Kennedy" are names; "pick option A. Then" is a boundary.
      const nameInitial =
        /(?:^|(?:^|\s)(?:[A-Z][a-z]+|[A-Z]\.)\s)[A-Z]\.$/.test(tail) && /^[A-Z][a-z]/.test(part.text);
      const initial = /(^|\s)[A-Za-z]\.$/.test(tail) && /^[A-Za-z]\./.test(part.text);
      // "etc. Then I left": these also end sentences, so only a lowercase word or digit continues one.
      const abbrev =
        word !== undefined &&
        SPEECH_ABBREVIATIONS.has(word) &&
        (!SENTENCE_FINAL_ABBREVIATIONS.has(word) || /^[a-z\d]/.test(part.text));
      // "The U.S. is large", "e.g. the docs": a lowercase word after an initialism continues it.
      const initialism = /(?:^|\s)(?:[A-Za-z]\.){2,}$/.test(tail) && /^[a-z]/.test(part.text);
      if (nameInitial || initial || abbrev || initialism) {
        join(prev, part);
        continue;
      }
    }
    out.push(part);
  }
  if (lead) out.push(lead);
  return out;
}

export function splitIntoSentences(text: string): string[] {
  return sentenceSpans(text).map((s) => s.text);
}

export function splitSpoken(text: string): SpokenSentence[] {
  const spans = sentenceSpans(speakableReply(text));
  return spans.length === 1 && !/\S/.test(spans[0].text) ? [{ text: "", start: 0, end: 0 }] : spans;
}

/**
 * Sentences of a reply still streaming that can no longer change. The last part may still grow
 * ("Dr." before "Dr. Smith", "about 2." before "2.5"), and whether it joins the one before turns
 * on its first two characters, so it stays partial until endStream.
 */
export function splitStreamingSpans(raw: string): { complete: SpokenSentence[]; partial: string } {
  const parts = splitSpoken(raw);
  const tail = parts[parts.length - 1]?.text ?? "";
  const settled = Math.max(0, parts.length - (tail.length >= 2 ? 1 : 2));
  return {
    complete: parts.slice(0, settled),
    partial: parts
      .slice(settled)
      .map((s) => s.text)
      .join(" "),
  };
}

export function splitStreaming(raw: string): { complete: string[]; partial: string } {
  const { complete, partial } = splitStreamingSpans(raw);
  return { complete: complete.map((s) => s.text), partial };
}
