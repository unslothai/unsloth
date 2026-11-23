// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  TRANSLATE_CHUNK_CHARS,
  restoreChunkWhitespace,
  splitForTranslation,
} from "./translate-text.ts";

export type SubtitleFormat = "srt" | "vtt";

type SubtitleCue = {
  /** The optional cue number or identifier and the timing line, as written. */
  head: string;
  newline: string;
  text: string;
};

export type SubtitleDocument = {
  format: SubtitleFormat;
  cues: SubtitleCue[];
  /** Verbatim text, or the index of a cue. */
  pieces: Array<string | number>;
  eol: string;
};

const TIME = String.raw`(?:\d+:)?\d{1,2}:\d{2}[,.]\d{1,3}`;
const TIMING = new RegExp(`^[ \\t]*${TIME}[ \\t]*-->[ \\t]*${TIME}`);
const VTT_HEADER = /^WEBVTT(?:[ \t]|\r?\n|$)/;

/** Null for anything that is not a whole SRT or WebVTT file; a clipped last SRT block is kept as is. */
export function parseSubtitles(source: string): SubtitleDocument | null {
  const lead = /^\s*/.exec(source)?.[0] ?? "";
  const body = source.slice(lead.length);
  const trail = /\s*$/.exec(body)?.[0] ?? "";
  const content = body.slice(0, body.length - trail.length);
  if (!content) return null;
  const format: SubtitleFormat = VTT_HEADER.test(content) ? "vtt" : "srt";
  const parts = content.split(/(\r?\n(?:[ \t]*\r?\n)+)/);
  const doc: SubtitleDocument = {
    format,
    cues: [],
    pieces: [lead],
    eol: source.includes("\r\n") ? "\r\n" : "\n",
  };
  for (let i = 0; i < parts.length; i += 2) {
    const block = parts[i];
    const lines = block.split(/(\r?\n)/);
    const timing = TIMING.test(lines[0])
      ? 0
      : lines.length > 2 &&
          TIMING.test(lines[2]) &&
          (format === "vtt"
            ? !lines[0].includes("-->")
            : /^\d+$/.test(lines[0].trim()))
        ? 2
        : -1;
    const header = format === "vtt" && i === 0;
    if (timing < 0 || header) {
      const kept = header || format === "vtt" || i === parts.length - 1;
      if (!kept) return null;
      doc.pieces.push(block);
    } else {
      doc.pieces.push(doc.cues.length);
      doc.cues.push({
        head: lines.slice(0, timing + 1).join(""),
        newline: lines[timing + 1] ?? doc.eol,
        text: lines.slice(timing + 2).join(""),
      });
    }
    if (parts[i + 1] !== undefined) doc.pieces.push(parts[i + 1]);
  }
  doc.pieces.push(trail);
  return doc.cues.length > 0 ? doc : null;
}

/** `texts[i]` replaces cue i's text; with `done`, the file stops after that many cues. */
export function rebuildSubtitles(
  doc: SubtitleDocument,
  texts: ReadonlyArray<string | null | undefined>,
  done?: number,
): string {
  let out = "";
  for (const piece of doc.pieces) {
    if (typeof piece === "string") {
      out += piece;
      continue;
    }
    if (done !== undefined && piece >= done) return out.trimEnd();
    const cue = doc.cues[piece];
    const text = texts[piece] ?? cue.text;
    out += text ? cue.head + cue.newline + text : cue.head;
  }
  return out;
}

/** Blank lines would end the cue early, so a translated cue keeps none. */
export function cueText(translated: string, eol: string): string {
  return translated
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .join(eol);
}

/** Consecutive cue indexes with text, each group within the passage size. */
export function subtitleBatches(
  texts: readonly string[],
  limit: number = TRANSLATE_CHUNK_CHARS,
): number[][] {
  const batches: number[][] = [];
  let current: number[] = [];
  let size = 0;
  texts.forEach((text, index) => {
    if (!text.trim()) return;
    const cost = text.length + 6;
    if (current.length > 0 && size + cost > limit) {
      batches.push(current);
      current = [];
      size = 0;
    }
    current.push(index);
    size += cost;
  });
  if (current.length > 0) batches.push(current);
  return batches;
}

export function numberedBatch(texts: readonly string[]): string {
  return texts
    .map((text, index) => `${index + 1}. ${cueText(text, "\n")}`)
    .join("\n");
}

const MARKER = /^\s*(\d+)\s*[.．、:：)]\s*/;

/** The cue texts of a numbered reply, or null unless it numbers exactly `count` items in order. */
export function splitNumberedBatch(
  reply: string,
  count: number,
): string[] | null {
  const items: string[][] = [];
  for (const line of reply.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const marker = MARKER.exec(line);
    if (marker) {
      if (Number(marker[1]) !== items.length + 1) return null;
      items.push([line.slice(marker[0].length)]);
    } else if (items.length === 0) {
      return null;
    } else {
      items[items.length - 1].push(line);
    }
  }
  if (items.length !== count) return null;
  const texts = items.map((lines) => lines.join("\n"));
  return texts.some((text) => !text.trim()) ? null : texts;
}

type BatchOptions = {
  signal: AbortSignal;
  onPhase: (phase: {
    kind: "translating";
    part: number;
    parts: number;
  }) => void;
};

/** Texts go out in numbered batches; a batch whose reply does not number every text goes again one by one.
 *  `onBatch` sees the translations so far (null = not yet) and how many texts lead up to the batch's end. */
export async function translateTexts(
  sources: readonly string[],
  complete: (source: string) => Promise<string>,
  options: BatchOptions & {
    eol?: string;
    onBatch?: (texts: ReadonlyArray<string | null>, done: number) => void;
  },
): Promise<string[]> {
  const { signal, onPhase, onBatch } = options;
  const eol = options.eol ?? "\n";
  const texts: Array<string | null> = sources.map(() => null);
  const batches = subtitleBatches(sources);
  const single = async (index: number) => {
    const translated = cueText(await complete(sources[index]), eol);
    texts[index] = translated || sources[index];
  };
  for (const [part, batch] of batches.entries()) {
    signal.throwIfAborted();
    onPhase({ kind: "translating", part: part + 1, parts: batches.length });
    // A line that already reads like a marker would shift the numbering.
    const numbered = batch.some((i) =>
      sources[i].split(/\r?\n/).some((line) => MARKER.test(line)),
    );
    if (batch.length === 1 || numbered) {
      for (const index of batch) {
        signal.throwIfAborted();
        await single(index);
      }
    } else {
      const reply = splitNumberedBatch(
        await complete(numberedBatch(batch.map((i) => sources[i]))),
        batch.length,
      );
      if (reply) {
        batch.forEach((index, i) => {
          texts[index] = cueText(reply[i], eol);
        });
      } else {
        for (const index of batch) {
          signal.throwIfAborted();
          await single(index);
        }
      }
    }
    onBatch?.(texts, batch[batch.length - 1] + 1);
  }
  return texts.map((text, index) => text ?? sources[index]);
}

/** Cue texts go out in numbered batches, keeping every cue number and timing. */
export async function translateSubtitles(
  doc: SubtitleDocument,
  complete: (source: string) => Promise<string>,
  options: BatchOptions & { onOutput: (output: string) => void },
): Promise<string> {
  const texts = await translateTexts(
    doc.cues.map((cue) => cue.text),
    complete,
    {
      signal: options.signal,
      onPhase: options.onPhase,
      eol: doc.eol,
      onBatch: (partial, done) =>
        options.onOutput(rebuildSubtitles(doc, partial, done)),
    },
  );
  return rebuildSubtitles(doc, texts);
}

const EXTENSION = /\.(srt|vtt|mp4|m4v|webm|mkv|mov)$/i;

/** `movie.srt` or `movie.mp4` translated into French saves as `movie.fr.srt`. */
export function subtitleFileName(
  sourceName: string | null,
  target: string,
  format: SubtitleFormat,
): string {
  const base = sourceName?.replace(EXTENSION, "").trim() || "subtitles";
  return `${base}.${target}.${format}`;
}

function srtTime(seconds: number): string {
  const ms = Math.round(Math.max(0, seconds) * 1000);
  const pad = (value: number, width = 2) => String(value).padStart(width, "0");
  return `${pad(Math.floor(ms / 3_600_000))}:${pad(Math.floor(ms / 60_000) % 60)}:${pad(Math.floor(ms / 1000) % 60)},${pad(ms % 1000, 3)}`;
}

/** An SRT file of timed transcript segments, or null when none has text. */
export function segmentsToSrt(
  segments: ReadonlyArray<{ start: number; end: number; text: string }>,
): string | null {
  const blocks = segments
    .map((segment) => ({ ...segment, text: cueText(segment.text, "\n") }))
    .filter((segment) => segment.text)
    .map(
      (segment, index) =>
        `${index + 1}\n${srtTime(segment.start)} --> ${srtTime(Math.max(segment.start, segment.end))}\n${segment.text}`,
    );
  return blocks.length > 0 ? `${blocks.join("\n\n")}\n` : null;
}

/** Like `translateTexts`, but a text longer than one passage is split like any text, never sent whole. */
export async function translateTextList(
  texts: readonly string[],
  complete: (source: string) => Promise<string>,
  options: Parameters<typeof translateTexts>[2],
): Promise<string[]> {
  const long = texts.map((text) => text.length > TRANSLATE_CHUNK_CHARS);
  const short = await translateTexts(
    texts.filter((_, i) => !long[i]),
    complete,
    options,
  );
  let next = 0;
  const out: string[] = [];
  for (const [i, text] of texts.entries()) {
    if (!long[i]) {
      out.push(short[next++]);
      continue;
    }
    let joined = "";
    for (const chunk of splitForTranslation(text)) {
      options.signal.throwIfAborted();
      joined += chunk.text
        ? restoreChunkWhitespace(chunk, await complete(chunk.text))
        : chunk.after;
    }
    out.push(joined.trim());
  }
  return out;
}
