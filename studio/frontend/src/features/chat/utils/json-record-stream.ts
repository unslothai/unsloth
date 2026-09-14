// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Text arriving from a file, with the byte count that produced it. */
export interface TextChunk {
  text: string;
  bytes: number;
}

/** A browser or desktop source that can be read incrementally. */
export interface ImportSource {
  name: string;
  size?: number;
  /** Decoded text and its source byte count for progress reporting. */
  chunks(): AsyncIterable<TextChunk>;
}

/** Decode byte ranges into text. `fatal` rejects invalid UTF-8 instead of substituting
 *  replacement characters, which the desktop path requires. */
export async function* decodeTextChunks(
  byteChunks: AsyncIterable<Uint8Array>,
  fatal = false,
): AsyncGenerator<TextChunk> {
  const decoder = new TextDecoder("utf-8", { fatal });
  for await (const bytes of byteChunks) {
    if (!bytes.byteLength) continue;
    // Preserve UTF-8 characters split across chunks.
    yield { text: decoder.decode(bytes, { stream: true }), bytes: bytes.byteLength };
  }
  const tail = decoder.decode();
  if (tail) yield { text: tail, bytes: 0 };
}

async function* fileBytes(file: File): AsyncGenerator<Uint8Array> {
  const reader = file.stream().getReader();
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) yield value;
    }
  } finally {
    reader.releaseLock();
  }
}

export function fileImportSource(file: File): ImportSource {
  return {
    name: file.name,
    size: file.size,
    // Browser imports stay lenient about invalid UTF-8, as `file.text()` was.
    chunks: () => decodeTextChunks(fileBytes(file)),
  };
}

/** Read a whole source. The bound is the bytes read, not the length of the decoded string:
 *  multibyte text is longer in bytes than in UTF-16 units. */
export async function readAllText(
  source: ImportSource,
  maxBytes: number,
  format: string,
): Promise<string> {
  let text = "";
  let bytes = 0;
  for await (const chunk of source.chunks()) {
    text += chunk.text;
    bytes += chunk.bytes;
    if (bytes > maxBytes) {
      throw new Error(
        `${source.name} is too large to import as ${format} (maximum ${Math.floor(maxBytes / 1024 / 1024)} MiB).`,
      );
    }
  }
  return text;
}

/** Stream top-level records from a JSON array or JSONL/NDJSON, including records formatted across multiple lines. */
export interface StreamJsonOptions {
  onBytes?: (bytes: number) => void;
  /** A framed record that does not parse. The stream continues past it. */
  onMalformed?: (text: string) => void;
}

interface Salvage {
  records: unknown[];
  damaged: string[];
}

/** A line-leading `{`/`[` that follows a finished value: the next record. */
const NEW_RECORD_LINE = /[^\s:,[{]\s*\n[[{]/;

/** Recover records after an unclosed JSONL row. Until line framing is proven, only unindented
 *  object or array lines qualify as record boundaries. */
function salvageLines(region: string, proven: boolean): Salvage {
  const records: unknown[] = [];
  const damaged: string[] = [];
  for (const line of region.split("\n")) {
    const text = line.trim();
    if (!text) continue;
    if (!proven && (/^\s/.test(line) || !/^[[{]/.test(text))) {
      damaged.push(text);
      continue;
    }
    try {
      records.push(JSON.parse(text));
    } catch {
      damaged.push(text);
    }
  }
  return { records, damaged };
}

export async function* streamJsonRecords(
  chunks: AsyncIterable<TextChunk>,
  options: StreamJsonOptions = {},
): AsyncGenerator<unknown> {
  const QUOTE = 34; // "
  const BACKSLASH = 92; // \
  const OPEN_BRACE = 123; // {
  const CLOSE_BRACE = 125; // }
  const OPEN_BRACKET = 91; // [
  const CLOSE_BRACKET = 93; // ]
  const SPACE = 32;
  const TAB = 9;
  const NEWLINE = 10;
  const RETURN = 13;

  let buffer = "";
  let scan = 0;
  let start = -1;
  let depth = 0;
  let inString = false;
  let escaped = false;
  let sawArrayStart = false;
  let sawArrayEnd = false;
  // True while every emitted record has occupied one line.
  let emittedRecord = false;
  let lineFramed = false;

  for await (const chunk of chunks) {
    options.onBytes?.(chunk.bytes);
    // Retain only the record still being read.
    const consumed = start >= 0 ? start : scan;
    try {
      buffer = consumed > 0 ? buffer.slice(consumed) + chunk.text : buffer + chunk.text;
    } catch (error) {
      // Records are dropped from the buffer as they are emitted, so the only way to reach the
      // engine's maximum string length is ONE record that long. Untranslated, that surfaces as a
      // bare "Invalid string length".
      if (error instanceof RangeError) {
        throw new RangeError(
          "One chat in this export is too large to read in a single piece. " +
            "Split the export and import the pieces separately.",
        );
      }
      throw error;
    }
    scan -= consumed;
    if (start >= 0) start = 0;

    // Recovery can replace the buffer and request another scan.
    let rescan = true;
    while (rescan) {
      rescan = false;

      while (scan < buffer.length) {
        const code = buffer.charCodeAt(scan);

        if (inString) {
          if (escaped) escaped = false;
          else if (code === BACKSLASH) escaped = true;
          else if (code === QUOTE) inString = false;
          scan++;
          continue;
        }

        if (start < 0) {
          // Past the array's own close, only whitespace belongs to the file. A record after it means this
          // is not the single export it claims to be.
          if (sawArrayEnd) {
            if (code !== SPACE && code !== TAB && code !== NEWLINE && code !== RETURN) {
              throw new SyntaxError(
                "The export continues after its closing bracket, so it is not one JSON array.",
              );
            }
            scan++;
            continue;
          }
          // Between records: the array's own brackets, separators and whitespace.
          if (code === OPEN_BRACKET && !sawArrayStart) {
            sawArrayStart = true;
            scan++;
            continue;
          }
          if (code !== OPEN_BRACE && code !== OPEN_BRACKET) {
            // The array's own closing bracket: the file is complete from here on.
            if (code === CLOSE_BRACKET && sawArrayStart) sawArrayEnd = true;
            // Commas, newlines, and any stray scalar between records: nothing to import.
            scan++;
            continue;
          }
          start = scan;
          depth = 0;
        }

        if (code === QUOTE) inString = true;
        else if (code === OPEN_BRACE || code === OPEN_BRACKET) depth++;
        else if (code === CLOSE_BRACE || code === CLOSE_BRACKET) depth--;
        scan++;

        if (depth === 0) {
          const text = buffer.slice(start, scan);
          start = -1;
          // Framing is intact, so a parse failure affects only this record.
          let record: unknown;
          try {
            record = JSON.parse(text);
          } catch {
            options.onMalformed?.(text);
            continue;
          }
          lineFramed = emittedRecord
            ? lineFramed && !text.includes("\n")
            : !text.includes("\n");
          emittedRecord = true;
          yield record;
        }
      }

      // A pending record is damaged once a later line starts a record of its own: a line-leading `{`
      // or `[` which JSON could not continue the pending text with. Deciding it that way keeps the
      // outcome independent of where the chunk boundary fell, and recovering here rather than only
      // at end of input keeps a file whose first row is broken from buffering whole.
      if (!sawArrayStart && start >= 0) {
        const lastNewline = buffer.lastIndexOf("\n");
        const boundary =
          lastNewline > start ? NEW_RECORD_LINE.exec(buffer.slice(start, lastNewline)) : null;
        if (boundary) {
          // Only the text before the boundary is damaged. The scanner resumes from the boundary itself,
          // so a pretty-printed record after a broken row is framed by its nesting rather than shredded.
          const at = start + boundary.index + boundary[0].length - 1;
          const damaged = buffer.slice(start, at).trim();
          if (damaged) options.onMalformed?.(damaged);
          buffer = buffer.slice(at);
          scan = 0;
          start = -1;
          depth = 0;
          inString = false;
          escaped = false;
          rescan = true;
        }
      }
    }
  }

  const tail = start >= 0 ? buffer.slice(start).trim() : "";
  if (tail) {
    if (sawArrayStart) {
      // Arrays have no line boundary on which to recover a truncated record. The failure is the same
      // one the closing-bracket check below reports, so it says so in the same words rather than
      // surfacing the engine's own "Unterminated string in JSON" to the user.
      let record: unknown;
      try {
        record = JSON.parse(tail);
      } catch {
        throw new SyntaxError(
          "The JSON array ends in the middle of a record, so the export is incomplete.",
        );
      }
      yield record;
    } else {
      // Recover complete JSONL rows after a broken one.
      const salvaged = salvageLines(tail, lineFramed);
      if (salvaged.records.length === 0) {
        // Report one truncated record instead of each of its lines.
        options.onMalformed?.(tail);
      } else {
        for (const text of salvaged.damaged) options.onMalformed?.(text);
        yield* salvaged.records;
      }
    }
  }

  // A download cut between two records leaves every record read so far intact, so the missing
  // bracket is the only sign that the rest never arrived.
  if (sawArrayStart && !sawArrayEnd) {
    throw new SyntaxError(
      "The JSON array ends before its closing bracket, so the export is incomplete.",
    );
  }
}
