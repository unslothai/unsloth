// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface TextChunk {
  text: string;
  bytes: number;
}

export interface ImportSource {
  name: string;
  size?: number;
  chunks(): AsyncIterable<TextChunk>;
}

const ENCODING_SAMPLE_BYTES = 1024;

function isJsonWhitespace(byte: number): boolean {
  return byte === 0x20 || byte === 0x09 || byte === 0x0a || byte === 0x0d;
}

/** Detect a BOM or the UTF-16 form of the first JSON object/array opener. */
function sniffJsonEncoding(head: Uint8Array, atEnd: boolean): string | undefined {
  if (head.length >= 2 && head[0] === 0xfe && head[1] === 0xff) return "utf-16be";
  if (head.length >= 2 && head[0] === 0xff && head[1] === 0xfe) {
    if (head.length < 4 && !atEnd) return undefined;
    // FF FE 00 00 is a UTF-32LE BOM, which TextDecoder does not support.
    return head[2] === 0 && head[3] === 0 ? "utf-8" : "utf-16le";
  }

  let littleEndian: boolean | undefined;
  for (let i = 0; i + 1 < head.length; i += 2) {
    const first = head[i];
    const second = head[i + 1];
    const byte = first || second;
    if ((first === 0) === (second === 0)) return "utf-8";
    if (littleEndian !== undefined && littleEndian !== (second === 0)) return "utf-8";
    littleEndian = second === 0;
    if (isJsonWhitespace(byte)) continue;
    if (byte === 0x7b || byte === 0x5b) {
      return littleEndian ? "utf-16le" : "utf-16be";
    }
    return "utf-8";
  }
  return atEnd || head.length === ENCODING_SAMPLE_BYTES ? "utf-8" : undefined;
}

/** The desktop importer uses fatal decoding to reject malformed text. */
export async function* decodeTextChunks(
  byteChunks: AsyncIterable<Uint8Array>,
  fatal = false,
): AsyncGenerator<TextChunk> {
  const head = new Uint8Array(ENCODING_SAMPLE_BYTES);
  let headBytes = 0;
  let decoder: TextDecoder | undefined;
  for await (const bytes of byteChunks) {
    if (!bytes.byteLength) continue;
    if (!decoder) {
      const size = Math.min(head.length - headBytes, bytes.byteLength);
      head.set(bytes.subarray(0, size), headBytes);
      headBytes += size;
      const sample = head.subarray(0, headBytes);
      const encoding = sniffJsonEncoding(sample, false);
      if (!encoding) continue;
      decoder = new TextDecoder(encoding, { fatal });
      const prefix = decoder.decode(sample, { stream: true });
      const suffix = decoder.decode(bytes.subarray(size), { stream: true });
      yield { text: prefix + suffix, bytes: headBytes + bytes.byteLength - size };
    } else {
      yield { text: decoder.decode(bytes, { stream: true }), bytes: bytes.byteLength };
    }
  }
  if (!decoder) {
    if (!headBytes) return;
    const sample = head.subarray(0, headBytes);
    decoder = new TextDecoder(sniffJsonEncoding(sample, true), { fatal });
    yield { text: decoder.decode(sample), bytes: headBytes };
    return;
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
    chunks: () => decodeTextChunks(fileBytes(file)),
  };
}

/** The bound is bytes read, not decoded string length. */
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
  onMalformed?: (text: string) => void;
}

interface Salvage {
  records: unknown[];
  damaged: string[];
}

const NEW_RECORD_LINE = /[^\s:,[{]\s*\n[[{]/;

/** Until line framing is proven, only unindented object or array lines are boundaries. */
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
  let emittedRecord = false;
  let lineFramed = false;

  for await (const chunk of chunks) {
    options.onBytes?.(chunk.bytes);
    const consumed = start >= 0 ? start : scan;
    try {
      buffer = consumed > 0 ? buffer.slice(consumed) + chunk.text : buffer + chunk.text;
    } catch (error) {
      // Only a single huge record can hit the engine's max string length; translate the RangeError.
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
          if (sawArrayEnd) {
            if (code !== SPACE && code !== TAB && code !== NEWLINE && code !== RETURN) {
              throw new SyntaxError(
                "The export continues after its closing bracket, so it is not one JSON array.",
              );
            }
            scan++;
            continue;
          }
          if (code === OPEN_BRACKET && !sawArrayStart) {
            sawArrayStart = true;
            scan++;
            continue;
          }
          if (code !== OPEN_BRACE && code !== OPEN_BRACKET) {
            if (code === CLOSE_BRACKET && sawArrayStart) sawArrayEnd = true;
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

      // A pending record is damaged once a later line starts a record; independent of chunk boundaries.
      if (!sawArrayStart && start >= 0) {
        const lastNewline = buffer.lastIndexOf("\n");
        const boundary =
          lastNewline > start ? NEW_RECORD_LINE.exec(buffer.slice(start, lastNewline)) : null;
        if (boundary) {
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
      // Report a truncated array record in the same words as the closing-bracket check.
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
      const salvaged = salvageLines(tail, lineFramed);
      if (salvaged.records.length === 0) {
        options.onMalformed?.(tail);
      } else {
        for (const text of salvaged.damaged) options.onMalformed?.(text);
        yield* salvaged.records;
      }
    }
  }

  // A missing closing bracket is the only sign a download was cut between records.
  if (sawArrayStart && !sawArrayEnd) {
    throw new SyntaxError(
      "The JSON array ends before its closing bracket, so the export is incomplete.",
    );
  }
}
