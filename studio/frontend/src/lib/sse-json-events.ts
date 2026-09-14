// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { takeSseFrame } from "./sse-framing.ts";

const LINE_BREAK = /\r?\n/;

/** Pull every complete frame out of `buffer`, returning their data payloads and the rest. */
function drainFrames(buffer: string): { payloads: string[]; rest: string } {
  const payloads: string[] = [];
  let rest = buffer;
  let frame = takeSseFrame(rest);
  while (frame) {
    rest = frame.remainder;
    const data = frame.event
      .split(LINE_BREAK)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart())
      .join("\n");
    if (data) payloads.push(data);
    frame = takeSseFrame(rest);
  }
  return { payloads, rest };
}

function readWithStall(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  onStall: () => void,
  stallMs: number | undefined,
): Promise<ReadableStreamReadResult<Uint8Array>> {
  if (stallMs === undefined) return reader.read();
  const stall = setTimeout(onStall, stallMs);
  return reader.read().finally(() => clearTimeout(stall));
}

/**
 * Yield the JSON payload of each `data:` frame until the stream ends or sends [DONE].
 *
 * `stallMs` bounds the silence between frames. A reverse proxy such as a Cloudflare
 * tunnel can hold a whole event stream until the response completes, and no origin
 * header or padding prevents it, so the only recourse is to abandon a silent stream.
 * The generator ends rather than throwing, which is indistinguishable from a normal
 * close and lets a caller reconcile by polling.
 */
export async function* readSseJsonEvents<T>(
  body: ReadableStream<Uint8Array>,
  stallMs?: number,
): AsyncGenerator<T> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  // cancelling resolves the pending read as done, so the generator ends without an error
  const cancel = () => {
    reader.cancel().catch(() => undefined);
  };
  let buffer = "";
  try {
    while (true) {
      const chunk = await readWithStall(reader, cancel, stallMs);
      if (chunk.done) break;
      const { payloads, rest } = drainFrames(
        buffer + decoder.decode(chunk.value, { stream: true }),
      );
      buffer = rest;
      for (const data of payloads) {
        if (data === "[DONE]") return;
        try {
          yield JSON.parse(data) as T;
        } catch {
          // ignore unparseable frames; [DONE] still ends the loop
        }
      }
    }
  } finally {
    // release the stream lock now instead of leaking the reader until GC
    cancel();
  }
}
