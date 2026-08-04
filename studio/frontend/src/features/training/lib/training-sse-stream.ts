// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { takeSseFrame } from "../api/sse-framing.ts";
import type { TrainingProgressPayload } from "../types/runtime";

export type TrainingProgressEventName =
  | "progress"
  | "heartbeat"
  | "complete"
  | "error";

export interface ParsedTrainingProgressEvent {
  event: TrainingProgressEventName;
  payload: TrainingProgressPayload;
  id: number | null;
}

function parseTrainingProgressEvent(
  rawEvent: string,
): ParsedTrainingProgressEvent | null {
  const lines = rawEvent.split(/\r?\n/);
  let eventName: TrainingProgressEventName = "progress";
  let id: number | null = null;
  const dataLines: string[] = [];

  for (const line of lines) {
    if (!line) {
      continue;
    }
    if (line.startsWith("event:")) {
      const value = line.slice(6).trim();
      if (
        value === "progress" ||
        value === "heartbeat" ||
        value === "complete" ||
        value === "error"
      ) {
        eventName = value;
      }
      continue;
    }
    if (line.startsWith("id:")) {
      const value = Number(line.slice(3).trim());
      id = Number.isFinite(value) ? value : null;
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }

  if (dataLines.length === 0) {
    return null;
  }

  const payload = JSON.parse(dataLines.join("\n")) as TrainingProgressPayload;
  return { event: eventName, payload, id };
}

export async function consumeTrainingProgressStream(options: {
  body: ReadableStream<Uint8Array>;
  signal: AbortSignal;
  onEvent: (event: ParsedTrainingProgressEvent) => void;
}): Promise<void> {
  const reader = options.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (!options.signal.aborted) {
      const { value, done } = await reader.read();
      if (done || options.signal.aborted) {
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      let frame = takeSseFrame(buffer);
      while (frame && !options.signal.aborted) {
        const rawEvent = frame.event;
        buffer = frame.remainder;

        if (!rawEvent.startsWith("retry:")) {
          const event = parseTrainingProgressEvent(rawEvent);
          if (event) {
            options.onEvent(event);
          }
        }

        frame = takeSseFrame(buffer);
      }
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
