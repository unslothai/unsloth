// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface TranscriptRecord {
  id: string;
  title: string;
  text: string;
  model: string;
  language: string | null;
  duration: number | null;
  created_at: string;
  archived: boolean;
}

export interface TranscriptProgress {
  text: string;
  processed_seconds?: number;
  duration?: number;
}

export interface TranscriptResult {
  text: string;
  model: string;
  duration: number | null;
  record: TranscriptRecord | null;
}

export async function readTranscriptStream(
  body: ReadableStream<Uint8Array>,
  onProgress: (progress: TranscriptProgress) => void,
): Promise<TranscriptResult> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      if (done && buffer.trim()) lines.push(buffer);
      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "error") throw new Error(event.message);
        if (event.type === "progress") onProgress(event);
        if (event.type === "complete") return event as TranscriptResult;
      }
      if (done)
        throw new Error(
          "The transcription connection ended before the result arrived.",
        );
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
}
