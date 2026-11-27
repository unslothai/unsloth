// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Run event routes send a keep-alive every 15s, so a stream this quiet died without the browser
 *  hearing of it (a phone back from sleep). Wall clock, so time spent asleep counts. */
export const STREAM_SILENCE_TIMEOUT_MS = 45_000;

export function watchStreamSilence(
  reader: ReadableStreamDefaultReader<unknown>,
  timeoutMs = STREAM_SILENCE_TIMEOUT_MS,
): { touch: () => void; stop: () => void } {
  let lastReadAt = Date.now();
  const timer = globalThis.setInterval(
    () => {
      if (Date.now() - lastReadAt > timeoutMs)
        reader.cancel().catch(() => undefined);
    },
    Math.min(5_000, timeoutMs),
  );
  return {
    touch: () => {
      lastReadAt = Date.now();
    },
    stop: () => globalThis.clearInterval(timer),
  };
}
