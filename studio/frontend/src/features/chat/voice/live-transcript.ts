// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";

export type LiveTranscript = {
  feed(samples: Float32Array): void;
  /** Rejects on failure: callers fall back to the recording. */
  finish(): Promise<string>;
  drop(): void;
};

type Fetcher = typeof authFetch;

// Shorter adds requests; longer delays the words.
const POST_MS = 200;
const FINISH_TIMEOUT_MS = 8000;
// Statuses meaning no live route on this model or build.
const REFUSED = new Set([404, 405, 409, 501]);

/** Some transcribers refuse the mic's own 44.1 or 48 kHz. */
const LIVE_SAMPLE_RATE = 16000;

export function createResampler(from: number, to: number): (input: Float32Array) => Float32Array {
  if (from === to) return (input) => input;
  const step = from / to;
  // Cycles per input sample, under the new rate's limit.
  const cutoff = 0.46 * Math.min(1, to / from);
  // Kernel half-width: eight sinc zero crossings.
  const half = Math.ceil(8 / (2 * cutoff));
  let held = new Float32Array(half);
  let at = half;
  return (input) => {
    const all = new Float32Array(held.length + input.length);
    all.set(held);
    all.set(input, held.length);
    const out = new Float32Array(Math.max(0, Math.ceil((all.length - half - at) / step)));
    let n = 0;
    for (; at + half < all.length && n < out.length; at += step) {
      const centre = Math.floor(at);
      let sum = 0;
      for (let k = centre - half; k <= centre + half; k++) {
        const x = k - at;
        const phase = 2 * Math.PI * cutoff * x;
        const sinc = x === 0 ? 1 : Math.sin(phase) / phase;
        const window = 0.5 + 0.5 * Math.cos((Math.PI * x) / (half + 1));
        sum += all[k] * 2 * cutoff * sinc * window;
      }
      out[n++] = sum;
    }
    const keep = Math.max(0, Math.floor(at) - half);
    held = all.slice(keep);
    at -= keep;
    return n === out.length ? out : out.slice(0, n);
  };
}

const tidy = (text: string) => text.replace(/\s+/g, " ").trim();
const asError = (error: unknown) => (error instanceof Error ? error : new Error(String(error)));

export function openLiveTranscript(options: {
  /** Query keys the server route reads: model, engine, language. */
  params: URLSearchParams;
  sampleRate: number;
  /** Full text so far, not just the new part. */
  onText?: (text: string) => void;
  onRefused?: () => void;
  fetcher?: Fetcher;
}): LiveTranscript {
  const { params, sampleRate, onText, onRefused, fetcher = authFetch } = options;
  const abort = new AbortController();
  const resample = createResampler(sampleRate, LIVE_SAMPLE_RATE);
  const minSamples = (LIVE_SAMPLE_RATE * POST_MS) / 1000;
  params.set("sample_rate", String(LIVE_SAMPLE_RATE));
  let id: string | null = null;
  let pending: Int16Array[] = [];
  let pendingSamples = 0;
  let posting = false;
  let ending = false;
  let lastSent = false;
  let settled = false;
  let giveUp: ReturnType<typeof setTimeout> | undefined;
  let heard = "";
  let resolve: (text: string) => void = () => {};
  let reject: (error: Error) => void = () => {};
  const done = new Promise<string>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  done.catch(() => {});
  const fail = (error: Error) => {
    if (settled) return;
    settled = true;
    clearTimeout(giveUp);
    abort.abort();
    reject(error);
  };

  const pump = () => {
    if (!id || posting || settled || lastSent) return;
    if (!ending && pendingSamples < minSamples) return;
    const body = new Int16Array(pendingSamples);
    let at = 0;
    for (const part of pending) {
      body.set(part, at);
      at += part.length;
    }
    pending = [];
    pendingSamples = 0;
    const last = ending;
    posting = true;
    // Never retried: a post that arrived would be heard twice.
    void fetcher(
      `/api/inference/audio/stt/live/${id}/audio${last ? "?last=true" : ""}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/octet-stream" },
        body: body.buffer,
        signal: abort.signal,
      },
      { retryNetworkErrors: false },
    )
      .then((response) => {
        if (!response.ok) throw new Error(`The live transcript refused audio (HTTP ${response.status}).`);
        if (last) lastSent = true;
      })
      .catch((error: unknown) => fail(asError(error)))
      .finally(() => {
        posting = false;
        pump();
      });
  };

  void (async () => {
    try {
      const response = await fetcher(
        `/api/inference/audio/stt/live?${params.toString()}`,
        { method: "POST", signal: abort.signal },
        { retryNetworkErrors: false },
      );
      if (!response.ok || !response.body) {
        if (REFUSED.has(response.status)) onRefused?.();
        throw new Error(`No live transcript (HTTP ${response.status}).`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { value, done: over } = await reader.read();
        if (over) break;
        buffer += decoder.decode(value, { stream: true });
        for (let end = buffer.indexOf("\n"); end >= 0; end = buffer.indexOf("\n")) {
          const line = buffer.slice(0, end).trim();
          buffer = buffer.slice(end + 1);
          if (!line.startsWith("data:")) continue;
          const event = JSON.parse(line.slice(5)) as { type?: string; id?: string; text?: string; message?: string };
          if (event.type === "ready" && event.id) {
            id = event.id;
            pump();
          } else if (event.type === "delta") {
            heard += event.text ?? "";
            if (!settled) onText?.(tidy(heard));
          } else if (event.type === "done") {
            if (!settled) {
              settled = true;
              clearTimeout(giveUp);
              resolve(tidy(event.text ?? heard));
            }
            return;
          } else if (event.type === "error") {
            throw new Error(event.message || "The live transcript failed.");
          }
        }
      }
      throw new Error("The live transcript ended early.");
    } catch (error) {
      fail(asError(error));
    }
  })();

  const drop = () => {
    if (settled) return;
    const session = id;
    fail(new Error("The live transcript was dropped."));
    // Explicit DELETE: a proxy can hold the stream open.
    if (session) {
      void fetcher(`/api/inference/audio/stt/live/${session}`, { method: "DELETE" }, { retryNetworkErrors: false }).catch(() => {});
    }
  };

  return {
    feed(input) {
      if (settled || ending || input.length === 0) return;
      const samples = resample(input);
      const pcm = new Int16Array(samples.length);
      for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }
      pending.push(pcm);
      pendingSamples += pcm.length;
      pump();
    },
    finish() {
      if (!ending) {
        ending = true;
        pump();
        giveUp = setTimeout(drop, FINISH_TIMEOUT_MS);
      }
      return done;
    },
    drop,
  };
}
