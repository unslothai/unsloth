// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HighlightResult } from "@streamdown/code";
import { useChatRuntimeStore } from "@/features/chat";
import {
  reasoningHighlightFailure,
  type ReasoningHighlightMessage,
  type ReasoningHighlightReply,
  type ReasoningHighlightRequest,
} from "./reasoning-highlight";
import { createStreamActivity, type StreamActivity } from "./stream-activity";

export type HighlightWorkerState =
  | "untested"
  | "ready"
  | "stalled"
  | "unavailable";

const READY_TIMEOUT_MS = 5_000;
const REPLY_SILENCE_MS = 15_000;
const IDLE_TEARDOWN_MS = 10_000;
let state: HighlightWorkerState = "untested";
let worker: Worker | null = null;
let boot: { instance: Worker; timer: ReturnType<typeof setTimeout> } | null =
  null;
let idle: ReturnType<typeof setTimeout> | undefined;
let nextClient = 0;
const listeners = new Map<number, (reply: ReasoningHighlightReply) => void>();
const patientClients = new Set<number>();
const awaitingReply = new Set<number>();
let silence: ReturnType<typeof setTimeout> | undefined;
let activity: StreamActivity | null = null;

export const highlightWorkerState = (): HighlightWorkerState => state;

export const streamActive = (): boolean =>
  (activity ??= createStreamActivity(useChatRuntimeStore)).active();

const clearBoot = (instance: Worker): void => {
  if (boot?.instance !== instance) return;
  clearTimeout(boot.timer);
  boot = null;
};

const afterQueuedMessages = (callback: () => void): void => {
  const channel = new MessageChannel();
  channel.port1.onmessage = () => {
    channel.port1.close();
    callback();
  };
  channel.port2.postMessage(0);
};

const flushListeners = (spared?: ReadonlySet<number>): void => {
  const waiting = [...listeners].filter(([id]) => !spared?.has(id));
  const deliver = (): void => {
    const next = waiting.shift();
    if (next === undefined) return;
    const [id, listener] = next;
    if (listeners.get(id) === listener) {
      listeners.delete(id);
      listener(reasoningHighlightFailure(id, 0));
    }
    if (waiting.length > 0) setTimeout(deliver, 0);
  };
  if (waiting.length > 0) setTimeout(deliver, 0);
};

function fail(instance: Worker, reason: string): void {
  if (worker !== instance) return;
  state = "unavailable";
  clearBoot(instance);
  if (idle) clearTimeout(idle);
  worker = null;
  instance.terminate();
  console.warn(
    "[Unsloth Code] highlight worker unavailable, highlighting on the main thread:",
    reason,
  );
  flushListeners();
}

function stall(instance: Worker): void {
  if (worker !== instance || boot?.instance !== instance) return;
  state = "stalled";
  console.warn(
    "[Unsloth Code] highlight worker did not answer within",
    READY_TIMEOUT_MS,
    "ms, highlighting on the main thread until it does",
  );
  flushListeners(patientClients);
}

function silenced(instance: Worker): void {
  if (worker !== instance || boot?.instance === instance) return;
  if (awaitingReply.size === 0) return;
  state = "stalled";
  console.warn(
    "[Unsloth Code] highlight worker sent nothing for",
    REPLY_SILENCE_MS,
    "ms, highlighting on the main thread until it does",
  );
  flushListeners(patientClients);
}

const watchSilence = (instance: Worker): void => {
  if (silence) clearTimeout(silence);
  silence = undefined;
  if (awaitingReply.size === 0) return;
  silence = setTimeout(() => {
    silence = undefined;
    afterQueuedMessages(() => silenced(instance));
  }, REPLY_SILENCE_MS);
};

function getWorker(patient = false): Worker | null {
  if (state === "unavailable") return null;
  if (state === "stalled" && !patient) return null;
  if (idle) clearTimeout(idle);
  if (worker) return worker;
  if (typeof Worker === "undefined") {
    state = "unavailable";
    return null;
  }
  let instance: Worker;
  try {
    instance = new Worker(
      new URL("./reasoning-highlight.worker.ts", import.meta.url),
      { type: "module" },
    );
  } catch (error) {
    state = "unavailable";
    console.warn(
      "[Unsloth Code] highlight worker unavailable, highlighting on the main thread:",
      error,
    );
    return null;
  }
  instance.onmessage = ({ data }: MessageEvent<ReasoningHighlightMessage>) => {
    if (instance !== worker) return;
    if ("ready" in data) {
      clearBoot(instance);
      state = "ready";
      watchSilence(instance);
      if (listeners.size === 0) scheduleIdle();
      return;
    }
    if (state === "stalled" && boot === null) state = "ready";
    listeners.get(data.client)?.(data);
    watchSilence(instance);
  };
  instance.onerror = (event) => fail(instance, event.message || event.type);
  worker = instance;
  boot = {
    instance,
    timer: setTimeout(
      () => afterQueuedMessages(() => stall(instance)),
      READY_TIMEOUT_MS,
    ),
  };
  return instance;
}

function scheduleIdle(): void {
  if (idle) clearTimeout(idle);
  idle = setTimeout(() => {
    if (!worker || boot?.instance === worker) return;
    worker.terminate();
    worker = null;
    if (state === "stalled") state = "untested";
  }, IDLE_TEARDOWN_MS);
}

export function requestFullHighlight(
  source: string,
  language: string | null,
  onResult: (result: HighlightResult | null) => void,
  speculative = false,
  patient = !speculative,
): (() => void) | null {
  const instance = getWorker(patient);
  if (!instance) return null;
  const id = ++nextClient;
  if (patient) patientClients.add(id);
  listeners.set(id, (reply) => {
    if (!reply.failed && (reply.revision !== 1 || !reply.result)) return;
    listeners.delete(id);
    patientClients.delete(id);
    awaitingReply.delete(id);
    onResult(reply.failed ? null : (reply.result as HighlightResult));
    if (listeners.size === 0 && worker) scheduleIdle();
  });
  const request: ReasoningHighlightRequest = {
    client: id,
    revision: 1,
    source,
    language,
    lines: [],
    full: true,
    ...(speculative ? { speculative: true } : {}),
  };
  instance.postMessage(request);
  awaitingReply.add(id);
  if (!silence) watchSilence(instance);
  return () => {
    if (!listeners.has(id)) return;
    listeners.delete(id);
    patientClients.delete(id);
    awaitingReply.delete(id);
    worker?.postMessage({ cancel: id });
    if (listeners.size === 0 && worker) scheduleIdle();
  };
}
