// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

// One prompt per file, each in its own new chat, one chat at a time: the new chat's composer claims and sends the step.

export interface BatchThreadSnapshot {
  isRunning: boolean;
  messages: readonly {
    role: string;
    status?: { type: string; reason?: string } | null;
    metadata?: {
      custom?: {
        researchRun?: { status?: string } | null;
        generationLocallyInterrupted?: boolean;
        incomplete?: { reason?: string } | null;
      } | null;
    } | null;
  }[];
}

export type BatchStepOutcome = "pending" | "done" | "failed" | "stopped";

/** Settled once the chat's last message is a finished reply; a cut-off reply still counts. */
export function batchStepOutcome(
  snapshot: BatchThreadSnapshot | null,
): BatchStepOutcome {
  if (!snapshot || snapshot.isRunning) return "pending";
  const last = snapshot.messages.at(-1);
  if (!last || last.role !== "assistant") return "pending";
  const status = last.status;
  if (!status || status.type === "running" || status.type === "requires-action") {
    return "pending";
  }
  if (status.type === "incomplete") {
    if (status.reason === "error") return "failed";
    if (status.reason === "cancelled") return "stopped";
  }
  // These replies complete even when the research failed or was stopped, or the stream stalled.
  const custom = last.metadata?.custom;
  if (custom?.generationLocallyInterrupted === true || custom?.incomplete?.reason === "empty") {
    return "failed";
  }
  const research = custom?.researchRun?.status;
  if (research === "failed") return "failed";
  if (research === "cancelled") return "stopped";
  return "done";
}

export interface ThreadListLike {
  getItemById: (id: string) => { getState: () => { id: string; remoteId?: string | null } };
  getById: (id: string) => { getState: () => BatchThreadSnapshot };
}

/** Reads one chat by id, following its id from local to saved. */
export function batchThreadReader(threads: ThreadListLike, itemId: string) {
  const ids = new Set([itemId]);
  return (): BatchThreadSnapshot | null => {
    for (const id of [...ids]) {
      try {
        const item = threads.getItemById(id).getState();
        ids.add(item.id);
        if (item.remoteId) ids.add(item.remoteId);
      } catch {
        // Try the other id.
      }
    }
    for (const id of ids) {
      try {
        return threads.getById(id).getState();
      } catch {
        // Try the other id.
      }
    }
    return null;
  };
}

// Natural order: clip-2 before clip-10.
export function orderBatchFiles(files: readonly File[]): File[] {
  return [...files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { numeric: true, sensitivity: "base" }),
  );
}

export function batchTargetKey(nonce: string): string {
  return `single:${nonce}`;
}

export type BatchPhase = "opening" | "sending" | "running";

export interface BatchRun {
  id: number;
  prompt: string;
  files: File[];
  index: number;
  nonce: string;
  phase: BatchPhase;
}

export type BatchFinish =
  | { kind: "done"; total: number }
  | { kind: "stopped"; sent: number; total: number }
  | { kind: "failed"; index: number; total: number; fileName: string; reason: string };

export interface BatchNotifier {
  progress: (index: number, total: number, fileName: string) => void;
  finish: (result: BatchFinish) => void;
}

export interface BatchDeps {
  openNewChat: (nonce: string) => void;
  newNonce: () => string;
  notify: BatchNotifier;
}

export interface BatchStepWatch {
  /** The chat this step sent into, or null while it cannot be read. */
  read: () => BatchThreadSnapshot | null;
  /** True while the send is still preparing (an attachment upload), before the message lands. */
  sending: () => boolean;
}

export interface BatchClaim {
  file: File;
  prompt: string;
  index: number;
  total: number;
  sent: (watch: BatchStepWatch) => void;
  failed: (reason: string) => void;
}

export const BATCH_POLL_MS = 500;
// A new chat that no composer claims (the user left Chat) ends the batch instead of hanging it.
export const BATCH_OPEN_TIMEOUT_MS = 30_000;
// Long enough for a send parked behind a model load or a file still being read.
export const BATCH_SEND_TIMEOUT_MS = 120_000;
// A reply cut at Max Tokens auto-continues a moment after it ends; wait that out before moving on.
export const BATCH_SETTLE_MS = 2_000;

interface BatchState {
  run: BatchRun | null;
}

export const useBatchNewChatsStore = create<BatchState>(() => ({ run: null }));

// Batch chats keep their own draft slot, so they never restore or consume the New chat draft.
const batchNonces = new Set<string>();

export function isBatchTargetKey(key: string | null | undefined): boolean {
  return key != null && key.startsWith("single:") && batchNonces.has(key.slice("single:".length));
}

let runSeq = 0;
let deps: BatchDeps | null = null;
let timer: ReturnType<typeof setTimeout> | ReturnType<typeof setInterval> | null = null;

function clearTimer(): void {
  if (timer === null) return;
  clearInterval(timer as ReturnType<typeof setInterval>);
  clearTimeout(timer as ReturnType<typeof setTimeout>);
  timer = null;
}

function currentRun(id: number): BatchRun | null {
  const run = useBatchNewChatsStore.getState().run;
  return run && run.id === id ? run : null;
}

function finish(result: BatchFinish): void {
  clearTimer();
  const notify = deps?.notify;
  useBatchNewChatsStore.setState({ run: null });
  deps = null;
  notify?.finish(result);
}

function openStep(id: number, index: number): void {
  const run = currentRun(id);
  if (!run || !deps) return;
  clearTimer();
  const nonce = deps.newNonce();
  batchNonces.add(nonce);
  useBatchNewChatsStore.setState({ run: { ...run, index, nonce, phase: "opening" } });
  deps.notify.progress(index, run.files.length, run.files[index].name);
  const expire = () => {
    const now = currentRun(id);
    if (!now || now.phase !== "opening" || now.nonce !== nonce) return;
    // A hidden tab throttles the composer's claim poll, so only a visible Studio times out.
    if (typeof document !== "undefined" && document.hidden) {
      timer = setTimeout(expire, BATCH_OPEN_TIMEOUT_MS);
      return;
    }
    fail(id, "Couldn't open a new chat for this file. Keep Chat open while the batch runs.");
  };
  timer = setTimeout(expire, BATCH_OPEN_TIMEOUT_MS);
  deps.openNewChat(nonce);
}

function fail(id: number, reason: string): void {
  const run = currentRun(id);
  if (!run) return;
  finish({
    kind: "failed",
    index: run.index,
    total: run.files.length,
    fileName: run.files[run.index].name,
    reason,
  });
}

function watchStep(id: number, index: number, watch: BatchStepWatch): void {
  clearTimer();
  const startedAt = Date.now();
  let doneSince: number | null = null;
  timer = setInterval(() => {
    const run = currentRun(id);
    if (!run || run.index !== index) {
      clearTimer();
      return;
    }
    const snapshot = watch.read();
    const outcome = batchStepOutcome(snapshot);
    if (outcome !== "done") doneSince = null;
    if (outcome === "pending") {
      const nothingSent = !snapshot || snapshot.messages.length === 0;
      // A refused send keeps the prompt in the composer, a failed attachment send clears it: either
      // way nothing reaches this chat.
      if (nothingSent && !watch.sending() && Date.now() - startedAt > BATCH_SEND_TIMEOUT_MS) {
        fail(id, "Studio didn't send this chat. Check its message box, then start the batch again.");
      }
      return;
    }
    if (outcome === "failed") {
      fail(id, "The reply ended with an error.");
      return;
    }
    if (outcome === "stopped") {
      finish({ kind: "stopped", sent: index + 1, total: run.files.length });
      return;
    }
    doneSince ??= Date.now();
    if (Date.now() - doneSince < BATCH_SETTLE_MS) return;
    if (index + 1 < run.files.length) {
      openStep(id, index + 1);
    } else {
      finish({ kind: "done", total: run.files.length });
    }
  }, BATCH_POLL_MS);
}

export function batchNewChatsRunning(): boolean {
  return useBatchNewChatsStore.getState().run !== null;
}

export function startBatchNewChats(
  prompt: string,
  files: readonly File[],
  batchDeps: BatchDeps,
): boolean {
  const text = prompt.trim();
  if (batchNewChatsRunning() || !text || files.length === 0) return false;
  runSeq += 1;
  deps = batchDeps;
  const id = runSeq;
  useBatchNewChatsStore.setState({
    run: { id, prompt: text, files: orderBatchFiles(files), index: 0, nonce: "", phase: "opening" },
  });
  openStep(id, 0);
  return true;
}

/** Stops before the next chat; the chat already sending keeps its reply. */
export function stopBatchNewChats(): void {
  const run = useBatchNewChatsStore.getState().run;
  if (!run) return;
  const sent = run.phase === "opening" ? run.index : run.index + 1;
  finish({ kind: "stopped", sent, total: run.files.length });
}

export function claimBatchStep(targetKey: string): BatchClaim | null {
  const run = useBatchNewChatsStore.getState().run;
  if (!run || run.phase !== "opening" || batchTargetKey(run.nonce) !== targetKey) {
    return null;
  }
  const { id, index } = run;
  clearTimer();
  useBatchNewChatsStore.setState({ run: { ...run, phase: "sending" } });
  const stillThisStep = () => {
    const now = currentRun(id);
    return now !== null && now.index === index && now.phase === "sending";
  };
  return {
    file: run.files[index],
    prompt: run.prompt,
    index,
    total: run.files.length,
    sent: (watch) => {
      const now = currentRun(id);
      if (!now || !stillThisStep()) return;
      useBatchNewChatsStore.setState({ run: { ...now, phase: "running" } });
      watchStep(id, index, watch);
    },
    failed: (reason) => {
      if (!stillThisStep()) return;
      fail(id, reason);
    },
  };
}

export function resetBatchNewChats(): void {
  clearTimer();
  deps = null;
  useBatchNewChatsStore.setState({ run: null });
}
