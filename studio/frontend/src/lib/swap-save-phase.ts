// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// While an Images or Video load POST is pending, load-progress says only that the chat context is
// saving for this load; other phases describe the previous model, so ready or null is not the end.

export const SAVING_CHAT_CONTEXT = "saving_chat_context";

/** Calls onSaving on each tick that reports the save, and ignores every other phase.
 *  stop() is idempotent and drops a read still in flight. */
export function watchSavingPhase(
  read: (signal: AbortSignal) => Promise<{ phase: unknown }>,
  onSaving: () => void,
  opts: { intervalMs?: number; firstDelayMs?: number } = {},
): { stop(): void } {
  const intervalMs = opts.intervalMs ?? 1000;
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let inFlight: AbortController | null = null;

  const tick = async () => {
    if (stopped) return;
    const controller = new AbortController();
    inFlight = controller;
    try {
      const progress = await read(controller.signal);
      if (!stopped && progress?.phase === SAVING_CHAT_CONTEXT) onSaving();
    } catch {
      // A failed read says nothing about the save; try again next tick.
    }
    if (inFlight === controller) inFlight = null;
    if (!stopped) timer = setTimeout(() => void tick(), intervalMs);
  };

  timer = setTimeout(() => void tick(), opts.firstDelayMs ?? intervalMs);
  return {
    stop() {
      if (stopped) return;
      stopped = true;
      if (timer) clearTimeout(timer);
      inFlight?.abort();
    },
  };
}

/** Whether a load-progress phase is a load still running (every nonterminal phase, the chat-context save included). */
export function isLoadInFlightPhase(phase: unknown): boolean {
  return phase === "downloading" || phase === "finalizing" || phase === SAVING_CHAT_CONTEXT;
}
