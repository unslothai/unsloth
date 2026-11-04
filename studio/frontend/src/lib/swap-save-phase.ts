// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// While an Images or Video load POST is pending, the backend may first save the chat model's
// context so chat can resume after the swap. That save is the only thing load-progress says
// about THIS load before the POST answers: every other phase still describes the previous
// model, so reading "ready" or null here as the end of the load would be wrong.

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
