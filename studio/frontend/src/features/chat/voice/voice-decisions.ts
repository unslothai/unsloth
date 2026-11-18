// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { useSyncExternalStore } from "react";

export interface VoiceDecisions {
  turn: { p: number; sent: boolean; at: number } | null;
  filler: { phrase: string; p: number; at: number } | null;
  interject: {
    action: "restart" | "ignore" | "queue";
    kind?: "amend" | "replace";
    confidence: number;
    at: number;
  } | null;
  vad: { kept: boolean; at: number } | null;
  stt: { kept: boolean; why: string; at: number } | null;
}

let state: VoiceDecisions = { turn: null, filler: null, interject: null, vad: null, stt: null };
const listeners = new Set<() => void>();

function emit(next: Partial<VoiceDecisions>): void {
  state = { ...state, ...next };
  for (const l of listeners) l();
}

export const publishTurnDecision = (p: number, sent: boolean): void =>
  emit({ turn: { p, sent, at: Date.now() } });
export const publishFillerDecision = (phrase: string, p: number = phrase ? 1 : 0): void =>
  emit({ filler: { phrase, p, at: Date.now() } });
export const publishInterjectDecision = (
  action: "restart" | "ignore" | "queue",
  confidence: number,
  kind?: "amend" | "replace",
): void => emit({ interject: { action, kind, confidence, at: Date.now() } });
export const publishVadDecision = (kept: boolean): void =>
  emit({ vad: { kept, at: Date.now() } });
export const publishSttDecision = (kept: boolean, why: string): void =>
  emit({ stt: { kept, why, at: Date.now() } });
export const clearVoiceDecisions = (): void =>
  emit({ turn: null, filler: null, interject: null, vad: null, stt: null });

export function useVoiceDecisions(): VoiceDecisions {
  return useSyncExternalStore(
    (l) => {
      listeners.add(l);
      return () => listeners.delete(l);
    },
    () => state,
    () => state,
  );
}
