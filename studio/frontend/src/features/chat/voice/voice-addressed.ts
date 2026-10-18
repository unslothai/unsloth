// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The "only answer when I'm talking to it" gate the loop runs on a finished turn:
// asks the addressed model, keeps what was not for the assistant in an overheard
// buffer that the next real turn carries as context, and reports the outcomes.

import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import {
  addressedOutcome,
  type AddressedLabel,
  type AddressedPrev,
  disownsReply,
  isAddressedByName,
  type OverheardEntry,
  pushOverheard,
  withOverheardContext,
} from "./addressed-gate";
import { REPEAT_WINDOW_MS } from "./laya-outcomes";
import { publishAddressedDecision } from "./voice-decisions";
import { addressedLabel, reportLayaOutcome } from "./voice-models-client";
import { voiceTrace } from "./voice-trace";

let prev: AddressedPrev = "none";
let overheard: OverheardEntry[] = [];
// The log rows behind the last verdicts, waiting for the next utterance to say whether they held.
let heldBack: { id: string; text: string; at: number; timer: ReturnType<typeof setTimeout> } | null = null;
let answered: { id: string; at: number } | null = null;

function settle(text: string): void {
  if (heldBack) {
    clearTimeout(heldBack.timer);
    const gapMs = Math.round(performance.now() - heldBack.at);
    reportLayaOutcome(heldBack.id, addressedOutcome(heldBack.text, text, gapMs), { gap_ms: gapMs });
    heldBack = null;
  }
  if (answered) {
    const gapMs = Math.round(performance.now() - answered.at);
    reportLayaOutcome(answered.id, disownsReply(text) ? "cut" : "ok", { gap_ms: gapMs });
    answered = null;
  }
}

export interface AddressedVerdict {
  submit: boolean;
  /** The text to send: the turn with any overheard speech quoted in front. */
  text: string;
  label: AddressedLabel;
}

/** Decide a finished turn. Off, or without the model, every turn is submitted as before. */
export async function gateAddressed(text: string): Promise<AddressedVerdict> {
  settle(text);
  const send = (label: AddressedLabel, id: string | null): AddressedVerdict => {
    prev = "assistant";
    const out = withOverheardContext(text, overheard, Date.now());
    overheard = [];
    if (id) answered = { id, at: performance.now() };
    return { submit: true, text: out, label };
  };
  if (!useVoiceSettingsStore.getState().voiceAddressedGate) return send("assistant", null);
  if (isAddressedByName(text)) {
    publishAddressedDecision("assistant", 1);
    voiceTrace("laya.addressed", { label: "assistant", confidence: 1, text, why: "named" });
    return send("assistant", null);
  }
  const r = await addressedLabel(text, prev);
  if (!r) return send("assistant", null);
  if (r.label === "assistant") return send("assistant", r.id);
  prev = r.label;
  overheard = pushOverheard(overheard, { text, label: r.label, at: Date.now() });
  if (r.id) {
    const id = r.id;
    heldBack = {
      id,
      text,
      at: performance.now(),
      timer: setTimeout(() => {
        if (heldBack?.id === id) heldBack = null;
        reportLayaOutcome(id, "ok");
      }, REPEAT_WINDOW_MS),
    };
  }
  return { submit: false, text, label: r.label };
}

/** Forget the context and buffer: voice off, or a different chat. */
export function resetAddressed(): void {
  prev = "none";
  overheard = [];
  if (heldBack) clearTimeout(heldBack.timer);
  heldBack = null;
  answered = null;
}

/** What the loop overheard and has not yet handed to a turn, for tests and the trace. */
export function getOverheard(): readonly OverheardEntry[] {
  return overheard;
}
