// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Per-model reasoning effort, keyed by checkpoint id. The chat's own `reasoningEffort` is one
// global scalar, so switching between a model you want thinking hard and one you want quick meant
// moving the same control back and forth.
//
// Kept out of the inference-params memory next door: effort is not an InferenceParam but a
// capability-gated level the catalogue publishes per model, and a model that names none must not
// be handed one.

import { create } from "zustand";

const KEY = "unsloth_model_reasoning_effort";

function readEfforts(): Record<string, string> {
  try {
    const raw = JSON.parse(localStorage.getItem(KEY) ?? "{}");
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
    const out: Record<string, string> = {};
    for (const [id, effort] of Object.entries(raw)) {
      if (typeof effort === "string" && effort.length > 0) out[id] = effort;
    }
    return out;
  } catch {
    return {};
  }
}

function writeEfforts(efforts: Record<string, string>): void {
  try {
    localStorage.setItem(KEY, JSON.stringify(efforts));
  } catch {
    // Ignore unavailable storage; the choice stays session-only.
  }
}

interface ModelReasoningEffortState {
  effortByModel: Record<string, string>;
  /** Pin an effort for one model, or pass null to go back to following the chat's own level. */
  setModelReasoningEffort: (modelId: string, effort: string | null) => void;
}

export const useModelReasoningEffortStore = create<ModelReasoningEffortState>(
  (set) => ({
    effortByModel: readEfforts(),
    setModelReasoningEffort: (modelId, effort) =>
      set((state) => {
        if (!modelId) return state;
        const next = { ...state.effortByModel };
        if (effort) next[modelId] = effort;
        else delete next[modelId];
        writeEfforts(next);
        return { effortByModel: next };
      }),
  }),
);

/** The effort pinned for `modelId`, read outside React. Null when the model follows the chat's
 *  own level, and when the pin is not a level this model accepts: a catalogue refresh can
 *  withdraw one, and asking for a level a provider rejects fails the request. */
export function pinnedReasoningEffort(
  modelId: string | null | undefined,
  allowed: readonly string[] | null | undefined,
): string | null {
  if (!modelId) return null;
  const effort = useModelReasoningEffortStore.getState().effortByModel[modelId];
  if (!effort) return null;
  if (allowed && allowed.length > 0 && !allowed.includes(effort)) return null;
  return effort;
}

if (typeof window !== "undefined") {
  window.addEventListener("storage", (event) => {
    if (event.key === KEY || event.key === null) {
      useModelReasoningEffortStore.setState({ effortByModel: readEfforts() });
    }
  });
}
