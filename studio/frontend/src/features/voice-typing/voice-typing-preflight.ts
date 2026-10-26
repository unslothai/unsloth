// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Reads the chosen dictation model's state from the backend so voice typing can
// say what pressing the shortcut will do: start instantly, load first, ask for a
// download, or refuse because the local engine is missing. Mirrors the rules the
// Voice tab applies to the same status payload.

import {
  fetchSttStatus,
  loadSttModel,
  sttEngineStatusFor,
} from "@/features/chat";
import {
  requestSttDownload,
  sttModelVariant,
  useVoiceSettingsStore,
} from "@/features/settings";
import type { VoiceTypingReadiness } from "./voice-typing-store";

export interface VoiceTypingModelState {
  readiness: VoiceTypingReadiness;
  /** The model this answer is about, for display. */
  model: string;
}

export async function checkVoiceTypingModel(
  signal?: AbortSignal,
): Promise<VoiceTypingModelState> {
  const settings = useVoiceSettingsStore.getState();
  const model = settings.sttModel;
  if (settings.dictationEngine === "custom") {
    return { readiness: "custom", model };
  }
  // A pinned quant is only resident when that quant is the one loaded; the same
  // rule the dictation card applies, so the two surfaces cannot disagree.
  const variant = sttModelVariant(model, settings.sttGgufVariant);
  const status = await fetchSttStatus(undefined, model, signal);
  const engine = sttEngineStatusFor(status, model);
  if (!engine) return { readiness: "unknown", model };
  if (!engine.available) return { readiness: "engine-missing", model };
  if (engine.loading) return { readiness: "loading", model };
  if (
    engine.loaded_model === model &&
    (!variant || engine.loaded_variant === variant)
  ) {
    return { readiness: "ready", model };
  }
  if (engine.downloaded_models.includes(model)) {
    return { readiness: "will-load", model };
  }
  return { readiness: "needs-download", model };
}

/** Load the configured model so the first dictation does not wait on it. */
export function warmVoiceTypingModel(): Promise<void> {
  const { sttModel, sttGgufVariant } = useVoiceSettingsStore.getState();
  return loadSttModel(
    sttModel,
    undefined,
    undefined,
    undefined,
    sttModelVariant(sttModel, sttGgufVariant),
  );
}

/** Ask the app's own download flow to fetch the model (same prompt the mic uses). */
export function requestVoiceTypingModelDownload(): void {
  const { sttModel, sttGgufVariant } = useVoiceSettingsStore.getState();
  requestSttDownload(sttModel, {
    selectLocalEngine: true,
    ggufVariant: sttModelVariant(sttModel, sttGgufVariant),
  });
}
