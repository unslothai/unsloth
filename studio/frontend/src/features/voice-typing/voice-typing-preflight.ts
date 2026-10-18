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
  resetMicrophonePermission,
  sttModelVariant,
  useVoiceSettingsStore,
} from "@/features/settings";
import { translate } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { toast } from "@/lib/toast";
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
  const status = await fetchSttStatus(undefined, model, signal);
  const engine = sttEngineStatusFor(status, model);
  if (!engine) return { readiness: "unknown", model };
  if (!engine.available) return { readiness: "engine-missing", model };
  if (engine.loading) return { readiness: "loading", model };
  if (engine.loaded_model === model) return { readiness: "ready", model };
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

export type MicrophonePermission = "granted" | "denied" | "prompt" | "unknown";

/** What the webview currently thinks of microphone access. */
export async function queryMicrophonePermission(): Promise<MicrophonePermission> {
  try {
    const status = await navigator.permissions?.query({
      name: "microphone" as PermissionName,
    });
    return (status?.state as MicrophonePermission) ?? "unknown";
  } catch {
    // The API or the name is unsupported; callers must not guess.
    return "unknown";
  }
}

/** The grant flow the Voice tab uses: forget a saved deny first, then ask. */
export async function requestVoiceTypingMicrophoneAccess(): Promise<boolean> {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast.error(translate("settings.voice.dictation.micAccessUnsupported"));
    return false;
  }
  // WebView2 keeps a saved "Don't allow" with no settings UI of its own (#9001),
  // so clear it first: this request is the only way back from one.
  await resetMicrophonePermission();
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    for (const track of stream.getTracks()) track.stop();
    return true;
  } catch {
    // A browser keeps its own saved deny that the reset cannot touch, so only
    // the desktop build can promise another prompt.
    toast.error(
      translate(
        isTauri
          ? "settings.voice.dictation.micAccessBlockedDesktop"
          : "settings.voice.dictation.micAccessBlocked",
      ),
    );
    return false;
  }
}
