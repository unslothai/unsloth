// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import {
  type DictationEngine,
  useVoiceSettingsStore,
} from "@/features/settings/stores/voice-settings-store";
import { toast } from "@/lib/toast";
import type { DictationAdapter } from "@assistant-ui/react";
import { StudioModelDictationAdapter } from "./studio-model-dictation-adapter";
import {
  type StudioDictationSession,
  StudioWebSpeechDictationAdapter,
} from "./studio-web-speech-dictation-adapter";

// The one live dictation session, so the recording bar's discard (X) can cancel
// it without going through assistant-ui (which only exposes stop, i.e.
// transcribe). Cancelling emits no transcript, so composer text is untouched.
let activeSession: StudioDictationSession | null = null;

/** Discard the current dictation without transcribing. Safe to call when idle. */
export function cancelActiveStudioDictation(): void {
  const session = activeSession;
  activeSession = null;
  session?.cancel();
}

/**
 * Routes dictation to the engine chosen in Voice settings, resolved at listen()
 * time so switching engines applies without reloading the chat runtime.
 */
/** Both local engines (Transformers and GGUF) record via MediaRecorder. */
function usesModelRecording(dictationEngine: DictationEngine): boolean {
  return dictationEngine === "model";
}

export class StudioDictationAdapter implements DictationAdapter {
  // Chat linked in Recent dictations. undefined follows the active single chat;
  // null records no chat (composers outside it, e.g. Compare).
  private readonly chatId: string | null | undefined;

  constructor(options: { chatId?: string | null } = {}) {
    this.chatId = options.chatId;
  }

  static isSupported(
    dictationEngine: DictationEngine = useVoiceSettingsStore.getState()
      .dictationEngine,
  ): boolean {
    return usesModelRecording(dictationEngine)
      ? StudioModelDictationAdapter.isSupported()
      : StudioWebSpeechDictationAdapter.isSupported();
  }

  listen(): StudioDictationSession {
    const session = this.createSession();
    // A second entry point (chat, Compare, settings test) replaces the active
    // session; cancel the old one so it cannot keep the mic open or save a
    // transcript with no discard button pointing at it.
    cancelActiveStudioDictation();
    activeSession = session;
    // Forget the session once it ends so a later cancel is a no-op.
    const clear = () => {
      if (activeSession === session) {
        activeSession = null;
      }
    };
    session.onSpeechEnd(clear);
    session.onEnd?.(clear);
    return session;
  }

  private createSession(): StudioDictationSession {
    const { dictationEngine } = useVoiceSettingsStore.getState();
    if (usesModelRecording(dictationEngine)) {
      if (StudioModelDictationAdapter.isSupported()) {
        return new StudioModelDictationAdapter({ chatId: this.chatId }).listen();
      }
      throw new Error(
        "Local model dictation is not supported in this browser.",
      );
    }
    if (StudioWebSpeechDictationAdapter.isSupported()) {
      return new StudioWebSpeechDictationAdapter({ chatId: this.chatId }).listen();
    }
    throw new Error("Browser dictation is not supported in this browser.");
  }
}

/** Whether dictation can run now for the chosen engine. */
export function isStudioDictationAvailable(
  dictationEngine: DictationEngine = useVoiceSettingsStore.getState()
    .dictationEngine,
): boolean {
  return StudioDictationAdapter.isSupported(dictationEngine);
}

/** Explain why dictation can't start and point the user to the local model. */
export function notifyStudioDictationUnavailable(
  dictationEngine: DictationEngine = useVoiceSettingsStore.getState()
    .dictationEngine,
): void {
  // Both engines need a secure context (localhost or HTTPS).
  if (typeof window !== "undefined" && !window.isSecureContext) {
    toast.error("Voice typing needs a secure connection.", {
      description:
        "Open Studio at http://127.0.0.1 (localhost) or over HTTPS to dictate.",
    });
    return;
  }
  if (usesModelRecording(dictationEngine)) {
    // Defensive: MediaRecorder is effectively always present here.
    toast.error("Voice recording isn't available in this browser.");
    return;
  }
  // Browser Web Speech is missing (e.g. Firefox). Stack text and button so the
  // action sits below, not squeezed into a side column.
  const toastId = toast.error("Voice typing isn't available in this browser.", {
    description: (
      <div className="mt-0.5 flex flex-col items-start gap-2 pb-1.5">
        <span>
          Choose the local speech-to-text model in Voice settings to dictate
          here.
        </span>
        <button
          type="button"
          onClick={() => {
            useSettingsDialogStore.getState().openDialog("voice");
            toast.dismiss(toastId);
          }}
          className="rounded-full bg-foreground px-2.5 pt-1 pb-1.5 text-xs font-medium text-background transition-colors hover:bg-foreground/90"
        >
          Open Voice settings
        </button>
      </div>
    ),
  });
}
