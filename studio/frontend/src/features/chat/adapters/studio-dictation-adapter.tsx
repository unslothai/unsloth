


import { requestSttDownload } from "@/features/settings/stores/stt-download-prompt-store";
import {
  type DictationEngine,
  useVoiceSettingsStore,
} from "@/features/settings/stores/voice-settings-store";
import { toast } from "@/lib/toast";
import type { DictationAdapter } from "@assistant-ui/react";
import {
  StudioModelDictationAdapter,
  fetchSttStatus,
  sttEngineStatusFor,
} from "./studio-model-dictation-adapter";
import {
  type StudioDictationSession,
  StudioWebSpeechDictationAdapter,
} from "./studio-web-speech-dictation-adapter";

// The one live dictation session, so Escape can discard it without going
// through assistant-ui (which only exposes stop, i.e. transcribe). Cancelling
// emits no transcript, so composer text is untouched.
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
  // Browser Web Speech is missing (e.g. Firefox). Local dictation is the only
  // way to type by voice here, so offer it rather than describing it.
  void offerLocalDictation();
}

/**
 * Move a browser with no speech service onto local dictation. Already
 * downloaded means one switch; otherwise the same confirmation the mic raises,
 * which flips the engine only if it is accepted.
 */
async function offerLocalDictation(): Promise<void> {
  const { sttModel, setDictationEngine } = useVoiceSettingsStore.getState();
  try {
    const status = await fetchSttStatus(undefined, sttModel);
    const engine = sttEngineStatusFor(status, sttModel);
    // An engine with no runtime installed cannot load what it downloads, so
    // say what is missing rather than asking for gigabytes first.
    if (engine && !engine.available) {
      toast.error("Local transcription isn't installed on this server.", {
        description:
          "Run `unsloth studio update` to install it, then choose a model in Voice settings.",
      });
      return;
    }
    if (engine?.downloaded_models.includes(sttModel)) {
      setDictationEngine("model");
      toast.success("Switched to local transcription.", {
        description:
          "Voice typing isn't available in this browser. Press the mic again to dictate.",
      });
      return;
    }
  } catch {
    // Status is unreachable; the download path reports its own failure.
  }
  requestSttDownload(sttModel, { selectLocalEngine: true });
}
