import {
  PLATFORM_CHAT_AUDIO_MAX_DURATION_MS,
  transcribePlatformChatAudio,
} from "@/integrations/platform-backend";
import {
  applyDictationDictionary,
  recordRecentDictation,
} from "@/features/settings/stores/voice-settings-store";
import type { DictationAdapter } from "@assistant-ui/react";
import { toast } from "sonner";
import {
  beginDictationSession,
  markDictationFailed,
  markDictationTranscript,
} from "./dictation-outcome";
import {
  describeMediaError,
  resolveDictationChatId,
  type StudioDictationSession,
} from "./studio-web-speech-dictation-adapter";

const MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/mp4",
];

function supportedMimeType(): string | undefined {
  return MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type));
}

function stopStream(stream: MediaStream | null): void {
  for (const track of stream?.getTracks() ?? []) track.stop();
}

/** Microphone recording and transcription through the active Rag Platform route. */
export class PlatformDictationAdapter implements DictationAdapter {
  private readonly chatId: string | null | undefined;

  constructor(options: { chatId?: string | null } = {}) {
    this.chatId = options.chatId;
  }

  static isSupported(): boolean {
    return (
      typeof window !== "undefined" &&
      window.isSecureContext &&
      typeof MediaRecorder !== "undefined" &&
      navigator.mediaDevices?.getUserMedia !== undefined
    );
  }

  listen(): StudioDictationSession {
    if (!PlatformDictationAdapter.isSupported()) {
      throw new Error("Rag Platform ses kaydı bu tarayıcıda desteklenmiyor.");
    }

    beginDictationSession();
    const controller = new AbortController();
    const chunks: Blob[] = [];
    const startCallbacks = new Set<() => void>();
    const speechCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();
    const endCallbacks = new Set<(result: DictationAdapter.Result) => void>();
    const lifecycleCallbacks = new Set<() => void>();
    const sessionChatId = resolveDictationChatId(this.chatId);
    let stream: MediaStream | null = null;
    let recorder: MediaRecorder | null = null;
    let ended = false;
    let cancelled = false;
    let durationTimer = 0;
    let resolveEnded: (() => void) | null = null;
    const endedPromise = new Promise<void>((resolve) => {
      resolveEnded = resolve;
    });

    const cleanup = () => {
      if (durationTimer) window.clearTimeout(durationTimer);
      durationTimer = 0;
      stopStream(stream);
      stream = null;
    };
    const finish = (
      reason: "stopped" | "cancelled" | "error",
      transcript = "",
    ) => {
      if (ended) return;
      ended = true;
      cleanup();
      const corrected =
        reason === "cancelled"
          ? ""
          : applyDictationDictionary(transcript).trim();
      session.status = { type: "ended", reason };
      if (corrected) {
        markDictationTranscript();
        for (const callback of speechCallbacks) {
          callback({ transcript: corrected, isFinal: true });
        }
        recordRecentDictation(corrected, sessionChatId);
      }
      for (const callback of endCallbacks) callback({ transcript: corrected });
      for (const callback of lifecycleCallbacks) callback();
      resolveEnded?.();
    };

    const transcribe = async () => {
      if (cancelled || ended) return;
      try {
        const type = recorder?.mimeType || chunks[0]?.type || "audio/webm";
        const text = await transcribePlatformChatAudio(
          new Blob(chunks, { type }),
          controller.signal,
        );
        finish("stopped", text);
      } catch (error) {
        if (cancelled || controller.signal.aborted) return;
        markDictationFailed();
        toast.error(
          error instanceof Error
            ? error.message
            : "Rag Platform ses kaydını yazıya çeviremedi.",
        );
        finish("error");
      }
    };

    const session: StudioDictationSession = {
      status: { type: "starting" },
      stop: async () => {
        if (!ended && recorder?.state === "recording") recorder.stop();
        else if (!ended && !recorder) finish("stopped");
        await endedPromise;
      },
      cancel: () => {
        if (ended) return;
        cancelled = true;
        controller.abort();
        if (recorder?.state === "recording") recorder.stop();
        finish("cancelled");
      },
      onSpeechStart: (callback) => {
        startCallbacks.add(callback);
        return () => startCallbacks.delete(callback);
      },
      onSpeech: (callback) => {
        speechCallbacks.add(callback);
        return () => speechCallbacks.delete(callback);
      },
      onSpeechEnd: (callback) => {
        endCallbacks.add(callback);
        return () => endCallbacks.delete(callback);
      },
      onEnd: (callback) => {
        lifecycleCallbacks.add(callback);
        return () => lifecycleCallbacks.delete(callback);
      },
    };

    void navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((mediaStream) => {
        if (ended || cancelled) {
          stopStream(mediaStream);
          return;
        }
        stream = mediaStream;
        const mimeType = supportedMimeType();
        recorder = mimeType
          ? new MediaRecorder(mediaStream, { mimeType })
          : new MediaRecorder(mediaStream);
        recorder.addEventListener("dataavailable", (event) => {
          if (event.data.size > 0) chunks.push(event.data);
        });
        recorder.addEventListener("stop", () => void transcribe(), {
          once: true,
        });
        recorder.addEventListener(
          "error",
          () => {
            markDictationFailed();
            finish("error");
          },
          { once: true },
        );
        recorder.start(250);
        session.status = { type: "running" };
        for (const callback of startCallbacks) callback();
        durationTimer = window.setTimeout(() => {
          if (recorder?.state === "recording") recorder.stop();
        }, PLATFORM_CHAT_AUDIO_MAX_DURATION_MS);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        markDictationFailed();
        toast.error(describeMediaError(error));
        finish("error");
      });

    return session;
  }
}
