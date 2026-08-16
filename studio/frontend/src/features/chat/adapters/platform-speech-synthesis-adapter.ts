import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { synthesizePlatformChatSpeech } from "@/integrations/platform-backend";
import type { SpeechSynthesisAdapter } from "@assistant-ui/react";
import { toast } from "sonner";

/** Read-aloud through the active Rag Platform Python speech route. */
export class PlatformSpeechSynthesisAdapter implements SpeechSynthesisAdapter {
  static isSupported(): boolean {
    return typeof window !== "undefined" && typeof window.Audio !== "undefined";
  }

  speak(text: string): SpeechSynthesisAdapter.Utterance {
    const subscribers = new Set<() => void>();
    const controller = new AbortController();
    let audio: HTMLAudioElement | null = null;
    let objectUrl: string | null = null;
    let ended = false;

    const notify = () => {
      for (const callback of subscribers) callback();
    };
    const cleanup = () => {
      if (audio) {
        audio.pause();
        audio.removeAttribute("src");
        audio.load();
        audio = null;
      }
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
        objectUrl = null;
      }
    };
    const finish = (
      reason: "finished" | "error" | "cancelled",
      error?: unknown,
    ) => {
      if (ended) return;
      ended = true;
      cleanup();
      utterance.status = { type: "ended", reason, error };
      if (reason === "error") {
        toast.error(
          error instanceof Error
            ? error.message
            : "Rag Platform sesli okuma isteğini tamamlayamadı.",
        );
      }
      notify();
    };

    const utterance: SpeechSynthesisAdapter.Utterance = {
      status: { type: "starting" },
      cancel: () => {
        controller.abort();
        finish("cancelled");
      },
      subscribe: (callback) => {
        subscribers.add(callback);
        return () => subscribers.delete(callback);
      },
    };

    void (async () => {
      try {
        const blob = await synthesizePlatformChatSpeech(
          text,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        objectUrl = URL.createObjectURL(blob);
        audio = new Audio(objectUrl);
        const { ttsRate, ttsVolume } = useVoiceSettingsStore.getState();
        audio.playbackRate = ttsRate;
        audio.volume = ttsVolume;
        audio.addEventListener("ended", () => finish("finished"), {
          once: true,
        });
        audio.addEventListener(
          "error",
          () =>
            finish(
              "error",
              new Error("Rag Platform ses çıktısı oynatılamadı."),
            ),
          { once: true },
        );
        await audio.play();
        if (ended) return;
        utterance.status = { type: "running" };
        notify();
      } catch (error) {
        if (controller.signal.aborted) return;
        finish("error", error);
      }
    })();

    return utterance;
  }
}
