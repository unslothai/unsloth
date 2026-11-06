// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { MAX_AUDIO_SIZE, MAX_AUDIO_SIZE_LABEL } from "@/lib/audio-utils";
import type { DictationAdapter } from "@assistant-ui/react";
import { toast } from "sonner";
import { micClipFilename } from "../utils/mic-target";
import { startDictationLevelMeter } from "./dictation-level";
import {
  beginDictationSession,
  markDictationClip,
  markDictationFailed,
} from "./dictation-outcome";
import { PcmRecorder } from "./pcm-recorder";
import {
  type StudioDictationSession,
  isMissingDeviceError,
} from "./studio-web-speech-dictation-adapter";

/** Records one clip and hands it to `onClip` as a WAV file instead of transcribing it.
 *  WAV because the audio attachment path takes it as is; MediaRecorder's webm reads as video. */
export function listenForAudioClip(
  onClip: (file: File) => Promise<unknown> | unknown,
): StudioDictationSession {
  beginDictationSession();
  const speechStartCallbacks = new Set<() => void>();
  const speechEndCallbacks = new Set<
    (result: DictationAdapter.Result) => void
  >();
  const endCallbacks = new Set<() => void>();
  let stream: MediaStream | null = null;
  let recorder: PcmRecorder | null = null;
  let stopLevelMeter = () => {};
  let limitTimer: ReturnType<typeof setTimeout> | undefined;
  let ended = false;
  let resolveEnded: () => void = () => {};
  const endedPromise = new Promise<void>((resolve) => {
    resolveEnded = resolve;
  });

  const release = () => {
    clearTimeout(limitTimer);
    stopLevelMeter();
    for (const track of stream?.getTracks() ?? []) track.stop();
    stream = null;
  };
  const finish = (reason: "stopped" | "cancelled" | "error") => {
    if (ended) return;
    ended = true;
    release();
    session.status = { type: "ended", reason };
    for (const callback of speechEndCallbacks) callback({ transcript: "" });
    for (const callback of endCallbacks) callback();
    resolveEnded();
  };

  const session: StudioDictationSession = {
    status: { type: "starting" },
    stop: async () => {
      const active = recorder;
      recorder = null;
      if (ended || !active || active.state === "inactive") {
        finish("stopped");
        return endedPromise;
      }
      stopLevelMeter();
      let delivering = false;
      // PcmRecorder emits its one blob synchronously from stop().
      active.addEventListener(
        "dataavailable",
        (event) => {
          if (event.data.size === 0) return;
          delivering = true;
          const file = new File([event.data], micClipFilename(), {
            type: "audio/wav",
          });
          void Promise.resolve()
            .then(() => onClip(file))
            .then(
              () => markDictationClip(),
              (error: unknown) => {
                // The attachment adapter already toasts its own refusals.
                console.error("Could not attach the recording:", error);
                markDictationFailed();
              },
            )
            .finally(() => finish("stopped"));
        },
        { once: true },
      );
      active.stop();
      // A tap shorter than one buffer yields no clip.
      if (!delivering) finish("stopped");
      return endedPromise;
    },
    cancel: () => {
      const active = recorder;
      recorder = null;
      // Discard, not stop: stop() encodes the whole WAV just to drop it.
      active?.discard();
      finish("cancelled");
    },
    onSpeechStart: (callback) => {
      speechStartCallbacks.add(callback);
      return () => speechStartCallbacks.delete(callback);
    },
    onSpeechEnd: (callback) => {
      speechEndCallbacks.add(callback);
      return () => speechEndCallbacks.delete(callback);
    },
    onSpeech: () => () => {},
    onEnd: (callback: () => void) => {
      endCallbacks.add(callback);
      return () => endCallbacks.delete(callback);
    },
  };

  void (async () => {
    const { micDeviceId } = useVoiceSettingsStore.getState();
    const audio: MediaTrackConstraints = {
      echoCancellation: true,
      noiseSuppression: true,
    };
    try {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio:
            micDeviceId && micDeviceId !== "default"
              ? { ...audio, deviceId: { exact: micDeviceId } }
              : audio,
        });
      } catch (error) {
        // Saved mic may be unplugged; fall back to the default.
        if (micDeviceId === "default" || !isMissingDeviceError(error)) {
          throw error;
        }
        stream = await navigator.mediaDevices.getUserMedia({ audio });
      }
      if (ended) {
        release();
        return;
      }
      recorder = new PcmRecorder(stream);
      recorder.start();
      // The attachment refuses a clip over the audio limit, so stop just under it instead.
      limitTimer = setTimeout(() => {
        toast.info(`Recording stopped at the ${MAX_AUDIO_SIZE_LABEL} audio limit.`);
        void session.stop();
      }, recorder.secondsWithin(MAX_AUDIO_SIZE) * 1000);
      stopLevelMeter = startDictationLevelMeter(stream);
      session.status = { type: "running" };
      for (const callback of speechStartCallbacks) callback();
    } catch (error) {
      console.error("Microphone error:", error);
      toast.error(
        isMissingDeviceError(error)
          ? "No microphone was found."
          : "Could not access the microphone.",
      );
      markDictationFailed();
      finish("error");
    }
  })();

  return session;
}
