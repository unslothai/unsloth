// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { MAX_AUDIO_SIZE, MAX_AUDIO_SIZE_LABEL } from "@/lib/audio-utils";
import type { DictationAdapter } from "@assistant-ui/react";
import { toast } from "sonner";
import { type MicClipRoom, micClipFilename } from "../utils/mic-target";
import { startDictationLevelMeter } from "./dictation-level";
import {
  beginDictationSession,
  markDictationClip,
  markDictationFailed,
} from "./dictation-outcome";
import { holdMicClipRoom, releaseMicClipRoom } from "./mic-clip-state";
import { PcmRecorder } from "./pcm-recorder";
import {
  type StudioDictationSession,
  isMissingDeviceError,
} from "./studio-web-speech-dictation-adapter";

// WAV, not MediaRecorder webm: the attachment path routes webm as video.
export function listenForAudioClip(
  onClip: (file: File) => Promise<unknown> | unknown,
  room: MicClipRoom = { maxBytes: MAX_AUDIO_SIZE },
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
    releaseMicClipRoom(session);
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
          releaseMicClipRoom(session);
          if (event.data.size === 0) return;
          delivering = true;
          const file = new File([event.data], micClipFilename(), {
            type: "audio/wav",
          });
          void Promise.resolve()
            .then(() => onClip(file))
            .then(
              () => markDictationClip(file),
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
      if (!delivering) finish("stopped");
      return endedPromise;
    },
    cancel: () => {
      const active = recorder;
      recorder = null;
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

  if ("reason" in room) {
    toast.error(room.reason);
    markDictationFailed();
    queueMicrotask(() => finish("error"));
    return session;
  }
  const maxBytes = room.maxBytes;
  holdMicClipRoom(session);

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
        if (micDeviceId === "default" || !isMissingDeviceError(error)) {
          throw error;
        }
        if (ended) return;
        stream = await navigator.mediaDevices.getUserMedia({ audio });
      }
      if (ended) {
        release();
        return;
      }
      recorder = new PcmRecorder(stream);
      // The pre-check assumes 16 kHz; a device-rate fallback needs more room for the same second.
      if (maxBytes < recorder.bytesFor(1)) {
        recorder.discard();
        recorder = null;
        toast.error(
          `Audio files together exceed the ${MAX_AUDIO_SIZE_LABEL} per-message limit`,
        );
        markDictationFailed();
        finish("error");
        return;
      }
      recorder.maxBytes = maxBytes;
      recorder.start();
      limitTimer = setTimeout(() => {
        toast.info(`Recording stopped at the ${MAX_AUDIO_SIZE_LABEL} audio limit.`);
        void session.stop();
      }, recorder.secondsWithin(maxBytes) * 1000);
      stopLevelMeter = startDictationLevelMeter(stream);
      session.status = { type: "running" };
      for (const callback of speechStartCallbacks) callback();
    } catch (error) {
      // Cancelled already: the outcome flags now belong to the next session.
      if (ended) return;
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

type ModelState = {
  params: { checkpoint: string | null | undefined };
  modelLoading?: boolean;
  loadingModelPick?: unknown;
};
type CheckpointStore = {
  getState: () => ModelState;
  subscribe: (listener: (state: ModelState) => void) => () => void;
};

/** A load starting counts as a change: `checkpoint` only moves once the load succeeds. */
export function discardClipWhenModelChanges(
  session: StudioDictationSession,
  store: CheckpointStore,
): void {
  const startedOn = store.getState().params.checkpoint;
  const unsubscribe = store.subscribe((state) => {
    const switching =
      state.params.checkpoint !== startedOn ||
      state.modelLoading === true ||
      state.loadingModelPick != null;
    if (!switching) return;
    unsubscribe();
    if (session.status.type === "ended") return;
    toast.info("Recording discarded because the chat model changed.");
    session.cancel();
  });
  session.onEnd?.(unsubscribe);
}

export function discardClipOnRelease(
  session: StudioDictationSession,
  onRelease: Set<() => void>,
): void {
  const release = () => {
    if (session.status.type === "ended") return;
    toast.info("Recording discarded because the chat changed.");
    session.cancel();
  };
  onRelease.add(release);
  session.onEnd?.(() => onRelease.delete(release));
}
