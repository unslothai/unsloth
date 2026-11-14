// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The store directly: the feature barrel imports back here through the dictation dispatcher.
import { encodeWav } from "@/features/chat/adapters/pcm-recorder";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  createBargeGate,
  stepBargeGate,
} from "@/features/chat/voice/barge-gate";
import {
  isBackchannel,
  joinTurn,
  looksUnfinished,
} from "@/features/chat/voice/turn-complete";
import {
  VadUnavailableError,
  detectSpeech,
  loadVadModel,
} from "@/features/chat/voice/vad-client";
import {
  isVoicePlaybackActive,
  requestVoiceBargeAbort,
  requestVoiceBargeIn,
  requestVoiceResume,
  requestVoiceSubmit,
  voiceLoopSupported,
} from "@/features/chat/voice/voice-loop-bridge";
import { requestSttDownload } from "@/features/settings/stores/stt-download-prompt-store";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { toast } from "@/lib/toast";
import type { DictationAdapter } from "@assistant-ui/react";
import { SttModelNotDownloadedError } from "./stt-errors";
import {
  loadSttModel,
  transcribeAudioBlob,
} from "./studio-model-dictation-adapter";
import { isMissingDeviceError } from "./studio-web-speech-dictation-adapter";


const SILENCE_RMS = 0.012;
const SPEECH_RMS = 0.02;
const SILENCE_HANG_MS = 1000;
const CONTINUATION_HANG_MS = 1800;
const CONTINUATION_FLUSH_MS = 4000;
const CONTINUATION_MAX_MS = 20000;
const MAX_UTTERANCE_MS = 30000;
const NO_SPEECH_TIMEOUT_MS = 8000;
const MIN_SPEECH_MS = 350;
const BARGE_PREROLL_MS = 600;
const BARGE_UNCONFIRMED_MS = 450;
const BARGE_CONFIRMED_MS = 250;
const VAD_MIN_SPEECH_MS = 300;

let vadDisabled = false;
const SILENCE_PAD_MS = 200;
const REVEAL_CHAR_MS = 12;
const REVEAL_MAX_MS = 1000;

type AudioContextCtor = typeof AudioContext;

const getAudioContextCtor = (): AudioContextCtor | undefined => {
  if (typeof window === "undefined") return undefined;
  return (
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: AudioContextCtor })
      .webkitAudioContext
  );
};

function isHallucinatedTranscript(text: string): boolean {
  const words = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
  if (words.length < 6) return false;
  return new Set(words).size / words.length < 0.3;
}

let heldTurn: { text: string; at: number } | null = null;
// The fragment cut into the reply: the next session keeps listening as that barge-in, not
// through the paused reply's barge gate, which resets its clocks and never flushes the fragment.
let heldFromBarge = false;

// The last transcript this loop put in the composer: the re-arm clears only that, never a typed draft.
let voiceBoxText = "";

export function voiceWroteBox(text: string): boolean {
  return text.trim() !== "" && text.trim() === voiceBoxText.trim();
}

export function clearHeldTurn(): void {
  heldTurn = null;
  heldFromBarge = false;
}

export class StudioVoiceLoopAdapter implements DictationAdapter {
  static isSupported(): boolean {
    return voiceLoopSupported();
  }

  listen(): DictationAdapter.Session {
    const AudioCtx = getAudioContextCtor();
    if (!AudioCtx || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Voice mode can't use the microphone in this browser.");
    }

    const speechStartCallbacks = new Set<() => void>();
    const speechEndCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();
    const speechCallbacks = new Set<
      (result: DictationAdapter.Result) => void
    >();

    let stream: MediaStream | null = null;
    let audioCtx: AudioContext | null = null;
    let processor: ScriptProcessorNode | null = null;
    let source: MediaStreamAudioSourceNode | null = null;
    const chunks: Float32Array[] = [];
    const chunkVoiced: boolean[] = [];
    let voicedMs = 0;
    const bargeGate = createBargeGate(
      vadDisabled ? BARGE_UNCONFIRMED_MS : BARGE_CONFIRMED_MS,
    );
    let vadReady = false;
    let bargeConfirming = false;
    // The longest loud run while the detector was asked: quiet frames reset the gate meanwhile.
    let bargePeakMs = 0;
    const preroll: Float32Array[] = [];
    let bargedIn = heldTurn !== null && heldFromBarge;
    let sampleRate = 16000;
    let heardSpeech = false;
    let lastVoiceAt = 0;
    let startedAt = 0;
    let ended = false;
    let resolveEnded: (() => void) | null = null;
    const endedPromise = new Promise<void>((resolve) => {
      resolveEnded = resolve;
    });

    const session: DictationAdapter.Session = {
      status: { type: "starting" },
      stop: async () => {
        await finalize();
        await endedPromise;
      },
      cancel: () => {
        teardown();
        finish("cancelled");
      },
      onSpeechStart: (cb) => {
        speechStartCallbacks.add(cb);
        return () => speechStartCallbacks.delete(cb);
      },
      onSpeechEnd: (cb) => {
        speechEndCallbacks.add(cb);
        return () => speechEndCallbacks.delete(cb);
      },
      onSpeech: (cb) => {
        speechCallbacks.add(cb);
        return () => speechCallbacks.delete(cb);
      },
    };
    const emitSpeech = (transcript: string, isFinal: boolean) => {
      voiceBoxText = transcript;
      for (const cb of speechCallbacks) cb({ transcript, isFinal });
    };

    const teardown = () => {
      useChatRuntimeStore.getState().setVoiceHearing(false);
      if (processor) {
        processor.onaudioprocess = null;
        processor.disconnect();
        processor = null;
      }
      source?.disconnect();
      source = null;
      stream?.getTracks().forEach((t) => t.stop());
      stream = null;
      void audioCtx?.close().catch(() => {});
      audioCtx = null;
    };

    const finish = (
      reason: "stopped" | "cancelled" | "error",
      transcript?: string,
    ) => {
      if (ended) return;
      ended = true;
      session.status = { type: "ended", reason };
      if (transcript) {
        for (const cb of speechEndCallbacks) cb({ transcript });
      }
      resolveEnded?.();
    };

    const clipHasSpeech = async (blob: Blob): Promise<boolean | null> => {
      if (!vadReady || vadDisabled) return null;
      try {
        const result = await detectSpeech(
          blob,
          useChatRuntimeStore.getState().selectedVadModelId,
        );
        return result.speech_ms >= VAD_MIN_SPEECH_MS;
      } catch (error) {
        if (error instanceof VadUnavailableError) vadDisabled = true;
        return null;
      }
    };

    const endWithoutSpeech = () => {
      if (bargedIn) requestVoiceBargeAbort();
      if (heldTurn) {
        const text = heldTurn.text;
        heldTurn = null;
        emitSpeech(text, true);
        finish("stopped", text);
        setTimeout(() => requestVoiceSubmit(), 0);
        return;
      }
      finish("stopped");
      setTimeout(() => requestVoiceResume(), 0);
    };

    let finalizing = false;
    const finalize = async () => {
      if (finalizing || ended) return;
      finalizing = true;
      teardown();

      if (!heardSpeech || chunks.length === 0 || voicedMs < MIN_SPEECH_MS) {
        endWithoutSpeech();
        return;
      }

      const frameMs = chunks[0] ? (chunks[0].length / sampleRate) * 1000 : 0;
      const padFrames = frameMs > 0 ? Math.ceil(SILENCE_PAD_MS / frameMs) : 0;
      const firstVoiced = chunkVoiced.indexOf(true);
      const lastVoiced = chunkVoiced.lastIndexOf(true);
      const start = Math.max(0, firstVoiced - padFrames);
      const end = Math.min(chunks.length - 1, lastVoiced + padFrames);
      const span = chunks.slice(start, end + 1);

      const total = span.reduce((n, c) => n + c.length, 0);
      const merged = new Float32Array(total);
      let offset = 0;
      for (const c of span) {
        merged.set(c, offset);
        offset += c.length;
      }
      const wav = new Blob([encodeWav(merged, sampleRate)], { type: "audio/wav" });

      if ((await clipHasSpeech(wav)) === false) {
        if (ended) return;
        endWithoutSpeech();
        return;
      }
      if (ended) return;

      try {
        const modelId =
          useChatRuntimeStore.getState().selectedSttModelId ?? undefined;
        const store = useChatRuntimeStore.getState();
        store.setVoiceTranscribing(true);
        let transcript = "";
        try {
          transcript = (
            await transcribeAudioBlob(wav, { model: modelId })
          ).trim();
          if (ended) return;
        } catch (error) {
          if (error instanceof SttModelNotDownloadedError) {
            requestSttDownload(
              modelId ?? useVoiceSettingsStore.getState().sttModel,
            );
          }
          throw error;
        } finally {
          store.setVoiceTranscribing(false);
        }
        if (transcript && isHallucinatedTranscript(transcript)) {
          transcript = "";
        }
        if (transcript && isBackchannel(transcript)) {
          if (bargedIn) requestVoiceBargeAbort();
          finish("stopped");
          setTimeout(() => requestVoiceResume(), 0);
          return;
        }
        if (transcript && heldTurn) {
          transcript = joinTurn(heldTurn.text, transcript);
        }
        if (
          transcript &&
          looksUnfinished(transcript) &&
          performance.now() - (heldTurn?.at ?? performance.now()) <
            CONTINUATION_MAX_MS
        ) {
          heldTurn = {
            text: transcript,
            at: heldTurn?.at ?? performance.now(),
          };
          heldFromBarge = bargedIn;
          emitSpeech(transcript, false);
          finish("stopped");
          setTimeout(() => requestVoiceResume(), 0);
          return;
        }
        heldTurn = null;
        if (transcript) {
          const total = transcript.length;
          const stepChars = Math.max(
            1,
            Math.ceil(total / Math.max(1, REVEAL_MAX_MS / REVEAL_CHAR_MS)),
          );
          for (let i = stepChars; i < total; i += stepChars) {
            if (ended) return;
            const partial = transcript.slice(0, i);
            emitSpeech(partial, false);
            await new Promise<void>((r) => setTimeout(r, REVEAL_CHAR_MS));
          }
          if (ended) return;
          emitSpeech(transcript, true);
          finish("stopped", transcript);
          setTimeout(() => requestVoiceSubmit(), 0);
        } else {
          if (bargedIn) requestVoiceBargeAbort();
          finish("stopped");
          setTimeout(() => requestVoiceResume(), 0);
        }
      } catch (error) {
        if (ended) return;
        console.error("Voice transcription error:", error);
        toast.error("Transcription failed.");
        if (bargedIn) requestVoiceBargeAbort();
        finish("error");
        setTimeout(() => requestVoiceResume(), 0);
      }
    };

    void (async () => {
      try {
        const micDeviceId = useVoiceSettingsStore.getState().micDeviceId;
        const pinnedMic = micDeviceId && micDeviceId !== "default";
        const baseAudio: MediaTrackConstraints = {
          echoCancellation: true,
          noiseSuppression: true,
        };
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: pinnedMic
              ? { ...baseAudio, deviceId: { exact: micDeviceId } }
              : baseAudio,
          });
        } catch (error) {
          if (pinnedMic && isMissingDeviceError(error)) {
            stream = await navigator.mediaDevices.getUserMedia({
              audio: baseAudio,
            });
          } else {
            throw error;
          }
        }
        if (ended) {
          teardown();
          return;
        }
        audioCtx = new AudioCtx();
        // Created outside the user gesture: a suspended context never fires audioprocess.
        await audioCtx.resume();
        sampleRate = audioCtx.sampleRate;
        source = audioCtx.createMediaStreamSource(stream);
        processor = audioCtx.createScriptProcessor(4096, 1, 1);
        startedAt = performance.now();
        lastVoiceAt = startedAt;
        session.status = { type: "running" };

        void loadSttModel(
          useChatRuntimeStore.getState().selectedSttModelId ??
            useVoiceSettingsStore.getState().sttModel,
        ).catch(() => {});
        if (!vadDisabled) {
          void loadVadModel(useChatRuntimeStore.getState().selectedVadModelId)
            .then(() => {
              vadReady = true;
              useChatRuntimeStore.getState().setVadSlotLoaded(true);
            })
            .catch((error: unknown) => {
              if (error instanceof VadUnavailableError) vadDisabled = true;
              useChatRuntimeStore.getState().setVadSlotLoaded(false);
            });
        }

        processor.onaudioprocess = (event) => {
          if (ended || finalizing) return;
          const input = event.inputBuffer.getChannelData(0);

          let sumSquares = 0;
          for (let i = 0; i < input.length; i++)
            sumSquares += input[i] * input[i];
          const rms = Math.sqrt(sumSquares / input.length);
          const now = performance.now();

          const frameMs = (input.length / sampleRate) * 1000;
          if (!bargedIn && isVoicePlaybackActive()) {
            preroll.push(new Float32Array(input));
            // While the detector is asked, keep every frame: the probe's audio opens the utterance.
            if (!bargeConfirming) while (preroll.length * frameMs > BARGE_PREROLL_MS) preroll.shift();
            startedAt = now;
            lastVoiceAt = now;
            const cut = stepBargeGate(bargeGate, rms, frameMs);
            if (bargeConfirming) bargePeakMs = Math.max(bargePeakMs, bargeGate.loudMs);
            if (!cut) return;
            const commitBarge = () => {
              bargedIn = true;
              requestVoiceBargeIn();
              for (const frame of preroll.splice(0)) {
                chunks.push(frame);
                chunkVoiced.push(true);
              }
              voicedMs = Math.max(bargeGate.loudMs, bargePeakMs);
              bargePeakMs = 0;
              heardSpeech = true;
              useChatRuntimeStore.getState().setVoiceHearing(true);
              for (const cb of speechStartCallbacks) cb();
            };
            if (!vadReady || vadDisabled) {
              if (bargeGate.loudMs < BARGE_UNCONFIRMED_MS) return;
              commitBarge();
              return;
            }
            if (bargeConfirming) return;
            bargeConfirming = true;
            bargePeakMs = bargeGate.loudMs;
            const total = preroll.reduce((n, c) => n + c.length, 0);
            const probe = new Float32Array(total);
            let at = 0;
            for (const c of preroll) {
              probe.set(c, at);
              at += c.length;
            }
            void clipHasSpeech(new Blob([encodeWav(probe, sampleRate)], { type: "audio/wav" })).then((voiced) => {
              bargeConfirming = false;
              if (ended || bargedIn) return;
              if (voiced === false) {
                bargeGate.loudMs = 0;
                bargePeakMs = 0;
                return;
              }
              commitBarge();
            });
            return;
          }
          chunks.push(new Float32Array(input));
          const notSilent = rms >= SILENCE_RMS;
          const speech = rms >= SPEECH_RMS;
          chunkVoiced.push(speech);
          if (notSilent) lastVoiceAt = now;
          if (speech) {
            voicedMs += frameMs;
            if (!heardSpeech) {
              heardSpeech = true;
              useChatRuntimeStore.getState().setVoiceHearing(true);
              for (const cb of speechStartCallbacks) cb();
            }
          }
          const sinceVoice = now - lastVoiceAt;
          const elapsed = now - startedAt;
          if (
            (heardSpeech &&
              sinceVoice >=
                (heldTurn ? CONTINUATION_HANG_MS : SILENCE_HANG_MS)) ||
            elapsed >= MAX_UTTERANCE_MS ||
            (!heardSpeech &&
              elapsed >=
                (heldTurn ? CONTINUATION_FLUSH_MS : NO_SPEECH_TIMEOUT_MS))
          ) {
            void finalize();
          }
        };

        // onaudioprocess only fires while the processor reaches a destination; a muted sink
        const sink = audioCtx.createGain();
        sink.gain.value = 0;
        source.connect(processor);
        processor.connect(sink);
        sink.connect(audioCtx.destination);
      } catch (error) {
        teardown();
        console.error("Voice mic error:", error);
        toast.error("Could not access the microphone for dictation.");
        finish("error");
      }
    })();

    return session;
  }
}
