// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { requestVoiceResume } from "@/components/assistant-ui/thread";
import { authFetch } from "@/features/auth";
import type { DictationAdapter } from "@assistant-ui/react";
import { toast } from "sonner";

// Backend Whisper STT dictation: capture the mic, detect end-of-utterance with a
// simple energy VAD, then POST the recorded audio to /api/audio/transcribe and
// emit the transcript. Unlike the Web Speech adapter this needs no browser cloud
// speech service, so it works in Edge/Brave and the Tauri desktop webview.
//
// Whisper is batch, not streaming, so there are no interim results: one final
// transcript per utterance, mirroring how the Web Speech adapter ends a session
// on silence and lets the voice loop re-arm.

const SILENCE_RMS = 0.012;       // RMS below this counts as silence
const SILENCE_HANG_MS = 1000;    // silence after speech ends the utterance
const MAX_UTTERANCE_MS = 30000;  // hard cap so a stuck mic can't record forever
const NO_SPEECH_TIMEOUT_MS = 8000;  // give up quietly if no speech is heard

type AudioContextCtor = typeof AudioContext;

const getAudioContextCtor = (): AudioContextCtor | undefined => {
  if (typeof window === "undefined") return undefined;
  return (
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: AudioContextCtor }).webkitAudioContext
  );
};

// Encode mono float samples as a 16-bit PCM WAV blob (soundfile reads this; no
// ffmpeg needed server-side).
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i] ?? 0));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}

export class StudioWhisperDictationAdapter implements DictationAdapter {
  static isSupported(): boolean {
    return (
      typeof window !== "undefined" &&
      navigator.mediaDevices?.getUserMedia !== undefined &&
      getAudioContextCtor() !== undefined
    );
  }

  listen(): DictationAdapter.Session {
    const AudioCtx = getAudioContextCtor();
    if (!AudioCtx || !navigator.mediaDevices?.getUserMedia) {
      throw new Error("Whisper dictation is not supported in this browser.");
    }

    const speechStartCallbacks = new Set<() => void>();
    const speechEndCallbacks = new Set<(result: DictationAdapter.Result) => void>();
    const speechCallbacks = new Set<(result: DictationAdapter.Result) => void>();

    let stream: MediaStream | null = null;
    let audioCtx: AudioContext | null = null;
    let processor: ScriptProcessorNode | null = null;
    let source: MediaStreamAudioSourceNode | null = null;
    const chunks: Float32Array[] = [];
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

    const teardown = () => {
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

    const finish = (reason: "stopped" | "cancelled" | "error", transcript?: string) => {
      if (ended) return;
      ended = true;
      session.status = { type: "ended", reason };
      if (transcript) {
        for (const cb of speechEndCallbacks) cb({ transcript });
      }
      resolveEnded?.();
    };

    // End the utterance: tear down capture, transcribe the buffer, emit result.
    let finalizing = false;
    const finalize = async () => {
      if (finalizing || ended) return;
      finalizing = true;
      teardown();

      if (!heardSpeech || chunks.length === 0) {
        // Nothing said this window -- end quietly and re-arm, mirroring the Web
        // Speech "no-speech" path so the voice loop keeps listening.
        finish("stopped");
        setTimeout(() => requestVoiceResume(), 0);
        return;
      }

      const total = chunks.reduce((n, c) => n + c.length, 0);
      const merged = new Float32Array(total);
      let offset = 0;
      for (const c of chunks) {
        merged.set(c, offset);
        offset += c.length;
      }
      const wav = encodeWav(merged, sampleRate);

      try {
        const form = new FormData();
        form.append("file", wav, "speech.wav");
        const response = await authFetch("/api/audio/transcribe", {
          method: "POST",
          body: form,
        });
        if (ended) return;
        if (!response.ok) throw new Error(`Transcription failed (${response.status})`);
        const data = (await response.json()) as { text?: string };
        const transcript = (data.text ?? "").trim();
        if (transcript) {
          for (const cb of speechCallbacks) cb({ transcript, isFinal: true });
          finish("stopped", transcript);
        } else {
          finish("stopped");
          setTimeout(() => requestVoiceResume(), 0);
        }
      } catch (error) {
        if (ended) return;
        console.error("Whisper dictation error:", error);
        toast.error("Whisper transcription failed.");
        finish("error");
      }
    };

    void (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true },
        });
        if (ended) {
          teardown();
          return;
        }
        audioCtx = new AudioCtx();
        sampleRate = audioCtx.sampleRate;
        source = audioCtx.createMediaStreamSource(stream);
        processor = audioCtx.createScriptProcessor(4096, 1, 1);
        startedAt = performance.now();
        lastVoiceAt = startedAt;
        session.status = { type: "running" };

        processor.onaudioprocess = (event) => {
          if (ended || finalizing) return;
          const input = event.inputBuffer.getChannelData(0);
          chunks.push(new Float32Array(input));

          let sumSquares = 0;
          for (let i = 0; i < input.length; i++) sumSquares += input[i] * input[i];
          const rms = Math.sqrt(sumSquares / input.length);
          const now = performance.now();

          if (rms >= SILENCE_RMS) {
            lastVoiceAt = now;
            if (!heardSpeech) {
              heardSpeech = true;
              for (const cb of speechStartCallbacks) cb();
            }
          }

          const sinceVoice = now - lastVoiceAt;
          const elapsed = now - startedAt;
          if (
            (heardSpeech && sinceVoice >= SILENCE_HANG_MS) ||
            elapsed >= MAX_UTTERANCE_MS ||
            (!heardSpeech && elapsed >= NO_SPEECH_TIMEOUT_MS)
          ) {
            void finalize();
          }
        };

        source.connect(processor);
        processor.connect(audioCtx.destination); // required for onaudioprocess to fire
      } catch (error) {
        teardown();
        console.error("Whisper dictation mic error:", error);
        toast.error("Could not access the microphone for dictation.");
        finish("error");
      }
    })();

    return session;
  }
}
