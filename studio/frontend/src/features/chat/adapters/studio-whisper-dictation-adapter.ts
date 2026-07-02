// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  requestVoiceBargeIn,
  requestVoiceResume,
  requestVoiceSubmit,
} from "@/components/assistant-ui/thread";
import { authFetch } from "@/features/auth";
import { useChatRuntimeStore } from "@/features/chat";
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
// Minimum total voiced audio required to treat a window as real speech and
// transcribe it. Coughs, clicks, a blip of the model's own voice, or ambient
// noise fall under this and are dropped -- this is what stops Whisper being fed
// junk (and hallucinating repeated tokens) during barge-in / idle listening.
const MIN_SPEECH_MS = 350;
// Sustained voiced audio that triggers a real-time barge-in (cut the TTS) while
// the model is speaking, without waiting for end-of-utterance + transcription.
// Slightly below MIN_SPEECH_MS so the interrupt lands fast; the utterance still
// has to clear MIN_SPEECH_MS to actually be transcribed and sent.
const BARGE_IN_MS = 250;
// Silence trimmed around the voiced span before sending to Whisper (Whisper
// hallucinates on long leading/trailing silence). Keep a little context.
const SILENCE_PAD_MS = 200;
// Faked streaming reveal: Whisper is batch (one transcript per utterance), but
// once it lands we replay it as growing interim results so the composer types it
// out char-by-char like the streaming Web Speech engine, instead of the whole
// line appearing at once. REVEAL_MAX_MS caps the total so long transcripts don't
// lag the send much.
const REVEAL_CHAR_MS = 12;
const REVEAL_MAX_MS = 1000;

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
    // Parallel to chunks: whether each frame was above the silence floor, used to
    // measure voiced duration and to trim silence before transcription.
    const chunkVoiced: boolean[] = [];
    let voicedMs = 0;
    let bargedIn = false;
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

      if (!heardSpeech || chunks.length === 0 || voicedMs < MIN_SPEECH_MS) {
        // Nothing said this window (or only a sub-threshold blip: cough, click,
        // a bit of the model's own voice, ambient noise). End quietly and re-arm,
        // mirroring the Web Speech "no-speech" path so the loop keeps listening
        // without feeding Whisper junk to hallucinate on.
        finish("stopped");
        setTimeout(() => requestVoiceResume(), 0);
        return;
      }

      // Trim to the voiced span (+ a little padding) so Whisper isn't handed long
      // leading/trailing silence, which it tends to hallucinate words from.
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
      const wav = encodeWav(merged, sampleRate);

      try {
        const form = new FormData();
        form.append("file", wav, "speech.wav");
        const modelId = useChatRuntimeStore.getState().selectedSttModelId;
        if (modelId) form.append("model", modelId);
        // Flag transcribing so the orb shows a processing state (not idle green)
        // while Whisper runs -- the first call can take many seconds on ROCm.
        const store = useChatRuntimeStore.getState();
        store.setVoiceTranscribing(true);
        let transcript = "";
        try {
          const response = await authFetch("/api/audio/transcribe", {
            method: "POST",
            body: form,
          });
          if (ended) return;
          if (!response.ok) throw new Error(`Transcription failed (${response.status})`);
          const data = (await response.json()) as { text?: string };
          transcript = (data.text ?? "").trim();
        } finally {
          store.setVoiceTranscribing(false);
        }
        if (transcript) {
          // Fake a streaming reveal: replay the transcript as growing interim
          // results so the composer types it out char-by-char (like Web Speech),
          // then commit + send. Bails immediately if the session is superseded.
          const total = transcript.length;
          const stepChars = Math.max(
            1,
            Math.ceil(total / Math.max(1, REVEAL_MAX_MS / REVEAL_CHAR_MS)),
          );
          for (let i = stepChars; i < total; i += stepChars) {
            if (ended) return;
            const partial = transcript.slice(0, i);
            for (const cb of speechCallbacks) cb({ transcript: partial, isFinal: false });
            await new Promise<void>((r) => setTimeout(r, REVEAL_CHAR_MS));
          }
          if (ended) return;
          // onSpeech(isFinal) commits the full transcript into the composer text;
          // onSpeechEnd (via finish) ends the session. Then, deferred so those
          // state updates land first, submit the turn. requestVoiceSubmit is a
          // no-op outside voice mode, so plain Dictate-button use just fills the
          // composer as before.
          for (const cb of speechCallbacks) cb({ transcript, isFinal: true });
          finish("stopped", transcript);
          setTimeout(() => requestVoiceSubmit(), 0);
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
        // Pin capture to the user-chosen input device when set, so the loop
        // listens to their headset mic and not a loopback / "Stereo Mix" /
        // default-communications device that mixes in system/app audio (e.g.
        // Discord). null -> browser default.
        const micDeviceId = useChatRuntimeStore.getState().selectedMicDeviceId;
        stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            ...(micDeviceId ? { deviceId: { exact: micDeviceId } } : {}),
          },
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

          const voiced = rms >= SILENCE_RMS;
          chunkVoiced.push(voiced);
          if (voiced) {
            voicedMs += (input.length / sampleRate) * 1000;
            lastVoiceAt = now;
            if (!heardSpeech) {
              heardSpeech = true;
              for (const cb of speechStartCallbacks) cb();
            }
            // Real-time barge-in: as soon as speech is sustained past the
            // threshold, cut the TTS immediately (once). The VoiceEngine handler
            // no-ops unless the model is actually speaking, so this is safe to
            // fire during normal listening too.
            if (!bargedIn && voicedMs >= BARGE_IN_MS) {
              bargedIn = true;
              requestVoiceBargeIn();
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
