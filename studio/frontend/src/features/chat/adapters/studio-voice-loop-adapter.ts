// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The store directly, not the feature barrel: the barrel pulls in the runtime
// provider, which reaches back here through the dictation dispatcher.
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  createBargeGate,
  stepBargeGate,
} from "@/features/chat/voice/barge-gate";
import {
  isBackchannel,
  isReplyEcho,
  joinTurn,
} from "@/features/chat/voice/turn-complete";
import {
  VadUnavailableError,
  detectSpeech,
  loadVadModel,
} from "@/features/chat/voice/vad-client";
import {
  JUDGE_SEND_THRESHOLD,
  judgeTurnComplete,
} from "@/features/chat/voice/voice-models-client";
import {
  isVoicePlaybackActive,
  getVoiceReplyText,
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

// Voice mode's listener: capture the mic, detect end-of-utterance with a
// simple energy VAD, then transcribe the recording with the picked STT model and
// emit the transcript. Unlike the Web Speech adapter this needs no browser cloud
// speech service, so it works in Edge/Brave and the Tauri desktop webview.
//
// STT is batch, not streaming, so there are no interim results: one final
// transcript per utterance, mirroring how the Web Speech adapter ends a session
// on silence and lets the voice loop re-arm.

const SILENCE_RMS = 0.012; // RMS below this counts as silence (ends the utterance)
// A frame only counts as real SPEECH toward the transcribe gate (voicedMs /
// heardSpeech) at or above this, comfortably over the ambient-noise floor. Frames
// between SILENCE_RMS and SPEECH_RMS keep the utterance alive but don't count, so
// room noise or a mic click can't quietly accumulate the 350ms of "speech" that
// would get fed to STT (which then hallucinates "Thank you." etc. from noise).
// Lower it if genuinely soft speech is being dropped.
const SPEECH_RMS = 0.02;
const SILENCE_HANG_MS = 1000; // silence after speech ends the utterance
// A transcript that reads cut off ("Could you tell me just like") is held, not
// sent, and the next window gets a longer hang so the rest of the thought can
// land. If nothing more is said, the fragment goes as-is after the flush wait.
const CONTINUATION_HANG_MS = 1800;
const CONTINUATION_FLUSH_MS = 4000;
const CONTINUATION_MAX_MS = 20000; // a fragment never waits longer than this in total
const MAX_UTTERANCE_MS = 30000; // hard cap so a stuck mic can't record forever
const NO_SPEECH_TIMEOUT_MS = 8000; // give up quietly if no speech is heard
// Minimum total voiced audio required to treat a window as real speech and
// transcribe it. Coughs, clicks, a blip of the model's own voice, or ambient
// noise fall under this and are dropped -- this is what stops STT being fed
// junk (and hallucinating repeated tokens) during barge-in / idle listening.
const MIN_SPEECH_MS = 350;
// Audio kept from just before a barge-in fires, so the first word of what cut in
// still reaches STT.
const BARGE_PREROLL_MS = 600;
// A burst this long cuts the reply on loudness alone (no detector); with the detector confirming
// a voice, half of it is enough and a cough no longer needs the long hold to be ruled out.
const BARGE_UNCONFIRMED_MS = 450;
const BARGE_CONFIRMED_MS = 250;
// What the detector must hear in a clip for it to count as speech at all.
const VAD_MIN_SPEECH_MS = 300;

// The backend said it cannot detect speech here (no runtime, model refused): stop asking for
// the rest of the app session and fall back to the loudness gate alone.
let vadDisabled = false;
// Silence trimmed around the voiced span before sending to STT (STT
// hallucinates on long leading/trailing silence). Keep a little context.
const SILENCE_PAD_MS = 200;
// Faked streaming reveal: STT is batch (one transcript per utterance), but
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
    (window as unknown as { webkitAudioContext?: AudioContextCtor })
      .webkitAudioContext
  );
};

// Encode mono float samples as a 16-bit PCM WAV blob (soundfile reads this; no
// ffmpeg needed server-side).
function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset: number, s: string) => {
    for (let i = 0; i < s.length; i++)
      view.setUint8(offset + i, s.charCodeAt(i));
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

// STT hallucinates a repeated word/phrase on non-speech audio (noise,
// breath, silence) -- e.g. "Ha, ha, ha..." or "Thank you. Thank you...". Reject
// a transcript dominated by one repeated token (very low unique-word ratio) so
// it isn't sent as a prompt. Short utterances (< 6 words) are always allowed.
function isHallucinatedTranscript(text: string): boolean {
  const words = text
    .toLowerCase()
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
  if (words.length < 6) return false;
  return new Set(words).size / words.length < 0.3;
}

// The unfinished fragment carried across listen() sessions; the re-arm clears
// the composer, so the text lives here until the turn is whole.
let heldTurn: { text: string; at: number } | null = null;

// The turn judge decides whether to wait for more; without it the turn is sent.
async function turnUnfinished(text: string): Promise<boolean> {
  const p = await judgeTurnComplete(text);
  return p !== null && p < JUDGE_SEND_THRESHOLD;
}

/** Drop a fragment waiting for its continuation: a thread switch or voice going off. */
/** The fragment the judge is holding for its continuation, shown in the composer meanwhile. */
export function getHeldTurnText(): string {
  return heldTurn?.text ?? "";
}

export function clearHeldTurn(): void {
  heldTurn = null;
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
    // Parallel to chunks: whether each frame was above the silence floor, used to
    // measure voiced duration and to trim silence before transcription.
    const chunkVoiced: boolean[] = [];
    let voicedMs = 0;
    const bargeGate = createBargeGate(
      vadDisabled ? BARGE_UNCONFIRMED_MS : BARGE_CONFIRMED_MS,
    );
    let vadReady = false;
    let bargeConfirming = false;
    const preroll: Float32Array[] = [];
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
        // The fragment outlives the session: the engine cancels and reopens the mic
        // around every run end, and a held turn must survive that.
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
      // Mic is closing: no longer hearing the user, so drop the orb's salmon
      // "hearing you" state (the utterance ended or was cancelled).
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

    // Whether the clip holds a voice. null: no detector, decide on loudness as before.
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

    // Nothing said this window (or only a sub-threshold blip: cough, click, a bit of the
    // model's own voice, ambient noise). End quietly and re-arm, mirroring the Web Speech
    // "no-speech" path so the loop keeps listening without feeding STT junk to hallucinate on.
    const endWithoutSpeech = () => {
      // A burst that held the reply was not the user after all.
      if (bargedIn) requestVoiceBargeAbort();
      if (heldTurn) {
        // The rest never came: the fragment was the whole turn after all.
        const text = heldTurn.text;
        heldTurn = null;
        for (const cb of speechCallbacks)
          cb({ transcript: text, isFinal: true });
        finish("stopped", text);
        setTimeout(() => requestVoiceSubmit(), 0);
        return;
      }
      finish("stopped");
      setTimeout(() => requestVoiceResume(), 0);
    };

    // End the utterance: tear down capture, transcribe the buffer, emit result.
    let finalizing = false;
    const finalize = async () => {
      if (finalizing || ended) return;
      finalizing = true;
      teardown();

      if (!heardSpeech || chunks.length === 0 || voicedMs < MIN_SPEECH_MS) {
        endWithoutSpeech();
        return;
      }

      // Trim to the voiced span (+ a little padding) so STT isn't handed long
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

      // Loud is not the same as spoken: a cough, a hum or a chair clears the loudness gate.
      // The detector rules those out before anything reaches STT.
      if ((await clipHasSpeech(wav)) === false) {
        if (ended) return;
        endWithoutSpeech();
        return;
      }
      if (ended) return;

      try {
        // The PR posted its own multipart to /api/audio/transcribe and let the
        // backend pick a model. Upstream since grew a real STT stack: a curated
        // model list, per-model engines, downloads, and language handling. Go
        // through its helper so the loop transcribes with the model the Voice
        // settings tab actually downloaded, instead of whatever the route
        // defaults to (which is how this asked for an absent "small").
        const modelId =
          useChatRuntimeStore.getState().selectedSttModelId ?? undefined;
        // Flag transcribing so the orb shows a processing state (not idle green)
        // while STT runs -- the first call can take many seconds on ROCm.
        const store = useChatRuntimeStore.getState();
        store.setVoiceTranscribing(true);
        let transcript = "";
        try {
          transcript = (
            await transcribeAudioBlob(wav, { model: modelId })
          ).trim();
          if (ended) return;
        } catch (error) {
          // A model the user has selected but never downloaded is the common
          // first-run case, not a failure: raise the same download prompt the
          // Dictate button raises and keep the loop alive.
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
          // STT invents a repeated token ("Ha, ha, ha...", "Thank you.
          // Thank you...") from noise/breath/silence. Drop it and keep listening
          // rather than sending garbage as a prompt.
          transcript = "";
        }
        if (transcript && bargedIn && isReplyEcho(transcript, getVoiceReplyText())) {
          // The reply's own voice got through the detector. Not a turn: let the
          // reply pick back up and keep listening.
          requestVoiceBargeAbort();
          finish("stopped");
          setTimeout(() => requestVoiceResume(), 0);
          return;
        }
        if (transcript && isBackchannel(transcript)) {
          // A breath or "mm" that cleared the speech gate. Not a turn: keep listening,
          // and keep any held fragment waiting for its real continuation.
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
          (await turnUnfinished(transcript)) &&
          performance.now() - (heldTurn?.at ?? performance.now()) <
            CONTINUATION_MAX_MS
        ) {
          // Mid-thought: show it in the composer, keep it, and listen for the rest.
          heldTurn = {
            text: transcript,
            at: heldTurn?.at ?? performance.now(),
          };
          for (const cb of speechCallbacks) cb({ transcript, isFinal: false });
          finish("stopped");
          setTimeout(() => requestVoiceResume(), 0);
          return;
        }
        heldTurn = null;
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
            for (const cb of speechCallbacks)
              cb({ transcript: partial, isFinal: false });
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
          // STT heard nothing usable in the burst that held the reply.
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
        // Keep the voice loop alive: a transcribe failure (transient backend
        // hiccup, etc.) shouldn't leave the mic dead so it never hears you again.
        // Re-arm so the next utterance still gets a shot.
        setTimeout(() => requestVoiceResume(), 0);
      }
    };

    void (async () => {
      try {
        // Pin capture to the input device chosen in Settings > Voice, so the loop
        // listens to their headset mic and not a loopback / "Stereo Mix" /
        // default-communications device that mixes in system/app audio (e.g.
        // Discord). "default" -> browser default.
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
          // Same fallback as the Dictate adapters: a pinned headset that is
          // unplugged must not kill the loop, just lose its pin for this turn.
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
        // Created after the mic await, outside the user gesture; a suspended context delivers
        // no audioprocess callbacks and the loop only ever hits its no-speech timeout.
        await audioCtx.resume();
        sampleRate = audioCtx.sampleRate;
        source = audioCtx.createMediaStreamSource(stream);
        processor = audioCtx.createScriptProcessor(4096, 1, 1);
        startedAt = performance.now();
        lastVoiceAt = startedAt;
        session.status = { type: "running" };

        // Warm the STT model while the user is still drawing breath. The STT model's
        // first load is slow enough (tens of seconds on ROCm) that paying it
        // after the first utterance reads as the loop having missed what was
        // said. Fire-and-forget: transcription awaits the same load, so a
        // failure here surfaces there rather than killing the session.
        void loadSttModel(
          useChatRuntimeStore.getState().selectedSttModelId ??
            useVoiceSettingsStore.getState().sttModel,
        ).catch(() => {});
        // The speech detector is small and loads in seconds; until it is up, clips pass on
        // loudness alone as before.
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
          // The reply is playing: listen only for the user cutting in. Nothing is
          // recorded and the no-speech clock waits, so when playback ends this
          // session simply becomes the next turn's listener.
          if (!bargedIn && isVoicePlaybackActive()) {
            preroll.push(new Float32Array(input));
            while (preroll.length * frameMs > BARGE_PREROLL_MS) preroll.shift();
            startedAt = now;
            lastVoiceAt = now;
            if (!stepBargeGate(bargeGate, rms, frameMs)) return;
            const commitBarge = () => {
              bargedIn = true;
              requestVoiceBargeIn();
              for (const frame of preroll.splice(0)) {
                chunks.push(frame);
                chunkVoiced.push(true);
              }
              voicedMs = bargeGate.loudMs;
              heardSpeech = true;
              useChatRuntimeStore.getState().setVoiceHearing(true);
              for (const cb of speechStartCallbacks) cb();
            };
            if (!vadReady || vadDisabled) {
              // No detector: only a long burst cuts in, so a cough decays before it counts.
              if (bargeGate.loudMs < BARGE_UNCONFIRMED_MS) return;
              commitBarge();
              return;
            }
            if (bargeConfirming) return;
            // Loud enough for long enough: ask the detector whether it is a voice before
            // cutting the reply. The preroll keeps filling meanwhile, so a confirmed barge
            // still carries its first word.
            bargeConfirming = true;
            const total = preroll.reduce((n, c) => n + c.length, 0);
            const probe = new Float32Array(total);
            let at = 0;
            for (const c of preroll) {
              probe.set(c, at);
              at += c.length;
            }
            void clipHasSpeech(encodeWav(probe, sampleRate)).then((voiced) => {
              bargeConfirming = false;
              if (ended || bargedIn) return;
              if (voiced === false) {
                bargeGate.loudMs = 0;
                return;
              }
              commitBarge();
            });
            return;
          }
          chunks.push(new Float32Array(input));
          // notSilent keeps the utterance alive (resets the silence-hang timer);
          // speech is the stricter bar that actually counts toward transcription,
          // so noise/clicks between the two thresholds never reach STT.
          const notSilent = rms >= SILENCE_RMS;
          const speech = rms >= SPEECH_RMS;
          chunkVoiced.push(speech);
          if (notSilent) lastVoiceAt = now;
          if (speech) {
            voicedMs += frameMs;
            if (!heardSpeech) {
              heardSpeech = true;
              // Light the orb's salmon "hearing you" state the instant the mic
              // picks up real voice -- immediate proof the mic is capturing you.
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
        // keeps it running without ever playing the mic back.
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
