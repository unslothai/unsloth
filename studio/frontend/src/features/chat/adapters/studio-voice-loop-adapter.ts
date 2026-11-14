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
  createClipSpeculation,
  SPECULATE_AFTER_MS,
  utteranceSpan,
} from "@/features/chat/voice/clip-speculation";
import { createHearingHint } from "@/features/chat/voice/hearing-hint";
import { createHeldEdit } from "@/features/chat/voice/held-edit";
import { QUICK_CONTINUE_MS } from "@/features/chat/voice/laya-outcomes";
import {
  createReacquireBackoff,
  createSilenceWatch,
  createSingleFlight,
} from "@/features/chat/voice/mic-health";
import {
  isBackchannel,
  isReplyEcho,
  isSttNoise,
  joinTurn,
} from "@/features/chat/voice/turn-complete";
import {
  VadUnavailableError,
  detectSpeech,
  loadVadModel,
} from "@/features/chat/voice/vad-client";
import {
  publishSttDecision,
  publishVadDecision,
} from "@/features/chat/voice/voice-decisions";
import {
  getVoiceParkedText,
  getVoiceReplyText,
  getVoiceMode,
  isVoicePlaybackActive,
  requestVoiceBargeAbort,
  requestVoiceBargeIn,
  requestVoiceEarlyInterject,
  requestVoiceResume,
  requestVoiceSubmit,
  voiceLoopSupported,
} from "@/features/chat/voice/voice-loop-bridge";
import {
  JUDGE_SEND_THRESHOLD,
  judgeTurnComplete,
  reportLayaOutcome,
} from "@/features/chat/voice/voice-models-client";
import { createTurnAudio } from "@/features/chat/voice/turn-audio";
import { voiceHearsAudio } from "@/features/chat/voice/voice-models";
import { attachInputSpectrum, detachInputSpectrum } from "@/features/chat/voice/voice-spectrum";
import { voiceTrace } from "@/features/chat/voice/voice-trace";
import { requestSttDownload } from "@/features/settings/stores/stt-download-prompt-store";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import type { DictationAdapter } from "@assistant-ui/react";
import { SttModelNotDownloadedError } from "./stt-errors";
import {
  loadSttModel,
  transcribeAudioBlob,
} from "./studio-model-dictation-adapter";
import { isMissingDeviceError } from "./studio-web-speech-dictation-adapter";

const wavBlob = (samples: Float32Array, rate: number) =>
  new Blob([encodeWav(samples, rate)], { type: "audio/wav" });

// Thresholds follow the room's ambient RMS; only frames over the speech bar count toward MIN_SPEECH_MS,
// so room noise between the two bars keeps an utterance alive but cannot accumulate into a fake one.
const SILENCE_RMS_MIN = 0.006;
const SPEECH_RMS_MIN = 0.011;
const SILENCE_OVER_FLOOR = 1.6;
const SPEECH_OVER_FLOOR = 2.6;
const SILENCE_HANG_MS = 1000; // silence after speech ends the utterance
// A transcript that reads cut off is held, and the next window gets this longer hang for the rest.
const CONTINUATION_HANG_MS = 1800;
const CONTINUATION_FLUSH_MS = 4000;
const CONTINUATION_MAX_MS = 20000;
const MAX_UTTERANCE_MS = 30000;
const NO_SPEECH_TIMEOUT_MS = 8000;
const MIN_SPEECH_MS = 350;
const ZERO_RUN_TRACE_MS = 200;
// Audio kept from before a barge-in fires, so the first word of what cut in still reaches STT.
const BARGE_PREROLL_MS = 600;
const BARGE_UNCONFIRMED_MS = 450;
const BARGE_CONFIRMED_MS = 250;
const VAD_MIN_SPEECH_MS = 300;
// The probe clip holds only ~BARGE_CONFIRMED_MS of voice: the full bar failed the first check every time.
const BARGE_VAD_MIN_SPEECH_MS = 200;

// Scoped to the detector pick that failed: choosing another one asks again.
let vadDisabled = false;
let vadDisabledFor: string | null = null;
function disableVad(failedPick: string): void {
  vadDisabled = true;
  vadDisabledFor = failedPick;
}
// STT hallucinates on long leading/trailing silence, so the clip is trimmed to the voiced span plus this.
const SILENCE_PAD_MS = 200;
const REVEAL_CHAR_MS = 12;
const REVEAL_MAX_MS = 250;
let reveal: { shown: string; rush: boolean } | null = null;

export function voiceRevealText(): string | null {
  return reveal?.shown ?? null;
}

export function rushVoiceReveal(): void {
  if (reveal) reveal.rush = true;
}

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
// The fragment cut into the reply: the next session keeps listening as that barge-in,
// not as the paused reply's barge gate, which never records and never times out.
let heldFromBarge = false;
// The last transcript this loop put in the composer: the re-arm clears only that, never a typed draft.
let voiceBoxText = "";

export function voiceWroteBox(text: string): boolean {
  return text.trim() !== "" && text.trim() === voiceBoxText.trim();
}

/** Take a draft typed ahead of the loop's own text: it starts the spoken turn, so the box is the turn's now. */
export function takeTypedDraft(box: string): string {
  const own = voiceBoxText.trim();
  const text = box.trim();
  if (!own || text === own || !text.endsWith(own)) return "";
  voiceBoxText = text;
  return text.slice(0, text.length - own.length).trim();
}
// Bumped by clearHeldTurn (voice off, thread switch, stop, typed send): a transcript still
// in STT or the turn judge when it runs belongs to the cleared turn and must not be held.
let heldEpoch = 0;

let micMuted = false;

// A second listen() used to leave an orphan holding the mic and the loop went deaf: the older one is cancelled.
let liveSession: { cancel: () => void; reacquire: (reason: string) => void } | null = null;
export function setMicMuted(muted: boolean): void {
  micMuted = muted;
}

const openMicOnce = createSingleFlight();
const reacquireBackoff = createReacquireBackoff();

// WebKit may leave the track open but silent after the floating orb, so a fresh one is opened.
export function reacquireMic(reason: string): void {
  liveSession?.reacquire(reason);
}

let heldJudgeId: string | null = null;
let finishedJudge: {
  id: string;
  at: number;
  quick: boolean;
  timer: ReturnType<typeof setTimeout> | null;
} | null = null;

function watchFinished(id: string): void {
  if (finishedJudge?.timer) clearTimeout(finishedJudge.timer);
  const entry = { id, at: performance.now(), quick: false, timer: null as ReturnType<typeof setTimeout> | null };
  entry.timer = setTimeout(() => {
    if (finishedJudge === entry) finishedJudge = null;
    reportLayaOutcome(id, "sent");
  }, QUICK_CONTINUE_MS);
  finishedJudge = entry;
}

// A cough clears the gate too, so the continuation report waits for the transcript.
function noteSpeechStart(): void {
  const f = finishedJudge;
  if (!f || f.quick || !f.timer) return;
  clearTimeout(f.timer);
  f.timer = null;
  f.quick = true;
}

function settleFinished(realSpeech: boolean): void {
  const f = finishedJudge;
  if (!f?.quick) return;
  finishedJudge = null;
  const gapMs = Math.round(performance.now() - f.at);
  reportLayaOutcome(f.id, realSpeech ? "continued_quickly" : "sent", { gap_ms: gapMs });
}

async function turnUnfinished(text: string): Promise<boolean> {
  const r = await judgeTurnComplete(text);
  const unfinished = r !== null && r.p < JUDGE_SEND_THRESHOLD;
  heldJudgeId = unfinished ? (r?.id ?? null) : null;
  if (r?.id && !unfinished) watchFinished(r.id);
  return unfinished;
}

export function getHeldTurnText(): string {
  return heldTurn?.text ?? "";
}

// The recorded speech of the turn, kept only while the chat model hears audio (voiceHearsAudio).
const turnAudio = createTurnAudio();

export function takeVoiceTurnAudio(): string | null {
  return turnAudio.take();
}

/** A queued utterance keeps its speech for the turn it is sent with. */
export function parkVoiceTurnAudio(): void {
  turnAudio.park();
}

/** An ignored utterance drops only its own speech. */
export function dropVoiceUtteranceAudio(): void {
  turnAudio.dropCurrent();
}

/** Drop a fragment waiting for its continuation: a thread switch or voice going off. */
export function clearHeldTurn(): void {
  turnAudio.clear();
  heldEpoch += 1;
  heldTurn = null;
  heldJudgeId = null;
  heldEdit.reset();
}

let readComposerText: () => string = () => "";
export function registerComposerTextReader(read: (() => string) | null): void {
  readComposerText = read ?? (() => "");
}

const heldEdit = createHeldEdit({
  readBox: () => readComposerText(),
  onChange: (owned) => useChatRuntimeStore.getState().setVoiceHeldForEdit(owned),
  // The speech kept for that draft goes with it, not into the next turn.
  onCleared: () => turnAudio.clear(),
  trace: voiceTrace,
});

export function claimHeldTurnForEdit(): boolean {
  if (!heldTurn || !heldEdit.claim(heldTurn.text)) return false;
  heldTurn = null;
  heldJudgeId = null;
  heldFromBarge = false;
  return true;
}

export function isComposerHeldForEdit(): boolean {
  return !heldEdit.autoSend();
}

export function cancelVoiceSession(): void {
  turnAudio.clear();
  liveSession?.cancel();
}

export function endHeldEdit(): void {
  heldEdit.sent();
}

// True until the session emits its transcript, so a mic cycle elsewhere can wait instead of losing the turn.
let utteranceInFlight = false;
export function isUtteranceInFlight(): boolean {
  return utteranceInFlight;
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
    // The STT clip starts at the first of these before the first loud frame, so a soft opening word is kept.
    const chunkNotSilent: boolean[] = [];
    let voicedMs = 0;
    // Taken as the session opens: a clear while it still records drops what it hears.
    const epoch = heldEpoch;
    if (vadDisabled && useChatRuntimeStore.getState().selectedVadModelId !== vadDisabledFor) {
      vadDisabled = false;
    }
    let noiseFloor = 0;
    let noiseFrames = 0;
    const bargeGate = createBargeGate(
      vadDisabled ? BARGE_UNCONFIRMED_MS : BARGE_CONFIRMED_MS,
    );
    let vadReady = false;
    let bargeConfirming = false;
    // The gate resets on the first quiet frame, so a word ending before the answer would otherwise count as none.
    let bargePeakMs = 0;
    const preroll: Float32Array[] = [];
    let bargedIn = heldTurn !== null && heldFromBarge;
    let peakRms = 0;
    let zeroMs = 0;
    let zeroRunMs = 0;
    let speechBarSeen = 0;
    const endZeroRun = () => {
      if (zeroRunMs >= ZERO_RUN_TRACE_MS)
        voiceTrace("mic.zeros", { ms: Math.round(zeroRunMs), device: trackLabel });
      zeroRunMs = 0;
    };
    let trackLabel = "";
    let watchFrame: (now: number, live: boolean) => void = () => {};
    let trackMutedAt = 0;
    let sampleRate = 16000;
    let heardSpeech = false;
    const hearingHint = createHearingHint();
    let lastVoiceAt = 0;
    let startedAt = 0;
    let ended = false;
    let resolveEnded: (() => void) | null = null;
    const endedPromise = new Promise<void>((resolve) => {
      resolveEnded = resolve;
    });

    const shown = (transcript: string): string => {
      const parked = getVoiceParkedText();
      return parked && transcript ? `${parked} ${transcript}` : parked || transcript;
    };
    const emitSpeech = (transcript: string, isFinal: boolean) => {
      voiceBoxText = shown(transcript);
      for (const cb of speechCallbacks) cb({ transcript: voiceBoxText, isFinal });
    };

    const session: DictationAdapter.Session = {
      status: { type: "starting" },
      stop: async () => {
        await finalize();
        await endedPromise;
      },
      cancel: () => {
        teardown();
        early.clear();
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
        // Each emitted transcript is the whole turn, so parked turns and the held fragment go through it (pre-filling doubled it).
        const lead = shown(heldTurn?.text ?? "");
        if (lead) {
          voiceBoxText = lead;
          cb({ transcript: lead, isFinal: false });
        }
        return () => speechCallbacks.delete(cb);
      },
    };
    if (liveSession) {
      voiceTrace("session.supersede", {});
      liveSession.cancel();
    }
    const handle = {
      cancel: () => session.cancel(),
      reacquire: (reason: string) => {
        voiceTrace("mic.reacquire", { reason });
        reacquireBackoff.tried(performance.now());
        reopen(reason);
      },
    };
    liveSession = handle;
    voiceTrace("session.start", { playback: isVoicePlaybackActive() });
    const reopen = (why: string, detail?: Record<string, unknown>) => {
      if (ended) return;
      // Mid-utterance the dead mic may deliver no frame to end it: send what was heard, then re-arm.
      if (utteranceInFlight) {
        voiceTrace("session.reopenFinalize", { why, ...detail });
        void finalize();
        return;
      }
      voiceTrace("session.reopen", { why, ...detail });
      teardown();
      finish("stopped");
      setTimeout(() => requestVoiceResume(), 0);
    };
    let unsubMic: (() => void) | null = useVoiceSettingsStore.subscribe(
      (state, prev) => {
        if (state.micDeviceId !== prev.micDeviceId) reopen("micSwitch", { to: state.micDeviceId });
      },
    );
    // Output routes change too (macOS fires this when a reply plays): only a changed set of inputs reopens.
    let inputIds: string | null = null;
    const audioInputIds = async () => {
      const devices = await navigator.mediaDevices.enumerateDevices?.();
      return (devices ?? [])
        .filter((d) => d.kind === "audioinput")
        .map((d) => d.deviceId)
        .sort()
        .join(",");
    };
    void audioInputIds().then((ids) => {
      if (inputIds === null) inputIds = ids;
    });
    const onDevicesChanged = () => {
      void audioInputIds().then((ids) => {
        if (inputIds !== null && ids === inputIds) return;
        inputIds = ids;
        reopen("deviceChange");
      });
    };
    navigator.mediaDevices.addEventListener?.("devicechange", onDevicesChanged);
    // WebKit mutes a capture track while the window is away and may never unmute it: reopen it on focus.
    const onFocus = () => {
      const t = stream?.getAudioTracks()[0];
      if (t?.muted) reopen("mutedOnFocus", { mutedMs: Math.round(performance.now() - trackMutedAt) });
    };
    window.addEventListener("focus", onFocus);

    let silenceTimer: ReturnType<typeof setInterval> | null = null;
    const teardown = () => {
      if (silenceTimer) clearInterval(silenceTimer);
      silenceTimer = null;
      unsubMic?.();
      unsubMic = null;
      navigator.mediaDevices.removeEventListener?.("devicechange", onDevicesChanged);
      window.removeEventListener("focus", onFocus);
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
      detachInputSpectrum();
      void audioCtx?.close().catch(() => {});
      audioCtx = null;
    };

    const finish = (
      reason: "stopped" | "cancelled" | "error",
      transcript?: string,
    ) => {
      if (ended) return;
      ended = true;
      utteranceInFlight = false;
      if (liveSession === handle) liveSession = null;
      endZeroRun();
      voiceTrace("session.finish", {
        reason,
        transcript: transcript ?? null,
        bargedIn,
        peakRms: Number(peakRms.toFixed(4)),
        voicedMs: Math.round(voicedMs),
        speechBar: Number(speechBarSeen.toFixed(4)),
        zeroMs: Math.round(zeroMs),
        device: trackLabel,
      });
      session.status = { type: "ended", reason };
      if (transcript) {
        for (const cb of speechEndCallbacks) cb({ transcript });
      }
      resolveEnded?.();
    };

    const detectClip = async (
      blob: Blob,
      minSpeechMs: number,
    ): Promise<{ kept: boolean; speech_ms: number } | null> => {
      if (!vadReady || vadDisabled) return null;
      const vadPick = useChatRuntimeStore.getState().selectedVadModelId;
      try {
        const { speech_ms } = await detectSpeech(blob, vadPick);
        return { kept: speech_ms >= minSpeechMs, speech_ms };
      } catch (error) {
        if (error instanceof VadUnavailableError) disableVad(vadPick);
        return null;
      }
    };
    const publishVad = (vad: { kept: boolean; speech_ms: number } | null): boolean | null => {
      if (!vad) return null;
      publishVadDecision(vad.kept);
      voiceTrace("vad", { ...vad, bargedIn, bargeConfirming });
      return vad.kept;
    };
    const clipHasSpeech = async (blob: Blob, minSpeechMs: number) =>
      publishVad(await detectClip(blob, minSpeechMs));

    type ClipAnalysis = {
      modelId: string | undefined;
      samples: Float32Array;
      vad: Promise<{ kept: boolean; speech_ms: number } | null>;
      text: Promise<string>;
      cancel(): void;
    };
    const padFrames = () => {
      const frameMs = chunks[0] ? (chunks[0].length / sampleRate) * 1000 : 0;
      return frameMs > 0 ? Math.ceil(SILENCE_PAD_MS / frameMs) : 0;
    };
    const clipKey = (span: { start: number; end: number }) => {
      const rt = useChatRuntimeStore.getState();
      const vad = vadReady && !vadDisabled ? rt.selectedVadModelId : "off";
      return `${span.start}:${span.end}:${rt.selectedSttModelId ?? ""}:${vad}`;
    };
    const analyseClip = (span: { start: number; end: number }): ClipAnalysis => {
      const total = chunks.slice(span.start, span.end + 1).reduce((n, c) => n + c.length, 0);
      const merged = new Float32Array(total);
      let offset = 0;
      for (let i = span.start; i <= span.end; i++) {
        merged.set(chunks[i], offset);
        offset += chunks[i].length;
      }
      const wav = wavBlob(merged, sampleRate);
      const modelId = useChatRuntimeStore.getState().selectedSttModelId ?? undefined;
      const abort = new AbortController();
      const vad = detectClip(wav, bargedIn ? BARGE_VAD_MIN_SPEECH_MS : VAD_MIN_SPEECH_MS);
      const text = vad.then(async (v) =>
        v?.kept === false
          ? ""
          : (await transcribeAudioBlob(wav, { model: modelId, signal: abort.signal })).trim(),
      );
      text.catch(() => {});
      return { modelId, samples: merged, vad, text, cancel: () => abort.abort() };
    };
    const early = createClipSpeculation<ClipAnalysis>();

    const endWithoutSpeech = () => {
      voiceTrace("session.noSpeech", { bargedIn });
      settleFinished(false);
      if (heldTurn) {
        // The rest never came: the fragment was the whole turn after all (the
        // submit cuts the reply it barged into, so no abort here).
        const text = heldTurn.text;
        heldTurn = null;
        reportLayaOutcome(heldJudgeId, "flushed", { text });
        heldJudgeId = null;
        emitSpeech(text, true);
        finish("stopped", text);
        setTimeout(() => requestVoiceSubmit(text), 0);
        return;
      }
      if (bargedIn) requestVoiceBargeAbort();
      finish("stopped");
      setTimeout(() => requestVoiceResume(), 0);
    };

    let finalizing = false;
    const finalize = async () => {
      if (finalizing || ended) return;
      finalizing = true;
      teardown();

      // A barge-in already passed its own bar: a quick "stop" is shorter than a turn.
      const minMs = bargedIn ? BARGE_CONFIRMED_MS : MIN_SPEECH_MS;
      if (!heardSpeech || chunks.length === 0 || voicedMs < minMs) {
        endWithoutSpeech();
        return;
      }

      const span = utteranceSpan(chunkVoiced, chunkNotSilent, padFrames());
      if (!span) {
        endWithoutSpeech();
        return;
      }
      voiceTrace("session.finalize", {
        trailingMs: Math.round((chunks.length - 1 - chunkVoiced.lastIndexOf(true)) * (chunks[0].length / sampleRate) * 1000),
      });
      const key = clipKey(span);
      const ready = early.take(key);
      const analysis = ready ?? analyseClip(span);
      voiceTrace("stt.clip", { early: ready !== null });
      if (publishVad(await analysis.vad) === false) {
        if (ended) return;
        endWithoutSpeech();
        return;
      }
      if (ended) return;

      try {
        const modelId = analysis.modelId;
        const store = useChatRuntimeStore.getState();
        store.setVoiceTranscribing(true);
        let transcript = "";
        try {
          transcript = await analysis.text;
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
        // Voice was turned off while this clip was heard: its words are discarded, never held for the next session.
        if (getVoiceMode() === "off") {
          finish("cancelled");
          return;
        }
        if (transcript && isHallucinatedTranscript(transcript)) {
          publishSttDecision(false, "repeated word");
          transcript = "";
        }
        voiceTrace("stt", { transcript, bargedIn, heldTurn: heldTurn?.text ?? null });
        if (transcript && bargedIn && isReplyEcho(transcript, getVoiceReplyText())) {
          voiceTrace("barge.abort", { why: "echo" });
          settleFinished(false);
          publishSttDecision(false, "echo of the reply");
          requestVoiceBargeAbort();
          finish("stopped");
          setTimeout(() => requestVoiceResume(), 0);
          return;
        }
        if (transcript && (isBackchannel(transcript) || isSttNoise(transcript))) {
          voiceTrace("stt.drop", { transcript, why: "noise" });
          settleFinished(false);
          publishSttDecision(false, isBackchannel(transcript) ? "breath" : "cough");
          if (bargedIn) {
            voiceTrace("barge.abort", { why: "backchannel" });
            requestVoiceBargeAbort();
          }
          finish("stopped");
          setTimeout(() => requestVoiceResume(), 0);
          return;
        }
        settleFinished(Boolean(transcript));
        // Clicked into: what was said joins the box and waits for the user's send.
        // Decided before keeping this speech, so an emptied draft's audio is dropped first.
        const editing = !heldEdit.autoSend();
        const rt = useChatRuntimeStore.getState();
        // A clear during STT already emptied the buffer; this turn's speech must not refill it.
        if (
          transcript &&
          epoch === heldEpoch &&
          voiceHearsAudio(rt, useVoiceSettingsStore.getState().micTarget)
        ) {
          turnAudio.add(analysis.samples, sampleRate);
        }
        if (transcript && heldTurn) {
          transcript = joinTurn(heldTurn.text, transcript);
          reportLayaOutcome(heldJudgeId, "continued", { text: transcript });
          heldJudgeId = null;
        }
        const unfinished =
          Boolean(transcript) &&
          !editing &&
          (await turnUnfinished(transcript));
        if (epoch !== heldEpoch) {
          voiceTrace("stt.drop", { transcript, why: "cleared" });
          finish("stopped");
          // The clear's own re-arm found this session still live and skipped.
          setTimeout(() => requestVoiceResume(), 0);
          return;
        }
        if (
          unfinished &&
          performance.now() - (heldTurn?.at ?? performance.now()) <
            CONTINUATION_MAX_MS
        ) {
          heldTurn = {
            text: transcript,
            at: heldTurn?.at ?? performance.now(),
          };
          heldFromBarge = bargedIn;
          requestVoiceEarlyInterject(transcript);
          emitSpeech(transcript, false);
          finish("stopped");
          setTimeout(() => requestVoiceResume(), 0);
          return;
        }
        // An empty continuation keeps the held phrase: the next silence still flushes it.
        if (transcript) heldTurn = null;
        if (transcript) {
          const total = transcript.length;
          const stepChars = Math.max(
            1,
            Math.ceil(total / Math.max(1, REVEAL_MAX_MS / REVEAL_CHAR_MS)),
          );
          const mine = { shown: shown(transcript), rush: false };
          reveal = mine;
          try {
            for (let i = stepChars; i < total && !mine.rush; i += stepChars) {
              if (ended || epoch !== heldEpoch) break;
              const partial = transcript.slice(0, i);
              emitSpeech(partial, false);
              await new Promise<void>((r) => setTimeout(r, REVEAL_CHAR_MS));
            }
          } finally {
            if (reveal === mine) reveal = null;
          }
          if (ended) return;
          if (epoch !== heldEpoch) {
            // A Stop or reset during the reveal let this turn go.
            voiceTrace("stt.drop", { transcript, why: "cleared" });
            finish("stopped");
            setTimeout(() => requestVoiceResume(), 0);
            return;
          }
          // Speech then finish land before the deferred submit; requestVoiceSubmit no-ops outside voice mode.
          emitSpeech(transcript, true);
          finish("stopped", transcript);
          if (editing) {
            voiceTrace("hold.userEditAppend", { transcript });
            setTimeout(() => requestVoiceResume(), 0);
          } else {
            setTimeout(() => requestVoiceSubmit(transcript), 0);
          }
        } else {
          if (bargedIn) {
            voiceTrace("barge.abort", { why: "empty" });
            requestVoiceBargeAbort();
          }
          finish("stopped");
          setTimeout(() => requestVoiceResume(), 0);
        }
      } catch (error) {
        if (ended) return;
        console.error("Voice transcription error:", error);
        toast.error(translate("voiceControls.transcriptionFailed"));
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
        // One open at a time; a session cancelled while it waited stops its own tracks before the next opens.
        stream = await openMicOnce(async () => {
          if (ended) return null;
          let opened: MediaStream;
          try {
            opened = await navigator.mediaDevices.getUserMedia({
              audio: pinnedMic
                ? { ...baseAudio, deviceId: { exact: micDeviceId } }
                : baseAudio,
            });
          } catch (error) {
            // Same fallback as the Dictate adapters: a pinned headset that is
            // unplugged must not kill the loop, just lose its pin for this turn.
            if (pinnedMic && isMissingDeviceError(error)) {
              opened = await navigator.mediaDevices.getUserMedia({
                audio: baseAudio,
              });
            } else {
              throw error;
            }
          }
          if (ended) {
            opened.getTracks().forEach((t) => t.stop());
            return null;
          }
          return opened;
        });
        if (ended || !stream) {
          teardown();
          return;
        }
        audioCtx = new AudioCtx();
        // Created outside the user gesture: a suspended context never fires audioprocess.
        await audioCtx.resume();
        // A cancel during that await tore the context down.
        if (ended || !audioCtx) {
          teardown();
          return;
        }
        sampleRate = audioCtx.sampleRate;
        for (const track of stream.getAudioTracks()) {
          trackLabel = track.label;
          track.addEventListener("ended", () => reopen("trackEnded", { label: track.label }));
          track.addEventListener("mute", () => {
            trackMutedAt = performance.now();
            voiceTrace("mic.muted", { label: track.label });
          });
          track.addEventListener("unmute", () => {
            voiceTrace("mic.unmuted", { label: track.label, mutedMs: Math.round(performance.now() - trackMutedAt) });
            const t = performance.now();
            startedAt = t;
            lastVoiceAt = t;
          });
        }
        voiceTrace("mic.open", { label: trackLabel, muted: stream.getAudioTracks()[0]?.muted ?? null });
        source = audioCtx.createMediaStreamSource(stream);
        attachInputSpectrum(audioCtx, source);
        processor = audioCtx.createScriptProcessor(4096, 1, 1);
        startedAt = performance.now();
        lastVoiceAt = startedAt;
        session.status = { type: "running" };
        // An open track that never delivers a non-zero sample is a dead mic, not a quiet room.
        const silence = createSilenceWatch(startedAt);
        watchFrame = (now, live) => {
          silence.frame(now, live);
          if (live) reacquireBackoff.heard();
        };
        silenceTimer = setInterval(() => {
          const now = performance.now();
          if (ended || finalizing || micMuted) {
            silence.reset(now);
            return;
          }
          // Backoff first: due() fires once, so asking it while the backoff refuses would end recovery for good.
          if (!reacquireBackoff.allow(now) || !silence.due(now)) return;
          handle.reacquire("silent");
        }, 250);

        void loadSttModel(
          useChatRuntimeStore.getState().selectedSttModelId ??
            useVoiceSettingsStore.getState().sttModel,
        ).catch(() => {});
        if (!vadDisabled) {
          const vadPick = useChatRuntimeStore.getState().selectedVadModelId;
          void loadVadModel(vadPick)
            .then(() => {
              vadReady = true;
              useChatRuntimeStore.getState().setVadSlotLoaded(true);
            })
            .catch((error: unknown) => {
              if (error instanceof VadUnavailableError) disableVad(vadPick);
              useChatRuntimeStore.getState().setVadSlotLoaded(false);
            });
        }

        let wasMuted = false;
        processor.onaudioprocess = (event) => {
          if (ended || finalizing) return;
          if (micMuted || stream?.getAudioTracks()[0]?.muted) {
            wasMuted = true;
            return;
          }
          if (wasMuted) {
            wasMuted = false;
            const t = performance.now();
            startedAt = t;
            lastVoiceAt = t;
          }
          const input = event.inputBuffer.getChannelData(0);

          let sumSquares = 0;
          for (let i = 0; i < input.length; i++)
            sumSquares += input[i] * input[i];
          const rms = Math.sqrt(sumSquares / input.length);
          if (rms > peakRms) peakRms = rms;
          const now = performance.now();
          watchFrame(now, sumSquares > 0);

          const frameMs = (input.length / sampleRate) * 1000;
          if (sumSquares === 0) {
            zeroMs += frameMs;
            zeroRunMs += frameMs;
          } else if (zeroRunMs > 0) {
            endZeroRun();
          }
          // While the reply plays (or a barge-in probe is out, keeping frames in order) only a barge-in is listened for.
          if (!bargedIn && (isVoicePlaybackActive() || bargeConfirming)) {
            preroll.push(new Float32Array(input));
            // While the detector is asked, keep every frame: the probe's audio opens the utterance.
            if (!bargeConfirming) while (preroll.length * frameMs > BARGE_PREROLL_MS) preroll.shift();
            startedAt = now;
            lastVoiceAt = now;
            const cut = stepBargeGate(bargeGate, rms, frameMs);
            if (bargeConfirming) bargePeakMs = Math.max(bargePeakMs, bargeGate.loudMs);
            const rt = useChatRuntimeStore.getState();
            const hint = hearingHint.step(
              bargeGate.loudMs > 0,
              rt.voiceOrbState,
              now,
            );
            if (hint !== null) {
              voiceTrace("orb.hearingHint", {
                on: hint,
                orb: rt.voiceOrbState,
              });
              rt.setVoiceHearing(hint);
            }
            if (!cut) return;
            const commitBarge = () => {
              bargedIn = true;
              hearingHint.commit();
              voiceTrace("barge.commit", { loudMs: bargeGate.loudMs });
              requestVoiceBargeIn();
              for (const frame of preroll.splice(0)) {
                chunks.push(frame);
                chunkVoiced.push(true);
                chunkNotSilent.push(true);
              }
              voicedMs = Math.max(bargeGate.loudMs, bargePeakMs);
              bargePeakMs = 0;
              heardSpeech = true;
              utteranceInFlight = true;
              noteSpeechStart();
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
            void clipHasSpeech(wavBlob(probe, sampleRate), BARGE_VAD_MIN_SPEECH_MS).then((voiced) => {
              bargeConfirming = false;
              if (ended || bargedIn) return;
              if (voiced === false) {
                bargeGate.loudMs = 0;
                bargePeakMs = 0;
                if (hearingHint.reject() === false)
                  useChatRuntimeStore.getState().setVoiceHearing(false);
                return;
              }
              commitBarge();
            });
            return;
          }
          // Playback ended before the barge-in gate fired: the preroll is this utterance's opening.
          for (const frame of preroll.splice(0)) {
            let sq = 0;
            for (let i = 0; i < frame.length; i++) sq += frame[i] * frame[i];
            const level = Math.sqrt(sq / frame.length);
            const loud = level >= Math.max(SPEECH_RMS_MIN, noiseFloor * SPEECH_OVER_FLOOR);
            chunks.push(frame);
            chunkVoiced.push(loud);
            chunkNotSilent.push(level >= Math.max(SILENCE_RMS_MIN, noiseFloor * SILENCE_OVER_FLOOR));
            if (loud) voicedMs += frameMs;
          }
          chunks.push(new Float32Array(input));
          if (
            hearingHint.on &&
            !heardSpeech &&
            hearingHint.step(false, null, now) === false
          ) {
            useChatRuntimeStore.getState().setVoiceHearing(false);
          }
          // notSilent keeps the utterance alive (resets the silence-hang timer);
          // speech is the stricter bar that actually counts toward transcription,
          // so noise/clicks between the two thresholds never reach STT.
          const speechBar = Math.max(SPEECH_RMS_MIN, noiseFloor * SPEECH_OVER_FLOOR);
          speechBarSeen = speechBar;
          const notSilent = rms >= Math.max(SILENCE_RMS_MIN, noiseFloor * SILENCE_OVER_FLOOR);
          const speech = rms >= speechBar;
          if (!heardSpeech && !speech) {
            // After the first quiet frames the floor drifts slowly, so a soft word does not lift it under itself.
            noiseFrames++;
            const w = noiseFrames <= 4 ? 1 / noiseFrames : 0.05;
            noiseFloor += (rms - noiseFloor) * w;
          }
          chunkVoiced.push(speech);
          chunkNotSilent.push(notSilent);
          if (notSilent) lastVoiceAt = now;
          if (speech || (voicedMs > 0 && !heardSpeech)) {
            if (speech) voicedMs += frameMs;
            if (!heardSpeech) {
              heardSpeech = true;
              utteranceInFlight = true;
              noteSpeechStart();
              useChatRuntimeStore.getState().setVoiceHearing(true);
              for (const cb of speechStartCallbacks) cb();
            }
          }
          const sinceVoice = now - lastVoiceAt;
          const elapsed = now - startedAt;
          if (heardSpeech && !finalizing && voicedMs >= (bargedIn ? BARGE_CONFIRMED_MS : MIN_SPEECH_MS) && sinceVoice >= SPECULATE_AFTER_MS) {
            const span = utteranceSpan(chunkVoiced, chunkNotSilent, padFrames());
            if (span) early.start(clipKey(span), () => analyseClip(span));
          }
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
        const name = error instanceof Error ? error.name : "";
        const message = error instanceof Error ? error.message : String(error);
        voiceTrace("session.error", { name, message, ended });
        // A session cancelled while the mic was still opening is not a mic fault.
        if (!ended) {
          toast.error("Could not access the microphone for dictation.", {
            description: name ? `${name}: ${message}` : message,
          });
        }
        finish("error");
      }
    })();

    return session;
  }
}
