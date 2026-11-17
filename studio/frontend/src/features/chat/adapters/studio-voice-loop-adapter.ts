// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The store directly: the feature barrel imports back here through the dictation dispatcher.
import { encodeWav } from "@/features/chat/adapters/pcm-recorder";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  bargeOnset,
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
import type { LiveTranscript } from "@/features/chat/voice/live-transcript";
import { QUICK_CONTINUE_MS } from "@/features/chat/voice/laya-outcomes";
import {
  createReacquireBackoff,
  createSilenceWatch,
  createSingleFlight,
} from "@/features/chat/voice/mic-health";
import {
  isAcknowledgement,
  isBackchannel,
  answersClosingQuestion,
  isReplyEcho,
  isSttNoise,
  joinTurn,
} from "@/features/chat/voice/turn-complete";
import {
  VadUnavailableError,
  detectSpeech,
  loadVadModel,
  minSpeechMs,
  vadFallback,
} from "@/features/chat/voice/vad-client";
import {
  publishSttDecision,
  publishVadDecision,
} from "@/features/chat/voice/voice-decisions";
import {
  getVoiceParkedText,
  getVoiceReplyText,
  getVoiceMode,
  voiceOffEpoch,
  isVoicePlaybackActive,
  requestVoiceBargeAbort,
  requestVoiceBargeIn,
  requestVoiceEarlyInterject,
  requestVoiceResume,
  requestVoiceSubmit,
  voiceLoopSupported,
} from "@/features/chat/voice/voice-loop-bridge";
import {
  judgeTurnComplete,
  reportLayaOutcome,
} from "@/features/chat/voice/voice-models-client";
import { createTurnAudio } from "@/features/chat/voice/turn-audio";
import { TURN_THRESHOLDS, fuseTurn } from "@/features/chat/voice/turn-fuser";
import {
  TURN_ASK_AFTER_MS,
  TURN_END_AFTER_MS,
  askTurnEnd,
  createTurnEndCheck,
  hasClosingMark,
  readsAsQuestion,
  loadTurnModel,
  readsUnfinished,
  turnClipSpan,
  turnClipTail,
} from "@/features/chat/voice/turn-end";
import { voiceHearsAudio } from "@/features/chat/voice/voice-models";
import { attachInputSpectrum, detachInputSpectrum } from "@/features/chat/voice/voice-spectrum";
import { voiceTrace } from "@/features/chat/voice/voice-trace";
import { isTrackingSttDownload } from "@/features/settings/lib/stt-download-mirror";
import { requestSttDownload } from "@/features/settings/stores/stt-download-prompt-store";
import { useVoiceSettingsStore } from "@/features/settings/stores/voice-settings-store";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import type { DictationAdapter } from "@assistant-ui/react";
import { SttModelNotDownloadedError } from "./stt-errors";
import {
  loadSttModel,
  openSttLive,
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
// Held while the reply plays so a barge-in keeps the words before the gate fired (bargeOnset).
const BARGE_LOOKBACK_MS = 2500;
// A pause inside an interruption ("Wait, stop. Make it ...") is still the same interruption.
const BARGE_GAP_MS = 350;
const BARGE_UNCONFIRMED_MS = 450;
const BARGE_CONFIRMED_MS = 250;

// The transcriber the download prompt was last raised for as listening began: a dismissed prompt
// is not raised again every time the mic re-arms (an utterance that needs the model still asks).

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

let heldTurn: { text: string; at: number; turnId: string | null } | null = null;
// Not randomUUID alone: Studio on a LAN address is not a secure context.
const newTurnId = (): string =>
  globalThis.crypto?.randomUUID?.() ?? `t${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
// The fragment cut into the reply: the next session keeps listening as that barge-in,
// not as the paused reply's barge gate, which never records and never times out.
let heldFromBarge = false;
// The last transcript this loop put in the composer: the re-arm clears only that, never a typed draft.
let voiceBoxText = "";

export function voiceWroteBox(text: string): boolean {
  return text.trim() !== "" && text.trim() === voiceBoxText.trim();
}

// The box minus this loop's last transcript, or null when that transcript is not its tail:
// dictation prepends a typed draft, so a held fragment sits after it and must go before re-arming.
// Once a send has cleared the box, its transcript no longer claims later drafts; a newer utterance's does.
export function voiceTextReleaser(): () => void {
  const said = voiceBoxText;
  return () => {
    if (voiceBoxText === said) voiceBoxText = "";
  };
}

export function withoutVoiceText(text: string): string | null {
  const said = voiceBoxText.trim();
  const box = text.trimEnd();
  if (!said || !box.endsWith(said)) return null;
  const before = box.slice(0, box.length - said.length);
  // Only a whole tail: a typed word that merely ends in the same letters stays.
  if (before && !/\s$/.test(before)) return null;
  return before.trimEnd();
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

// The transcriber that has written a closing mark: only then does a missing one mean anything.
let punctuatingStt: string | null = null;
const sttKey = (modelId: string | undefined) => modelId ?? "";
const readsCutOff = (text: string, modelId: string | undefined) =>
  readsUnfinished(text, punctuatingStt === sttKey(modelId));

// soundedP: Smart Turn called the turn finished with this probability, and the judge must agree too
// (the "Smart Turn + judge" lane, fuseTurn); null asks the judge alone.
async function turnUnfinished(
  text: string,
  modelId: string | undefined,
  soundedP: number | null = null,
  judgeBar: number = TURN_THRESHOLDS.text,
): Promise<boolean> {
  const r = await judgeTurnComplete(text);
  // No decision model to ask: a transcript that reads cut off is still held for the rest.
  if (r === null) return readsCutOff(text, modelId);
  const unfinished =
    soundedP === null ? !r.finished : fuseTurn({ pText: r.p, pAudio: soundedP }, { text: judgeBar, audio: TURN_THRESHOLDS.audio }) !== "confirmed";
  heldJudgeId = unfinished ? (r?.id ?? null) : null;
  if (r?.id && !unfinished) watchFinished(r.id);
  return unfinished;
}

export function getHeldTurnText(): string {
  return heldTurn?.text ?? "";
}

let utterancesPending = 0;
// Hearing ends before VAD and STT run: the coalesced reply must still wait for this phrase.
export function voiceUtterancePending(): boolean {
  return utterancesPending > 0;
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

// Thread switch or voice off: the box it named belongs to the chat being left, so a matching draft in
// the next one is the user's. (Not in clearHeldTurn: a stop or typed send keeps this chat's box.)
export function forgetVoiceBox(): void {
  voiceBoxText = "";
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
    const offAtOpen = voiceOffEpoch();
    // Off now, or off and back on since this recording opened: either way its words are dropped.
    const switchedOff = () => getVoiceMode() === "off" || voiceOffEpoch() !== offAtOpen;
    // Taken as the session opens: a clear while it still records drops what it hears.
    const epoch = heldEpoch;
    if (vadDisabled && useChatRuntimeStore.getState().selectedVadModelId !== vadDisabledFor) {
      vadDisabled = false;
    }
    let noiseFloor = 0;
    let noiseFrames = 0;
    // A thread switch or Stop queued behind finish() resets the turn: the old words never submit into it.
    // A re-arm may have given up while this clip was in STT: voice still on listens again.
    const dropSwitched = () => {
      finish("cancelled");
      if (getVoiceMode() === "active") setTimeout(() => requestVoiceResume(), 0);
    };
    const submitLater = (text: string, meta?: { turnId?: string; confirmedEarly?: boolean }) =>
      setTimeout(() => {
        if (!switchedOff() && epoch === heldEpoch) requestVoiceSubmit(text, meta);
      }, 0);
    let prerollVoicedMs = 0;
    const bargeGate = createBargeGate(
      vadDisabled ? BARGE_UNCONFIRMED_MS : BARGE_CONFIRMED_MS,
    );
    let vadReady = false;
    // The end-of-turn model is up, and it called this turn finished.
    let turnReady = false;
    let soundedFinished = false;
    // The early transcript of this clip ends on a question mark (readsAsQuestion).
    let questionKey: string | null = null;
    let questionReadKey: string | null = null;
    // Smart Turn's last answer for this turn, for the "Smart Turn + judge" lane.
    let turnP: number | null = null;
    // One id per turn for the voice trace; a continuation keeps the id of the fragment it joins.
    let turnId: string | null = null;
    const startTurn = () => {
      turnId ??= heldTurn?.turnId ?? newTurnId();
    };
    let bargeConfirming = false;
    // The gate resets on the first quiet frame, so a word ending before the answer would otherwise count as none.
    let bargePeakMs = 0;
    const preroll: Float32Array[] = [];
    // Per preroll frame: it was over the barge-in bar.
    const prerollLoud: boolean[] = [];
    // The utterance going to a transcriber that follows along as it is spoken (live-transcript.ts):
    // `fed` is the next chunk it has not been given.
    let live: { session: LiveTranscript; modelId: string | undefined; fed: number } | null = null;
    // A pause already took the live transcript: only more speech starts another.
    let liveTaken = false;
    // This pick gives no live transcript: it is not asked again this session.
    let liveOff = false;
    // The longest live text put in the box, so the finished transcript is not typed out again.
    let liveShown = "";
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
        turnEnd.clear();
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
      live?.session.drop();
      live = null;
      // Words shown as they were heard, then not kept: the box goes back to what it held.
      if (!transcript && liveShown) emitSpeech(heldTurn?.text ?? "", false);
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
      barge: boolean,
    ): Promise<{ kept: boolean; speech_ms: number } | null> => {
      if (!vadReady || vadDisabled) return null;
      const vadPick = useChatRuntimeStore.getState().selectedVadModelId;
      try {
        const { speech_ms, bars } = await detectSpeech(blob, vadPick);
        return { kept: speech_ms >= minSpeechMs(bars, barge), speech_ms };
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
    const clipHasSpeech = async (blob: Blob, barge: boolean) =>
      publishVad(await detectClip(blob, barge));

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
    const spanSamples = (span: { start: number; end: number }): Float32Array => {
      const total = chunks.slice(span.start, span.end + 1).reduce((n, c) => n + c.length, 0);
      const merged = new Float32Array(total);
      let offset = 0;
      for (let i = span.start; i <= span.end; i++) {
        merged.set(chunks[i], offset);
        offset += chunks[i].length;
      }
      return merged;
    };
    const feedLive = () => {
      if (!live) return;
      while (live.fed < chunks.length) live.session.feed(chunks[live.fed++]);
    };
    // Opens a live transcript of the utterance so far and keeps it fed; a pick that cannot give
    // one leaves `live` empty, and the clip is transcribed when it ends, as before.
    const startLive = () => {
      const span = utteranceSpan(chunkVoiced, chunkNotSilent, padFrames());
      if (!span) return;
      const modelId = useChatRuntimeStore.getState().selectedSttModelId ?? undefined;
      const session = openSttLive(sampleRate, modelId, (text) => {
        // Words over a reply may be its echo or an "mm-hm", and a box being edited is the user's.
        if (ended || bargedIn || !heldEdit.autoSend()) return;
        // A transcript restarted after a pause catches up before it adds anything.
        if (text.length < liveShown.length) return;
        liveShown = text;
        emitSpeech(heldTurn ? joinTurn(heldTurn.text, text) : text, false);
      });
      liveTaken = false;
      if (!session) {
        liveOff = true;
        return;
      }
      voiceTrace("stt.live", { from: span.start, frames: chunks.length - span.start });
      live = { session, modelId, fed: span.start };
      feedLive();
    };
    const turnEnd = createTurnEndCheck(
      (wav, signal) => askTurnEnd(wav, signal, useVoiceSettingsStore.getState().dictationLanguage),
      (answer, ms) => {
        turnP = answer.p;
        voiceTrace("turn.end", { ...answer, ms, turnId });
      },
    );
    // Words spoken over a reply that ask nothing of it: the reply carries on.
    const nothingToActOn = (text: string): boolean => {
      const reply = getVoiceReplyText();
      return (
        !text ||
        isBackchannel(text) ||
        isSttNoise(text) ||
        isReplyEcho(text, reply) ||
        isAcknowledgement(text, reply)
      );
    };
    // The barge-in clip whose words turned out to be nothing to act on, and the one being read.
    let bargeNothingKey: string | null = null;
    let bargeReadingKey: string | null = null;
    const analyseClip = (span: { start: number; end: number }): ClipAnalysis => {
      const merged = spanSamples(span);
      const wav = wavBlob(merged, sampleRate);
      const modelId = useChatRuntimeStore.getState().selectedSttModelId ?? undefined;
      let dropped = false;
      const vad = detectClip(wav, bargedIn);
      // The live transcript has heard all of this already: ending it gives the text at once.
      let heard: Promise<string> | null = null;
      if (live && live.modelId === modelId) {
        feedLive();
        heard = live.session.finish();
        heard.catch(() => {});
        live = null;
        liveTaken = true;
      }
      // A clip dropped before its transcription starts is never sent. One already in flight is left
      // to finish: the server answers an aborted transcription by restarting the transcriber, and
      // that model load costs the next clip far more than these few hundred milliseconds.
      const text = vad.then(async (v) => {
        if (v?.kept === false || dropped) return "";
        if (heard) {
          try {
            return await heard;
          } catch (error) {
            voiceTrace("stt.liveFailed", { message: error instanceof Error ? error.message : String(error) });
          }
        }
        return (await transcribeAudioBlob(wav, { model: modelId })).trim();
      });
      text.catch(() => {});
      return {
        modelId,
        samples: merged,
        vad,
        text,
        cancel: () => {
          dropped = true;
        },
      };
    };
    const early = createClipSpeculation<ClipAnalysis>();

    const endWithoutSpeech = () => {
      voiceTrace("session.noSpeech", { bargedIn });
      settleFinished(false);
      if (heldTurn) {
        // The rest never came: the fragment was the whole turn after all (the
        // submit cuts the reply it barged into, so no abort here).
        const text = heldTurn.text;
        const heldId = heldTurn.turnId ?? undefined;
        heldTurn = null;
        reportLayaOutcome(heldJudgeId, "flushed", { text });
        heldJudgeId = null;
        emitSpeech(text, true);
        finish("stopped", text);
        voiceTrace("turn.confirmed", { turnId: heldId, flushed: true });
        submitLater(text, { turnId: heldId });
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
      utterancesPending += 1;
      try {
        await finalizeHeard();
      } finally {
        utterancesPending -= 1;
      }
    };
    const finalizeHeard = async () => {
      // Speech ended this long before this mark; a continuation records another, and the last counts.
      voiceTrace("turn.speechEnd", { turnId, silenceMs: Math.round(performance.now() - lastVoiceAt) });
      turnEnd.clear();

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
        why: soundedFinished ? "sounded finished" : "silence",
        trailingMs: Math.round((chunks.length - 1 - chunkVoiced.lastIndexOf(true)) * (chunks[0].length / sampleRate) * 1000),
      });
      const key = clipKey(span);
      // Words heard across a voice off or reset are dropped anyway: no VAD or STT run for them.
      if (switchedOff()) {
        early.clear();
        dropSwitched();
        return;
      }
      const ready = early.take(key);
      const analysis = ready ?? analyseClip(span);
      voiceTrace("stt.clip", { early: ready !== null });
      const vadResult = await analysis.vad;
      voiceTrace("turn.vadDone", { turnId, kept: vadResult?.kept ?? null });
      if (publishVad(vadResult) === false) {
        if (ended) return;
        endWithoutSpeech();
        return;
      }
      if (ended) return;
      if (switchedOff()) {
        analysis.cancel();
        dropSwitched();
        return;
      }

      try {
        const modelId = analysis.modelId;
        const store = useChatRuntimeStore.getState();
        store.setVoiceTranscribing(true);
        let transcript = "";
        try {
          transcript = await analysis.text;
          if (ended) return;
        } catch (error) {
          if (error instanceof SttModelNotDownloadedError && !switchedOff()) {
            const missing = modelId ?? useVoiceSettingsStore.getState().sttModel;
            const downloading = isTrackingSttDownload(missing);
            if (!downloading) requestSttDownload(missing);
            // Not a failed transcription: the voice waits for the transcriber (the orb and pill say so)
            // and listens again once it is loaded. This clip cannot be heard.
            const rt = useChatRuntimeStore.getState();
            if (rt.selectedSttModelId === missing) rt.setSttSlotDownload(downloading ? "downloading" : "needed");
            voiceTrace("stt.notDownloaded", { model: missing, downloading });
            if (bargedIn) requestVoiceBargeAbort();
            finish("cancelled");
            return;
          }
          throw error;
        } finally {
          store.setVoiceTranscribing(false);
        }
        // Voice was turned off while this clip was heard: its words are discarded, never held for the next session.
        if (switchedOff()) {
          dropSwitched();
          return;
        }
        if (transcript && isHallucinatedTranscript(transcript)) {
          publishSttDecision(false, "repeated word");
          transcript = "";
        }
        if (hasClosingMark(transcript)) punctuatingStt = sttKey(modelId);
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
        const acknowledged =
          Boolean(transcript) && bargedIn && isAcknowledgement(transcript, getVoiceReplyText());
        // A noise word that continues a held turn ("Translate the phrase" + "thank you") or answers the
        // reply's question ("you or me?" + "you") is speech, not a hallucination.
        const noise =
          isSttNoise(transcript) && !heldTurn && !answersClosingQuestion(transcript, getVoiceReplyText());
        if (transcript && (acknowledged || isBackchannel(transcript) || noise)) {
          voiceTrace("stt.drop", { transcript, why: acknowledged ? "acknowledgement" : "noise" });
          settleFinished(false);
          publishSttDecision(
            false,
            acknowledged ? "acknowledgement" : isBackchannel(transcript) ? "breath" : "cough",
          );
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
        let editing = !heldEdit.autoSend();
        const spokenNow = transcript;
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
        // A turn that sounded finished is sent unless its words read cut off; the text judge is asked
        // only about one that trailed off.
        voiceTrace("turn.transcript", { turnId, chars: transcript.length });
        const judged = Boolean(transcript) && !editing;
        const { turnEnding, turnJudgeBar } = useVoiceSettingsStore.getState();
        const unfinished =
          judged &&
          (soundedFinished
            ? turnEnding === "smartTurnJudge"
              ? await turnUnfinished(transcript, modelId, turnP, turnJudgeBar)
              : readsCutOff(transcript, modelId)
            : await turnUnfinished(transcript, modelId));
        if (judged) voiceTrace("turn.judgeDone", { turnId, unfinished, soundedFinished });
        // The user may claim the held fragment while the judge decides: their edit is never auto-sent,
        // and the box already holds the fragment, so only the new words join it.
        if (!editing && !heldEdit.autoSend()) {
          editing = true;
          transcript = spokenNow;
        }
        if (epoch !== heldEpoch) {
          voiceTrace("stt.drop", { transcript, why: "cleared" });
          // A dropped barge-in cut nothing: the reply it paused plays on, as after any other drop.
          if (bargedIn) requestVoiceBargeAbort();
          finish("stopped");
          // The clear's own re-arm found this session still live and skipped.
          setTimeout(() => requestVoiceResume(), 0);
          return;
        }
        if (
          unfinished &&
          !editing &&
          performance.now() - (heldTurn?.at ?? performance.now()) <
            CONTINUATION_MAX_MS
        ) {
          heldTurn = {
            text: transcript,
            at: heldTurn?.at ?? performance.now(),
            turnId,
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
            // A turn that sounded finished is sent at once: typing it out first is a wait before the reply.
            // Text that was shown live as it was heard is already in the box.
            if (liveShown) mine.rush = true;
            for (let i = stepChars; i < total && !mine.rush && !soundedFinished; i += stepChars) {
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
            if (bargedIn) requestVoiceBargeAbort();
            finish("stopped");
            setTimeout(() => requestVoiceResume(), 0);
            return;
          }
          // Speech then finish land before the deferred submit; requestVoiceSubmit no-ops outside voice mode.
          emitSpeech(transcript, true);
          finish("stopped", transcript);
          if (editing || !heldEdit.autoSend()) {
            voiceTrace("hold.userEditAppend", { transcript });
            setTimeout(() => requestVoiceResume(), 0);
          } else {
            voiceTrace("turn.confirmed", { turnId, early: soundedFinished });
            submitLater(transcript, { turnId: turnId ?? undefined, confirmedEarly: soundedFinished });
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
        if (switchedOff()) {
          dropSwitched();
          return;
        }
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

        const sttPick =
          useChatRuntimeStore.getState().selectedSttModelId ??
          useVoiceSettingsStore.getState().sttModel;
        const sttPicked = () => useChatRuntimeStore.getState().selectedSttModelId === sttPick;
        // A transcriber known to be missing is not loaded again until its download ends: every
        // listen session (one per 8 s of quiet) asked, and got a 409 each time.
        if (useChatRuntimeStore.getState().sttSlotDownload === null) {
          void loadSttModel(sttPick, undefined, undefined, undefined, undefined, undefined, true)
            .then(() => {
              if (sttPicked()) useChatRuntimeStore.getState().setSttSlotLoaded(true);
            })
            .catch((error: unknown) => {
              if (!sttPicked()) return;
              const rt = useChatRuntimeStore.getState();
              rt.setSttSlotLoaded(false);
              if (!(error instanceof SttModelNotDownloadedError)) return;
              const downloading = isTrackingSttDownload(sttPick);
              rt.setSttSlotDownload(downloading ? "downloading" : "needed");
              // Asked as listening begins, not after the first utterance is lost to the missing model.
              if (!downloading) requestSttDownload(sttPick);
            });
        }
        // Off in Settings > Voice > Voice mode: the fixed silence wait alone ends a turn.
        if (useVoiceSettingsStore.getState().turnEnding !== "wait") {
          void loadTurnModel().then((ready) => {
            turnReady = ready;
          });
        }
        if (!vadDisabled) {
          const vadPick = useChatRuntimeStore.getState().selectedVadModelId;
          void loadVadModel(vadPick)
            .then(() => {
              vadReady = true;
              useChatRuntimeStore.getState().setVadSlotLoaded(true);
            })
            .catch((error: unknown) => {
              useChatRuntimeStore.getState().setVadSlotLoaded(false);
              if (!(error instanceof VadUnavailableError)) return;
              const fallback = vadFallback(vadPick);
              // Not saved: an install that gains the detector later (a runtime update) gets the pick back.
              if (fallback) useChatRuntimeStore.setState({ selectedVadModelId: fallback });
              else disableVad(vadPick);
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
          // Speech already underway when the reply starts playing barges in: never cut down to the louder bar.
          if (heardSpeech && !bargedIn && isVoicePlaybackActive()) {
            bargedIn = true;
            requestVoiceBargeIn();
          }
          // While the reply plays (or a barge-in probe is out, keeping frames in order) only a barge-in is listened for.
          if (!bargedIn && (isVoicePlaybackActive() || bargeConfirming)) {
            preroll.push(new Float32Array(input));
            // While the detector is asked, keep every frame: the probe's audio opens the utterance.
            if (!bargeConfirming) {
              while (preroll.length * frameMs > BARGE_LOOKBACK_MS) {
                preroll.shift();
                prerollLoud.shift();
              }
            }
            startedAt = now;
            lastVoiceAt = now;
            const cut = stepBargeGate(bargeGate, rms, frameMs);
            prerollLoud.push(bargeGate.wasLoud);
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
              const onset = bargeOnset(
                prerollLoud,
                Math.ceil(BARGE_GAP_MS / frameMs),
                Math.ceil(BARGE_PREROLL_MS / frameMs),
              );
              voiceTrace("barge.onset", {
                keptMs: Math.round((preroll.length - onset) * frameMs),
              });
              preroll.splice(0, onset);
              prerollLoud.length = 0;
              for (const frame of preroll.splice(0)) {
                chunks.push(frame);
                chunkVoiced.push(true);
                chunkNotSilent.push(true);
              }
              voicedMs = Math.max(bargeGate.loudMs, bargePeakMs);
              bargePeakMs = 0;
              heardSpeech = true;
              startTurn();
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
            // The detector hears only the burst: the older lookback is mostly the reply's own echo.
            const recent = preroll.slice(
              -Math.ceil(BARGE_PREROLL_MS / frameMs),
            );
            const total = recent.reduce((n, c) => n + c.length, 0);
            const probe = new Float32Array(total);
            let at = 0;
            for (const c of recent) {
              probe.set(c, at);
              at += c.length;
            }
            void clipHasSpeech(wavBlob(probe, sampleRate), true).then((voiced) => {
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
          preroll.splice(
            0,
            Math.max(0, preroll.length - Math.ceil(BARGE_PREROLL_MS / frameMs)),
          );
          prerollLoud.length = 0;
          for (const frame of preroll.splice(0)) {
            let sq = 0;
            for (let i = 0; i < frame.length; i++) sq += frame[i] * frame[i];
            const level = Math.sqrt(sq / frame.length);
            const loud = level >= Math.max(SPEECH_RMS_MIN, noiseFloor * SPEECH_OVER_FLOOR);
            chunks.push(frame);
            chunkVoiced.push(loud);
            chunkNotSilent.push(level >= Math.max(SILENCE_RMS_MIN, noiseFloor * SILENCE_OVER_FLOOR));
            // Counted only once live speech follows: echo left after the reply must not become a turn.
            if (loud) prerollVoicedMs += frameMs;
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
          if (speech) {
            voicedMs += frameMs;
            if (!heardSpeech) {
              voicedMs += prerollVoicedMs;
              prerollVoicedMs = 0;
              heardSpeech = true;
              startTurn();
              utteranceInFlight = true;
              noteSpeechStart();
              useChatRuntimeStore.getState().setVoiceHearing(true);
              for (const cb of speechStartCallbacks) cb();
            }
          }
          if (heardSpeech) {
            if (live) feedLive();
            else if (!liveOff && (!liveTaken || speech)) startLive();
          }
          const sinceVoice = now - lastVoiceAt;
          const elapsed = now - startedAt;
          const enoughSpeech =
            heardSpeech && !finalizing && voicedMs >= (bargedIn ? BARGE_CONFIRMED_MS : MIN_SPEECH_MS);
          // The end-of-turn model is not asked about speech over a reply: on a second or two of
          // audio it calls a quarter to a half of unfinished clips finished.
          const askTurn = turnReady && !bargedIn;
          // With the turn asked, the transcript is asked for at the same moment: an early end needs it.
          if (enoughSpeech && sinceVoice >= (askTurn ? TURN_ASK_AFTER_MS : SPECULATE_AFTER_MS)) {
            const span = utteranceSpan(chunkVoiced, chunkNotSilent, padFrames());
            if (span) {
              const key = clipKey(span);
              const analysis = early.start(key, () => analyseClip(span));
              if (bargedIn && bargeNothingKey === key) {
                // An "mm-hm" or an echo: end the clip now, so the reply it paused resumes without the full wait.
                void finalize();
                return;
              }
              if (!bargedIn && questionKey === key && sinceVoice >= TURN_END_AFTER_MS) {
                voiceTrace("turn.endByQuestion", { turnId, sinceVoice: Math.round(sinceVoice) });
                soundedFinished = true;
                void finalize();
                return;
              }
              if (!bargedIn && questionReadKey !== key) {
                questionReadKey = key;
                analysis.text.then(
                  (text) => {
                    if (questionReadKey === key && readsAsQuestion(text))
                      questionKey = key;
                  },
                  () => {},
                );
              }
              if (bargedIn && bargeReadingKey !== key) {
                bargeReadingKey = key;
                analysis.text.then(
                  (text) => {
                    if (nothingToActOn(text)) bargeNothingKey = key;
                  },
                  () => {},
                );
              }
            }
          }
          if (enoughSpeech && askTurn && sinceVoice >= TURN_ASK_AFTER_MS) {
            const span = turnClipSpan(chunkVoiced, chunkNotSilent);
            const key = span ? `${span.start}:${span.end}` : null;
            if (span && key) {
              turnEnd.ask(key, () => wavBlob(turnClipTail(spanSamples(span), sampleRate), sampleRate));
              if (turnEnd.finished(key) && sinceVoice >= TURN_END_AFTER_MS) {
                soundedFinished = true;
                void finalize();
                return;
              }
            }
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
