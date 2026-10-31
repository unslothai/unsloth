// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Background orchestration for system-wide voice typing.
//
// The native layer watches the global shortcut and emits start/stop/cancel
// signals. This controller records and transcribes through the same local
// dictation stack the composer microphone uses (engine, model, language, mic and
// dictionary all come from Voice settings), then hands the finished transcript
// back to the native layer, which types it into whichever app has focus.
//
// This file is also the diagnostics surface: every phase is logged and every
// failure either arrives from the adapter with its own toast or becomes one here.
// A chord that does nothing must never be unexplainable.

import { authFetch } from "@/features/auth";
import {
  type StudioDictationSession,
  StudioModelDictationAdapter,
  isStudioDictationAvailable,
  joinTranscriptParts,
  keepSttModelAlive,
  notifyStudioDictationUnavailable,
  openMicrophone,
  prewarmPcmCapture,
  sttEngineFor,
  stopMediaStream,
  subscribeDictationLevel,
  useChatRuntimeStore,
} from "@/features/chat";
import {
  ensureMicrophonePermission,
  readMicrophonePermission,
} from "@/features/microphone";
import {
  applyDictationDictionary,
  useSettingsDialogStore,
  useVoiceSettingsStore,
} from "@/features/settings";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { fixGrammar } from "./grammar-fixer";
import { containsAbusiveLanguage } from "./abusive-language";
import { invoke } from "@tauri-apps/api/core";
import { type UnlistenFn, listen } from "@tauri-apps/api/event";
import {
  type VoiceTypingModelState,
  checkVoiceTypingModel,
  requestVoiceTypingModelDownload,
  warmVoiceTypingModel,
} from "./voice-typing-preflight";
import { type VoiceTypingCue, playVoiceTypingCue } from "./voice-typing-sounds";
import {
  DEFAULT_LLM_POST_PROCESS_PROMPT,
  useVoiceTypingStore,
} from "./voice-typing-store";

interface VoiceTypingInfo {
  os: string;
  shortcutSupported: boolean;
  holdSupported: boolean;
  pasteSupported: boolean;
}

type SignalAction = "start" | "stop" | "cancel";
type HudPhase =
  | "idle"
  | "preparing"
  | "recording"
  | "transcribing"
  | "fixing"
  | "refining";
type Cue = VoiceTypingCue;

/** Safety net so a forgotten toggle cannot record forever. */
const MAX_RECORDING_MS = 5 * 60 * 1000;
/** Minimum perceptual level (0..1, where level = rms * 3.2) that counts as speech
 *  for auto-stop, combined with an adaptive ambient noise floor. */
const MIN_VOICED_LEVEL = 0.12;
/** Shorter recordings that produced nothing had no speech worth reporting. */
const NO_SPEECH_NOTICE_MIN_MS = 2000;
/** Mic-level frames are throttled on their way to the overlay window. */
const LEVEL_THROTTLE_MS = 45;
/** How often to check or hold the loaded dictation model. */
const MODEL_KEEPALIVE_MS = 60_000;
/** Retry interval when boot-time model check runs before the Python backend is up. */
const BOOT_MODEL_RETRY_MS = 3_000;
const MAX_BOOT_RETRIES = 6;

/**
 * Harper-backed on-device grammar correction. The TypeScript rules stay as a
 * fallback so an older desktop backend (or a dev web build) never blocks a
 * transcript.
 */
/**
 * Dictation often drops the final full stop. Add one only to a complete-ish
 * sentence (four or more words), and use a question mark for a question. Short
 * fragments are left alone: a pause is not a sentence.
 */
function ensureTerminalPeriod(text: string): string {
  const trimmed = text.trimEnd();
  if (!trimmed) return text;
  if (/[.!?…:;)\]"']$/.test(trimmed)) return text;
  const question =
    /^(what|why|how|when|where|who|whom|whose|which|is|are|was|were|do|does|did|can|could|will|would|should|may|might|have|has|had)\b/i.test(
      trimmed,
    );
  if (trimmed.split(/\s+/).length < 4 && !trimmed.includes("=")) return text;
  return `${trimmed}${question ? "?" : "."}`;
}

async function fixGrammarOnDevice(
  text: string,
): Promise<{ text: string; fixes: number }> {  const { dictationLanguage, dictionary } = useVoiceSettingsStore.getState();
  try {
    const result = await invoke<{ text: string; fixes: number }>(
      "fix_grammar",
      {
        text,
        language: dictationLanguage,
        customWords: dictionary,
      },
    );
    if (result?.text?.trim()) {
      return { text: result.text, fixes: result.fixes ?? 0 };
    }
    return { text, fixes: 0 };
  } catch (error) {
    console.warn("[voice typing] native grammar fixer unavailable:", error);
    return {
      text: fixGrammar(text, { language: dictationLanguage }),
      fixes: 0,
    };
  }
}

export class VoiceTypingController {
  private info: VoiceTypingInfo | null = null;
  private phase: HudPhase = "idle";
  private starting = false;
  private session: StudioDictationSession | null = null;
  private transcript = "";
  private ready = false;
  private disposed = false;
  private activating = false;
  private bootRetries = 0;
  private unlistenSignal: UnlistenFn | null = null;
  private unsubscribeLevel: (() => void) | null = null;
  private unsubscribeStore: (() => void) | null = null;
  private unsubscribeVoiceSettings: (() => void) | null = null;
  private timers = new Set<number>();
  private lastLevelSentAt = 0;
  private recordingStartedAt = 0;
  private speechHeard = false;
  private voicedFrameStreak = 0;
  private lastVoiceAt = 0;
  /** Fires after `idleStopMs` without speech, independent of mic level frames. */
  private idleTimer = 0;
  private noiseFloorLevel = 0.025;
  /** Bumped by cancel; an in-flight delivery from an older session is discarded. */
  private cancelToken = 0;
  private llmAbortController: AbortController | null = null;
  private deniedMicNoticeShown = false;
  /** Microphone kept open between sessions so the next press records at once. */
  private warmStream: MediaStream | null = null;
  private warmTimer: number | null = null;
  /** Repeats the backend keepalive so a loaded model is not evicted between presses. */
  private keepAliveTimer: number | null = null;
  /** Bumped when a warm stream is released; an open that resolves afterwards is
   *  stopped instead of leaving the microphone on after a cancel. */
  private warmToken = 0;
  /** The warm stream a running session borrowed, reclaimed when it ends. */
  private sessionStream: MediaStream | null = null;
  private pressStartedAt = 0;
  private micLiveAt = 0;
  private firstFinalAt = 0;
  private stopRequestedAt = 0;

  async init(): Promise<void> {
    this.info = await invoke<VoiceTypingInfo>("voice_typing_info").catch(
      () => null,
    );
    if (this.disposed) return;
    if (!this.info?.shortcutSupported) {
      console.info("[voice typing] not available on this platform; dormant");
      return;
    }
    const unlisten = await listen<{ action: SignalAction }>(
      "voice-typing:signal",
      (event) => {
        void this.onSignal(event.payload.action);
      },
    );
    if (this.disposed) {
      unlisten();
      return;
    }
    this.unlistenSignal = unlisten;
    this.ready = true;
    console.info("[voice typing] controller ready", this.info);
    await this.pushConfig();
    this.unsubscribeStore = useVoiceTypingStore.subscribe((state, previous) => {
      void this.onPreferencesChanged(state, previous);
    });
    // The chosen dictation model and engine live in Voice settings; a change
    // there decides what the next dictation loads.
    this.unsubscribeVoiceSettings = useVoiceSettingsStore.subscribe(
      (state, previous) => {
        // A warm stream belongs to the device it was opened from. Changing the
        // saved microphone must not leave the old one open until its timer ends.
        if (state.micDeviceId !== previous.micDeviceId) {
          this.releaseWarmMicrophone();
        }
        if (state.sttKeepAliveSeconds !== previous.sttKeepAliveSeconds) {
          const model = state.sttModel.trim();
          if (model) {
            void keepSttModelAlive(
              sttEngineFor(model),
              model,
              state.sttKeepAliveSeconds,
            ).catch(() => {});
          }
        }
        if (!useVoiceTypingStore.getState().enabled) return;
        if (
          state.sttModel === previous.sttModel &&
          state.dictationEngine === previous.dictationEngine &&
          state.sttGgufVariant === previous.sttGgufVariant
        ) {
          return;
        }
        void this.activate(false);
      },
    );
    if (useVoiceTypingStore.getState().enabled) {
      // Already on at boot: report the model state and warm the model, but never
      // prompt for the microphone without a user gesture.
      void this.activate(false);
    }
  }

  dispose(): void {
    this.disposed = true;
    this.cancelToken += 1;
    this.unlistenSignal?.();
    this.unlistenSignal = null;
    this.releaseLevel();
    this.unsubscribeStore?.();
    this.unsubscribeStore = null;
    this.unsubscribeVoiceSettings?.();
    this.unsubscribeVoiceSettings = null;
    this.clearTimers();
    this.clearModelKeepAlive();
    this.discardSessionStream();
    this.releaseWarmMicrophone();
    try {
      this.session?.cancel();
    } catch {
      // The session may already have ended.
    }
    this.session = null;
    this.phase = "idle";
  }

  private async onPreferencesChanged(
    state: ReturnType<typeof useVoiceTypingStore.getState>,
    previous: ReturnType<typeof useVoiceTypingStore.getState>,
  ): Promise<void> {
    const configChanged =
      state.enabled !== previous.enabled ||
      state.shortcut !== previous.shortcut ||
      state.mode !== previous.mode ||
      state.muteWhileRecording !== previous.muteWhileRecording ||
      state.hudPosition !== previous.hudPosition;
    if (configChanged) await this.pushConfig();
    if (!state.enabled && previous.enabled) {
      // Turning voice typing off should also close a microphone that was kept
      // open for the next press, and stop a dictation that is still running.
      this.releaseWarmMicrophone();
      this.clearModelKeepAlive();
      if (this.phase !== "idle" || this.starting) this.cancelSession();
    }
    if (state.enabled && !previous.enabled) {
      // An explicit switch-on is the only place the mic prompt may appear.
      void this.activate(true);
      return;
    }
    if (state.enabled && state.preloadModel && !previous.preloadModel) {
      void this.activate(false);
    }
    if (state.micWarmSeconds !== previous.micWarmSeconds) {
      if (state.micWarmSeconds === 0) this.releaseWarmMicrophone();
      else if (this.warmStream) this.armWarmTimer();
    }
  }

  private async pushConfig(): Promise<void> {
    if (!this.ready || this.disposed) return;
    const { enabled, shortcut, mode, muteWhileRecording, hudPosition } =
      useVoiceTypingStore.getState();
    console.info("[voice typing] pushing config", {
      enabled,
      shortcut,
      mode,
      muteWhileRecording,
      hudPosition,
    });
    try {
      await invoke("voice_typing_configure", {
        enabled,
        shortcut,
        mode,
        muteWhileRecording,
        hudPosition,
      });
    } catch (error) {
      const message =
        typeof error === "string"
          ? error
          : translate("settings.voice.voiceTyping.configureFailed");
      console.warn("[voice typing] config rejected:", error);
      toast.error(message);
    }
  }

  /** Bring the dictation stack up: report model readiness, then optionally load.
   *  `explicit` marks a user gesture (switching voice typing on), the only case
   *  that may prompt for the microphone and surface failures loudly. */
  private async activate(explicit: boolean): Promise<void> {
    if (this.disposed || this.activating) return;
    this.activating = true;
    try {
      // Opening the shared audio context is part of getting ready: doing it here
      // keeps the first press from paying for it before a sample arrives.
      prewarmPcmCapture();
      // Building the overlay is the other first-press cost; a hidden window now
      // means the pill appears as soon as the shortcut is pressed.
      void invoke("voice_typing_hud_prewarm").catch(() => {
        // The overlay is still created on first use.
      });
      if (explicit) {
        // A user gesture: the only moment the consent dialog may be raised.
        if (!(await ensureMicrophonePermission("voice-typing"))) {
          this.notifyMicBlocked();
          return;
        }
      }
      const { readiness, model } = await this.refreshModelStatus();
      if (this.disposed) return;
      if (readiness === "unknown") {
        // Backend may still be booting when Tauri first mounts the provider;
        // retry quietly until the server responds.
        if (this.bootRetries < MAX_BOOT_RETRIES) {
          this.bootRetries += 1;
          this.setTimer(BOOT_MODEL_RETRY_MS, () => {
            if (!this.disposed && useVoiceTypingStore.getState().enabled) {
              void this.activate(false);
            }
          });
        }
        return;
      }
      this.bootRetries = 0;
      // A saved deny cannot resolve on its own, so it is worth saying out loud at
      // boot; an unknown answer waits for the first press instead.
      const modelUsable =
        readiness === "ready" ||
        readiness === "will-load" ||
        readiness === "loading";
      if (!explicit && modelUsable && !this.deniedMicNoticeShown) {
        if ((await readMicrophonePermission()) === "denied") {
          this.deniedMicNoticeShown = true;
          this.notifyMicBlocked();
        }
      }
      if (explicit && readiness === "needs-download") {
        // Ask through the app's own download flow instead of failing later.
        requestVoiceTypingModelDownload();
        return;
      }
      if (explicit && readiness === "engine-missing") {
        toast.error(
          translate("settings.voice.voiceTyping.modelEngineMissing"),
          {
            description: translate(
              "settings.voice.voiceTyping.engineMissingHint",
            ),
          },
        );
        return;
      }
      if (!useVoiceTypingStore.getState().preloadModel) return;
      if (readiness !== "will-load") return;
      try {
        await warmVoiceTypingModel();
        if (!this.disposed) {
          console.info("[voice typing] model warmed:", model);
          useVoiceTypingStore.getState().setReadiness("ready");
        }
      } catch (error) {
        // Correct the displayed state and, for a user gesture, explain.
        await this.refreshModelStatus();
        if (!explicit) return;
        const message =
          error instanceof Error && error.message
            ? error.message
            : translate("settings.voice.voiceTyping.startFailed");
        console.warn("[voice typing] model warm-up failed:", error);
        toast.error(message, {
          action: {
            label: translate("settings.voice.voiceTyping.openSettingsAction"),
            onClick: openVoiceSettings,
          },
        });
      }
    } finally {
      this.activating = false;
      // Whatever the outcome, a resident model is worth holding on to from here.
      // The heartbeat itself checks readiness before it sends anything.
      this.armModelKeepAlive();
    }
  }

  private async refreshModelStatus(): Promise<VoiceTypingModelState> {
    try {
      const state = await checkVoiceTypingModel();
      if (!this.disposed) {
        useVoiceTypingStore.getState().setReadiness(state.readiness);
      }
      console.info(
        "[voice typing] model status:",
        state.readiness,
        state.model,
      );
      return state;
    } catch (error) {
      console.warn("[voice typing] model status check failed:", error);
      if (!this.disposed)
        useVoiceTypingStore.getState().setReadiness("unknown");
      return { readiness: "unknown", model: "" };
    }
  }

  private async onSignal(action: SignalAction): Promise<void> {
    console.info("[voice typing] signal:", action, "phase:", this.phase);
    switch (action) {
      case "start":
        if (this.phase === "idle" && !this.starting) {
          await this.begin();
        }
        return;
      case "stop":
        if (this.phase === "recording") this.finish();
        else if (
          this.starting ||
          this.phase === "preparing" ||
          this.phase === "transcribing"
        ) {
          this.cancelSession();
        }
        return;
      case "cancel":
        if (this.phase !== "idle" || this.starting) this.cancelSession();
        return;
    }
  }

  private notifyMicBlocked(): void {
    toast.error(translate("settings.voice.dictation.micAccessBlockedDesktop"), {
      action: {
        label: translate("settings.voice.voiceTyping.openSettingsAction"),
        onClick: openVoiceSettings,
      },
    });
  }

  private async begin(): Promise<void> {
    if (this.phase !== "idle") return;
    if (this.session || this.starting) return;
    if (!isStudioDictationAvailable()) {
      // The canonical explanation the composer also shows.
      notifyStudioDictationUnavailable();
      return;
    }
    this.starting = true;
    const pressToken = this.cancelToken;
    // The consent dialog belongs to the settings gesture that switched voice
    // typing on, where the app is already focused. Opening it from a global
    // shortcut would steal focus from the app being dictated into, so a missing
    // grant is reported with a way back to Voice settings instead.
    const permission = await readMicrophonePermission();
    if (this.disposed || pressToken !== this.cancelToken) {
      this.starting = false;
      return;
    }
    if (permission !== "granted") {
      this.starting = false;
      this.notifyMicBlocked();
      this.setPhase("idle");
      return;
    }
    this.pressStartedAt = performance.now();
    this.micLiveAt = 0;
    this.firstFinalAt = 0;
    this.stopRequestedAt = 0;
    this.setPhase("preparing");
    this.starting = false;
    this.transcript = "";
    this.speechHeard = false;
    this.voicedFrameStreak = 0;
    this.lastVoiceAt = 0;
    this.noiseFloorLevel = 0.025;
    this.recordingStartedAt = 0;
    const { spokenPunctuation } = useVoiceTypingStore.getState();
    const warm = this.takeWarmStream();
    this.sessionStream = warm;
    let session: StudioDictationSession;
    try {
      session = new StudioModelDictationAdapter({
        chatId: null,
        microphone: warm,
        removeFillerWords: true,
        spokenPunctuation,
        // Voice typing is short-form: a 1 s first segment reaches the model
        // sooner than the composer's 2.8 s window.
        minSegmentMs: 1_000,
        silenceCutMs: 320,
        // Use a single JSON response per cut segment so system/UI resources are not
        // spent streaming partial tokens and HTTP disconnect aborts the backend immediately.
        stream: false,
      }).listen();
    } catch (error) {
      console.error("[voice typing] could not start:", error);
      this.retireSessionStream();
      toast.error(
        error instanceof Error && error.message
          ? error.message
          : translate("settings.voice.voiceTyping.startFailed"),
      );
      this.setPhase("idle");
      return;
    }
    if (this.disposed || pressToken !== this.cancelToken) {
      session.cancel();
      this.discardSessionStream();
      return;
    }
    this.session = session;
    session.onSpeechStart(() => {
      if (this.session !== session || this.phase !== "preparing") return;
      this.recordingStartedAt = Date.now();
      this.micLiveAt = performance.now();
      this.setPhase("recording");
      // After the first samples, so the cue means "the mic is live".
      this.playCue("start");
    });
    // The adapter applies the dictionary and records the session for history.
    session.onSpeech((result) => {
      const text = result.transcript ?? "";
      if (result.isFinal === false) {
        if (text && (this.phase === "recording" || this.phase === "transcribing")) {
          const preview = joinTranscriptParts([this.transcript, text]).trim();
          if (preview) this.setPhase(this.phase, preview);
        }
        return;
      }
      if (text) {
        // Keep a segment that is only a spoken "new paragraph": it is a break,
        // not empty text.
        const next = joinTranscriptParts([this.transcript, text]);
        if (next !== this.transcript) {
          this.transcript = next;
          if (!this.firstFinalAt) this.firstFinalAt = performance.now();
          const preview = this.transcript.trim();
          if (
            preview &&
            (this.phase === "recording" || this.phase === "transcribing")
          ) {
            this.setPhase(this.phase, preview);
          }
        }
      }
    });
    session.onEnd?.(() => {
      if (this.session !== session) return;
      this.session = null;
      this.releaseLevel();
      const status = session.status;
      const reason = status.type === "ended" ? status.reason : "unknown";
      // A normal stop keeps the borrowed microphone warm for the next press; an
      // error or a cancel gives it back.
      if (reason === "stopped") this.retireSessionStream();
      else this.discardSessionStream();
      if (this.phase === "preparing" || this.phase === "recording") {
        // The adapter ended the session on its own (missing model, microphone
        // failure, a download prompt) and reported the reason itself.
        console.info("[voice typing] session ended by the adapter:", {
          reason,
        });
        this.clearTimers();
        this.setPhase("idle");
      }
    });
    this.watchLevels(session);
    this.setTimer(MAX_RECORDING_MS, () => {
      if (this.phase === "recording") this.finish();
    });
  }

  private watchLevels(session: StudioDictationSession): void {
    this.releaseLevel();
    this.lastLevelSentAt = 0;
    this.unsubscribeLevel = subscribeDictationLevel((level) => {
      if (this.session !== session || this.phase !== "recording") return;
      const now = Date.now();
      if (now - this.lastLevelSentAt >= LEVEL_THROTTLE_MS) {
        this.lastLevelSentAt = now;
        void this.invokeQuietly("voice_typing_hud_level", { level });
      }
      const voiceThreshold = Math.max(
        MIN_VOICED_LEVEL,
        this.noiseFloorLevel * 2.5,
      );
      if (level >= voiceThreshold) {
        this.voicedFrameStreak += 1;
        if (this.voicedFrameStreak >= 2 || level >= voiceThreshold * 1.4) {
          this.speechHeard = true;
          this.lastVoiceAt = now;
          this.armIdleTimer();
        }
        // Slowly adapt noise floor upward if steady ambient hum is below 0.16
        // so constant background noise cannot lock `speechHeard` active forever.
        if (level < 0.16 && level < this.noiseFloorLevel * 4.2) {
          this.noiseFloorLevel = Math.min(
            0.18,
            Math.max(0.015, this.noiseFloorLevel * 0.98 + level * 0.02),
          );
        }
        return;
      }
      this.voicedFrameStreak = 0;
      this.noiseFloorLevel = Math.min(
        0.18,
        Math.max(0.015, this.noiseFloorLevel * 0.95 + level * 0.05),
      );
      const { idleStopMs } = useVoiceTypingStore.getState();
      // Only one auto-stop exists: the idle timeout. There is no separate
      // short-pause timer, so the same setting governs hold and toggle modes.
      if (
        idleStopMs > 0 &&
        this.speechHeard &&
        this.lastVoiceAt > 0 &&
        now - this.lastVoiceAt >= idleStopMs
      ) {
        console.info("[voice typing] auto-stop after idle silence");
        this.finish();
        return;
      }
      if (!this.speechHeard) {
        // User activated voice typing and spoke nothing (only silence or ambient noise):
        // close the pill after the same idle window without transcribing.
        const silentWaitMs = idleStopMs > 0 ? idleStopMs : 2500;
        if (
          this.recordingStartedAt > 0 &&
          now - this.recordingStartedAt >= silentWaitMs
        ) {
          console.info("[voice typing] auto-cancel after initial silence");
          this.cancelSession();
        }
        return;
      }
    });
  }

  private finish(): void {
    const session = this.session;
    if (!session || this.phase !== "recording") return;
    if (!this.speechHeard && !this.transcript.trim()) {
      // Nothing above the noise floor was spoken; close immediately without
      // entering the transcribing phase or calling the backend.
      this.cancelSession();
      return;
    }
    this.clearTimers();
    this.releaseLevel();
    this.stopRequestedAt = performance.now();
    const token = this.cancelToken;
    const recordedMs = this.recordingStartedAt
      ? Date.now() - this.recordingStartedAt
      : 0;
    this.setPhase("transcribing", this.transcript.trim() || undefined);
    void (async () => {
      try {
        // Resolves once the final segment is transcribed and onSpeech has fired.
        await session.stop();
      } catch (error) {
        console.error("[voice typing] could not finish the recording:", error);
      }
      await this.deliver(session, token, recordedMs);
    })();
  }

  private async postProcessTranscript(
    rawText: string,
    token: number,
  ): Promise<string> {
    const {
      llmPostProcessEnabled,
      llmThinkingEnabled,
      llmPostProcessModel,
      llmPostProcessPrompt,
    } = useVoiceTypingStore.getState();
    if (!llmPostProcessEnabled || !rawText.trim()) return rawText;
    const runtime = useChatRuntimeStore.getState();
    const model =
      llmPostProcessModel.trim() ||
      runtime.residentCheckpoint?.trim() ||
      runtime.params.checkpoint.trim() ||
      runtime.loadedModels[0]?.id?.trim() ||
      "";
    if (!model) return rawText;

    const template =
      llmPostProcessPrompt.trim() || DEFAULT_LLM_POST_PROCESS_PROMPT;
    const userContent = template.includes("${output}")
      ? template.replaceAll("${output}", rawText)
      : `${template}\n\n<transcript>\n${rawText}\n</transcript>`;

    const controller = new AbortController();
    this.llmAbortController = controller;
    try {
      const response = await authFetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: userContent }],
          stream: false,
          temperature: 0.1,
          max_tokens: Math.max(256, Math.min(2048, rawText.length * 2)),
          enable_thinking: llmThinkingEnabled,
          ...(llmThinkingEnabled
            ? {}
            : { thinking: { type: "disabled" }, reasoning_effort: "low" }),
          enable_tools: false,
        }),
        signal: controller.signal,
      });
      if (!response.ok || this.disposed || token !== this.cancelToken) {
        return rawText;
      }
      const data = (await response.json().catch(() => null)) as {
        choices?: Array<{ message?: { content?: string | null } }>;
      } | null;
      const content = data?.choices?.[0]?.message?.content;
      if (typeof content !== "string") return rawText;
      const cleaned = content
        .replace(/<think>[\s\S]*?<\/think>/gi, "")
        .trim();
      return cleaned || rawText;
    } catch (error) {
      if (!controller.signal.aborted) {
        console.warn("[voice typing] LLM post-processing skipped:", error);
      }
      return rawText;
    } finally {
      if (this.llmAbortController === controller) {
        this.llmAbortController = null;
      }
    }
  }

  private async deliver(
    session: StudioDictationSession,
    token: number,
    recordedMs: number,
  ): Promise<void> {
    if (this.session === session) this.session = null;
    const rawText = this.transcript.trim();
    this.transcript = "";
    if (this.disposed || token !== this.cancelToken) return;
    if (!rawText) {
      // Genuine silence (or an aborted session): close the pill immediately.
      this.setPhase("idle");
      if (recordedMs >= NO_SPEECH_NOTICE_MIN_MS && this.speechHeard) {
        toast.info(translate("settings.voice.voiceTyping.noSpeechNotice"));
      }
      return;
    }
    // Voice typing only: an abusive note is discarded before any refinement,
    // insertion, clipboard copy or history write.
    if (containsAbusiveLanguage(rawText)) {
      this.rejectAbusive(rawText);
      return;
    }
    const storeState = useVoiceTypingStore.getState();
    const { grammarFixerEnabled, llmPostProcessEnabled } = storeState;
    if (llmPostProcessEnabled) {
      this.setPhase("refining");
    }
    let text = await this.postProcessTranscript(rawText, token);
    if (this.disposed || token !== this.cancelToken) return;
    let fixes = 0;
    if (grammarFixerEnabled) {
      this.setPhase("fixing");
      const corrected = await fixGrammarOnDevice(text);
      text = corrected.text;
      fixes = corrected.fixes;
    }
    // Grammar rules can re-case a custom word; make the user's spelling win.
    text = applyDictationDictionary(text, undefined, { snippets: false });
    const withTerminal = ensureTerminalPeriod(text);
    if (withTerminal !== text) fixes += 1;
    text = withTerminal;
    if (containsAbusiveLanguage(text)) {
      this.rejectAbusive(rawText);
      return;
    }
    // The dictation adapter records the pre-refinement transcript for the
    // history; replace it with what was actually inserted.
    useVoiceSettingsStore
      .getState()
      .updateLatestRecentDictation(rawText, text, fixes);
    if (this.disposed || token !== this.cancelToken) return;
    if (!text.trim()) {
      this.setPhase("idle");
      return;
    }
    // Hide the floating HUD pill immediately before inserting text so there is
    // zero post-transcription linger delay.
    this.setPhase("idle");
    useVoiceTypingStore.getState().setLastTranscript(text);
    this.playCue("stop");
    const {
      appendTrailingSpace,
      copyLastTranscript,
      insertSubmitKey,
      pasteAutomatically,
      pasteMethod,
    } = useVoiceTypingStore.getState();
    try {
      const outcome = await invoke<string>("voice_typing_insert", {
        text,
        // A manual paste needs the transcript to stay put. The explicit copy
        // preference keeps it after an automatic paste too, without a second copy.
        copyLastTranscript: copyLastTranscript || !pasteAutomatically,
        paste: pasteAutomatically,
        pasteMethod,
        // The key is only sent after an actual paste.
        submitKey: pasteAutomatically ? insertSubmitKey : "none",
        trailingSpace: appendTrailingSpace,
      });
      // A cancel during the insert cannot undo the paste, but it must not
      // overwrite the phase of a session the user already started next.
      if (this.disposed || token !== this.cancelToken) return;
      if (outcome === "copied") {
        toast.info(translate("settings.voice.voiceTyping.copiedNotice"));
      }
      console.info("[voice typing] transcript delivered:", outcome, {
        characters: text.length,
        micMs: this.micLiveAt
          ? Math.round(this.micLiveAt - this.pressStartedAt)
          : null,
        firstTextMs: this.firstFinalAt
          ? Math.round(this.firstFinalAt - this.pressStartedAt)
          : null,
        insertMs: this.stopRequestedAt
          ? Math.round(performance.now() - this.stopRequestedAt)
          : null,
      });
      // A completed session proves the model is resident.
      useVoiceTypingStore.getState().setReadiness("ready");
      this.warmMicrophone();
    } catch (error) {
      if (this.disposed || token !== this.cancelToken) return;
      console.error("[voice typing] could not insert the transcript:", error);
      toast.error(translate("settings.voice.voiceTyping.insertFailed"));
      this.setPhase("idle");
    }
  }

  private rejectAbusive(rawText: string): void {
    // The adapter records the note before delivery; remove that record (and
    // its audio) so nothing survives in history or local storage.
    useVoiceSettingsStore.getState().removeLatestRecentDictation(rawText);
    this.transcript = "";
    this.setPhase("idle");
    toast.error(
      translate("settings.voice.voiceTyping.abusiveLanguage"),
      { duration: 3000 },
    );
  }

  private cancelSession(): void {
    // Invalidates any in-flight delivery from a session stopped earlier.
    this.cancelToken += 1;
    this.starting = false;
    this.llmAbortController?.abort();
    this.llmAbortController = null;
    const wasActive =
      this.phase === "recording" || this.phase === "transcribing";
    const session = this.session;
    this.session = null;
    this.transcript = "";
    this.clearTimers();
    this.releaseLevel();
    try {
      session?.cancel();
    } catch {
      // Already gone.
    }
    // A cancelled dictation gives the microphone back instead of keeping it warm,
    // including one a just-finished session had already returned to the pool.
    this.discardSessionStream();
    this.releaseWarmMicrophone();
    if (wasActive) this.playCue("cancel");
    this.setPhase("idle");
  }

  private setPhase(phase: HudPhase, sublabel?: string): void {
    this.phase = phase;
    const { hudPosition } = useVoiceTypingStore.getState();
    const labelKey =
      phase === "preparing"
        ? "settings.voice.voiceTyping.hudPreparing"
        : phase === "recording"
          ? "settings.voice.voiceTyping.hudListening"
          : phase === "transcribing"
            ? "settings.voice.voiceTyping.hudTranscribing"
            : null;
    const label = labelKey ? translate(labelKey) : undefined;
    void this.invokeQuietly("voice_typing_hud_state", {
      state: phase,
      label,
      sublabel: sublabel ?? null,
      hudPosition,
    });
  }

  /** A stream opened for a different saved device would record from the wrong
   *  microphone; leave it to the timer unless the browser hides the device id. */
  private streamMatchesSavedDevice(stream: MediaStream): boolean {
    const saved = useVoiceSettingsStore.getState().micDeviceId.trim();
    if (!saved || saved === "default") return true;
    const track = stream.getAudioTracks()[0];
    const actual = track?.getSettings().deviceId?.trim();
    return !actual || actual === saved;
  }

  /** Claim the warm microphone for a session, if it is still alive. */
  private takeWarmStream(): MediaStream | null {
    this.clearWarmTimer();
    const stream = this.warmStream;
    this.warmStream = null;
    if (!stream) return null;
    const live = stream
      .getAudioTracks()
      .some((track) => track.readyState === "live");
    if (!live || !this.streamMatchesSavedDevice(stream)) {
      stopMediaStream(stream);
      return null;
    }
    return stream;
  }

  /** Give a normally finished session's borrowed microphone back to the warm
   *  pool, or stop it when warming is off. */
  private retireSessionStream(): void {
    const stream = this.sessionStream;
    this.sessionStream = null;
    if (!stream) return;
    if (
      useVoiceTypingStore.getState().micWarmSeconds === 0 ||
      this.disposed ||
      !this.streamMatchesSavedDevice(stream)
    ) {
      stopMediaStream(stream);
      return;
    }
    if (this.warmStream) {
      // A warm spare was opened while the session ran; the session's copy is
      // redundant.
      stopMediaStream(stream);
      return;
    }
    this.warmStream = stream;
    this.armWarmTimer();
  }

  /** Stop the stream a session borrowed, without keeping it warm. */
  private discardSessionStream(): void {
    const stream = this.sessionStream;
    this.sessionStream = null;
    if (stream) stopMediaStream(stream);
  }

  /** Open the microphone after a dictation so the next press records at once. */
  private warmMicrophone(): void {
    const { enabled, micWarmSeconds: seconds } = useVoiceTypingStore.getState();
    if (!enabled || this.disposed || seconds === 0) return;
    if (this.warmStream) {
      // A session gives its borrowed stream back before the insert completes.
      // Restart the hold window here so the time spent inserting is not taken
      // out of the time the microphone stays ready for the next press.
      this.armWarmTimer();
      return;
    }
    const token = this.warmToken;
    const { micDeviceId } = useVoiceSettingsStore.getState();
    void openMicrophone(micDeviceId)
      .then((stream) => {
        if (
          this.disposed ||
          token !== this.warmToken ||
          this.session ||
          !useVoiceTypingStore.getState().enabled ||
          useVoiceTypingStore.getState().micWarmSeconds === 0 ||
          this.warmStream ||
          !this.streamMatchesSavedDevice(stream)
        ) {
          stopMediaStream(stream);
          return;
        }
        this.warmStream = stream;
        this.armWarmTimer();
      })
      .catch((error) => {
        // Warmth is an optimization; a failure here must not surface.
        console.info("[voice typing] microphone not kept warm:", error);
      });
  }

  private armWarmTimer(): void {
    this.clearWarmTimer();
    const seconds = useVoiceTypingStore.getState().micWarmSeconds;
    if (!this.warmStream || seconds <= 0) return;
    this.warmTimer = window.setTimeout(() => {
      this.warmTimer = null;
      this.releaseWarmMicrophone();
    }, seconds * 1000);
  }

  private clearWarmTimer(): void {
    if (this.warmTimer !== null) {
      window.clearTimeout(this.warmTimer);
      this.warmTimer = null;
    }
  }

  /** Hold the loaded dictation model past the sidecar's idle unload.
   *
   *  A press after a pause otherwise pays for a load before it can record, which
   *  is most of the gap the user sees. The backend only rearms its timer: it never
   *  loads on a heartbeat, so an idle app cannot pin VRAM it is not using, and a
   *  model that was evicted anyway is simply loaded by the next press. */
  private armModelKeepAlive(): void {
    const { sttModel, sttKeepAliveSeconds } = useVoiceSettingsStore.getState();
    const model = sttModel.trim();
    if (model) {
      void keepSttModelAlive(
        sttEngineFor(model),
        model,
        sttKeepAliveSeconds,
      ).catch(() => {});
    }
    if (this.keepAliveTimer !== null || this.disposed) return;
    if (!useVoiceTypingStore.getState().enabled) return;
    this.keepAliveTimer = window.setInterval(() => {
      this.heartbeatModel();
    }, MODEL_KEEPALIVE_MS);
  }

  private clearModelKeepAlive(): void {
    if (this.keepAliveTimer !== null) {
      window.clearInterval(this.keepAliveTimer);
      this.keepAliveTimer = null;
    }
  }

  private heartbeatModel(): void {
    if (this.disposed) {
      this.clearModelKeepAlive();
      return;
    }
    const { readiness, enabled } = useVoiceTypingStore.getState();
    if (!enabled) {
      this.clearModelKeepAlive();
      return;
    }
    if (readiness === "unknown") {
      void this.refreshModelStatus();
      return;
    }
    // Nothing resident to hold while the model is missing or still downloading;
    // the press that finds it ready starts the timer again.
    if (readiness === "needs-download" || readiness === "engine-missing") {
      return;
    }
    const { sttModel, sttKeepAliveSeconds } = useVoiceSettingsStore.getState();
    const model = sttModel.trim();
    if (!model) return;
    if (sttKeepAliveSeconds === 0) {
      void keepSttModelAlive(sttEngineFor(model), model, 0)
        .then((held) => {
          if (!held && !this.disposed && readiness === "ready") {
            void this.refreshModelStatus();
          }
        })
        .catch(() => {});
    } else if (readiness === "ready") {
      void this.refreshModelStatus();
    }
  }

  private releaseWarmMicrophone(): void {
    // Any openMicrophone() still pending belongs to the release that just
    // happened; its result must be stopped rather than kept warm.
    this.warmToken += 1;
    this.clearWarmTimer();
    const stream = this.warmStream;
    this.warmStream = null;
    if (stream) stopMediaStream(stream);
  }

  private releaseLevel(): void {
    this.unsubscribeLevel?.();
    this.unsubscribeLevel = null;
  }

  private async invokeQuietly(
    command: string,
    args: Record<string, unknown>,
  ): Promise<void> {
    if (this.disposed) return;
    await invoke(command, args).catch(() => {});
  }

  private setTimer(delay: number, run: () => void): void {
    const id = window.setTimeout(() => {
      this.timers.delete(id);
      run();
    }, delay);
    this.timers.add(id);
  }

  private clearTimers(): void {
    for (const id of this.timers) window.clearTimeout(id);
    this.timers.clear();
    if (this.idleTimer) {
      window.clearTimeout(this.idleTimer);
      this.idleTimer = 0;
    }
  }

  /** Re-arm the forgotten-session safety stop after each voiced frame. */
  private armIdleTimer(): void {
    if (this.idleTimer) window.clearTimeout(this.idleTimer);
    const { idleStopMs } = useVoiceTypingStore.getState();
    if (idleStopMs <= 0) {
      this.idleTimer = 0;
      return;
    }
    this.idleTimer = window.setTimeout(() => {
      this.idleTimer = 0;
      if (this.phase === "recording" && this.speechHeard) {
        console.info("[voice typing] auto-stop after idle silence");
        this.finish();
      }
    }, idleStopMs);
  }

  /** Short feedback tones; a nicety that must never break dictation. */
  private playCue(kind: Cue): void {
    const { soundEnabled, soundTheme, soundVolume } =
      useVoiceTypingStore.getState();
    if (!soundEnabled) return;
    playVoiceTypingCue(kind, soundTheme, soundVolume);
  }
}

function openVoiceSettings(): void {
  useSettingsDialogStore.getState().openDialog("voice");
}
