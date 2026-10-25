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

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  isStudioDictationAvailable,
  joinTranscriptParts,
  notifyStudioDictationUnavailable,
  openMicrophone,
  stopMediaStream,
  StudioModelDictationAdapter,
  subscribeDictationLevel,
  type StudioDictationSession,
} from "@/features/chat";
import {
  useSettingsDialogStore,
  useVoiceSettingsStore,
} from "@/features/settings";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  checkVoiceTypingModel,
  queryMicrophonePermission,
  requestVoiceTypingMicrophoneAccess,
  requestVoiceTypingModelDownload,
  warmVoiceTypingModel,
  type MicrophonePermission,
  type VoiceTypingModelState,
} from "./voice-typing-preflight";
import { useVoiceTypingStore } from "./voice-typing-store";

interface VoiceTypingInfo {
  os: string;
  shortcutSupported: boolean;
  holdSupported: boolean;
  pasteSupported: boolean;
}

type SignalAction = "start" | "stop" | "cancel" | "disabled";
type HudPhase = "idle" | "preparing" | "recording" | "transcribing" | "done";
type Cue = "start" | "stop" | "cancel";

/** Safety net so a forgotten toggle cannot record forever. */
const MAX_RECORDING_MS = 5 * 60 * 1000;
/** Raw RMS above this counts as speech for the auto-stop. */
const VOICED_LEVEL = 0.05;
/** Shorter recordings that produced nothing had no speech worth reporting. */
const NO_SPEECH_NOTICE_MIN_MS = 2000;
/** How long the "inserted" state stays on the overlay before it hides. */
const DONE_VISIBLE_MS = 900;
/** Mic-level frames are throttled on their way to the overlay window. */
const LEVEL_THROTTLE_MS = 55;
/** Live transcript updates are throttled more than the level: text is heavier. */
const HUD_TEXT_THROTTLE_MS = 120;
/** The native side throttles too; this keeps double notices out of the queue. */
const DISABLED_NOTICE_COOLDOWN_MS = 15_000;
/** "Waiting for the microphone prompt" hints go out at most this often. */
const MIC_NOTICE_COOLDOWN_MS = 30_000;
/** Recheck the microphone permission this long into a stuck "preparing". */
const MIC_PERMISSION_WATCHDOG_MS = 2_500;

export class VoiceTypingController {
  private info: VoiceTypingInfo | null = null;
  private phase: HudPhase = "idle";
  private session: StudioDictationSession | null = null;
  private transcript = "";
  private ready = false;
  private disposed = false;
  private activating = false;
  private unlistenSignal: UnlistenFn | null = null;
  private unsubscribeLevel: (() => void) | null = null;
  private unsubscribeStore: (() => void) | null = null;
  private unsubscribeVoiceSettings: (() => void) | null = null;
  private timers = new Set<number>();
  private lastLevelSentAt = 0;
  private audioContext: AudioContext | null = null;
  private recordingStartedAt = 0;
  private speechHeard = false;
  private lastVoiceAt = 0;
  /** Bumped by cancel; an in-flight delivery from an older session is discarded. */
  private cancelToken = 0;
  private lastDisabledNoticeAt = 0;
  private lastMicNoticeAt = 0;
  private deniedMicNoticeShown = false;
  /** Last permission answer; a grant is reused to skip a query on the hot path. */
  private micPermission: MicrophonePermission | null = null;
  /** Microphone kept open between sessions so the next press records at once. */
  private warmStream: MediaStream | null = null;
  private warmTimer: number | null = null;
  /** The warm stream a running session borrowed, reclaimed when it ends. */
  private sessionStream: MediaStream | null = null;
  private hudText = "";
  private hudTextSentAt = 0;
  private partialText = "";
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
      state.hudEnabled !== previous.hudEnabled;
    if (configChanged) await this.pushConfig();
    if (!state.enabled && previous.enabled) {
      // Turning voice typing off should also close a microphone that was kept
      // open for the next press, and stop a dictation that is still running.
      this.releaseWarmMicrophone();
      if (this.phase !== "idle") this.cancelSession();
    }
    if (state.enabled && !previous.enabled) {
      // An explicit switch-on is the only place the mic prompt may appear.
      void this.activate(true);
      return;
    }
    if (state.enabled && state.preloadModel && !previous.preloadModel) {
      void this.activate(false);
    }
    if (state.hudShowTranscript !== previous.hudShowTranscript) {
      if (!state.hudShowTranscript) this.clearHudText();
      else this.sendHudText(this.composeHudText(), true);
    }
    if (state.micWarmSeconds !== previous.micWarmSeconds) {
      if (state.micWarmSeconds <= 0) this.releaseWarmMicrophone();
      else if (this.warmStream) this.armWarmTimer();
    }
  }

  private async pushConfig(): Promise<void> {
    if (!this.ready || this.disposed) return;
    const { enabled, shortcut, mode, hudEnabled } =
      useVoiceTypingStore.getState();
    console.info("[voice typing] pushing config", {
      enabled,
      shortcut,
      mode,
      hud: hudEnabled,
    });
    try {
      await invoke("voice_typing_configure", {
        enabled,
        shortcut,
        mode,
        hud: hudEnabled,
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
      if (explicit) {
        // A user gesture: the only moment the microphone prompt may be raised.
        if (await requestVoiceTypingMicrophoneAccess()) {
          this.micPermission = "granted";
        }
      }
      const { readiness, model } = await this.refreshModelStatus();
      if (this.disposed) return;
      // A saved deny cannot resolve on its own, so it is worth saying out loud at
      // boot; a fresh "ask" resolves on first use and stays quiet here.
      const modelUsable =
        readiness === "ready" ||
        readiness === "will-load" ||
        readiness === "loading";
      if (!explicit && modelUsable && !this.deniedMicNoticeShown) {
        if ((await queryMicrophonePermission()) === "denied") {
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
      if (!this.disposed) useVoiceTypingStore.getState().setReadiness("unknown");
      return { readiness: "unknown", model: "" };
    }
  }

  private async onSignal(action: SignalAction): Promise<void> {
    console.info("[voice typing] signal:", action, "phase:", this.phase);
    switch (action) {
      case "disabled":
        this.notifyDisabled();
        return;
      case "start":
        // "done" lingers briefly after an insert; a new press may start already.
        if (this.phase === "idle" || this.phase === "done") await this.begin();
        return;
      case "stop":
        if (this.phase === "recording") this.finish();
        else if (this.phase === "preparing" || this.phase === "transcribing") {
          this.cancelSession();
        }
        return;
      case "cancel":
        if (this.phase !== "idle") this.cancelSession();
        return;
    }
  }

  private notifyDisabled(): void {
    const now = Date.now();
    if (now - this.lastDisabledNoticeAt < DISABLED_NOTICE_COOLDOWN_MS) return;
    this.lastDisabledNoticeAt = now;
    toast.info(translate("settings.voice.voiceTyping.disabledNotice"), {
      action: {
        label: translate("settings.voice.voiceTyping.openSettingsAction"),
        onClick: openVoiceSettings,
      },
    });
  }

  /** Rechecks the permission when a session is still waiting to start. */
  private async assistMicPermission(
    session: StudioDictationSession,
  ): Promise<void> {
    if (this.session !== session || this.phase !== "preparing") return;
    const permission = await queryMicrophonePermission();
    if (permission !== "prompt") return;
    if (this.session !== session || this.phase !== "preparing") return;
    void this.focusMainWindow();
    this.notifyMicPermissionWaiting();
  }

  private notifyMicPermissionWaiting(): void {
    const now = Date.now();
    if (now - this.lastMicNoticeAt < MIC_NOTICE_COOLDOWN_MS) return;
    this.lastMicNoticeAt = now;
    toast.info(translate("settings.voice.voiceTyping.micPermissionWaiting"));
  }

  private notifyMicBlocked(): void {
    toast.error(translate("settings.voice.dictation.micAccessBlockedDesktop"), {
      action: {
        label: translate("settings.voice.dictation.allowMicrophone"),
        onClick: () => {
          void requestVoiceTypingMicrophoneAccess();
        },
      },
    });
  }

  private async focusMainWindow(): Promise<void> {
    try {
      // Dynamic import: the window module is also lazily loaded elsewhere, and a
      // second static edge makes the bundler keep it in the main chunk.
      const { getCurrentWindow } = await import("@tauri-apps/api/window");
      await getCurrentWindow().setFocus();
    } catch {
      // Focus is best effort; the toast still explains what to do.
    }
  }

  private async begin(): Promise<void> {
    if (this.phase !== "idle" && this.phase !== "done") return;
    if (this.session) return;
    if (!isStudioDictationAvailable()) {
      // The canonical explanation the composer also shows.
      notifyStudioDictationUnavailable();
      return;
    }
    this.pressStartedAt = performance.now();
    this.micLiveAt = 0;
    this.firstFinalAt = 0;
    this.stopRequestedAt = 0;
    this.clearHudText();
    this.setPhase("preparing");
    this.transcript = "";
    this.partialText = "";
    this.speechHeard = false;
    this.lastVoiceAt = 0;
    this.recordingStartedAt = 0;
    // The permission prompt can only be answered on this window, and a background
    // press would otherwise wait on an invisible dialog: bring the window forward
    // and explain what is happening. A saved deny does not open the mic at all.
    const pressToken = this.cancelToken;
    // A granted permission cannot be revoked from inside the app, so the hot path
    // skips the query; getUserMedia reports a revocation from the OS itself.
    const permission =
      this.micPermission === "granted"
        ? "granted"
        : await queryMicrophonePermission();
    this.micPermission = permission;
    if (this.disposed || pressToken !== this.cancelToken) return;
    if (permission === "denied") {
      void this.focusMainWindow();
      this.notifyMicBlocked();
      this.setPhase("idle");
      return;
    }
    if (permission === "prompt") {
      void this.focusMainWindow();
      this.notifyMicPermissionWaiting();
    }
    const { removeFillerWords, spokenPunctuation } =
      useVoiceTypingStore.getState();
    const warm = this.takeWarmStream();
    this.sessionStream = warm;
    let session: StudioDictationSession;
    try {
      session = new StudioModelDictationAdapter({
        chatId: null,
        microphone: warm,
        removeFillerWords,
        spokenPunctuation,
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
    if (this.disposed) {
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
        // A segment's live text; the HUD shows it after the finished ones.
        this.partialText = text.trim();
        this.sendHudText(this.composeHudText());
        return;
      }
      this.partialText = "";
      if (text) {
        // Keep a segment that is only a spoken "new paragraph": it is a break,
        // not empty text.
        const next = joinTranscriptParts([this.transcript, text]);
        if (next !== this.transcript) {
          this.transcript = next;
          if (!this.firstFinalAt) this.firstFinalAt = performance.now();
        }
      }
      this.sendHudText(this.composeHudText(), true);
    });
    session.onEnd?.(() => {
      if (this.session !== session) return;
      this.session = null;
      this.releaseLevel();
      this.partialText = "";
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
    // A press-time permission answer can go stale, so recheck once while the
    // session still waits for the microphone to go live.
    this.setTimer(MIC_PERMISSION_WATCHDOG_MS, () => {
      void this.assistMicPermission(session);
    });
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
      if (
        useVoiceTypingStore.getState().hudEnabled &&
        now - this.lastLevelSentAt >= LEVEL_THROTTLE_MS
      ) {
        this.lastLevelSentAt = now;
        void this.invokeQuietly("voice_typing_hud_level", { level });
      }
      if (level > VOICED_LEVEL) {
        this.speechHeard = true;
        this.lastVoiceAt = now;
        return;
      }
      const { autoStopSilence, autoStopDelayMs, mode } =
        useVoiceTypingStore.getState();
      if (!autoStopSilence || mode === "hold") return;
      if (!this.speechHeard || !this.lastVoiceAt) return;
      if (now - this.lastVoiceAt < autoStopDelayMs) return;
      console.info("[voice typing] auto-stop after silence");
      toast.info(translate("settings.voice.voiceTyping.stoppedAfterSilence"));
      this.finish();
    });
  }

  private finish(): void {
    const session = this.session;
    if (!session || this.phase !== "recording") return;
    this.clearTimers();
    this.releaseLevel();
    this.stopRequestedAt = performance.now();
    const token = this.cancelToken;
    const recordedMs = this.recordingStartedAt
      ? Date.now() - this.recordingStartedAt
      : 0;
    this.setPhase("transcribing");
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

  private async deliver(
    session: StudioDictationSession,
    token: number,
    recordedMs: number,
  ): Promise<void> {
    if (this.session === session) this.session = null;
    const text = this.transcript.trim();
    this.transcript = "";
    this.partialText = "";
    if (this.disposed || token !== this.cancelToken) return;
    if (!text) {
      // Genuine silence (or an aborted session): nothing to insert.
      if (recordedMs >= NO_SPEECH_NOTICE_MIN_MS) {
        toast.info(translate("settings.voice.voiceTyping.noSpeechNotice"));
      }
      this.setPhase("idle");
      return;
    }
    this.playCue("stop");
    const {
      appendTrailingSpace,
      keepOnClipboard,
      pasteAutomatically,
      submitAfterInsert,
    } = useVoiceTypingStore.getState();
    try {
      const outcome = await invoke<string>("voice_typing_insert", {
        text: appendTrailingSpace ? `${text} ` : text,
        // A transcript the user still has to paste by hand must not be restored away.
        keepOnClipboard: pasteAutomatically ? keepOnClipboard : true,
        paste: pasteAutomatically,
        // Enter only makes sense once the text was actually pasted.
        submit: pasteAutomatically && submitAfterInsert,
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
      this.setPhase("done");
      this.setTimer(DONE_VISIBLE_MS, () => {
        if (this.phase === "done") this.setPhase("idle");
      });
      // Keep the microphone open for a follow-up dictation; the first session
      // opens it here because it owned the stream it recorded from.
      this.warmMicrophone();
    } catch (error) {
      if (this.disposed || token !== this.cancelToken) return;
      console.error("[voice typing] could not insert the transcript:", error);
      toast.error(translate("settings.voice.voiceTyping.insertFailed"));
      this.setPhase("idle");
    }
  }

  private cancelSession(): void {
    // Invalidates any in-flight delivery from a session stopped earlier.
    this.cancelToken += 1;
    const wasRecording = this.phase === "recording";
    const session = this.session;
    this.session = null;
    this.transcript = "";
    this.partialText = "";
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
    if (wasRecording) this.playCue("cancel");
    this.setPhase("idle");
  }

  private setPhase(phase: HudPhase): void {
    this.phase = phase;
    if (phase === "idle") this.clearHudText();
    void this.invokeQuietly("voice_typing_hud_state", { state: phase });
  }

  private composeHudText(): string {
    return joinTranscriptParts([this.transcript, this.partialText]);
  }

  private sendHudText(text: string, force = false): void {
    if (this.disposed) return;
    const { hudEnabled, hudShowTranscript } = useVoiceTypingStore.getState();
    if (!hudEnabled || !hudShowTranscript) return;
    const now = Date.now();
    if (!force && now - this.hudTextSentAt < HUD_TEXT_THROTTLE_MS) return;
    if (text === this.hudText) return;
    this.hudText = text;
    this.hudTextSentAt = now;
    void this.invokeQuietly("voice_typing_hud_text", { text });
  }

  private clearHudText(): void {
    this.hudText = "";
    this.partialText = "";
    if (this.disposed) return;
    void this.invokeQuietly("voice_typing_hud_text", { text: "" });
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
      useVoiceTypingStore.getState().micWarmSeconds <= 0 ||
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
    if (!enabled || this.disposed || seconds <= 0 || this.warmStream) return;
    const { micDeviceId } = useVoiceSettingsStore.getState();
    void openMicrophone(micDeviceId)
      .then((stream) => {
        if (
          this.disposed ||
          this.session ||
          !useVoiceTypingStore.getState().enabled ||
          useVoiceTypingStore.getState().micWarmSeconds <= 0 ||
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

  private releaseWarmMicrophone(): void {
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
  }

  /** Tiny Web Audio blips; a nicety that must never break dictation. */
  private playCue(kind: Cue): void {
    if (!useVoiceTypingStore.getState().soundEnabled) return;
    try {
      this.audioContext ??= new AudioContext();
      const context = this.audioContext;
      if (context.state === "suspended") void context.resume();
      const [frequency, duration] =
        kind === "start"
          ? [880, 0.07]
          : kind === "stop"
            ? [660, 0.09]
            : [320, 0.1];
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0.0001, context.currentTime);
      gain.gain.exponentialRampToValueAtTime(
        0.06,
        context.currentTime + 0.012,
      );
      gain.gain.exponentialRampToValueAtTime(
        0.0001,
        context.currentTime + duration,
      );
      oscillator.connect(gain).connect(context.destination);
      oscillator.start();
      oscillator.stop(context.currentTime + duration + 0.02);
    } catch {
      // No audio output available.
    }
  }
}

function openVoiceSettings(): void {
  useSettingsDialogStore.getState().openDialog("voice");
}
