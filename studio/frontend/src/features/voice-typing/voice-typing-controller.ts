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
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  isStudioDictationAvailable,
  notifyStudioDictationUnavailable,
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
/** Silence after speech that ends a dictation on its own. */
const AUTO_STOP_SILENCE_MS = 6500;
/** Raw RMS above this counts as speech for the auto-stop. */
const VOICED_LEVEL = 0.05;
/** Shorter recordings that produced nothing had no speech worth reporting. */
const NO_SPEECH_NOTICE_MIN_MS = 2000;
/** How long the "inserted" state stays on the overlay before it hides. */
const DONE_VISIBLE_MS = 900;
/** Mic-level frames are throttled on their way to the overlay window. */
const LEVEL_THROTTLE_MS = 55;
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
    if (state.enabled && !previous.enabled) {
      // An explicit switch-on is the only place the mic prompt may appear.
      void this.activate(true);
      return;
    }
    if (state.enabled && state.preloadModel && !previous.preloadModel) {
      void this.activate(false);
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
        await requestVoiceTypingMicrophoneAccess();
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
    this.setPhase("preparing");
    this.transcript = "";
    this.speechHeard = false;
    this.lastVoiceAt = 0;
    this.recordingStartedAt = 0;
    // The permission prompt can only be answered on this window, and a background
    // press would otherwise wait on an invisible dialog: bring the window forward
    // and explain what is happening. A saved deny does not open the mic at all.
    const pressToken = this.cancelToken;
    const permission = await queryMicrophonePermission();
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
    let session: StudioDictationSession;
    try {
      session = new StudioModelDictationAdapter({ chatId: null }).listen();
    } catch (error) {
      console.error("[voice typing] could not start:", error);
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
      return;
    }
    this.session = session;
    session.onSpeechStart(() => {
      if (this.session !== session || this.phase !== "preparing") return;
      this.recordingStartedAt = Date.now();
      this.setPhase("recording");
      // After the first samples, so the cue means "the mic is live".
      this.playCue("start");
    });
    // The adapter applies the dictionary and records the session for history.
    session.onSpeech((result) => {
      if (!result.isFinal) return;
      const part = result.transcript?.trim() ?? "";
      if (part) {
        this.transcript = this.transcript ? `${this.transcript} ${part}` : part;
      }
    });
    session.onEnd?.(() => {
      if (this.session !== session) return;
      this.session = null;
      this.releaseLevel();
      if (this.phase === "preparing" || this.phase === "recording") {
        // The adapter ended the session on its own (missing model, microphone
        // failure, a download prompt) and reported the reason itself.
        const status = session.status;
        const reason = status.type === "ended" ? status.reason : "unknown";
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
      const { autoStopSilence, mode } = useVoiceTypingStore.getState();
      if (!autoStopSilence || mode === "hold") return;
      if (!this.speechHeard || !this.lastVoiceAt) return;
      if (now - this.lastVoiceAt < AUTO_STOP_SILENCE_MS) return;
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
    const { appendTrailingSpace, keepOnClipboard, pasteAutomatically } =
      useVoiceTypingStore.getState();
    try {
      const outcome = await invoke<string>("voice_typing_insert", {
        text: appendTrailingSpace ? `${text} ` : text,
        // A transcript the user still has to paste by hand must not be restored away.
        keepOnClipboard: pasteAutomatically ? keepOnClipboard : true,
        paste: pasteAutomatically,
      });
      if (outcome === "copied") {
        toast.info(translate("settings.voice.voiceTyping.copiedNotice"));
      }
      console.info("[voice typing] transcript delivered:", outcome, {
        characters: text.length,
      });
      // A completed session proves the model is resident.
      useVoiceTypingStore.getState().setReadiness("ready");
      this.setPhase("done");
      this.setTimer(DONE_VISIBLE_MS, () => {
        if (this.phase === "done") this.setPhase("idle");
      });
    } catch (error) {
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
    this.clearTimers();
    this.releaseLevel();
    try {
      session?.cancel();
    } catch {
      // Already gone.
    }
    if (wasRecording) this.playCue("cancel");
    this.setPhase("idle");
  }

  private setPhase(phase: HudPhase): void {
    this.phase = phase;
    void this.invokeQuietly("voice_typing_hud_state", { state: phase });
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
