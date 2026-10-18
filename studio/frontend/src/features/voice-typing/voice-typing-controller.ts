// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Background orchestration for system-wide voice typing.
//
// The native layer watches the global shortcut and emits start/stop/cancel
// signals. This controller records and transcribes through the same local
// dictation stack the composer microphone uses (engine, model, language, mic and
// dictionary all come from Voice settings), then hands the finished transcript
// back to the native layer, which types it into whichever app has focus.

import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import {
  StudioModelDictationAdapter,
  subscribeDictationLevel,
  type StudioDictationSession,
} from "@/features/chat";
import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { useVoiceTypingStore } from "./voice-typing-store";

interface VoiceTypingInfo {
  os: string;
  shortcutSupported: boolean;
  holdSupported: boolean;
  pasteSupported: boolean;
}

type SignalAction = "start" | "stop" | "cancel";
type HudPhase = "idle" | "recording" | "transcribing" | "done";
type Cue = "start" | "stop" | "cancel";

/** Safety net so a forgotten toggle cannot record forever. */
const MAX_RECORDING_MS = 5 * 60 * 1000;
/** How long the "inserted" state stays on the overlay before it hides. */
const DONE_VISIBLE_MS = 900;
/** Mic-level frames are throttled on their way to the overlay window. */
const LEVEL_THROTTLE_MS = 55;

export class VoiceTypingController {
  private info: VoiceTypingInfo | null = null;
  private phase: HudPhase = "idle";
  private session: StudioDictationSession | null = null;
  private transcript = "";
  private ready = false;
  private disposed = false;
  private wasEnabled = false;
  private unlistenSignal: UnlistenFn | null = null;
  private unsubscribeLevel: (() => void) | null = null;
  private unsubscribeStore: (() => void) | null = null;
  private timers = new Set<number>();
  private lastLevelSentAt = 0;
  private audioContext: AudioContext | null = null;

  async init(): Promise<void> {
    this.info = await invoke<VoiceTypingInfo>("voice_typing_info").catch(
      () => null,
    );
    if (this.disposed || !this.info?.shortcutSupported) return;
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
    this.wasEnabled = useVoiceTypingStore.getState().enabled;
    this.ready = true;
    await this.pushConfig();
    this.unsubscribeStore = useVoiceTypingStore.subscribe(() => {
      const state = useVoiceTypingStore.getState();
      void this.pushConfig();
      if (state.enabled && !this.wasEnabled) void this.warmMicrophone();
      this.wasEnabled = state.enabled;
    });
  }

  dispose(): void {
    this.disposed = true;
    this.unlistenSignal?.();
    this.unlistenSignal = null;
    this.unsubscribeLevel?.();
    this.unsubscribeLevel = null;
    this.unsubscribeStore?.();
    this.unsubscribeStore = null;
    this.clearTimers();
    try {
      this.session?.cancel();
    } catch {
      // The session may already have ended.
    }
    this.session = null;
    this.phase = "idle";
  }

  private async pushConfig(): Promise<void> {
    if (!this.ready || this.disposed) return;
    const { enabled, shortcut, mode, hudEnabled } = useVoiceTypingStore.getState();
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
      toast.error(message);
    }
  }

  /** Resolving the microphone prompt needs a visible window; the enable toggle is it. */
  private async warmMicrophone(): Promise<void> {
    if (!navigator.mediaDevices?.getUserMedia) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      for (const track of stream.getTracks()) track.stop();
    } catch {
      toast.error(translate("settings.voice.voiceTyping.micDenied"));
    }
  }

  private async onSignal(action: SignalAction): Promise<void> {
    if (action === "start") {
      if (this.phase === "idle") await this.begin();
      return;
    }
    if (action === "stop") {
      if (this.phase === "recording") this.finish();
      else if (this.phase === "transcribing") this.cancelSession();
      return;
    }
    if (this.phase !== "idle") this.cancelSession();
  }

  private async begin(): Promise<void> {
    if (this.phase !== "idle" || this.session) return;
    if (!StudioModelDictationAdapter.isSupported()) {
      toast.error(translate("settings.voice.voiceTyping.micDenied"));
      return;
    }
    let session: StudioDictationSession;
    try {
      session = new StudioModelDictationAdapter({ chatId: null }).listen();
    } catch (error) {
      console.error("Voice typing could not start:", error);
      toast.error(translate("settings.voice.voiceTyping.startFailed"));
      return;
    }
    this.session = session;
    this.transcript = "";
    this.setPhase("recording");
    this.playCue("start");
    // The adapter applies the dictionary and records the session for history.
    session.onSpeech((result) => {
      if (!result.isFinal) return;
      const part = result.transcript?.trim() ?? "";
      if (part) {
        this.transcript = this.transcript ? `${this.transcript} ${part}` : part;
      }
    });
    session.onEnd?.(() => {
      if (this.session === session) this.session = null;
    });
    this.unsubscribeLevel?.();
    this.unsubscribeLevel = subscribeDictationLevel((level) => {
      if (this.phase !== "recording") return;
      const now = Date.now();
      if (now - this.lastLevelSentAt < LEVEL_THROTTLE_MS) return;
      this.lastLevelSentAt = now;
      void this.invokeQuietly("voice_typing_hud_level", { level });
    });
    this.setTimer(MAX_RECORDING_MS, () => {
      if (this.phase === "recording") this.finish();
    });
  }

  private finish(): void {
    const session = this.session;
    if (!session || this.phase !== "recording") return;
    this.clearTimers();
    this.unsubscribeLevel?.();
    this.unsubscribeLevel = null;
    this.setPhase("transcribing");
    void (async () => {
      try {
        // Resolves once the final segment is transcribed and onSpeech has fired.
        await session.stop();
      } catch (error) {
        console.error("Voice typing could not finish the recording:", error);
      }
      await this.deliver(session);
    })();
  }

  private async deliver(session: StudioDictationSession): Promise<void> {
    if (this.session === session) this.session = null;
    const text = this.transcript.trim();
    this.transcript = "";
    if (this.disposed) return;
    if (!text) {
      // Genuine silence (or a cancelled download prompt): nothing to insert.
      this.setPhase("idle");
      return;
    }
    this.playCue("stop");
    const { appendTrailingSpace, keepOnClipboard } =
      useVoiceTypingStore.getState();
    try {
      const outcome = await invoke<string>("voice_typing_insert", {
        text: appendTrailingSpace ? `${text} ` : text,
        keepOnClipboard,
      });
      if (outcome === "copied") {
        toast.info(translate("settings.voice.voiceTyping.copiedNotice"));
      }
      this.setPhase("done");
      this.setTimer(DONE_VISIBLE_MS, () => {
        if (this.phase === "done") this.setPhase("idle");
      });
    } catch (error) {
      console.error("Voice typing could not insert the transcript:", error);
      toast.error(translate("settings.voice.voiceTyping.insertFailed"));
      this.setPhase("idle");
    }
  }

  private cancelSession(): void {
    const wasRecording = this.phase === "recording";
    const session = this.session;
    this.session = null;
    this.transcript = "";
    this.clearTimers();
    this.unsubscribeLevel?.();
    this.unsubscribeLevel = null;
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
        kind === "start" ? [880, 0.07] : kind === "stop" ? [660, 0.09] : [320, 0.1];
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      oscillator.type = "sine";
      oscillator.frequency.value = frequency;
      gain.gain.setValueAtTime(0.0001, context.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.06, context.currentTime + 0.012);
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
