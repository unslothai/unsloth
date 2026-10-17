// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Module-level registry for the voice conversation loop: its mode, and the two
 * calls other parts of the app make back into it.
 *
 * Everything here exists because the loop's own component remounts. Sending the
 * first message swaps ThreadWelcome for ThreadComposerDock, so a subscription or a
 * React ref would be torn down at exactly the moment the loop must not forget it is
 * running. These module bindings survive it.
 *
 * It is also the only thing a dictation adapter or the chat page has to import to
 * reach the loop. Importing the loop itself would be a cycle
 * (runtime-provider -> adapter -> voice-engine -> ...), and bundlers resolve those
 * by handing one side a partially initialised module.
 */

import type { DictationAdapter } from "@assistant-ui/react";
import { useSyncExternalStore } from "react";

/** Whether the conversation loop is off, being configured, or running. */
export type VoiceMode = "off" | "configuring" | "active";

// The loop's own mode, distinct from the store field of the same name: this one
// survives the ThreadWelcome -> ThreadComposerDock remount that the first send
// triggers, which is exactly when the loop must not forget it is running.
let voiceMode: VoiceMode = "off";

let voiceToggle: (() => void) | null = null;
// Toggles clicked before the lazy loop mounted. Only the parity matters: two clicks cancel out.
let voiceTogglesPending = 0;
// Set by the first Voice click and kept: the loop's components mount only from then on.
let voiceRequested = false;
const voiceRequestedListeners = new Set<() => void>();
let voicePlayCtx: AudioContext | null = null;
let voiceThreadReset: (() => void) | null = null;
let voiceResume: (() => void) | null = null;
let voiceBargeIn: (() => void) | null = null;
let voiceBargeAbort: (() => void) | null = null;
let voiceSubmit: ((text?: string) => void) | null = null;
let voicePlayback: (() => boolean) | null = null;
let voiceListen: (() => DictationAdapter.Session | null) | null = null;

/**
 * The loop's module, loaded on use so voice mode stays off the first screen.
 * Everything voice loads through this one entry: a second would split shared
 * modules into extra startup chunks.
 */
export const loadVoiceMode = () => import("./voice-engine");

/** Whether Voice has been clicked this session; the lazy loop is not rendered before it. */
export function useVoiceRequested(): boolean {
  return useSyncExternalStore(
    subscribeVoiceRequested,
    isVoiceRequested,
    isVoiceRequested,
  );
}

export function isVoiceRequested(): boolean {
  return voiceRequested;
}

export function subscribeVoiceRequested(listener: () => void): () => void {
  voiceRequestedListeners.add(listener);
  return () => voiceRequestedListeners.delete(listener);
}

/** The context voice mode plays through. Here, not in the lazy player, so the first click can unlock it. */
export function getVoicePlayContext(): AudioContext | null {
  if (typeof window === "undefined") return null;
  const Ctor =
    window.AudioContext ??
    (window as unknown as { webkitAudioContext?: typeof AudioContext })
      .webkitAudioContext;
  if (!Ctor) return null;
  if (!voicePlayCtx) voicePlayCtx = new Ctor();
  return voicePlayCtx;
}

/** What the loop's adapter needs to listen: getUserMedia (secure contexts only) and Web Audio. */
export function voiceLoopSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    navigator.mediaDevices?.getUserMedia !== undefined &&
    (window.AudioContext ??
      (window as unknown as { webkitAudioContext?: unknown })
        .webkitAudioContext) !== undefined
  );
}

/**
 * Whether the plus menu may offer Voice at all. Listening is the gate. Speaking is
 * not gated: voice mode always has an audio.cpp voice picked (Kokoro by default).
 */
export function useVoiceAvailable(): boolean {
  // Model transcription only. The loop used to follow the Dictate button's
  // dictationEngine setting and run on the browser's SpeechRecognition when that
  // said "browser" -- which meant voice mode silently depended on an online
  // Google endpoint, and did not exist at all in the desktop WebView. It now
  // always runs a local transcription model, so this is the only thing to check.
  return voiceLoopSupported();
}

/** The loop's current mode. Safe to call from anywhere, including render. */
export function getVoiceMode(): VoiceMode {
  return voiceMode;
}

/** Written only by the loop itself as it transitions. */
export function setVoiceMode(next: VoiceMode): void {
  voiceMode = next;
}

/** Registered by the mounted loop. Pass null on teardown. */
export function registerVoiceToggle(fn: (() => void) | null): void {
  voiceToggle = fn;
  if (!fn) return;
  const replay = voiceTogglesPending % 2 === 1;
  voiceTogglesPending = 0;
  if (replay) fn();
}

/**
 * Drive the same off/configuring/active toggle the plus menu uses -- from the orb
 * overlay's close button, its Esc handler, or the chat page's thread-switch reset.
 * Before the lazy loop has mounted, the toggle waits for it instead of being lost,
 * and the playback context is unlocked here, inside the click.
 */
export function requestVoiceToggle(): void {
  if (voiceToggle) {
    voiceToggle();
    return;
  }
  voiceTogglesPending += 1;
  void getVoicePlayContext()?.resume().catch(() => {});
  if (!voiceRequested) {
    voiceRequested = true;
    for (const listener of voiceRequestedListeners) listener();
  }
  // A failed chunk surfaces through the lazy components' own load.
  void loadVoiceMode().catch(() => {});
}

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceThreadReset(fn: (() => void) | null): void {
  voiceThreadReset = fn;
}

/**
 * Carry the loop across a thread switch: cut whatever the last thread had in
 * flight (playing reply, open mic, pending timers) and re-arm on the new one.
 *
 * Voice mode itself survives, the same way a loaded chat model does. This used to
 * call the plain toggle, which turned voice OFF -- so starting a new chat silently
 * ended the conversation mode and the controls vanished from the header.
 * A no-op while the loop is unmounted or off.
 */
export function requestVoiceThreadReset(): void {
  voiceThreadReset?.();
}

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceResume(fn: (() => void) | null): void {
  voiceResume = fn;
}

/**
 * Re-arm the mic after a recoverable end of session -- a Web Speech "no-speech"
 * timeout, say, where the engine heard nothing and gave up but the conversation is
 * still going. A no-op unless the loop is mounted and voice mode is still active, so
 * an adapter used outside voice mode can call it freely.
 */
export function requestVoiceResume(): void {
  voiceResume?.();
}

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceBargeIn(fn: (() => void) | null): void {
  voiceBargeIn = fn;
}

/**
 * Hold the reply because the user may have started talking over it: the adapter
 * raises this once a burst over its barge gate holds during playback. The reply is
 * only cut once the utterance proves real (requestVoiceSubmit); a false alarm
 * resumes it (requestVoiceBargeAbort). No-ops unless something is audible.
 */
export function requestVoiceBargeIn(): void {
  voiceBargeIn?.();
}

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceBargeAbort(fn: (() => void) | null): void {
  voiceBargeAbort = fn;
}

/**
 * The burst that paused the reply turned out not to be the user (a cough, a hum,
 * nothing STT could use): let the reply carry on from where it was held.
 */
export function requestVoiceBargeAbort(): void {
  voiceBargeAbort?.();
}

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceSubmit(fn: ((text?: string) => void) | null): void {
  voiceSubmit = fn;
}

/**
 * Send the utterance the adapter has just finished (the composer text when none
 * is given). The
 * adapter owns end-of-turn, so it is also what says the turn is ready to go: a
 * no-op outside voice mode, which is what leaves the plain Dictate button filling
 * the composer without sending.
 */
export function requestVoiceSubmit(text?: string): void {
  voiceSubmit?.(text);
}

let voiceStopSpeaking: (() => void) | null = null;

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceStopSpeaking(fn: (() => void) | null): void {
  voiceStopSpeaking = fn;
}

/** Cut the reply being spoken (and the run behind it) and go back to listening. */
export function requestVoiceStopSpeaking(): void {
  voiceStopSpeaking?.();
}

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoicePlayback(fn: (() => boolean) | null): void {
  voicePlayback = fn;
}

/** Whether a reply is being spoken; the adapter listens only for barge-in then. */
export function isVoicePlaybackActive(): boolean {
  return voicePlayback?.() ?? false;
}

let voiceReplyText: (() => string) | null = null;

export function registerVoiceReplyText(fn: (() => string) | null): void {
  voiceReplyText = fn;
}

/** The reply being spoken, so a transcript of its own leak is never a turn. */
export function getVoiceReplyText(): string {
  return voiceReplyText?.() ?? "";
}

let voiceParkedText: (() => string) | null = null;

export function registerVoiceParkedText(fn: (() => string) | null): void {
  voiceParkedText = fn;
}

/** Turns parked while the voice loads, so the adapter can show them stacking in the composer. */
export function getVoiceParkedText(): string {
  return voiceParkedText?.() ?? "";
}

/** Set once by the loop's module as it loads, which voice mode needs before it can go active. */
export function registerVoiceListen(
  fn: () => DictationAdapter.Session | null,
): void {
  voiceListen = fn;
}

/** The loop's own dictation session; null before its module loads or where it can't listen. */
export function startVoiceListen(): DictationAdapter.Session | null {
  return voiceListen?.() ?? null;
}
