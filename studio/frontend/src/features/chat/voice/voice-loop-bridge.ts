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

/** Whether the conversation loop is off, being configured, or running. */
export type VoiceMode = "off" | "configuring" | "active";

// The loop's own mode, distinct from the store field of the same name: this one
// survives the ThreadWelcome -> ThreadComposerDock remount that the first send
// triggers, which is exactly when the loop must not forget it is running.
let voiceMode: VoiceMode = "off";

let voiceToggle: (() => void) | null = null;
// A toggle clicked before the lazy loop mounted; it runs once the loop registers.
let voiceTogglePending = false;
let voiceThreadReset: (() => void) | null = null;
let voiceResume: (() => void) | null = null;
let voiceBargeIn: (() => void) | null = null;
let voiceSubmit: (() => void) | null = null;
let voicePlayback: (() => boolean) | null = null;
let voiceListen: (() => DictationAdapter.Session | null) | null = null;

/**
 * The loop's module, loaded on use so voice mode stays off the first screen.
 * Everything voice loads through this one entry: a second would split shared
 * modules into extra startup chunks.
 */
export const loadVoiceMode = () => import("./voice-engine");

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
  if (fn && voiceTogglePending) {
    voiceTogglePending = false;
    fn();
  }
}

/**
 * Drive the same off/configuring/active toggle the plus menu uses -- from the orb
 * overlay's close button, its Esc handler, or the chat page's thread-switch reset.
 * Before the lazy loop has mounted, the toggle waits for it instead of being lost;
 * audio is then primed after the click rather than inside it.
 */
export function requestVoiceToggle(): void {
  if (voiceToggle) {
    voiceToggle();
    return;
  }
  voiceTogglePending = true;
  void loadVoiceMode();
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
 * Cut the reply because the user has started talking over it: the adapter raises
 * this once a burst over its barge gate holds during playback. The handler no-ops
 * unless something is actually audible, so stray calls are harmless.
 */
export function requestVoiceBargeIn(): void {
  voiceBargeIn?.();
}

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceSubmit(fn: (() => void) | null): void {
  voiceSubmit = fn;
}

/**
 * Send the utterance the adapter has just committed into the composer. The
 * adapter owns end-of-turn, so it is also what says the turn is ready to go: a
 * no-op outside voice mode, which is what leaves the plain Dictate button filling
 * the composer without sending.
 */
export function requestVoiceSubmit(): void {
  voiceSubmit?.();
}

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoicePlayback(fn: (() => boolean) | null): void {
  voicePlayback = fn;
}

/** Whether a reply is being spoken; the adapter listens only for barge-in then. */
export function isVoicePlaybackActive(): boolean {
  return voicePlayback?.() ?? false;
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
