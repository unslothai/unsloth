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

/** Whether the conversation loop is off, being configured, or running. */
export type VoiceMode = "off" | "configuring" | "active";

// The loop's own mode, distinct from the store field of the same name: this one
// survives the ThreadWelcome -> ThreadComposerDock remount that the first send
// triggers, which is exactly when the loop must not forget it is running.
let voiceMode: VoiceMode = "off";

let voiceToggle: (() => void) | null = null;
let voiceResume: (() => void) | null = null;
let voiceBargeIn: (() => void) | null = null;
let voiceSubmit: (() => void) | null = null;

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
}

/**
 * Drive the same off/configuring/active toggle the plus menu uses -- from the orb
 * overlay's close button, its Esc handler, or the chat page's thread-switch reset.
 * A no-op while the loop is unmounted.
 */
export function requestVoiceToggle(): void {
  voiceToggle?.();
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
 * Cut the currently-playing reply because the user has started talking over it.
 * The adapter raises this the instant an utterance commits to being transcribed,
 * so the model is not still speaking into a turn it is about to lose. The handler
 * no-ops unless something is actually audible, so stray calls are harmless.
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
