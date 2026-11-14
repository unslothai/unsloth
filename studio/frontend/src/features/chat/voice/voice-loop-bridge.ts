// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import type { DictationAdapter } from "@assistant-ui/react";
import { useSyncExternalStore } from "react";

export type VoiceMode = "off" | "configuring" | "active";

let voiceMode: VoiceMode = "off";

let voiceToggle: (() => void) | null = null;
// Toggles clicked before the lazy loop mounted. Only the parity matters: two clicks cancel out.
let voiceTogglesPending = 0;
let voiceRequested = false;
const voiceRequestedListeners = new Set<() => void>();
let voicePlayCtx: AudioContext | null = null;
let voiceThreadReset: ((prevThreadId: string | null) => void) | null = null;
let voiceResume: (() => void) | null = null;
let voiceBargeIn: (() => void) | null = null;
let voiceBargeAbort: (() => void) | null = null;
let voiceSubmit: ((text?: string) => void) | null = null;
let voicePlayback: (() => boolean) | null = null;
let voiceListen: (() => DictationAdapter.Session | null) | null = null;

export const loadVoiceMode = () => import("./voice-engine");

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

export function voiceLoopSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    navigator.mediaDevices?.getUserMedia !== undefined &&
    (window.AudioContext ??
      (window as unknown as { webkitAudioContext?: unknown })
        .webkitAudioContext) !== undefined
  );
}

export function useVoiceAvailable(): boolean {
  return voiceLoopSupported();
}

export function getVoiceMode(): VoiceMode {
  return voiceMode;
}

export function setVoiceMode(next: VoiceMode): void {
  voiceMode = next;
}

export function registerVoiceToggle(fn: (() => void) | null): void {
  voiceToggle = fn;
  if (!fn) return;
  const replay = voiceTogglesPending % 2 === 1;
  voiceTogglesPending = 0;
  if (replay) fn();
}

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
  void loadVoiceMode().catch(() => {});
}

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceThreadReset(fn: ((prevThreadId: string | null) => void) | null): void {
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
export function requestVoiceThreadReset(prevThreadId: string | null): void {
  voiceThreadReset?.(prevThreadId);
}

/** The run to stop when the loop leaves `prevThreadId`: the one it was voicing, while it still
 *  runs. Left going, it kept generating beside the new chat's run under the same voice. */
export function voiceRunToStop(
  prevThreadId: string | null,
  voicedThreadId: string | null,
  runningByThreadId: Record<string, boolean>,
): string | null {
  if (!prevThreadId || prevThreadId !== voicedThreadId) return null;
  return runningByThreadId[prevThreadId] ? prevThreadId : null;
}

export function registerVoiceResume(fn: (() => void) | null): void {
  voiceResume = fn;
}

export function requestVoiceResume(): void {
  voiceResume?.();
}

export function registerVoiceBargeIn(fn: (() => void) | null): void {
  voiceBargeIn = fn;
}

export function requestVoiceBargeIn(): void {
  voiceBargeIn?.();
}

export function registerVoiceBargeAbort(fn: (() => void) | null): void {
  voiceBargeAbort = fn;
}

export function requestVoiceBargeAbort(): void {
  voiceBargeAbort?.();
}

export type ToolCallInfo = { toolName: string; args: unknown };
let toolCallDescriber: ((toolCallId: string) => ToolCallInfo | null) | null = null;

/** Called by the mounted voice loop, which holds the thread. Pass null on teardown. */
export function registerToolCallDescriber(
  fn: ((toolCallId: string) => ToolCallInfo | null) | null,
): void {
  toolCallDescriber = fn;
}

/** The tool and arguments behind a pending approval, for the floating orb's popup. */
export function describeToolCall(toolCallId: string): ToolCallInfo | null {
  return toolCallDescriber?.(toolCallId) ?? null;
}

let voiceEarlyInterject: ((fragment: string) => void) | null = null;

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceEarlyInterject(fn: ((fragment: string) => void) | null): void {
  voiceEarlyInterject = fn;
}

/** The turn judge is holding `fragment` for more: a run it talks over can be cut now. */
export function requestVoiceEarlyInterject(fragment: string): void {
  voiceEarlyInterject?.(fragment);
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

let voiceTypedSubmit: ((box: string) => boolean) | null = null;

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceTypedSubmit(fn: ((box: string) => boolean) | null): void {
  voiceTypedSubmit = fn;
}

/**
 * Enter or Send while the loop is listening. True when the loop took the box as a
 * typed turn and sent it; false means the box is only what voice put there, so the
 * caller ends the spoken turn instead.
 */
export function requestVoiceTypedSubmit(box: string): boolean {
  return voiceTypedSubmit?.(box) ?? false;
}

let voicePauseToggle: (() => void) | null = null;

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoicePauseToggle(fn: (() => void) | null): void {
  voicePauseToggle = fn;
}

/** The composer's tap: pause the loop (mic muted, playback frozen) or resume it. */
export function requestVoicePauseToggle(): void {
  voicePauseToggle?.();
}

let voiceStopSpeaking: (() => void) | null = null;

/** Called by the mounted voice loop. Pass null on teardown. */
export function registerVoiceStopSpeaking(fn: (() => void) | null): void {
  voiceStopSpeaking = fn;
}

/** The composer's stop: cut the reply being spoken, the run behind it and anything held, and go back to listening. Voice mode stays on. */
export function requestVoiceStopSpeaking(): void {
  voiceStopSpeaking?.();
}

export function registerVoicePlayback(fn: (() => boolean) | null): void {
  voicePlayback = fn;
}

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

export function startVoiceListen(): DictationAdapter.Session | null {
  return voiceListen?.() ?? null;
}
