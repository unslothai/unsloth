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

export function registerVoiceThreadReset(fn: ((prevThreadId: string | null) => void) | null): void {
  voiceThreadReset = fn;
}

// Must not call the plain toggle: that turned voice OFF on every new chat.
export function requestVoiceThreadReset(prevThreadId: string | null): void {
  voiceThreadReset?.(prevThreadId);
}

// Left going, the old run kept generating beside the new chat's run under the same voice.
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

export function registerToolCallDescriber(
  fn: ((toolCallId: string) => ToolCallInfo | null) | null,
): void {
  toolCallDescriber = fn;
}

export function describeToolCall(toolCallId: string): ToolCallInfo | null {
  return toolCallDescriber?.(toolCallId) ?? null;
}

let voiceEarlyInterject: ((fragment: string) => void) | null = null;

export function registerVoiceEarlyInterject(fn: ((fragment: string) => void) | null): void {
  voiceEarlyInterject = fn;
}

export function requestVoiceEarlyInterject(fragment: string): void {
  voiceEarlyInterject?.(fragment);
}

export function registerVoiceSubmit(fn: ((text?: string) => void) | null): void {
  voiceSubmit = fn;
}

// A no-op outside voice mode: that is what leaves the plain Dictate button filling the composer without sending.
export function requestVoiceSubmit(text?: string): void {
  voiceSubmit?.(text);
}

let voiceTypedSubmit: ((box: string) => boolean) | null = null;

export function registerVoiceTypedSubmit(fn: ((box: string) => boolean) | null): void {
  voiceTypedSubmit = fn;
}

// False means the box is only what voice put there, so the caller ends the spoken turn instead.
export function requestVoiceTypedSubmit(box: string): boolean {
  return voiceTypedSubmit?.(box) ?? false;
}

let voicePauseToggle: (() => void) | null = null;

export function registerVoicePauseToggle(fn: (() => void) | null): void {
  voicePauseToggle = fn;
}

export function requestVoicePauseToggle(): void {
  voicePauseToggle?.();
}

let voiceStopSpeaking: (() => void) | null = null;

export function registerVoiceStopSpeaking(fn: (() => void) | null): void {
  voiceStopSpeaking = fn;
}

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

export function getVoiceParkedText(): string {
  return voiceParkedText?.() ?? "";
}

export function registerVoiceListen(
  fn: () => DictationAdapter.Session | null,
): void {
  voiceListen = fn;
}

export function startVoiceListen(): DictationAdapter.Session | null {
  return voiceListen?.() ?? null;
}

let voiceSpokenAudio: ((threadId: string | null, wavBase64: string) => void) | null = null;

export function registerVoiceSpokenAudio(
  fn: ((threadId: string | null, wavBase64: string) => void) | null,
): void {
  voiceSpokenAudio = fn;
}

export function publishVoiceSpokenAudio(threadId: string | null, wavBase64: string): void {
  voiceSpokenAudio?.(threadId, wavBase64);
}
