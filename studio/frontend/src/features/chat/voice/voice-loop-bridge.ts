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
let voiceThreadReset: (() => void) | null = null;
let voiceResume: (() => void) | null = null;
let voiceBargeIn: (() => void) | null = null;
let voiceBargeAbort: (() => void) | null = null;
let voiceSubmit: (() => void) | null = null;
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

export function registerVoiceThreadReset(fn: (() => void) | null): void {
  voiceThreadReset = fn;
}

export function requestVoiceThreadReset(): void {
  voiceThreadReset?.();
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

export function registerVoiceSubmit(fn: (() => void) | null): void {
  voiceSubmit = fn;
}

export function requestVoiceSubmit(): void {
  voiceSubmit?.();
}

export function registerVoicePlayback(fn: (() => boolean) | null): void {
  voicePlayback = fn;
}

export function isVoicePlaybackActive(): boolean {
  return voicePlayback?.() ?? false;
}

export function registerVoiceListen(
  fn: () => DictationAdapter.Session | null,
): void {
  voiceListen = fn;
}

export function startVoiceListen(): DictationAdapter.Session | null {
  return voiceListen?.() ?? null;
}
