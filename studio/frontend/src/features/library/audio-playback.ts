// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { translate } from "@/i18n";
import { toast } from "@/lib/toast";
import { useSyncExternalStore } from "react";
import { type LibraryItem, errorMessage, fetchLibraryStreamUrl } from "./api";

// One player for every card, so starting a clip stops the one before it.
let player: HTMLAudioElement | null = null;
let playingId: string | null = null;
// Bumped by every play and stop, so a link that arrives late never starts a clip left behind.
let generation = 0;
const listeners = new Set<() => void>();

function setPlaying(id: string | null): void {
  playingId = id;
  for (const listener of listeners) listener();
}

export function stopLibraryAudio(): void {
  generation += 1;
  player?.pause();
  if (playingId !== null) setPlaying(null);
}

/** A hidden card is the clip's only control, so its clip stops. */
export function stopLibraryAudioUnlessShown(shown: ReadonlySet<string>): void {
  if (playingId !== null && !shown.has(playingId)) stopLibraryAudio();
}

function failed(item: LibraryItem, error: unknown, mine: number): void {
  if (mine !== generation) return;
  // A failed stream can both reject play() and fire error: report it once.
  generation += 1;
  setPlaying(null);
  toast.error(translate("library.audio.playFailed", { name: item.name }), {
    description: errorMessage(error),
  });
}

export async function toggleLibraryAudio(item: LibraryItem): Promise<void> {
  const wasPlaying = playingId === item.id;
  stopLibraryAudio();
  if (wasPlaying) return;
  const mine = generation;
  // Shown as playing while the link is fetched, so a second click stops it rather than restarts.
  setPlaying(item.id);
  let url: string;
  try {
    url = await fetchLibraryStreamUrl(item);
  } catch (error) {
    failed(item, error, mine);
    return;
  }
  if (mine !== generation) return;
  player ??= new Audio();
  player.onended = () => mine === generation && setPlaying(null);
  // A stream that fails after play() resolved never rejects it.
  player.onerror = () => failed(item, player?.error?.message ?? "", mine);
  player.src = url;
  player.play().catch((error: unknown) => failed(item, error, mine));
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function useLibraryAudioPlaying(id: string): boolean {
  return useSyncExternalStore(subscribe, () => playingId === id);
}
