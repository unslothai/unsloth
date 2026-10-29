// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { persist } from "zustand/middleware";
import type { AudioSourceSelection } from "../audio-run-request";
import type {
  MusicDrafts,
  MusicEditAction,
  MusicEditDraft,
  MusicMode,
  MusicSfxDraft,
  MusicSongDraft,
} from "../music/music-types";

export const AUDIO_MUSIC_STORAGE_KEY = "unsloth_audio_music_v1";

export const DEFAULT_MUSIC_DRAFTS: MusicDrafts = {
  song: { instrumental: false, durationS: null, variations: 1 },
  sfx: { prompt: "", durationS: null, variations: 1 },
  edit: {
    source: null,
    action: null,
    ranges: [],
    strength: null,
    extendS: 15,
    prompt: "",
  },
};

type DraftPatch<M extends MusicMode> = M extends "song"
  ? Partial<MusicSongDraft>
  : M extends "sfx"
    ? Partial<MusicSfxDraft>
    : Partial<MusicEditDraft>;

interface AudioMusicState extends MusicDrafts {
  mode: MusicMode;
  setMode: (mode: MusicMode) => void;
  patchDraft: <M extends MusicMode>(mode: M, patch: DraftPatch<M>) => void;
  pushClipToEdit: (
    clip: AudioSourceSelection,
    action?: MusicEditAction | null,
  ) => void;
  loadedEditActions: readonly MusicEditAction[];
  setLoadedEditActions: (actions: readonly MusicEditAction[]) => void;
}

export const useAudioMusicStore = create<AudioMusicState>()(
  persist(
    (set) => ({
      mode: "song",
      ...DEFAULT_MUSIC_DRAFTS,
      loadedEditActions: [],
      setLoadedEditActions: (loadedEditActions) => set({ loadedEditActions }),
      setMode: (mode) => set({ mode }),
      patchDraft: (mode, patch) =>
        set(
          (state) =>
            ({
              [mode]: { ...state[mode], ...patch },
            }) as Partial<AudioMusicState>,
        ),
      pushClipToEdit: (clip, action = null) =>
        set((state) => ({
          mode: "edit",
          edit: {
            ...state.edit,
            source: clip,
            // A new clip's old ranges point at the wrong audio.
            ranges: state.edit.source?.id === clip.id ? state.edit.ranges : [],
            action: action ?? state.edit.action,
            prompt: state.edit.prompt.trim() ? state.edit.prompt : clip.name,
          },
        })),
    }),
    {
      name: AUDIO_MUSIC_STORAGE_KEY,
      version: 1,
      partialize: (state) => ({
        mode: state.mode,
        song: state.song,
        sfx: state.sfx,
        edit: state.edit,
      }),
      // Drafts saved by an older build may miss fields added since.
      merge: (persisted, current) => {
        const saved = (persisted ?? {}) as Partial<
          MusicDrafts & { mode: MusicMode }
        >;
        return {
          ...current,
          mode: saved.mode ?? current.mode,
          song: { ...current.song, ...saved.song },
          sfx: { ...current.sfx, ...saved.sfx },
          edit: { ...current.edit, ...saved.edit },
        };
      },
    },
  ),
);
