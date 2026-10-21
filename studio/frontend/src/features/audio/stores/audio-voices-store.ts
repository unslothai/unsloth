// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import {
  type AudioVoice,
  createVoice,
  deleteVoice,
  listVoices,
  updateVoice,
} from "../api";
import type { AudioVoiceCreateRequest } from "../audio-run-request";

interface AudioVoicesState {
  voices: AudioVoice[];
  loaded: boolean;
  loading: boolean;
  error: string | null;
  refresh: () => Promise<void>;
  save: (request: AudioVoiceCreateRequest) => Promise<AudioVoice>;
  rename: (
    id: string,
    patch: Pick<AudioVoice, "name" | "transcript" | "language">,
  ) => Promise<void>;
  remove: (id: string) => Promise<void>;
}

// Bumped by each save, rename and delete: a list fetched across one is stale and is fetched again.
let mutations = 0;

const message = (error: unknown, fallback: string) =>
  error instanceof Error ? error.message : fallback;

export const useAudioVoicesStore = create<AudioVoicesState>()((set, get) => ({
  voices: [],
  loaded: false,
  loading: false,
  error: null,
  refresh: async () => {
    if (get().loading) return;
    set({ loading: true });
    const started = mutations;
    try {
      const voices = await listVoices();
      if (started !== mutations) {
        set({ loading: false });
        return get().refresh();
      }
      set({ voices, loaded: true, loading: false, error: null });
    } catch (error) {
      set({
        loading: false,
        loaded: true,
        error: message(error, "Could not list saved voices."),
      });
    }
  },
  save: async (request) => {
    const voice = await createVoice(request);
    mutations += 1;
    set((state) => ({
      voices: [voice, ...state.voices.filter((item) => item.id !== voice.id)],
    }));
    return voice;
  },
  rename: async (id, patch) => {
    const previous = get().voices.find((voice) => voice.id === id);
    set((state) => ({
      voices: state.voices.map((voice) =>
        voice.id === id ? { ...voice, ...patch } : voice,
      ),
    }));
    try {
      const saved = await updateVoice(id, patch);
      mutations += 1;
      set((state) => ({
        voices: state.voices.map((voice) => (voice.id === id ? saved : voice)),
      }));
    } catch (error) {
      // Put back only this voice: a save or deletion may have finished meanwhile.
      if (previous) {
        set((state) => ({
          voices: state.voices.map((voice) =>
            voice.id === id ? previous : voice,
          ),
        }));
      }
      throw error;
    }
  },
  remove: async (id) => {
    const before = get().voices;
    const index = before.findIndex((voice) => voice.id === id);
    set({ voices: before.filter((voice) => voice.id !== id) });
    try {
      await deleteVoice(id);
      mutations += 1;
    } catch (error) {
      // Put back only this voice: another deletion may have finished meanwhile.
      const failed = before[index];
      if (failed) {
        set((state) => {
          if (state.voices.some((voice) => voice.id === id)) return {};
          const voices = [...state.voices];
          voices.splice(Math.min(index, voices.length), 0, failed);
          return { voices };
        });
      }
      throw error;
    }
  },
}));
