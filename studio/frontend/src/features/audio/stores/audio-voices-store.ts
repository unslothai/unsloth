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
    try {
      const voices = await listVoices();
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
    set((state) => ({
      voices: [voice, ...state.voices.filter((item) => item.id !== voice.id)],
    }));
    return voice;
  },
  rename: async (id, patch) => {
    const before = get().voices;
    set({
      voices: before.map((voice) =>
        voice.id === id ? { ...voice, ...patch } : voice,
      ),
    });
    try {
      const saved = await updateVoice(id, patch);
      set((state) => ({
        voices: state.voices.map((voice) => (voice.id === id ? saved : voice)),
      }));
    } catch (error) {
      set({ voices: before });
      throw error;
    }
  },
  remove: async (id) => {
    const before = get().voices;
    set({ voices: before.filter((voice) => voice.id !== id) });
    try {
      await deleteVoice(id);
    } catch (error) {
      set({ voices: before });
      throw error;
    }
  },
}));
