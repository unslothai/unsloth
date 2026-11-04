// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { AUTH_SESSION_CLEARED_EVENT } from "@/features/auth";
import {
  getChatIndex,
  getChatScene,
  listCharacters,
  listPersonas,
  updateChatScene,
} from "./api";
import type { Character, ChatScene, Persona, SceneSettings } from "./types";

type LoadState = "idle" | "loading" | "ready" | "error";

interface CharactersState {
  characters: Record<string, Character>;
  order: string[];
  status: LoadState;
  personas: Persona[];
  personasStatus: LoadState;
  /** threadId -> cast ids for every character chat; drives sidebar portraits. */
  chatIndex: Record<string, string[]>;
  /** threadId -> the chat's scene; null = the thread is an ordinary chat. */
  scenes: Record<string, ChatScene | null>;
  refresh: () => Promise<void>;
  refreshPersonas: () => Promise<void>;
  refreshIndex: () => Promise<void>;
  upsert: (character: Character) => void;
  remove: (id: string) => void;
  upsertPersona: (persona: Persona) => void;
  removePersona: (id: string) => void;
  loadScene: (threadId: string, force?: boolean) => Promise<ChatScene | null>;
  setScene: (scene: ChatScene) => void;
  patchScene: (
    threadId: string,
    patch: {
      cast?: { characterId: string; muted?: boolean }[];
      personaId?: string | null;
      settings?: SceneSettings;
    },
  ) => Promise<ChatScene>;
}

const inflightScenes = new Map<string, Promise<ChatScene | null>>();
const sceneWrites = new Map<string, Promise<unknown>>();
let refreshPromise: Promise<void> | null = null;
// Bumped on sign-out: an answer to a request made before it belongs to the previous account.
let epoch = 0;

function sortIds(characters: Record<string, Character>): string[] {
  return Object.values(characters)
    .sort(
      (a, b) =>
        (b.lastChatAt ?? b.updatedAt) - (a.lastChatAt ?? a.updatedAt) ||
        a.name.localeCompare(b.name),
    )
    .map((character) => character.id);
}

export const useCharactersStore = create<CharactersState>((set, get) => ({
  characters: {},
  order: [],
  status: "idle",
  personas: [],
  personasStatus: "idle",
  chatIndex: {},
  scenes: {},
  refresh: () => {
    if (refreshPromise) return refreshPromise;
    if (get().status === "idle") set({ status: "loading" });
    const started = epoch;
    refreshPromise = listCharacters()
      .then((list) => {
        if (started !== epoch) return;
        const characters = Object.fromEntries(list.map((c) => [c.id, c]));
        set({ characters, order: sortIds(characters), status: "ready" });
      })
      .catch((error) => {
        set({ status: get().status === "ready" ? "ready" : "error" });
        throw error;
      })
      .finally(() => {
        refreshPromise = null;
      });
    return refreshPromise;
  },
  refreshPersonas: async () => {
    if (get().personasStatus === "idle") set({ personasStatus: "loading" });
    const started = epoch;
    try {
      const personas = await listPersonas();
      if (started === epoch) set({ personas, personasStatus: "ready" });
    } catch (error) {
      set({ personasStatus: get().personasStatus === "ready" ? "ready" : "error" });
      throw error;
    }
  },
  refreshIndex: async () => {
    const started = epoch;
    const chatIndex = await getChatIndex();
    if (started === epoch) set({ chatIndex });
  },
  upsert: (character) =>
    set((state) => {
      const characters = { ...state.characters, [character.id]: character };
      return { characters, order: sortIds(characters) };
    }),
  remove: (id) =>
    set((state) => {
      const characters = { ...state.characters };
      delete characters[id];
      const chatIndex = Object.fromEntries(
        Object.entries(state.chatIndex).map(([thread, ids]) => [thread, ids.filter((c) => c !== id)]),
      );
      const scenes = Object.fromEntries(
        Object.entries(state.scenes).map(([thread, scene]) => [
          thread,
          scene ? { ...scene, cast: scene.cast.filter((m) => m.characterId !== id) } : scene,
        ]),
      );
      return { characters, order: sortIds(characters), chatIndex, scenes };
    }),
  upsertPersona: (persona) =>
    set((state) => {
      const others = state.personas.filter((p) => p.id !== persona.id);
      const personas = persona.isDefault
        ? [persona, ...others.map((p) => ({ ...p, isDefault: false }))]
        : [...others, persona];
      return { personas: personas.sort((a, b) => Number(b.isDefault) - Number(a.isDefault) || a.name.localeCompare(b.name)) };
    }),
  removePersona: (id) => set((state) => ({ personas: state.personas.filter((p) => p.id !== id) })),
  loadScene: (threadId, force = false) => {
    if (!force && threadId in get().scenes) return Promise.resolve(get().scenes[threadId] ?? null);
    const pending = inflightScenes.get(threadId);
    if (pending && !force) return pending;
    const started = epoch;
    const request = getChatScene(threadId)
      .then((scene) => {
        if (started === epoch) set((state) => ({ scenes: { ...state.scenes, [threadId]: scene } }));
        return scene;
      })
      .finally(() => inflightScenes.delete(threadId));
    inflightScenes.set(threadId, request);
    return request;
  },
  setScene: (scene) =>
    set((state) => ({
      scenes: { ...state.scenes, [scene.threadId]: scene },
      chatIndex: { ...state.chatIndex, [scene.threadId]: scene.cast.map((m) => m.characterId) },
    })),
  patchScene: (threadId, patch) => {
    // One write at a time per chat, each merged into the settings the previous one left, so two
    // quick edits (a note and the view) never undo each other.
    const previous = sceneWrites.get(threadId) ?? Promise.resolve();
    const write = previous
      .catch(() => undefined)
      .then(async () => {
        const settings = patch.settings
          ? { ...(get().scenes[threadId]?.settings ?? {}), ...patch.settings }
          : undefined;
        const scene = await updateChatScene(threadId, { ...patch, ...(settings ? { settings } : {}) });
        get().setScene(scene);
        return scene;
      });
    sceneWrites.set(threadId, write);
    void write.finally(() => {
      if (sceneWrites.get(threadId) === write) sceneWrites.delete(threadId);
    }).catch(() => undefined);
    return write;
  },
}));

if (typeof window !== "undefined") {
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    epoch += 1;
    refreshPromise = null;
    inflightScenes.clear();
    sceneWrites.clear();
    useCharactersStore.setState({
      characters: {},
      order: [],
      status: "idle",
      personas: [],
      personasStatus: "idle",
      chatIndex: {},
      scenes: {},
    });
  });
}
