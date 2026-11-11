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
  type ScenePatch,
} from "./api";
import { mergeCharacters } from "./meta";
import type { Character, ChatScene, Persona } from "./types";

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
  patchScene: (threadId: string, patch: ScenePatch) => Promise<ChatScene>;
}

const inflightScenes = new Map<string, Promise<ChatScene | null>>();
const sceneWrites = new Map<string, Promise<unknown>>();
// Bumped by every saved scene write, so a read that started before one never replaces its result.
const sceneWriteGen = new Map<string, number>();
let refreshPromise: Promise<void> | null = null;
let personasPromise: Promise<void> | null = null;
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
        const characters = mergeCharacters(get().characters, list);
        if (characters === get().characters) set({ status: "ready" });
        else set({ characters, order: sortIds(characters), status: "ready" });
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
  refreshPersonas: () => {
    // Shared while in flight, as refresh() does, so every component opening at once makes one request.
    if (personasPromise) return personasPromise;
    if (get().personasStatus === "idle") set({ personasStatus: "loading" });
    const started = epoch;
    personasPromise = listPersonas()
      .then((personas) => {
        if (started !== epoch) return;
        // Kept as is when nothing changed, so a re-read before each send re-renders nothing.
        const same = JSON.stringify(personas) === JSON.stringify(get().personas);
        set(same ? { personasStatus: "ready" } : { personas, personasStatus: "ready" });
      })
      .catch((error) => {
        set({ personasStatus: get().personasStatus === "ready" ? "ready" : "error" });
        throw error;
      })
      .finally(() => {
        personasPromise = null;
      });
    return personasPromise;
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
    // A queued write answers with the newest scene, so a re-read waits for it instead.
    const write = sceneWrites.get(threadId);
    if (force && write) {
      const current = () => get().scenes[threadId] ?? null;
      return write.then(current, current);
    }
    const pending = inflightScenes.get(threadId);
    if (pending && !force) return pending;
    const started = epoch;
    const gen = sceneWriteGen.get(threadId) ?? 0;
    const request = getChatScene(threadId)
      .then((scene) => {
        if (started !== epoch) return scene;
        // A write queued or saved during this read answers with a newer scene.
        if (sceneWrites.has(threadId) || (sceneWriteGen.get(threadId) ?? 0) !== gen) {
          return get().scenes[threadId] ?? scene;
        }
        set((state) => ({ scenes: { ...state.scenes, [threadId]: scene } }));
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
    // One write at a time per chat, so the scene kept here answers the latest edit.
    const previous = sceneWrites.get(threadId) ?? Promise.resolve();
    const write = previous
      .catch(() => undefined)
      .then(async () => {
        const scene = await updateChatScene(threadId, patch);
        sceneWriteGen.set(threadId, (sceneWriteGen.get(threadId) ?? 0) + 1);
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
  // Coming back to this tab picks up characters and personas edited in another one before the next send.
  document.addEventListener("visibilitychange", () => {
    const state = useCharactersStore.getState();
    if (document.visibilityState !== "visible") return;
    if (state.status === "ready") void state.refresh().catch(() => undefined);
    if (state.personasStatus === "ready") void state.refreshPersonas().catch(() => undefined);
  });
  window.addEventListener(AUTH_SESSION_CLEARED_EVENT, () => {
    epoch += 1;
    refreshPromise = null;
    personasPromise = null;
    inflightScenes.clear();
    sceneWrites.clear();
    sceneWriteGen.clear();
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
