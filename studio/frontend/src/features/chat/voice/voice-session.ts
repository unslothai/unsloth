// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { isChatModelVoice, VOICE_DEFAULT_TTS_MODEL } from "./voice-models.ts";
import {
  holdForServing,
  releaseBeforeBorrow,
  settleVoiceLoad,
  type VoiceHold,
  voiceToUnload,
} from "./voice-ownership.ts";

export type VoiceMode = "off" | "configuring" | "active";

// A release's unload that the server never answers holds a new load back at most this long.
const UNLOAD_WAIT_MS = 10_000;

export interface VoiceSlot {
  loadVoice(id: string): Promise<void>;
  servedVoice(): Promise<string | null>;
  unloadVoice(id: string): Promise<void>;
  voiceServing(id: string): Promise<boolean>;
}

// What the session reads and changes outside itself; chat-page wires these to the store, toasts and model clients.
export interface VoiceSessionPorts {
  state(): {
    voiceMode: VoiceMode;
    selectedVoiceModelId: string;
    checkpoint: string;
    modelLoading: boolean;
  };
  setSelectedVoiceModelId(id: string): void;
  setVoiceSlotLoaded(loaded: boolean): void;
  setVoiceSlotLoading(loading: boolean): void;
  slot(): Promise<VoiceSlot>;
  // The listening models: transcriber, speech detector and turn judge.
  releaseListening(): void;
  chatModelCannotSpeak(): void;
  loadFailed(error: unknown): void;
}

export interface VoiceSession {
  ensureVoiceSlotLoaded(): Promise<boolean>;
  // The pick changed, or the chat model it follows did: re-checks once a load in flight settles.
  follow(selectedVoiceModelId: string): void;
  setMode(mode: VoiceMode): void;
  activate(): void;
  dispose(): void;
}

export function createVoiceSession(ports: VoiceSessionPorts, initialMode: VoiceMode): VoiceSession {
  // Only a voice this session loaded is ever unloaded; whichever is picked when a load settles ends up loaded.
  let held: VoiceHold | null = null;
  // The load in flight, the pick it is for, and the epoch it started in.
  let loading: Promise<boolean> | null = null;
  let loadingFor = "";
  let loadingEpoch = 0;
  // The unload of a voice this session let go; a load waits for it, so it never sees that voice as still serving.
  let unloading: Promise<void> | null = null;
  // Bumped whenever voice mode lets go, so a load still in flight knows it is no longer wanted.
  let epoch = 0;
  let mode = initialMode;
  let leaving = false;

  const ensureVoiceSlotLoaded = (): Promise<boolean> => {
    if (ports.state().voiceMode !== "active") return Promise.resolve(false);
    if (loading) {
      // Joined only when it loads what is picked now; otherwise asked again once it settles.
      if (loadingFor === ports.state().selectedVoiceModelId && loadingEpoch === epoch) return loading;
      return loading.then(ensureVoiceSlotLoaded);
    }
    const run = (async () => {
      const start = ports.state();
      let id = start.selectedVoiceModelId;
      const startEpoch = epoch;
      try {
        if (unloading) {
          let timer: ReturnType<typeof setTimeout> | undefined;
          await Promise.race([unloading, new Promise<void>((resolve) => (timer = setTimeout(resolve, UNLOAD_WAIT_MS)))]);
          clearTimeout(timer);
          if (startEpoch !== epoch) return false;
        }
        const { loadVoice, servedVoice, unloadVoice, voiceServing } = await ports.slot();
        if (isChatModelVoice(id) && !(await voiceServing(id))) {
          // A chat model that cannot speak would park every turn behind a voice that never loads.
          if (!start.checkpoint || ports.state().modelLoading) return false;
          if (startEpoch !== epoch) return false;
          id = VOICE_DEFAULT_TTS_MODEL;
          loadingFor = id;
          ports.setSelectedVoiceModelId(id);
          ports.chatModelCannotSpeak();
        }
        if (await voiceServing(id)) {
          if (startEpoch !== epoch) return false;
          const release = releaseBeforeBorrow(held, id);
          if (release) {
            held = null;
            await unloadVoice(release).catch(() => {});
            if (startEpoch !== epoch) return false;
          }
          held = holdForServing(held, id);
          ports.setVoiceSlotLoaded(ports.state().selectedVoiceModelId === id);
          return true;
        }
        // A voice already serving becomes the pick instead of a second voice loading beside it;
        // never for the chat model's own voice, which is the user's opt-in.
        const served = held === null && !isChatModelVoice(id) ? await servedVoice() : null;
        if (served) {
          if (startEpoch !== epoch) return false;
          held = holdForServing(null, served);
          loadingFor = served;
          ports.setSelectedVoiceModelId(served);
          ports.setVoiceSlotLoaded(true);
          return true;
        }
        // With no chat model the voice would take the empty seat and the next chat load would replace it.
        if (!start.checkpoint) return false;
        ports.setVoiceSlotLoaded(false);
        ports.setVoiceSlotLoading(true);
        const previous = held;
        if (previous && previous.id !== id) {
          held = null;
          const own = voiceToUnload(previous);
          if (own) await unloadVoice(own).catch(() => {});
          if (startEpoch !== epoch) return false;
        }
        await loadVoice(id);
        const settled = settleVoiceLoad(id, startEpoch, epoch);
        if (settled.unload) {
          await unloadVoice(settled.unload).catch(() => {});
          return false;
        }
        held = settled.hold;
        ports.setVoiceSlotLoaded(ports.state().selectedVoiceModelId === id);
        return true;
      } catch (error) {
        ports.loadFailed(error);
        return false;
      } finally {
        ports.setVoiceSlotLoading(false);
        loading = null;
      }
    })();
    loading = run;
    loadingFor = ports.state().selectedVoiceModelId;
    loadingEpoch = epoch;
    return run;
  };

  const releaseOwnVoice = () => {
    const id = voiceToUnload(held);
    held = null;
    epoch += 1;
    ports.setVoiceSlotLoaded(false);
    if (!id) return;
    const done: Promise<void> = ports
      .slot()
      .then(({ unloadVoice }) => unloadVoice(id))
      .catch(() => {})
      .finally(() => {
        if (unloading === done) unloading = null;
      });
    unloading = done;
  };

  return {
    ensureVoiceSlotLoaded,
    follow(selectedVoiceModelId) {
      if (held?.id !== selectedVoiceModelId) ports.setVoiceSlotLoaded(false);
      const at = epoch;
      void (loading ?? Promise.resolve(true)).then(() => (at === epoch ? ensureVoiceSlotLoaded() : false));
    },
    setMode(next) {
      const was = mode;
      mode = next;
      if (next !== "off" || was === "off") return;
      releaseOwnVoice();
      ports.releaseListening();
    },
    activate() {
      leaving = false;
    },
    // Leaving Chat: the voice this session loaded always goes; the listening models only if voice mode was on.
    // Settled a microtask later, so StrictMode's dev remount (cleanup, then mount again at once) keeps them.
    dispose() {
      leaving = true;
      queueMicrotask(() => {
        if (!leaving) return;
        leaving = false;
        releaseOwnVoice();
        if (mode !== "off") ports.releaseListening();
      });
    },
  };
}
