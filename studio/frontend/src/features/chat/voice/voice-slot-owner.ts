// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


type VoiceSlotApi = {
  loadVoice(id: string): Promise<void>;
  unloadVoice(id: string): Promise<void>;
  voiceServing(id: string): Promise<boolean>;
};

export type VoiceSlotOwnerDeps = {
  api: () => Promise<VoiceSlotApi>;
  pickedVoice: () => string;
  hasChatModel: () => boolean;
  voiceOff: () => boolean;
  setLoaded: (loaded: boolean) => void;
  setLoading: (loading: boolean) => void;
  onError: (error: unknown) => void;
};

export type VoiceSlotOwner = {
  ensure(): Promise<boolean>;
  unloadOwn(): void;
  open(): void;
  close(): void;
  readonly loadedId: string | null;
  readonly pending: Promise<boolean> | null;
};

// Serialized across owners: a page that left can still be unloading the voice a new page adopts.
let slotTail: Promise<unknown> = Promise.resolve();
function serial<T>(op: () => Promise<T>): Promise<T> {
  const next = slotTail.then(op, op);
  slotTail = next.catch(() => {});
  return next;
}

export function createVoiceSlotOwner(deps: VoiceSlotOwnerDeps): VoiceSlotOwner {
  let loaded: string | null = null;
  let owned = false;
  let pending: Promise<boolean> | null = null;
  let closed = false;
  const stale = () => closed || deps.voiceOff();

  const unloadOwn = () => {
    const id = owned ? loaded : null;
    loaded = null;
    owned = false;
    deps.setLoaded(false);
    if (id) {
      void serial(() =>
        deps.api().then(({ unloadVoice }) => unloadVoice(id)),
      ).catch(() => {});
    }
  };

  const ensure = (): Promise<boolean> => {
    if (pending) return pending;
    const run = serial(async () => {
      const id = deps.pickedVoice();
      try {
        const { loadVoice, unloadVoice, voiceServing } = await deps.api();
        if (await voiceServing(id)) {
          if (stale()) return false;
          if (owned && loaded && loaded !== id) {
            const previous = loaded;
            loaded = null;
            owned = false;
            await unloadVoice(previous).catch(() => {});
          }
          owned = owned && loaded === id;
          loaded = id;
          deps.setLoaded(true);
          return true;
        }
        if (!deps.hasChatModel()) return false;
        deps.setLoaded(false);
        deps.setLoading(true);
        const previous = loaded;
        if (previous && previous !== id) {
          const mine = owned;
          loaded = null;
          owned = false;
          if (mine) await unloadVoice(previous).catch(() => {});
        }
        await loadVoice(id);
        if (stale()) {
          await unloadVoice(id).catch(() => {});
          return false;
        }
        loaded = id;
        owned = true;
        deps.setLoaded(deps.pickedVoice() === id);
        return true;
      } catch (error) {
        deps.onError(error);
        return false;
      } finally {
        deps.setLoading(false);
        pending = null;
      }
    });
    pending = run;
    return run;
  };

  return {
    ensure,
    unloadOwn,
    open() {
      closed = false;
    },
    close() {
      closed = true;
      unloadOwn();
    },
    get loadedId() {
      return loaded;
    },
    get pending() {
      return pending;
    },
  };
}
