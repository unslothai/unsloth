// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The chat page's hold on the voice slot: it only ever unloads the voice it loaded
// itself. One load at a time; whichever voice is picked when it settles is the one
// that ends up loaded.

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

// Every owner's loads and unloads run one after another: a page that left can
// still be unloading the voice a new page is about to adopt by id.
let slotTail: Promise<unknown> = Promise.resolve();
function serial<T>(op: () => Promise<T>): Promise<T> {
  const next = slotTail.then(op, op);
  slotTail = next.catch(() => {});
  return next;
}

export function createVoiceSlotOwner(deps: VoiceSlotOwnerDeps): VoiceSlotOwner {
  let loaded: string | null = null;
  let pending: Promise<boolean> | null = null;
  let closed = false;
  // Voice turned off, or the page left, while the load was in flight: the off
  // path already ran and found nothing of ours to unload.
  const stale = () => closed || deps.voiceOff();

  const unloadOwn = () => {
    const id = loaded;
    loaded = null;
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
          loaded = id;
          deps.setLoaded(true);
          return true;
        }
        // With no chat model the voice would take the empty seat, and the next chat
        // load would replace it.
        if (!deps.hasChatModel()) return false;
        deps.setLoaded(false);
        deps.setLoading(true);
        const previous = loaded;
        if (previous && previous !== id) {
          loaded = null;
          await unloadVoice(previous).catch(() => {});
        }
        await loadVoice(id);
        if (stale()) {
          await unloadVoice(id).catch(() => {});
          return false;
        }
        loaded = id;
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
