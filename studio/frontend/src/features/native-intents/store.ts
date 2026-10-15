import { create } from "zustand";
import {
  type PendingNativeAttachments,
  dequeueNativeAttachments,
  enqueueNativeAttachments,
} from "./attachment-queue";
import type { NativeIntent } from "./types";

function moveQueuedAttachments(
  queues: PendingNativeAttachments,
  staleKeys: string[],
  targetKey: string,
): PendingNativeAttachments {
  let pending = queues;
  for (const key of staleKeys) {
    const queued = queues[key] ?? [];
    if (queued.length > 0) {
      pending = enqueueNativeAttachments(pending, targetKey, queued);
    }
    if (key in pending) {
      pending = { ...pending };
      delete pending[key];
    }
  }
  return pending;
}

interface NativeIntentState {
  pendingModelIntent: NativeIntent | null;
  // Keyed by chat: registration is async, so the active chat may change before these arrive.
  pendingAttachments: PendingNativeAttachments;
  pendingImageAttachments: PendingNativeAttachments;
  pendingOpenDocumentAttachments: PendingNativeAttachments;
  pendingAudioAttachments: PendingNativeAttachments;
  pendingVideoAttachments: PendingNativeAttachments;
  // Covers the registration window so a fast submit waits for the attachment.
  registeringImageDrops: number;
  registeringAudioDrops: number;
  registeringVideoDrops: number;
  // Per chat, so a failure elsewhere cannot cancel this composer's parked send.
  imageDropFailures: Record<string, number>;
  audioDropFailures: Record<string, number>;
  videoDropFailures: Record<string, number>;
  // A remounted composer cannot hand its batches over itself, so it leaves a note.
  imageDropOwners: Record<string, string>;
  audioDropOwners: Record<string, string>;
  videoDropOwners: Record<string, string>;
  addIntent: (intent: NativeIntent) => void;
  addAttachments: (targetKey: string, intents: NativeIntent[]) => void;
  addImageAttachments: (targetKey: string, intents: NativeIntent[]) => void;
  addOpenDocumentAttachments: (targetKey: string, intents: NativeIntent[]) => void;
  addAudioAttachments: (targetKey: string, intents: NativeIntent[]) => void;
  addVideoAttachments: (targetKey: string, intents: NativeIntent[]) => void;
  takeAttachments: (targetKey: string) => NativeIntent[];
  takeImageAttachments: (targetKey: string) => NativeIntent[];
  takeOpenDocumentAttachments: (targetKey: string) => NativeIntent[];
  takeAudioAttachments: (targetKey: string) => NativeIntent[];
  takeVideoAttachments: (targetKey: string) => NativeIntent[];
  beginImageDropRegistration: () => void;
  endImageDropRegistration: () => void;
  beginAudioDropRegistration: () => void;
  endAudioDropRegistration: () => void;
  beginVideoDropRegistration: () => void;
  endVideoDropRegistration: () => void;
  failImageDropRegistration: (targetKey: string) => void;
  failAudioDropRegistration: (targetKey: string) => void;
  failVideoDropRegistration: (targetKey: string) => void;
  noteImageDropOwner: (targetKey: string, identity: string) => void;
  claimImageAttachments: (identity: string, targetKey: string) => void;
  noteAudioDropOwner: (targetKey: string, identity: string) => void;
  claimAudioAttachments: (identity: string, targetKey: string) => void;
  noteVideoDropOwner: (targetKey: string, identity: string) => void;
  claimVideoAttachments: (identity: string, targetKey: string) => void;
  clearModelIntent: (intentId?: string) => void;
}

export const useNativeIntentStore = create<NativeIntentState>((set, get) => ({
  pendingModelIntent: null,
  pendingAttachments: {},
  pendingImageAttachments: {},
  pendingOpenDocumentAttachments: {},
  pendingAudioAttachments: {},
  pendingVideoAttachments: {},
  registeringImageDrops: 0,
  registeringAudioDrops: 0,
  registeringVideoDrops: 0,
  imageDropFailures: {},
  audioDropFailures: {},
  videoDropFailures: {},
  imageDropOwners: {},
  audioDropOwners: {},
  videoDropOwners: {},
  addAttachments: (targetKey, intents) => {
    const current = get().pendingAttachments;
    const pendingAttachments = enqueueNativeAttachments(
      current,
      targetKey,
      intents,
    );
    if (pendingAttachments !== current) {
      set({ pendingAttachments });
    }
  },
  addImageAttachments: (targetKey, intents) => {
    const current = get().pendingImageAttachments;
    const pendingImageAttachments = enqueueNativeAttachments(
      current,
      targetKey,
      intents,
    );
    if (pendingImageAttachments !== current) {
      set({ pendingImageAttachments });
    }
  },
  addOpenDocumentAttachments: (targetKey, intents) => {
    const current = get().pendingOpenDocumentAttachments;
    const pendingOpenDocumentAttachments = enqueueNativeAttachments(
      current,
      targetKey,
      intents,
    );
    if (pendingOpenDocumentAttachments !== current) {
      set({ pendingOpenDocumentAttachments });
    }
  },
  addAudioAttachments: (targetKey, intents) => {
    const current = get().pendingAudioAttachments;
    const pendingAudioAttachments = enqueueNativeAttachments(
      current,
      targetKey,
      intents,
    );
    if (pendingAudioAttachments !== current) {
      set({ pendingAudioAttachments });
    }
  },
  takeAudioAttachments: (targetKey) => {
    const current = get().pendingAudioAttachments;
    const [queued, pendingAudioAttachments] = dequeueNativeAttachments(
      current,
      targetKey,
    );
    if (pendingAudioAttachments !== current) {
      set({ pendingAudioAttachments });
    }
    return queued;
  },
  takeAttachments: (targetKey) => {
    const current = get().pendingAttachments;
    const [queued, pendingAttachments] = dequeueNativeAttachments(
      current,
      targetKey,
    );
    if (pendingAttachments !== current) {
      set({ pendingAttachments });
    }
    return queued;
  },
  takeImageAttachments: (targetKey) => {
    const current = get().pendingImageAttachments;
    const [queued, pendingImageAttachments] = dequeueNativeAttachments(
      current,
      targetKey,
    );
    if (pendingImageAttachments !== current) {
      set({ pendingImageAttachments });
    }
    return queued;
  },
  takeOpenDocumentAttachments: (targetKey) => {
    const current = get().pendingOpenDocumentAttachments;
    const [queued, pendingOpenDocumentAttachments] = dequeueNativeAttachments(
      current,
      targetKey,
    );
    if (pendingOpenDocumentAttachments !== current) {
      set({ pendingOpenDocumentAttachments });
    }
    return queued;
  },
  beginImageDropRegistration: () => {
    set({ registeringImageDrops: get().registeringImageDrops + 1 });
  },
  endImageDropRegistration: () => {
    set({ registeringImageDrops: Math.max(0, get().registeringImageDrops - 1) });
  },
  beginAudioDropRegistration: () => {
    set({ registeringAudioDrops: get().registeringAudioDrops + 1 });
  },
  endAudioDropRegistration: () => {
    set({ registeringAudioDrops: Math.max(0, get().registeringAudioDrops - 1) });
  },
  failImageDropRegistration: (targetKey) => {
    const current = get().imageDropFailures;
    set({
      imageDropFailures: {
        ...current,
        [targetKey]: (current[targetKey] ?? 0) + 1,
      },
    });
  },
  failAudioDropRegistration: (targetKey) => {
    const current = get().audioDropFailures;
    set({
      audioDropFailures: {
        ...current,
        [targetKey]: (current[targetKey] ?? 0) + 1,
      },
    });
  },
  noteImageDropOwner: (targetKey, identity) => {
    if (!identity) return;
    set({ imageDropOwners: { ...get().imageDropOwners, [targetKey]: identity } });
  },
  claimImageAttachments: (identity, targetKey) => {
    if (!identity) return;
    const owners = get().imageDropOwners;
    const stale = Object.keys(owners).filter(
      (key) => owners[key] === identity && key !== targetKey,
    );
    if (stale.length === 0) return;
    const pendingImageAttachments = moveQueuedAttachments(
      get().pendingImageAttachments,
      stale,
      targetKey,
    );
    const pendingOpenDocumentAttachments = moveQueuedAttachments(
      get().pendingOpenDocumentAttachments,
      stale,
      targetKey,
    );
    const nextOwners = { ...owners };
    for (const key of stale) delete nextOwners[key];
    set({
      pendingImageAttachments,
      pendingOpenDocumentAttachments,
      imageDropOwners: nextOwners,
    });
  },
  noteAudioDropOwner: (targetKey, identity) => {
    if (!identity) return;
    set({ audioDropOwners: { ...get().audioDropOwners, [targetKey]: identity } });
  },
  claimAudioAttachments: (identity, targetKey) => {
    if (!identity) return;
    const owners = get().audioDropOwners;
    const stale = Object.keys(owners).filter(
      (key) => owners[key] === identity && key !== targetKey,
    );
    if (stale.length === 0) return;
    const queues = get().pendingAudioAttachments;
    let pendingAudioAttachments = queues;
    for (const key of stale) {
      const queued = queues[key] ?? [];
      if (queued.length > 0) {
        pendingAudioAttachments = enqueueNativeAttachments(
          pendingAudioAttachments,
          targetKey,
          queued,
        );
      }
      if (key in pendingAudioAttachments) {
        pendingAudioAttachments = { ...pendingAudioAttachments };
        delete pendingAudioAttachments[key];
      }
    }
    const nextOwners = { ...owners };
    for (const key of stale) delete nextOwners[key];
    set({ pendingAudioAttachments, audioDropOwners: nextOwners });
  },
  addVideoAttachments: (targetKey, intents) => {
    const current = get().pendingVideoAttachments;
    const pendingVideoAttachments = enqueueNativeAttachments(
      current,
      targetKey,
      intents,
    );
    if (pendingVideoAttachments !== current) {
      set({ pendingVideoAttachments });
    }
  },
  takeVideoAttachments: (targetKey) => {
    const current = get().pendingVideoAttachments;
    const [queued, pendingVideoAttachments] = dequeueNativeAttachments(
      current,
      targetKey,
    );
    if (pendingVideoAttachments !== current) {
      set({ pendingVideoAttachments });
    }
    return queued;
  },
  beginVideoDropRegistration: () => {
    set({ registeringVideoDrops: get().registeringVideoDrops + 1 });
  },
  endVideoDropRegistration: () => {
    set({ registeringVideoDrops: Math.max(0, get().registeringVideoDrops - 1) });
  },
  failVideoDropRegistration: (targetKey) => {
    const current = get().videoDropFailures;
    set({
      videoDropFailures: {
        ...current,
        [targetKey]: (current[targetKey] ?? 0) + 1,
      },
    });
  },
  noteVideoDropOwner: (targetKey, identity) => {
    if (!identity) return;
    set({ videoDropOwners: { ...get().videoDropOwners, [targetKey]: identity } });
  },
  claimVideoAttachments: (identity, targetKey) => {
    if (!identity) return;
    const owners = get().videoDropOwners;
    const stale = Object.keys(owners).filter(
      (key) => owners[key] === identity && key !== targetKey,
    );
    if (stale.length === 0) return;
    const queues = get().pendingVideoAttachments;
    let pendingVideoAttachments = queues;
    for (const key of stale) {
      const queued = queues[key] ?? [];
      if (queued.length > 0) {
        pendingVideoAttachments = enqueueNativeAttachments(
          pendingVideoAttachments,
          targetKey,
          queued,
        );
      }
      if (key in pendingVideoAttachments) {
        pendingVideoAttachments = { ...pendingVideoAttachments };
        delete pendingVideoAttachments[key];
      }
    }
    const nextOwners = { ...owners };
    for (const key of stale) delete nextOwners[key];
    set({ pendingVideoAttachments, videoDropOwners: nextOwners });
  },
  addIntent: (intent) => {
    if (intent.kind !== "model") {
      return;
    }
    const current = get().pendingModelIntent;
    if (current?.path.token === intent.path.token) {
      return;
    }
    set({ pendingModelIntent: intent });
  },
  clearModelIntent: (intentId) => {
    const current = get().pendingModelIntent;
    if (intentId && current?.id !== intentId) {
      return;
    }
    set({ pendingModelIntent: null });
  },
}));
