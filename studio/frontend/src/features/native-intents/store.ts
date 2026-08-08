import { create } from "zustand";
import {
  type PendingNativeAttachments,
  dequeueNativeAttachments,
  enqueueNativeAttachments,
} from "./attachment-queue";
import type { NativeIntent } from "./types";

interface NativeIntentState {
  pendingModelIntent: NativeIntent | null;
  // Key each batch to the chat that received the OS drop. Registration crosses an
  // async Rust boundary, so the active chat may change before these arrive.
  pendingAttachments: PendingNativeAttachments;
  pendingImageAttachments: PendingNativeAttachments;
  // Image drops registering with Rust, before they have a queue to sit in. Not
  // keyed: until the intents land there is no settled target, and the OS drop
  // went to the window, which has one composer to send from.
  registeringImageDrops: number;
  // Bumped, per chat, when a drop fails before it reaches a queue. The composer
  // watches its own key so a failure elsewhere cannot cancel its parked send.
  imageDropFailures: Record<string, number>;
  addIntent: (intent: NativeIntent) => void;
  addAttachments: (targetKey: string, intents: NativeIntent[]) => void;
  addImageAttachments: (targetKey: string, intents: NativeIntent[]) => void;
  takeAttachments: (targetKey: string) => NativeIntent[];
  takeImageAttachments: (targetKey: string) => NativeIntent[];
  beginImageDropRegistration: () => void;
  endImageDropRegistration: () => void;
  failImageDropRegistration: (targetKey: string) => void;
  clearModelIntent: (intentId?: string) => void;
}

export const useNativeIntentStore = create<NativeIntentState>((set, get) => ({
  pendingModelIntent: null,
  pendingAttachments: {},
  pendingImageAttachments: {},
  registeringImageDrops: 0,
  imageDropFailures: {},
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
  beginImageDropRegistration: () => {
    set({ registeringImageDrops: get().registeringImageDrops + 1 });
  },
  endImageDropRegistration: () => {
    set({ registeringImageDrops: Math.max(0, get().registeringImageDrops - 1) });
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
