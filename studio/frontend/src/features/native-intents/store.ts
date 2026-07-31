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
  addIntent: (intent: NativeIntent) => void;
  addAttachments: (targetKey: string, intents: NativeIntent[]) => void;
  takeAttachments: (targetKey: string) => NativeIntent[];
  clearModelIntent: (intentId?: string) => void;
}

export const useNativeIntentStore = create<NativeIntentState>((set, get) => ({
  pendingModelIntent: null,
  pendingAttachments: {},
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
