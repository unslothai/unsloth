import { create } from "zustand";
import type { NativeIntent } from "./types";

interface NativeIntentState {
  pendingModelIntent: NativeIntent | null;
  // Dropped documents wait here until the thread bar (which owns the RAG upload and
  // the lazy thread id) drains them.
  pendingAttachments: NativeIntent[];
  addIntent: (intent: NativeIntent) => void;
  addAttachments: (intents: NativeIntent[]) => void;
  takeAttachments: () => NativeIntent[];
  clearModelIntent: (intentId?: string) => void;
}

export const useNativeIntentStore = create<NativeIntentState>((set, get) => ({
  pendingModelIntent: null,
  pendingAttachments: [],
  addAttachments: (intents) => {
    const fresh = intents.filter((intent) => intent.kind === "attachment");
    if (fresh.length === 0) return;
    set({ pendingAttachments: [...get().pendingAttachments, ...fresh] });
  },
  takeAttachments: () => {
    const queued = get().pendingAttachments;
    if (queued.length > 0) set({ pendingAttachments: [] });
    return queued;
  },
  addIntent: (intent) => {
    if (intent.kind !== "model") return;
    const current = get().pendingModelIntent;
    if (current?.path.token === intent.path.token) return;
    set({ pendingModelIntent: intent });
  },
  clearModelIntent: (intentId) => {
    const current = get().pendingModelIntent;
    if (intentId && current?.id !== intentId) return;
    set({ pendingModelIntent: null });
  },
}));
