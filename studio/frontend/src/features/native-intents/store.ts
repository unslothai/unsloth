import { create } from "zustand";
import type { NativeIntent } from "./types";

interface NativeIntentState {
  pendingModelIntent: NativeIntent | null;
  addIntent: (intent: NativeIntent) => void;
  clearModelIntent: (intentId?: string) => void;
}

export const useNativeIntentStore = create<NativeIntentState>((set, get) => ({
  pendingModelIntent: null,
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
