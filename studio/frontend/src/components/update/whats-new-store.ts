import { create } from "zustand";

interface ReleaseOffer {
  version: string;
  url?: string | null;
  // The version the user came from, which opens the catch-up view.
  since?: string;
}

export const useWhatsNew = create<{
  open: boolean;
  offer?: ReleaseOffer;
  show: (offer?: ReleaseOffer) => void;
  close: () => void;
}>((set) => ({
  open: false,
  show: (offer) => set({ open: true, offer }),
  close: () => set({ open: false, offer: undefined }),
}));
