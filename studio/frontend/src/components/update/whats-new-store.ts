import { create } from "zustand";

interface ReleaseOffer {
  version: string;
  url?: string | null;
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
