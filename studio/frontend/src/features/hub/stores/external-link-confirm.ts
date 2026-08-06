


import { create } from "zustand";

interface ExternalLinkConfirmStore {
  pendingUrl: string | null;
  request: (url: string) => void;
  dismiss: () => void;
}

export const useExternalLinkConfirm = create<ExternalLinkConfirmStore>((set) => ({
  pendingUrl: null,
  request: (url) => set({ pendingUrl: url }),
  dismiss: () => set({ pendingUrl: null }),
}));

export function confirmExternalLink(url: string): boolean {
  const trimmed = url?.trim() ?? "";
  if (!trimmed || trimmed.startsWith("#")) return false;
  const lower = trimmed.toLowerCase();
  if (
    !lower.includes("://") &&
    !lower.startsWith("//") &&
    !lower.startsWith("mailto:")
  ) {
    return false;
  }
  useExternalLinkConfirm.getState().request(trimmed);
  return true;
}
