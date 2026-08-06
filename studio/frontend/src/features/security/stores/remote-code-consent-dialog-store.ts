


import { create } from "zustand";
import type { RemoteCodeScan } from "../types";

type Resolver = (confirmed: boolean) => void;

// One in-flight consent at a time; a new request resolves any prior pending one as
// declined so its promise never leaks.
let pendingResolver: Resolver | null = null;

interface RemoteCodeConsentDialogStore {
  open: boolean;
  scan: RemoteCodeScan | null;
  requestConsent: (scan: RemoteCodeScan) => Promise<boolean>;
  resolve: (confirmed: boolean) => void;
}

export const useRemoteCodeConsentDialogStore = create<RemoteCodeConsentDialogStore>()(
  (set) => ({
    open: false,
    scan: null,
    requestConsent: (scan) =>
      new Promise<boolean>((resolve) => {
        pendingResolver?.(false);
        pendingResolver = resolve;
        set({ open: true, scan });
      }),
    resolve: (confirmed) => {
      const resolver = pendingResolver;
      pendingResolver = null;
      set({ open: false, scan: null });
      resolver?.(confirmed);
    },
  }),
);
