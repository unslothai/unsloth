


import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

/**
 * localStorage that cannot throw: private browsing, blocked cookies and opaque webview
 * origins all throw on access. Losing the preference is fine; losing the panel is not.
 */
const safeStorage = {
  getItem: (name: string): string | null => {
    try {
      return window.localStorage.getItem(name);
    } catch {
      return null;
    }
  },
  setItem: (name: string, value: string): void => {
    try {
      window.localStorage.setItem(name, value);
    } catch {
      // Quota exceeded or denied; the preference stays session-only.
    }
  },
  removeItem: (name: string): void => {
    try {
      window.localStorage.removeItem(name);
    } catch {
      // Same.
    }
  },
};

interface ApiMonitorOverlayState {
  /** Whether the floating panel is on screen right now. Session state. */
  isOpen: boolean;
  /** Set on close so the panel does not pop back in the same burst. */
  suppressed: boolean;
  /** Persisted opt out: when false the panel never opens itself. */
  autoOpen: boolean;
  open: () => void;
  close: () => void;
  setAutoOpen: (autoOpen: boolean) => void;
}

/** Only `autoOpen` persists; a dismissal lasts the sitting, not forever. */
export const useApiMonitorOverlayStore = create<ApiMonitorOverlayState>()(
  persist(
    (set) => ({
      isOpen: false,
      suppressed: false,
      autoOpen: true,
      open: () => set({ isOpen: true, suppressed: false }),
      close: () => set({ isOpen: false, suppressed: true }),
      setAutoOpen: (autoOpen) => set({ autoOpen }),
    }),
    {
      name: "unsloth_api_monitor_overlay",
      version: 1,
      storage: createJSONStorage(() => safeStorage),
      partialize: (state) => ({ autoOpen: state.autoOpen }),
      // Without this a version bump discards the payload and revives a disabled popup.
      migrate: (persisted) => persisted,
      // Explicit merge so an older stored payload cannot resurrect `isOpen`.
      merge: (persisted, current) => ({
        ...current,
        autoOpen:
          typeof (persisted as { autoOpen?: unknown } | null)?.autoOpen ===
          "boolean"
            ? (persisted as { autoOpen: boolean }).autoOpen
            : current.autoOpen,
      }),
    },
  ),
);
