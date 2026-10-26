// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  loadDownloadTransportSettings,
  subscribeDownloadTransportSettings,
  updateDownloadTransportSettings,
} from "@/features/settings";
import { toast } from "@/lib/toast";
import { useCallback, useEffect, useState } from "react";
import {
  type DownloadTransportCapabilities,
  getDownloadTransportCapabilities,
} from "./api";
import {
  type TransportMode,
  isTransportMode,
  pickTransportMode,
} from "./constants";
export type { TransportMode } from "./constants";

/** Outranks the install-wide setting, so "Reset all local preferences" must clear it. */
export const TRANSPORT_MODE_STORAGE_KEY = "unsloth.studio.transportMode";
const STORAGE_KEY = TRANSPORT_MODE_STORAGE_KEY;
const CHANGE_EVENT = "unsloth:transport-preference-change";

type TransportCapabilitiesState = {
  capabilities: DownloadTransportCapabilities | null;
  isLoading: boolean;
};

// Cached because the download path reads it synchronously.
let installMode: TransportMode | null = null;
let installModeInFlight: Promise<TransportMode | null> | null = null;
let installModeInFlightIsRefresh = false;

function readStored(): TransportMode | null {
  if (typeof window === "undefined") {
    return null;
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return isTransportMode(raw) ? raw : null;
  } catch {
    return null;
  }
}

function loadInstallMode(refresh: boolean): Promise<TransportMode | null> {
  return loadDownloadTransportSettings({ refresh }).then((settings) => {
    installMode = isTransportMode(settings.mode) ? settings.mode : null;
    return installMode;
  });
}

function hydrateInstallMode(refresh = false): Promise<TransportMode | null> {
  // A refresh must not share an ordinary in-flight hydration that returns the stale value.
  if (installModeInFlight && (!refresh || installModeInFlightIsRefresh)) {
    return installModeInFlight;
  }
  // Keep the superseded request as a fallback if the refresh fails.
  const superseded = refresh ? installModeInFlight : null;
  const pending = loadInstallMode(refresh)
    // Keep what we loaded: discarding it on a transient failure sent the next download to Auto.
    .catch(() => (installMode === null && superseded ? superseded : installMode))
    .finally(() => {
      if (installModeInFlight === pending) {
        installModeInFlight = null;
        installModeInFlightIsRefresh = false;
      }
    });
  installModeInFlight = pending;
  installModeInFlightIsRefresh = refresh;
  return pending;
}

export function getTransportMode(): TransportMode {
  return pickTransportMode(readStored(), installMode);
}

/** Re-read rather than cached, since another browser can change it mid-session. */
export async function resolveTransportMode(): Promise<TransportMode> {
  const stored = readStored();
  if (stored !== null) {
    return stored;
  }
  return pickTransportMode(null, await hydrateInstallMode(true));
}

export function useTransportMode(): [
  TransportMode,
  (next: TransportMode, opts?: { persist?: boolean }) => void,
] {
  const [mode, setMode] = useState<TransportMode>(getTransportMode);

  useEffect(() => {
    const handleLocal = () => setMode(getTransportMode());
    const handleStorage = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage) {
        return;
      }
      if (event.key !== null && event.key !== STORAGE_KEY) {
        return;
      }
      setMode(getTransportMode());
    };
    window.addEventListener(CHANGE_EVENT, handleLocal);
    window.addEventListener("storage", handleStorage);
    const unsubscribe = subscribeDownloadTransportSettings((settings) => {
      installMode = isTransportMode(settings.mode)
        ? settings.mode
        : installMode;
      setMode(getTransportMode());
    });
    // Refresh rather than read the cache, or the toggle shows a stale mode.
    void hydrateInstallMode(true).then(() => setMode(getTransportMode()));
    return () => {
      window.removeEventListener(CHANGE_EVENT, handleLocal);
      window.removeEventListener("storage", handleStorage);
      unsubscribe();
    };
  }, []);

  const set = useCallback((
    next: TransportMode,
    opts: { persist?: boolean } = {},
  ) => {
    // A forced fallback, not a choice: do not persist, or it outranks the install setting.
    if (opts.persist === false) {
      setMode(next);
      return;
    }
    // Persist first, reflect after: downloads re-read localStorage.
    let savedLocally = true;
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      savedLocally = false;
    }
    if (savedLocally) {
      setMode(next);
      window.dispatchEvent(new Event(CHANGE_EVENT));
    }
    void updateDownloadTransportSettings(next).catch((error) => {
      console.warn(
        "Couldn't save the download transport for this install.",
        error,
      );
      toast.error(
        savedLocally
          ? "Saved for this browser, but not for this install."
          : "Couldn't save the download transport preference.",
      );
    });
  }, []);

  return [mode, set];
}

/** False until capabilities land, so no card flashes a resume promise. */
export function useHttpPartialsResumable(): boolean {
  const { capabilities } = useDownloadTransportCapabilities();
  return capabilities?.partials_resumable === true;
}

export function useDownloadTransportCapabilities(): TransportCapabilitiesState {
  const [state, setState] = useState<TransportCapabilitiesState>({
    capabilities: null,
    isLoading: true,
  });

  useEffect(() => {
    let cancelled = false;
    getDownloadTransportCapabilities()
      .then((capabilities) => {
        if (cancelled) {
          return;
        }
        setState({ capabilities, isLoading: false });
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        setState({ capabilities: null, isLoading: false });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return state;
}
