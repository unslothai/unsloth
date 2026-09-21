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

/** This browser's own override. Exported because it outranks the install-wide setting, so
 * "Reset all local preferences" has to clear it. */
export const TRANSPORT_MODE_STORAGE_KEY = "unsloth.studio.transportMode";
const STORAGE_KEY = TRANSPORT_MODE_STORAGE_KEY;
const CHANGE_EVENT = "unsloth:transport-preference-change";

type TransportCapabilitiesState = {
  capabilities: DownloadTransportCapabilities | null;
  isLoading: boolean;
};

// The install's setting, cached because the download path reads it synchronously.
let installMode: TransportMode | null = null;
let installModeInFlight: Promise<TransportMode | null> | null = null;
let installModeInFlightIsRefresh = false;

/** This browser's own choice, or null when it has never made one. */
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
  // Only the API layer decides what may be shared: riding on an ordinary hydration already in
  // flight would hand back the value this refresh went to replace.
  if (installModeInFlight && (!refresh || installModeInFlightIsRefresh)) {
    return installModeInFlight;
  }
  // Not shared, but not thrown away either: a download started before the first hydration
  // answers overtakes it with a refresh, and if that refresh fails there is still a request
  // coming with the install's real choice.
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

/** The preference as currently known, without waiting on the install setting. */
export function getTransportMode(): TransportMode {
  return pickTransportMode(readStored(), installMode);
}

/** The preference, waiting for the install setting when this browser has no choice of its own.
 * Re-read rather than cached, since another browser can change it mid-session. */
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
    // Another surface saved it: adopt it unless this browser has its own choice.
    const unsubscribe = subscribeDownloadTransportSettings((settings) => {
      installMode = isTransportMode(settings.mode)
        ? settings.mode
        : installMode;
      setMode(getTransportMode());
    });
    // Refreshed, like the download path: reading the cache here showed the old mode in the
    // toggle while the next download already ran on the one another browser had set.
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
    // A fallback this machine forced, not a choice: reflect it and write nothing. Stored, it
    // would outrank the install setting and survive hf_xet being repaired.
    if (opts.persist === false) {
      setMode(next);
      return;
    }
    // Persist first, reflect after: downloads re-read localStorage, so an optimistic setMode()
    // before a failed write would show a transport the downloads do not use.
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
    // And the install's setting, so other browsers follow. With storage blocked this is the
    // only store, and the subscription keeps the reflect-after rule holding through it.
    void updateDownloadTransportSettings(next).catch((error) => {
      console.warn(
        "Couldn't save the download transport for this install.",
        error,
      );
      // Said out loud either way: the row calls this setting install-wide, so a local-only
      // save is a mismatch and no save at all is a click that did nothing.
      toast.error(
        savedLocally
          ? "Saved for this browser, but not for this install."
          : "Couldn't save the download transport preference.",
      );
    });
  }, []);

  return [mode, set];
}

/** Whether an interrupted HTTP transfer leaves resumable bytes on this backend. False until
 * the capabilities land, so no card flashes a resume promise it may take back. */
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
