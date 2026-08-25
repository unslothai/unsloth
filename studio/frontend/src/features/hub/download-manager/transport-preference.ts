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

const STORAGE_KEY = "unsloth.studio.transportMode";
const CHANGE_EVENT = "unsloth:transport-preference-change";

type TransportCapabilitiesState = {
  capabilities: DownloadTransportCapabilities | null;
  isLoading: boolean;
};

// The install's setting, so the choice follows the install rather than one browser. Cached
// because the download path reads the preference synchronously.
let installMode: TransportMode | null = null;
let installModeInFlight: Promise<TransportMode | null> | null = null;

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

function hydrateInstallMode(refresh = false): Promise<TransportMode | null> {
  installModeInFlight ??= loadDownloadTransportSettings({ refresh })
    .then((settings) => {
      installMode = isTransportMode(settings.mode) ? settings.mode : null;
      return installMode;
    })
    .catch(() => null)
    .finally(() => {
      installModeInFlight = null;
    });
  return installModeInFlight;
}

/** The preference as currently known, without waiting on the install setting. */
export function getTransportMode(): TransportMode {
  return pickTransportMode(readStored(), installMode);
}

/** The preference, waiting for the install setting when this browser has no choice of its own.
 *
 * Re-reads it rather than trusting the cache: this runs once per download start, and the whole
 * point of an install-level setting is that another browser can change it. A cached answer held
 * for the life of the tab would keep downloading on the old transport until a reload. */
export async function resolveTransportMode(): Promise<TransportMode> {
  const stored = readStored();
  if (stored !== null) {
    return stored;
  }
  return pickTransportMode(null, await hydrateInstallMode(true));
}

export function useTransportMode(): [
  TransportMode,
  (next: TransportMode, opts?: { persistInstall?: boolean }) => void,
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
    void hydrateInstallMode().then(() => setMode(getTransportMode()));
    return () => {
      window.removeEventListener(CHANGE_EVENT, handleLocal);
      window.removeEventListener("storage", handleStorage);
      unsubscribe();
    };
  }, []);

  const set = useCallback((
    next: TransportMode,
    opts: { persistInstall?: boolean } = {},
  ) => {
    // Persist first, reflect after: the engine reads getTransportMode() fresh
    // from localStorage at download time, so an optimistic setMode() before a
    // failed write (private mode / quota) would show the new transport while
    // downloads still used the old one. On failure leave everything untouched.
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      toast.error("Couldn't save the download transport preference.");
      return;
    }
    setMode(next);
    window.dispatchEvent(new Event(CHANGE_EVENT));
    // And the install's setting, so scripted callers and other browsers follow. A failure here
    // leaves this browser on its own choice, which already applies.
    //
    // Opt-out for a fallback the user did not ask for: the Hub toggle drops a stored "xet" to
    // "http" by itself when hf_xet is missing, and persisting that would replace everyone's
    // choice for good, with repairing hf_xet later not bringing it back.
    if (opts.persistInstall === false) {
      return;
    }
    void updateDownloadTransportSettings(next).catch((error) => {
      console.warn(
        "Couldn't save the download transport for this install.",
        error,
      );
    });
  }, []);

  return [mode, set];
}

/** Whether an interrupted HTTP transfer leaves resumable bytes on this backend.
 * False until the capabilities land, so a card never flashes a resume promise
 * it may have to take back. */
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
