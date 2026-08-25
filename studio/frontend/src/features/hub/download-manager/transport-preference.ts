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

/** This browser's own transport override. Exported because it outranks the install-wide
 * setting, so "Reset all local preferences" has to clear it or the browser keeps ignoring
 * transport changes made elsewhere on the same install. */
export const TRANSPORT_MODE_STORAGE_KEY = "unsloth.studio.transportMode";
const STORAGE_KEY = TRANSPORT_MODE_STORAGE_KEY;
const CHANGE_EVENT = "unsloth:transport-preference-change";

type TransportCapabilitiesState = {
  capabilities: DownloadTransportCapabilities | null;
  isLoading: boolean;
};

// The install's setting, so the choice follows the install rather than one browser. Cached
// because the download path reads the preference synchronously.
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
  return loadDownloadTransportSettings({ refresh })
    .then((settings) => {
      installMode = isTransportMode(settings.mode) ? settings.mode : null;
      return installMode;
    })
    // Keep what we already loaded. Discarding it on a transient failure sent the next
    // download to Auto even though the install's choice was still known here.
    .catch(() => installMode);
}

function hydrateInstallMode(refresh = false): Promise<TransportMode | null> {
  // A refresh must reach the API layer, which decides what may be shared with what. Riding on
  // an ordinary hydration already in flight would hand back the value it went to replace.
  if (installModeInFlight && (!refresh || installModeInFlightIsRefresh)) {
    return installModeInFlight;
  }
  const pending = loadInstallMode(refresh).finally(() => {
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
    void hydrateInstallMode().then(() => setMode(getTransportMode()));
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
    // A fallback this machine forced, not a choice: reflect it and write nothing. Storing it
    // locally pinned the browser to HTTP for good, because a local value outranks the
    // install-wide one, so repairing hf_xet later never brought the Xet choice back.
    if (opts.persist === false) {
      setMode(next);
      return;
    }
    // Persist first, reflect after: the engine reads getTransportMode() fresh
    // from localStorage at download time, so an optimistic setMode() before a
    // failed write (private mode / quota) would show the new transport while
    // downloads still used the old one.
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
    // And the install's setting, so scripted callers and other browsers follow. A browser with
    // storage blocked reaches here too: the install is the store then, and the subscription
    // updates installMode, so the reflect-after rule holds through the server instead.
    void updateDownloadTransportSettings(next).catch((error) => {
      console.warn(
        "Couldn't save the download transport for this install.",
        error,
      );
      // Said out loud either way. The row calls this setting install-wide, so a browser that
      // kept the choice locally while the install did not is a mismatch worth knowing about,
      // and a browser that kept it nowhere has silently ignored the click.
      toast.error(
        savedLocally
          ? "Saved for this browser, but not for this install."
          : "Couldn't save the download transport preference.",
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
