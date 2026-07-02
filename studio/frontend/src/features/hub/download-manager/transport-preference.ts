// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";
import { toast } from "@/lib/toast";
import {
  type DownloadTransportCapabilities,
  getDownloadTransportCapabilities,
} from "./api";
import {
  DEFAULT_TRANSPORT_MODE,
  type TransportMode,
  isTransportMode,
} from "./constants";
export type { TransportMode } from "./constants";

const STORAGE_KEY = "unsloth.studio.transportMode";
const CHANGE_EVENT = "unsloth:transport-preference-change";

type TransportCapabilitiesState = {
  capabilities: DownloadTransportCapabilities | null;
  isLoading: boolean;
};

function readStored(): TransportMode {
  if (typeof window === "undefined") {
    return DEFAULT_TRANSPORT_MODE;
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return isTransportMode(raw) ? raw : DEFAULT_TRANSPORT_MODE;
  } catch {
    return DEFAULT_TRANSPORT_MODE;
  }
}

export function getTransportMode(): TransportMode {
  return readStored();
}

export function useTransportMode(): [
  TransportMode,
  (next: TransportMode) => void,
] {
  const [mode, setMode] = useState<TransportMode>(readStored);

  useEffect(() => {
    const handleLocal = () => setMode(readStored());
    const handleStorage = (event: StorageEvent) => {
      if (event.storageArea !== window.localStorage) {
        return;
      }
      if (event.key !== null && event.key !== STORAGE_KEY) {
        return;
      }
      setMode(readStored());
    };
    window.addEventListener(CHANGE_EVENT, handleLocal);
    window.addEventListener("storage", handleStorage);
    return () => {
      window.removeEventListener(CHANGE_EVENT, handleLocal);
      window.removeEventListener("storage", handleStorage);
    };
  }, []);

  const set = useCallback((next: TransportMode) => {
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
  }, []);

  return [mode, set];
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
