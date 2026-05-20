// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useState } from "react";

export type TransportMode = "http" | "xet";

const STORAGE_KEY = "unsloth.studio.transportMode";
const DEFAULT_MODE: TransportMode = "http";
const CHANGE_EVENT = "unsloth:transport-preference-change";

function readStored(): TransportMode {
  if (typeof window === "undefined") return DEFAULT_MODE;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    return raw === "xet" || raw === "http" ? raw : DEFAULT_MODE;
  } catch {
    return DEFAULT_MODE;
  }
}

export function getTransportMode(): TransportMode {
  return readStored();
}

export function useTransportMode(): [TransportMode, (next: TransportMode) => void] {
  const [mode, setMode] = useState<TransportMode>(readStored);

  useEffect(() => {
    const sync = () => setMode(readStored());
    window.addEventListener(CHANGE_EVENT, sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener(CHANGE_EVENT, sync);
      window.removeEventListener("storage", sync);
    };
  }, []);

  const set = useCallback((next: TransportMode) => {
    setMode(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
      window.dispatchEvent(new Event(CHANGE_EVENT));
    } catch {
      return;
    }
  }, []);

  return [mode, set];
}
