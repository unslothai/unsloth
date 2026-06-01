// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { bumpInventoryVersion } from "./inventory-events";

const HF_TOKEN_KEY = "unsloth_hf_token";
const LEGACY_TRAINING_KEY = "unsloth_training_config_v1";
let storageSyncStarted = false;
let storageSyncListener: ((event: StorageEvent) => void) | null = null;

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function loadInitial(): string {
  if (!canUseStorage()) return "";
  try {
    const direct = window.localStorage.getItem(HF_TOKEN_KEY);
    if (direct !== null) return direct;
    const legacy = window.localStorage.getItem(LEGACY_TRAINING_KEY);
    if (legacy) {
      const parsed = JSON.parse(legacy) as {
        state?: Record<string, unknown>;
        [key: string]: unknown;
      };
      const fromTraining = parsed?.state?.hfToken;
      if (typeof fromTraining === "string" && fromTraining.length > 0) {
        window.localStorage.setItem(HF_TOKEN_KEY, fromTraining);
        if (parsed.state && "hfToken" in parsed.state) {
          delete parsed.state.hfToken;
          window.localStorage.setItem(
            LEGACY_TRAINING_KEY,
            JSON.stringify(parsed),
          );
        }
        return fromTraining;
      }
    }
  } catch {
    return "";
  }
  return "";
}

function persist(value: string): void {
  if (!canUseStorage()) return;
  try {
    window.localStorage.setItem(HF_TOKEN_KEY, value);
  } catch {
    void 0;
  }
}

function normalize(raw: string): string {
  return raw.replace(/^[\s"']+|[\s"']+$/g, "");
}

function stopStorageSync(): void {
  if (!canUseStorage() || storageSyncListener === null) return;
  window.removeEventListener("storage", storageSyncListener);
  storageSyncListener = null;
  storageSyncStarted = false;
}

if (import.meta.hot) {
  import.meta.hot.dispose(stopStorageSync);
}

interface HfTokenStore {
  token: string;
  setToken: (value: string) => void;
  clearToken: () => void;
}

export const useHfTokenStore = create<HfTokenStore>((set) => {
  const applyToken = (value: string, shouldPersist: boolean) => {
    const next = normalize(value);
    if (shouldPersist) persist(next);
    let changed = false;
    set((state) => {
      if (state.token === next) return state;
      changed = true;
      return { token: next };
    });
    if (changed) bumpInventoryVersion();
  };

  if (!storageSyncStarted && canUseStorage()) {
    storageSyncStarted = true;
    storageSyncListener = (event) => {
      if (event.key !== HF_TOKEN_KEY) return;
      applyToken(event.newValue ?? "", false);
    };
    window.addEventListener("storage", storageSyncListener);
  }

  return {
    token: loadInitial(),
    setToken: (value) => applyToken(value, true),
    clearToken: () => applyToken("", true),
  };
});

export function getHfToken(): string {
  return useHfTokenStore.getState().token;
}
