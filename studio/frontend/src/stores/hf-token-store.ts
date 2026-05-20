// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

const HF_TOKEN_KEY = "unsloth_hf_token";
const LEGACY_TRAINING_KEY = "unsloth_training_config_v1";

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
      const parsed = JSON.parse(legacy) as { state?: { hfToken?: unknown } };
      const fromTraining = parsed?.state?.hfToken;
      if (typeof fromTraining === "string" && fromTraining.length > 0) {
        window.localStorage.setItem(HF_TOKEN_KEY, fromTraining);
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
  } catch {}
}

function normalize(raw: string): string {
  return raw.trim().replace(/^["']+|["']+$/g, "");
}

interface HfTokenStore {
  token: string;
  setToken: (value: string) => void;
  clearToken: () => void;
}

export const useHfTokenStore = create<HfTokenStore>((set) => ({
  token: loadInitial(),
  setToken: (value) =>
    set(() => {
      const next = normalize(value);
      persist(next);
      return { token: next };
    }),
  clearToken: () =>
    set(() => {
      persist("");
      return { token: "" };
    }),
}));

export function getHfToken(): string {
  return useHfTokenStore.getState().token;
}
