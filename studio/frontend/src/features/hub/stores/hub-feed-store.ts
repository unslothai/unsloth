// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";
import type { HfModelResult } from "../hooks/use-hub-model-search";
import { type ChannelId, findChannel } from "../lib/channels";
import { createThrottledStorage, noopStorage } from "./persist-storage";

export interface ChannelFeedEntry {
  results: HfModelResult[];
  fetchedAt: number;
  tokenFingerprint: string;
}

export interface HubFeedState {
  channels: Partial<Record<ChannelId, ChannelFeedEntry>>;
  tokenFingerprint: string;
  setChannelEntry: (
    id: ChannelId,
    results: HfModelResult[],
    tokenFingerprint: string,
  ) => void;
  clearForToken: (tokenFingerprint: string) => void;
}

export const FEED_CACHE_CAP = 24;

export const FEED_TTL_MS: Record<ChannelId, number> = {
  "unsloth-trending": 30 * 60 * 1000,
  "unsloth-latest": 5 * 60 * 1000,
  "unsloth-safetensors": 30 * 60 * 1000,
};

const PERSIST_KEY = "unsloth.studio.hubFeed";
const PERSIST_VERSION = 1;
const PERSIST_THROTTLE_MS = 1_000;

export function isChannelEntryFresh(
  entry: ChannelFeedEntry | undefined,
  id: ChannelId,
  tokenFingerprint: string,
  now: number = Date.now(),
): boolean {
  if (!entry) return false;
  if (entry.tokenFingerprint !== tokenFingerprint) return false;
  if (entry.results.length === 0) return false;
  return now - entry.fetchedAt < FEED_TTL_MS[id];
}

export const selectChannelEntry =
  (id: ChannelId) =>
  (state: HubFeedState): ChannelFeedEntry | undefined =>
    state.channels[id];

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function sanitizeEntry(value: unknown): ChannelFeedEntry | null {
  if (!isRecord(value)) return null;
  if (!Array.isArray(value.results)) return null;
  if (
    typeof value.fetchedAt !== "number" ||
    !Number.isFinite(value.fetchedAt)
  ) {
    return null;
  }
  const tokenFingerprint =
    typeof value.tokenFingerprint === "string"
      ? value.tokenFingerprint
      : "anon";
  const results = value.results.filter(
    (item): item is HfModelResult =>
      isRecord(item) && typeof item.id === "string",
  );
  if (results.length === 0) return null;
  return {
    results: results.slice(0, FEED_CACHE_CAP),
    fetchedAt: value.fetchedAt,
    tokenFingerprint,
  };
}

function sanitizeFeedState(persisted: unknown): Partial<HubFeedState> {
  if (!isRecord(persisted) || !isRecord(persisted.channels)) {
    return { channels: {}, tokenFingerprint: "anon" };
  }
  const channels: Partial<Record<ChannelId, ChannelFeedEntry>> = {};
  for (const [key, value] of Object.entries(persisted.channels)) {
    if (!findChannel(key as ChannelId)) continue;
    const entry = sanitizeEntry(value);
    if (entry) channels[key as ChannelId] = entry;
  }
  const tokenFingerprint =
    typeof persisted.tokenFingerprint === "string"
      ? persisted.tokenFingerprint
      : "anon";
  return { channels, tokenFingerprint };
}

export const useHubFeedStore = create<HubFeedState>()(
  persist(
    (set) => ({
      channels: {},
      tokenFingerprint: "anon",
      setChannelEntry: (id, results, tokenFingerprint) =>
        set((state) => {
          const base =
            state.tokenFingerprint === tokenFingerprint ? state.channels : {};
          return {
            tokenFingerprint,
            channels: {
              ...base,
              [id]: {
                results: results.slice(0, FEED_CACHE_CAP),
                fetchedAt: Date.now(),
                tokenFingerprint,
              },
            },
          };
        }),
      clearForToken: (tokenFingerprint) =>
        set((state) =>
          state.tokenFingerprint === tokenFingerprint
            ? state
            : { tokenFingerprint, channels: {} },
        ),
    }),
    {
      name: PERSIST_KEY,
      version: PERSIST_VERSION,
      storage: createJSONStorage(() =>
        typeof window === "undefined"
          ? noopStorage
          : createThrottledStorage(window.localStorage, PERSIST_THROTTLE_MS),
      ),
      migrate: (persisted) => sanitizeFeedState(persisted),
      merge: (persisted, current) => ({
        ...current,
        ...sanitizeFeedState(persisted),
      }),
      partialize: (state) => ({
        channels: state.channels,
        tokenFingerprint: state.tokenFingerprint,
      }),
    },
  ),
);
