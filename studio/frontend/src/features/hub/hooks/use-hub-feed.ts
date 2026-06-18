// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { type ChannelId, findChannel } from "../lib/channels";
import { fingerprintToken } from "../lib/token-fingerprint";
import { isChannelEntryFresh, useHubFeedStore } from "../stores/hub-feed-store";
import {
  type HfModelResult,
  fetchChannelFirstPage,
} from "./use-hub-model-search";

export type FeedStatus = "idle" | "loading" | "refreshing" | "ready" | "error";

export interface HubFeedSection {
  channelId: ChannelId;
  results: HfModelResult[];
  status: FeedStatus;
  isLoading: boolean;
  isRefreshing: boolean;
  error: string | null;
}

export interface UseHubFeedResult {
  trending: HubFeedSection;
  refetch: (id: ChannelId) => void;
}

// Only the trending row is rendered in the feed now; "Fine-tune ready" moved
// into the format dropdown and loads its channel on demand (channel-list mode).
const CHANNEL_IDS: readonly ChannelId[] = ["unsloth-trending"];
const FEED_PAGE_SIZE = 20;
const MAX_RETRIES = 5;
const BACKOFF_BASE_MS = 800;
const BACKOFF_CAP_MS = 15_000;

function backoffDelay(attempt: number): number {
  const base = Math.min(BACKOFF_CAP_MS, BACKOFF_BASE_MS * 2 ** attempt);
  return Math.round(base * (0.5 + Math.random() * 0.5));
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === "AbortError";
}

function channelRecord<T>(value: T): Record<ChannelId, T> {
  return {
    "unsloth-trending": value,
    "unsloth-latest": value,
    "unsloth-safetensors": value,
  };
}

export function useHubFeed(opts: {
  accessToken: string | undefined;
  online: boolean;
  enabled: boolean;
  deviceType: string | null;
}): UseHubFeedResult {
  const { accessToken, online, enabled, deviceType } = opts;
  const tokenFingerprint = useMemo(
    () => fingerprintToken(accessToken),
    [accessToken],
  );

  const channels = useHubFeedStore((s) => s.channels);
  const setChannelEntry = useHubFeedStore((s) => s.setChannelEntry);
  const clearForToken = useHubFeedStore((s) => s.clearForToken);

  const [statuses, setStatuses] = useState<Record<ChannelId, FeedStatus>>(() =>
    channelRecord<FeedStatus>("idle"),
  );
  const [errors, setErrors] = useState<Record<ChannelId, string | null>>(() =>
    channelRecord<string | null>(null),
  );

  const abortRefs = useRef<Record<ChannelId, AbortController | null>>(
    channelRecord<AbortController | null>(null),
  );
  const timerRefs = useRef<
    Record<ChannelId, ReturnType<typeof setTimeout> | null>
  >(channelRecord<ReturnType<typeof setTimeout> | null>(null));
  const inFlightRefs = useRef<Record<ChannelId, boolean>>(
    channelRecord<boolean>(false),
  );
  const versionRefs = useRef<Record<ChannelId, number>>(
    channelRecord<number>(0),
  );
  const onlineRef = useRef(online);

  const setStatus = useCallback((id: ChannelId, status: FeedStatus) => {
    setStatuses((prev) =>
      prev[id] === status ? prev : { ...prev, [id]: status },
    );
  }, []);
  const setError = useCallback((id: ChannelId, error: string | null) => {
    setErrors((prev) => (prev[id] === error ? prev : { ...prev, [id]: error }));
  }, []);

  const runChannel = useCallback(
    (id: ChannelId) => {
      if (inFlightRefs.current[id]) return;
      const preset = findChannel(id);
      if (!preset) return;

      const hadCache =
        (useHubFeedStore.getState().channels[id]?.results.length ?? 0) > 0;
      const version = (versionRefs.current[id] += 1);
      inFlightRefs.current[id] = true;
      abortRefs.current[id]?.abort();
      const controller = new AbortController();
      abortRefs.current[id] = controller;

      const isStale = () =>
        controller.signal.aborted || versionRefs.current[id] !== version;

      const settle = (status: FeedStatus, message: string | null) => {
        inFlightRefs.current[id] = false;
        setStatus(id, status);
        if (status === "error") setError(id, message);
      };

      const scheduleRetry = (attempt: number, message: string | null) => {
        if (isStale()) {
          inFlightRefs.current[id] = false;
          return;
        }
        if (attempt >= MAX_RETRIES) {
          settle(hadCache ? "ready" : "error", message ?? "No results");
          return;
        }
        if (!onlineRef.current) {
          inFlightRefs.current[id] = false;
          return;
        }
        timerRefs.current[id] = setTimeout(() => {
          timerRefs.current[id] = null;
          void attemptFetch(attempt + 1);
        }, backoffDelay(attempt));
      };

      const attemptFetch = async (attempt: number): Promise<void> => {
        if (isStale()) {
          inFlightRefs.current[id] = false;
          return;
        }
        if (!onlineRef.current) {
          inFlightRefs.current[id] = false;
          return;
        }
        if (attempt === 0) {
          setStatus(id, hadCache ? "refreshing" : "loading");
          setError(id, null);
        }
        try {
          const { results } = await fetchChannelFirstPage({
            channel: {
              owner: preset.owner,
              tags: preset.tags,
              query: preset.query,
              idSuffix: preset.idSuffix,
            },
            sortBy: preset.sort,
            sortDirection: "desc",
            accessToken,
            signal: controller.signal,
            pageSize: FEED_PAGE_SIZE,
            deviceType,
            keepUnsupportedTags: true,
          });
          if (isStale()) return;
          if (results.length > 0) {
            setChannelEntry(id, results, tokenFingerprint);
            settle("ready", null);
            return;
          }
          settle("ready", null);
        } catch (err) {
          if (isStale() || isAbortError(err)) return;
          scheduleRetry(
            attempt,
            err instanceof Error ? err.message : "Failed to load",
          );
        }
      };

      queueMicrotask(() => void attemptFetch(0));
    },
    [
      accessToken,
      tokenFingerprint,
      deviceType,
      setChannelEntry,
      setStatus,
      setError,
    ],
  );

  useEffect(() => {
    onlineRef.current = online;
    const cleanup = () => {
      for (const id of CHANNEL_IDS) {
        abortRefs.current[id]?.abort();
        abortRefs.current[id] = null;
        const timer = timerRefs.current[id];
        if (timer) clearTimeout(timer);
        timerRefs.current[id] = null;
        inFlightRefs.current[id] = false;
        versionRefs.current[id] += 1;
      }
    };
    if (!enabled) return cleanup;
    if (useHubFeedStore.getState().tokenFingerprint !== tokenFingerprint) {
      clearForToken(tokenFingerprint);
    }
    if (!online) return cleanup;
    for (const id of CHANNEL_IDS) {
      const entry = useHubFeedStore.getState().channels[id];
      if (isChannelEntryFresh(entry, id, tokenFingerprint)) continue;
      runChannel(id);
    }
    return cleanup;
  }, [enabled, online, tokenFingerprint, clearForToken, runChannel]);

  const refetch = useCallback(
    (id: ChannelId) => {
      const timer = timerRefs.current[id];
      if (timer) {
        clearTimeout(timer);
        timerRefs.current[id] = null;
      }
      inFlightRefs.current[id] = false;
      runChannel(id);
    },
    [runChannel],
  );

  const buildSection = useCallback(
    (id: ChannelId): HubFeedSection => {
      const results = channels[id]?.results ?? [];
      const status = statuses[id];
      return {
        channelId: id,
        results,
        status,
        isLoading: status === "loading" && results.length === 0,
        isRefreshing: status === "refreshing",
        error: results.length === 0 ? errors[id] : null,
      };
    },
    [channels, statuses, errors],
  );

  return {
    trending: buildSection("unsloth-trending"),
    refetch,
  };
}
