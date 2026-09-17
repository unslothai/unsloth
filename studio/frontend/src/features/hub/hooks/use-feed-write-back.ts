// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useRef } from "react";
import { useHfEndpoint } from "@/lib/hf-endpoint";
import type { ChannelId } from "../lib/channels";
import { feedIdentity, useHubFeedStore } from "../stores/hub-feed-store";
import type { HfModelResult } from "./use-hub-model-search";

export function useFeedWriteBack(opts: {
  channelId: ChannelId | null;
  results: HfModelResult[];
  isLoading: boolean;
  accessToken: string | undefined;
}): void {
  const { channelId, results, isLoading, accessToken } = opts;
  const setChannelEntry = useHubFeedStore((s) => s.setChannelEntry);
  const hfEndpoint = useHfEndpoint();
  const tokenFingerprint = useMemo(
    () => feedIdentity(hfEndpoint, accessToken),
    [hfEndpoint, accessToken],
  );
  const writtenKeyRef = useRef<string | null>(null);

  useEffect(() => {
    if (!channelId || isLoading || results.length === 0) return;
    const key = `${channelId}:${tokenFingerprint}`;
    if (writtenKeyRef.current === key) return;
    writtenKeyRef.current = key;
    setChannelEntry(channelId, results, tokenFingerprint);
  }, [channelId, isLoading, results, tokenFingerprint, setChannelEntry]);
}
