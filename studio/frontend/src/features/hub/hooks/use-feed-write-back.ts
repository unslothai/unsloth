// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useMemo, useRef } from "react";
import type { ChannelId } from "../lib/channels";
import { fingerprintToken } from "../lib/token-fingerprint";
import { useHubFeedStore } from "../stores/hub-feed-store";
import type { HfModelResult } from "./use-hub-model-search";

export function useFeedWriteBack(opts: {
  channelId: ChannelId | null;
  results: HfModelResult[];
  isLoading: boolean;
  accessToken: string | undefined;
}): void {
  const { channelId, results, isLoading, accessToken } = opts;
  const setChannelEntry = useHubFeedStore((s) => s.setChannelEntry);
  const tokenFingerprint = useMemo(
    () => fingerprintToken(accessToken),
    [accessToken],
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
