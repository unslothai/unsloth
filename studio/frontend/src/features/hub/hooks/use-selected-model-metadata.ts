// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { usePlatformStore } from "@/config/env";
import type { HfModelResult } from "@/features/hub/hooks/use-hub-model-search";
import { getHfEndpoint } from "@/lib/hf-endpoint";
import { cachedModelInfo } from "../lib/hf-cache";
import { useEffect, useState } from "react";
import { toHfModelResult } from "../lib/view-models";

type SelectedModelMetadataResult = ReturnType<typeof toHfModelResult>;

export function useSelectedModelMetadata(
  repoId: string | null,
  {
    accessToken,
    enabled,
    online,
  }: {
    accessToken: string | undefined;
    enabled: boolean;
    online: boolean;
  },
): { result: SelectedModelMetadataResult; error: boolean } {
  const [state, setState] = useState<{
    repoId: string;
    result: HfModelResult | null;
    error: boolean;
  }>(() => ({ repoId: "", result: null, error: false }));

  // getHfEndpoint() is a plain module read React cannot see change, so an
  // endpoint that lands after this effect has run (a /api/health that first
  // failed, or a cold desktop start) would leave the pane showing the default
  // hub's answer, or its error. Taking it from the store puts it in the effect's
  // identity, so a late-arriving mirror refetches.
  const hfEndpoint = usePlatformStore((s) => s.hfEndpoint);

  useEffect(() => {
    if (!(repoId && enabled && online)) {
      return;
    }

    // No AbortController: cachedModelInfo shares one in-flight request per repo
    // across callers (hf-cache.ts), so aborting would cancel it for everyone.
    // The `cancelled` flag plus the state.repoId guard prevent stale writes.
    let cancelled = false;

    cachedModelInfo({
      hubUrl: getHfEndpoint(),
      name: repoId,
      ...(accessToken ? { accessToken } : {}),
    })
      .then((result) => {
        if (cancelled) {
          return;
        }
        setState({
          repoId,
          result: toHfModelResult(result),
          error: false,
        });
      })
      .catch(() => {
        if (cancelled) {
          return;
        }
        setState({
          repoId,
          result: null,
          error: true,
        });
      });

    return () => {
      cancelled = true;
    };
  }, [repoId, accessToken, enabled, online, hfEndpoint]);

  if (state.repoId !== repoId) {
    return { result: null, error: false };
  }

  return { result: state.result, error: state.error };
}
