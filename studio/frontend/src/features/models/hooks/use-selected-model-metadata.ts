// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { HfModelResult } from "@/hooks";
import { cachedModelInfo } from "@/lib/hf-cache";
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

  useEffect(() => {
    if (!(repoId && enabled && online)) {
      return;
    }

    let cancelled = false;
    const controller = new AbortController();

    cachedModelInfo(
      {
        name: repoId,
        ...(accessToken ? { accessToken } : {}),
      },
      { signal: controller.signal },
    )
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
        if (cancelled || controller.signal.aborted) {
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
      controller.abort();
    };
  }, [repoId, accessToken, enabled, online]);

  if (state.repoId !== repoId) {
    return { result: null, error: false };
  }

  return { result: state.result, error: state.error };
}
