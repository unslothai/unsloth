// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useOnlineStatus } from "@/features/hub/hooks/use-online-status";
import { fingerprintToken } from "@/features/hub/lib/token-fingerprint";
import { useEffect, useState } from "react";
import {
  type DatasetSizeInfo,
  fetchDatasetSize,
} from "../lib/dataset-size";

export function useDatasetSize(
  repoId: string | null | undefined,
  options: { enabled?: boolean; token?: string | null } = {},
): DatasetSizeInfo | null {
  const online = useOnlineStatus();
  const enabled = options.enabled ?? true;
  const token = options.token || undefined;
  const repoKey = repoId ?? "";
  const key = repoKey ? `${repoKey}::${fingerprintToken(token)}` : "";
  const [state, setState] = useState<{
    key: string;
    value: DatasetSizeInfo | null;
  }>(() => ({ key, value: null }));

  useEffect(() => {
    if (!repoKey || !enabled || !online) return;
    const controller = new AbortController();
    void fetchDatasetSize(repoKey, token, controller.signal).then((value) => {
      if (!controller.signal.aborted) {
        setState({ key, value });
      }
    });
    return () => {
      controller.abort();
    };
  }, [enabled, key, online, repoKey, token]);

  return state.key === key ? state.value : null;
}
