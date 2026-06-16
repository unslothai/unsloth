// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";
import { useDebouncedValue } from "./use-debounced-value";

export interface HubPrecheckState {
  valid: boolean | null;
  message: string | null;
  isChecking: boolean;
  details?: Record<string, unknown> | null;
}

const INITIAL: HubPrecheckState = {
  valid: null,
  message: null,
  isChecking: false,
  details: null,
};

/**
 * Validates Hugging Face Hub upload settings before a long export starts.
 * Debounced to avoid excessive requests while typing repo fields.
 */
export function useHubExportPrecheck(params: {
  enabled: boolean;
  repoId: string;
  hfToken: string;
  privateRepo: boolean;
}): HubPrecheckState {
  const debouncedRepoId = useDebouncedValue(params.repoId.trim(), 500);
  const debouncedToken = useDebouncedValue(
    params.hfToken.trim().replace(/^["']+|["']+$/g, ""),
    500,
  );
  const [state, setState] = useState<HubPrecheckState>(INITIAL);
  const versionRef = useRef(0);

  const runCheck = useCallback(
    async (
      repoId: string | null,
      token: string,
      privateRepo: boolean,
    ) => {
      const v = ++versionRef.current;
      setState((prev) => ({ ...prev, isChecking: true, message: null }));

      try {
        const { precheckHubExport } = await import(
          "@/features/export/api/export-api"
        );
        const response = await precheckHubExport({
          ...(repoId ? { repo_id: repoId } : {}),
          hf_token: token || null,
          private: privateRepo,
        });
        if (versionRef.current !== v) return;
        setState({
          valid: response.valid,
          message: response.message,
          isChecking: false,
          details: response.details ?? null,
        });
      } catch (err) {
        if (versionRef.current !== v) return;
        setState({
          valid: false,
          message: err instanceof Error ? err.message : "Hub precheck failed",
          isChecking: false,
          details: null,
        });
      }
    },
    [],
  );

  useEffect(() => {
    if (!params.enabled) {
      setState(INITIAL);
      return;
    }
    if (debouncedRepoId && debouncedRepoId.includes("/")) {
      runCheck(debouncedRepoId, debouncedToken, params.privateRepo);
      return;
    }
    runCheck(null, debouncedToken, params.privateRepo);
  }, [
    params.enabled,
    debouncedRepoId,
    debouncedToken,
    params.privateRepo,
    runCheck,
  ]);

  return state;
}
