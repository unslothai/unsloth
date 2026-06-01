// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  type GgufVariantDetail,
  listGgufVariants,
  useGgufVariantsCacheVersion,
} from "../inventory";
import { fingerprintToken } from "../lib/token-fingerprint";

interface GgufVariantFetchState {
  key: string;
  variants: GgufVariantDetail[] | null;
  defaultVariant: string | null;
  loading: boolean;
  error: string | null;
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export function useGgufVariantFetchState({
  repoId,
  hfToken,
  preferLocalCache = false,
  localPath = null,
  enabled = true,
  errorFallback = "Failed to load variants",
}: {
  repoId: string;
  hfToken?: string | null;
  preferLocalCache?: boolean;
  localPath?: string | null;
  enabled?: boolean;
  errorFallback?: string;
}) {
  const variantsVersion = useGgufVariantsCacheVersion(repoId);
  const localVariantPath = localPath?.trim() || null;
  const variantKey = useMemo(
    () =>
      `${repoId}::${fingerprintToken(hfToken)}::${
        preferLocalCache ? "local" : "remote"
      }::${localVariantPath ?? ""}::${variantsVersion}`,
    [hfToken, localVariantPath, preferLocalCache, repoId, variantsVersion],
  );
  const [state, setState] = useState<GgufVariantFetchState>(() => ({
    key: variantKey,
    variants: null,
    defaultVariant: null,
    loading: enabled,
    error: null,
  }));
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async (): Promise<void> => {
    abortRef.current?.abort();
    if (!enabled) {
      setState({
        key: variantKey,
        variants: null,
        defaultVariant: null,
        loading: false,
        error: null,
      });
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setState({
      key: variantKey,
      variants: null,
      defaultVariant: null,
      loading: true,
      error: null,
    });
    try {
      const response = await listGgufVariants(repoId, hfToken || undefined, {
        preferLocalCache,
        localPath: localVariantPath,
        signal: controller.signal,
      });
      if (controller.signal.aborted) return;
      setState({
        key: variantKey,
        variants: response.variants,
        defaultVariant: response.default_variant,
        loading: false,
        error: null,
      });
    } catch (error) {
      if (controller.signal.aborted || isAbortError(error)) return;
      setState({
        key: variantKey,
        variants: null,
        defaultVariant: null,
        loading: false,
        error: error instanceof Error ? error.message : errorFallback,
      });
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
      }
    }
  }, [
    enabled,
    errorFallback,
    hfToken,
    localVariantPath,
    preferLocalCache,
    repoId,
    variantKey,
  ]);

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => {
      window.clearTimeout(timer);
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, [refresh]);

  return state.key === variantKey
    ? { ...state, refresh }
    : {
        key: variantKey,
        variants: null,
        defaultVariant: null,
        loading: enabled,
        error: null,
        refresh,
      };
}
