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
  scopeKey: string;
  variants: GgufVariantDetail[] | null;
  defaultVariant: string | null;
  loading: boolean;
  error: string | null;
  refreshError: string | null;
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
  const variantScopeKey = useMemo(
    () =>
      `${repoId}::${fingerprintToken(hfToken)}::${
        preferLocalCache ? "local" : "remote"
      }::${localVariantPath ?? ""}`,
    [hfToken, localVariantPath, preferLocalCache, repoId],
  );
  const variantKey = useMemo(
    () => `${variantScopeKey}::${variantsVersion}`,
    [variantScopeKey, variantsVersion],
  );
  const [state, setState] = useState<GgufVariantFetchState>(() => ({
    key: variantKey,
    scopeKey: variantScopeKey,
    variants: null,
    defaultVariant: null,
    loading: enabled,
    error: null,
    refreshError: null,
  }));
  const abortRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async (): Promise<void> => {
    abortRef.current?.abort();
    if (!enabled) {
      setState({
        key: variantKey,
        scopeKey: variantScopeKey,
        variants: null,
        defaultVariant: null,
        loading: false,
        error: null,
        refreshError: null,
      });
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setState((prev) => {
      const sameScope = prev.scopeKey === variantScopeKey;
      return {
        key: variantKey,
        scopeKey: variantScopeKey,
        variants: sameScope ? prev.variants : null,
        defaultVariant: sameScope ? prev.defaultVariant : null,
        loading: true,
        error: null,
        refreshError: null,
      };
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
        scopeKey: variantScopeKey,
        variants: response.variants,
        defaultVariant: response.default_variant,
        loading: false,
        error: null,
        refreshError: null,
      });
    } catch (error) {
      if (controller.signal.aborted || isAbortError(error)) return;
      setState((prev) => {
        const sameScope = prev.scopeKey === variantScopeKey;
        const variants = sameScope ? prev.variants : null;
        const hasUsableVariants = !!variants && variants.length > 0;
        const message = error instanceof Error ? error.message : errorFallback;
        return {
          key: variantKey,
          scopeKey: variantScopeKey,
          variants,
          defaultVariant: sameScope ? prev.defaultVariant : null,
          loading: false,
          error: hasUsableVariants ? null : message,
          refreshError: hasUsableVariants ? message : null,
        };
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
    variantScopeKey,
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

  if (state.key === variantKey) {
    return { ...state, refresh };
  }
  if (state.scopeKey === variantScopeKey) {
    return {
      ...state,
      key: variantKey,
      loading: true,
      error: null,
      refreshError: null,
      refresh,
    };
  }
  return {
    key: variantKey,
    scopeKey: variantScopeKey,
    variants: null,
    defaultVariant: null,
    loading: enabled,
    error: null,
    refreshError: null,
    refresh,
  };
}
