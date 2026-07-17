// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useHfTokenStore } from "@/features/hub";
import { useEffect, useState } from "react";
import { fetchModelMaxPositionEmbeddings } from "../api/model-metadata";
import { fetchDefaultChatTemplate } from "../api/templates";

export interface DefaultChatTemplateState {
  template: string | null;
  loading: boolean;
  error: string | null;
}

export interface ModelMaxPositionState {
  maxPositionEmbeddings: number | null;
  loading: boolean;
  error: string | null;
}

const TEMPLATE_CACHE_MAX_ENTRIES = 50;
const templateCache = new Map<string, string | null>();
const maxPositionCache = new Map<string, number | null>();

function cacheTemplate(key: string, template: string | null): void {
  templateCache.delete(key);
  templateCache.set(key, template);
  while (templateCache.size > TEMPLATE_CACHE_MAX_ENTRIES) {
    const oldest = templateCache.keys().next().value;
    if (oldest === undefined) {
      break;
    }
    templateCache.delete(oldest);
  }
}

function cacheMaxPosition(key: string, value: number | null): void {
  maxPositionCache.delete(key);
  maxPositionCache.set(key, value);
  while (maxPositionCache.size > TEMPLATE_CACHE_MAX_ENTRIES) {
    const oldest = maxPositionCache.keys().next().value;
    if (oldest === undefined) {
      break;
    }
    maxPositionCache.delete(oldest);
  }
}

export function useDefaultChatTemplate(
  modelId: string | null,
  ggufVariant: string | null | undefined,
  enabled: boolean,
): DefaultChatTemplateState {
  const token = useHfTokenStore((s) => s.token);
  const cacheKey =
    enabled && modelId ? `${modelId}::${ggufVariant ?? ""}::${token}` : null;
  const [fetched, setFetched] = useState<{
    key: string;
    state: DefaultChatTemplateState;
  } | null>(null);

  useEffect(() => {
    if (cacheKey == null || !modelId || templateCache.has(cacheKey)) {
      return;
    }
    setFetched(null);
    const controller = new AbortController();
    fetchDefaultChatTemplate(modelId, ggufVariant, token, controller.signal)
      .then((template) => {
        if (controller.signal.aborted) {
          return;
        }
        // Cache the terminal result, including a null "no default template",
        // so reopening the viewer for such a model reuses it instead of
        // re-running the backend/Hugging Face lookup every time.
        cacheTemplate(cacheKey, template);
        setFetched({
          key: cacheKey,
          state: { template, loading: false, error: null },
        });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setFetched({
          key: cacheKey,
          state: {
            template: null,
            loading: false,
            error:
              err instanceof Error ? err.message : "Failed to load template",
          },
        });
      });

    return () => controller.abort();
  }, [cacheKey, modelId, ggufVariant, token]);

  if (cacheKey == null) {
    return { template: null, loading: false, error: null };
  }
  if (templateCache.has(cacheKey)) {
    return {
      template: templateCache.get(cacheKey) ?? null,
      loading: false,
      error: null,
    };
  }
  if (fetched?.key === cacheKey) {
    return fetched.state;
  }
  return { template: null, loading: true, error: null };
}

export function useModelMaxPositionEmbeddings(
  modelId: string | null,
  enabled: boolean,
): ModelMaxPositionState {
  const token = useHfTokenStore((s) => s.token);
  const cacheKey = enabled && modelId ? `${modelId}::${token}` : null;
  const [fetched, setFetched] = useState<{
    key: string;
    state: ModelMaxPositionState;
  } | null>(null);

  useEffect(() => {
    if (cacheKey == null || !modelId || maxPositionCache.has(cacheKey)) {
      return;
    }
    setFetched(null);
    const controller = new AbortController();
    fetchModelMaxPositionEmbeddings(modelId, token, controller.signal)
      .then((maxPositionEmbeddings) => {
        if (controller.signal.aborted) {
          return;
        }
        cacheMaxPosition(cacheKey, maxPositionEmbeddings);
        setFetched({
          key: cacheKey,
          state: { maxPositionEmbeddings, loading: false, error: null },
        });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setFetched({
          key: cacheKey,
          state: {
            maxPositionEmbeddings: null,
            loading: false,
            error:
              err instanceof Error
                ? err.message
                : "Failed to load model metadata",
          },
        });
      });

    return () => controller.abort();
  }, [cacheKey, modelId, token]);

  if (cacheKey == null) {
    return { maxPositionEmbeddings: null, loading: false, error: null };
  }
  if (maxPositionCache.has(cacheKey)) {
    return {
      maxPositionEmbeddings: maxPositionCache.get(cacheKey) ?? null,
      loading: false,
      error: null,
    };
  }
  if (fetched?.key === cacheKey) {
    return fetched.state;
  }
  return { maxPositionEmbeddings: null, loading: true, error: null };
}
