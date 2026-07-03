// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { looksLikeLocalPath } from "@/features/hub/lib/local-path";
import { getHfToken } from "@/features/hub/stores/hf-token-store";
import { useEffect, useState } from "react";
import { fetchDefaultChatTemplate } from "../api/templates";

export interface DefaultChatTemplateState {
  template: string | null;
  loading: boolean;
  error: string | null;
}

const TEMPLATE_CACHE_MAX_ENTRIES = 50;
const templateCache = new Map<string, string | null>();

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

export function useDefaultChatTemplate(
  modelId: string | null,
  ggufVariant: string | null | undefined,
  enabled: boolean,
): DefaultChatTemplateState {
  const [state, setState] = useState<DefaultChatTemplateState>({
    template: null,
    loading: false,
    error: null,
  });

  useEffect(() => {
    if (!(enabled && modelId)) {
      return;
    }
    const token = getHfToken();
    const cacheKey = `${modelId}::${ggufVariant ?? ""}::${token}`;
    if (templateCache.has(cacheKey)) {
      setState({
        template: templateCache.get(cacheKey) ?? null,
        loading: false,
        error: null,
      });
      return;
    }

    const controller = new AbortController();
    setState({ template: null, loading: true, error: null });
    fetchDefaultChatTemplate(modelId, ggufVariant, token, controller.signal)
      .then((template) => {
        if (!(template === null && looksLikeLocalPath(modelId))) {
          cacheTemplate(cacheKey, template);
        }
        setState({ template, loading: false, error: null });
      })
      .catch((err: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setState({
          template: null,
          loading: false,
          error: err instanceof Error ? err.message : "Failed to load template",
        });
      });

    return () => controller.abort();
  }, [modelId, ggufVariant, enabled]);

  return state;
}
