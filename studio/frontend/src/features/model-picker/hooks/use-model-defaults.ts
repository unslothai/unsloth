// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getHfToken } from "@/features/hub/stores/hf-token-store";
import { useEffect, useState } from "react";
import { fetchDefaultChatTemplate } from "../api/templates";

export interface DefaultChatTemplateState {
  template: string | null;
  loading: boolean;
  error: string | null;
}

const templateCache = new Map<string, string | null>();

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
        templateCache.set(cacheKey, template);
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
