// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  fetchModelDefaults,
  readCachedModelDefaults,
} from "./model-defaults-fetch";

export function isAbortError(err: unknown): boolean {
  return (err as { name?: string } | null)?.name === "AbortError";
}

export interface ModelDefaults {
  maxContext: number | null;
  chatTemplate: string | null;
  setChatTemplate: (template: string | null) => void;
}

export function useModelDefaults(
  modelId: string,
  options: { skip?: boolean } = {},
): ModelDefaults {
  const { skip = false } = options;
  const [maxContext, setMaxContext] = useState<number | null>(
    () => readCachedModelDefaults(modelId)?.maxPositionEmbeddings ?? null,
  );
  const [chatTemplate, setChatTemplate] = useState<string | null>(
    () => readCachedModelDefaults(modelId)?.chatTemplate ?? null,
  );
  const fetchedForRef = useRef<string | null>(null);

  useEffect(() => {
    const cached = readCachedModelDefaults(modelId);
    setMaxContext(cached?.maxPositionEmbeddings ?? null);
    setChatTemplate(cached?.chatTemplate ?? null);
  }, [modelId]);

  useEffect(() => {
    if (skip) return;
    if (maxContext != null && chatTemplate != null) return;
    if (fetchedForRef.current === modelId) return;
    const controller = new AbortController();
    fetchModelDefaults(modelId, controller.signal)
      .then((defaults) => {
        if (controller.signal.aborted) return;
        fetchedForRef.current = modelId;
        if (maxContext == null) setMaxContext(defaults.maxPositionEmbeddings);
        if (chatTemplate == null) setChatTemplate(defaults.chatTemplate);
      })
      .catch((err) => {
        if (controller.signal.aborted || isAbortError(err)) return;
        toast.error("Couldn't load model defaults", {
          description: err instanceof Error ? err.message : undefined,
        });
      });
    return () => controller.abort();
  }, [chatTemplate, maxContext, skip, modelId]);

  return { maxContext, chatTemplate, setChatTemplate };
}
