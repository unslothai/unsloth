// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { useInventoryVersion } from "@/stores/inventory-events";
import { useHfTokenStore } from "@/stores/hf-token-store";
import {
  fetchModelDefaults,
  modelDefaultsRequestKey,
  type ModelDefaultsFetchOptions,
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

type ModelDefaultsFields = Pick<ModelDefaults, "maxContext" | "chatTemplate">;

type ModelDefaultsState = ModelDefaultsFields & {
  requestKey: string;
  maxContextVersion: number;
  chatTemplateVersion: number;
};

function readDefaultsState(
  modelId: string,
  hfToken: string,
  fetchOptions: ModelDefaultsFetchOptions,
  inventoryVersion: number,
): ModelDefaultsFields {
  const cached = readCachedModelDefaults(
    modelId,
    hfToken,
    fetchOptions,
    inventoryVersion,
  );
  return {
    maxContext: cached?.maxPositionEmbeddings ?? null,
    chatTemplate: cached?.chatTemplate ?? null,
  };
}

function createDefaultsState(
  requestKey: string,
  defaults: ModelDefaultsFields,
): ModelDefaultsState {
  return {
    requestKey,
    ...defaults,
    maxContextVersion: 0,
    chatTemplateVersion: 0,
  };
}

function hasCompleteDefaults(defaults: ModelDefaultsFields): boolean {
  return defaults.maxContext != null && defaults.chatTemplate != null;
}

export function useModelDefaults(
  modelId: string,
  options: { skip?: boolean } & ModelDefaultsFetchOptions = {},
): ModelDefaults {
  const {
    skip = false,
    preferLocalCache = false,
    localPath = null,
    modelFormat = null,
  } = options;
  const fetchOptions = useMemo(
    () => ({ preferLocalCache, localPath, modelFormat }),
    [preferLocalCache, localPath, modelFormat],
  );
  const hfToken = useHfTokenStore((s) => s.token);
  const inventoryVersion = useInventoryVersion();
  const requestKey = useMemo(
    () =>
      modelDefaultsRequestKey(modelId, hfToken, fetchOptions, inventoryVersion),
    [
      modelId,
      hfToken,
      fetchOptions,
      inventoryVersion,
    ],
  );
  const cachedDefaults = useMemo(
    () => readDefaultsState(modelId, hfToken, fetchOptions, inventoryVersion),
    [modelId, hfToken, fetchOptions, inventoryVersion],
  );
  const [storedDefaults, setStoredDefaults] = useState<ModelDefaultsState>(() =>
    createDefaultsState(requestKey, cachedDefaults),
  );
  const activeDefaults = useMemo(
    () =>
      storedDefaults.requestKey === requestKey
        ? storedDefaults
        : createDefaultsState(requestKey, cachedDefaults),
    [storedDefaults, requestKey, cachedDefaults],
  );
  const defaultsRef = useRef(activeDefaults);
  const fetchedForRef = useRef<string | null>(null);
  const fetchGenerationRef = useRef(0);

  useEffect(() => {
    defaultsRef.current = activeDefaults;
  }, [activeDefaults]);

  const setChatTemplate = useCallback(
    (template: string | null) => {
      setStoredDefaults((current) => {
        const base =
          current.requestKey === requestKey
            ? current
            : createDefaultsState(requestKey, cachedDefaults);
        const next = {
          ...base,
          chatTemplate: template,
          chatTemplateVersion: base.chatTemplateVersion + 1,
        };
        defaultsRef.current = next;
        return next;
      });
    },
    [requestKey, cachedDefaults],
  );

  useEffect(() => {
    if (skip) return;
    if (fetchedForRef.current === requestKey) return;
    const current = defaultsRef.current;
    if (current.maxContext != null && current.chatTemplate != null) return;
    const controller = new AbortController();
    const generation = ++fetchGenerationRef.current;
    const maxContextVersion = current.maxContextVersion;
    const chatTemplateVersion = current.chatTemplateVersion;
    fetchModelDefaults(modelId, controller.signal, fetchOptions)
      .then((defaults) => {
        if (controller.signal.aborted) return;
        if (fetchGenerationRef.current !== generation) return;
        setStoredDefaults((stored) => {
          const base =
            stored.requestKey === requestKey
              ? stored
              : createDefaultsState(requestKey, cachedDefaults);
          const next = { ...base };
          if (
            base.maxContext == null &&
            base.maxContextVersion === maxContextVersion
          ) {
            next.maxContext = defaults.maxPositionEmbeddings;
          }
          if (
            base.chatTemplate == null &&
            base.chatTemplateVersion === chatTemplateVersion
          ) {
            next.chatTemplate = defaults.chatTemplate;
          }
          if (hasCompleteDefaults(next)) {
            fetchedForRef.current = requestKey;
          }
          defaultsRef.current = next;
          if (
            next.maxContext === base.maxContext &&
            next.chatTemplate === base.chatTemplate &&
            stored.requestKey === requestKey
          ) {
            return stored;
          }
          return next;
        });
      })
      .catch((err) => {
        if (controller.signal.aborted || isAbortError(err)) return;
        if (fetchGenerationRef.current !== generation) return;
        toast.error("Couldn't load model defaults", {
          description: err instanceof Error ? err.message : undefined,
        });
      });
    return () => {
      controller.abort();
      if (fetchGenerationRef.current === generation) {
        fetchGenerationRef.current += 1;
      }
    };
  }, [skip, modelId, requestKey, fetchOptions, cachedDefaults]);

  return {
    maxContext: activeDefaults.maxContext,
    chatTemplate: activeDefaults.chatTemplate,
    setChatTemplate,
  };
}
