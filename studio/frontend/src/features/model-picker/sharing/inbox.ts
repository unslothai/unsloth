// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelConfigHandoffRequest } from "../model-config/model-config-handoff";
import type { PerModelConfig } from "../model-config/per-model-config";
import { SHARED_CONFIG_KEYS } from "./fields";
import type { SharedRunConfig } from "./links";

export type RunConfigRequest = {
  id: string;
  value: SharedRunConfig;
  selectedModel?: string;
  target?: Pick<ModelConfigHandoffRequest, "id" | "meta">;
  draftKey?: string;
  replaceHistory?: boolean;
  newChatId?: string | null;
};

export function mergeSharedRunConfig(
  defaults: PerModelConfig,
  patch: Partial<PerModelConfig>,
  isGguf: boolean,
): PerModelConfig {
  const provided = Object.fromEntries(
    SHARED_CONFIG_KEYS.filter((key) => Object.hasOwn(patch, key))
      .map((key) => [key, patch[key]] as const)
      .filter(([, value]) => value !== undefined)
      .map(([key, value]) => [key, Array.isArray(value) ? [...value] : value]),
  );
  if (isGguf && Object.hasOwn(provided, "maxSeqLength")) {
    provided.customContextLength ??= provided.maxSeqLength;
    provided.maxSeqLength = null;
  }
  if (
    Object.hasOwn(provided, "customContextLength") &&
    !Object.hasOwn(provided, "maxSeqLength")
  ) {
    provided.maxSeqLength = null;
  }
  if (
    Object.hasOwn(provided, "maxSeqLength") &&
    !Object.hasOwn(provided, "customContextLength")
  ) {
    provided.customContextLength = null;
  }
  return { ...defaults, ...provided };
}

export function createRunConfigInbox() {
  let pending: RunConfigRequest | null = null;
  const listeners = new Set<() => void>();
  const editors = new Map<string, number>();
  const publish = (next: RunConfigRequest | null) => {
    pending = next;
    for (const listener of listeners) {
      listener();
    }
  };
  return {
    getSnapshot: () => pending,
    subscribe: (listener: () => void) => {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    submit: (request: RunConfigRequest) => publish(request),
    retainEditor: (
      draftKey: string,
      onCancel?: (request: RunConfigRequest) => void,
    ) => {
      editors.set(draftKey, (editors.get(draftKey) ?? 0) + 1);
      let released = false;
      return () => {
        if (released) {
          return;
        }
        released = true;
        const remaining = (editors.get(draftKey) ?? 1) - 1;
        if (remaining > 0) {
          editors.set(draftKey, remaining);
        } else {
          editors.delete(draftKey);
        }
        const request = pending;
        queueMicrotask(() => {
          if (
            request &&
            pending === request &&
            !editors.has(draftKey) &&
            pending?.draftKey === draftKey
          ) {
            publish(null);
            if (Object.keys(request.value.config).length > 0) {
              onCancel?.(request);
            }
          }
        });
      };
    },
    bind: (id: string, draftKey: string) => {
      if (pending?.id === id) {
        publish({ ...pending, draftKey });
      }
    },
    clear: (id: string) => {
      if (pending?.id === id) {
        publish(null);
      }
    },
    take: (id: string, draftKey: string): Partial<PerModelConfig> | null => {
      if (pending?.id !== id || pending.draftKey !== draftKey) {
        return null;
      }
      const patch = pending.value.config;
      publish(null);
      return patch;
    },
  };
}

export const runConfigInbox = createRunConfigInbox();
