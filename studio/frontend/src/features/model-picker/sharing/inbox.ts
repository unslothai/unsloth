// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelConfigHandoffRequest } from "../model-config/model-config-handoff";
import type { PerModelConfig } from "../model-config/per-model-config";
import type { SharedRunConfig } from "./links";

export type RunConfigRequest = {
  id: string;
  value: SharedRunConfig;
  selectedModel?: string;
  target?: Pick<ModelConfigHandoffRequest, "id" | "meta">;
  draftKey?: string;
  replaceHistory?: boolean;
  newChatId?: ModelConfigHandoffRequest["newChatId"];
};

export function createRunConfigInbox() {
  let pending: RunConfigRequest | null = null;
  const listeners = new Set<() => void>();
  const editors = new Map<string, number>();
  const editedDrafts = new Set<string>();
  const publish = (next: RunConfigRequest | null) => {
    if (next?.id !== pending?.id) {
      editedDrafts.clear();
    }
    pending = next;
    for (const listener of listeners) {
      listener();
    }
  };
  return {
    getSnapshot: () => pending,
    recordEdit: (draftKey: string) => {
      if (pending) editedDrafts.add(draftKey);
    },
    wasEdited: (draftKey: string) => editedDrafts.has(draftKey),
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
