// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "@/lib/toast";
import {
  clearExtraArgsEditForDraft,
  markModelConfigDraftEdited,
  patchModelConfigDraft,
  readModelConfigDraft,
} from "../model-config/model-config-draft";
import type { PerModelConfig } from "../model-config/per-model-config";
import { SHARED_CONFIG_KEYS } from "./fields";
import {
  type RunConfigRequest,
  mergeSharedRunConfig,
  runConfigInbox,
} from "./inbox";

export function scheduleRunConfigImport({
  canImport,
  ready,
  pending,
  key,
  hydrated,
  isGguf,
  onImport,
}: {
  canImport: boolean;
  ready: boolean;
  pending: RunConfigRequest | null;
  key: string;
  hydrated: boolean;
  isGguf: boolean;
  onImport: (changes: Partial<PerModelConfig>) => void;
}): (() => void) | undefined {
  if (
    !(canImport && ready && pending) ||
    pending.draftKey !== key ||
    !readModelConfigDraft(key)
  ) {
    return;
  }
  let cancelled = false;
  queueMicrotask(() => {
    const draft = readModelConfigDraft(key);
    if (cancelled || !draft || runConfigInbox.getSnapshot() !== pending) {
      return;
    }
    if (Object.keys(pending.value.config).length === 0) {
      runConfigInbox.take(pending.id, key);
      return;
    }
    if (!hydrated) {
      toast.error("Could not import run settings", {
        id: pending.id,
        description:
          "Saved model settings are unavailable. Close the editor and reopen this link to retry.",
      });
      return;
    }
    const patch = runConfigInbox.take(pending.id, key);
    if (!patch) {
      return;
    }
    const merged = mergeSharedRunConfig(draft.config, patch, isGguf);
    const changes = Object.fromEntries(
      SHARED_CONFIG_KEYS.filter(
        (field) =>
          JSON.stringify(draft.config[field]) !== JSON.stringify(merged[field]),
      ).map((field) => [field, merged[field]]),
    );
    if (Object.hasOwn(patch, "llamaExtraArgs")) {
      clearExtraArgsEditForDraft(key);
    }
    markModelConfigDraftEdited(key);
    patchModelConfigDraft(key, (current) =>
      mergeSharedRunConfig(current, patch, isGguf),
    );
    onImport(changes);
    toast.success("Settings imported from link", {
      id: pending.id,
      description: "Review before loading.",
    });
  });
  return () => {
    cancelled = true;
  };
}
