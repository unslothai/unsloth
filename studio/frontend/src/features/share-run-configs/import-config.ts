import type { PerModelConfig } from "@/features/model-picker";
import { toast } from "@/lib/toast";
import {
  clearExtraArgsEditForDraft,
  markModelConfigDraftEdited,
  patchModelConfigDraft,
  readModelConfigDraft,
} from "../model-picker/model-config/model-config-draft";
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
  onImport,
}: {
  canImport: boolean;
  ready: boolean;
  pending: RunConfigRequest | null;
  key: string;
  hydrated: boolean;
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
    if (cancelled || !draft) {
      return;
    }
    const patch = runConfigInbox.take(pending.id, key);
    if (!patch || Object.keys(patch).length === 0) {
      return;
    }
    if (!hydrated) {
      toast.error("Could not import run settings", {
        description:
          "Saved model settings are unavailable. Close the editor and reopen this link to retry.",
      });
      return;
    }
    const merged = mergeSharedRunConfig(draft.config, patch);
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
      mergeSharedRunConfig(current, patch),
    );
    onImport(changes);
    toast.success("Settings imported from link", {
      description: "Review before loading.",
    });
  });
  return () => {
    cancelled = true;
  };
}
