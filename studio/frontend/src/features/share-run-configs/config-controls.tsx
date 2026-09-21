// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { SHARED_RUN_CONFIG_FOCUS_ATTRIBUTE } from "@/lib/shared-run-config-focus";
import { toast } from "@/lib/toast";
import {
  useEffect,
  useLayoutEffect,
  useState,
  useSyncExternalStore,
} from "react";
import type { ModelPickTarget } from "../model-picker/components/model-selector/types";
import {
  clearExtraArgsEditForDraft,
  markModelConfigDraftEdited,
  modelConfigDraftKey,
  patchModelConfigDraft,
  readModelConfigDraft,
} from "../model-picker/model-config/model-config-draft";
import type { PerModelConfig } from "../model-picker/model-config/per-model-config";
import { mergeSharedRunConfig, runConfigInbox } from "./inbox";
import { ShareRunConfigDialog } from "./share-dialog";

export function SharedRunConfigControls({
  target,
  config,
  ready,
  hydrated,
  canImport,
  disabled,
  onImport,
}: {
  target: ModelPickTarget;
  config: PerModelConfig;
  ready: boolean;
  hydrated: boolean;
  canImport: boolean;
  disabled: boolean;
  onImport: () => void;
}) {
  const [sharing, setSharing] = useState(false);
  const pending = useSyncExternalStore(
    runConfigInbox.subscribe,
    runConfigInbox.getSnapshot,
  );
  const key = modelConfigDraftKey(
    target.configId ?? target.id,
    target.ggufVariant,
  );
  useLayoutEffect(() => {
    if (canImport) {
      return runConfigInbox.retainEditor(key);
    }
  }, [canImport, key]);
  useEffect(() => {
    if (
      !(canImport && ready && pending) ||
      pending.draftKey !== key ||
      !readModelConfigDraft(key)
    ) {
      return;
    }
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) {
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
      if (Object.hasOwn(patch, "llamaExtraArgs")) {
        clearExtraArgsEditForDraft(key);
      }
      markModelConfigDraftEdited(key);
      patchModelConfigDraft(key, (current) =>
        mergeSharedRunConfig(current, patch),
      );
      onImport();
      toast.success("Settings imported from link", {
        description: "Review before loading.",
      });
    });
    return () => {
      cancelled = true;
    };
  }, [canImport, hydrated, key, onImport, pending, ready]);
  return (
    <>
      <Button
        type="button"
        size="sm"
        variant="ghost"
        className="h-8"
        {...{
          [SHARED_RUN_CONFIG_FOCUS_ATTRIBUTE]:
            sharing || (canImport && pending?.draftKey === key)
              ? ""
              : undefined,
        }}
        disabled={!ready || disabled}
        onClick={() => setSharing(true)}
      >
        Share
      </Button>
      {sharing && (
        <ShareRunConfigDialog
          target={target}
          config={config}
          onClose={() => setSharing(false)}
        />
      )}
    </>
  );
}
