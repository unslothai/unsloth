// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import type { ModelPickTarget, PerModelConfig } from "@/features/model-picker";
import {
  useEffect,
  useLayoutEffect,
  useState,
  useSyncExternalStore,
} from "react";
import { modelConfigDraftKey } from "../model-picker/model-config/model-config-draft";
import { SHARED_RUN_CONFIG_FOCUS_ATTRIBUTE } from "./editor-events";
import { scheduleRunConfigImport } from "./import-config";
import { runConfigInbox } from "./inbox";

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
  onImport: (changes: Partial<PerModelConfig>) => void;
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
  useEffect(
    () =>
      scheduleRunConfigImport({
        canImport,
        ready,
        pending,
        key,
        hydrated,
        onImport,
      }),
    [canImport, hydrated, key, onImport, pending, ready],
  );
  return (
    <>
      <Button
        type="button"
        size="sm"
        variant="ghost"
        className="h-8"
        {...{
          [SHARED_RUN_CONFIG_FOCUS_ATTRIBUTE]: sharing ? "" : undefined,
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
