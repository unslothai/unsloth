// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useEffect, useState, useSyncExternalStore } from "react";
import type { ModelPickTarget } from "../components/model-selector/types";
import { modelConfigDraftKey } from "../model-config/model-config-draft";
import type { PerModelConfig } from "../model-config/per-model-config";
import { scheduleRunConfigImport } from "./import-config";
import { runConfigInbox } from "./inbox";
import { ShareRunConfigDialog } from "./share-dialog";

export function SharedRunConfigActions({
  target,
  config,
  ready,
  hydrated,
  canImport,
  disabled,
  className,
  onImport,
}: {
  target: ModelPickTarget;
  config: PerModelConfig;
  ready: boolean;
  hydrated: boolean;
  canImport: boolean;
  disabled: boolean;
  className: string;
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
  useEffect(
    () =>
      scheduleRunConfigImport({
        canImport,
        ready,
        pending,
        key,
        hydrated,
        isGguf: target.isGguf,
        onImport,
      }),
    [canImport, hydrated, key, onImport, pending, ready, target.isGguf],
  );
  return (
    <>
      <Button
        type="button"
        size="sm"
        variant="outline"
        className={className}
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

export { SharedRunConfigReview } from "./config-review";
