// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { toast } from "@/lib/toast";
import {
  Suspense,
  lazy,
  useEffect,
  useLayoutEffect,
  useState,
  useSyncExternalStore,
} from "react";
import type { ModelPickTarget } from "../components/model-selector/types";
import { modelConfigDraftKey } from "../model-config/model-config-draft";
import type { PerModelConfig } from "../model-config/per-model-config";
import { scheduleRunConfigImport } from "./import-config";
import { runConfigInbox } from "./inbox";

const ShareRunConfigDialog = lazy(() =>
  import("./share-dialog").then((module) => ({
    default: module.ShareRunConfigDialog,
  })),
);

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
      return runConfigInbox.retainEditor(key, (request) => {
        toast.info("Run settings import cancelled", {
          id: request.id,
          description:
            "The editor closed before the settings were imported. Reopen the link to try again.",
        });
      });
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
        data-shared-run-config={sharing ? "" : undefined}
        disabled={!ready || disabled}
        onClick={() => setSharing(true)}
      >
        Share
      </Button>
      {sharing && (
        <Suspense fallback={null}>
          <ShareRunConfigDialog
            target={target}
            config={config}
            onClose={() => setSharing(false)}
          />
        </Suspense>
      )}
    </>
  );
}
