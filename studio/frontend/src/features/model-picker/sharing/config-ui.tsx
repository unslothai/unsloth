// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useEffect, useState, useSyncExternalStore } from "react";
import type { ModelPickTarget } from "../components/model-selector/types";
import { modelConfigDraftKey } from "../model-config/model-config-draft";
import type { PerModelConfig } from "../model-config/per-model-config";
import {
  SHARED_CONFIG_FIELDS,
  SHARED_CONFIG_KEYS,
  formatSharedConfigValue,
} from "./fields";

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
  onImport: (changes: Partial<PerModelConfig>, model?: string) => void;
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

export function SharedRunConfigReview({
  config,
  model,
  draftConfig,
  currentConfig,
}: {
  config: Partial<PerModelConfig> | null;
  model?: string;
  draftConfig: PerModelConfig;
  currentConfig: PerModelConfig;
}) {
  if (!config) {
    return null;
  }
  const keys = SHARED_CONFIG_KEYS.filter((key) => Object.hasOwn(config, key));
  if (
    keys.some(
      (key) => JSON.stringify(config[key]) !== JSON.stringify(draftConfig[key]),
    )
  ) {
    return null;
  }
  return (
    <details open={true} className="mb-5 rounded-lg border p-3 text-sm">
      <summary className="cursor-pointer font-medium">
        {keys.length > 0
          ? `Settings changed by link (${keys.length})`
          : model
            ? "Model selected by link"
            : "Link settings already match this editor"}
      </summary>
      {model && (
        <p className="my-2 break-words text-xs text-muted-foreground">
          This link selected {model} from Hugging Face. Loading downloads any
          model files that are not already cached.
        </p>
      )}
      <p className="my-2 text-xs text-muted-foreground">
        Loading with “Remember for this model” checked saves these settings for
        future loads, replacing any saved settings. Loading with it unchecked
        deletes any saved settings for this model. Close this editor without
        loading to keep your saved settings.
      </p>
      {keys.length > 0 && (
        <>
          <p className="my-2 text-xs text-muted-foreground">
            Adjustments for this device or model are shown below. This summary
            closes when you edit the settings.
          </p>
          <dl className="max-h-48 space-y-2 overflow-y-auto">
            {keys.map((key) => (
              <div key={key}>
                <dt className="font-medium">
                  {SHARED_CONFIG_FIELDS[key].label}
                </dt>
                <dd className="whitespace-pre-wrap break-words text-xs text-muted-foreground">
                  {formatSharedConfigValue(key, currentConfig)}
                  {JSON.stringify(config[key]) !==
                    JSON.stringify(currentConfig[key]) && (
                    <span className="block">
                      Requested: {formatSharedConfigValue(key, config)}.
                      Adjusted for this device or model; unsupported values will
                      not be used.
                    </span>
                  )}
                </dd>
              </div>
            ))}
          </dl>
        </>
      )}
    </details>
  );
}
