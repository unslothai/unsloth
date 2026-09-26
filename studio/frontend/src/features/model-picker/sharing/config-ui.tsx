// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { ChevronDown } from "lucide-react";
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
  onImport: (
    changes: Partial<PerModelConfig>,
    model?: string,
    ggufVariant?: string,
  ) => void;
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

function reviewTitle(changed: boolean, model?: string, ggufVariant?: string) {
  if (changed) {
    return "Settings changed by link";
  }
  if (model) {
    return "Model selected by link";
  }
  return ggufVariant
    ? "GGUF variant selected by link"
    : "Link settings already match this editor";
}

function downloadNote(model?: string, ggufVariant?: string) {
  if (model) {
    return "If the model isn’t on this device, loading downloads it from Hugging Face.";
  }
  return ggufVariant
    ? "If this variant isn’t on this device, loading downloads it."
    : null;
}

function savedSettingsNote(remember: boolean, hasSavedSettings: boolean) {
  if (remember) {
    return hasSavedSettings
      ? "Loading replaces your saved settings for this model. Close without loading to keep them."
      : "Loading saves these settings for this model.";
  }
  return hasSavedSettings
    ? "Loading clears your saved settings for this model. Close without loading to keep them."
    : "To keep these settings for next time, tick “Remember for this model”.";
}

export function SharedRunConfigReview({
  config,
  model,
  ggufVariant,
  draftConfig,
  currentConfig,
  remember = false,
  hasSavedSettings = false,
}: {
  config: Partial<PerModelConfig> | null;
  model?: string;
  ggufVariant?: string;
  draftConfig: PerModelConfig;
  currentConfig: PerModelConfig;
  remember?: boolean;
  hasSavedSettings?: boolean;
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
  const selection = [model, ggufVariant].filter(Boolean).join(" · ");
  const download = downloadNote(model, ggufVariant);
  return (
    <details
      open={true}
      className="group mb-5 rounded-2xl border border-border/60"
    >
      <summary className="flex cursor-pointer list-none items-center justify-between gap-3 rounded-2xl px-4 py-3 outline-none focus-visible:ring-2 focus-visible:ring-ring/60 [&::-webkit-details-marker]:hidden">
        <span className="min-w-0">
          <span className="flex min-w-0 items-center gap-2">
            <span className="min-w-0 truncate text-ui-13 font-medium leading-[1.25] tracking-nav text-foreground">
              {reviewTitle(keys.length > 0, model, ggufVariant)}
            </span>
            {keys.length > 0 && (
              <span className="shrink-0 rounded-md bg-[rgb(0_0_0_/_calc(0.04*var(--contrast-wash-gain,1)))] px-1.5 py-0.5 text-ui-10 font-medium tabular-nums text-muted-foreground dark:bg-muted">
                {keys.length}
              </span>
            )}
          </span>
          {selection && (
            <span className="mt-1 block text-ui-11 text-muted-foreground [overflow-wrap:anywhere]">
              {selection}
            </span>
          )}
        </span>
        <ChevronDown
          aria-hidden="true"
          className="size-3.5 shrink-0 -rotate-90 text-muted-foreground transition-transform duration-200 group-open:rotate-0 group-hover:text-foreground motion-reduce:transition-none"
        />
      </summary>
      <div className="space-y-3 px-4 pb-3.5">
        {keys.length > 0 && (
          <dl className="space-y-2">
            {keys.map((key) => (
              <div
                key={key}
                className="grid grid-cols-[auto_minmax(0,1fr)] items-baseline gap-x-4 gap-y-0.5"
              >
                <dt className="text-ui-12 text-muted-foreground">
                  {SHARED_CONFIG_FIELDS[key].label}
                </dt>
                <dd className="min-w-0 whitespace-pre-wrap text-right text-ui-12 tabular-nums text-foreground [overflow-wrap:anywhere]">
                  {formatSharedConfigValue(key, currentConfig)}
                </dd>
                {JSON.stringify(config[key]) !==
                  JSON.stringify(currentConfig[key]) && (
                  <dd className="col-span-2 min-w-0 whitespace-pre-wrap text-ui-11 leading-relaxed text-muted-foreground/80 [overflow-wrap:anywhere]">
                    Requested {formatSharedConfigValue(key, config)}, adjusted
                    for this device or model.
                  </dd>
                )}
              </div>
            ))}
          </dl>
        )}
        <div
          className={`space-y-1 text-ui-11 leading-relaxed text-muted-foreground ${keys.length > 0 ? "border-t border-border/60 pt-3" : ""}`}
        >
          {download && <p>{download}</p>}
          <p>{savedSettingsNote(remember, hasSavedSettings)}</p>
        </div>
      </div>
    </details>
  );
}
