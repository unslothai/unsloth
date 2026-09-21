// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { PerModelConfig } from "../model-config/per-model-config";
import { SHARED_CONFIG_FIELDS, SHARED_CONFIG_KEYS } from "./fields";

function displayValue(value: unknown): string {
  return value == null
    ? "Default"
    : typeof value === "string" && value !== ""
      ? value
      : JSON.stringify(value);
}

export function SharedRunConfigReview({
  config,
  draftConfig,
  currentConfig,
}: {
  config: Partial<PerModelConfig> | null;
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
    <details open className="mb-5 rounded-lg border p-3 text-sm">
      <summary className="cursor-pointer font-medium">
        {keys.length
          ? `Settings changed by link (${keys.length})`
          : "Link settings already match this editor"}
      </summary>
      {keys.length > 0 && (
        <>
          <p className="my-2 text-xs text-muted-foreground">
            Review text and extra arguments before editing or loading. Adjustments
            for this device or model are shown below. This summary closes when you
            edit the settings.
          </p>
          <dl className="max-h-48 space-y-2 overflow-y-auto">
            {keys.map((key) => (
              <div key={key}>
                <dt className="font-medium">
                  {SHARED_CONFIG_FIELDS[key].label}
                </dt>
                <dd className="whitespace-pre-wrap break-words text-xs text-muted-foreground">
                  {displayValue(currentConfig[key])}
                  {JSON.stringify(config[key]) !==
                    JSON.stringify(currentConfig[key]) && (
                    <span className="block">
                      Requested: {displayValue(config[key])}. Adjusted for this
                      device or model; unsupported values will not be used.
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
