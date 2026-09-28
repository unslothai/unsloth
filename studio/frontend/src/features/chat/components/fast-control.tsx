// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useContext, useState } from "react";
import { ZapIcon } from "lucide-react";
import { Switch } from "@/components/ui/switch";
// eslint-disable-next-line no-restricted-imports -- Store-only access avoids importing the settings dialog and its chat cycle.
import { useSettingsDialogStore } from "@/features/settings/stores/settings-dialog-store";
import { FastModelSelectionContext } from "../lib/fast-selection-context";
import { exactModelPricing } from "../model-catalog";
import { currentFast, toggleFast } from "../lib/fast-controls";
import { priceDifference } from "../lib/fast-variants";
import { tokenRate } from "../lib/model-pricing";
import { ModelPriceLine } from "./model-pricing";

export function FastControl() {
  const [enabling, setEnabling] = useState(false);
  const select = useContext(FastModelSelectionContext);
  const fast = currentFast();
  if (!fast.variant && !fast.native)
    return fast.provider?.providerType === "openrouter" ? (
      <button
        type="button"
        className="min-h-9 text-xs text-muted-foreground underline"
        onClick={() =>
          useSettingsDialogStore
            .getState()
            .openConnectionSettings(fast.provider!.id)
        }
      >
        Configure Fast models
      </button>
    ) : null;
  const reason = fast.variant?.reason;
  const tierUnavailable = fast.tier && !fast.tier.available;
  const delta =
    fast.variant && fast.selection
      ? priceDifference(
          exactModelPricing("openrouter", fast.selection.modelId),
          exactModelPricing("openrouter", fast.variant.destination),
        )
      : null;
  return (
    <div className="min-w-0 flex-1 space-y-2">
      <label className="flex min-h-9 items-center justify-between gap-2 text-sm font-medium">
        <span className="flex items-center gap-1.5">
          <ZapIcon className="size-4" aria-hidden="true" /> Fast
        </span>
        <Switch
          aria-label="Fast mode"
          checked={!!fast.isFast}
          disabled={
            fast.busy ||
            enabling ||
            !fast.connected ||
            !!reason ||
            (!!tierUnavailable && !fast.isFast) ||
            (!!fast.variant && !select)
          }
          onCheckedChange={async () => {
            setEnabling(true);
            try {
              await toggleFast(select ?? (() => {}));
            } finally {
              setEnabling(false);
            }
          }}
        />
      </label>
      {fast.variant ? (
        <div className="space-y-1 text-xs text-muted-foreground">
          <p className="break-all">Switch to {fast.variant.destination}</p>
          <p>
            {fast.variant.pair.source === "user"
              ? "Your Fast pair"
              : "Detected from provider catalog"}
          </p>
          <ModelPriceLine modelId={fast.variant.destination} />
          {delta ? <p className="tabular-nums">{delta}</p> : null}
          {reason ? (
            <button
              type="button"
              onClick={() =>
                fast.provider &&
                useSettingsDialogStore
                  .getState()
                  .openConnectionSettings(fast.provider.id)
              }
              className="min-h-9 text-left underline underline-offset-2"
            >
              {reason}
            </button>
          ) : null}
        </div>
      ) : (
        <div className="space-y-1 text-xs text-muted-foreground">
          <p>
            {fast.tier
              ? "OpenRouter Fast tier · same model"
              : "Anthropic Fast mode"}
          </p>
          {fast.tier?.endpoints.map((endpoint) => (
            <p key={endpoint.tag} className="tabular-nums">
              {endpoint.tag}: Input {tokenRate(endpoint.pricing?.rates.prompt)}{" "}
              · Output {tokenRate(endpoint.pricing?.rates.completion)} / 1M
              tokens
            </p>
          ))}
          {fast.tier ? (
            <p>
              Premium rates. OpenRouter may fall back to standard; response
              details record the served tier.
              {fast.tier.cached ? " Cached catalog." : ""}
            </p>
          ) : null}
          {tierUnavailable ? <p>Fast tier is currently unavailable.</p> : null}
        </div>
      )}
      {fast.busy ? (
        <p className="text-xs text-muted-foreground">
          Available after generation finishes.
        </p>
      ) : null}
    </div>
  );
}
