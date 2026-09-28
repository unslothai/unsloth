// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useSyncExternalStore } from "react";
import { useShallow } from "zustand/react/shallow";
import { ZapIcon } from "lucide-react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { currentFast } from "../lib/fast-controls";
import { refreshOpenRouterFastTier } from "../lib/openrouter-fast-tier";
import { modelCatalogVersion, subscribeModelCatalog } from "../model-catalog";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useExternalProvidersStore } from "../stores/external-providers-store";
import { FastControl } from "./fast-control";
import { ModelPricing } from "./model-pricing";

/** Works alongside the existing composer and any local/cloud thinking control. */
export function ComposerFastControl({
  side = "top",
}: { side?: "top" | "bottom" }) {
  useChatRuntimeStore(
    useShallow((s) => [
      s.params.checkpoint,
      s.params.fastMode,
      s.runningByThreadId,
      s.modelLoading,
    ]),
  );
  useExternalProvidersStore(
    useShallow((s) => [s.providers, s.connectionsEnabled]),
  );
  useSyncExternalStore(subscribeModelCatalog, modelCatalogVersion);
  const fast = currentFast();
  const model = fast.selection?.modelId;
  const type = fast.provider?.providerType;
  useEffect(() => {
    if (type === "openrouter" && model) void refreshOpenRouterFastTier(model);
  }, [type, model]);
  if (!fast.connected || (!fast.native && !fast.variant)) return null;
  const tierPrice = fast.isFast ? fast.tier?.endpoints[0]?.pricing : undefined;
  return (
    <Popover modal={false}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="composer-pill-btn"
          data-active={fast.isFast ? "true" : "false"}
          aria-label={fast.isFast ? "Fast mode on" : "Fast mode off"}
        >
          <ZapIcon
            className="size-4 shrink-0"
            aria-hidden="true"
            fill={fast.isFast ? "currentColor" : "none"}
          />
          <span>Fast{fast.isFast ? " · On" : ""}</span>
        </button>
      </PopoverTrigger>
      <PopoverContent
        side={side}
        align="end"
        collisionPadding={12}
        className="w-80 max-h-[var(--radix-popover-content-available-height)] overflow-y-auto rounded-2xl p-4"
        aria-label="Fast settings"
      >
        <FastControl />
        {type === "openrouter" && model ? (
          <ModelPricing
            modelId={model}
            label={
              fast.isFast && fast.tier ? "Published Fast rates" : undefined
            }
            pricingOverride={
              fast.isFast && fast.tier
                ? tierPrice
                  ? {
                      ...tierPrice,
                      fetchedAt: fast.tier.fetchedAt,
                      cached: fast.tier.cached,
                    }
                  : null
                : undefined
            }
            source={fast.isFast && fast.tier ? fast.tier.source : undefined}
          />
        ) : null}
      </PopoverContent>
    </Popover>
  );
}
