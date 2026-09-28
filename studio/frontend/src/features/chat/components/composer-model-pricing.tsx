// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { parseExternalModelId } from "../external-providers";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useExternalProvidersStore } from "../stores/external-providers-store";
import { ModelPricing } from "./model-pricing";

/** Independent of the local/cloud thinking controls, including non-reasoning models. */
export function ComposerModelPricing({
  side = "top",
}: { side?: "top" | "bottom" }) {
  const checkpoint = useChatRuntimeStore((s) => s.params.checkpoint);
  const providers = useExternalProvidersStore((s) => s.providers);
  const enabled = useExternalProvidersStore((s) => s.connectionsEnabled);
  const selection = parseExternalModelId(checkpoint);
  const provider = providers.find((p) => p.id === selection?.providerId);
  if (!enabled || !selection || provider?.providerType !== "openrouter")
    return null;
  return (
    <Popover modal={false}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="composer-pill-btn"
          aria-label="Published model rates"
        >
          <span aria-hidden="true">$</span>
          <span>Rates</span>
        </button>
      </PopoverTrigger>
      <PopoverContent
        side={side}
        align="end"
        collisionPadding={12}
        className="w-80 max-h-[var(--radix-popover-content-available-height)] overflow-y-auto rounded-2xl p-4"
        aria-label="Published model rates"
      >
        <p className="break-all text-xs text-muted-foreground">
          {selection.modelId}
        </p>
        <ModelPricing modelId={selection.modelId} />
      </PopoverContent>
    </Popover>
  );
}
