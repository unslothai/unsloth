// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  describePlacement,
  useChatPreferencesStore,
  useChatRuntimeStore,
} from "@/features/chat";

/** Where the resident GGUF landed. Under Auto, --fit decides, and the layer count alone hides a MoE
 *  spill, so the experts and the RAM side are shown too. */
export function LoadedPlacementRow() {
  const enabled = useChatPreferencesStore((s) => s.showGenerationDetails);
  const placement = useChatRuntimeStore((s) => s.loadedPlacement);
  if (!enabled || placement === null) return null;
  const { layers, memory } = describePlacement(placement);
  if (layers === null && memory === null) return null;
  return (
    <div
      data-slot="loaded-placement"
      className="flex flex-col gap-1 border-b border-border pb-5"
    >
      <span className="text-ui-13 font-medium leading-[1.25] tracking-nav text-foreground">
        Loaded placement
      </span>
      {layers !== null && (
        <span className="text-ui-12 leading-snug text-muted-foreground">{layers}</span>
      )}
      {memory !== null && (
        <span className="font-mono text-ui-12 leading-snug text-muted-foreground tabular-nums">
          {memory}
        </span>
      )}
    </div>
  );
}
