// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useStore } from "@xyflow/react";
import { createPortal } from "react-dom";
import type { NodeSpec } from "../types";
import { nodeIcon } from "./icons";
import { NodeIconTile } from "./node-card";
import { CATEGORY_TONES } from "./node-tones";
import { GRAB } from "./placement";

/** The card-to-be under the pointer, at canvas zoom, exactly where placementPosition puts it.
 *  Portaled to body: a transformed ancestor would otherwise shift a `fixed` box. */
export function PlacementGhost({
  spec,
  x,
  y,
}: { spec: NodeSpec; x: number; y: number }) {
  const zoom = useStore((s) => s.transform[2]);
  return createPortal(
    <div
      className="corner-squircle pointer-events-none fixed z-[1000] flex w-65 origin-top-left items-center gap-2.5 rounded-xl border border-border bg-card p-3 opacity-80 shadow-md"
      style={{
        left: x - GRAB[0] * zoom,
        top: y - GRAB[1] * zoom,
        transform: `scale(${zoom})`,
      }}
      data-testid="flows-placement-ghost"
    >
      <NodeIconTile
        icon={nodeIcon(spec.icon)}
        tone={CATEGORY_TONES[spec.category]}
      />
      <span className="truncate font-semibold text-sm text-foreground">
        {spec.title}
      </span>
    </div>,
    document.body,
  );
}
