// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useStore } from "@xyflow/react";
import { createPortal } from "react-dom";
import type { RerouteDir } from "../stores/reroute-graph";
import type { NodeSpec } from "../types";
import { nodeIcon } from "./icons";
import { NodeIconTile } from "./node-card";
import { CATEGORY_TONES } from "./node-tones";
import { GRAB, PLACE_GROUP, PLACE_SHAPES } from "./placement";

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

/** Outline of the group, note or reroute an armed toolbar button places, held at its grab point. */
export function ShapeGhost({
  type,
  x,
  y,
  dir = "right",
}: { type: string; x: number; y: number; dir?: RerouteDir }) {
  const zoom = useStore((s) => s.transform[2]);
  const base = PLACE_SHAPES[type];
  if (!base) return null;
  const reroute = type === "reroute";
  const shape =
    reroute && dir !== "right"
      ? {
          size: [base.size[1], base.size[0]],
          grab: [base.grab[1], base.grab[0]],
        }
      : base;
  return createPortal(
    <div
      className={`pointer-events-none fixed z-[1000] border-2 border-dashed border-primary/60 bg-primary/5 opacity-80 ${reroute ? "rounded-full" : "rounded-lg"}`}
      style={{
        left: x - shape.grab[0] * zoom,
        top: y - shape.grab[1] * zoom,
        width: shape.size[0] * zoom,
        height: shape.size[1] * zoom,
      }}
      data-testid="flows-placement-ghost"
      data-dir={reroute ? dir : undefined}
    >
      {!reroute && (
        <span className="absolute top-1 left-2 font-medium text-muted-foreground text-xs">
          {type === PLACE_GROUP ? "Group" : "Note"}
        </span>
      )}
    </div>,
    document.body,
  );
}
