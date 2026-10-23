// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ReactElement } from "react";
import type { Edge, EdgeProps } from "@xyflow/react";
import { RoutedWire } from "../graph/wire-routing";

export type DataEdge = Edge<{
  path?: "auto" | "bezier" | "smoothstep" | "step" | "straight";
  active?: boolean;
}>;

export function DataEdge({
  data = { path: "auto" },
  id,
  source,
  target,
  markerEnd,
  selected,
  sourcePosition,
  sourceX,
  sourceY,
  style,
  targetPosition,
  targetX,
  targetY,
}: EdgeProps<DataEdge>): ReactElement {
  const isActive = Boolean(data.active);
  const edgeStyle = {
    stroke: isActive || selected ? "var(--primary)" : "var(--muted-foreground)",
    strokeWidth: isActive ? 2.6 : selected ? 2.6 : 2.1,
    opacity: isActive ? 1 : selected ? 0.96 : 0.7,
    strokeDasharray: isActive ? "8 6" : undefined,
    ...style,
  };

  return (
    <RoutedWire
      id={id}
      source={source}
      target={target}
      data={data}
      sourceX={sourceX}
      sourceY={sourceY}
      sourcePosition={sourcePosition}
      targetX={targetX}
      targetY={targetY}
      targetPosition={targetPosition}
      markerEnd={markerEnd}
      style={edgeStyle}
    />
  );
}
