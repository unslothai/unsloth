// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { EdgeProps } from "@xyflow/react";
import { memo, type ReactElement } from "react";
import { RoutedWire } from "./graph/wire-routing";

export const RecipeGraphSemanticEdge = memo(function RecipeGraphSemanticEdge({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  markerEnd,
  selected,
  data,
}: EdgeProps): ReactElement {
  const isActive = Boolean((data as { active?: boolean } | undefined)?.active);

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
      style={{
        strokeDasharray: isActive ? "8 6" : selected ? "7 5" : "6 5",
        strokeWidth: isActive ? 2.4 : selected ? 2.3 : 1.8,
        stroke: isActive || selected ? "var(--primary)" : "var(--muted-foreground)",
        opacity: isActive ? 1 : selected ? 0.95 : 0.62,
        ...style,
      }}
    />
  );
});
