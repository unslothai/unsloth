// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { BaseEdge, type EdgeProps, useStore } from "@xyflow/react";
import { type ReactElement, memo, useMemo } from "react";
import { collectEdgeObstacles } from "../utils/graph/edge-obstacles";
import {
  laneOffsetFromId,
  routeOrthogonalPath,
} from "../utils/graph/orthogonal-router";

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
  const nodeLookup = useStore((store) => store.nodeLookup);
  const obstacles = useMemo(
    () => collectEdgeObstacles(nodeLookup, source, target),
    [nodeLookup, source, target],
  );
  const path = routeOrthogonalPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    obstacles,
    laneOffset: laneOffsetFromId(id),
  });

  return (
    <BaseEdge
      id={id}
      path={path}
      markerEnd={markerEnd}
      style={{
        strokeDasharray: isActive ? "8 6" : selected ? "7 5" : "6 5",
        strokeWidth: isActive ? 2.4 : selected ? 2.3 : 1.8,
        stroke:
          isActive || selected ? "var(--primary)" : "var(--muted-foreground)",
        opacity: isActive ? 1 : selected ? 0.95 : 0.62,
        ...style,
      }}
    />
  );
});
