// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BaseEdge,
  EdgeLabelRenderer,
  type EdgeProps,
  useStore,
} from "@xyflow/react";
import { type ReactElement, memo, useMemo } from "react";
import { collectObstacles } from "../utils/graph/edge-obstacles";
import {
  laneOffsetFromId,
  routeOrthogonalPath,
} from "../utils/graph/orthogonal-router";
import { WireLabel } from "./wire-label";

export const RecipeGraphSemanticEdge = memo(function RecipeGraphSemanticEdge({
  id,
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
  const edgeData = data as { active?: boolean; label?: string } | undefined;
  const isActive = Boolean(edgeData?.active);
  const label = edgeData?.label;
  const nodeLookup = useStore((store) => store.nodeLookup);
  const obstacles = useMemo(() => collectObstacles(nodeLookup), [nodeLookup]);
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
    <>
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
      {label ? (
        <EdgeLabelRenderer>
          <WireLabel
            label={label}
            x={sourceX}
            y={sourceY}
            active={isActive || Boolean(selected)}
          />
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
});
