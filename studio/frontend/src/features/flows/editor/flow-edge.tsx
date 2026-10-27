// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  BaseEdge,
  type EdgeProps,
  getBezierPath,
  getSmoothStepPath,
} from "@xyflow/react";
import { memo } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useUiStore } from "../stores/ui-store";
import type { RfEdge } from "./derive-rf";
import { BusEdge } from "./wire-routing";

function CurvedEdgeImpl({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  selected,
  data,
}: EdgeProps<RfEdge>) {
  const color =
    useCatalogStore((s) => s.catalog?.port_colors[data?.portType ?? "any"]) ??
    "#94a3b8";
  if (data?.kind === "flow") {
    const [path] = getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      borderRadius: 14,
    });
    return (
      <BaseEdge
        id={id}
        path={path}
        style={{
          stroke: selected ? "#ffffff" : "#e4e4e7",
          strokeWidth: selected ? 3 : 2.25,
          strokeDasharray: "8 5",
          opacity: selected ? 1 : 0.8,
        }}
      />
    );
  }
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });
  return (
    <BaseEdge
      id={id}
      path={path}
      style={{
        stroke: color,
        strokeWidth: selected ? 3 : 1.75,
        opacity: selected ? 1 : 0.75,
      }}
    />
  );
}

const CurvedEdge = memo(CurvedEdgeImpl);

function FlowEdgeViewImpl(props: EdgeProps<RfEdge>) {
  const bus = useUiStore((s) => s.wireStyle === "bus");
  return bus ? <BusEdge {...props} /> : <CurvedEdge {...props} />;
}

export const FlowEdgeView = memo(FlowEdgeViewImpl);
