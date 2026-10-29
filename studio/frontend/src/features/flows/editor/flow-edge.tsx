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
import { WireDelete, WireHalo, useWireHover } from "./wire-delete";
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
  deletable,
  data,
}: EdgeProps<RfEdge>) {
  const color =
    useCatalogStore((s) => s.catalog?.port_colors[data?.portType ?? "any"]) ??
    "#94a3b8";
  const { hover, enter, leave } = useWireHover();
  const flow = data?.kind === "flow";
  const [path, labelX, labelY] = flow
    ? getSmoothStepPath({
        sourceX,
        sourceY,
        targetX,
        targetY,
        sourcePosition,
        targetPosition,
        borderRadius: 14,
      })
    : getBezierPath({
        sourceX,
        sourceY,
        targetX,
        targetY,
        sourcePosition,
        targetPosition,
      });
  const style = flow
    ? {
        stroke: selected ? "#ffffff" : "#e4e4e7",
        strokeWidth: selected ? 3 : 2.25,
        strokeDasharray: "8 5",
        opacity: selected ? 1 : 0.8,
      }
    : {
        stroke: color,
        strokeWidth: selected ? 2.5 : 1.75,
        opacity: selected || hover ? 1 : 0.75,
      };
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: hover only reveals the × button
    <g onMouseEnter={enter} onMouseLeave={leave}>
      {selected && <WireHalo path={path} />}
      <BaseEdge id={id} path={path} style={style} />
      <WireDelete
        id={id}
        x={labelX}
        y={labelY}
        show={deletable !== false && (hover || !!selected)}
        onEnter={enter}
        onLeave={leave}
      />
    </g>
  );
}

const CurvedEdge = memo(CurvedEdgeImpl);

function FlowEdgeViewImpl(props: EdgeProps<RfEdge>) {
  const bus = useUiStore((s) => s.wireStyle === "bus");
  return bus ? <BusEdge {...props} /> : <CurvedEdge {...props} />;
}

export const FlowEdgeView = memo(FlowEdgeViewImpl);
