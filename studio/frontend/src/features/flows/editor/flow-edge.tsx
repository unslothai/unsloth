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
import { portInk } from "./theme-colors";
import { WireDelete, WireHalo, useWireHover, wireStroke } from "./wire-delete";
import { wireDeleteShown } from "./wire-live";
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
  const style = wireStroke(flow, selected, hover, portInk(color));
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: hover only reveals the × button
    <g onMouseEnter={enter} onMouseLeave={leave}>
      {selected && <WireHalo path={path} />}
      <BaseEdge id={id} path={path} style={style} />
      <WireDelete
        id={id}
        x={labelX}
        y={labelY}
        show={wireDeleteShown(deletable, hover, selected)}
        onEnter={enter}
        onLeave={leave}
      />
    </g>
  );
}

const CurvedEdge = memo(CurvedEdgeImpl);

function FlowEdgeViewImpl(props: EdgeProps<RfEdge>) {
  const global = useUiStore((s) => s.wireStyle);
  return (props.data?.wireStyle ?? global) === "bus" ? <BusEdge {...props} /> : <CurvedEdge {...props} />;
}

export const FlowEdgeView = memo(FlowEdgeViewImpl);
