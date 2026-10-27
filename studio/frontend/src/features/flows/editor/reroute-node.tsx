// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Handle, type NodeProps, Position } from "@xyflow/react";
import { memo } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import type { RfNode } from "./derive-rf";
import {
  REROUTE_IN_HANDLE,
  REROUTE_OUT_HANDLE,
  REROUTE_SIZE,
  reroutePort,
} from "./reroutes";

/** A waypoint for steering a wire. It carries whatever type is wired into it. */
function RerouteNodeImpl({ id, selected }: NodeProps<RfNode>) {
  const type = useDocStore((s) =>
    s.graph.reroutes[id] ? reroutePort(s.graph, id).type : "any",
  );
  const color =
    useCatalogStore((s) => s.catalog?.port_colors[type]) ?? "#94a3b8";
  return (
    <div
      className={`relative rounded-full border-2 bg-zinc-900 shadow-sm shadow-black/40 ${selected ? "ring-2 ring-white/60" : ""}`}
      style={{ ...REROUTE_SIZE, borderColor: color }}
      title="Reroute: drag to steer the wire, drop on a wire to split it"
      data-testid="flows-reroute"
      data-reroute-id={id}
    >
      <Handle
        type="target"
        id={REROUTE_IN_HANDLE}
        position={Position.Left}
        className="!h-2 !w-2 !min-w-0 !border-0"
        style={{ background: color, left: -4 }}
      />
      <Handle
        type="source"
        id={REROUTE_OUT_HANDLE}
        position={Position.Right}
        className="!h-2 !w-2 !min-w-0 !border-0"
        style={{ background: color, right: -4 }}
      />
    </div>
  );
}

export const RerouteNode = memo(RerouteNodeImpl);
