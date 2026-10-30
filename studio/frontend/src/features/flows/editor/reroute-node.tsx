// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Handle, type NodeProps, Position } from "@xyflow/react";
import { memo } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import type { RerouteDir } from "../stores/reroute-graph";
import type { RfNode } from "./derive-rf";
import {
  REROUTE_IN_HANDLE,
  REROUTE_OUT_HANDLE,
  reroutePort,
  rerouteSize,
} from "./reroutes";

const HANDLE_SIDES: Record<RerouteDir, [Position, Position]> = {
  right: [Position.Left, Position.Right],
  down: [Position.Top, Position.Bottom],
  up: [Position.Bottom, Position.Top],
};
import { portInk } from "./theme-colors";

/** A waypoint for steering a wire. It carries whatever type is wired into it. */
function RerouteNodeImpl({ id, selected }: NodeProps<RfNode>) {
  const type = useDocStore((s) =>
    s.graph.reroutes[id] ? reroutePort(s.graph, id).type : "any",
  );
  const dir = useDocStore((s) => s.graph.reroutes[id]?.dir ?? "right");
  const color =
    useCatalogStore((s) => s.catalog?.port_colors[type]) ?? "#94a3b8";
  const [inSide, outSide] = HANDLE_SIDES[dir];
  return (
    <div
      className={`relative rounded-full border-2 bg-card shadow-sm ${selected ? "ring-2 ring-ring" : ""}`}
      style={{ ...rerouteSize(dir), borderColor: portInk(color) }}
      data-dir={dir}
      title="Reroute: drag to steer the wire, drop on a wire to split it, R turns it"
      data-testid="flows-reroute"
      data-reroute-id={id}
    >
      <Handle
        type="target"
        id={REROUTE_IN_HANDLE}
        position={inSide}
        className="!h-2 !w-2 !min-w-0 !border-0"
        style={{ background: color, [inSide]: -4 }}
      />
      <Handle
        type="source"
        id={REROUTE_OUT_HANDLE}
        position={outSide}
        className="!h-2 !w-2 !min-w-0 !border-0"
        style={{ background: color, [outSide]: -4 }}
      />
    </div>
  );
}

export const RerouteNode = memo(RerouteNodeImpl);
