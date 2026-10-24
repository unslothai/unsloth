// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { NodeProps } from "@xyflow/react";
import { memo, type ReactElement } from "react";
import { REROUTE_PORTS } from "../../utils/ports";
import { PortHandle } from "./node-ports";

/** A waypoint the user drags to steer a wire; it passes the connection straight through. */
export const RerouteNode = memo(function RerouteNode({ selected }: NodeProps): ReactElement {
  return (
    <div
      title="Reroute: drop on a wire to split it, drag to steer it"
      className={
        selected
          ? "relative size-full rounded-full border border-ring-strong bg-card shadow-md"
          : "relative size-full rounded-full border border-border bg-card shadow-sm hover:border-ring"
      }
    >
      {REROUTE_PORTS.map((port) => (
        <PortHandle key={port.id} port={port} />
      ))}
    </div>
  );
});
