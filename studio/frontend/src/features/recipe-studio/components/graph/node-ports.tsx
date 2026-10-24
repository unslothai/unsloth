// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { cn } from "@/lib/utils";
import { Handle, Position } from "@xyflow/react";
import type { ReactElement } from "react";
import { HANDLE_IDS } from "../../utils/handles";
import {
  RECIPE_PORT_KINDS,
  type RecipePort,
  type RecipePortKind,
} from "../../utils/ports";

export const PORT_KIND_DOT: Record<RecipePortKind, string> = {
  data: "!border-slate-500 !bg-slate-300 dark:!bg-slate-500",
  model: "!border-orange-500 !bg-orange-300 dark:!bg-orange-500",
  provider: "!border-amber-500 !bg-amber-200 dark:!bg-amber-500",
  tool: "!border-cyan-500 !bg-cyan-200 dark:!bg-cyan-500",
};

const PORT_HANDLE_CLASS =
  "!size-2.5 rounded-full !border shadow-[0_0_0_2px_var(--card)] transition-transform hover:scale-125";

// Filled by the canvas itself, never by a drag.
const UNCONNECTABLE = new Set<string>([
  HANDLE_IDS.promptIn,
  HANDLE_IDS.llmInputOut,
  HANDLE_IDS.rerouteIn,
]);

export function PortHandle({ port }: { port: RecipePort }): ReactElement {
  const connectable = !UNCONNECTABLE.has(port.id);
  return (
    <Handle
      id={port.id}
      type={port.side === "left" ? "target" : "source"}
      position={port.side === "left" ? Position.Left : Position.Right}
      isConnectable={connectable}
      isConnectableStart={connectable}
      title={port.label || undefined}
      className={cn(PORT_HANDLE_CLASS, PORT_KIND_DOT[port.kind])}
    />
  );
}

/** A card's inputs down the left and outputs down the right, one labelled row each. */
export function NodePorts({
  ports,
  className,
}: {
  ports: RecipePort[] | undefined;
  className?: string;
}): ReactElement | null {
  if (!ports?.length) {
    return null;
  }
  const inputs = ports.filter((port) => port.side === "left");
  const outputs = ports.filter((port) => port.side === "right");
  const rows = Math.max(inputs.length, outputs.length);
  return (
    <div className={cn("border-b border-border/50 py-1", className)}>
      {Array.from({ length: rows }, (_, row) => {
        const input = inputs[row];
        const output = outputs[row];
        return (
          <div
            key={`${input?.id ?? ""}:${output?.id ?? ""}`}
            className="relative flex h-6 items-center justify-between gap-2 px-3 text-xs"
          >
            <span className="min-w-0 truncate text-muted-foreground">
              {input?.label}
            </span>
            <span className="min-w-0 truncate text-right font-medium text-foreground/80">
              {output?.label}
            </span>
            {input && <PortHandle port={input} />}
            {output && <PortHandle port={output} />}
          </div>
        );
      })}
    </div>
  );
}

/** Which dot colour means what; sits next to Auto layout. */
export function PortLegend(): ReactElement {
  return (
    <div className="flex items-center gap-3 rounded-md border border-border/60 bg-background/90 px-2.5 py-1.5 text-xs text-muted-foreground">
      {RECIPE_PORT_KINDS.map(({ kind, label }) => (
        <span key={kind} className="flex items-center gap-1.5">
          <span
            className={cn("inline-block size-2.5 rounded-full border", PORT_KIND_DOT[kind])}
          />
          {label}
        </span>
      ))}
    </div>
  );
}
