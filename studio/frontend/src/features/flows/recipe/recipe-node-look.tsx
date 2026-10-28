// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A recipe column on the shared Flows node card: the block's icon and tone, the column name as title, its kind
// as subtitle, and two or three key facts. Everything else is in the inspector.

import { Badge } from "@/components/ui/badge";
import { Handle, Position } from "@xyflow/react";
import { HEADER_MID, NodeFieldRows } from "../editor/node-card";
import type { NodeLook } from "../editor/node-looks";
import type { FlowNode } from "../types";
import { columnOf } from "./recipe-columns";
import { columnBlock, columnFacts } from "./recipe-kinds";
import { useColumnSamples } from "./recipe-panels";

function ColumnBody({ node }: { node: FlowNode }) {
  const { chips, fields } = columnFacts(columnOf(node));
  const [sample] = useColumnSamples(node, 1);
  return (
    <>
      {/* where a model preset or tool profile card's dashed link lands */}
      <Handle
        type="target"
        id="setup-in"
        position={Position.Left}
        isConnectable={false}
        className="!size-2 !border-0 !bg-transparent"
        style={{ top: HEADER_MID }}
      />
      {chips.length > 0 && (
        <div className="flow-detail flex flex-wrap gap-1 px-3">
          {chips.map((chip) => (
            <Badge
              key={chip}
              variant="secondary"
              className="h-5 max-w-full px-1.5 text-ui-10"
            >
              <span className="truncate">{chip}</span>
            </Badge>
          ))}
        </div>
      )}
      <NodeFieldRows rows={fields} />
      {sample && (
        <p className="flow-detail mx-3 line-clamp-2 whitespace-pre-wrap rounded-md border border-border/60 bg-muted/40 px-2 py-1 text-ui-11 leading-snug text-foreground">
          {sample}
        </p>
      )}
    </>
  );
}

function ColumnHeaderExtra({ node }: { node: FlowNode }) {
  if (columnOf(node).drop !== true) return null;
  return (
    <Badge
      variant="outline"
      className="h-5 px-1.5 text-ui-10 text-muted-foreground"
      title="Used while generating, left out of the final dataset"
    >
      Helper
    </Badge>
  );
}

export const RECIPE_COLUMN_LOOK: NodeLook = {
  icon: (node) => columnBlock(columnOf(node)).icon,
  tone: (node) => columnBlock(columnOf(node)).tone,
  subtitle: (node) => columnBlock(columnOf(node)).label,
  HeaderExtra: ColumnHeaderExtra,
  Body: ColumnBody,
};
