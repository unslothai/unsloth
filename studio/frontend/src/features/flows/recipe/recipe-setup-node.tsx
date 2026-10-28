// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { RECIPE_STUDIO_WARNING_BADGE_TONE } from "@/features/recipe-studio/utils/ui-tones";
import { cn } from "@/lib/utils";
import {
  BaseEdge,
  type EdgeProps,
  Handle,
  type NodeProps,
  Position,
  getBezierPath,
} from "@xyflow/react";
import { memo } from "react";
import type { RfNode } from "../editor/derive-rf";
import {
  NodeCard,
  NodeCardHeader,
  NodeFieldRows,
} from "../editor/node-card";
import { IDLE_RUNTIME } from "../stores/runtime-store";
import { setupLook } from "./recipe-kinds";
import type { SetupCard } from "./recipe-setup";

export const SETUP_IN = "setup-in";
export const SETUP_OUT = "setup-out";

export type SetupData = { card: SetupCard; seedType?: string };

const HIDDEN = "!size-2 !border-0 !bg-transparent";

function RecipeSetupNodeImpl({ data, selected }: NodeProps<RfNode>) {
  const { card, seedType } = data as unknown as SetupData;
  const { icon, tone } = setupLook(card.kind, seedType);
  return (
    <NodeCard
      runtime={IDLE_RUNTIME}
      className={cn("w-60", selected && "ring-2 ring-ring/40")}
      data-testid="recipe-setup-card"
      data-setup-kind={card.kind}
    >
      <Handle
        type="target"
        id={SETUP_IN}
        position={Position.Left}
        isConnectable={false}
        className={HIDDEN}
      />
      <NodeCardHeader
        icon={icon}
        tone={tone}
        title={card.title}
        subtitle={card.subtitle}
      >
        {card.warning && (
          <Badge
            variant="outline"
            className={cn("h-5 px-1.5 text-ui-10", RECIPE_STUDIO_WARNING_BADGE_TONE)}
          >
            {card.kind === "seed" ? "Needs file" : "Needs server"}
          </Badge>
        )}
      </NodeCardHeader>
      <div className="py-2">
        <NodeFieldRows rows={card.fields.slice(0, 3)} />
      </div>
      <Handle
        type="source"
        id={SETUP_OUT}
        position={Position.Right}
        isConnectable={false}
        className={HIDDEN}
      />
    </NodeCard>
  );
}

export const RecipeSetupNode = memo(RecipeSetupNodeImpl);

/** Which preset or tool profile a column uses: a quiet dashed line, as on Recipe Studio's canvas. */
function RecipeSetupEdgeImpl({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
}: EdgeProps) {
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
        stroke: "var(--muted-foreground)",
        strokeWidth: 1.25,
        strokeDasharray: "5 4",
        opacity: 0.55,
      }}
    />
  );
}

export const RecipeSetupEdge = memo(RecipeSetupEdgeImpl);
