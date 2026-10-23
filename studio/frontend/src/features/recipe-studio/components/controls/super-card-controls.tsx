// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  ArrowLeft01Icon,
  Layers01Icon,
  UngroupLayersIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  NodeToolbar,
  Position,
  type ReactFlowState,
  ViewportPortal,
  useStore,
} from "@xyflow/react";
import { type ReactElement, useCallback } from "react";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import type { SuperCard } from "../../utils/graph/super-cards";

type SuperCardBreadcrumbBarProps = {
  card: SuperCard;
  recipeName: string;
  locked: boolean;
  onExit: () => void;
  onRename: (id: string, title: string) => void;
  onUngroup: (id: string) => void;
};

// Sits above the canvas, not on it, so it never covers a block.
export function SuperCardBreadcrumbBar({
  card,
  recipeName,
  locked,
  onExit,
  onRename,
  onUngroup,
}: SuperCardBreadcrumbBarProps): ReactElement {
  return (
    <div className="flex h-12 shrink-0 items-center justify-between gap-3 border-b bg-muted/30 px-3">
      <Breadcrumb className="min-w-0">
        <BreadcrumbList className="flex-nowrap">
          <BreadcrumbItem>
            <BreadcrumbLink asChild={true}>
              <button
                type="button"
                onClick={onExit}
                className="flex items-center gap-1.5"
              >
                <HugeiconsIcon icon={ArrowLeft01Icon} className="size-4" />
                <span className="max-w-48 truncate">
                  {recipeName.trim() || "Recipe"}
                </span>
              </button>
            </BreadcrumbLink>
          </BreadcrumbItem>
          <BreadcrumbSeparator />
          <BreadcrumbItem className="gap-2">
            <HugeiconsIcon
              icon={Layers01Icon}
              className="size-4 text-primary"
            />
            <Input
              aria-label="Super card name"
              value={card.title}
              disabled={locked}
              onChange={(event) => onRename(card.id, event.target.value)}
              className="h-8 w-52 text-sm font-medium text-foreground"
            />
            <span className="whitespace-nowrap text-xs text-muted-foreground">
              {card.nodeIds.length} blocks
            </span>
          </BreadcrumbItem>
        </BreadcrumbList>
      </Breadcrumb>
      <Button
        type="button"
        size="sm"
        variant="outline"
        className="corner-squircle"
        disabled={locked}
        onClick={() => onUngroup(card.id)}
      >
        <HugeiconsIcon icon={UngroupLayersIcon} className="size-4" />
        Ungroup
      </Button>
    </div>
  );
}

type SuperCardCombineToolbarProps = {
  nodeIds: string[];
  locked: boolean;
  onCombine: () => void;
};

type OutlineRect = { id: string; x: number; y: number; w: number; h: number };

// Dashed outlines preview which blocks the card will take.
function SelectionOutlines({ nodeIds }: { nodeIds: string[] }): ReactElement {
  const key = useStore(
    useCallback(
      (state: ReactFlowState) =>
        JSON.stringify(
          nodeIds.flatMap((id): OutlineRect[] => {
            const node = state.nodeLookup.get(id);
            if (!node) {
              return [];
            }
            const { x, y } = node.internals.positionAbsolute;
            return [
              {
                id,
                x,
                y,
                w: node.measured.width ?? 0,
                h: node.measured.height ?? 0,
              },
            ];
          }),
        ),
      [nodeIds],
    ),
  );
  const rects = JSON.parse(key) as OutlineRect[];
  return (
    <ViewportPortal>
      {rects.map((rect) => (
        <div
          key={rect.id}
          className="pointer-events-none absolute rounded-2xl border-2 border-dashed border-primary/60"
          style={{
            transform: `translate(${rect.x - 7}px, ${rect.y - 7}px)`,
            width: rect.w + 14,
            height: rect.h + 14,
          }}
        />
      ))}
    </ViewportPortal>
  );
}

// Floats above the selected blocks once two or more are picked.
export function SuperCardCombineToolbar({
  nodeIds,
  locked,
  onCombine,
}: SuperCardCombineToolbarProps): ReactElement | null {
  if (nodeIds.length < 2) {
    return null;
  }
  return (
    <>
      <SelectionOutlines nodeIds={nodeIds} />
      <NodeToolbar
        nodeId={nodeIds}
        isVisible={true}
        position={Position.Top}
        offset={14}
      >
        <Button
          type="button"
          size="sm"
          className="corner-squircle shadow-md"
          disabled={locked}
          onClick={onCombine}
        >
          <HugeiconsIcon icon={Layers01Icon} className="size-4" />
          Combine {nodeIds.length} blocks into a super card
        </Button>
      </NodeToolbar>
    </>
  );
}
