import { Position } from "@xyflow/react";
import type { LayoutDirection } from "../types";

export const NODE_HANDLE_CLASS =
  "pointer-events-auto !size-2.5 !border-border/80 !bg-muted shadow-sm hover:!border-primary/70 hover:!bg-primary/20";

export const AUX_HANDLE_CLASS =
  "!size-2 !border-border/80 !bg-muted/80 shadow-sm";

export type NodeHandleLayout = {
  isTopBottom: boolean;
  dataInPosition: Position;
  dataOutPosition: Position;
  semanticInPosition: Position;
  semanticOutPosition: Position;
};

export function getNodeHandleLayout(
  direction: LayoutDirection,
): NodeHandleLayout {
  const isTopBottom = direction === "TB";
  return {
    isTopBottom,
    dataInPosition: isTopBottom ? Position.Top : Position.Left,
    dataOutPosition: isTopBottom ? Position.Bottom : Position.Right,
    semanticInPosition: isTopBottom ? Position.Left : Position.Top,
    semanticOutPosition: isTopBottom ? Position.Right : Position.Bottom,
  };
}

export function getAuxSourceHandlePosition(
  direction: LayoutDirection,
): Position {
  return direction === "TB" ? Position.Bottom : Position.Right;
}

