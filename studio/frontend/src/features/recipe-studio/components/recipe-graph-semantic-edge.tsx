import { BaseEdge, type EdgeProps, getSmoothStepPath } from "@xyflow/react";
import { memo, type ReactElement } from "react";

export const RecipeGraphSemanticEdge = memo(function RecipeGraphSemanticEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  markerEnd,
  selected,
}: EdgeProps): ReactElement {
  const [path] = getSmoothStepPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
    borderRadius: 0,
    offset: 16,
  });

  return (
    <BaseEdge
      id={id}
      path={path}
      markerEnd={markerEnd}
      style={{
        strokeDasharray: selected ? "7 5" : "6 5",
        strokeWidth: selected ? 2.3 : 1.8,
        stroke: selected
          ? "hsl(var(--primary) / 0.9)"
          : "hsl(var(--foreground) / 0.38)",
        opacity: selected ? 1 : 0.92,
        ...style,
      }}
    />
  );
});
