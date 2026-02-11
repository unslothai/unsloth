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
        strokeDasharray: "4 4",
        strokeWidth: 1.5,
        stroke: "var(--muted-foreground)",
        ...style,
      }}
    />
  );
});
