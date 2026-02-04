import { BaseEdge, type EdgeProps, getSmoothStepPath } from "@xyflow/react";
import { memo } from "react";

export const CanvasEdge = memo(function CanvasEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
}: EdgeProps): JSX.Element {
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

  return <BaseEdge id={id} path={path} style={style} />;
});
