import { BaseEdge, type EdgeProps, getSmoothStepPath } from "@xyflow/react";
import { memo, type ReactElement } from "react";

export const CanvasEdge = memo(function CanvasEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  type,
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

  const nextStyle =
    type === "semantic"
      ? { ...style, strokeDasharray: "4 4" }
      : style;

  return <BaseEdge id={id} path={path} style={nextStyle} />;
});
