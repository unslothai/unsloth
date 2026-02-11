import { useMemo, type ReactElement } from "react";
import {
  BaseEdge,
  EdgeLabelRenderer,
  getBezierPath,
  getSmoothStepPath,
  getStraightPath,
  Position,
  useStore,
  type Edge,
  type EdgeProps,
  type Node,
} from "@xyflow/react";

export type DataEdge<T extends Node = Node> = Edge<{
  key?: keyof T["data"];
  path?: "auto" | "bezier" | "smoothstep" | "step" | "straight";
}>;

export function DataEdge({
  data = { path: "auto" },
  id,
  markerEnd,
  source,
  sourcePosition,
  sourceX,
  sourceY,
  style,
  targetPosition,
  targetX,
  targetY,
}: EdgeProps<DataEdge>): ReactElement {
  const nodeData = useStore((state) => state.nodeLookup.get(source)?.data);
  const resolvedPathType = resolvePathType({
    type: data.path ?? "auto",
    sourceX,
    sourceY,
    targetX,
    targetY,
    sourcePosition,
    targetPosition,
  });
  const [edgePath, labelX, labelY] = getPath({
    type: resolvedPathType,
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  const label = useMemo(() => {
    if (data.key && nodeData) {
      const value = nodeData[data.key];
      if (typeof value === "string" || typeof value === "number") {
        return value;
      }
      if (typeof value === "object") {
        return JSON.stringify(value);
      }
    }
    return "";
  }, [data, nodeData]);

  const transform = `translate(${labelX}px,${labelY}px) translate(-50%, -50%)`;

  return (
    <>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style} />
      {data.key && (
        <EdgeLabelRenderer>
          <div
            className="absolute rounded border bg-background px-1 text-foreground"
            style={{ transform }}
          >
            <pre className="text-xs">{label}</pre>
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

function getPath({
  type,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
}: {
  type: "bezier" | "smoothstep" | "step" | "straight";
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  sourcePosition: Position;
  targetPosition: Position;
}): [string, number, number, ...number[]] {
  if (type === "bezier") {
    return getBezierPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
    });
  }
  if (type === "smoothstep") {
    return getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
    });
  }
  if (type === "step") {
    return getSmoothStepPath({
      sourceX,
      sourceY,
      targetX,
      targetY,
      sourcePosition,
      targetPosition,
      borderRadius: 0,
    });
  }
  return getStraightPath({
    sourceX,
    sourceY,
    targetX,
    targetY,
  });
}

function resolvePathType({
  type,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
}: {
  type: "auto" | "bezier" | "smoothstep" | "step" | "straight";
  sourceX: number;
  sourceY: number;
  targetX: number;
  targetY: number;
  sourcePosition: Position;
  targetPosition: Position;
}): "bezier" | "smoothstep" | "step" | "straight" {
  if (type !== "auto") {
    return type;
  }

  const isVerticalFlow =
    (sourcePosition === Position.Bottom && targetPosition === Position.Top) ||
    (sourcePosition === Position.Top && targetPosition === Position.Bottom);
  if (isVerticalFlow && Math.abs(sourceX - targetX) <= 18) {
    return "straight";
  }

  const isHorizontalFlow =
    (sourcePosition === Position.Right && targetPosition === Position.Left) ||
    (sourcePosition === Position.Left && targetPosition === Position.Right);
  if (isHorizontalFlow && Math.abs(sourceY - targetY) <= 18) {
    return "straight";
  }

  const deltaX = Math.abs(sourceX - targetX);
  const deltaY = Math.abs(sourceY - targetY);
  if (deltaX < 40 || deltaY < 40) {
    return "smoothstep";
  }

  return "bezier";
}
