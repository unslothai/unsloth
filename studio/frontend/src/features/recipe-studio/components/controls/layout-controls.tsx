import { type ReactElement, useCallback } from "react";
import {
  Panel,
  useReactFlow,
  useUpdateNodeInternals,
} from "@xyflow/react";
import { Button } from "@/components/ui/button";

type LayoutControlsProps = {
  direction: "LR" | "TB";
  onLayout: () => void;
  onToggleDirection: () => void;
};

export function LayoutControls({
  direction,
  onLayout,
  onToggleDirection,
}: LayoutControlsProps): ReactElement {
  const { fitView, getNodes } = useReactFlow();
  const updateNodeInternals = useUpdateNodeInternals();

  const refreshNodeInternals = useCallback(() => {
    const nodeIds = getNodes().map((node) => node.id);
    if (nodeIds.length > 0) {
      updateNodeInternals(nodeIds);
    }
  }, [getNodes, updateNodeInternals]);

  const handleLayout = useCallback(() => {
    onLayout();
    requestAnimationFrame(() => {
      refreshNodeInternals();
      requestAnimationFrame(() => {
        fitView({ duration: 250 });
      });
    });
  }, [fitView, onLayout, refreshNodeInternals]);

  const handleToggleDirection = useCallback(() => {
    onToggleDirection();
    requestAnimationFrame(() => {
      refreshNodeInternals();
      requestAnimationFrame(() => {
        refreshNodeInternals();
      });
    });
  }, [onToggleDirection, refreshNodeInternals]);

  return (
    <Panel position="top-left" className="m-3 flex items-center gap-2">
      <Button size="sm" className="corner-squircle" variant="secondary" onClick={handleLayout}>
        Auto layout
      </Button>
      <Button size="sm" className="corner-squircle" variant="outline" onClick={handleToggleDirection}>
        {direction}
      </Button>
    </Panel>
  );
}
