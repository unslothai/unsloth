import { type ReactElement, useCallback } from "react";
import { Lock, LockOpen, Maximize2, Minus, Plus } from "lucide-react";
import { Panel, useReactFlow } from "@xyflow/react";
import { Button } from "@/components/ui/button";
import { getFitNodeIdsIgnoringNotes } from "../../utils/graph/fit-view";
import { RECIPE_FLOATING_ICON_BUTTON_CLASS } from "../recipe-floating-icon-button-class";

type ViewportControlsProps = {
  interactive: boolean;
  onToggleInteractive: () => void;
};

export function ViewportControls({
  interactive,
  onToggleInteractive,
}: ViewportControlsProps): ReactElement {
  const { zoomIn, zoomOut, fitView, getNodes } = useReactFlow();

  const handleZoomIn = useCallback(() => {
    zoomIn({ duration: 150 });
  }, [zoomIn]);

  const handleZoomOut = useCallback(() => {
    zoomOut({ duration: 150 });
  }, [zoomOut]);

  const handleFitView = useCallback(() => {
    fitView({
      duration: 250,
      nodes: getFitNodeIdsIgnoringNotes(getNodes()),
    });
  }, [fitView, getNodes]);

  return (
    <Panel position="bottom-left" className="m-3 flex items-center gap-2">
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleZoomIn}
        aria-label="Zoom in"
      >
        <Plus className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleZoomOut}
        aria-label="Zoom out"
      >
        <Minus className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleFitView}
        aria-label="Fit view"
      >
        <Maximize2 className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={onToggleInteractive}
        aria-label={interactive ? "Lock interaction" : "Unlock interaction"}
      >
        {interactive ? <LockOpen className="size-4" /> : <Lock className="size-4" />}
      </Button>
    </Panel>
  );
}
