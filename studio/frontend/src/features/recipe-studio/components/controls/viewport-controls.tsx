// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Panel, useReactFlow } from "@xyflow/react";
import { Focus, Lock, LockOpen } from "lucide-react";
import { type ReactElement, useCallback } from "react";
import { buildFitViewOptions } from "../../utils/graph/fit-view";
import { RECIPE_FLOATING_ICON_BUTTON_CLASS } from "../recipe-floating-icon-button-class";
import { MinusSignIcon, PlusSignIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { ContractArrowsIcon, ExpandArrowsIcon } from "@/lib/expand-icons";

type ViewportControlsProps = {
  interactive: boolean;
  lockDisabled?: boolean;
  onToggleInteractive: () => void;
  maximized: boolean;
  onToggleMaximize: () => void;
};

export function ViewportControls({
  interactive,
  lockDisabled = false,
  onToggleInteractive,
  maximized,
  onToggleMaximize,
}: ViewportControlsProps): ReactElement {
  const { zoomIn, zoomOut, fitView, getNodes } = useReactFlow();

  const handleZoomIn = useCallback(() => {
    zoomIn({ duration: 150 });
  }, [zoomIn]);

  const handleZoomOut = useCallback(() => {
    zoomOut({ duration: 150 });
  }, [zoomOut]);

  const handleFitView = useCallback(() => {
    fitView(buildFitViewOptions(getNodes()));
  }, [fitView, getNodes]);

  return (
    <Panel
      position="bottom-left"
      className="m-3 flex items-center gap-2 @max-[48rem]/canvas:flex-col-reverse"
    >
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleZoomIn}
        aria-label="Zoom in"
      >
        <HugeiconsIcon icon={PlusSignIcon} className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleZoomOut}
        aria-label="Zoom out"
      >
        <HugeiconsIcon icon={MinusSignIcon} className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleFitView}
        aria-label="Center view"
      >
        <Focus className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={onToggleMaximize}
        aria-label={maximized ? "Exit full view" : "Expand to full view"}
      >
        {maximized ? (
          <HugeiconsIcon icon={ContractArrowsIcon} className="size-4" />
        ) : (
          <HugeiconsIcon icon={ExpandArrowsIcon} className="size-4" />
        )}
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        disabled={lockDisabled}
        onClick={onToggleInteractive}
        aria-label={interactive ? "Lock interaction" : "Unlock interaction"}
      >
        {interactive ? (
          <LockOpen className="size-4" />
        ) : (
          <Lock className="size-4" />
        )}
      </Button>
    </Panel>
  );
}
