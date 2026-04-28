// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactElement, useCallback } from "react";
import { Lock, LockOpen, Maximize2, Minus, Plus } from "lucide-react";
import { Panel, useReactFlow } from "@xyflow/react";
import { Button } from "@/components/ui/button";
import { useI18n } from "@/features/i18n";
import { buildFitViewOptions } from "../../utils/graph/fit-view";
import { RECIPE_FLOATING_ICON_BUTTON_CLASS } from "../recipe-floating-icon-button-class";

type ViewportControlsProps = {
  interactive: boolean;
  lockDisabled?: boolean;
  onToggleInteractive: () => void;
};

export function ViewportControls({
  interactive,
  lockDisabled = false,
  onToggleInteractive,
}: ViewportControlsProps): ReactElement {
  const { t } = useI18n();
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
    <Panel position="bottom-left" className="m-3 flex items-center gap-2">
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleZoomIn}
        aria-label={t("recipe.viewport.zoomIn")}
      >
        <Plus className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleZoomOut}
        aria-label={t("recipe.viewport.zoomOut")}
      >
        <Minus className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        onClick={handleFitView}
        aria-label={t("recipe.viewport.fitView")}
      >
        <Maximize2 className="size-4" />
      </Button>
      <Button
        type="button"
        variant="ghost"
        size="icon"
        className={RECIPE_FLOATING_ICON_BUTTON_CLASS}
        disabled={lockDisabled}
        onClick={onToggleInteractive}
        aria-label={
          interactive
            ? t("recipe.viewport.lockInteraction")
            : t("recipe.viewport.unlockInteraction")
        }
      >
        {interactive ? <LockOpen className="size-4" /> : <Lock className="size-4" />}
      </Button>
    </Panel>
  );
}
