// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { HugeiconsIcon } from "@hugeicons/react";
import { MiniMap, Panel, useReactFlow } from "@xyflow/react";
import { memo } from "react";
import { useUiStore } from "../stores/ui-store";
import type { RfNode } from "./derive-rf";
import {
  CenterFocusIcon,
  MapsIcon,
  MinusSignIcon,
  PlusSignIcon,
} from "./icon-data";

const FIT_VIEW_OPTIONS = { padding: 0.2, maxZoom: 1.1, duration: 200 };
const MINIMAP_STYLE = { width: 168, height: 112 };
// Recipe Studio's floating button, as a true circle (its class adds corner-squircle).
const ROUND_BUTTON =
  "btn-fill group size-11 rounded-full border-0 p-0 text-muted-foreground hover:text-foreground";

function minimapColor(node: RfNode): string {
  if (node.type !== "flowNode")
    return "color-mix(in oklab, var(--muted-foreground) 14%, transparent)";
  return node.selected
    ? "var(--primary)"
    : "color-mix(in oklab, var(--muted-foreground) 55%, transparent)";
}

function FloatingButton({
  icon,
  label,
  onClick,
  active,
}: {
  icon: typeof PlusSignIcon;
  label: string;
  onClick: () => void;
  active?: boolean;
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      aria-label={label}
      aria-pressed={active}
      title={label}
      onClick={onClick}
      className={`${ROUND_BUTTON} ${active ? "text-foreground" : ""}`}
    >
      <HugeiconsIcon icon={icon} className="size-4" />
    </Button>
  );
}

/** Recipe Studio's canvas controls: round floating buttons, the minimap behind a toggle. */
function CanvasControlsImpl({ readOnly }: { readOnly: boolean }) {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const paletteOpen = useUiStore((s) => s.paletteOpen);
  const [minimap, setMinimap] = usePersistedToggle("unsloth_flows_minimap");
  return (
    <>
      {!paletteOpen && !readOnly && (
        <Panel position="top-left" className="m-3">
          <FloatingButton
            icon={PlusSignIcon}
            label="Show nodes"
            onClick={() => useUiStore.getState().togglePalette()}
          />
        </Panel>
      )}
      <Panel position="bottom-left" className="m-3 flex items-center gap-2">
        <FloatingButton
          icon={PlusSignIcon}
          label="Zoom in"
          onClick={() => void zoomIn({ duration: 150 })}
        />
        <FloatingButton
          icon={MinusSignIcon}
          label="Zoom out"
          onClick={() => void zoomOut({ duration: 150 })}
        />
        <FloatingButton
          icon={CenterFocusIcon}
          label="Center view"
          onClick={() => void fitView(FIT_VIEW_OPTIONS)}
        />
        <FloatingButton
          icon={MapsIcon}
          label={minimap ? "Hide minimap" : "Show minimap"}
          onClick={() => setMinimap(!minimap)}
          active={minimap}
        />
      </Panel>
      {minimap && (
        <MiniMap
          pannable={true}
          zoomable={true}
          nodeColor={minimapColor}
          nodeStrokeWidth={0}
          nodeBorderRadius={6}
          bgColor="var(--card)"
          maskColor="color-mix(in oklab, var(--background) 55%, transparent)"
          maskStrokeColor="var(--border)"
          maskStrokeWidth={1}
          style={MINIMAP_STYLE}
          className="corner-squircle !m-3 overflow-hidden !rounded-xl !border !border-border"
        />
      )}
    </>
  );
}

export const CanvasControls = memo(CanvasControlsImpl);
