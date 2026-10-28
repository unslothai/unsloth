// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { usePersistedToggle } from "@/hooks/use-persisted-toggle";
import { HugeiconsIcon } from "@hugeicons/react";
import { MiniMap, Panel, useReactFlow } from "@xyflow/react";
import { memo, useEffect, useState } from "react";
import { useCatalogStore } from "../stores/catalog-store";
import { useUiStore } from "../stores/ui-store";
import type { RfNode } from "./derive-rf";
import { nodeIcon } from "./icons";
import { NodeIconTile } from "./node-card";
import { addNodeConnected } from "./node-factory";
import { CATEGORY_TONES } from "./node-tones";
import { placementPosition, stopPlacement, usePlacement } from "./placement";
import {
  CenterFocusIcon,
  MapsIcon,
  MinusSignIcon,
  PlusSignIcon,
} from "./icon-data";

// React Flow Controls' fit (default padding), animated.
const FIT_VIEW_OPTIONS = { duration: 200 };
const MINIMAP_STYLE = { width: 168, height: 112 };
// Above every node, recipe notes and frames included.
const PANEL_STYLE = { zIndex: 1001 };
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

/** While a Nodes panel item is armed: each click on empty canvas places one; Esc, right-click
 *  or leaving the editor stops. Clicks on nodes and groups are left alone. */
function ArmedPlacement() {
  const armed = usePlacement((s) => s.armed);
  const spec = useCatalogStore((s) =>
    armed ? s.catalog?.nodes.find((n) => n.type === armed) : undefined,
  );
  const { screenToFlowPosition } = useReactFlow();
  const [pointer, setPointer] = useState<{ x: number; y: number } | null>(
    null,
  );

  useEffect(() => stopPlacement, []);

  useEffect(() => {
    const canvas = document.querySelector<HTMLElement>(
      '[data-testid="flows-canvas"]',
    );
    if (!armed || !canvas) return;
    const onPane = (e: Event) =>
      (e.target as Element).classList.contains("react-flow__pane");
    const click = (e: MouseEvent) => {
      if (e.button !== 0 || e.detail > 1 || !onPane(e)) return;
      const snap = useUiStore.getState().snapToGrid;
      addNodeConnected(
        armed,
        placementPosition(
          screenToFlowPosition({ x: e.clientX, y: e.clientY }),
          snap,
        ),
      );
    };
    const dblclick = (e: MouseEvent) => {
      if (onPane(e)) e.stopPropagation();
    };
    const menu = (e: MouseEvent) => {
      e.preventDefault();
      stopPlacement();
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") stopPlacement();
    };
    const move = (e: PointerEvent) =>
      setPointer(onPane(e) ? { x: e.clientX, y: e.clientY } : null);
    canvas.classList.add("flows-armed");
    canvas.addEventListener("click", click);
    canvas.addEventListener("dblclick", dblclick, true);
    canvas.addEventListener("contextmenu", menu);
    canvas.addEventListener("pointermove", move);
    window.addEventListener("keydown", key);
    return () => {
      canvas.classList.remove("flows-armed");
      canvas.removeEventListener("click", click);
      canvas.removeEventListener("dblclick", dblclick, true);
      canvas.removeEventListener("contextmenu", menu);
      canvas.removeEventListener("pointermove", move);
      window.removeEventListener("keydown", key);
      setPointer(null);
    };
  }, [armed, screenToFlowPosition]);

  if (!armed) return null;
  return (
    <>
      <Panel position="top-center" className="m-3" style={PANEL_STYLE}>
        <span className="rounded-full border border-border bg-card px-3 py-1 text-xs text-muted-foreground shadow-sm">
          Click to place · Esc to stop
        </span>
      </Panel>
      {pointer && spec && (
        <div
          className="pointer-events-none fixed z-[1000] flex items-center gap-2 rounded-xl border border-border bg-card px-2.5 py-1.5 text-sm font-semibold opacity-80 shadow-md"
          style={{ left: pointer.x + 12, top: pointer.y + 12 }}
        >
          <NodeIconTile
            icon={nodeIcon(spec.icon)}
            tone={CATEGORY_TONES[spec.category]}
          />
          {spec.title}
        </div>
      )}
    </>
  );
}

/** Recipe Studio's canvas controls: round floating buttons, the minimap behind a toggle. */
function CanvasControlsImpl({ readOnly }: { readOnly: boolean }) {
  const { zoomIn, zoomOut, fitView } = useReactFlow();
  const paletteOpen = useUiStore((s) => s.paletteOpen);
  const [minimap, setMinimap] = usePersistedToggle("unsloth_flows_minimap");
  return (
    <>
      {!readOnly && <ArmedPlacement />}
      {!paletteOpen && !readOnly && (
        <Panel position="top-left" className="m-3" style={PANEL_STYLE}>
          <FloatingButton
            icon={PlusSignIcon}
            label="Show nodes"
            onClick={() => useUiStore.getState().togglePalette()}
          />
        </Panel>
      )}
      <Panel
        position="bottom-left"
        className="m-3 flex items-center gap-2"
        style={PANEL_STYLE}
      >
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
          style={{ ...MINIMAP_STYLE, ...PANEL_STYLE }}
          className="corner-squircle !m-3 overflow-hidden !rounded-xl !border !border-border"
        />
      )}
    </>
  );
}

export const CanvasControls = memo(CanvasControlsImpl);
