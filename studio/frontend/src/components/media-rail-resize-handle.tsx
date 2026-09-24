// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLayoutEffect, useRef, useState } from "react";

import { PanelResizeHandle } from "@/components/ui/panel-resize-handle";
import {
  MEDIA_RAIL_ROOT_ATTR,
  MEDIA_RAIL_WIDTH_MIN,
  type MediaRailKind,
  useMediaRailWidth,
} from "@/hooks/use-media-rail-width";
import { cn } from "@/lib/utils";

/** The rail's rendered width, mirroring the pages' `min(rail, 100% - 13rem)`. */
const RAIL_BOX_WIDTH: Record<MediaRailKind, string> = {
  images: "min(var(--media-rail-width,calc(408px*var(--ui-space-scale,1))),calc(100% - 13rem))",
  video: "min(var(--media-rail-width,calc(400px*var(--ui-space-scale,1))),calc(100% - 13rem))",
  audio: "min(var(--media-rail-width,calc(408px*var(--ui-space-scale,1))),calc(100% - 13rem))",
};

/**
 * Drag handle on the right edge of a media page's settings rail. Paints `--media-rail-width` on
 * the page root so the header and rail move together.
 *
 * `placement="rail"` sits inside the (relative) rail. `placement="page"` sits in the (relative)
 * page root and spans its full height, for pages whose divider also runs through the header.
 * `className` hides it where the panes stack.
 */
export function MediaRailResizeHandle({
  kind,
  placement = "rail",
  className,
}: {
  kind: MediaRailKind;
  placement?: "rail" | "page";
  className?: string;
}) {
  const rail = useMediaRailWidth(kind);
  const anchorRef = useRef<HTMLSpanElement>(null);
  const root = () =>
    anchorRef.current?.closest<HTMLElement>(`[${MEDIA_RAIL_ROOT_ATTR}]`) ?? null;
  // The `100% - 13rem` cap in layout px, so drags and keys start from the width actually shown.
  const [cap, setCap] = useState<number | null>(null);
  useLayoutEffect(() => {
    const anchor = anchorRef.current;
    // What the rail's 100% resolves against: the page root's padding box, or the rail's parent.
    const block = placement === "page" ? anchor?.parentElement : anchor?.parentElement?.parentElement;
    if (!block) return;
    const update = () => {
      const style = getComputedStyle(block);
      const inner =
        placement === "page"
          ? block.clientWidth
          : block.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      const rem = parseFloat(getComputedStyle(document.documentElement).fontSize) || 16;
      setCap((inner - 13 * rem) / rail.scale);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(block);
    return () => observer.disconnect();
  }, [placement, rail.scale]);
  const max = cap === null ? rail.max : Math.max(MEDIA_RAIL_WIDTH_MIN, Math.min(rail.max, cap));
  const width = Math.min(rail.width, max);
  const handle = (
    <PanelResizeHandle
      edge="right"
      open={true}
      width={width}
      stored={rail.stored}
      min={MEDIA_RAIL_WIDTH_MIN}
      max={max}
      scale={rail.scale}
      clamp={(px) => Math.min(rail.clamp(px), max)}
      // Commits what is shown; a capped pull past the edge keeps the stored preference.
      setWidth={(px) => rail.setWidth(Math.min(px, max))}
      resetWidth={rail.resetWidth}
      // The rail does not collapse; click does nothing, drag and arrow keys resize.
      onToggle={() => {}}
      target={root}
      cssVar="--media-rail-width"
      measure={() => width}
      label="Resize settings panel"
      toggleLabel="Resize settings panel"
      hideTooltip={true}
      dataSlot="media-rail-resize-handle"
      className="pointer-events-auto"
    />
  );
  if (placement === "page") {
    // A rail-wide box under the top inset, so the handle on its right edge covers the whole divider.
    return (
      <span
        ref={anchorRef}
        className={cn(
          "pointer-events-none absolute bottom-0 left-0 top-[var(--studio-content-top-inset,0px)] z-50",
          className,
        )}
        style={{ width: RAIL_BOX_WIDTH[kind] }}
      >
        {handle}
      </span>
    );
  }
  return (
    <span ref={anchorRef} className={className}>
      {handle}
    </span>
  );
}
