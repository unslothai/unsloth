// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type React from "react";
import { useCallback, useRef, useState } from "react";

/** Where a drop would land: on one side of one tile. */
export interface StripDropCue {
  id: string;
  edge: "before" | "after";
}

// Custom type so other drop targets do not read the drag as a file or link.
const DRAG_TYPE = "application/x-unsloth-gallery-item";

/** Whether the event came from the tile's own DOM, not a portalled menu or dialog inside it. */
function fromTile(event: React.SyntheticEvent<HTMLElement>): boolean {
  return event.currentTarget.contains(event.target as Node);
}

/**
 * Drag-to-reorder for a gallery strip: horizontal (Images, Video) or, with `axis: "y"`, a vertical
 * list (Audio history).
 *
 * Tracked on the strip so gaps and ends are drop targets. Tiles carry `data-reorder-id` in display
 * order. A drop reports the id the tile now follows (null = front). Alt + Left / Right (Up / Down
 * on a list) moves the focused tile one slot. Drags that did not start on a tile are ignored.
 */
export function useStripReorder(
  onMove: (id: string, afterId: string | null) => void,
  { axis = "x" }: { axis?: "x" | "y" } = {},
) {
  const vertical = axis === "y";
  const backKey = vertical ? "ArrowUp" : "ArrowLeft";
  const forwardKey = vertical ? "ArrowDown" : "ArrowRight";
  const dragIdRef = useRef<string | null>(null);
  const [draggingId, setDraggingId] = useState<string | null>(null);
  const [cue, setCue] = useState<StripDropCue | null>(null);

  const end = useCallback(() => {
    dragIdRef.current = null;
    setDraggingId(null);
    setCue(null);
  }, []);

  /** The drop at pointer position `at` along the axis, or null if the tile would not move. */
  function resolve(
    strip: HTMLElement,
    at: number,
  ): { afterId: string | null; cue: StripDropCue } | null {
    const dragged = dragIdRef.current;
    if (!dragged) return null;
    const tiles = [...strip.querySelectorAll<HTMLElement>("[data-reorder-id]")];
    const order = tiles.map((tile) => tile.dataset.reorderId ?? "");
    let index = tiles.findIndex((tile) => {
      const rect = tile.getBoundingClientRect();
      return vertical ? at < rect.top + rect.height / 2 : at < rect.left + rect.width / 2;
    });
    if (index < 0) index = tiles.length;
    const from = order.indexOf(dragged);
    // Either side of the dragged tile is a no-op.
    if (from < 0 || index === from || index === from + 1) return null;
    return {
      afterId: index === 0 ? null : order[index - 1],
      cue:
        index < order.length
          ? { id: order[index], edge: "before" }
          : { id: order[order.length - 1], edge: "after" },
    };
  }

  const stripProps = {
    onDragOver: (event: React.DragEvent<HTMLElement>) => {
      if (!dragIdRef.current) return;
      event.preventDefault();
      event.dataTransfer.dropEffect = "move";
      const next = resolve(event.currentTarget, vertical ? event.clientY : event.clientX)?.cue ?? null;
      setCue((prev) =>
        prev?.id === next?.id && prev?.edge === next?.edge ? prev : next,
      );
    },
    onDragLeave: (event: React.DragEvent<HTMLElement>) => {
      if (!event.currentTarget.contains(event.relatedTarget as Node | null)) {
        setCue(null);
      }
    },
    onDrop: (event: React.DragEvent<HTMLElement>) => {
      const dragged = dragIdRef.current;
      if (!dragged) return;
      event.preventDefault();
      const drop = resolve(event.currentTarget, vertical ? event.clientY : event.clientX);
      end();
      if (drop) onMove(dragged, drop.afterId);
    },
  };

  function tileProps(id: string) {
    return {
      "data-reorder-id": id,
      draggable: true,
      onDragStart: (event: React.DragEvent<HTMLElement>) => {
        if (!fromTile(event)) return;
        dragIdRef.current = id;
        setDraggingId(id);
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData(DRAG_TYPE, id);
      },
      onDragEnd: end,
      onKeyDown: (event: React.KeyboardEvent<HTMLElement>) => {
        if (
          !fromTile(event) ||
          !event.altKey ||
          (event.key !== backKey && event.key !== forwardKey)
        ) {
          return;
        }
        const strip = event.currentTarget.parentElement;
        if (!strip) return;
        const order = [
          ...strip.querySelectorAll<HTMLElement>("[data-reorder-id]"),
        ].map((tile) => tile.dataset.reorderId ?? "");
        const from = order.indexOf(id);
        event.preventDefault();
        if (event.key === backKey && from > 0) {
          onMove(id, from > 1 ? order[from - 2] : null);
        } else if (event.key === forwardKey && from >= 0 && from < order.length - 1) {
          onMove(id, order[from + 1]);
        }
      },
    };
  }

  return { stripProps, tileProps, cue, draggingId };
}
