// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { EdgeLabelRenderer, useReactFlow } from "@xyflow/react";
import { useCallback, useEffect, useRef, useState } from "react";

/** Hover state for one wire that survives the move from the wire onto its × button. */
export function useWireHover() {
  const [hover, setHover] = useState(false);
  const timer = useRef<number | undefined>(undefined);
  const enter = useCallback(() => {
    window.clearTimeout(timer.current);
    setHover(true);
  }, []);
  const leave = useCallback(() => {
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setHover(false), 200);
  }, []);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  return { hover, enter, leave };
}

/** Drawn under a selected wire so the selection reads at any wire colour. */
export function WireHalo({ path }: { path: string }) {
  return (
    <path
      d={path}
      fill="none"
      stroke="var(--primary)"
      strokeWidth={7}
      strokeLinecap="round"
      strokeLinejoin="round"
      opacity={0.35}
      className="pointer-events-none"
    />
  );
}

/** A small × at a wire's midpoint, shown on hover or selection; removes the wire as one undo step. */
export function WireDelete({
  id,
  x,
  y,
  show,
  onEnter,
  onLeave,
}: {
  id: string;
  x: number;
  y: number;
  show: boolean;
  onEnter: () => void;
  onLeave: () => void;
}) {
  const { deleteElements } = useReactFlow();
  if (!show) return null;
  return (
    <EdgeLabelRenderer>
      <button
        type="button"
        aria-label="Remove wire"
        title="Remove wire"
        data-testid="wire-delete"
        className="nodrag nopan pointer-events-auto absolute flex size-5 items-center justify-center rounded-full border border-border bg-card text-muted-foreground shadow-sm hover:border-destructive/60 hover:text-destructive"
        style={{ transform: `translate(-50%, -50%) translate(${x}px, ${y}px)` }}
        onMouseEnter={onEnter}
        onMouseLeave={onLeave}
        onClick={(e) => {
          e.stopPropagation();
          void deleteElements({ edges: [{ id }] });
        }}
      >
        <HugeiconsIcon icon={Cancel01Icon} size={11} strokeWidth={2} />
      </button>
    </EdgeLabelRenderer>
  );
}
