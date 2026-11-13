// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactNode, type RefObject, useEffect, useRef, useState } from "react";

import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { InformationCircleIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";

// index.css reveals the "i" when the row one or two levels up holds keyboard focus. That used to be
// `ROW:has(:focus-visible) > [data-slot=info-hint]`, a style after a :has(), which WebKit re-checks
// over the whole document on every keystroke, so the hint marks itself instead.
const FIELD_ROW = ":is(.diffusion-surface, .hint-on-hover) *";

function useNearFocus(hint: RefObject<HTMLElement | null>): void {
  useEffect(() => {
    const el = hint.current;
    const rows = [el?.parentElement, el?.parentElement?.parentElement].filter(
      (row): row is HTMLElement => row?.matches(FIELD_ROW) ?? false,
    );
    if (!el || rows.length === 0) return;
    const sync = () => {
      const near = rows.some((row) => row.querySelector(":focus-visible") !== null);
      if (near !== el.hasAttribute("data-near-focus")) el.toggleAttribute("data-near-focus", near);
    };
    // After the event: :focus-visible has settled on the element gaining or losing focus. Keyboard
    // input can turn it on without moving focus (a mouse-focused slider nudged with an arrow key),
    // and a pointer press can turn it off, so those re-check too.
    const later = () => queueMicrotask(sync);
    const top = rows[rows.length - 1];
    const events = ["focusin", "focusout", "keydown", "pointerdown"] as const;
    for (const type of events) top.addEventListener(type, later);
    sync();
    return () => {
      for (const type of events) top.removeEventListener(type, later);
    };
  }, [hint]);
}

/** Small "i" affordance that reveals a styled tooltip on hover/focus. The
 *  standard inline help control across the settings UI. */
export function InfoHint({
  children,
  side = "top",
  hoverOnly = false,
}: {
  children: ReactNode;
  side?: "top" | "right" | "bottom" | "left";
  /** Hover only, so a click cannot leave it open; pair it with sr-only text. */
  hoverOnly?: boolean;
}) {
  const [hovered, setHovered] = useState(false);
  const button = useRef<HTMLButtonElement>(null);
  useNearFocus(button);
  return (
    <Tooltip
      {...(hoverOnly ? { open: hovered, onOpenChange: () => {} } : {})}
    >
      <TooltipTrigger asChild>
        <button
          ref={button}
          onPointerEnter={hoverOnly ? () => setHovered(true) : undefined}
          onPointerLeave={hoverOnly ? () => setHovered(false) : undefined}
          type="button"
          data-slot="info-hint"
          aria-label="More info"
          className="inline-flex size-3 shrink-0 cursor-help items-center justify-center rounded-full text-muted-foreground/70 transition-colors hover:text-[#383835] dark:hover:text-[#e8e8e8] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        >
          <HugeiconsIcon
            icon={InformationCircleIcon}
            strokeWidth={1.75}
            className="size-2.5"
          />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side={side}
        align="center"
        sideOffset={6}
        collisionPadding={12}
        className="[&_span>svg]:hidden! duration-0 max-w-[calc(240px*var(--ui-space-scale,1))] text-left"
      >
        {children}
      </TooltipContent>
    </Tooltip>
  );
}
