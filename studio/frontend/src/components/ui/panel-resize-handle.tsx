// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client"

import * as React from "react"

import { cn } from "@/lib/utils"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { getClientPlatform } from "@/components/tauri/window-titlebar"

/** Pointer travel (px) below which a drag counts as a plain click. */
const DRAG_SLOP = 4
/** A compatibility click lands immediately after pointer-up. */
const CLICK_COMPAT_WINDOW_MS = 300
/** Arrow-key resize step for keyboard users. */
const RESIZE_STEP = 16

type DragState = {
  startX: number
  startWidth: number
  moved: boolean
}

export type PanelResizeHandleProps = {
  /** Which edge of the panel the handle sits on. */
  edge: "left" | "right"
  open: boolean
  width: number
  /** Uncapped stored preference, so a capped drag does not lower it. */
  stored: number
  min: number
  max: number
  clamp: (px: number) => number
  setWidth: (px: number) => void
  resetWidth: () => void
  onToggle: () => void
  /** Element to paint the live width onto, and the property to paint. */
  target: () => HTMLElement | null
  cssVar: string
  /** Measured to start a drag from the rendered size when collapsed. */
  measure: () => number
  label: string
  toggleLabel: string
  /** Translated tooltip copy; the caller owns the translation layer. */
  collapseHint: string
  expandHint: string
  dragHint: string
  /** Shown in the tooltip when the panel has a toggle shortcut. */
  shortcut?: string
  dataSlot?: string
  className?: string
  /** Mirrors the live width onto :root for chrome outside the panel. */
  rootVar?: string
}

/**
 * A draggable panel edge: drag to resize, click to collapse or expand. Arrow
 * keys resize, Home restores the default. The width is painted straight to the
 * target while dragging and only persisted on release.
 */
export function PanelResizeHandle({
  edge,
  open,
  width,
  stored,
  min,
  max,
  clamp,
  setWidth,
  resetWidth,
  onToggle,
  target,
  cssVar,
  measure,
  label,
  toggleLabel,
  collapseHint,
  expandHint,
  dragHint,
  shortcut,
  dataSlot = "panel-resize-handle",
  className,
  rootVar,
}: PanelResizeHandleProps) {
  const ref = React.useRef<HTMLButtonElement>(null)
  const dragRef = React.useRef<DragState | null>(null)
  const [dragging, setDragging] = React.useState(false)
  const [hovered, setHovered] = React.useState(false)
  const [focused, setFocused] = React.useState(false)
  const [isMacPlatform] = React.useState(() => getClientPlatform().includes("mac"))
  const hint = shortcut ? shortcut.replace("Mod", isMacPlatform ? "⌘" : "Ctrl+") : null

  // Cached on pointer down so no DOM walk per move.
  const targetRef = React.useRef<HTMLElement | null>(null)
  const frameRef = React.useRef(0)
  const pendingRef = React.useRef(0)
  // What the pointer asked for, before the viewport cap. Committing the capped
  // value instead would quietly downgrade a stored preference on a narrow window.
  const rawRef = React.useRef(0)
  // When a pointer sequence last ended. The browser's compatibility click
  // lands in the same tick, so only a click that close behind is a duplicate.
  // A timestamp cannot go stale the way an armed flag does: a genuine cancel
  // emits no click, and a later assistive-tech click still gets through.
  const handledAtRef = React.useRef(0)
  const committedRef = React.useRef(width)
  React.useEffect(() => {
    committedRef.current = width
  }, [width])

  const paint = React.useCallback(
    (value: string) => {
      targetRef.current?.style.setProperty(cssVar, value)
      if (rootVar) {
        document.documentElement.style.setProperty(rootVar, value)
      }
    },
    [cssVar, rootVar],
  )

  // Resizing relayouts the whole shell, and pointermove fires faster than the
  // display refreshes, so coalesce to one paint per frame.
  const paintWidth = React.useCallback(
    (px: number) => {
      pendingRef.current = px
      if (frameRef.current) return
      frameRef.current = requestAnimationFrame(() => {
        frameRef.current = 0
        paint(`${pendingRef.current}px`)
      })
    },
    [paint],
  )

  const endDrag = React.useCallback(() => {
    // Only a sequence that actually started can produce a compatibility click.
    // This also runs as the effect cleanup, where no drag happened.
    if (dragRef.current) handledAtRef.current = Date.now()
    dragRef.current = null
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current)
      frameRef.current = 0
    }
    // Hand the property back to the committed value. A commit re-renders with
    // the new width; a cancel or a no-commit drag keeps DOM and store in step.
    paint(`${committedRef.current}px`)
    if (rootVar) document.documentElement.style.removeProperty(rootVar)
    targetRef.current?.removeAttribute("data-resizing")
    document.documentElement.removeAttribute("data-panel-resizing")
    targetRef.current = null
    setDragging(false)
    document.body.style.removeProperty("cursor")
    document.body.style.removeProperty("user-select")
  }, [paint, rootVar])

  const handlePointerDown = (event: React.PointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    targetRef.current = target()
    // Collapsed: grow from the rendered size so the edge tracks the pointer.
    const start = open ? width : measure()
    dragRef.current = { startX: event.clientX, startWidth: start, moved: false }
    pendingRef.current = start
    rawRef.current = start
    targetRef.current?.setAttribute("data-resizing", "true")
    document.documentElement.setAttribute("data-panel-resizing", "true")
    setDragging(true)
    document.body.style.setProperty("cursor", "col-resize")
    document.body.style.setProperty("user-select", "none")
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current
    if (!drag) return
    // A panel whose handle is on its left edge grows as the pointer moves left.
    const delta = (edge === "left" ? -1 : 1) * (event.clientX - drag.startX)
    if (!drag.moved && Math.abs(delta) < DRAG_SLOP) return
    drag.moved = true

    const next = drag.startWidth + delta
    rawRef.current = next
    if (!open) {
      // Past the minimum, dragging the collapsed edge reopens it.
      if (next >= min) {
        paintWidth(clamp(next))
        onToggle()
      }
      return
    }
    // Dragging inward stops at the minimum. Collapsing is click or the shortcut.
    paintWidth(clamp(next))
  }

  const handlePointerUp = (event: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current
    if (!drag) return
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId)
    }
    endDrag()

    if (!drag.moved) {
      onToggle()
      return
    }
    // A drag below the minimum leaves the stored width alone.
    if (!open) return
    // Capped: the visible edge is already at the cap, so an outward pull cannot
    // express intent beyond it. Committing would silently lower the larger
    // hidden preference. A deliberate inward drag still commits.
    if (stored > max && rawRef.current >= max) return
    // Commit what was asked for, not the capped paint, so a drag on a narrow
    // window cannot shrink a larger stored preference. setWidth clamps.
    setWidth(rawRef.current)
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    // The collapse/expand the label advertises, for keyboard users. Pointer-up
    // handles it for the mouse; a synthesized click never reaches it.
    if (event.key === "Enter" || event.key === " ") {
      // preventDefault cancels the native click, so nothing follows to guard
      // against; arming here would swallow the next assistive-tech click.
      event.preventDefault()
      onToggle()
      return
    }
    const outward = edge === "left" ? "ArrowLeft" : "ArrowRight"
    const inward = edge === "left" ? "ArrowRight" : "ArrowLeft"
    if (event.key === outward || event.key === inward) {
      event.preventDefault()
      if (!open) {
        // Collapsed there is nothing to resize, so the outward arrow reopens.
        if (event.key === outward) onToggle()
        return
      }
      if (event.key === outward && stored > max && width >= max) return
      setWidth(width + (event.key === outward ? RESIZE_STEP : -RESIZE_STEP))
      return
    }
    if (event.key === "Home") {
      event.preventDefault()
      resetWidth()
    }
  }

  // Clear a stuck cursor override if we unmount mid-drag.
  React.useEffect(() => endDrag, [endDrag])

  return (
    <Tooltip open={(hovered || focused) && !dragging}>
      <TooltipTrigger asChild>
        <button
          ref={ref}
          type="button"
          data-slot={dataSlot}
          data-dragging={dragging || undefined}
          aria-label={open ? label : toggleLabel}
          {...(open ? { "aria-orientation": "vertical" as const } : {})}
          {...(open
            ? { "aria-valuenow": width, "aria-valuemin": min, "aria-valuemax": max }
            : {})}
          role={open ? "separator" : "button"}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={endDrag}
          onKeyDown={handleKeyDown}
          onClick={() => {
            // Switch and voice control activate by dispatching a bare click
            // with no pointer or key events, which nothing else here catches.
            if (Date.now() - handledAtRef.current < CLICK_COMPAT_WINDOW_MS) return
            onToggle()
          }}
          onPointerEnter={() => setHovered(true)}
          onPointerLeave={() => setHovered(false)}
          onFocus={(event) => setFocused(event.target.matches(":focus-visible"))}
          onBlur={() => setFocused(false)}
          className={cn(
            "absolute inset-y-0 z-30 hidden w-2 touch-none select-none sm:block",
            edge === "left" ? "-left-1" : "-right-1",
            // `!` overrides the app-wide hand cursor on buttons.
            open
              ? "cursor-col-resize!"
              : edge === "left"
                ? "cursor-w-resize!"
                : "cursor-e-resize!",
            // Sits exactly on the panel border so hover recolours one line.
            "after:absolute after:inset-y-0 after:w-px after:bg-transparent after:transition-colors after:duration-150",
            edge === "left" ? "after:left-1" : "after:right-1",
            "hover:after:bg-sidebar-ring/25 data-dragging:after:bg-sidebar-ring/25",
            // The app zeroes the native outline on buttons, so mark focus here.
            "focus-visible:outline-none focus-visible:after:bg-sidebar-ring/60",
            className,
          )}
        />
      </TooltipTrigger>
      <TooltipContent
        side={edge === "left" ? "left" : "right"}
        align="center"
        className="tooltip-compact"
      >
        <span className="flex flex-col gap-px">
          <span>
            {open ? collapseHint : expandHint}
            {hint ? ` ${hint}` : ""}
          </span>
          <span className="opacity-70">{dragHint}</span>
        </span>
      </TooltipContent>
    </Tooltip>
  )
}
