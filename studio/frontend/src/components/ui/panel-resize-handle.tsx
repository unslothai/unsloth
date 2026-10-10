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
import { PANEL_RESIZE_SCOPED_VARS_ENABLED } from "@/components/ui/panel-resize-recalc-flags"
import { Z_LAYER } from "@/lib/z-layers"

/** Pointer travel (px) below which a drag counts as a plain click. */
const DRAG_SLOP = 4
/** A compatibility click lands immediately after pointer-up. */
const CLICK_COMPAT_WINDOW_MS = 300
const RESIZE_STEP = 16

// One fixed transparent overlay owns the drag cursor and hit testing. Inherited cursor,
// user-select or pointer-events writes on <body> would restyle the whole document.
const DRAG_OVERLAY_SLOT = "panel-resize-drag-overlay"
/** Explicit ownership count, since a stuck overlay would swallow the whole UI. */
let dragOverlayOwners = 0

function acquireDragOverlay(): void {
  dragOverlayOwners += 1
  if (dragOverlayOwners > 1) return
  const el = document.createElement("div")
  el.setAttribute("data-slot", DRAG_OVERLAY_SLOT)
  el.setAttribute("aria-hidden", "true")
  const s = el.style
  s.position = "fixed"
  s.inset = "0"
  // Top of the named scale: it replaces `!important` rules that blanked whole subtrees.
  s.zIndex = String(Z_LAYER.DRAG_CURSOR_OVERLAY)
  s.background = "transparent"
  // Load-bearing: as the viewport's hit target it keeps hover and click off the content.
  s.pointerEvents = "auto"
  s.cursor = "col-resize"
  s.userSelect = "none"
  s.touchAction = "none"
  document.body.appendChild(el)
}

function releaseDragOverlay(): void {
  if (dragOverlayOwners === 0) return
  dragOverlayOwners -= 1
  if (dragOverlayOwners > 0) return
  document
    .querySelector(`[data-slot="${DRAG_OVERLAY_SLOT}"]`)
    ?.remove()
}

type DragState = {
  startX: number
  startWidth: number
  moved: boolean
}

export type PanelResizeHandleProps = {
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
  /** Measured to start a drag from the rendered size when collapsed, in layout px. */
  measure: () => number
  /** Browser interface scale: widths are layout px, painted times this. */
  scale?: number
  label: string
  toggleLabel: string
  collapseHint?: string
  expandHint?: string
  dragHint?: string
  shortcut?: string
  hideTooltip?: boolean
  onHoverChange?: (hovered: boolean) => void
  dataSlot?: string
  className?: string
  rootVar?: string
  /** Narrower `cssVar` target under PANEL_RESIZE_SCOPED_VARS_ENABLED; must hold every consumer. */
  scopedTarget?: () => HTMLElement | null
  /** Consumers of `rootVar` under PANEL_RESIZE_SCOPED_VARS_ENABLED; empty skips the write. */
  rootVarTargets?: () => HTMLElement[]
}

/** Drag to resize, click to collapse; the width is painted while dragging and persisted on release. */
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
  scale = 1,
  label,
  toggleLabel,
  collapseHint,
  expandHint,
  dragHint,
  shortcut,
  onHoverChange,
  hideTooltip = false,
  dataSlot = "panel-resize-handle",
  className,
  rootVar,
  scopedTarget,
  rootVarTargets,
}: PanelResizeHandleProps) {
  const ref = React.useRef<HTMLButtonElement>(null)
  const dragRef = React.useRef<DragState | null>(null)
  const [dragging, setDragging] = React.useState(false)
  const [hovered, setHovered] = React.useState(false)
  const [focused, setFocused] = React.useState(false)
  const [isMacPlatform] = React.useState(() => getClientPlatform().includes("mac"))
  const hint = shortcut ? shortcut.replace("Mod", isMacPlatform ? "⌘" : "Ctrl+") : null

  const targetRef = React.useRef<HTMLElement | null>(null)
  const frameRef = React.useRef(0)
  const pendingRef = React.useRef(0)
  // Pre-cap value, so a narrow window cannot downgrade the stored preference.
  const rawRef = React.useRef(0)
  // The compatibility click lands in the same tick; a timestamp cannot go stale like a flag.
  const handledAtRef = React.useRef(0)
  const committedRef = React.useRef(width)
  React.useEffect(() => {
    committedRef.current = width
  }, [width])
  const scaleRef = React.useRef(scale)
  React.useEffect(() => {
    scaleRef.current = scale
  }, [scale])

  // Where `rootVar` is painted, resolved on pointer down; documentElement with the flag off.
  const rootTargetsRef = React.useRef<HTMLElement[]>([])
  // Whether THIS handle holds the overlay; endDrag also runs as cleanup with nothing held.
  const overlayHeldRef = React.useRef(false)

  const paint = React.useCallback(
    (value: string) => {
      targetRef.current?.style.setProperty(cssVar, value)
      if (rootVar) {
        for (const el of rootTargetsRef.current) {
          el.style.setProperty(rootVar, value)
        }
      }
    },
    [cssVar, rootVar],
  )

  // pointermove outpaces the display; coalesce to one paint per frame.
  const paintWidth = React.useCallback(
    (px: number) => {
      pendingRef.current = px
      if (frameRef.current) return
      frameRef.current = requestAnimationFrame(() => {
        frameRef.current = 0
        paint(`${pendingRef.current * scaleRef.current}px`)
      })
    },
    [paint],
  )

  const endDrag = React.useCallback(() => {
    // Only a started sequence can produce a compatibility click; this also runs as cleanup.
    if (dragRef.current) handledAtRef.current = Date.now()
    dragRef.current = null
    if (frameRef.current) {
      cancelAnimationFrame(frameRef.current)
      frameRef.current = 0
    }
    // Restore the committed value so DOM and store stay in step on cancel or no-commit.
    paint(`${committedRef.current * scaleRef.current}px`)
    if (rootVar) {
      for (const el of rootTargetsRef.current) el.style.removeProperty(rootVar)
    }
    rootTargetsRef.current = []
    targetRef.current?.removeAttribute("data-resizing")
    document.documentElement.removeAttribute("data-panel-resizing")
    targetRef.current = null
    setDragging(false)
    if (overlayHeldRef.current) {
      overlayHeldRef.current = false
      releaseDragOverlay()
    }
  }, [paint, rootVar])

  const handlePointerDown = (event: React.PointerEvent<HTMLButtonElement>) => {
    if (event.button !== 0) return
    event.preventDefault()
    event.currentTarget.setPointerCapture(event.pointerId)
    targetRef.current =
      (PANEL_RESIZE_SCOPED_VARS_ENABLED && scopedTarget?.()) || target()
    rootTargetsRef.current = !rootVar
      ? []
      : PANEL_RESIZE_SCOPED_VARS_ENABLED
        ? (rootVarTargets?.() ?? [])
        : [document.documentElement]
    // Collapsed: grow from the rendered size so the edge tracks the pointer.
    const start = open ? width : measure()
    dragRef.current = { startX: event.clientX, startWidth: start, moved: false }
    pendingRef.current = start
    rawRef.current = start
    targetRef.current?.setAttribute("data-resizing", "true")
    document.documentElement.setAttribute("data-panel-resizing", "true")
    setDragging(true)
    overlayHeldRef.current = true
    acquireDragOverlay()
  }

  const handlePointerMove = (event: React.PointerEvent<HTMLButtonElement>) => {
    const drag = dragRef.current
    if (!drag) return
    const delta = (edge === "left" ? -1 : 1) * (event.clientX - drag.startX)
    if (!drag.moved && Math.abs(delta) < DRAG_SLOP) return
    drag.moved = true

    // Screen px to layout px, so the edge stays under the pointer at any scale.
    const next = drag.startWidth + delta / scaleRef.current
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
    // Capped: an outward pull cannot express intent past the cap, so do not lower the stored preference.
    if (stored > max && rawRef.current >= max) return
    // Commit the raw request, not the capped paint, so narrow windows keep the preference.
    setWidth(rawRef.current)
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLButtonElement>) => {
    // Keyboard collapse/expand; pointer-up handles the mouse.
    if (event.key === "Enter" || event.key === " ") {
      // preventDefault cancels the native click; arming here would swallow the next AT click.
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

  const handle = (
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
            // Switch and voice control dispatch a bare click with no pointer or key events.
            if (Date.now() - handledAtRef.current < CLICK_COMPAT_WINDOW_MS) return
            onToggle()
          }}
          onPointerEnter={() => {
            setHovered(true)
            onHoverChange?.(true)
          }}
          onPointerLeave={() => {
            setHovered(false)
            onHoverChange?.(false)
          }}
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
            "after:absolute after:inset-y-0 after:w-px after:bg-transparent after:transition-colors after:duration-150",
            edge === "left" ? "after:left-1" : "after:right-1",
            "hover:after:bg-sidebar-ring/25 data-dragging:after:bg-sidebar-ring/25",
            // The app zeroes the native outline on buttons, so mark focus here.
            "focus-visible:outline-none focus-visible:after:bg-sidebar-ring/60",
            className,
          )}
        />
  )

  if (hideTooltip) {
    return handle
  }

  return (
    <Tooltip open={(hovered || focused) && !dragging}>
      <TooltipTrigger asChild>{handle}</TooltipTrigger>
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
