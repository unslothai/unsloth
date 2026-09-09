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
/** Arrow-key resize step for keyboard users. */
const RESIZE_STEP = 16

// A drag needs one cursor over the whole viewport, because the pointer travels
// across buttons and text that would otherwise claim their own. That used to be
// `html[data-panel-resizing] *` plus `cursor`/`user-select` on <body>. Both
// reach every element in the document: the universal selector matches all of
// them, and `cursor` and `user-select` are inherited, so writing them on <body>
// marks inherited style dirty for everything below it. The cost is therefore
// proportional to the STANDING DOM rather than to the one thing that changed,
// which is the same shape as the sidebar-width writes scoped in #9400/#9441.
//
// The same is true of the rule that blanked pointer events on the sidebar and
// on [data-slot="sidebar-inset"], the <main> holding the whole app including
// the thread: `pointer-events` is inherited too, so that write dirtied the
// thread's subtree on both flips as well.
//
// A single fixed element on top of the viewport does both jobs with an
// invalidation set of one element. It carries the cursor, and by being the hit
// test target for the whole viewport it keeps hover and click off the content
// underneath. It is transparent and it is removed the instant the drag ends, so
// nothing about what the user sees changes.
const DRAG_OVERLAY_SLOT = "panel-resize-drag-overlay"
/** Nested drags cannot happen through pointer capture, but a stuck overlay would
 *  swallow the whole UI, so ownership is explicit rather than assumed. */
let dragOverlayOwners = 0

function acquireDragOverlay(): void {
  dragOverlayOwners += 1
  if (dragOverlayOwners > 1) return
  const el = document.createElement("div")
  el.setAttribute("data-slot", DRAG_OVERLAY_SLOT)
  // Decorative and non-interactive as far as assistive tech is concerned: it
  // exists only to own the cursor while the pointer is already captured.
  el.setAttribute("aria-hidden", "true")
  const s = el.style
  s.position = "fixed"
  s.inset = "0"
  // Top of the named scale, not a hand-picked large number: the rules it
  // replaces were `!important` and blanked whole subtrees, so anything it did
  // not out-rank it would only partly stand in for.
  s.zIndex = String(Z_LAYER.DRAG_CURSOR_OVERLAY)
  s.background = "transparent"
  // Explicit, because it is load-bearing rather than incidental. Being the hit
  // test target for the whole viewport is what keeps hover and click off the
  // content underneath, which is the job `pointer-events: none` on
  // [data-slot="sidebar-inset"] used to do by dirtying the thread's subtree.
  s.pointerEvents = "auto"
  // col-resize unconditionally, which is what the replaced rule did even when a
  // collapsed edge was being dragged open.
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
  /**
   * Narrower element for `cssVar`, replacing `target()` under
   * PANEL_RESIZE_SCOPED_VARS_ENABLED. Must hold every consumer and nothing
   * else: the write invalidates inherited style for everything below it.
   */
  scopedTarget?: () => HTMLElement | null
  /**
   * The elements that actually read `rootVar`, replacing
   * `document.documentElement` under PANEL_RESIZE_SCOPED_VARS_ENABLED. Empty
   * means no consumer, so the write is skipped.
   */
  rootVarTargets?: () => HTMLElement[]
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

  // Where `rootVar` is painted, resolved once on pointer down. Always
  // [document.documentElement] with the flag off, which is what shipped.
  const rootTargetsRef = React.useRef<HTMLElement[]>([])
  // Whether THIS handle is holding the cursor overlay. endDrag also runs as the
  // effect cleanup, where no drag happened and nothing was acquired.
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
    // Resolved once per drag, never per frame: the write is what costs, and a
    // DOM walk here is already how `target()` worked.
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
