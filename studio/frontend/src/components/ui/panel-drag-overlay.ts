// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Z_LAYER } from "@/lib/z-layers"

// One fixed overlay owns the drag cursor and hit testing. Not `html[data-panel-resizing] *` or
// inherited cursor / user-select / pointer-events on <body> or the inset: those restyle the whole DOM.
const DRAG_OVERLAY_SLOT = "panel-resize-drag-overlay"
const PANEL_RESIZING_ATTRIBUTE = "data-panel-resizing"
let dragOverlayOwners = 0
let resizingHolders = 0
const DRAG_SLOP = 4

function holdResizing(): void {
  resizingHolders += 1
  if (resizingHolders > 1) return
  document.documentElement.setAttribute(PANEL_RESIZING_ATTRIBUTE, "true")
}

function releaseResizing(): void {
  if (resizingHolders === 0) return
  resizingHolders -= 1
  if (resizingHolders > 0) return
  document.documentElement.removeAttribute(PANEL_RESIZING_ATTRIBUTE)
}

export function acquireDragOverlay(): void {
  dragOverlayOwners += 1
  if (dragOverlayOwners > 1) return
  holdResizing()
  const el = document.createElement("div")
  el.setAttribute("data-slot", DRAG_OVERLAY_SLOT)
  el.setAttribute("aria-hidden", "true")
  const s = el.style
  s.position = "fixed"
  s.inset = "0"
  s.zIndex = String(Z_LAYER.DRAG_CURSOR_OVERLAY)
  s.background = "transparent"
  // The hit target for the whole viewport: keeps hover and click off the content below.
  s.pointerEvents = "auto"
  s.cursor = "col-resize"
  s.userSelect = "none"
  s.touchAction = "none"
  document.body.appendChild(el)
}

export function releaseDragOverlay(): void {
  if (dragOverlayOwners === 0) return
  dragOverlayOwners -= 1
  if (dragOverlayOwners > 0) return
  document
    .querySelector(`[data-slot="${DRAG_OVERLAY_SLOT}"]`)
    ?.remove()
  releaseResizing()
}

export function armPanelDrag(startX: number): {
  move: (x: number) => void
  release: () => void
} {
  holdResizing()
  let overlaid = false
  let released = false
  return {
    move: (x) => {
      if (overlaid || released || Math.abs(x - startX) < DRAG_SLOP) return
      overlaid = true
      acquireDragOverlay()
    },
    release: () => {
      if (released) return
      released = true
      if (overlaid) releaseDragOverlay()
      releaseResizing()
    },
  }
}

export function panelDragInProgress(): boolean {
  return dragOverlayOwners > 0
}
