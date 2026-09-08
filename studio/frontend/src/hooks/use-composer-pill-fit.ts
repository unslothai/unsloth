// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useLayoutEffect, useState } from "react";

/**
 * Value for the pill row's `data-pill-compact` attribute.
 * - `undefined`: every pill keeps its label.
 * - `"first"`: only the leading permission pill drops to its icon.
 * - `"true"`: every pill drops to its icon (the existing compact look).
 */
export type PillCompact = undefined | "first" | "true";

/** Escalation order: give up the least label space that still fits. */
const STAGES: PillCompact[] = [undefined, "first", "true"];

function applyStage(row: HTMLElement, stage: PillCompact) {
  if (stage === undefined) {
    row.removeAttribute("data-pill-compact");
  } else {
    row.setAttribute("data-pill-compact", stage);
  }
}

/**
 * True while the row is one visual line: not wrapped inside itself, and nothing
 * after it (the dictate/send cluster) pushed underneath it.
 */
function fitsOnOneLine(row: HTMLElement) {
  const rect = row.getBoundingClientRect();
  // Tallest child is one line; a wrapped row is about double. Measured, not a
  // constant, so the UI font-scale setting cannot break the check.
  let lineHeight = 0;
  for (const child of Array.from(row.children)) {
    lineHeight = Math.max(lineHeight, child.getBoundingClientRect().height);
  }
  if (lineHeight === 0) {
    return true;
  }
  if (rect.height > lineHeight + 4) {
    return false;
  }
  for (
    let sibling = row.nextElementSibling;
    sibling;
    sibling = sibling.nextElementSibling
  ) {
    const siblingRect = sibling.getBoundingClientRect();
    // Skips the hidden file inputs, and the textarea that flex `order` moves
    // to the row above in single chat.
    if (siblingRect.width === 0) {
      continue;
    }
    if (siblingRect.top >= rect.bottom - 2) {
      return false;
    }
  }
  return true;
}

/**
 * Collapses `row` by the least amount that keeps it on one line, writes the
 * winning `data-pill-compact` value to it, and returns that value.
 * `forceCompact` skips measuring: nothing narrower is on offer.
 *
 * Exported for the unit test; components use the hook below.
 */
export function measurePillCompact(
  row: HTMLElement,
  forceCompact: boolean,
): PillCompact {
  if (forceCompact) {
    applyStage(row, "true");
    return "true";
  }
  // Falls through to the narrowest stage when even that overflows.
  let fitted: PillCompact = "true";
  for (const stage of STAGES) {
    applyStage(row, stage);
    // Reading geometry after the write forces the reflow this needs.
    if (fitsOnOneLine(row)) {
      fitted = stage;
      break;
    }
  }
  applyStage(row, fitted);
  return fitted;
}

/**
 * Keeps the composer's tool pills on one line with the dictate/send controls.
 *
 * The count rule (`forceCompact`) cannot see label widths, so four long labels
 * ("Run automatically" next to "Deep research") still overflowed and dropped
 * the mic and send button onto a second line. This measures the laid-out row
 * and collapses only as far as it takes to fit.
 *
 * Returns a callback ref for the row and the `data-pill-compact` value to
 * render on it, so React keeps the attribute the measurement settled on.
 */
export function useComposerPillFit(forceCompact: boolean) {
  const [row, setRow] = useState<HTMLElement | null>(null);
  const [compact, setCompact] = useState<PillCompact>(
    forceCompact ? "true" : undefined,
  );

  const measure = useCallback(
    (el: HTMLElement) => {
      const next = measurePillCompact(el, forceCompact);
      setCompact((prev) => (prev === next ? prev : next));
    },
    [forceCompact],
  );

  useLayoutEffect(() => {
    if (!row) {
      return;
    }
    // One measurement per frame: the observers also fire for the transient
    // sizes the escalation loop itself produces.
    let frame = 0;
    const schedule = () => {
      if (frame) {
        return;
      }
      frame = requestAnimationFrame(() => {
        frame = 0;
        measure(row);
      });
    };
    // First pass runs before paint, so the row never flashes wrapped.
    measure(row);
    const resize = new ResizeObserver(schedule);
    resize.observe(row);
    // The composer line owns the width the row has to fit into.
    if (row.parentElement) {
      resize.observe(row.parentElement);
    }
    // A model capability toggle adds or removes a pill without necessarily
    // resizing what the observer above watches.
    const mutations = new MutationObserver(schedule);
    mutations.observe(row, { childList: true });
    return () => {
      if (frame) {
        cancelAnimationFrame(frame);
      }
      resize.disconnect();
      mutations.disconnect();
    };
  }, [row, measure]);

  return { pillRowRef: setRow, pillCompact: compact };
}
