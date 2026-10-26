// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useLayoutEffect, useState } from "react";

/**
 * - `undefined`: every pill keeps its label.
 * - `"first"`: only the leading permission pill drops to its icon.
 * - `"true"`: every pill drops to its icon.
 */
export type PillCompact = undefined | "first" | "true";

const STAGES: PillCompact[] = [undefined, "first", "true"];

function applyStage(row: HTMLElement, stage: PillCompact) {
  if (stage === undefined) {
    row.removeAttribute("data-pill-compact");
  } else {
    row.setAttribute("data-pill-compact", stage);
  }
}

/** Not wrapped, and nothing after it (dictate/send) pushed underneath. */
function fitsOnOneLine(row: HTMLElement) {
  const rect = row.getBoundingClientRect();
  // Measured, not a constant, so the UI font-scale setting cannot break the check.
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
    // Skips hidden file inputs and the textarea that flex `order` moves above in single chat.
    if (siblingRect.width === 0) {
      continue;
    }
    if (siblingRect.top >= rect.bottom - 2) {
      return false;
    }
  }
  return true;
}

/** `forceCompact` skips measuring. Exported for the unit test. */
export function measurePillCompact(
  row: HTMLElement,
  forceCompact: boolean,
): PillCompact {
  if (forceCompact) {
    applyStage(row, "true");
    return "true";
  }
  let fitted: PillCompact = "true";
  for (const stage of STAGES) {
    applyStage(row, stage);
    // Reading geometry after the write forces the needed reflow.
    if (fitsOnOneLine(row)) {
      fitted = stage;
      break;
    }
  }
  applyStage(row, fitted);
  return fitted;
}

/** `forceCompact` counts pills but cannot see label widths, so measure the laid-out row. */
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
    // One measurement per frame: observers also fire for the escalation loop's own sizes.
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
    // Before paint, so the row never flashes wrapped.
    measure(row);
    const resize = new ResizeObserver(schedule);
    resize.observe(row);
    if (row.parentElement) {
      resize.observe(row.parentElement);
    }
    // A capability toggle can add a pill without resizing anything observed.
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
