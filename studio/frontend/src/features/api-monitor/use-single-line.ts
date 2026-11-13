// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useLayoutEffect, useState } from "react";

/** Callback ref, so a box that mounts later is measured too; ResizeObserver catches wrapping. */
export function useSingleLine<T extends HTMLElement>(): [
  (element: T | null) => void,
  boolean,
] {
  const [element, setElement] = useState<T | null>(null);
  const [single, setSingle] = useState(true);
  useLayoutEffect(() => {
    if (!element) return;
    const measure = () => {
      const style = getComputedStyle(element);
      const line = Number.parseFloat(style.lineHeight);
      const padding =
        Number.parseFloat(style.paddingTop) +
        Number.parseFloat(style.paddingBottom);
      setSingle(element.scrollHeight - padding <= line * 1.5);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [element]);
  return [setElement, single];
}
