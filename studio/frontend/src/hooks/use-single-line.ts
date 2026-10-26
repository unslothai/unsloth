// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useCallback, useEffect, useRef, useState } from "react";

function fitsOneLine(el: HTMLElement): boolean {
  const style = getComputedStyle(el);
  const fontSize = Number.parseFloat(style.fontSize) || 16;
  const lineHeight = Number.parseFloat(style.lineHeight) || fontSize * 1.2;
  const content =
    el.clientHeight -
    (Number.parseFloat(style.paddingTop) || 0) -
    (Number.parseFloat(style.paddingBottom) || 0);
  return content < lineHeight * 1.5;
}

/**
 * Whether the element's text currently fits on one line, tracked across resizes. A box that is
 * usually one line but can wrap (a command, a URL) is a pill on one line and a 16px box once
 * wrapped; CSS cannot tell the two apart.
 */
export function useSingleLine<T extends HTMLElement>(): [
  (node: T | null) => void,
  boolean,
] {
  const [single, setSingle] = useState(true);
  const observer = useRef<ResizeObserver | null>(null);

  const ref = useCallback((node: T | null) => {
    observer.current?.disconnect();
    observer.current = null;
    if (!node || typeof ResizeObserver === "undefined") {
      return;
    }
    const update = () => setSingle(fitsOneLine(node));
    update();
    observer.current = new ResizeObserver(update);
    observer.current.observe(node);
  }, []);

  useEffect(() => () => observer.current?.disconnect(), []);

  return [ref, single];
}
