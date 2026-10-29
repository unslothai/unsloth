// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type UIEvent, useCallback, useEffect, useState } from "react";

import { cn } from "@/lib/utils";

/** Pass `attach` as the ref and pair with a mask class such as `.panel-scroll-fade`. */
export function useScrollFades() {
  // State, not a ref, so the observer re-attaches when the node changes.
  const [node, setNode] = useState<HTMLElement | null>(null);
  const [scrolled, setScrolled] = useState(false);
  const [moreBelow, setMoreBelow] = useState(false);

  const update = useCallback((el: HTMLElement) => {
    const top = el.scrollTop > 0;
    setScrolled((prev) => (prev === top ? prev : top));
    const below = el.scrollHeight - el.scrollTop - el.clientHeight > 1;
    setMoreBelow((prev) => (prev === below ? prev : below));
  }, []);

  // ResizeObserver fires once on observe, which seeds the state.
  useEffect(() => {
    if (!node) {
      return;
    }
    const observer = new ResizeObserver(() => update(node));
    observer.observe(node);
    if (node.firstElementChild) {
      observer.observe(node.firstElementChild);
    }
    return () => observer.disconnect();
  }, [node, update]);

  return {
    attach: setNode,
    onScroll: (e: UIEvent<HTMLElement>) => update(e.currentTarget),
    className: cn(scrolled && "is-scrolled", moreBelow && "is-bottom-faded"),
  };
}
