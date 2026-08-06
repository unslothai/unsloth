


import { type UIEvent, useCallback, useEffect, useState } from "react";

import { cn } from "@/lib/utils";

/**
 * Edge fades for a scroll container: the top fades once scrolled, the bottom while more
 * content sits below. Pass `attach` as the element's ref and pair with a mask class such
 * as `.panel-scroll-fade`.
 */
export function useScrollFades() {
  // The node is state, not a ref, so the observer re-attaches when it changes.
  const [node, setNode] = useState<HTMLDivElement | null>(null);
  const [scrolled, setScrolled] = useState(false);
  const [moreBelow, setMoreBelow] = useState(false);

  const update = useCallback((el: HTMLElement) => {
    const top = el.scrollTop > 0;
    setScrolled((prev) => (prev === top ? prev : top));
    const below = el.scrollHeight - el.scrollTop - el.clientHeight > 1;
    setMoreBelow((prev) => (prev === below ? prev : below));
  }, []);

  // ResizeObserver fires once on observe, so mounting seeds the state too.
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
