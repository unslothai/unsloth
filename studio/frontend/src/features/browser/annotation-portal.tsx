// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ReactNode, useLayoutEffect, useState } from "react";
import { createPortal } from "react-dom";
import { BROWSER_ANNOTATION_ROOT } from "./browser-runtime-root";

export function AnnotationPortal({ page, tabId, hidden, children }: { page: HTMLElement; tabId: string; hidden: boolean; children: ReactNode }) {
  const [host] = useState(() => document.createElement("div"));
  useLayoutEffect(() => {
    let frame = 0;
    host.setAttribute("data-browser-annotation-portal", "");
    host.setAttribute("data-browser-scope-tab", tabId);
    Object.assign(host.style, { position: "fixed", zIndex: "30", overflow: "hidden", pointerEvents: "none" });
    const measure = () => {
      frame = 0;
      const box = page.getBoundingClientRect();
      Object.assign(host.style, { left: `${box.left}px`, top: `${box.top}px`, width: `${box.width}px`, height: `${box.height}px` });
    };
    const schedule = () => { if (!host.hidden && !frame) frame = requestAnimationFrame(measure); };
    const observer = new ResizeObserver(schedule);
    observer.observe(page);
    document.getElementById(BROWSER_ANNOTATION_ROOT)!.append(host);
    measure();
    window.addEventListener("resize", schedule);
    window.addEventListener("scroll", schedule, true);
    const timer = window.setInterval(schedule, 150);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", schedule);
      window.removeEventListener("scroll", schedule, true);
      window.clearInterval(timer);
      cancelAnimationFrame(frame);
      host.remove();
    };
  }, [page, host, tabId]);
  useLayoutEffect(() => {
    if (hidden && host.contains(document.activeElement) && document.activeElement instanceof HTMLElement) document.activeElement.blur();
    host.toggleAttribute("hidden", hidden);
    host.toggleAttribute("inert", hidden);
  }, [hidden, host]);
  return createPortal(children, host);
}
