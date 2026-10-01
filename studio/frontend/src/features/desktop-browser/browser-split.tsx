// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { ResizableHandle, ResizablePanel, ResizablePanelGroup } from "@/components/ui/resizable";
import { isTauri } from "@/lib/api-base";
import { useCallback, useEffect, useRef, type ReactNode } from "react";
import type { PanelImperativeHandle } from "react-resizable-panels";
import { BrowserPane } from "./browser-pane";

export function BrowserSplit({ children, open, onClose }: { children: ReactNode; open: boolean; onClose: () => void }) {
  const chatRef = useRef<PanelImperativeHandle | null>(null);
  const measureRef = useRef<(() => void) | null>(null);
  const frameRef = useRef(0);
  const resampleBrowser = useCallback(() => {
    if (frameRef.current) return;
    frameRef.current = requestAnimationFrame(() => {
      frameRef.current = 0;
      measureRef.current?.();
    });
  }, []);
  useEffect(() => {
    if (!isTauri || !open) return;
    const frame = requestAnimationFrame(() => {
      chatRef.current?.resize("50%");
      // Initial registration can precede the split's first layout. Sample after
      // the browser panel has actually expanded, even if ResizeObserver misses it.
      resampleBrowser();
    });
    return () => cancelAnimationFrame(frame);
  }, [open, resampleBrowser]);
  useEffect(() => () => { if (frameRef.current) cancelAnimationFrame(frameRef.current); }, []);
  if (!isTauri) return children;
  // Keep the handle's scoped cursor; do not rewrite a document-wide cursor stylesheet.
  return (
    <ResizablePanelGroup orientation="horizontal" disableCursor onLayoutChange={resampleBrowser} onLayoutChanged={resampleBrowser} className="min-h-0 min-w-0 flex-1 basis-0 overflow-hidden" data-testid="desktop-browser-layout">
      <ResizablePanel id="desktop-chat" panelRef={chatRef} defaultSize="100%" minSize={open ? "320px" : "100%"} className="flex min-h-0 min-w-0 flex-col overflow-hidden">
        {children}
      </ResizablePanel>
      {open ? (
        <>
          <ResizableHandle id="desktop-browser-splitter" aria-label="Resize chat and browser" />
          <ResizablePanel id="desktop-browser" defaultSize="50%" minSize="320px" className="min-h-0 min-w-0 overflow-hidden">
            <BrowserPane onClose={onClose} measureRef={measureRef} />
          </ResizablePanel>
        </>
      ) : null}
    </ResizablePanelGroup>
  );
}
