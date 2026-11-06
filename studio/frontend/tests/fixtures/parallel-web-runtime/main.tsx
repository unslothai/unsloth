// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import { useState } from "react";
import { createRoot } from "react-dom/client";
import { Dialog, DialogContent, DialogTitle } from "../../../src/components/ui/dialog";
import "../../../src/index.css";
// Discover the lazy runtime's Vite dependencies before the fixture starts.
import "../../../src/features/browser/web-runtime";
import { BrowserRuntimeRoot } from "../../../src/features/browser/browser-runtime-root";
import { AnnotationPortal } from "../../../src/features/browser/annotation-portal";
import { browserScopeContains } from "../../../src/features/browser/browser-scope";
import { WebAnnotateLayer } from "../../../src/features/browser/annotate-layer";
import { useBrowserAgentStore } from "../../../src/features/browser/agent/agent-store";
import { useBrowserStore } from "../../../src/features/browser/store";
import { requestFrameAgent, whenWebPageReady } from "../../../src/features/browser/page-frame";
function Fixture() {
  const state = useBrowserStore();
  const tab = state.tabs.find((item) => item.id === state.activeTabId);
  const [page, setPage] = useState<HTMLDivElement | null>(null);
  const [annotating, setAnnotating] = useState(false);
  const [observing, setObserving] = useState(false);
  const [watched, setWatched] = useState<string | null>(null);
  const [menu, setMenu] = useState(false);
  const [scaled, setScaled] = useState(false);
  Object.assign(window, { showAnnotation: setAnnotating, observe: setObserving, watchTab: setWatched, showMenu: setMenu, setPreviewScale: setScaled });
  return <><div id="slot" ref={setPage} style={{ position: "relative", width: scaled ? 300 : 600, height: scaled ? 200 : 400, overflow: "hidden" }}>
    <div data-browser-page="" style={{ width: 600, height: 400, transform: scaled ? "scale(0.5)" : undefined, transformOrigin: "top left", ...{ "--browser-floating-scale": scaled ? 0.5 : 1 } }}>
      {state.open && tab && <div data-web-page={tab.id} style={{ width: 600, height: "100%", display: watched ? "none" : undefined }} />}
    </div>
    {watched && <div data-web-observe={watched} style={{ width: "100%", height: "100%" }} />}
  </div><input id="foreground" /><BrowserRuntimeRoot />{annotating && page && tab && <AnnotationPortal page={page} tabId={tab.id} hidden={observing || !!watched}><WebAnnotateLayer tabId={tab.id} title="Test page" page={page} suspended={observing || !!watched} /></AnnotationPortal>}<Dialog open={menu} onOpenChange={setMenu}><DialogContent><DialogTitle>Test dialog</DialogTitle></DialogContent></Dialog></>;
}
Object.assign(window, { store: useBrowserStore, agents: useBrowserAgentStore, requestFrameAgent, whenWebPageReady, browserScopeContains });
createRoot(document.getElementById("root")!).render(<Fixture />);
