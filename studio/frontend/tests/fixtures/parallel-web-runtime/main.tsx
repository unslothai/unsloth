// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import { useState } from "react";
import { createRoot } from "react-dom/client";
import { Dialog, DialogContent, DialogTitle } from "../../../src/components/ui/dialog";
import "../../../src/index.css";
import { WebRuntime } from "../../../src/features/browser/web-runtime";
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
  const [menu, setMenu] = useState(false);
  Object.assign(window, { showAnnotation: setAnnotating, observe: setObserving, showMenu: setMenu });
  return <><div id="slot" ref={setPage} style={{ position: "relative", width: 600, height: 400 }}>{state.open && tab && <div data-web-page={tab.id} style={{ width: "100%", height: "100%" }} />}</div><input id="foreground" /><WebRuntime />{annotating && page && tab && <AnnotationPortal page={page} tabId={tab.id} hidden={observing}><WebAnnotateLayer tabId={tab.id} title="Test page" page={page} suspended={observing} /></AnnotationPortal>}<Dialog open={menu} onOpenChange={setMenu}><DialogContent><DialogTitle>Test dialog</DialogTitle></DialogContent></Dialog></>;
}
Object.assign(window, { store: useBrowserStore, agents: useBrowserAgentStore, requestFrameAgent, whenWebPageReady, browserScopeContains });
createRoot(document.getElementById("root")!).render(<Fixture />);
