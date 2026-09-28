// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Real panels and stores; the browser test supplies deterministic API responses.
import { createRoot } from "react-dom/client";
import {
  createMemoryHistory,
  createRootRoute,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router";
import { FloatingMonitor } from "@/components/floating-monitor";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ApiMonitorOverlay } from "@/features/api-monitor/api-monitor-overlay";
import { useApiMonitorOverlayStore } from "@/features/api-monitor/overlay-store";
import { useMonitorOverlayStore } from "@/features/settings";
import "./src/index.css";

useApiMonitorOverlayStore.getState().setAutoOpen(false);
useMonitorOverlayStore.getState().setIsOpen(false);

function Harness() {
  return (
    <TooltipProvider>
      <div className="flex gap-4 p-4">
        <button onClick={() => useApiMonitorOverlayStore.getState().open()}>
          Open API
        </button>
        <button
          onClick={() => useMonitorOverlayStore.getState().setIsOpen(true)}
        >
          Open hardware
        </button>
      </div>
      <FloatingMonitor />
      <ApiMonitorOverlay />
    </TooltipProvider>
  );
}

const routeTree = createRootRoute({ component: Harness });
const router = createRouter({
  routeTree,
  history: createMemoryHistory({ initialEntries: ["/"] }),
});
createRoot(document.getElementById("root")!).render(
  <RouterProvider router={router} />,
);
