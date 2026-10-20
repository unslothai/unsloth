// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { createRoot } from "react-dom/client";
import { createMemoryHistory, createRootRoute, createRouter, RouterProvider } from "@tanstack/react-router";
import { LinkedFoldersManager } from "../../../src/features/rag/components/linked-folders-manager";
import { setApiBase } from "@/lib/api-base";
import { initializeLocale } from "@/i18n";
import { Toaster } from "sonner";
import "../../../src/index.css";

setApiBase(Number(location.port));
initializeLocale();
export function App() {
  return <main className="aui-root mx-auto max-w-xl p-4">
    <LinkedFoldersManager scope={{ type: "knowledge_base", id: "knowledge" }} />
    <Toaster />
  </main>;
}
const router = createRouter({
  routeTree: createRootRoute({ component: App }),
  history: createMemoryHistory({ initialEntries: ["/"] }),
});
createRoot(document.getElementById("root")!).render(<RouterProvider router={router} />);
