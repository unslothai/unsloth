// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Production project dialog, with native IPC and HTTP supplied by the Playwright driver.
import { createRoot } from "react-dom/client";
import { useState } from "react";
import {
  createMemoryHistory,
  createRootRoute,
  createRouter,
  RouterProvider,
} from "@tanstack/react-router";
import { NewProjectDialog } from "@/features/chat/components/new-project-dialog";
import { EditProjectDialog } from "@/features/chat/components/edit-project-dialog";
import { getChatProject } from "@/features/chat/api/chat-api";
import type { ProjectRecord } from "@/features/chat/types";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "sonner";
import { setApiBase } from "@/lib/api-base";
import { initializeLocale } from "@/i18n";
import "./src/index.css";

setApiBase(Number(location.port));
initializeLocale();
function App() {
  const [open, setOpen] = useState(true);
  const [created, setCreated] = useState<ProjectRecord | null>(null);
  const [editing, setEditing] = useState(false);
  return (
    <TooltipProvider>
      <div className="aui-root" style={{ minHeight: "100vh" }}>
        <NewProjectDialog
          open={open}
          onOpenChange={setOpen}
          onCreated={setCreated}
        />
        <p data-testid="created">{created?.id}</p>
        {created && (
          <button
            onClick={async () => {
              setCreated(await getChatProject(created.id));
              setEditing(true);
            }}
          >
            Edit project
          </button>
        )}
        <EditProjectDialog
          project={editing ? created : null}
          onOpenChange={setEditing}
          onDelete={() => {
            setEditing(false);
            setCreated(null);
          }}
        />
        <Toaster />
      </div>
    </TooltipProvider>
  );
}
const root = createRootRoute({ component: App });
const router = createRouter({
  routeTree: root,
  history: createMemoryHistory({ initialEntries: ["/"] }),
});
createRoot(document.getElementById("root")!).render(
  <RouterProvider router={router} />,
);
