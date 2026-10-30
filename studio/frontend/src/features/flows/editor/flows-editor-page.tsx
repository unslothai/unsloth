// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import "@xyflow/react/dist/style.css";
import "./flows-editor.css";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty";
import { useOptionalSidebar } from "@/components/ui/sidebar";
import { Spinner } from "@/components/ui/spinner";
import { shouldUseNativeMacWindowTitlebar } from "@/components/tauri/window-titlebar";
import { useBlocker, useNavigate } from "@tanstack/react-router";
import { ReactFlowProvider } from "@xyflow/react";
import { useReactFlow } from "@xyflow/react";
import { type ReactElement, useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { getFlow } from "../api";
import { ensureCatalog } from "../stores/catalog-loader";
import { subscribeResidentStatusRefresh } from "@/features/hub/lib/resident-status-refresh";
import { needsNewerStudio } from "../stores/catalog-store";
import { documentSignature, useDocStore } from "../stores/doc-store";
import { usePinsStore } from "../stores/pins-store";
import { useUiStore } from "../stores/ui-store";
import { FlowCanvas } from "./canvas";
import { CapabilityBanner } from "./capability-banner";
import { Inspector } from "./inspector";
import { addNodeConnected } from "./node-factory";
import { Palette } from "./palette";
import { placementPosition } from "./placement";
import { QuickAdd } from "./quick-add";
import { detachRun, hydrateRuntime } from "./run-controller";
import { syncResidentModel } from "./resident-sync";
import { RunBar } from "./run-bar";
import { RunDrawer } from "./run-drawer";
import { Toolbar } from "./toolbar";
import {
  beginSaveSession,
  endSaveSession,
  flushSave,
  useAutosave,
} from "./use-autosave";
import { useEditorShortcuts } from "./use-editor-shortcuts";

const SAVE_WAIT_MS = 10_000;
// The shell gives this chat-like route no top inset; start below the titlebar on Mac and
// Windows the way Train and Recipe Studio do.
const BELOW_TITLEBAR =
  "pt-[var(--studio-non-chat-content-top-inset,var(--studio-content-top-inset,0px))]";

/** Whether to stay: saves first, and a failed or stuck save asks instead of leaving the user stuck. */
function stayUnlessSaved(): Promise<boolean> {
  const wait = new Promise<"stuck">((resolve) =>
    setTimeout(() => resolve("stuck"), SAVE_WAIT_MS),
  );
  return Promise.race([flushSave(), wait]).then((saved) => {
    if (saved === true) return false;
    return new Promise<boolean>((resolve) => {
      toast.warning(
        saved === "stuck" ? "Still saving this flow" : "This flow isn't saved",
        {
          description: "Leave anyway and lose the unsaved changes, or stay.",
          duration: 15_000,
          action: { label: "Leave anyway", onClick: () => resolve(false) },
          onDismiss: () => resolve(true),
          onAutoClose: () => resolve(true),
        },
      );
    });
  });
}

type LoadState =
  | { status: "loading" }
  | { status: "ready"; readOnly: boolean }
  | { status: "error"; message: string };

function EditorBody({
  flowId,
  readOnly,
}: { flowId: string; readOnly: boolean }) {
  const navigate = useNavigate();
  const paletteOpen = useUiStore((s) => s.paletteOpen);
  const { screenToFlowPosition } = useReactFlow();
  const sidebar = useOptionalSidebar();
  const sidebarBeside = !!sidebar?.open && !sidebar.isMobile;
  // On Mac the toolbar takes the titlebar row itself, on the canvas colour, as chat's header does.
  const [macChrome] = useState(shouldUseNativeMacWindowTitlebar);
  useEditorShortcuts(!readOnly);
  useAutosave(!readOnly);
  // Every way out (the window's back arrow, a link, the sidebar, Play) saves first.
  useBlocker({ shouldBlockFn: stayUnlessSaved, enableBeforeUnload: false });

  const dropAt = useCallback(
    (type: string, x: number, y: number) => {
      const target = document.elementFromPoint(x, y);
      if (!target?.closest('[data-testid="flows-canvas"]')) return;
      const snap = useUiStore.getState().snapToGrid;
      addNodeConnected(
        type,
        placementPosition(screenToFlowPosition({ x, y }), snap),
      );
    },
    [screenToFlowPosition],
  );

  return (
    <div
      // The titlebar strip and toolbar share the sidebar's colour, so the top reads as one surface with it.
      className={`flows-editor flex h-full min-h-0 w-full flex-col text-foreground ${macChrome ? "bg-background" : `bg-sidebar ${BELOW_TITLEBAR}`}`}
      data-testid="flows-editor"
    >
      <Toolbar
        readOnly={readOnly}
        titlebar={macChrome}
        clearNav={macChrome && !(sidebar?.pinned ?? true)}
      />
      {readOnly && (
        <Alert className="rounded-none border-x-0 border-t-0 py-2">
          <AlertDescription className="text-xs text-amber-700 dark:text-amber-300">
            This flow uses blocks this version of Unsloth Studio does not know,
            so it opens read-only. Update Studio to edit or run it.
          </AlertDescription>
        </Alert>
      )}
      {!readOnly && <CapabilityBanner flowId={flowId} />}
      <div
        // Curves into the app sidebar only while it is open beside it.
        className={`relative flex min-h-0 flex-1 overflow-hidden bg-background transition-[border-radius] duration-200 ease-out ${sidebarBeside && !macChrome ? "rounded-tl-xl" : "rounded-tl-none"}`}
      >
        {paletteOpen && !readOnly && <Palette onDrop={dropAt} />}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="relative min-h-0 flex-1">
            <FlowCanvas readOnly={readOnly} />
          </div>
          <RunDrawer />
        </div>
        <Inspector />
        {/* Over the whole editor body, not the canvas, so it stays put as panels open. */}
        <RunBar
          onPlay={() =>
            void navigate({ to: "/flows/$flowId/play", params: { flowId } })
          }
          onApp={() =>
            void navigate({ to: "/flows/$flowId/app", params: { flowId } })
          }
        />
      </div>
      <QuickAdd />
    </div>
  );
}

export function FlowsEditorPage({ flowId }: { flowId: string }): ReactElement {
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    syncResidentModel();
    return subscribeResidentStatusRefresh(syncResidentModel);
  }, []);

  useEffect(() => {
    let alive = true;
    setState({ status: "loading" });
    useUiStore.getState().reset();
    Promise.all([ensureCatalog(), getFlow(flowId)])
      .then(([, record]) => {
        if (!alive) return;
        useDocStore.getState().load(record.doc);
        usePinsStore.getState().load(record.id);
        const readOnly = record.doc.nodes.some(needsNewerStudio);
        if (!readOnly) {
          beginSaveSession(
            record.id,
            record.updatedAt,
            documentSignature(record.doc),
          );
        }
        setState({ status: "ready", readOnly });
        void hydrateRuntime(record.id);
      })
      .catch((error: unknown) => {
        if (alive)
          setState({
            status: "error",
            message:
              error instanceof Error
                ? error.message
                : "Could not open the flow",
          });
      });
    return () => {
      alive = false;
      detachRun();
      endSaveSession();
    };
  }, [flowId]);

  if (state.status === "loading") {
    return (
      <div className="flex h-full flex-1 items-center justify-center gap-2 bg-background text-sm text-muted-foreground">
        <Spinner /> Opening flow…
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <Empty className="h-full bg-background">
        <EmptyHeader>
          <EmptyTitle>Could not open the flow</EmptyTitle>
          <EmptyDescription>{state.message}</EmptyDescription>
        </EmptyHeader>
        <Button variant="outline" asChild={true}>
          <a href="/flows">Back to flows</a>
        </Button>
      </Empty>
    );
  }
  return (
    <ReactFlowProvider>
      <EditorBody flowId={flowId} readOnly={state.readOnly} />
    </ReactFlowProvider>
  );
}
