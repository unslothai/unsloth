// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import "@xyflow/react/dist/style.css";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyTitle,
} from "@/components/ui/empty";
import { Spinner } from "@/components/ui/spinner";
import {
  holdSidebarUnpinned,
  releaseSidebarPinned,
} from "@/hooks/use-sidebar-pin";
import { useNavigate } from "@tanstack/react-router";
import { ReactFlowProvider } from "@xyflow/react";
import { useReactFlow } from "@xyflow/react";
import { type ReactElement, useCallback, useEffect, useState } from "react";
import { toast } from "sonner";
import { getFlow } from "../api";
import { ensureCatalog } from "../stores/catalog-loader";
import { needsNewerStudio } from "../stores/catalog-store";
import { documentSignature, useDocStore } from "../stores/doc-store";
import { usePinsStore } from "../stores/pins-store";
import { useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import { FlowCanvas } from "./canvas";
import { CapabilityBanner } from "./capability-banner";
import { Inspector } from "./inspector";
import { addNodeConnected } from "./node-factory";
import { Palette } from "./palette";
import { placementPosition } from "./placement";
import { QuickAdd } from "./quick-add";
import { detachRun, hydrateRuntime } from "./run-controller";
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

/** Saves, then leaves; a failed or stuck save asks instead of leaving the user stuck. */
function leaveAfterSave(go: () => void): void {
  const wait = new Promise<"stuck">((resolve) =>
    setTimeout(() => resolve("stuck"), SAVE_WAIT_MS),
  );
  void Promise.race([flushSave(), wait]).then((saved) => {
    if (saved === true) return go();
    toast.warning(
      saved === "stuck" ? "Still saving this flow" : "This flow isn't saved",
      {
        description: "Leave anyway and lose the unsaved changes, or stay.",
        duration: 15_000,
        action: { label: "Leave anyway", onClick: go },
      },
    );
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
  const drawerOpen = useUiStore((s) => s.drawerOpen);
  const runStatus = useRuntimeStore((s) => s.runStatus);
  const { screenToFlowPosition } = useReactFlow();
  useEditorShortcuts(!readOnly);
  // The Nodes panel owns the left edge here: the app sidebar unpins (Desktop peeks it from the
  // window edge) without touching the saved pin, which comes back on leaving.
  useEffect(() => {
    holdSidebarUnpinned();
    return releaseSidebarPinned;
  }, []);
  useAutosave(!readOnly);

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
      className="flows-editor flex h-full min-h-0 w-full flex-col bg-background text-foreground"
      data-testid="flows-editor"
    >
      <Toolbar
        onBack={() => leaveAfterSave(() => void navigate({ to: "/flows" }))}
        onPlay={() =>
          leaveAfterSave(
            () =>
              void navigate({ to: "/flows/$flowId/play", params: { flowId } }),
          )
        }
        onApp={() =>
          leaveAfterSave(
            () =>
              void navigate({ to: "/flows/$flowId/app", params: { flowId } }),
          )
        }
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
      <div className="flex min-h-0 flex-1">
        {paletteOpen && !readOnly && <Palette onDrop={dropAt} />}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="relative min-h-0 flex-1">
            <FlowCanvas readOnly={readOnly} />
            {!drawerOpen && runStatus && (
              <Button
                type="button"
                variant="outline"
                size="xs"
                onClick={() => useUiStore.getState().setDrawer(true, "run")}
                className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full capitalize shadow-sm"
              >
                Run: {runStatus}
              </Button>
            )}
          </div>
          <RunDrawer />
        </div>
        <Inspector />
      </div>
      <QuickAdd />
    </div>
  );
}

export function FlowsEditorPage({ flowId }: { flowId: string }): ReactElement {
  const [state, setState] = useState<LoadState>({ status: "loading" });

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
