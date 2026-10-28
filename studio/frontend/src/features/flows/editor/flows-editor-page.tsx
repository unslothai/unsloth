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
import { useNavigate } from "@tanstack/react-router";
import { ReactFlowProvider } from "@xyflow/react";
import { useReactFlow } from "@xyflow/react";
import {
  type ReactElement,
  Suspense,
  lazy,
  useCallback,
  useEffect,
  useState,
} from "react";
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

const RecipeEditor = lazy(() => import("../recipe/recipe-editor"));
const RecipeRuns = lazy(() => import("../recipe/recipe-runs"));

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
  const recipeEditorOpen = useUiStore((s) => s.recipeEditor !== null);
  const recipeRunsOpen = useUiStore((s) => s.recipeRuns !== null);
  const runStatus = useRuntimeStore((s) => s.runStatus);
  const { screenToFlowPosition } = useReactFlow();
  useEditorShortcuts(!readOnly);
  useAutosave(!readOnly);

  const addAtCenter = useCallback(
    (type: string) => {
      const pane = document
        .querySelector('[data-testid="flows-canvas"]')
        ?.getBoundingClientRect();
      const center = pane
        ? screenToFlowPosition({
            x: pane.left + pane.width / 2,
            y: pane.top + pane.height / 2,
          })
        : { x: 0, y: 0 };
      const jitter = () => Math.round((Math.random() - 0.5) * 60);
      const node = addNodeConnected(type, [
        Math.round(center.x - 130 + jitter()),
        Math.round(center.y - 60 + jitter()),
      ]);
      if (node) {
        useUiStore.getState().setSelection([node.id], []);
        useUiStore.getState().openInspector(node.id);
      }
    },
    [screenToFlowPosition],
  );

  return (
    <div
      className="flows-editor flex h-full min-h-0 w-full flex-col bg-background text-foreground"
      data-testid="flows-editor"
    >
      <Toolbar
        onBack={() =>
          void flushSave().then((saved) => {
            if (saved) void navigate({ to: "/flows" });
          })
        }
        onPlay={() =>
          void flushSave().then((saved) => {
            if (saved)
              void navigate({ to: "/flows/$flowId/play", params: { flowId } });
          })
        }
        onApp={() =>
          void flushSave().then((saved) => {
            if (saved)
              void navigate({ to: "/flows/$flowId/app", params: { flowId } });
          })
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
        {paletteOpen && !readOnly && <Palette onAdd={addAtCenter} />}
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
      {recipeEditorOpen && (
        <Suspense fallback={null}>
          <RecipeEditor />
        </Suspense>
      )}
      {recipeRunsOpen && (
        <Suspense fallback={null}>
          <RecipeRuns />
        </Suspense>
      )}
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
