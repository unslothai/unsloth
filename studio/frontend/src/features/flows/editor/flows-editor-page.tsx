// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import "@xyflow/react/dist/style.css";
import { useNavigate } from "@tanstack/react-router";
import { ReactFlowProvider } from "@xyflow/react";
import { useReactFlow } from "@xyflow/react";
import { type ReactElement, useCallback, useEffect, useState } from "react";
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
        <div className="border-b border-amber-500/30 bg-amber-500/10 px-3 py-1.5 text-[12px] text-amber-700 dark:text-amber-200">
          This flow uses blocks this version of Unsloth Studio does not know, so
          it opens read-only. Update Studio to edit or run it.
        </div>
      )}
      {!readOnly && <CapabilityBanner flowId={flowId} />}
      <div className="flex min-h-0 flex-1">
        {paletteOpen && !readOnly && <Palette onAdd={addAtCenter} />}
        <div className="flex min-w-0 flex-1 flex-col">
          <div className="relative min-h-0 flex-1">
            <FlowCanvas readOnly={readOnly} />
            {!drawerOpen && runStatus && (
              <button
                type="button"
                onClick={() => useUiStore.getState().setDrawer(true, "run")}
                className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full border border-border bg-popover px-3 py-1 text-[12px] capitalize text-foreground shadow-border hover:bg-accent"
              >
                Run: {runStatus}
              </button>
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
      <div className="flex h-full flex-1 items-center justify-center bg-background text-sm text-muted-foreground">
        Opening flow…
      </div>
    );
  }
  if (state.status === "error") {
    return (
      <div className="flex h-full flex-1 flex-col items-center justify-center gap-2 bg-background text-sm text-muted-foreground">
        <p>{state.message}</p>
        <a href="/flows" className="text-primary hover:underline">
          Back to flows
        </a>
      </div>
    );
  }
  return (
    <ReactFlowProvider>
      <EditorBody flowId={flowId} readOnly={state.readOnly} />
    </ReactFlowProvider>
  );
}
